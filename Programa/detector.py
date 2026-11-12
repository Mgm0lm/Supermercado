"""
Thread de deteccao. Roda separada da thread de camera, na velocidade
que o hardware permitir - é isso que possibilita manter o video fluido
mesmo em maquina fraca (ver config.DESYNC_BBOXES).
"""

import logging
import math
import time
import traceback
from threading import Lock

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)

MAX_CONSECUTIVE_ERRORS = 5


class Detection:
    __slots__ = ("cls_name", "conf", "box")

    def __init__(self, cls_name, conf, box):
        self.cls_name = cls_name
        self.conf = conf
        self.box = box  # (x1, y1, x2, y2)


class DetectionWorker(QThread):
    # lista[Detection], quadro original, frame_id, revisão da área
    detections_ready = Signal(list, object, int, int, object)
    model_error = Signal(str)
    classifier_error = Signal(str)
    model_ready = Signal()

    def __init__(self, model_path, shared_frame, conf_threshold, imgsz=640, parent=None,
                 classifier_path=None):
        super().__init__(parent)
        self.shared_frame = shared_frame
        self.model_path = model_path
        self.imgsz = imgsz
        self.classifier_path = classifier_path

        self._conf_lock = Lock()
        self._conf_threshold = conf_threshold
        self._zone = None
        self._zone_revision = 0
        self._zone_pending = False
        self._running = False
        self._model = None

    def set_conf_threshold(self, value):
        with self._conf_lock:
            self._conf_threshold = value

    def set_zone(self, zone, revision, pending=False):
        with self._conf_lock:
            self._zone = zone
            self._zone_revision = revision
            self._zone_pending = pending

    def run(self):
        logger.info("Thread de detecção iniciada. Carregando modelo: %s", self.model_path)
        try:
            import os as _os
            _os.environ.setdefault("YOLO_AUTOINSTALL", "False")

            from ultralytics import YOLO
            try:
                from ultralytics import settings as _yolo_settings
                _yolo_settings.update({"sync": False})
            except Exception:
                pass  # nao critico - so evita telemetria/rede em background

            self._model = YOLO(self.model_path)
        except Exception:
            logger.exception("Falha ao carregar o modelo (%s)", self.model_path)
            self.model_error.emit(
                f"Erro ao carregar o modelo ({self.model_path}).\n"
                "Detalhes completos foram salvos em app.log."
            )
            return

        logger.info("Modelo carregado com sucesso. Classes: %s", getattr(self._model, "names", "?"))
        self.model_ready.emit()
        classifier = None
        if self.classifier_path is not None:
            try:
                import config
                from classifier import ProductClassifier
                classifier = ProductClassifier(self.classifier_path)
            except Exception as error:
                logger.exception("Falha ao iniciar o classificador")
                self.classifier_error.emit(f"Classificador indisponível: {error}")
        self._running = True
        last_frame_id = -1
        consecutive_errors = 0

        while self._running:
            with self._conf_lock:
                conf, zone = self._conf_threshold, self._zone
                revision, pending = self._zone_revision, self._zone_pending
            if pending:
                time.sleep(0.01)
                continue
            frame, frame_id = self.shared_frame.get()
            if frame is None or frame_id == last_frame_id:
                time.sleep(0.01)
                continue
            last_frame_id = frame_id

            source = frame
            left = top = 0
            if zone is not None:
                h, w = frame.shape[:2]
                left, top = math.floor(zone[0] * w), math.floor(zone[1] * h)
                right, bottom = math.ceil(zone[2] * w), math.ceil(zone[3] * h)
                source = frame[top:bottom, left:right].copy()
            try:
                results = self._model.predict(
                    source, conf=conf, imgsz=self.imgsz, verbose=False
                )
                consecutive_errors = 0
            except Exception:
                consecutive_errors += 1
                logger.exception(
                    "Erro durante a inferência (tentativa %d/%d)",
                    consecutive_errors, MAX_CONSECUTIVE_ERRORS
                )
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    self.model_error.emit(
                        "A detecção parou após vários erros seguidos de inferência.\n"
                        "Detalhes completos foram salvos em app.log."
                    )
                    self._running = False
                    break
                time.sleep(0.5)
                continue

            detections = []
            if results:
                r = results[0]
                names = r.names
                for b in r.boxes:
                    cls_id = int(b.cls[0])
                    cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                    conf_val = float(b.conf[0])
                    x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
                    detections.append(Detection(
                        cls_name, conf_val, (x1 + left, y1 + top, x2 + left, y2 + top)
                    ))

            prediction = None
            if classifier is not None:
                target = max(
                    (d for d in detections if d.cls_name in config.PRODUCTS and d.conf >= conf),
                    key=lambda d: d.conf, default=None,
                )
                if target is not None:
                    try:
                        prediction = classifier.predict(frame, target.box, conf, config.PRODUCTS)
                    except Exception as error:
                        logger.exception("Falha na inferência do classificador")
                        self.classifier_error.emit(f"Classificador indisponível: {error}")
                        classifier = None

            self.detections_ready.emit(detections, frame, frame_id, revision, prediction)

    def stop(self):
        self._running = False
