"""
OCR do rótulo do produto (RapidOCR sobre onnxruntime - sem PyTorch,
modelos já embutidos no pacote, funciona offline).

Lê SOMENTE a caixa (bounding box) do YOLO, com uma pequena margem
(OCR_USE_BOX_CROP = True, o padrão). Roda em thread própria, sempre
sobre o frame mais recente: nunca atrasa o vídeo nem o YOLO.
O resultado vira pontos em scoring.py.
"""

import logging
import time
from threading import Lock

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

import config

logger = logging.getLogger(__name__)


class OcrInput:
    """Guarda só o pedido mais recente (frame + caixa do YOLO), thread-safe."""

    def __init__(self):
        self._lock = Lock()
        self._frame = None
        self._box = None
        self._request_id = 0
        self._requested_at = 0.0

    def set(self, frame, box=None):
        with self._lock:
            self._frame, self._box = frame, box
            self._request_id += 1
            self._requested_at = time.time()

    def clear(self):
        with self._lock:
            self._frame, self._box = None, None

    def get(self):
        with self._lock:
            return self._frame, self._box, self._request_id, self._requested_at


def crop_for_ocr(frame, box, padding, max_side):
    """Recorta a caixa do YOLO (com margem). Só reduz se for maior que
    max_side (imagens menores são lidas na resolução original).
    Retorna (imagem, descrição do que foi lido)."""
    h, w = frame.shape[:2]
    desc = f"frame inteiro {w}x{h}px"
    if box is not None:
        x1, y1, x2, y2 = box
        pw, ph = int((x2 - x1) * padding), int((y2 - y1) * padding)
        x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
        x2, y2 = min(w, x2 + pw), min(h, y2 + ph)
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            frame = frame[y1:y2, x1:x2]
            desc = f"caixa do YOLO {x2 - x1}x{y2 - y1}px"
    ih, iw = frame.shape[:2]
    scale = max_side / max(ih, iw)
    if scale < 1.0:
        nw, nh = int(iw * scale), int(ih * scale)
        frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        desc += f" (reduzida p/ {nw}x{nh})"
    return frame, desc


class OcrEngine:
    """Aceita o pacote novo (rapidocr) e o antigo (rapidocr_onnxruntime)."""

    def __init__(self):
        # lado da detecção: o padrão do RapidOCR (736) amplia até recortes
        # pequenos e deixa a leitura lenta; 448 é bem mais rápido
        det_side = int(getattr(config, "OCR_DET_SIDE_LEN", 448))
        threads = int(getattr(config, "OCR_THREADS", 2))  # 0 = todos os núcleos

        try:
            from rapidocr import RapidOCR
            self._legacy = False
        except ImportError:
            from rapidocr_onnxruntime import RapidOCR  # ImportError aqui = nenhum instalado
            self._legacy = True

        # o RapidOCR loga a cada leitura vazia: silencia pra não lotar o app.log
        logging.getLogger("RapidOCR").setLevel(logging.ERROR)

        try:
            if self._legacy:
                kwargs = {"det_limit_side_len": det_side}
                if threads > 0:
                    kwargs["intra_op_num_threads"] = threads
                self._engine = RapidOCR(**kwargs)
            else:
                params = {"Det.limit_side_len": det_side}
                if threads > 0:
                    params["EngineConfig.onnxruntime.intra_op_num_threads"] = threads
                    params["EngineConfig.onnxruntime.inter_op_num_threads"] = 1
                self._engine = RapidOCR(params=params)
        except Exception:
            logger.exception("RapidOCR não aceitou os parâmetros de velocidade; usando o padrão")
            self._engine = RapidOCR()

    def read(self, image_bgr, min_score):
        if self._legacy:
            result, _ = self._engine(image_bgr, use_cls=False)
            items = [(str(t), float(s)) for _, t, s in (result or [])]
        else:
            out = self._engine(image_bgr, use_cls=False)
            txts = getattr(out, "txts", None) or ()
            scores = getattr(out, "scores", None) or ()
            items = [(str(t), float(s)) for t, s in zip(txts, scores)]
        return [(t, s) for t, s in items if s >= min_score]


class OcrWorker(QThread):
    # [(texto, score)], duração (s), instante do pedido, descrição da região lida
    ocr_ready = Signal(list, float, float, str)
    ocr_error = Signal(str)

    def __init__(self, ocr_input, parent=None):
        super().__init__(parent)
        self._input = ocr_input
        self._running = False

    def run(self):
        logger.info("Thread de OCR iniciada.")
        try:
            engine = OcrEngine()
        except Exception:
            logger.exception("Falha ao iniciar o OCR (pip install rapidocr)")
            self.ocr_error.emit("OCR indisponível (instale o pacote 'rapidocr'). Usando só o YOLO.")
            return
        try:  # aquecimento: a 1ª leitura de verdade não paga o custo de inicialização
            engine.read(np.full((96, 256, 3), 255, np.uint8), 1.0)
        except Exception:
            pass
        logger.info("OCR pronto.")

        interval = float(getattr(config, "OCR_INTERVAL_SECONDS", 0.5))
        min_score = float(getattr(config, "OCR_MIN_TEXT_SCORE", 0.5))
        padding = float(getattr(config, "OCR_BOX_PADDING", 0.10))
        max_side = int(getattr(config, "OCR_MAX_SIDE", 800))

        self._running = True
        last_id, last_end = -1, 0.0
        while self._running:
            frame, box, req_id, requested_at = self._input.get()
            if frame is None or req_id == last_id or time.time() - last_end < interval:
                time.sleep(0.03)
                continue
            last_id = req_id
            crop, region = crop_for_ocr(frame, box, padding, max_side)
            t0 = time.time()
            try:
                results = engine.read(crop, min_score)
            except Exception:
                logger.exception("Erro durante o OCR")
                last_end = time.time()
                continue
            last_end = time.time()
            self.ocr_ready.emit(results, last_end - t0, requested_at, region)

    def stop(self):
        self._running = False
