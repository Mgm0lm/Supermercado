"""
Backend alternativo ao detector.py: em vez de localizar o produto na
cena (YOLO), apenas classifica a imagem inteira - ideal para modelos
tipo Teachable Machine, onde só um produto aparece por vez na câmera.

Espera um modelo Keras (.h5 ou .keras) + um labels.txt no formato:
    0 Miojo
    1 Feijão
    ...
"""

import logging
import time
from threading import Lock

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

import config
from detector import Detection  # reaproveita a mesma estrutura de resultado

logger = logging.getLogger(__name__)


def _load_labels(path):
    """Le um labels.txt no formato "<indice> <nome>" por linha."""
    labels = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    labels[int(parts[0])] = parts[1]
                else:
                    labels[len(labels)] = line
    except FileNotFoundError:
        logger.warning("labels.txt não encontrado em %s", path)
    return labels


class ClassificationWorker(QThread):
    # lista[Detection] (com box=None), frame usado (BGR np.ndarray), frame_id
    detections_ready = Signal(list, object, int)
    model_error = Signal(str)
    model_ready = Signal()

    def __init__(self, model_path, labels_path, shared_frame, conf_threshold,
                 img_size=224, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.labels_path = labels_path
        self.shared_frame = shared_frame
        self.img_size = img_size

        self._conf_lock = Lock()
        self._conf_threshold = conf_threshold
        self._running = False
        self._model = None
        self._labels = {}

    def set_conf_threshold(self, value):
        with self._conf_lock:
            self._conf_threshold = value

    def _get_conf_threshold(self):
        with self._conf_lock:
            return self._conf_threshold

    def run(self):
        logger.info("Thread de classificação (Keras) iniciada. Modelo: %s", self.model_path)
        self._model = self._load_model_with_fallback()
        if self._model is None:
            return  # erro ja emitido/logado em _load_model_with_fallback

        self._labels = _load_labels(self.labels_path)
        if not self._labels:
            logger.warning("Sem labels.txt válido - usando índices numéricos como nome de classe.")

        logger.info("Modelo Keras carregado com sucesso. Classes: %s", self._labels)
        self.model_ready.emit()

        self._running = True
        last_frame_id = -1

        while self._running:
            frame, frame_id = self.shared_frame.get()
            if frame is None or frame_id == last_frame_id:
                time.sleep(0.01)
                continue
            last_frame_id = frame_id

            try:
                batch = self._preprocess(frame)
                preds = self._infer(batch)
            except Exception:
                logger.exception("Erro durante a inferência (Keras)")
                time.sleep(0.5)
                continue

            conf_threshold = self._get_conf_threshold()
            cls_id = int(np.argmax(preds))
            conf = float(preds[cls_id])
            raw_name = self._labels.get(cls_id, str(cls_id))
            cls_name = config.CLASS_NAME_ALIASES.get(raw_name, raw_name)

            detections = []
            if conf >= conf_threshold and cls_name not in config.BACKGROUND_CLASS_NAMES:
                # box=None: nao ha localizacao, so a classe da imagem inteira.
                detections.append(Detection(cls_name, conf, None))

            self.detections_ready.emit(detections, frame, frame_id)

    def _infer(self, batch):
        """Chama o modelo diretamente em vez de model.predict().

        model.predict() foi feito pra rodar em lote (dataset inteiro) e
        tem bastante overhead por chamada (progress bar, re-checagem do
        input, etc). Pra inferência frame a frame em tempo real, chamar
        o modelo como função (model(x)) é MUITO mais rápido - essa
        troca sozinha costuma multiplicar o FPS por vários x.
        """
        try:
            preds = self._model(batch, training=False)
        except TypeError:
            preds = self._model(batch)
        preds = np.asarray(preds)
        if preds.ndim > 1:
            preds = preds[0]
        return preds

    def _load_model_with_fallback(self):
        """Tenta carregar com Keras 3 (padrão). Modelos exportados pelo
        Teachable Machine costumam usar um formato de config mais
        antigo (ex: DepthwiseConv2D com 'groups' no config), que o
        deserializador mais rígido do Keras 3 rejeita. Nesse caso,
        reparamos o config do .h5 diretamente (removendo as chaves
        incompatíveis) e reconstruímos o modelo ainda dentro do Keras 3
        - evita depender do tf_keras, que usa imports dinâmicos que o
        PyInstaller não detecta e quebra silenciosamente no .exe."""
        try:
            import keras
            model = keras.models.load_model(self.model_path, compile=False)
            logger.info("Modelo Keras carregado com sucesso (Keras 3, direto).")
            return model
        except Exception as e_keras3:
            logger.warning(
                "Falha ao carregar com Keras 3 direto (%s). Tentando reparar o config do .h5...",
                e_keras3
            )

        try:
            model = self._load_h5_with_patched_config(self.model_path)
            logger.info("Modelo Keras carregado com sucesso (Keras 3, config reparado).")
            return model
        except Exception as e_patched:
            logger.warning(
                "Falha ao carregar com config reparado (%s). Tentando fallback tf_keras (legado)...",
                e_patched
            )

        try:
            import sys
            if getattr(sys, "frozen", False):
                raise RuntimeError(
                    "tf_keras usa imports dinâmicos incompatíveis com o "
                    "PyInstaller - pulando essa tentativa no .exe."
                )
            import tf_keras
            model = tf_keras.models.load_model(self.model_path, compile=False)
            logger.info("Modelo Keras carregado com sucesso (via tf_keras/legado).")
            return model
        except Exception as e_tfkeras:
            logger.warning("Fallback tf_keras não disponível/falhou: %s", e_tfkeras)

        logger.error(
            "Falha ao carregar o modelo Keras (%s): Keras 3 direto, config reparado e tf_keras falharam.",
            self.model_path
        )
        self.model_error.emit(
            f"Erro ao carregar o modelo Keras ({self.model_path}).\n"
            "Tentei Keras 3 direto, config reparado e o fallback tf_keras - todos falharam.\n"
            "Detalhes completos foram salvos em app.log."
        )
        return None

    @staticmethod
    def _load_h5_with_patched_config(path):
        """Le o model_config de dentro do .h5, remove chaves que o
        Keras 3 nao aceita mais em camadas antigas (ex: 'groups' em
        DepthwiseConv2D) e reconstroi o modelo + pesos a partir do
        config corrigido."""
        import json
        import h5py
        import keras

        with h5py.File(path, "r") as f:
            raw_config = f.attrs.get("model_config")
            if raw_config is None:
                raise ValueError("Arquivo .h5 sem 'model_config' - não parece ser um modelo Keras salvo com save().")
            if isinstance(raw_config, bytes):
                raw_config = raw_config.decode("utf-8")
            config_dict = json.loads(raw_config)

        ClassificationWorker._strip_incompatible_layer_kwargs(config_dict)

        class_name = config_dict.get("class_name", "Sequential")
        model_cls = getattr(keras.models, class_name, None) or getattr(keras, class_name, None)
        if model_cls is None or not hasattr(model_cls, "from_config"):
            raise ValueError(f"Não sei reconstruir modelo da classe '{class_name}'.")

        model = model_cls.from_config(config_dict["config"])
        model.load_weights(path)
        return model

    @staticmethod
    def _strip_incompatible_layer_kwargs(node):
        """Remove recursivamente kwargs que o Keras 3 rejeita em
        configs de camadas salvas por versoes antigas do TF/Keras."""
        if isinstance(node, dict):
            if node.get("class_name") == "DepthwiseConv2D":
                node.get("config", {}).pop("groups", None)
            for value in node.values():
                ClassificationWorker._strip_incompatible_layer_kwargs(value)
        elif isinstance(node, list):
            for item in node:
                ClassificationWorker._strip_incompatible_layer_kwargs(item)
            return None

    def _labels_preview(self):
        return self._labels if self._labels else "(labels.txt ainda não lido)"

    def _preprocess(self, frame_bgr):
        img = self._crop_roi(frame_bgr)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)
        img = (img / 127.5) - 1.0  # normalizacao padrao do Teachable Machine (MobileNet)
        return np.expand_dims(img, axis=0)

    @staticmethod
    def _crop_roi(frame_bgr):
        roi = config.CLASSIFIER_ROI
        if not roi:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        x1 = max(0, int(roi[0] * w))
        y1 = max(0, int(roi[1] * h))
        x2 = min(w, int(roi[2] * w))
        y2 = min(h, int(roi[3] * h))
        if x2 <= x1 or y2 <= y1:
            return frame_bgr  # ROI invalido - usa o frame inteiro
        return frame_bgr[y1:y2, x1:x2]

    def stop(self):
        self._running = False
