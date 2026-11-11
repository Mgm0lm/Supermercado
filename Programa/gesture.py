"""
Thread opcional: rastreia as duas mãos (MediaPipe HandLandmarker) e
classifica o gesto atual com um modelo leve (MLP do
scikit-learn, treinado por train_gesture_model.py) pra permitir
finalizar a compra sem precisar de mouse/teclado - interação 100%
pela câmera.

Roda desacoplada da thread de câmera e da de detecção de produtos,
no mesmo espírito das outras: cada uma na sua velocidade, sem travar
o vídeo.
"""

import logging
import time

from PySide6.QtCore import QThread, Signal

import config
import mediapipe_utils as mpu
from gesture_features import extract_features

logger = logging.getLogger(__name__)


class GestureWorker(QThread):
    # gesto_reconhecido (bool), confianca (0-1), hands (lista generica de
    # mediapipe_utils.detect_hands, ou None) - so pra desenhar na tela.
    gesture_status = Signal(bool, float, object, object)
    model_error = Signal(str)
    classifier_error = Signal(str)

    def __init__(self, shared_frame, hand_block=False, parent=None):
        super().__init__(parent)
        self.shared_frame = shared_frame
        self.hand_block = hand_block
        self._running = False
        self._landmarker = None
        self._mp = None
        self._model = None

    def run(self):
        logger.info("Thread de gesto iniciada.")
        try:
            self._landmarker, self._mp = mpu.create_hand_landmarker(running_mode="VIDEO")
        except Exception:
            logger.exception("Falha ao iniciar o MediaPipe HandLandmarker")
            self.model_error.emit(
                "Erro ao iniciar o MediaPipe HandLandmarker.\n"
                "Detalhes completos foram salvos em app.log.\n"
                "(Confira também se há internet na primeira execução - "
                "o modelo é baixado automaticamente.)"
            )
            return

        try:
            import joblib
            self._model = joblib.load(config.GESTURE_MODEL_PATH)
            logger.info("Modelo de gesto carregado: %s", config.GESTURE_MODEL_PATH)
        except Exception:
            logger.exception("Falha ao carregar o modelo de gesto (%s)", config.GESTURE_MODEL_PATH)
            self.classifier_error.emit(
                f"Erro ao carregar o modelo de gesto ({config.GESTURE_MODEL_PATH}).\n"
                "Rode train_gesture_model.py para gerar esse arquivo.\n"
                "Detalhes completos foram salvos em app.log."
            )
            if not self.hand_block:
                self._landmarker.close()
                return

        self._running = True
        last_frame_id = -1
        start_time = time.time()

        while self._running:
            sampled = time.monotonic()
            frame, frame_id = self.shared_frame.get()
            if frame is None or frame_id == last_frame_id:
                time.sleep(0.02)
                continue
            last_frame_id = frame_id

            timestamp_ms = int((time.time() - start_time) * 1000)
            try:
                hands = mpu.detect_hands(self._landmarker, frame, timestamp_ms, self._mp, "VIDEO")
            except Exception:
                logger.exception("Erro processando o gesto")
                self.gesture_status.emit(False, 0.0, None, None)
                time.sleep(0.3)
                continue

            is_gesture, confidence = False, 0.0
            if self._model is not None and len(hands) == 2:
                features = extract_features(hands)
                if features is not None:
                    confidence = self._predict_confidence(features)
                    is_gesture = confidence >= config.GESTURE_MIN_CONFIDENCE

            self.gesture_status.emit(is_gesture, confidence, hands or None, sampled)

        if self._landmarker is not None:
            self._landmarker.close()

    def _predict_confidence(self, features):
        X = features.reshape(1, -1)
        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(X)[0]
            classes = list(self._model.classes_)
            if 1 in classes:
                return float(proba[classes.index(1)])
            return float(proba.max())
        pred = self._model.predict(X)[0]  # fallback p/ modelos sem predict_proba
        return 1.0 if pred == 1 else 0.0

    def stop(self):
        self._running = False
