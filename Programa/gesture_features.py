"""
Extração de features a partir dos landmarks de mão.

Usado tanto pelo app em produção (gesture.py) quanto pelo script de
treino (train_gesture_model.py) - É CRÍTICO que os dois usem exatamente
a mesma lógica aqui, senão o modelo treinado não bate com o que o app
gera em tempo real.

Recebe o formato GENÉRICO produzido por mediapipe_utils.detect_hands():
    [{"label": "Left" ou "Right", "landmarks": [(x, y, z), ... 21]}]
Isso mantém esse módulo independente de qual versão/API do MediaPipe
está sendo usada por trás.
"""

import numpy as np

LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3  # x, y, z
FEATURES_PER_HAND = LANDMARKS_PER_HAND * COORDS_PER_LANDMARK  # 63
NUM_FEATURES = FEATURES_PER_HAND * 2  # duas maos = 126


def _normalize_hand(landmarks_xyz):
    """Recebe os 21 pontos (x, y, z) de uma mão e devolve um vetor de
    63 features:
    - Transladado pro pulso (ponto 0) = invariante à posição da mão
      na tela.
    - Escalado pela distância pulso -> base do dedo médio (ponto 9)
      = invariante ao tamanho da mão / distância da câmera.
    """
    pts = np.array(landmarks_xyz, dtype=np.float32)  # (21, 3)
    wrist = pts[0].copy()
    pts -= wrist

    palm_ref = float(np.linalg.norm(pts[9]))
    if palm_ref < 1e-6:
        palm_ref = 1.0
    pts /= palm_ref

    return pts.flatten()


def extract_features(hands):
    """Recebe a lista genérica de mãos (ver docstring do módulo).
    Retorna um vetor de NUM_FEATURES (126) valores, na ordem [mão
    esquerda, mão direita], SE exatamente 2 mãos (uma de cada lado)
    estiverem presentes - ou None caso contrário.
    """
    if not hands or len(hands) != 2:
        return None

    by_label = {}
    for hand in hands:
        by_label[hand["label"]] = _normalize_hand(hand["landmarks"])

    if "Left" not in by_label or "Right" not in by_label:
        return None  # duas deteccoes do mesmo lado (raro) - ignora o frame

    return np.concatenate([by_label["Left"], by_label["Right"]]).astype(np.float32)
