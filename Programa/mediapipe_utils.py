"""
Camada fina sobre a API atual do MediaPipe (Tasks API / HandLandmarker),
usada tanto por gesture.py (app em produção) quanto por
train_gesture_model.py (script de treino).

A API antiga "mp.solutions.hands" foi removida nas versões mais
recentes do mediapipe (1.x) - por isso usamos a Tasks API aqui, que é
a atual recomendada pelo Google e deve seguir estável daqui pra frente.
Isolar isso num módulo só facilita se a API mudar de novo no futuro.
"""

import os
import urllib.request

import cv2
import numpy as np

import config

# Modelo oficial do Google (poucos MB). Baixado automaticamente na
# primeira vez, se ainda não existir localmente em HAND_LANDMARKER_MODEL_PATH.
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Modelo do FaceLandmarker (usado só por train_gesture_model.py, pra
# detectar o piscar via Eye Aspect Ratio - EAR).
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# Índices clássicos (estilo dlib) dos 6 pontos de cada olho, na malha
# de 478 pontos do MediaPipe Face Landmarker: (canto, topo1, topo2,
# canto, base2, base1).
LEFT_EYE_EAR_IDX = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_EAR_IDX = (33, 160, 158, 133, 153, 144)

# Pares de índices dos 21 pontos da mão, pra desenhar o "esqueleto" sem
# depender de mp.solutions.drawing_utils (também removido na v1.x).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # polegar
    (0, 5), (5, 6), (6, 7), (7, 8),           # indicador
    (5, 9), (9, 10), (10, 11), (11, 12),      # medio
    (9, 13), (13, 14), (14, 15), (15, 16),    # anelar
    (13, 17), (17, 18), (18, 19), (19, 20),   # minimo
    (0, 17),
]


def ensure_hand_landmarker_model():
    """Garante que o .task do HandLandmarker existe localmente,
    baixando do Google se necessário (só precisa de internet nessa
    primeira vez)."""
    path = config.HAND_LANDMARKER_MODEL_PATH
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[mediapipe_utils] Baixando modelo HandLandmarker em {HAND_LANDMARKER_MODEL_URL} ...")
    urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, path)
    print(f"[mediapipe_utils] Modelo salvo em {path}")
    return path


def create_hand_landmarker(running_mode="VIDEO"):
    """Cria um HandLandmarker configurado pra até 2 mãos. Retorna
    (landmarker, módulo mediapipe) - o módulo é devolvido pra
    detect_hands() poder criar as mp.Image sem reimportar."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = ensure_hand_landmarker_model()

    mode_map = {"VIDEO": vision.RunningMode.VIDEO, "IMAGE": vision.RunningMode.IMAGE}
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_hands=2,
        min_hand_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        min_tracking_confidence=config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        running_mode=mode_map[running_mode],
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    return landmarker, mp


def detect_hands(landmarker, frame_bgr, timestamp_ms, mp_module, running_mode="VIDEO"):
    """Roda a detecção num frame BGR e devolve uma lista GENÉRICA de
    mãos (independente da versão da API):
        [{"label": "Left" ou "Right", "landmarks": [(x, y, z), ... 21]}]
    Coordenadas normalizadas (0.0 a 1.0), como o MediaPipe retorna.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)

    if running_mode == "VIDEO":
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
    else:
        result = landmarker.detect(mp_image)

    hands = []
    if not result.hand_landmarks:
        return hands
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name  # "Left" ou "Right"
        pts = [(lm.x, lm.y, lm.z) for lm in landmarks]
        hands.append({"label": label, "landmarks": pts})
    return hands


def draw_hands(frame_bgr, hands, color=(200, 200, 200)):
    """Desenha pontos + conexões das mãos (formato genérico de
    detect_hands) direto com cv2, sem depender de mp.solutions."""
    h, w = frame_bgr.shape[:2]
    for hand in hands:
        pts_px = [(int(x * w), int(y * h)) for x, y, _ in hand["landmarks"]]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame_bgr, pts_px[a], pts_px[b], color, 2)
        for px, py in pts_px:
            cv2.circle(frame_bgr, (px, py), 3, color, -1)
    return frame_bgr


# --------------------------------------------------------------------
# ROSTO / PISCAR - usado só por train_gesture_model.py (captura por
# piscar, já que as duas mãos ficam ocupadas segurando o gesto)
# --------------------------------------------------------------------

def ensure_face_landmarker_model():
    """Garante que o .task do FaceLandmarker existe localmente,
    baixando do Google se necessário."""
    path = getattr(
        config, "FACE_LANDMARKER_MODEL_PATH",
        os.path.join(os.path.dirname(config.HAND_LANDMARKER_MODEL_PATH), "face_landmarker.task")
    )
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[mediapipe_utils] Baixando modelo FaceLandmarker em {FACE_LANDMARKER_MODEL_URL} ...")
    urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, path)
    print(f"[mediapipe_utils] Modelo salvo em {path}")
    return path


def create_face_landmarker(running_mode="VIDEO"):
    """Cria um FaceLandmarker configurado pra 1 rosto. Retorna
    (landmarker, módulo mediapipe)."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = ensure_face_landmarker_model()

    mode_map = {"VIDEO": vision.RunningMode.VIDEO, "IMAGE": vision.RunningMode.IMAGE}
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_faces=1,
        output_face_blendshapes=False,
        running_mode=mode_map[running_mode],
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    return landmarker, mp


def detect_face_landmarks(landmarker, frame_bgr, timestamp_ms, mp_module, running_mode="VIDEO"):
    """Retorna a lista de (x, y, z) normalizados do primeiro rosto
    detectado, ou None se nenhum rosto foi encontrado."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp_module.Image(image_format=mp_module.ImageFormat.SRGB, data=rgb)

    if running_mode == "VIDEO":
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
    else:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None
    return [(lm.x, lm.y, lm.z) for lm in result.face_landmarks[0]]


def _eye_aspect_ratio(face_landmarks, eye_idx, frame_w, frame_h):
    """EAR de um olho só. Convertido pra pixel antes de medir as
    distâncias, pra não distorcer se o frame não for quadrado."""
    pts = np.array(
        [(face_landmarks[i][0] * frame_w, face_landmarks[i][1] * frame_h) for i in eye_idx],
        dtype=np.float32,
    )
    p1, p2, p3, p4, p5, p6 = pts
    vertical = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 1.0
    return float(vertical / (2.0 * horizontal))


def blink_ear(face_landmarks, frame_w, frame_h):
    """Eye Aspect Ratio médio dos dois olhos - cai bruscamente durante
    um piscar. Mesma técnica clássica usada em detectores de sonolência
    (Soukupová & Čech, 2016), adaptada pros 478 pontos do MediaPipe."""
    left = _eye_aspect_ratio(face_landmarks, LEFT_EYE_EAR_IDX, frame_w, frame_h)
    right = _eye_aspect_ratio(face_landmarks, RIGHT_EYE_EAR_IDX, frame_w, frame_h)
    return (left + right) / 2.0
