"""
Coleta amostras do gesto de "finalizar compra" (landmarks das duas
mãos, via MediaPipe) e treina o classificador usado por gesture.py.

USO:
    python train_gesture_model.py

CONTROLES (janela da webcam):
    P        - captura uma amostra POSITIVA (o gesto escolhido pra
               finalizar a compra) E define o "modo do piscar" como
               positivo. Segure a tecla pra capturar várias seguidas
               (autorepeat do teclado), variando um pouco a
               posição/ângulo das mãos entre uma captura e outra.
    O        - captura uma amostra NEGATIVA (qualquer outra coisa com
               as duas mãos: paradas, acenando, segurando produto,
               gestos aleatórios) E define o "modo do piscar" como
               negativo.
    Piscar   - captura uma amostra no modo atual (positivo ou
               negativo, o que foi definido por P/O pela última vez).
               Útil quando as duas mãos estão ocupadas segurando a
               pose: aperte P ou O uma vez pra escolher o modo, solte
               o teclado, segure a pose com as duas mãos e vá
               piscando pra capturar repetidamente sem tocar em nada
               (detectado via Eye Aspect Ratio na malha facial do
               MediaPipe - precisa de rosto visível).
    S        - salva as amostras acumuladas em modelo/gesture_dataset.csv
               (a captura P/O/piscar só guarda em memória até você salvar).
    T        - salva e treina o modelo com TODO o dataset salvo até
               agora (inclusive de sessões anteriores), gerando
               modelo/gesture_model.joblib.
    F        - alterna tela cheia.
    ESC      - sai (pergunta se quer salvar amostras pendentes).

Só é possível capturar quando exatamente 2 mãos estão visíveis - é a
mesma regra usada pelo app em produção. IMPORTANTE: a imagem NÃO é
espelhada aqui, de propósito - tem que ser igual ao app em produção,
senão o MediaPipe rotula "Left"/"Right" de forma diferente do que o
modelo vai ver depois, e o gesto não é reconhecido corretamente.

Na primeira execução, os modelos do MediaPipe (hand_landmarker.task e
face_landmarker.task, esse último só usado pra detectar o piscar) são
baixados automaticamente - precisa de internet só dessa vez. Nenhum
dos dois precisa ir junto no .exe final: são só ferramentas de coleta
de dados, não são usados pelo app em produção.

Recomendação: grave bem mais amostras negativas que positivas (ex:
100 positivas e 300+ negativas, incluindo momentos de "mão normal"
enquanto segura produtos, gesticula, etc.) para o modelo não ficar
"gatilho fácil".
"""

import csv
import os
import time

import cv2
import numpy as np

import config
import mediapipe_utils as mpu
from gesture_features import extract_features, NUM_FEATURES

WINDOW_NAME = "Treino de gesto - Caixa Autonomo"


def main():
    os.makedirs(os.path.dirname(config.GESTURE_DATASET_PATH), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Não consegui abrir a câmera (índice 0). Troque o índice na linha "
              "'cv2.VideoCapture(0)' se sua câmera não for a 0.")
        return

    hand_landmarker, mp_module = mpu.create_hand_landmarker(running_mode="VIDEO")
    face_landmarker, _ = mpu.create_face_landmarker(running_mode="VIDEO")
    start_time = time.time()

    blink_ear_threshold = getattr(config, "BLINK_EAR_THRESHOLD", 0.21)
    blink_cooldown_seconds = getattr(config, "BLINK_COOLDOWN_SECONDS", 0.6)

    samples = []  # [(features, label), ...] - ainda nao salvos em disco
    n_pos, n_neg = 0, 0

    # Estado do piscar: so dispara na transicao "aberto -> fechado", e
    # so pode disparar de novo depois do cooldown (evita varias
    # capturas no mesmo piscar, que dura vários frames).
    eye_was_open = True
    blink_cooldown_until = 0.0

    # [P]/[O] tambem definem o que o PISCAR captura - assim da pra
    # segurar o gesto positivo (ou a pose negativa) com as duas maos
    # ocupadas e so piscar repetidamente, sem tocar no teclado. Aperte
    # P ou O uma vez pra "escolher o modo", depois va piscando.
    blink_label = 1  # 1 = positivo (padrao), 0 = negativo

    fullscreen = True
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Controles: [P] positivo (+ define modo do piscar)  [O] negativo (+ define modo do piscar)")
    print("           [S] salvar  [T] treinar  [F] tela cheia  [ESC] sair")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_ms = int((time.time() - start_time) * 1000)

            hands = mpu.detect_hands(hand_landmarker, frame, timestamp_ms, mp_module, "VIDEO")
            if hands:
                mpu.draw_hands(frame, hands)

            features = None
            if len(hands) == 2:
                features = extract_features(hands)
            ready = features is not None

            # ---- deteccao de piscar (EAR) ----
            face_landmarks = mpu.detect_face_landmarks(
                face_landmarker, frame, timestamp_ms, mp_module, "VIDEO"
            )
            blink_event = False
            ear = None
            if face_landmarks is not None:
                h, w = frame.shape[:2]
                ear = mpu.blink_ear(face_landmarks, w, h)
                eye_open_now = ear >= blink_ear_threshold
                now = time.time()
                if eye_was_open and not eye_open_now and now >= blink_cooldown_until:
                    blink_event = True
                    blink_cooldown_until = now + blink_cooldown_seconds
                eye_was_open = eye_open_now

            captured_by_blink = False
            if blink_event and ready:
                samples.append((features, blink_label))
                if blink_label == 1:
                    n_pos += 1
                else:
                    n_neg += 1
                captured_by_blink = True

            # ---- interface na tela ----
            status = "2 MAOS OK - pronto pra capturar" if ready else "Mostre as DUAS maos pra capturar"
            color = (0, 200, 0) if ready else (0, 0, 200)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Positivos: {n_pos}   Negativos: {n_neg}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            mode_text = "POSITIVO" if blink_label == 1 else "NEGATIVO"
            mode_color = (0, 200, 0) if blink_label == 1 else (0, 165, 255)
            cv2.putText(frame, f"Modo do piscar: {mode_text}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2)

            if ear is not None:
                ear_color = (0, 200, 0) if ear >= blink_ear_threshold else (0, 165, 255)
                cv2.putText(frame, f"EAR: {ear:.3f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, ear_color, 2)
            else:
                cv2.putText(frame, "Rosto nao detectado (piscar desativado)", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200), 2)

            if captured_by_blink:
                cv2.putText(frame, "PISCADA CAPTURADA!", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

            cv2.putText(frame, "[P] positivo  [O] negativo  (ambos tb definem o modo do piscar)  "
                                "[S] salvar  [T] treinar  [F] tela cheia  [ESC] sair",
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                break
            elif key in (ord("p"), ord("P")):
                blink_label = 1
                if features is not None:
                    samples.append((features, 1))
                    n_pos += 1
            elif key in (ord("o"), ord("O")):
                blink_label = 0
                if features is not None:
                    samples.append((features, 0))
                    n_neg += 1
            elif key in (ord("s"), ord("S")):
                _save_dataset(samples)
                samples = []
            elif key in (ord("t"), ord("T")):
                _save_dataset(samples)
                samples = []
                _train_and_save()
            elif key in (ord("f"), ord("F")):
                fullscreen = not fullscreen
                prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, prop)
    finally:
        cap.release()
        hand_landmarker.close()
        face_landmarker.close()
        cv2.destroyAllWindows()

    if samples:
        resp = input(f"Você tem {len(samples)} amostra(s) não salva(s). Salvar agora? [s/N] ")
        if resp.strip().lower() == "s":
            _save_dataset(samples)


def _save_dataset(samples):
    if not samples:
        return
    file_exists = os.path.exists(config.GESTURE_DATASET_PATH)
    with open(config.GESTURE_DATASET_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([f"f{i}" for i in range(NUM_FEATURES)] + ["label"])
        for features, label in samples:
            writer.writerow(list(features) + [label])
    print(f"{len(samples)} amostra(s) salva(s) em {config.GESTURE_DATASET_PATH}")


def _load_dataset():
    X, y = [], []
    with open(config.GESTURE_DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # cabecalho
        for row in reader:
            *feats, label = row
            X.append([float(v) for v in feats])
            y.append(int(label))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def _train_and_save():
    if not os.path.exists(config.GESTURE_DATASET_PATH):
        print("Nenhum dataset encontrado ainda. Capture amostras com [P]/[O]/piscar antes de treinar.")
        return

    X, y = _load_dataset()
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    print(f"Dataset: {len(X)} amostra(s) ({n_pos} positiva(s), {n_neg} negativa(s))")

    if n_pos == 0 or n_neg == 0:
        print("Preciso de pelo menos 1 amostra positiva E 1 negativa pra treinar.")
        return

    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    import joblib

    can_split = len(X) >= 10 and n_pos >= 2 and n_neg >= 2
    if can_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = np.empty((0, X.shape[1])), np.empty((0,))

    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=42)
    model.fit(X_train, y_train)

    if len(X_test) > 0:
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"Acurácia no conjunto de teste: {acc * 100:.1f}%")
    else:
        print("Poucas amostras pra separar um conjunto de teste - treinado com tudo, sem validação.")
        print("Colete mais amostras (recomendado: 20+ de cada classe) pra uma avaliação confiável.")

    os.makedirs(os.path.dirname(config.GESTURE_MODEL_PATH), exist_ok=True)
    joblib.dump(model, config.GESTURE_MODEL_PATH)
    print(f"Modelo salvo em {config.GESTURE_MODEL_PATH}")


if __name__ == "__main__":
    main()
