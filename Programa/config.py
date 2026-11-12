"""
============================================================
 CONFIGURACOES DO SISTEMA - CAIXA COM VISAO COMPUTACIONAL
============================================================
Edite os valores abaixo para ajustar o comportamento do
programa sem precisar mexer no resto do codigo.
"""

import os
import sys

if getattr(sys, "frozen", False):
    # Executavel gerado pelo PyInstaller: os arquivos ficam na mesma
    # pasta do .exe (nao dentro de uma pasta temporaria descartavel),
    # entao da pra trocar o modelo sem precisar gerar o .exe de novo.
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------
# MODELO
# ------------------------------------------------------------------
# Caminho para o modelo treinado. Pode ser .pt (PyTorch) ou .onnx.
# Recomenda-se usar .onnx no computador da apresentacao (mais leve/rapido).
# Coloque o arquivo exportado dentro da pasta "modelo/".
MODEL_PATH = os.path.join(BASE_DIR, "modelo", "best.onnx")

# Confianca minima inicial (pode ser ajustada em tempo real na interface)
CONF_THRESHOLD_DEFAULT = 0.60
CONF_THRESHOLD_MIN = 0.10
CONF_THRESHOLD_MAX = 0.95
CONF_THRESHOLD_STEP = 0.05

# ------------------------------------------------------------------
# REGRAS DE DETECCAO / CARRINHO
# ------------------------------------------------------------------
# Tempo (segundos) que um produto precisa ficar sendo detectado
# ininterruptamente para ser adicionado ao carrinho.
MIN_DETECTION_SECONDS = 1.5

# Tempo (segundos) de "pausa" apos adicionar um item, para evitar que o
# mesmo produto seja lido duas vezes seguidas.
COOLDOWN_SECONDS = 2.5

# Depois de adicionada ao carrinho, a classe só volta a ser considerada
# se ficar pelo menos esse tempo SEM ser detectada (evita leituras
# repetidas do mesmo produto parado na frente da câmera).
RELEASE_ABSENCE_SECONDS = 0.15

# ------------------------------------------------------------------
# DESEMPENHO / CAMERA
# ------------------------------------------------------------------
CAMERA_FPS_TARGET = 30
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720

# Quando True: a imagem da camera roda sempre fluida (30fps) e as caixas
# do YOLO ficam "desincronizadas" (podem estar um pouco atrasadas em
# relacao ao frame atual). Ideal para hardware fraco - evita sensacao de
# travamento no video.
# Quando False: a imagem so e atualizada quando uma nova deteccao
# termina (100% sincronizado, porem mais "travado" em hardware fraco).
DESYNC_BBOXES = True

# Numero de indices de camera testados ao montar a lista (0..N-1)
MAX_CAMERAS_TO_SCAN = 5

# Tamanho de imagem usado na inferencia (deve bater com o treino: 640)
INFERENCE_IMGSZ = 640

# ------------------------------------------------------------------
# GESTO PARA FINALIZAR COMPRA (opcional - MediaPipe + MLP)
# ------------------------------------------------------------------
# Interação 100% pela câmera: levante as duas mãos no gesto treinado e
# segure por GESTURE_HOLD_SECONDS pra finalizar a compra, sem precisar
# de mouse/teclado. Ativado/desativado por um botão na interface (como
# o de "vídeo fluido"), desligado por padrão pra não gastar CPU à toa.
ENABLE_GESTURE_CHECKOUT_DEFAULT = False

GESTURE_MODEL_PATH = os.path.join(BASE_DIR, "modelo", "gesture_model.joblib")
GESTURE_DATASET_PATH = os.path.join(BASE_DIR, "modelo", "gesture_dataset.csv")

GESTURE_HOLD_SECONDS = 1.5     # tempo segurando o gesto pra confirmar
GESTURE_COOLDOWN_SECONDS = 3.0  # pausa apos finalizar, evita disparo duplo
GESTURE_MIN_CONFIDENCE = 0.80   # confianca minima do classificador (0-1)

MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.6
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

# Modelo do MediaPipe HandLandmarker (API atual - "mp.solutions" foi
# removida nas versões recentes do mediapipe). Baixado automaticamente
# na primeira vez (precisa de internet só nessa primeira vez) e salvo
# aqui pra reuso offline depois - inclua esse arquivo ao copiar a pasta
# modelo/ pro .exe.
HAND_LANDMARKER_MODEL_PATH = os.path.join(BASE_DIR, "modelo", "hand_landmarker.task")

# Usado só por train_gesture_model.py (captura por piscar, já que as
# duas mãos ficam ocupadas segurando o gesto). NÃO precisa ir junto no
# .exe final - é só uma ferramenta de coleta de dados.
FACE_LANDMARKER_MODEL_PATH = os.path.join(BASE_DIR, "modelo", "face_landmarker.task")
BLINK_EAR_THRESHOLD = 0.21      # abaixo disso, considera olho fechado
BLINK_COOLDOWN_SECONDS = 0.6    # evita capturar 2x no mesmo piscar

# ------------------------------------------------------------------
# SISTEMA DE PONTOS (YOLO, CLASSIFICADOR E OCR)
# ------------------------------------------------------------------
# Oculto na interface; use  python main.py --debug  (ou o .exe com
# --debug) para ver a pontuação de cada classe e o texto lido pelo OCR.
#
# - YOLO: a melhor caixa de cada classe vale até SCORE_YOLO_MAX_POINTS
#   (= 30 x confiança).
# - Desempate: se classes acima da confiança mínima estão a até
#   SCORE_CENTER_TIE_MARGIN (0.08 = 8 pontos percentuais) da melhor, a
#   mais perto do centro da imagem ganha até SCORE_CENTER_MAX_POINTS.
# - Classificador: 30 x confiança para a classe acima do limiar.
# - OCR: até 35 pontos pela fração de palavras esperadas reconhecidas.
# - Aceita a classe com maior total, desde que total >= 30 x confiança mínima.
SCORE_YOLO_MAX_POINTS = 30
SCORE_CLASSIFIER_MAX_POINTS = 30
SCORE_CENTER_MAX_POINTS = 5
SCORE_CENTER_TIE_MARGIN = 0.08
SCORE_OCR_MAX_POINTS = 35

ENABLE_OCR = True                # False = só YOLO (nem carrega o OCR)
OCR_INTERVAL_SECONDS = 0.5       # pausa mínima entre leituras (poupa CPU)
OCR_RESULT_TTL_SECONDS = 3.0     # quanto tempo uma leitura continua valendo
OCR_USE_BOX_CROP = True          # True: lê só a caixa do YOLO; False: frame inteiro
OCR_BOX_PADDING = 0.10           # margem extra em volta da caixa
OCR_MIN_TEXT_SCORE = 0.5         # descarta trechos com confiança menor
OCR_MAX_SIDE = 800               # só reduz recortes maiores que isso
OCR_DET_SIDE_LEN = 448           # menor = mais rápido (mas lê texto menor pior); padrão do RapidOCR: 736
OCR_THREADS = 2                  # núcleos do OCR (0 = todos); menos = o YOLO sofre menos
OCR_FUZZY_THRESHOLD = 0.85       # tolerância a erro de leitura (palavras 5+ letras)

# Palavras (uma só por item, sem acento/maiúscula) que aparecem no rótulo
# de cada classe. Evite palavras genéricas que existam em outros produtos.
# Ajuste olhando o texto lido no modo --debug. "Milho" só tem uma
# palavra: adicione a marca da sua lata.
# Estes são os padrões. O arquivo ocr_keywords.json (criado do lado do
# .exe/main.py na primeira execução) sobrescreve por classe, sem recompilar.
OCR_KEYWORDS = {
    "Azeitonas":       ["azeitonas", "saborosas", "fatiadas"],
    "Café":            ["cafe", "coracoes", "torrado", "moido"],
    "Coca 600ml":      ["coca", "cola"],
    "Feijão":          ["feijao", "pacha", "vermelho"],
    "Feijão Vermelho": ["feijao", "pacha", "vermelho"],
    "Ketchup":         ["ketchup", "hellmann"],
    "Milho":           ["milho"],
    "Molho de tomate": ["molho", "tomate", "predilecta"],
    "Mostarda":        ["mostarda", "flamboyant"],
    "Nissin Miojo":    ["miojo", "nissin", "galinha", "caipirinha"],
    "3 Corações":      ["cafe", "coracoes", "torrado", "moido"],
}

# ------------------------------------------------------------------
# PRODUTOS
# ------------------------------------------------------------------
# Mapeia o nome da CLASSE DO MODELO (exatamente como aparece no
# treinamento/YOLO) para o nome de exibicao no carrinho e o preco
# unitario em R$.
#
# ATENCAO 1: "Coca 600ml" e "Milho" nao tiveram preco informado -
# ajuste os valores abaixo antes da apresentacao.
#
# ATENCAO 2: a classe "3 Coracoes" teve apenas 1 imagem no dataset e
# metricas ruins (mAP50-95 = 0.304, recall = 0). O nome completo do
# produto "Cafe" ja e "Cafe Tradicional 3 Coracoes 500g", entao e bem
# provavel que essa seja uma classe duplicada por erro de rotulagem.
# Por enquanto ela foi mapeada para o mesmo produto/preco do "Cafe"
# para nao quebrar o carrinho, mas o ideal e corrigir o dataset
# (unificar as classes ou remover as anotacoes erradas) e retreinar.
# ATENCAO 3: no log mais recente o modelo aparece com uma classe extra,
# "Feijão Vermelho", além de "Feijão" (que já é exibido como "Feijão
# Vermelho Pachá 1kg"). Provavelmente é outra classe duplicada/rotulada
# por engano, no mesmo padrao do "3 Corações". Mapeada abaixo pro mesmo
# produto pra nao quebrar o carrinho - revisar o dataset e retreinar.
PRODUCTS = {
    "Azeitonas":       {"display_name": "Azeitonas verdes fatiadas Saborosas 220g", "price": 8.50},
    "Café":            {"display_name": "Café Tradicional 3 Corações 500g",         "price": 25.50},
    "Coca 600ml":      {"display_name": "Coca-Cola 600ml",                          "price": 8.00},
    "Feijão":          {"display_name": "Feijão Vermelho Pachá 1kg",                "price": 10.90},
    "Feijão Vermelho": {"display_name": "Feijão Vermelho Pachá 1kg",                "price": 10.90},  # ver ATENCAO 3
    "Ketchup":         {"display_name": "Ketchup Hellmann's 380g",                  "price": 8.90},
    "Milho":           {"display_name": "Milho",                                    "price": 6.50},
    "Molho de tomate": {"display_name": "Molho de tomate Predilecta 300g",          "price": 6.50},
    "Mostarda":        {"display_name": "Mostarda Flamboyant 180g",                 "price": 7.50},
    "Nissin Miojo":    {"display_name": "Nissin Miojo Galinha Caipirinha",          "price": 2.50},
    "3 Corações":      {"display_name": "Café Tradicional 3 Corações 500g",         "price": 25.50},  # ver ATENCAO 2
}

# ------------------------------------------------------------------
# INTERFACE
# ------------------------------------------------------------------
APP_TITLE = "Caixa Autônomo (Visão Computacional)"
CAMERA_PANEL_RATIO = 0.62  # fracao da largura da janela para o painel da camera
UI_RADIUS = 14             # raio das bordas arredondadas (px)

# Mostra um contador discreto de FPS (camera e IA) no canto do video.
SHOW_FPS = True

# Aviso temporário (ex: ao limpar o carrinho em "Nova compra").
TOAST_SECONDS = 3.5

# ------------------------------------------------------------------
# SOM
# ------------------------------------------------------------------
# Toca um "bip" (estilo caixa de mercado) quando um produto e
# adicionado ao carrinho.
ENABLE_SOUND = True
SOUND_FREQ_HZ = 1800       # frequencia do bip
SOUND_DURATION_MS = 110    # duracao do bip
SOUND_VOLUME = 1.0         # 0.0 a 1.0 - unico ponto que controla o volume
                           # (o .wav gerado sempre usa amplitude maxima;
                           # antes o volume era aplicado 2x e ficava baixo)

# ------------------------------------------------------------------
# OVERRIDE EXTERNO (edite SEM precisar recompilar o .exe)
# ------------------------------------------------------------------
# Se existir um arquivo "config_override.json" do lado do .exe (ou do
# main.py), os valores dele sobrescrevem qualquer variavel deste
# arquivo. Assim da pra trocar precos, thresholds, ativar o gesto, etc.
# depois de gerado o executavel, sem precisar rodar o PyInstaller de novo.
# Exemplo de conteudo (ver config_override.example.json):
#   { "CONF_THRESHOLD_DEFAULT": 0.5, "ENABLE_GESTURE_CHECKOUT_DEFAULT": true }
import json

# ------------------------------------------------------------------
# PALAVRAS-CHAVE DO OCR EDITÁVEIS (ocr_keywords.json)
# ------------------------------------------------------------------
# Arquivo do lado do .exe (ou do main.py). Se não existir, é criado com os
# padrões acima. Cada chave é o nome da classe do YOLO e o valor é a lista de
# palavras do rótulo (sem acento/maiúscula, uma palavra por item). Lista
# vazia desativa o OCR para aquela classe. Reinicie o programa para aplicar.
OCR_KEYWORDS_FILE = os.path.join(BASE_DIR, "ocr_keywords.json")


def _load_ocr_keywords_file():
    try:
        if not os.path.exists(OCR_KEYWORDS_FILE):
            payload = {"_comentario": "Edite as palavras de cada classe (nome da classe igual ao do YOLO). "
                                      "Sem acento nem maiuscula, uma palavra por item. "
                                      "Reinicie o programa para aplicar."}
            payload.update(OCR_KEYWORDS)
            with open(OCR_KEYWORDS_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return
        with open(OCR_KEYWORDS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for cls_name, words in data.items():
            if cls_name.startswith("_") or not isinstance(words, list):
                continue
            OCR_KEYWORDS[cls_name] = [str(w) for w in words]
        print(f"[config] Palavras-chave do OCR lidas de {OCR_KEYWORDS_FILE}")
    except Exception as e:
        print(f"[config] Falha ao ler {OCR_KEYWORDS_FILE}, usando os padrões: {e}")


_load_ocr_keywords_file()

CONFIG_OVERRIDE_PATH = os.path.join(BASE_DIR, "config_override.json")


def _apply_overrides():
    if not os.path.exists(CONFIG_OVERRIDE_PATH):
        return
    try:
        with open(CONFIG_OVERRIDE_PATH, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        for key, value in overrides.items():
            globals()[key] = value
        print(f"[config] Overrides aplicados de {CONFIG_OVERRIDE_PATH}: {list(overrides.keys())}")
    except Exception as e:
        print(f"[config] Falha ao ler {CONFIG_OVERRIDE_PATH}, usando padrões: {e}")


_apply_overrides()
