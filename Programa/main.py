"""
Caixa Autonomo com Visao Computacional - protótipo para apresentação.

Como rodar:
    pip install -r requirements.txt
    python main.py

ESC fecha o programa. Veja config.py para ajustar precos, classes,
sensibilidade e o modo "desincronizado" de bounding boxes.
"""

import argparse
import logging
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
from PySide6.QtCore import Qt, QRectF, QUrl, QTimer, QThread
from PySide6.QtGui import QImage, QPixmap, QPainterPath, QRegion, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QHBoxLayout,
    QVBoxLayout, QFrame, QComboBox, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QCheckBox, QSizePolicy
)

import config
from camera_manager import CameraWorker, SharedFrame, list_available_cameras
from detector import DetectionWorker
from gesture import GestureWorker
from ocr import OcrInput, OcrWorker
import scoring
from cart import Cart
from ui_styles import STYLESHEET
from sound import generate_checkout_beep

try:
    from PySide6.QtMultimedia import QSoundEffect
    _SOUND_AVAILABLE = True
except Exception:
    _SOUND_AVAILABLE = False


def cfg(name, default):
    """Lê uma variável do config.py, tolerando um arquivo desatualizado."""
    return getattr(config, name, default)


def setup_logging():
    """Grava logs em app.log (do lado do .exe/script) e também no
    console quando disponível. Sem isso, erros em threads ficam
    invisíveis no .exe gerado com --windowed (não tem console)."""
    log_path = os.path.join(config.BASE_DIR, "app.log")
    handlers = [logging.FileHandler(log_path, encoding="utf-8")]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler(sys.stderr))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

    def log_uncaught_exceptions(exc_type, exc_value, exc_tb):
        logging.getLogger("uncaught").critical(
            "Exceção não tratada", exc_info=(exc_type, exc_value, exc_tb)
        )

    sys.excepthook = log_uncaught_exceptions
    logging.info("Log iniciado em %s", log_path)


class FpsCounter:
    """Média móvel simples de FPS a partir dos timestamps dos últimos frames."""

    def __init__(self, window=30):
        self._times = deque(maxlen=window)

    def tick(self):
        self._times.append(time.time())

    @property
    def fps(self):
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        if span <= 0:
            return 0.0
        return (len(self._times) - 1) / span


def cv_frame_to_pixmap(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = frame_rgb.shape
    qimg = QImage(frame_rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def draw_detections(frame, detections, active_class=None):
    """Desenha as caixas do YOLO na imagem (BGR, modifica uma cópia)."""
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.box
        highlight = det.cls_name == active_class
        color = (37, 99, 235) if not highlight else (17, 138, 76)  # BGR
        thickness = 2 if not highlight else 3
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{det.cls_name} {det.conf * 100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
        cv2.putText(out, label, (x1 + 4, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_hand_landmarks(frame, hands, is_gesture_active):
    """Desenha pontinhos nas mãos detectadas (feedback visual leve).
    'hands' é o formato genérico de mediapipe_utils.detect_hands():
    [{"label": ..., "landmarks": [(x, y, z), ...21]}]."""
    if not hands:
        return frame
    h, w = frame.shape[:2]
    color = (17, 138, 76) if is_gesture_active else (200, 200, 200)
    for hand in hands:
        for x, y, _ in hand["landmarks"]:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 3, color, -1)
    return frame


def draw_fps_overlay(frame, cam_fps, det_fps):
    """Texto pequeno e discreto no canto, pra nao 'poluir' a interface."""
    text = f"cam {cam_fps:4.1f} fps  |  IA {det_fps:4.1f} fps"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = 10, th + 8

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 6, 2), (x + tw + 6, y + 6), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.putText(frame, text, (x, y), font, scale, (220, 220, 220), thickness, cv2.LINE_AA)
    return frame


class MainWindow(QMainWindow):
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug  # --debug: mostra o painel de pontuacao
        self.setWindowTitle(config.APP_TITLE)
        self.setStyleSheet(STYLESHEET)

        # -------- estado --------
        self.cart = Cart()
        self.shared_frame = SharedFrame()
        self.conf_threshold = config.CONF_THRESHOLD_DEFAULT
        self.desync_enabled = config.DESYNC_BBOXES

        self.last_detections = []           # usado no modo desincronizado
        self.candidate_class = None
        self.candidate_start = None
        self.cooldown_until = 0.0

        self.camera_worker = None
        self.detection_worker = None
        self.gesture_worker = None

        self.gesture_enabled = config.ENABLE_GESTURE_CHECKOUT_DEFAULT
        self.last_hand_landmarks = None
        self.gesture_hold_start = None
        self.gesture_cooldown_until = 0.0

        # OCR / sistema de pontos
        self.ocr_input = OcrInput()
        self.ocr_worker = None
        self.ocr_available = False
        self.ocr_matches = {}
        self.ocr_texts = []
        self.ocr_elapsed = 0.0
        self.ocr_updated_at = None
        self.ocr_ignore_before = 0.0
        self.ocr_target = None
        self.ocr_target_since = 0.0
        self.ocr_history = deque()  # (instante, textos) das leituras dentro do TTL
        self.ocr_region = ""

        # classes recem adicionadas: {classe: ultimo instante em que foi vista}
        self.blocked = {}
        self._debug_error_logged = False
        self._message_font_px = 0

        self.cam_fps_counter = FpsCounter()
        self.det_fps_counter = FpsCounter()

        self.beep_sound = self._init_sound()

        self._build_ui()
        self._start_camera(self.camera_combo.currentData())
        self._start_detector()
        if self.gesture_enabled:
            self._start_gesture_worker()

    def _init_sound(self):
        if not config.ENABLE_SOUND or not _SOUND_AVAILABLE:
            return None
        try:
            beep_path = generate_checkout_beep(
                freq=config.SOUND_FREQ_HZ,
                duration_ms=config.SOUND_DURATION_MS,
                volume=1.0,  # amplitude sempre maxima no .wav; quem
                             # controla o volume final e o QSoundEffect
                             # abaixo (antes os dois multiplicavam entre
                             # si e o som saia baixo demais)
            )
            effect = QSoundEffect()
            effect.setSource(QUrl.fromLocalFile(beep_path))
            effect.setVolume(config.SOUND_VOLUME)
            return effect
        except Exception:
            # Se o backend de audio nao estiver disponivel na maquina,
            # o app continua funcionando normalmente, so sem som.
            return None

    def _play_beep(self):
        if self.beep_sound is not None:
            self.beep_sound.play()

    # ------------------------------------------------------------------
    # CONSTRUCAO DA INTERFACE
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        main_layout = QHBoxLayout(root)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # ---------------- painel esquerdo: camera ----------------
        left_col = QVBoxLayout()
        left_col.setSpacing(10)

        header = QHBoxLayout()
        self.message_label = QLabel("Inicie a compra")
        self.message_label.setObjectName("mensagem")
        header.addWidget(self.message_label)
        header.addStretch()
        esc_hint = QLabel("Pressione ESC para sair")
        esc_hint.setObjectName("subtitulo")
        header.addWidget(esc_hint)
        left_col.addLayout(header)

        self.video_label = QLabel()
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(480, 360)
        left_col.addWidget(self.video_label, stretch=1)

        status_row = QHBoxLayout()
        self.status_label = QLabel("Aguardando produto...")
        self.status_label.setObjectName("statusDeteccao")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        status_row.addWidget(self.status_label, stretch=1)
        left_col.addLayout(status_row)
        left_col.addWidget(self.progress_bar)

        main_layout.addLayout(left_col, int(config.CAMERA_PANEL_RATIO * 100))

        # ---------------- painel direito: controles + carrinho ----------------
        right_col = QVBoxLayout()
        right_col.setSpacing(14)

        right_col.addWidget(self._build_camera_card())
        right_col.addWidget(self._build_threshold_card())
        if self.debug:
            right_col.addWidget(self._build_debug_card())
        right_col.addWidget(self._build_cart_card(), stretch=1)

        main_layout.addLayout(right_col, int((1 - config.CAMERA_PANEL_RATIO) * 100))

        # ESC fecha o app
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self.close)

        self._build_toast(root)

    def _update_message(self):
        if self.candidate_class is not None:
            text = "Aguarde..."
        elif not self.cart.items:
            text = "Inicie a compra"
        else:
            text = "Passe o próximo produto"
        if self.message_label.text() != text:
            self.message_label.setText(text)

    def _build_debug_card(self):
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 10, 14, 10)
        self.debug_label = QLabel("DEBUG: aguardando detecção...")
        self.debug_label.setObjectName("debugText")
        self.debug_label.setTextInteractionFlags(Qt.NoTextInteraction)
        lay.addWidget(self.debug_label)
        return card

    def _update_debug_panel(self, scores, winner, min_total, now):
        """Painel do --debug. Qualquer falha aqui é só registrada no app.log:
        nunca pode impedir a contagem dos produtos."""
        if not self.debug:
            return
        try:
            age = None if self.ocr_updated_at is None else now - self.ocr_updated_at
            args = (scores or {}, winner, min_total, self.ocr_available,
                    self.ocr_texts, age, self.ocr_elapsed)
            try:
                text = scoring.format_debug_text(*args, blocked=set(self.blocked))
            except TypeError:  # scoring.py de uma versão antiga
                text = scoring.format_debug_text(*args)
            if self.ocr_region:
                text += "\nOCR lê: " + self.ocr_region
            self.debug_label.setText(text)
        except Exception:
            if not self._debug_error_logged:
                self._debug_error_logged = True
                logging.exception("Falha no painel de debug (a contagem continua normalmente). "
                                  "Confira se todos os arquivos são da mesma versão.")

    # ------------------------------------------------------------------
    # AVISO TEMPORARIO (toast)
    # ------------------------------------------------------------------
    def _build_toast(self, parent):
        self.toast_label = QLabel(parent)
        self.toast_label.setObjectName("toast")
        self.toast_label.setAlignment(Qt.AlignCenter)
        self.toast_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.toast_label.hide()
        self.toast_timer = QTimer(self)
        self.toast_timer.setSingleShot(True)
        self.toast_timer.timeout.connect(self.toast_label.hide)

    def _show_toast(self, message, seconds=None):
        seconds = cfg("TOAST_SECONDS", 3.5) if seconds is None else seconds
        parent = self.toast_label.parentWidget()
        pw = parent.width() if parent is not None else 1280
        ph = parent.height() if parent is not None else 720

        # tamanho proporcional à tela: fonte = 9% da altura, largura mínima = 62%
        px = max(48, int(ph * 0.09))
        self.toast_label.setText(message)
        self.toast_label.setMinimumWidth(int(pw * 0.62))
        for _ in range(20):
            self.toast_label.setStyleSheet(
                f"font-size: {px}px; padding: {int(px * 0.7)}px {int(px * 1.6)}px;")
            self.toast_label.ensurePolished()
            self.toast_label.adjustSize()
            if self.toast_label.width() <= pw * 0.94 or px <= 28:
                break
            px -= 6  # telas pequenas: reduz a fonte até caber
        self._position_toast()
        self.toast_label.show()
        self.toast_label.raise_()
        self.toast_timer.start(int(seconds * 1000))

    def _position_toast(self):
        parent = self.toast_label.parentWidget()
        if parent is not None:
            self.toast_label.move((parent.width() - self.toast_label.width()) // 2,
                                  (parent.height() - self.toast_label.height()) // 2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        px = max(40, int(self.height() * 0.055))
        if hasattr(self, "message_label") and px != self._message_font_px:
            self._message_font_px = px
            self.message_label.setStyleSheet(f"font-size: {px}px;")
        if hasattr(self, "toast_label") and self.toast_label.isVisible():
            self._position_toast()

    def _build_camera_card(self):
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 14, 14, 14)

        title = QLabel("Câmera")
        title.setStyleSheet("font-weight: 600; font-size: 14px;")
        lay.addWidget(title)

        row = QHBoxLayout()
        self.camera_combo = QComboBox()
        for idx, label in list_available_cameras(config.MAX_CAMERAS_TO_SCAN):
            self.camera_combo.addItem(label, idx)
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        row.addWidget(self.camera_combo, stretch=1)

        refresh_btn = QPushButton("↻")
        refresh_btn.setFixedWidth(36)
        refresh_btn.clicked.connect(self._refresh_cameras)
        row.addWidget(refresh_btn)
        lay.addLayout(row)

        self.desync_checkbox = QCheckBox("Vídeo fluido (caixas desincronizadas)")
        self.desync_checkbox.setChecked(self.desync_enabled)
        self.desync_checkbox.stateChanged.connect(self._on_desync_toggled)
        lay.addWidget(self.desync_checkbox)

        self.gesture_checkbox = QCheckBox("Finalizar compra com gesto (2 mãos)")
        self.gesture_checkbox.setChecked(self.gesture_enabled)
        self.gesture_checkbox.stateChanged.connect(self._on_gesture_toggled)
        lay.addWidget(self.gesture_checkbox)

        self.gesture_status_label = QLabel("")
        self.gesture_status_label.setStyleSheet("font-size: 11px; color: #6b7280;")
        self.gesture_status_label.setVisible(self.gesture_enabled)
        lay.addWidget(self.gesture_status_label)

        self.gesture_progress_bar = QProgressBar()
        self.gesture_progress_bar.setRange(0, 100)
        self.gesture_progress_bar.setValue(0)
        self.gesture_progress_bar.setFixedHeight(8)
        self.gesture_progress_bar.setTextVisible(False)
        self.gesture_progress_bar.setVisible(self.gesture_enabled)
        lay.addWidget(self.gesture_progress_bar)
        lay.addWidget(self.desync_checkbox)

        return card

    def _build_threshold_card(self):
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 14, 14, 14)

        title = QLabel("Precisão mínima para detecção")
        title.setStyleSheet("font-weight: 600; font-size: 14px;")
        lay.addWidget(title)

        row = QHBoxLayout()
        minus_btn = QPushButton("-")
        minus_btn.setObjectName("thresholdBtn")
        minus_btn.clicked.connect(lambda: self._change_threshold(-config.CONF_THRESHOLD_STEP))

        self.threshold_label = QLabel(f"{self.conf_threshold * 100:.0f}%")
        self.threshold_label.setAlignment(Qt.AlignCenter)
        self.threshold_label.setStyleSheet("font-size: 18px; font-weight: 700;")

        plus_btn = QPushButton("+")
        plus_btn.setObjectName("thresholdBtn")
        plus_btn.clicked.connect(lambda: self._change_threshold(config.CONF_THRESHOLD_STEP))

        row.addWidget(minus_btn)
        row.addWidget(self.threshold_label, stretch=1)
        row.addWidget(plus_btn)
        lay.addLayout(row)

        return card

    def _build_cart_card(self):
        card = QFrame()
        card.setObjectName("card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 14, 14, 14)

        header_row = QHBoxLayout()
        title = QLabel("Carrinho")
        title.setStyleSheet("font-weight: 600; font-size: 14px;")
        header_row.addWidget(title)
        header_row.addStretch()

        new_purchase_btn = QPushButton("Nova compra")
        new_purchase_btn.setObjectName("secondaryBtn")
        new_purchase_btn.clicked.connect(self._new_purchase)
        header_row.addWidget(new_purchase_btn)
        lay.addLayout(header_row)

        self.cart_table = QTableWidget(0, 5)
        self.cart_table.setHorizontalHeaderLabels(["Produto", "Qtd", "Preço", "Subtotal", ""])
        self.cart_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.cart_table.verticalHeader().setVisible(False)
        self.cart_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cart_table.setSelectionMode(QTableWidget.NoSelection)
        lay.addWidget(self.cart_table, stretch=1)

        total_row = QHBoxLayout()
        total_row.addStretch()
        self.total_label = QLabel("Total: R$ 0,00")
        self.total_label.setObjectName("totalLabel")
        total_row.addWidget(self.total_label)
        lay.addLayout(total_row)

        return card

    # ------------------------------------------------------------------
    # CAMERA / DETECTOR
    # ------------------------------------------------------------------
    def _start_camera(self, index):
        self._invalidate_ocr(time.time())
        self.shared_frame.clear()
        self.camera_worker = CameraWorker(
            index, config.CAMERA_FPS_TARGET,
            config.CAMERA_FRAME_WIDTH, config.CAMERA_FRAME_HEIGHT,
            self.shared_frame
        )
        self.camera_worker.frame_ready.connect(self._on_frame_ready)
        self.camera_worker.camera_error.connect(self._on_camera_error)
        self.camera_worker.start()

    def _start_detector(self):
        if not os.path.exists(config.MODEL_PATH):
            QMessageBox.warning(
                self, "Modelo não encontrado",
                f"Não encontrei o modelo em:\n{config.MODEL_PATH}\n\n"
                "A câmera vai funcionar, mas sem detecção de produtos.\n"
                "Ajuste MODEL_PATH em config.py."
            )
            return

        self.detection_worker = DetectionWorker(
            config.MODEL_PATH, self.shared_frame, self.conf_threshold,
            imgsz=config.INFERENCE_IMGSZ
        )
        self.detection_worker.detections_ready.connect(self._on_detections_ready)
        self.detection_worker.model_error.connect(self._on_model_error)
        self.detection_worker.start()

        self._start_ocr_worker()

    def _start_ocr_worker(self):
        if not cfg("ENABLE_OCR", True):
            return
        self.ocr_available = True
        self.ocr_worker = OcrWorker(self.ocr_input)
        self.ocr_worker.ocr_ready.connect(self._on_ocr_ready)
        self.ocr_worker.ocr_error.connect(self._on_ocr_error)
        self.ocr_worker.start(QThread.Priority.LowPriority)  # nao disputa CPU com o YOLO

    def _on_ocr_ready(self, results, elapsed, requested_at, region=""):
        if requested_at < self.ocr_ignore_before:
            return  # leitura de um produto que já saiu / foi adicionado
        now = time.time()
        texts = [text for text, _ in results]
        # palavras achadas em leituras diferentes se somam durante o TTL
        # (cada leitura costuma pegar só parte do rótulo, por desfoque/ângulo)
        ttl = cfg("OCR_RESULT_TTL_SECONDS", 3.0)
        target = self.ocr_target
        if (
            self.ocr_available
            and target is not None
            and now >= self.cooldown_until
            and requested_at >= self.ocr_target_since
            and now - requested_at <= ttl
            and not any(scoring.same_product(target, name) for name in self.blocked)
            and scoring.ocr_confirms(target, scoring.match_keywords(texts))
        ):
            self._confirm_product(target, now)
            self.status_label.setText("Aguardando... (evitando leitura duplicada)")
            self.progress_bar.setValue(100)
            self._update_message()
            return
        self.ocr_history.append((now, texts))
        while self.ocr_history and now - self.ocr_history[0][0] > ttl:
            self.ocr_history.popleft()
        window_texts = [t for _, ts in self.ocr_history for t in ts]
        self.ocr_texts = texts
        self.ocr_elapsed = elapsed
        self.ocr_region = region
        self.ocr_matches = scoring.match_keywords(window_texts)
        self.ocr_updated_at = now

    def _on_ocr_error(self, msg):
        self.ocr_available = False
        self._show_toast(msg, seconds=6)

    def _invalidate_ocr(self, now):
        """Descarta leituras antigas (produto saiu, foi adicionado, etc)."""
        self.ocr_ignore_before = now
        self.ocr_target = None
        self.ocr_matches = {}
        self.ocr_texts = []
        self.ocr_updated_at = None
        self.ocr_history.clear()
        self.ocr_region = ""
        self.ocr_input.clear()

    def _start_gesture_worker(self):
        if not os.path.exists(config.GESTURE_MODEL_PATH):
            QMessageBox.warning(
                self, "Modelo de gesto não encontrado",
                f"Não encontrei o modelo de gesto em:\n{config.GESTURE_MODEL_PATH}\n\n"
                "Rode train_gesture_model.py pra gerar esse arquivo antes de ativar."
            )
            self.gesture_checkbox.blockSignals(True)
            self.gesture_checkbox.setChecked(False)
            self.gesture_checkbox.blockSignals(False)
            self.gesture_enabled = False
            return

        self.gesture_worker = GestureWorker(self.shared_frame)
        self.gesture_worker.gesture_status.connect(self._on_gesture_status)
        self.gesture_worker.model_error.connect(self._on_gesture_error)
        self.gesture_worker.start()

    def _stop_gesture_worker(self):
        if self.gesture_worker is not None:
            self.gesture_worker.stop()
            self.gesture_worker.wait(1500)
            self.gesture_worker = None
        self.last_hand_landmarks = None
        self.gesture_hold_start = None
        self.gesture_status_label.setText("")
        self.gesture_progress_bar.setValue(0)

    def _on_gesture_error(self, msg):
        self.gesture_checkbox.blockSignals(True)
        self.gesture_checkbox.setChecked(False)
        self.gesture_checkbox.blockSignals(False)
        self.gesture_enabled = False
        self._stop_gesture_worker()
        QMessageBox.critical(self, "Erro no gesto", msg)

    def _refresh_cameras(self):
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for idx, label in list_available_cameras(config.MAX_CAMERAS_TO_SCAN):
            self.camera_combo.addItem(label, idx)
        self.camera_combo.blockSignals(False)

    def _on_camera_changed(self, _idx):
        new_index = self.camera_combo.currentData()
        if self.camera_worker is not None:
            self.camera_worker.stop()
            self.camera_worker.wait(1500)
        self._start_camera(new_index)

    def _on_desync_toggled(self, state):
        self.desync_enabled = bool(state)

    def _on_gesture_toggled(self, state):
        self.gesture_enabled = bool(state)
        self.gesture_status_label.setVisible(self.gesture_enabled)
        self.gesture_progress_bar.setVisible(self.gesture_enabled)
        if self.gesture_enabled:
            self._start_gesture_worker()
        else:
            self._stop_gesture_worker()

    def _change_threshold(self, delta):
        new_val = round(self.conf_threshold + delta, 2)
        new_val = max(config.CONF_THRESHOLD_MIN, min(config.CONF_THRESHOLD_MAX, new_val))
        self.conf_threshold = new_val
        self._invalidate_ocr(time.time())
        self.threshold_label.setText(f"{self.conf_threshold * 100:.0f}%")
        if self.detection_worker is not None:
            self.detection_worker.set_conf_threshold(self.conf_threshold)

    def _on_camera_error(self, msg):
        self._invalidate_ocr(time.time())
        QMessageBox.critical(self, "Erro de câmera", msg)

    def _on_model_error(self, msg):
        self._invalidate_ocr(time.time())
        self.status_label.setText("Erro no modelo. Veja os detalhes")
        QMessageBox.critical(self, "Erro no modelo", msg)

    # ------------------------------------------------------------------
    # LOOP DE VIDEO
    # ------------------------------------------------------------------
    def _on_frame_ready(self, frame, frame_id):
        self.cam_fps_counter.tick()
        if self.desync_enabled:
            self._render_frame(frame, self.last_detections)

    def _on_detections_ready(self, detections, frame, frame_id):
        self.det_fps_counter.tick()
        if not self.desync_enabled:
            self._render_frame(frame, detections)
        else:
            self.last_detections = detections

        self._update_cart_logic(detections, frame)
        self._update_message()

    def _render_frame(self, frame_bgr, detections):
        frame_with_boxes = draw_detections(frame_bgr, detections, self.candidate_class)
        if self.gesture_enabled and self.last_hand_landmarks:
            frame_with_boxes = draw_hand_landmarks(
                frame_with_boxes, self.last_hand_landmarks, self.gesture_hold_start is not None
            )
        if config.SHOW_FPS:
            frame_with_boxes = draw_fps_overlay(
                frame_with_boxes, self.cam_fps_counter.fps, self.det_fps_counter.fps
            )
        pixmap = cv_frame_to_pixmap(frame_with_boxes)
        scaled = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)
        self._apply_rounded_mask()

    def _apply_rounded_mask(self):
        radius = config.UI_RADIUS
        rect = QRectF(self.video_label.rect())
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        region = QRegion(path.toFillPolygon().toPolygon())
        self.video_label.setMask(region)

    # ------------------------------------------------------------------
    # LOGICA DE CARRINHO (1 produto por vez, com tempo minimo + cooldown)
    # ------------------------------------------------------------------
    def _update_blocked(self, scores, min_total, now):
        """Classe recém adicionada só volta a ser considerada depois de
        RELEASE_ABSENCE_SECONDS (150 ms) sem ser detectada."""
        release = getattr(config, "RELEASE_ABSENCE_SECONDS", 0.15)
        for cls_name in list(self.blocked):
            s = scores.get(cls_name)
            if s is not None and s.total >= min_total - 1e-9:
                self.blocked[cls_name] = now          # ainda está na cena
            elif now - self.blocked[cls_name] >= release:
                del self.blocked[cls_name]

    def _update_cart_logic(self, detections, frame):
        now = time.time()

        if now < self.cooldown_until:
            self._reset_candidate()
            self.ocr_input.clear()
            self.status_label.setText("Aguardando... (evitando leitura duplicada)")
            self.progress_bar.setValue(0)
            return

        valid = [d for d in detections if d.cls_name in config.PRODUCTS]
        unblocked = [
            d for d in valid if d.cls_name not in self.blocked and d.conf >= self.conf_threshold
        ]
        top = max(unblocked, key=lambda d: d.conf, default=None)
        target = top.cls_name if top is not None else None
        if target != self.ocr_target:
            self.ocr_target = target
            self.ocr_target_since = now

        # alimenta o OCR (ignora o que já foi adicionado e continua na cena)
        use_crop = cfg("OCR_USE_BOX_CROP", True)
        if cfg("ENABLE_OCR", True) and self.ocr_available:
            if top is not None:
                self.ocr_input.set(frame, top.box if use_crop else None)
            elif not valid:
                if use_crop:
                    self._invalidate_ocr(now)  # cena vazia: esquece o rótulo anterior
                else:
                    self.ocr_input.set(frame, None)

        # pontuação: YOLO + desempate pelo centro + OCR
        ocr_matches = {}
        if self.ocr_updated_at is not None and now - self.ocr_updated_at <= cfg("OCR_RESULT_TTL_SECONDS", 3.0):
            ocr_matches = self.ocr_matches
        h, w = frame.shape[:2]
        scores = scoring.compute_scores(valid, (w, h), ocr_matches)
        min_total = scoring.min_total_for_conf(self.conf_threshold)
        self._update_blocked(scores, min_total, now)
        candidates = {c: s for c, s in scores.items() if c not in self.blocked}
        winner = scoring.pick_winner(candidates, min_total)
        self._update_debug_panel(scores, winner, min_total, now)

        if winner is None:
            self._reset_candidate()
            self.status_label.setText("Aguardando produto...")
            self.progress_bar.setValue(0)
            return

        best_name = winner.cls_name
        if self.candidate_class != best_name:
            self.candidate_class = best_name
            self.candidate_start = now

        elapsed = now - self.candidate_start
        progress = min(elapsed / config.MIN_DETECTION_SECONDS, 1.0)
        display_name = config.PRODUCTS.get(best_name, {}).get("display_name", best_name)
        suffix = f"  [{winner.total:.0f} pts]" if self.debug else ""
        self.status_label.setText(f"Lendo: {display_name}{suffix}")
        self.progress_bar.setValue(int(progress * 100))

        if elapsed >= config.MIN_DETECTION_SECONDS:
            self._confirm_product(best_name, now)

    def _confirm_product(self, class_name, now):
        self._add_to_cart(class_name)
        self.blocked[class_name] = now
        self.cooldown_until = now + config.COOLDOWN_SECONDS
        self._reset_candidate()
        self._invalidate_ocr(now)

    def _reset_candidate(self):
        self.candidate_class = None
        self.candidate_start = None

    def _add_to_cart(self, class_name):
        self.cart.add(class_name)
        self._refresh_cart_table()
        self._play_beep()

    def _remove_from_cart(self, class_name):
        self.cart.remove(class_name)
        self._refresh_cart_table()

    def _finish_purchase(self, from_gesture=False):
        """Encerra a compra (botão "Nova compra" ou gesto): mesmo pop-up."""
        if not self.cart.items:
            self._show_toast("Carrinho vazio\nInicie a compra")
            if from_gesture:
                self.gesture_status_label.setText("Carrinho vazio: nada a finalizar")
            return

        total = self.cart.total
        self._play_beep()
        self.cart.clear()
        self._refresh_cart_table()
        self._reset_candidate()
        self.cooldown_until = 0.0
        self._invalidate_ocr(time.time())
        if from_gesture:
            self.gesture_status_label.setText("Compra finalizada!")
            self.gesture_progress_bar.setValue(0)
        self._show_toast(f"Compra finalizada\nTotal: R$ {total:.2f}\nVolte sempre")

    def _new_purchase(self):
        self._finish_purchase()

    # ------------------------------------------------------------------
    # GESTO PARA FINALIZAR COMPRA
    # ------------------------------------------------------------------
    def _on_gesture_status(self, is_gesture, confidence, landmarks):
        self.last_hand_landmarks = landmarks
        now = time.time()

        if now < self.gesture_cooldown_until:
            self.gesture_hold_start = None
            self.gesture_status_label.setText("Aguardando...")
            self.gesture_progress_bar.setValue(0)
            return

        if not is_gesture:
            self.gesture_hold_start = None
            hint = "Mostre as duas mãos" if landmarks is None else "Gesto não reconhecido"
            self.gesture_status_label.setText(hint)
            self.gesture_progress_bar.setValue(0)
            return

        if self.gesture_hold_start is None:
            self.gesture_hold_start = now

        elapsed = now - self.gesture_hold_start
        progress = min(elapsed / config.GESTURE_HOLD_SECONDS, 1.0)
        self.gesture_status_label.setText(f"Segure o gesto... ({confidence * 100:.0f}%)")
        self.gesture_progress_bar.setValue(int(progress * 100))

        if elapsed >= config.GESTURE_HOLD_SECONDS:
            self.gesture_hold_start = None
            self.gesture_cooldown_until = now + config.GESTURE_COOLDOWN_SECONDS
            self._finalize_purchase_by_gesture()

    def _finalize_purchase_by_gesture(self):
        self._finish_purchase(from_gesture=True)

    def _refresh_cart_table(self):
        self.cart_table.setRowCount(0)
        for class_name, item in self.cart.items.items():
            row = self.cart_table.rowCount()
            self.cart_table.insertRow(row)
            self.cart_table.setItem(row, 0, QTableWidgetItem(item.display_name))
            self.cart_table.setItem(row, 1, QTableWidgetItem(str(item.qty)))
            self.cart_table.setItem(row, 2, QTableWidgetItem(f"R$ {item.price:.2f}"))
            self.cart_table.setItem(row, 3, QTableWidgetItem(f"R$ {item.subtotal:.2f}"))

            remove_btn = QPushButton("Remover")
            remove_btn.setObjectName("removeBtn")
            remove_btn.clicked.connect(lambda checked=False, cn=class_name: self._remove_from_cart(cn))
            self.cart_table.setCellWidget(row, 4, remove_btn)

        self.total_label.setText(f"Total: R$ {self.cart.total:.2f}")
        self._update_message()

    # ------------------------------------------------------------------
    # ENCERRAMENTO
    # ------------------------------------------------------------------
    @staticmethod
    def _stop_worker(worker, timeout_ms=1500):
        if worker is None:
            return
        worker.stop()
        if not worker.wait(timeout_ms):
            # ainda carregando um modelo (nao da pra interromper): sem isso o
            # Qt aborta ao sair ("QThread destroyed while thread is running")
            worker.terminate()
            worker.wait(500)

    def closeEvent(self, event):
        for worker in (self.camera_worker, self.detection_worker,
                       self.gesture_worker, self.ocr_worker):
            self._stop_worker(worker)
        event.accept()


def parse_args():
    parser = argparse.ArgumentParser(description=config.APP_TITLE)
    parser.add_argument("--debug", action="store_true",
                        help="mostra o painel com a pontuação (YOLO + OCR) de cada classe")
    return parser.parse_known_args()  # o resto vai pro Qt


def main():
    # .exe com --windowed nao tem console: bibliotecas que escrevem em
    # stdout/stderr (ex: RapidOCR) quebrariam com None
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")

    args, qt_args = parse_args()
    setup_logging()
    logging.info("MODEL_PATH resolvido para: %s (existe: %s) | debug=%s",
                 config.MODEL_PATH, os.path.exists(config.MODEL_PATH), args.debug)
    app = QApplication([sys.argv[0]] + qt_args)
    window = MainWindow(debug=args.debug)
    window.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":
    # OBRIGATORIO ao empacotar com PyInstaller no Windows quando alguma
    # dependencia (aqui, torch/ultralytics) pode usar multiprocessing.
    # Sem isso, o .exe relanca a si mesmo do zero a cada tentativa de
    # spawn - por isso janelas duplicadas e o modelo nunca terminando
    # de carregar.
    import multiprocessing
    multiprocessing.freeze_support()
    main()
