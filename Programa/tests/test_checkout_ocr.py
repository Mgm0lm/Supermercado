import unittest
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock, patch

from main import MainWindow


class CheckoutHarness:
    _update_cart_logic = MainWindow._update_cart_logic
    _on_ocr_ready = MainWindow._on_ocr_ready
    _update_blocked = MainWindow._update_blocked
    _invalidate_ocr = MainWindow._invalidate_ocr
    _reset_candidate = MainWindow._reset_candidate
    _confirm_product = MainWindow._confirm_product

    def __init__(self):
        self.ocr_input = Mock()
        self.status_label = Mock()
        self.progress_bar = Mock()
        self._update_debug_panel = Mock()
        self._update_message = Mock()
        self._add_to_cart = Mock()
        self.ocr_history = deque()
        self.ocr_target_since = 0.0
        self.ocr_available = True
        self.conf_threshold = 0.6
        self.cooldown_until = 0.0
        self.blocked = {}
        self.debug = False
        self._reset_candidate()
        self._invalidate_ocr(0.0)

    def detect(self, now, name):
        detections = [] if name is None else [
            SimpleNamespace(cls_name=name, conf=0.9, box=(0, 0, 30, 30))
        ]
        with patch("main.time.time", return_value=now):
            self._update_cart_logic(detections, SimpleNamespace(shape=(100, 100, 3)))

    def read(self, now, requested_at, text):
        with patch("main.time.time", return_value=now):
            self._on_ocr_ready([(text, 0.99)], now - requested_at, requested_at)


class CheckoutOcrTest(unittest.TestCase):
    def test_matching_read_launches_once_before_hold(self):
        scan = CheckoutHarness()
        scan.detect(10, "Milho")
        scan.read(10.1, 10, "milho")
        scan._add_to_cart.assert_called_once_with("Milho")
        scan.detect(10.3, "Milho")
        scan.detect(13.0, "Milho")
        scan._add_to_cart.assert_called_once()

    def test_divergence_preserves_hold(self):
        scan = CheckoutHarness()
        scan.detect(10, "Milho")
        scan.read(10.1, 10, "mostarda flamboyant")
        scan.detect(10.2, "Milho")
        scan.detect(11.6, "Milho")
        scan._add_to_cart.assert_not_called()
        scan.detect(11.8, "Milho")
        scan._add_to_cart.assert_called_once_with("Mostarda")

    def test_delayed_read_for_previous_target_cannot_skip_hold(self):
        scan = CheckoutHarness()
        scan.detect(10, "Milho")
        scan.detect(10.2, "Mostarda")
        scan.read(10.3, 10, "mostarda flamboyant")
        scan.detect(10.4, "Mostarda")
        scan._add_to_cart.assert_not_called()

    def test_aliases_with_same_name_and_price_can_confirm(self):
        scan = CheckoutHarness()
        scan.detect(10, "Café")
        scan.read(10.1, 10, "café 3 corações")
        scan._add_to_cart.assert_called_once_with("Café")
