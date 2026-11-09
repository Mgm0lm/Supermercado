from config import UI_RADIUS

STYLESHEET = f"""
QMainWindow, QWidget#root {{
    background-color: #ffffff;
}}

QLabel {{
    color: #1f2430;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}}

QLabel#videoLabel {{
    background-color: #101318;
    border-radius: {UI_RADIUS}px;
}}

QFrame#card {{
    background-color: #f4f6f9;
    border: 1px solid #e3e7ee;
    border-radius: {UI_RADIUS}px;
}}

QLabel#titulo {{
    font-size: 20px;
    font-weight: 600;
}}

QLabel#subtitulo {{
    font-size: 13px;
    color: #6b7280;
}}

QLabel#totalLabel {{
    font-size: 22px;
    font-weight: 700;
    color: #0f8a4c;
}}

QLabel#statusDeteccao {{
    font-size: 14px;
    font-weight: 600;
    color: #2563eb;
}}

QPushButton {{
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: {UI_RADIUS - 4}px;
    padding: 8px 14px;
    font-size: 14px;
    font-weight: 600;
}}

QPushButton:hover {{
    background-color: #1d4ed8;
}}

QPushButton:pressed {{
    background-color: #1e40af;
}}

QPushButton#thresholdBtn {{
    min-width: 36px;
    max-width: 36px;
    font-size: 18px;
    padding: 4px;
}}

QPushButton#removeBtn {{
    background-color: #ef4444;
    padding: 4px 10px;
    font-size: 12px;
}}

QPushButton#removeBtn:hover {{
    background-color: #dc2626;
}}

QPushButton#secondaryBtn {{
    background-color: #ffffff;
    color: #374151;
    border: 1px solid #d0d5dd;
    padding: 6px 12px;
    font-size: 12px;
}}

QPushButton#secondaryBtn:hover {{
    background-color: #f4f6f9;
}}

QComboBox, QCheckBox {{
    font-size: 13px;
    padding: 4px;
}}

QComboBox {{
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    padding: 6px 8px;
    background-color: white;
}}

QProgressBar {{
    border: 1px solid #d0d5dd;
    border-radius: 8px;
    background-color: #eef1f5;
    height: 14px;
    text-align: center;
    font-size: 10px;
}}

QProgressBar::chunk {{
    background-color: #2563eb;
    border-radius: 7px;
}}

QLabel#toast {{
    background-color: #ffffff;
    color: #000000;
    border: 5px solid #000000;
    border-radius: {UI_RADIUS * 2}px;
    padding: 70px 140px;
    font-size: 72px;
    font-weight: 700;
}}

QLabel#mensagem {{
    font-size: 48px;
    font-weight: 400;
    color: #111111;
}}

QLabel#debugText {{
    font-family: Consolas, "Courier New", monospace;
    font-size: 11px;
    color: #1f2430;
}}

QTableWidget {{
    background-color: white;
    border: 1px solid #e3e7ee;
    border-radius: {UI_RADIUS}px;
    gridline-color: #eef1f5;
    font-size: 13px;
}}

QHeaderView::section {{
    background-color: #f4f6f9;
    border: none;
    padding: 6px;
    font-weight: 600;
    color: #4b5563;
}}
"""
