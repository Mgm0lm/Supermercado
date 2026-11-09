"""
Sistema de pontos que combina YOLO + OCR para decidir qual produto
está na frente da câmera (sem depender de um único modelo).

Pontuação por classe (valores em config.py):
- YOLO: até 40 pts = 40 x confiança da melhor caixa daquela classe.
- Desempate por posição: se duas ou mais classes estão acima da
  confiança mínima e com até 8% de diferença entre si (em relação à
  melhor), a que estiver mais perto do centro da imagem ganha até 5 pts.
- OCR: até 60 pts se o texto lido no rótulo contiver palavras
  específicas da classe (OCR_KEYWORDS).

Uma classe é aceita se o total >= 40 x precisão mínima (os botões +/-
da interface continuam valendo: sem OCR, o comportamento é idêntico ao
de antes). Este módulo é puro (sem Qt/YOLO/OCR), fácil de testar.
"""

import math
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import config


def _cfg(name, default):
    # getattr: tolera um config.py local desatualizado, sem quebrar
    return getattr(config, name, default)


# ---------------------------------------------------------------- texto
def normalize_text(text):
    """minúsculas, sem acentos, só letras/números separados por espaço."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def tokenize(texts):
    tokens = []
    for t in texts:
        tokens.extend(normalize_text(t).split())
    return tokens


def _similarity(keyword, tokens, threshold):
    """1.0 = palavra exata; <1.0 = parecida (erro de leitura do OCR)."""
    best = 0.0
    for tok in tokens:
        if tok == keyword:
            return 1.0
        if len(keyword) >= 5:
            if keyword in tok:  # OCR às vezes cola palavras: "predilecta300g"
                best = max(best, 1.0)
                continue
            if len(tok) >= 4:
                ratio = SequenceMatcher(None, keyword, tok).ratio()
                if ratio >= threshold:
                    best = max(best, ratio)
    return best


@dataclass
class OcrMatch:
    points: float
    words: list = field(default_factory=list)


def match_keywords(texts, keywords_by_class=None):
    """Retorna {classe: OcrMatch} para as classes com palavras achadas.
    Cada palavra-chave achada vale 1 (ou a semelhança, se for aproximada);
    pontos = máx x min(1, soma / OCR_KEYWORDS_FOR_FULL_SCORE)."""
    if keywords_by_class is None:
        keywords_by_class = _cfg("OCR_KEYWORDS", {})
    max_points = float(_cfg("SCORE_OCR_MAX_POINTS", 60))
    full_matches = max(0.1, float(_cfg("OCR_KEYWORDS_FOR_FULL_SCORE", 2)))
    threshold = float(_cfg("OCR_FUZZY_THRESHOLD", 0.85))

    tokens = tokenize(texts)
    if not tokens:
        return {}

    result = {}
    for cls_name, keywords in keywords_by_class.items():
        total, words = 0.0, []
        unique = [k for k in dict.fromkeys(normalize_text(k) for k in keywords) if k]
        needed = max(0.1, min(full_matches, len(unique)))  # 1 palavra so: 1 acerto basta
        for kw in unique:
            if not kw:
                continue
            sim = _similarity(kw, tokens, threshold)
            if sim > 0:
                total += sim
                words.append(kw)
        if total > 0:
            result[cls_name] = OcrMatch(max_points * min(1.0, total / needed), words)
    return result


def same_product(first, second):
    a, b = config.PRODUCTS.get(first), config.PRODUCTS.get(second)
    return first == second or (
        a is not None and b is not None
        and (a["display_name"], a["price"]) == (b["display_name"], b["price"])
    )


def ocr_confirms(class_name, matches):
    """A leitura atual deve favorecer o mesmo produto, sem empate com outro."""
    own, other = 0.0, 0.0
    for name, match in matches.items():
        if name not in config.PRODUCTS:
            continue
        if same_product(name, class_name):
            own = max(own, match.points)
        else:
            other = max(other, match.points)
    return own > other + 1e-6


# ---------------------------------------------------------------- pontos
@dataclass
class ClassScore:
    cls_name: str
    conf: float = None          # None = nenhuma caixa do YOLO pra essa classe
    yolo: float = 0.0
    center: float = 0.0
    ocr: float = 0.0
    ocr_words: list = field(default_factory=list)

    @property
    def total(self):
        return self.yolo + self.center + self.ocr


def min_total_for_conf(min_conf):
    """Pontuação mínima pra aceitar = pontos de YOLO de uma detecção
    exatamente na confiança mínima (mantém o sentido dos botões +/-)."""
    return float(_cfg("SCORE_YOLO_MAX_POINTS", 40)) * min_conf


def compute_scores(detections, image_size, ocr_matches=None, products=None):
    """detections: objetos com .cls_name, .conf, .box (x1,y1,x2,y2), já
    acima da confiança mínima. image_size: (largura, altura)."""
    products = _cfg("PRODUCTS", {}) if products is None else products
    yolo_max = float(_cfg("SCORE_YOLO_MAX_POINTS", 40))
    center_max = float(_cfg("SCORE_CENTER_MAX_POINTS", 5))
    margin = float(_cfg("SCORE_CENTER_TIE_MARGIN", 0.08))
    w, h = image_size

    scores, best_box = {}, {}
    for det in detections:
        if det.cls_name not in products:
            continue
        cur = scores.get(det.cls_name)
        if cur is None or det.conf > cur.conf:
            scores[det.cls_name] = ClassScore(det.cls_name, det.conf, yolo_max * det.conf)
            best_box[det.cls_name] = det.box

    # desempate: entre as classes coladas na melhor, a mais central ganha
    if len(scores) >= 2:
        top_conf = max(s.conf for s in scores.values())
        close = [c for c, s in scores.items() if top_conf - s.conf <= margin + 1e-9]
        if len(close) >= 2:
            half_diag = math.hypot(w / 2, h / 2)

            def dist(c):
                x1, y1, x2, y2 = best_box[c]
                return math.hypot((x1 + x2) / 2 - w / 2, (y1 + y2) / 2 - h / 2)

            dists = {c: dist(c) for c in close}
            nearest = min(dists, key=dists.get)
            scores[nearest].center = center_max * max(0.0, 1.0 - dists[nearest] / half_diag)

    for cls_name, match in (ocr_matches or {}).items():
        if cls_name not in products:
            continue
        s = scores.setdefault(cls_name, ClassScore(cls_name))
        s.ocr = match.points
        s.ocr_words = list(match.words)

    return scores


def pick_winner(scores, min_total):
    ok = [s for s in scores.values() if s.total >= min_total - 1e-9]
    if not ok:
        return None
    return max(ok, key=lambda s: (s.total, s.conf if s.conf is not None else -1.0))


def format_debug_text(scores, winner, min_total, ocr_enabled, ocr_texts, ocr_age, ocr_elapsed, blocked=()):
    """Texto do painel --debug."""
    lines = [f"DEBUG: mínimo p/ aceitar {min_total:.1f} pts (x = bloqueada)",
             f"{'classe':<17}{'yolo':>6}{'cent':>6}{'ocr':>6}{'total':>7}"]
    ordered = sorted(scores.values(), key=lambda s: s.total, reverse=True)[:5]
    if not ordered:
        lines.append("(nenhuma classe pontuando)")
    for s in ordered:
        mark = ">" if winner is not None and s.cls_name == winner.cls_name else ("x" if s.cls_name in blocked else " ")
        lines.append(f"{mark}{s.cls_name[:16]:<16}{s.yolo:6.1f}{s.center:6.1f}{s.ocr:6.1f}{s.total:7.1f}")
    if not ocr_enabled:
        lines.append("OCR: desativado ou indisponível")
    elif ocr_age is None:
        lines.append("OCR: aguardando leitura...")
    else:
        text = " | ".join(ocr_texts)[:70] or "(nada lido)"
        lines.append(f"OCR ({ocr_elapsed * 1000:.0f} ms, {ocr_age:.1f}s atrás): {text}")
    found = [f"{s.cls_name}: {', '.join(s.ocr_words)}" for s in ordered if s.ocr_words]
    if found:
        lines.append("palavras: " + "; ".join(found))
    return "\n".join(lines)
