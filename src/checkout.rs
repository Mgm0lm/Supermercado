use crate::{config::Config, detector::Detection};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Default)]
pub struct Score {
    pub conf: f32,
    pub yolo: f32,
    pub center: f32,
    pub ocr: f32,
}

impl Score {
    pub fn total(&self) -> f32 {
        self.yolo + self.center + self.ocr
    }
}

pub type Keywords = BTreeMap<String, Vec<String>>;

pub fn parse_keywords(bytes: &[u8]) -> serde_json::Result<Keywords> {
    let entries: BTreeMap<String, serde_json::Value> = serde_json::from_slice(bytes)?;
    Ok(entries
        .into_iter()
        .filter_map(|(class, words)| {
            if class.starts_with('_') {
                return None;
            }
            let words = words
                .as_array()?
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_owned)
                .collect();
            Some((class, words))
        })
        .collect())
}

fn normalize(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|c| match c {
            'à' | 'á' | 'â' | 'ã' | 'ä' => 'a',
            'è' | 'é' | 'ê' | 'ë' => 'e',
            'ì' | 'í' | 'î' | 'ï' => 'i',
            'ò' | 'ó' | 'ô' | 'õ' | 'ö' => 'o',
            'ù' | 'ú' | 'û' | 'ü' => 'u',
            'ç' => 'c',
            'ñ' => 'n',
            'a'..='z' | '0'..='9' => c,
            _ => ' ',
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn matching_chars(a: &[u8], b: &[u8]) -> usize {
    let (mut start_a, mut start_b, mut length) = (0, 0, 0);
    for i in 0..a.len() {
        for j in 0..b.len() {
            let count = a[i..]
                .iter()
                .zip(&b[j..])
                .take_while(|(x, y)| x == y)
                .count();
            if count > length {
                (start_a, start_b, length) = (i, j, count);
            }
        }
    }
    if length == 0 {
        return 0;
    }
    length
        + matching_chars(&a[..start_a], &b[..start_b])
        + matching_chars(&a[start_a + length..], &b[start_b + length..])
}

fn similarity(key: &str, token: &str, threshold: f32) -> f32 {
    if key == token || (key.len() >= 5 && token.contains(key)) {
        return 1.0;
    }
    if key.len() < 5 || token.len() < 4 {
        return 0.0;
    }
    let combined = key.len() + token.len();
    if 2.0 * key.len().min(token.len()) as f32 / (combined as f32) < threshold {
        return 0.0;
    }
    let ratio = 2.0 * matching_chars(key.as_bytes(), token.as_bytes()) as f32 / combined as f32;
    if ratio >= threshold { ratio } else { 0.0 }
}

pub fn scores(
    cfg: &Config,
    detections: &[Detection],
    size: (u32, u32),
    confidence: f32,
    text: &str,
    keywords: &Keywords,
) -> BTreeMap<String, Score> {
    let mut best = BTreeMap::<&str, &Detection>::new();
    for detection in detections {
        if detection.confidence >= confidence
            && cfg.products.contains_key(&detection.class)
            && best
                .get(detection.class.as_str())
                .is_none_or(|prev| prev.confidence < detection.confidence)
        {
            best.insert(&detection.class, detection);
        }
    }
    let mut scores: BTreeMap<_, _> = best
        .iter()
        .map(|(name, d)| {
            (
                name.to_string(),
                Score {
                    conf: d.confidence,
                    yolo: cfg.score_yolo_max_points * d.confidence,
                    ..Score::default()
                },
            )
        })
        .collect();
    let top_conf = best.values().map(|d| d.confidence).fold(0.0, f32::max);
    let (center_x, center_y) = (size.0 as f32 / 2.0, size.1 as f32 / 2.0);
    let mut distances = best
        .values()
        .filter(|d| top_conf - d.confidence <= cfg.score_center_tie_margin + 1e-7)
        .map(|d| {
            let [x1, y1, x2, y2] = d.bounds;
            let dist = ((x1 + x2) / 2.0 - center_x).hypot((y1 + y2) / 2.0 - center_y);
            (dist, &d.class)
        })
        .collect::<Vec<_>>();
    if distances.len() >= 2 {
        distances.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (dist, class) = distances[0];
        let diag = center_x.hypot(center_y).max(1.0);
        scores.get_mut(class).unwrap().center =
            cfg.score_center_max_points * (1.0 - dist / diag).max(0.0);
    }
    let normalized = normalize(text);
    let tokens: Vec<_> = normalized.split_whitespace().collect();
    if !tokens.is_empty() {
        for (class, keys) in keywords {
            if !cfg.products.contains_key(class) {
                continue;
            }
            let unique: BTreeSet<_> = keys
                .iter()
                .map(|key| normalize(key))
                .filter(|k| !k.is_empty())
                .collect();
            let needed = cfg
                .ocr_keywords_for_full_score
                .min(unique.len() as f32)
                .max(0.1);
            let matches: f32 = unique
                .iter()
                .map(|key| {
                    tokens
                        .iter()
                        .map(|token| similarity(key, token, cfg.ocr_fuzzy_threshold))
                        .fold(0.0, f32::max)
                })
                .sum();
            if matches > 0.0 {
                scores.entry(class.clone()).or_default().ocr =
                    cfg.score_ocr_max_points * (matches / needed).min(1.0);
            }
        }
    }
    scores
}

fn same_product(cfg: &Config, first: &str, second: &str) -> bool {
    first == second
        || cfg
            .products
            .get(first)
            .zip(cfg.products.get(second))
            .is_some_and(|(a, b)| a.display_name == b.display_name && a.price == b.price)
}

fn ocr_confirms(cfg: &Config, scores: &BTreeMap<String, Score>, target: &str) -> bool {
    let (mut own, mut other) = (0.0_f32, 0.0_f32);
    for (name, score) in scores {
        if same_product(cfg, name, target) {
            own = own.max(score.ocr);
        } else {
            other = other.max(score.ocr);
        }
    }
    own > other + 1e-6
}

#[derive(Default)]
pub struct Checkout {
    pub cart: Vec<(String, u32)>,
    pub blocked: BTreeMap<String, f64>,
    pub candidate: Option<(String, f64)>,
    pub cooldown_until: f64,
    pub gesture_start: Option<f64>,
    pub gesture_cooldown_until: f64,
}

impl Checkout {
    pub fn confirm_ocr(
        &mut self,
        now: f64,
        cfg: &Config,
        threshold: f32,
        scores: &BTreeMap<String, Score>,
        target: &str,
    ) -> bool {
        if now < self.cooldown_until
            || !scores
                .get(target)
                .is_some_and(|score| score.conf >= threshold)
            || !ocr_confirms(cfg, scores, target)
            || self
                .blocked
                .keys()
                .any(|name| same_product(cfg, name, target))
        {
            return false;
        }
        self.add_product(now, cfg, target);
        true
    }

    pub fn product(
        &mut self,
        now: f64,
        cfg: &Config,
        threshold: f32,
        scores: &BTreeMap<String, Score>,
    ) -> bool {
        if now < self.cooldown_until {
            self.candidate = None;
            return false;
        }
        let minimum = threshold * cfg.score_yolo_max_points;
        self.blocked.retain(|name, seen| {
            if scores.get(name).is_some_and(|s| s.total() >= minimum) {
                *seen = now;
            }
            now - *seen < cfg.release_absence_seconds
        });
        let winner = scores
            .iter()
            .filter(|(name, s)| s.total() >= minimum && !self.blocked.contains_key(*name))
            .max_by(|(_, a), (_, b)| {
                a.total()
                    .total_cmp(&b.total())
                    .then(a.conf.total_cmp(&b.conf))
            });
        let Some((name, _)) = winner else {
            self.candidate = None;
            return false;
        };
        let (current, start) = self.candidate.get_or_insert_with(|| (name.clone(), now));
        if current != name {
            *current = name.clone();
            *start = now;
        }
        if now - *start < cfg.min_detection_seconds {
            return false;
        }
        self.add_product(now, cfg, name);
        true
    }

    fn add_product(&mut self, now: f64, cfg: &Config, name: &str) {
        if let Some((_, qty)) = self.cart.iter_mut().find(|(class, _)| class == name) {
            *qty += 1;
        } else {
            self.cart.push((name.to_owned(), 1));
        }
        self.blocked.insert(name.to_owned(), now);
        self.cooldown_until = now + cfg.cooldown_seconds;
        self.candidate = None;
    }

    pub fn gesture(&mut self, now: f64, cfg: &Config, confidence: f64) -> bool {
        if now < self.gesture_cooldown_until || confidence < cfg.gesture_min_confidence {
            self.gesture_start = None;
            return false;
        }
        let start = self.gesture_start.get_or_insert(now);
        if now - *start < cfg.gesture_hold_seconds {
            return false;
        }
        self.gesture_cooldown_until = now + cfg.gesture_cooldown_seconds;
        self.gesture_start = None;
        true
    }

    pub fn total(&self, cfg: &Config) -> f64 {
        self.cart
            .iter()
            .map(|(class, qty)| cfg.products[class].price * f64::from(*qty))
            .sum()
    }

    pub fn finish(&mut self, cfg: &Config) -> Option<f64> {
        if self.cart.is_empty() {
            return None;
        }
        let total = self.total(cfg);
        self.cart.clear();
        self.cooldown_until = 0.0;
        self.candidate = None;
        Some(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn packaged_keywords_allow_comments() {
        let keywords = parse_keywords(include_bytes!("../Programa/ocr_keywords.json")).unwrap();
        assert!(!keywords.contains_key("_comentario"));
        assert!(keywords["Café"].iter().any(|word| word == "cafe"));
    }

    #[test]
    fn requires_hold_and_absence_between_two_scans() {
        let cfg = Config::load(Path::new(".")).unwrap();
        let mut checkout = Checkout::default();
        let scores = BTreeMap::from([(
            "Café".into(),
            Score {
                yolo: 32.0,
                ..Score::default()
            },
        )]);
        assert!(!checkout.product(0.0, &cfg, 0.6, &scores));
        assert!(!checkout.product(1.4, &cfg, 0.6, &scores));
        assert!(checkout.product(1.5, &cfg, 0.6, &scores));
        assert!(!checkout.product(5.0, &cfg, 0.6, &scores));
        assert!(!checkout.product(5.2, &cfg, 0.6, &BTreeMap::new()));
        assert!(!checkout.product(5.3, &cfg, 0.6, &scores));
        assert!(checkout.product(7.0, &cfg, 0.6, &scores));
        assert_eq!(checkout.finish(&cfg), Some(51.0));
        assert!(checkout.cart.is_empty());
    }

    #[test]
    fn requires_a_continuous_gesture() {
        let cfg = Config::load(Path::new(".")).unwrap();
        let mut checkout = Checkout::default();
        assert!(!checkout.gesture(0.0, &cfg, 0.95));
        assert!(!checkout.gesture(1.4, &cfg, 0.2));
        assert!(!checkout.gesture(1.5, &cfg, 0.95));
        assert!(checkout.gesture(3.0, &cfg, 0.95));
        assert!(!checkout.gesture(4.5, &cfg, 0.95));
    }

    #[test]
    fn fresh_agreement_skips_hold_and_keeps_duplicate_protection() {
        let cfg = Config::load(Path::new(".")).unwrap();
        let keywords = parse_keywords(include_bytes!("../Programa/ocr_keywords.json")).unwrap();
        let detection = Detection {
            class: "Café".into(),
            confidence: 0.9,
            bounds: [0.0, 0.0, 40.0, 40.0],
        };
        let scores = scores(
            &cfg,
            &[detection],
            (100, 100),
            0.6,
            "café 3 corações",
            &keywords,
        );
        let mut checkout = Checkout::default();
        assert!(!checkout.product(0.0, &cfg, 0.6, &scores));
        assert!(checkout.confirm_ocr(0.1, &cfg, 0.6, &scores, "Café"));
        assert!(!checkout.confirm_ocr(0.2, &cfg, 0.6, &scores, "Café"));
        assert!(!checkout.confirm_ocr(3.0, &cfg, 0.6, &scores, "Café"));
        assert_eq!(checkout.cart, vec![("Café".into(), 1)]);
    }

    #[test]
    fn divergent_or_ambiguous_ocr_keeps_hold() {
        let cfg = Config::load(Path::new(".")).unwrap();
        let keywords = parse_keywords(include_bytes!("../Programa/ocr_keywords.json")).unwrap();
        let detection = Detection {
            class: "Milho".into(),
            confidence: 0.9,
            bounds: [0.0, 0.0, 40.0, 40.0],
        };
        for text in ["mostarda flamboyant", "milho mostarda flamboyant"] {
            let scores = scores(
                &cfg,
                std::slice::from_ref(&detection),
                (100, 100),
                0.6,
                text,
                &keywords,
            );
            let mut checkout = Checkout::default();
            assert!(!checkout.confirm_ocr(0.0, &cfg, 0.6, &scores, "Milho"));
            assert!(!checkout.product(0.0, &cfg, 0.6, &scores));
            assert!(!checkout.confirm_ocr(1.4, &cfg, 0.6, &scores, "Milho"));
            assert!(checkout.product(1.5, &cfg, 0.6, &scores));
        }
    }

    #[test]
    fn scores_accented_and_imperfect_ocr_words() {
        let cfg = Config::load(Path::new(".")).unwrap();
        let keywords = BTreeMap::from([("Café".into(), vec!["café".into(), "corações".into()])]);
        let scores = scores(&cfg, &[], (100, 100), 0.6, "CAFÉ CORACOES500g", &keywords);
        assert_eq!(scores["Café"].total(), 60.0);
        assert!(similarity("predilecta", "predilccta", 0.85) >= 0.85);
    }
}
