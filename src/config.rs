use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Deserialize)]
pub struct Product {
    pub display_name: String,
    pub price: f64,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub struct Config {
    pub app_title: String,
    pub products: BTreeMap<String, Product>,
    pub model_path: PathBuf,
    pub gesture_model_path: PathBuf,
    pub hand_landmarker_model_path: PathBuf,
    pub camera_panel_ratio: f32,
    pub camera_frame_width: u32,
    pub camera_frame_height: u32,
    pub camera_fps_target: u32,
    pub max_cameras_to_scan: usize,
    pub desync_bboxes: bool,
    pub conf_threshold_default: f32,
    pub conf_threshold_min: f32,
    pub conf_threshold_max: f32,
    pub conf_threshold_step: f32,
    pub min_detection_seconds: f64,
    pub cooldown_seconds: f64,
    pub release_absence_seconds: f64,
    pub enable_gesture_checkout_default: bool,
    pub gesture_hold_seconds: f64,
    pub gesture_cooldown_seconds: f64,
    pub gesture_min_confidence: f64,
    pub mediapipe_min_detection_confidence: f32,
    pub mediapipe_min_tracking_confidence: f32,
    pub score_yolo_max_points: f32,
    pub score_center_max_points: f32,
    pub score_center_tie_margin: f32,
    pub score_ocr_max_points: f32,
    pub enable_ocr: bool,
    pub ocr_interval_seconds: f64,
    pub ocr_result_ttl_seconds: f64,
    pub ocr_use_box_crop: bool,
    pub ocr_box_padding: f32,
    pub ocr_max_side: u32,
    pub ocr_det_side_len: u32,
    pub ocr_min_text_score: f32,
    pub ocr_threads: usize,
    pub ocr_keywords_for_full_score: f32,
    pub ocr_fuzzy_threshold: f32,
    pub show_fps: bool,
    pub toast_seconds: f64,
    pub enable_sound: bool,
    pub sound_freq_hz: u32,
    pub sound_duration_ms: u32,
    pub sound_volume: f32,
}

impl Config {
    pub fn load(base: &Path) -> Result<Self> {
        let mut defaults: serde_json::Value =
            serde_json::from_str(include_str!("../rust/defaults.json"))?;
        match fs::read(base.join("config_override.json")) {
            Ok(bytes) => {
                let overrides: BTreeMap<String, serde_json::Value> =
                    serde_json::from_slice(&bytes)?;
                for (key, value) in overrides {
                    if defaults.get(&key).is_some() {
                        defaults[key] = value;
                    }
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err.into()),
        }
        let mut config: Self = serde_json::from_value(defaults)?;
        if !(0.0..=1.0).contains(&config.conf_threshold_min)
            || !(config.conf_threshold_min..=1.0).contains(&config.conf_threshold_max)
            || config.min_detection_seconds <= 0.0
            || config.gesture_hold_seconds <= 0.0
            || !(0.2..=0.8).contains(&config.camera_panel_ratio)
            || config.camera_fps_target == 0
            || config.camera_frame_width == 0
            || config.camera_frame_height == 0
            || config.ocr_max_side == 0
            || !(32..=2000).contains(&config.ocr_det_side_len)
        {
            return Err("Valores fora da faixa em config_override.json".into());
        }
        config.model_path = base.join(&config.model_path);
        config.gesture_model_path = base.join(&config.gesture_model_path);
        config.hand_landmarker_model_path = base.join(&config.hand_landmarker_model_path);
        Ok(config)
    }
}
