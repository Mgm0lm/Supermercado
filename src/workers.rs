use crate::{
    config::{Config, Result},
    detector::{Detection, Detector},
    gesture::{self, Hand, HandTracker, Mlp},
    ocr::Reader,
};
use image::RgbImage;
use nokhwa::{
    Camera,
    pixel_format::RgbFormat,
    utils::{
        ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
        Resolution,
    },
};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

pub struct Frame {
    pub id: u64,
    pub pixels: RgbImage,
    pub captured: Instant,
}

pub struct Found {
    pub frame: Arc<Frame>,
    pub detections: Vec<Detection>,
    pub finished: Instant,
}

pub struct Gesture {
    pub frame: Arc<Frame>,
    pub hands: Vec<Hand>,
    pub confidence: f64,
}

#[derive(Clone)]
pub struct OcrRequest {
    pub frame: Arc<Frame>,
    pub bounds: Option<[f32; 4]>,
    pub token: u64,
}

pub struct OcrText {
    pub text: String,
    pub token: u64,
    pub frame_id: u64,
    pub captured: Instant,
    pub finished: Instant,
    pub duration: Duration,
}

pub struct Shared {
    pub stop: bool,
    pub camera_epoch: u64,
    pub camera: Option<CameraIndex>,
    pub cameras: Vec<(CameraIndex, String)>,
    pub refresh: bool,
    pub camera_error: String,
    pub detector_status: String,
    pub gesture_error: String,
    pub ocr_error: String,
    pub frame: Option<Arc<Frame>>,
    pub found: Option<Arc<Found>>,
    pub gesture: Option<Arc<Gesture>>,
    pub ocr_request: Option<OcrRequest>,
    pub ocr: Option<Arc<OcrText>>,
    pub threshold: f32,
    pub gesture_enabled: bool,
    pub gesture_epoch: u64,
}

impl Shared {
    fn new(cfg: &Config) -> Self {
        Self {
            stop: false,
            camera_epoch: 0,
            camera: None,
            cameras: Vec::new(),
            refresh: true,
            camera_error: String::new(),
            detector_status: "Carregando modelo YOLO26...".into(),
            gesture_error: String::new(),
            ocr_error: String::new(),
            frame: None,
            found: None,
            gesture: None,
            ocr_request: None,
            ocr: None,
            threshold: cfg
                .conf_threshold_default
                .clamp(cfg.conf_threshold_min, cfg.conf_threshold_max),
            gesture_enabled: cfg.enable_gesture_checkout_default,
            gesture_epoch: 0,
        }
    }

    pub fn select_camera(&mut self, camera: Option<CameraIndex>) {
        self.camera_epoch += 1;
        self.camera = camera;
        self.frame = None;
        self.found = None;
        self.gesture = None;
        self.ocr_request = None;
        self.ocr = None;
        self.camera_error.clear();
    }
}

pub type Workers = Arc<Mutex<Shared>>;

fn idle() {
    thread::sleep(Duration::from_millis(12));
}

pub fn open_camera(index: CameraIndex, cfg: &Config) -> Result<Camera> {
    let requested = CameraFormat::new(
        Resolution::new(cfg.camera_frame_width, cfg.camera_frame_height),
        FrameFormat::MJPEG,
        cfg.camera_fps_target,
    );
    let best = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(requested));
    let mut camera = Camera::with_backend(index.clone(), best, ApiBackend::MediaFoundation)
        .or_else(|_| {
            Camera::with_backend(
                index,
                RequestedFormat::new::<RgbFormat>(RequestedFormatType::None),
                ApiBackend::MediaFoundation,
            )
        })?;
    camera.open_stream()?;
    Ok(camera)
}

pub fn frame(camera: &mut Camera) -> Result<RgbImage> {
    let buffer = camera.frame()?;
    if buffer.source_frame_format() == FrameFormat::MJPEG {
        return Ok(
            image::load_from_memory_with_format(buffer.buffer(), image::ImageFormat::Jpeg)?
                .into_rgb8(),
        );
    }
    Ok(buffer.decode_image::<RgbFormat>()?)
}

fn capture(shared: Workers, cfg: Config) {
    let mut active = None;
    let mut epoch = u64::MAX;
    let mut id = 0;
    loop {
        let (refresh, target, next_epoch) = {
            let mut state = shared.lock().unwrap();
            if state.stop {
                return;
            }
            (
                std::mem::take(&mut state.refresh),
                state.camera.clone(),
                state.camera_epoch,
            )
        };
        if refresh {
            active = None;
            let cameras = nokhwa::query(ApiBackend::MediaFoundation).map(|cameras| {
                cameras
                    .into_iter()
                    .take(cfg.max_cameras_to_scan)
                    .enumerate()
                    .map(|(i, camera)| (camera.index().clone(), format!("Câmera {i}")))
                    .collect::<Vec<_>>()
            });
            let mut state = shared.lock().unwrap();
            match cameras {
                Ok(cameras) => {
                    let selected = state
                        .camera
                        .clone()
                        .filter(|id| cameras.iter().any(|(camera, _)| camera == id))
                        .or_else(|| cameras.first().map(|(camera, _)| camera.clone()));
                    state.cameras = cameras;
                    state.select_camera(selected);
                    if state.camera.is_none() {
                        state.camera_error =
                            "Nenhuma câmera encontrada. Use ↻ para atualizar.".into();
                    }
                }
                Err(error) => {
                    state.select_camera(None);
                    state.cameras.clear();
                    state.camera_error = format!("Falha ao listar câmeras: {error}");
                }
            }
            continue;
        }
        if next_epoch != epoch {
            active = None;
            epoch = next_epoch;
            if let Some(index) = target {
                match open_camera(index, &cfg) {
                    Ok(camera) => active = Some(camera),
                    Err(error) => {
                        let mut state = shared.lock().unwrap();
                        if state.camera_epoch == epoch {
                            state.camera_error =
                                format!("Não foi possível abrir a câmera: {error}");
                        }
                    }
                }
            }
        }
        let Some(camera) = &mut active else {
            idle();
            continue;
        };
        let started = Instant::now();
        match frame(camera) {
            Ok(pixels) => {
                id += 1;
                let mut state = shared.lock().unwrap();
                if state.camera_epoch == epoch {
                    state.camera_error.clear();
                    state.frame = Some(Arc::new(Frame {
                        id,
                        pixels,
                        captured: Instant::now(),
                    }));
                }
            }
            Err(error) => {
                let mut state = shared.lock().unwrap();
                if state.camera_epoch == epoch {
                    state.camera_error = format!("Câmera sem vídeo: {error}");
                    state.frame = None;
                    state.found = None;
                    state.gesture = None;
                    state.ocr_request = None;
                    state.ocr = None;
                }
            }
        }
        thread::sleep(
            Duration::from_secs_f64(1.0 / f64::from(cfg.camera_fps_target))
                .saturating_sub(started.elapsed()),
        );
    }
}

fn detect(shared: Workers, cfg: Config, base: PathBuf) {
    let mut detector = match Detector::new(&base, &cfg) {
        Ok(detector) => detector,
        Err(error) => {
            shared.lock().unwrap().detector_status = format!("Falha ao carregar YOLO26: {error}");
            return;
        }
    };
    shared.lock().unwrap().detector_status.clear();
    let mut previous = 0;
    loop {
        let (frame, threshold, epoch) = {
            let state = shared.lock().unwrap();
            if state.stop {
                return;
            }
            (state.frame.clone(), state.threshold, state.camera_epoch)
        };
        let Some(frame) = frame.filter(|f| f.id != previous) else {
            idle();
            continue;
        };
        previous = frame.id;
        let result = detector.detect(&frame.pixels, threshold);
        let mut state = shared.lock().unwrap();
        if state.camera_epoch != epoch || state.frame.is_none() {
            continue;
        }
        match result {
            Ok(detections) => {
                state.detector_status.clear();
                state.found = Some(Arc::new(Found {
                    frame,
                    detections,
                    finished: Instant::now(),
                }));
            }
            Err(error) => {
                state.found = None;
                state.detector_status = format!("Falha de inferência: {error}");
                drop(state);
                thread::sleep(Duration::from_millis(500));
            }
        }
    }
}

fn recognize_hands(shared: Workers, cfg: Config, base: PathBuf) {
    let mut models = None;
    let mut epoch = u64::MAX;
    let mut previous = 0;
    let mut failed = false;
    let started = Instant::now();
    let mut timestamp = 0;
    loop {
        let (frame, enabled, camera_epoch, next_epoch) = {
            let state = shared.lock().unwrap();
            if state.stop {
                return;
            }
            (
                state.frame.clone(),
                state.gesture_enabled,
                state.camera_epoch,
                state.gesture_epoch,
            )
        };
        if epoch != next_epoch {
            epoch = next_epoch;
            models = None;
            failed = false;
        }
        if !enabled || failed {
            idle();
            continue;
        }
        if models.is_none() {
            let load = HandTracker::new(&base, &cfg)
                .and_then(|tracker| Ok((tracker, Mlp::load(&cfg.gesture_model_path)?)));
            match load {
                Ok(pair) => models = Some(pair),
                Err(error) => {
                    let mut state = shared.lock().unwrap();
                    if state.gesture_epoch == epoch {
                        state.gesture_error = format!("Erro ao iniciar o gesto: {error}");
                    }
                    failed = true;
                    continue;
                }
            }
        }
        let Some(frame) = frame.filter(|f| f.id != previous) else {
            idle();
            continue;
        };
        previous = frame.id;
        let (tracker, model) = models.as_mut().unwrap();
        timestamp = (timestamp + 1).max(started.elapsed().as_millis() as i64);
        let result = tracker.hands(&frame.pixels, timestamp);
        let mut state = shared.lock().unwrap();
        if state.camera_epoch != camera_epoch
            || state.gesture_epoch != epoch
            || state.frame.is_none()
        {
            continue;
        }
        match result {
            Ok(hands) => {
                let confidence = gesture::features(&hands).map_or(0.0, |x| model.predict(&x));
                state.gesture_error.clear();
                state.gesture = Some(Arc::new(Gesture {
                    frame,
                    hands,
                    confidence,
                }));
            }
            Err(error) => {
                state.gesture = None;
                state.gesture_error = format!("Erro ao processar o gesto: {error}");
                drop(state);
                thread::sleep(Duration::from_millis(300));
            }
        }
    }
}

fn recognize_text(shared: Workers, cfg: Config, base: PathBuf) {
    let mut reader = match Reader::new(&base, &cfg) {
        Ok(reader) => reader,
        Err(error) => {
            shared.lock().unwrap().ocr_error = format!("OCR indisponível: {error}");
            return;
        }
    };
    let mut previous = 0;
    let mut last = None::<Instant>;
    loop {
        let request = {
            let state = shared.lock().unwrap();
            if state.stop {
                return;
            }
            state.ocr_request.clone()
        };
        let Some(request) = request.filter(|r| {
            r.frame.id != previous
                && last.is_none_or(|time| time.elapsed().as_secs_f64() >= cfg.ocr_interval_seconds)
        }) else {
            idle();
            continue;
        };
        previous = request.frame.id;
        let started = Instant::now();
        last = Some(started);
        let result = reader.read(&request.frame.pixels, request.bounds, cfg.ocr_box_padding);
        let mut state = shared.lock().unwrap();
        if state
            .ocr_request
            .as_ref()
            .is_none_or(|r| r.token != request.token)
        {
            continue;
        }
        match result {
            Ok(text) => {
                state.ocr_error.clear();
                state.ocr = Some(Arc::new(OcrText {
                    text,
                    token: request.token,
                    frame_id: request.frame.id,
                    captured: request.frame.captured,
                    finished: Instant::now(),
                    duration: started.elapsed(),
                }));
            }
            Err(error) => {
                state.ocr = None;
                state.ocr_error = format!("Falha no OCR: {error}");
            }
        }
    }
}

pub fn start(cfg: &Config, base: &Path) -> Workers {
    let shared = Arc::new(Mutex::new(Shared::new(cfg)));
    let (state, config) = (shared.clone(), cfg.clone());
    thread::spawn(move || capture(state, config));
    let (state, config, assets) = (shared.clone(), cfg.clone(), base.to_path_buf());
    thread::spawn(move || detect(state, config, assets));
    let (state, config, assets) = (shared.clone(), cfg.clone(), base.to_path_buf());
    thread::spawn(move || recognize_hands(state, config, assets));
    if cfg.enable_ocr {
        let (state, config) = (shared.clone(), cfg.clone());
        let assets = base.to_path_buf();
        thread::spawn(move || recognize_text(state, config, assets));
    }
    shared
}
