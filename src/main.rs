#![windows_subsystem = "windows"]

mod checkout;
mod config;
mod detector;
mod gesture;
mod ocr;
mod ui;
mod workers;

use config::{Config, Result};
use detector::Detector;
use gesture::{HandTracker, Mlp};
use std::{
    cell::Cell,
    env, fs,
    path::{Path, PathBuf},
    rc::Rc,
    time::Instant,
};
use windows::{
    Win32::UI::WindowsAndMessaging::{MB_ICONERROR, MessageBoxW},
    core::HSTRING,
};

fn run() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let executable = env::current_exe()?;
    let mut base = executable
        .parent()
        .ok_or("Executável sem pasta")?
        .to_path_buf();
    let mut debug = false;
    let mut zone = false;
    let mut hand_block = false;
    let mut check_fullscreen = false;
    let mut operation = None;
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--assets") => {
                base = fs::canonicalize(PathBuf::from(
                    args.next().ok_or("Informe a pasta após --assets")?,
                ))?
            }
            Some("--debug") => debug = true,
            Some("--zone") => zone = true,
            Some("--hand-block") => hand_block = true,
            Some("--check-fullscreen") => check_fullscreen = true,
            Some("--check-models") => operation = Some(Check::Models),
            Some("--check-camera") => operation = Some(Check::Camera),
            Some("--check-image" | "--check-ocr" | "--check-mlp") => {
                let path = PathBuf::from(
                    args.next()
                        .ok_or("Informe o arquivo após o argumento de teste")?,
                );
                operation = Some(match arg.to_str() {
                    Some("--check-image") => Check::Image(path),
                    Some("--check-ocr") => Check::Ocr(path),
                    _ => Check::Mlp(path),
                });
            }
            _ => return Err(format!("Argumento desconhecido: {arg:?}").into()),
        }
    }
    let mut cfg = Config::load(&base)?;
    if let Some(check) = operation {
        return check.run(&base, &cfg);
    }
    let title = cfg.app_title.clone();
    if check_fullscreen {
        cfg.max_cameras_to_scan = 0;
        cfg.enable_ocr = false;
        cfg.enable_gesture_checkout_default = false;
    }
    let result = Rc::new(Cell::new(false));
    let probe = check_fullscreen.then(|| result.clone());
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_fullscreen(true)
            .with_decorations(false),
        persist_window: false,
        ..Default::default()
    };
    eframe::run_native(
        &title,
        options,
        Box::new(move |context| {
            Ok(Box::new(
                ui::App::new(
                    &context.egui_ctx,
                    &base,
                    cfg,
                    debug,
                    zone,
                    hand_block,
                    probe,
                )
                .map_err(|err| err.to_string())?,
            ))
        }),
    )?;
    if check_fullscreen && !result.get() {
        return Err("A área do caixa não preencheu o monitor.".into());
    }
    Ok(())
}

enum Check {
    Models,
    Camera,
    Image(PathBuf),
    Ocr(PathBuf),
    Mlp(PathBuf),
}

impl Check {
    fn run(self, base: &Path, cfg: &Config) -> Result<()> {
        match self {
            Self::Models => {
                let image = image::RgbImage::new(640, 480);
                let mut detector = Detector::new(base, cfg)?;
                let detections = detector.detect(&image, cfg.conf_threshold_default)?;
                let mut tracker = HandTracker::new(base, cfg)?;
                let hands = tracker.hands(&image, 0)?;
                let mlp = Mlp::load(&cfg.gesture_model_path)?;
                let probability = mlp.predict(&[0.0; 126]);
                if !detections.is_empty() || !hands.is_empty() || !probability.is_finite() {
                    return Err("Resultado inválido no quadro sintético vazio".into());
                }
                println!(
                    "YOLO26 via {}, MediaPipe e MLP carregados e executados.",
                    detector.backend_name()
                );
            }
            Self::Camera => {
                let cameras = nokhwa::query(nokhwa::utils::ApiBackend::MediaFoundation)?;
                let selected = cameras.first().ok_or("Nenhuma câmera encontrada")?;
                let mut camera = workers::open_camera(selected.index().clone(), cfg)?;
                let started = Instant::now();
                let mut dimensions = (0, 0);
                for _ in 0..10 {
                    dimensions = workers::frame(&mut camera)?.dimensions();
                }
                println!(
                    "Câmera: 10 quadros RGB {}x{} em {} ms.",
                    dimensions.0,
                    dimensions.1,
                    started.elapsed().as_millis()
                );
            }
            Self::Image(path) => {
                let image = image::open(path)?.into_rgb8();
                let mut detector = Detector::new(base, cfg)?;
                let detections = detector.detect(&image, cfg.conf_threshold_default)?;
                let mut tracker = HandTracker::new(base, cfg)?;
                let hands = tracker.hands(&image, 0)?;
                let mlp = Mlp::load(&cfg.gesture_model_path)?;
                let probability = gesture::features(&hands).map(|f| mlp.predict(&f));
                let output = serde_json::json!({
                    "backend": detector.backend_name(),
                    "detections": detections.iter().map(|d| serde_json::json!({
                        "class": d.class, "confidence": d.confidence, "bounds": d.bounds,
                    })).collect::<Vec<_>>(),
                    "hands": hands.iter().map(|h| serde_json::json!({
                        "side": if h.left { "Left" } else { "Right" }, "landmarks": h.points.len(),
                    })).collect::<Vec<_>>(),
                    "gesture_probability": probability,
                });
                println!("{output}");
            }
            Self::Ocr(path) => {
                let image = image::open(path)?.into_rgb8();
                println!("{}", ocr::Reader::new(base, cfg)?.read(&image, None, 0.0)?);
            }
            Self::Mlp(path) => {
                let features: Vec<Vec<f64>> = serde_json::from_slice(&fs::read(path)?)?;
                let mlp = Mlp::load(&cfg.gesture_model_path)?;
                let mut predictions = Vec::new();
                for input in features {
                    let input: [f64; 126] = input
                        .try_into()
                        .map_err(|_| "Esperados 126 valores por amostra")?;
                    predictions.push(mlp.predict(&input));
                }
                println!("{}", serde_json::to_string(&predictions)?);
            }
        }
        Ok(())
    }
}

fn main() {
    if let Err(error) = run() {
        let message = format!("{error}");
        eprintln!("{message}");
        if !env::args_os().any(|arg| arg.to_string_lossy().starts_with("--check-")) {
            unsafe {
                MessageBoxW(
                    None,
                    &HSTRING::from(message),
                    &HSTRING::from("Caixa Autônomo"),
                    MB_ICONERROR,
                );
            }
        }
        std::process::exit(1);
    }
}
