use crate::config::{Config, Result};
use image::RgbImage;
use libloading::Library;
use serde::Deserialize;
use std::{
    ffi::{CStr, c_char, c_void},
    fs,
    path::Path,
    ptr, slice,
};

type Handle = *mut c_void;
type ErrorOut = *mut *mut c_char;
type ErrorFree = unsafe extern "C" fn(*mut c_char);
type ImageCreate =
    unsafe extern "C" fn(i32, i32, i32, *const u8, i32, *mut Handle, ErrorOut) -> i32;
type ImageFree = unsafe extern "C" fn(Handle);
type Create = unsafe extern "C" fn(*const Options, *mut Handle, ErrorOut) -> i32;
type Detect =
    unsafe extern "C" fn(Handle, Handle, *const c_void, i64, *mut HandResult, ErrorOut) -> i32;
type CloseResult = unsafe extern "C" fn(*mut HandResult);
type Close = unsafe extern "C" fn(Handle, ErrorOut) -> i32;

// Correspondem à ABI de MediaPipe Tasks C distribuída em libmediapipe.dll 1.0.1.
#[repr(C)]
struct BaseOptions {
    buffer: *const u8,
    size: u32,
    path: *const c_char,
    fd: i32,
    delegate: i32,
    host_environment: i32,
    host_system: i32,
    host_version: *const c_char,
    ca_bundle_path: *const c_char,
    app_id: *const c_char,
    app_version: *const c_char,
}

#[repr(C)]
struct Options {
    base: BaseOptions,
    running_mode: i32,
    num_hands: i32,
    min_detection: f32,
    min_presence: f32,
    min_tracking: f32,
    callback: Option<unsafe extern "C" fn(i32, *const HandResult, Handle, i64)>,
}

#[repr(C)]
struct Category {
    index: i32,
    score: f32,
    name: *const c_char,
    display_name: *const c_char,
}

#[repr(C)]
struct Landmark {
    xyz: [f32; 3],
    has_visibility: bool,
    visibility: f32,
    has_presence: bool,
    presence: f32,
    name: *const c_char,
}

#[repr(C)]
struct List<T> {
    items: *const T,
    len: u32,
}

impl<T> List<T> {
    unsafe fn as_slice(&self) -> &[T] {
        if self.items.is_null() || self.len == 0 {
            &[]
        } else {
            // O MediaPipe mantém o array válido até MpHandLandmarkerCloseResult.
            unsafe { slice::from_raw_parts(self.items, self.len as usize) }
        }
    }
}

#[repr(C)]
struct HandResult {
    handedness: List<List<Category>>,
    normalized: List<List<Landmark>>,
    world: List<List<Landmark>>,
}

pub struct Hand {
    pub left: bool,
    pub points: [[f32; 3]; 21],
}

pub struct HandTracker {
    handle: Handle,
    image_create: ImageCreate,
    image_free: ImageFree,
    detect: Detect,
    close_result: CloseResult,
    close: Close,
    error_free: ErrorFree,
    _model_data: Vec<u8>,
    _library: Library,
}

fn status(code: i32, error: *mut c_char, free: ErrorFree) -> Result<()> {
    let message = if error.is_null() {
        format!("MediaPipe retornou código {code}")
    } else {
        // A DLL devolve uma string NUL-terminada com seu próprio alocador.
        unsafe {
            let message = CStr::from_ptr(error).to_string_lossy().into_owned();
            free(error);
            message
        }
    };
    if code == 0 {
        Ok(())
    } else {
        Err(message.into())
    }
}

impl HandTracker {
    pub fn new(base: &Path, config: &Config) -> Result<Self> {
        let model = fs::read(&config.hand_landmarker_model_path)?;
        // Os símbolos e suas assinaturas são da versão indicada acima.
        unsafe {
            let library = Library::new(base.join("libmediapipe.dll"))?;
            let create = *library.get::<Create>(b"MpHandLandmarkerCreate")?;
            let detect = *library.get::<Detect>(b"MpHandLandmarkerDetectForVideo")?;
            let close_result = *library.get::<CloseResult>(b"MpHandLandmarkerCloseResult")?;
            let close = *library.get::<Close>(b"MpHandLandmarkerClose")?;
            let image_create = *library.get::<ImageCreate>(b"MpImageCreateFromUint8Data")?;
            let image_free = *library.get::<ImageFree>(b"MpImageFree")?;
            let error_free = *library.get::<ErrorFree>(b"MpErrorFree")?;
            let options = Options {
                base: BaseOptions {
                    buffer: model.as_ptr(),
                    size: model.len().try_into()?,
                    path: ptr::null(),
                    fd: -1,
                    delegate: 0,
                    host_environment: 0,
                    host_system: 3,
                    host_version: ptr::null(),
                    ca_bundle_path: ptr::null(),
                    app_id: ptr::null(),
                    app_version: ptr::null(),
                },
                running_mode: 2,
                num_hands: 2,
                min_detection: config.mediapipe_min_detection_confidence,
                min_presence: config.mediapipe_min_detection_confidence,
                min_tracking: config.mediapipe_min_tracking_confidence,
                callback: None,
            };
            let mut handle = ptr::null_mut();
            let mut error = ptr::null_mut();
            status(create(&options, &mut handle, &mut error), error, error_free)?;
            if handle.is_null() {
                return Err("MediaPipe não retornou um detector de mãos".into());
            }
            Ok(Self {
                handle,
                image_create,
                image_free,
                detect,
                close_result,
                close,
                error_free,
                _model_data: model,
                _library: library,
            })
        }
    }

    pub fn hands(&mut self, image: &RgbImage, timestamp: i64) -> Result<Vec<Hand>> {
        let len = image.as_raw().len().try_into()?;
        let width = image.width().try_into()?;
        let height = image.height().try_into()?;
        // RGB intercalado de 8 bits. O resultado e a imagem são liberados pela mesma DLL.
        unsafe {
            let mut handle = ptr::null_mut();
            let mut error = ptr::null_mut();
            let code = (self.image_create)(
                1,
                width,
                height,
                image.as_raw().as_ptr(),
                len,
                &mut handle,
                &mut error,
            );
            status(code, error, self.error_free)?;
            let mut result: HandResult = std::mem::zeroed();
            let mut error = ptr::null_mut();
            let code = (self.detect)(
                self.handle,
                handle,
                ptr::null(),
                timestamp,
                &mut result,
                &mut error,
            );
            (self.image_free)(handle);
            status(code, error, self.error_free)?;
            let mut hands = Vec::with_capacity(2);
            for (category, landmarks) in result
                .handedness
                .as_slice()
                .iter()
                .zip(result.normalized.as_slice())
            {
                let Some(category) = category.as_slice().first() else {
                    continue;
                };
                if category.name.is_null() {
                    continue;
                }
                let left = match CStr::from_ptr(category.name).to_bytes() {
                    b"Left" => true,
                    b"Right" => false,
                    _ => continue,
                };
                let landmarks = landmarks.as_slice();
                if landmarks.len() == 21 {
                    let points = std::array::from_fn(|i| landmarks[i].xyz);
                    hands.push(Hand { left, points });
                }
            }
            (self.close_result)(&mut result);
            Ok(hands)
        }
    }
}

impl Drop for HandTracker {
    fn drop(&mut self) {
        // O identificador pertence a este rastreador e a DLL ainda está carregada.
        unsafe {
            let mut error = ptr::null_mut();
            let code = (self.close)(self.handle, &mut error);
            let _ = status(code, error, self.error_free);
        }
    }
}

#[derive(Deserialize)]
struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

#[derive(Deserialize)]
pub struct Mlp {
    layers: Vec<Layer>,
    activation: String,
    mean: Vec<f64>,
    scale: Vec<f64>,
}

pub fn features(hands: &[Hand]) -> Option<[f64; 126]> {
    let [first, second] = hands else {
        return None;
    };
    if first.left == second.left {
        return None;
    }
    let sorted = if first.left {
        [first, second]
    } else {
        [second, first]
    };
    let mut features = [0.0; 126];
    for (side, hand) in sorted.into_iter().enumerate() {
        let wrist = hand.points[0];
        let middle: [f32; 3] = std::array::from_fn(|axis| hand.points[9][axis] - wrist[axis]);
        let palm = middle.iter().map(|v| v * v).sum::<f32>().sqrt();
        let scale = if palm < 1e-6 { 1.0 } else { palm };
        for (index, point) in hand.points.iter().enumerate() {
            for axis in 0..3 {
                features[side * 63 + index * 3 + axis] =
                    f64::from((point[axis] - wrist[axis]) / scale);
            }
        }
    }
    Some(features)
}

impl Mlp {
    pub fn load(path: &Path) -> Result<Self> {
        let mlp: Self = serde_json::from_slice(&fs::read(path)?)?;
        if mlp.mean.len() != 126
            || mlp.scale.len() != 126
            || mlp
                .scale
                .iter()
                .any(|scale| *scale <= 0.0 || !scale.is_finite())
            || !matches!(
                mlp.activation.as_str(),
                "relu" | "tanh" | "identity" | "logistic"
            )
            || mlp.layers.is_empty()
        {
            return Err("Parâmetros inválidos da MLP".into());
        }
        let mut inputs = 126;
        for layer in &mlp.layers {
            let outputs = layer.biases.len();
            if outputs == 0
                || layer.weights.len() != inputs
                || layer.weights.iter().any(|row| row.len() != outputs)
            {
                return Err("Dimensões inválidas da MLP".into());
            }
            inputs = outputs;
        }
        if inputs != 1 {
            return Err("A MLP deve ter saída binária".into());
        }
        Ok(mlp)
    }

    pub fn predict(&self, features: &[f64; 126]) -> f64 {
        let mut values: Vec<_> = features
            .iter()
            .zip(&self.mean)
            .zip(&self.scale)
            .map(|((x, mean), scale)| (x - mean) / scale)
            .collect();
        let logistic = |x: f64| 1.0 / (1.0 + (-x).exp());
        for (index, layer) in self.layers.iter().enumerate() {
            let mut outputs = layer.biases.clone();
            for (value, weights) in values.iter().zip(&layer.weights) {
                for (out, weight) in outputs.iter_mut().zip(weights) {
                    *out += value * weight;
                }
            }
            if index == self.layers.len() - 1 {
                return logistic(outputs[0]);
            }
            for out in &mut outputs {
                *out = match self.activation.as_str() {
                    "relu" => out.max(0.0),
                    "tanh" => out.tanh(),
                    "logistic" => logistic(*out),
                    _ => *out,
                };
            }
            values = outputs;
        }
        unreachable!("a carga valida que a MLP tem pelo menos uma camada")
    }
}
