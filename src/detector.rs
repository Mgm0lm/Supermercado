use crate::config::{Config, ModelBackend, Result};
use image::{Rgb, RgbImage, imageops};
use libloading::os::windows::{
    LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, Library,
};
use openvino::{CompiledModel, Core, DeviceType, ElementType, InferRequest, Shape};
use ort::{session::Session, value::Tensor};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub struct Detection {
    pub class: String,
    pub confidence: f32,
    pub bounds: [f32; 4],
}

type Classes = BTreeMap<usize, String>;

#[derive(Deserialize)]
struct IrMetadata {
    names: ModelNames,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ModelNames {
    Numeric(Classes),
    Quoted(BTreeMap<String, String>),
    List(Vec<String>),
}

impl ModelNames {
    fn into_classes(self) -> Result<Classes> {
        match self {
            Self::Numeric(names) => Ok(names),
            Self::Quoted(names) => names
                .into_iter()
                .map(|(id, name)| Ok((id.parse()?, name)))
                .collect(),
            Self::List(names) => Ok(names.into_iter().enumerate().collect()),
        }
    }
}

enum Engine {
    Onnx(Session),
    OpenVino {
        request: InferRequest,
        input: openvino::Tensor,
        _compiled: CompiledModel,
        _core: Core,
    },
}

pub struct Detector {
    engine: Engine,
    side: usize,
    names: Classes,
}

fn has_extension(path: &Path, extension: &str) -> bool {
    path.extension()
        .is_some_and(|value| value.eq_ignore_ascii_case(extension))
}

fn model_file(path: &Path) -> Result<PathBuf> {
    if !path.is_dir() {
        return Ok(path.to_path_buf());
    }
    let mut xml = None;
    for entry in fs::read_dir(path)? {
        let candidate = entry?.path();
        if has_extension(&candidate, "xml")
            && candidate.is_file()
            && xml.replace(candidate).is_some()
        {
            return Err(
                "A pasta tem mais de um modelo OpenVINO. Informe o arquivo .xml em MODEL_PATH."
                    .into(),
            );
        }
    }
    xml.ok_or_else(|| "A pasta do modelo não contém um arquivo .xml.".into())
}

fn input_side(shape: &[i64]) -> Result<usize> {
    match shape {
        [1, 3, h, w] if h == w && (32..=1280).contains(h) => Ok(*h as usize),
        _ => {
            Err(format!("Esperada entrada YOLO quadrada e fixa em NCHW, recebida {shape:?}").into())
        }
    }
}

fn onnx_session(base: &Path, model: &Path) -> Result<Session> {
    ort::init_from(base.join("onnxruntime.dll"))?.commit();
    Ok(Session::builder()?
        .with_intra_threads(4)
        .map_err(|error| error.to_string())?
        .commit_from_file(model)?)
}

fn onnx_names(session: &Session) -> Result<Classes> {
    let metadata = session
        .metadata()?
        .custom("names")
        .ok_or("Modelo ONNX sem nomes de classes")?;
    serde_yaml_ng::from_str::<ModelNames>(&metadata)?.into_classes()
}

impl Detector {
    pub fn new(base: &Path, config: &Config) -> Result<Self> {
        let model = model_file(&config.model_path)?;
        let ir = has_extension(&model, "xml");
        if !ir && !has_extension(&model, "onnx") {
            return Err(
                "MODEL_PATH deve apontar para um .onnx, um .xml ou uma pasta OpenVINO.".into(),
            );
        }
        if ir && config.model_backend == ModelBackend::Onnxruntime {
            return Err("Use MODEL_BACKEND 'auto' ou 'openvino' para abrir um modelo .xml.".into());
        }
        let (engine, side, names) = if ir || config.model_backend == ModelBackend::Openvino {
            let names = if ir {
                let metadata = model.with_file_name("metadata.yaml");
                let bytes = fs::read(&metadata)
                    .map_err(|error| format!("Falha ao ler {}: {error}", metadata.display()))?;
                serde_yaml_ng::from_slice::<IrMetadata>(&bytes)?
                    .names
                    .into_classes()?
            } else {
                onnx_names(&onnx_session(base, &model)?)?
            };
            let runtime = base.join("openvino_c.dll");
            // Carrega as dependências do runtime junto da DLL, inclusive quando --assets usa outra pasta.
            let _dll = unsafe {
                Library::load_with_flags(
                    &runtime,
                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
                )
            }
            .map_err(|error| format!("Falha ao abrir {}: {error}", runtime.display()))?;
            openvino_sys::library::load_from(&runtime).map_err(|error| {
                format!(
                    "Runtime OpenVINO indisponível em {}: {error}",
                    runtime.display()
                )
            })?;
            let mut core = Core::new()?;
            let model_path = model.to_str().ok_or("Caminho de modelo inválido")?;
            let weights = model.with_extension("bin");
            if ir && !weights.is_file() {
                return Err(
                    format!("Arquivo de pesos OpenVINO ausente: {}", weights.display()).into(),
                );
            }
            let weights_path = if ir {
                weights.to_str().ok_or("Caminho dos pesos inválido")?
            } else {
                ""
            };
            let network = core.read_model_from_file(model_path, weights_path)?;
            if network.get_inputs_len()? != 1 || network.get_outputs_len()? != 1 {
                return Err("O YOLO26 deve ter uma entrada e uma saída.".into());
            }
            let port = network.get_input_by_index(0)?;
            let shape = port.get_shape()?;
            let side = input_side(shape.get_dimensions())?;
            if port.get_element_type()? != ElementType::F32 {
                return Err("O modelo YOLO26 deve aceitar imagens em FP32.".into());
            }
            let mut compiled = core.compile_model(&network, DeviceType::CPU)?;
            let mut request = compiled.create_infer_request()?;
            let input =
                openvino::Tensor::new(ElementType::F32, &Shape::new(shape.get_dimensions())?)?;
            request.set_input_tensor(&input)?;
            (
                Engine::OpenVino {
                    request,
                    input,
                    _compiled: compiled,
                    _core: core,
                },
                side,
                names,
            )
        } else {
            let session = onnx_session(base, &model)?;
            let [input] = session.inputs() else {
                return Err("O YOLO26 deve ter uma entrada.".into());
            };
            let shape = input.dtype().tensor_shape().ok_or("Entrada YOLO ausente")?;
            let side = input_side(shape)?;
            let names = onnx_names(&session)?;
            (Engine::Onnx(session), side, names)
        };
        if !names
            .values()
            .any(|name| config.products.contains_key(name))
        {
            return Err(
                "Nenhuma classe do modelo está cadastrada em PRODUCTS no config_override.json."
                    .into(),
            );
        }
        Ok(Self {
            engine,
            side,
            names,
        })
    }

    pub fn backend_name(&self) -> &'static str {
        match self.engine {
            Engine::Onnx(_) => "ONNX Runtime",
            Engine::OpenVino { .. } => "OpenVINO (CPU)",
        }
    }

    pub fn detect(&mut self, image: &RgbImage, threshold: f32) -> Result<Vec<Detection>> {
        let (width, height) = image.dimensions();
        let scale = self.side as f32 / width.max(height) as f32;
        let w = ((width as f32 * scale).round() as u32).max(1);
        let h = ((height as f32 * scale).round() as u32).max(1);
        let x = (self.side - w as usize) / 2;
        let y = (self.side - h as usize) / 2;
        let resized = imageops::resize(image, w, h, imageops::FilterType::Triangle);
        let mut padded = RgbImage::from_pixel(self.side as u32, self.side as u32, Rgb([114; 3]));
        imageops::replace(&mut padded, &resized, x as i64, y as i64);
        let plane = self.side * self.side;
        let fill_input = |input: &mut [f32]| {
            for (i, pixel) in padded.pixels().enumerate() {
                input[i] = f32::from(pixel[0]) / 255.0;
                input[plane + i] = f32::from(pixel[1]) / 255.0;
                input[2 * plane + i] = f32::from(pixel[2]) / 255.0;
            }
        };
        let decode = |shape: &[i64], values: &[f32]| {
            if !matches!(shape, [1, _, 6]) {
                return Err(format!(
                    "Esperada saída YOLO26 end-to-end [1, N, 6], recebida {shape:?}"
                )
                .into());
            }
            let project_x = |p: f32| ((p - x as f32) / scale).clamp(0.0, width as f32);
            let project_y = |p: f32| ((p - y as f32) / scale).clamp(0.0, height as f32);
            Ok(values
                .chunks_exact(6)
                .filter_map(|row| {
                    if row[4] < threshold || !row.iter().all(|v| v.is_finite()) {
                        return None;
                    }
                    let class = self.names.get(&(row[5] as usize))?.clone();
                    let bounds = [
                        project_x(row[0]),
                        project_y(row[1]),
                        project_x(row[2]),
                        project_y(row[3]),
                    ];
                    (bounds[0] < bounds[2] && bounds[1] < bounds[3]).then_some(Detection {
                        class,
                        confidence: row[4],
                        bounds,
                    })
                })
                .collect())
        };
        match &mut self.engine {
            Engine::Onnx(session) => {
                let mut input = vec![0.0; 3 * plane];
                fill_input(&mut input);
                let input =
                    Tensor::from_array(([1, 3, self.side, self.side], input.into_boxed_slice()))?;
                let output = session.run(ort::inputs![input])?;
                let (shape, values) = output[0].try_extract_tensor::<f32>()?;
                decode(shape, values)
            }
            Engine::OpenVino { request, input, .. } => {
                fill_input(input.get_data_mut::<f32>()?);
                request.infer()?;
                let output = request.get_output_tensor()?;
                if output.get_element_type()? != ElementType::F32 {
                    return Err("O modelo YOLO26 deve devolver detecções em FP32.".into());
                }
                decode(
                    output.get_shape()?.get_dimensions(),
                    output.get_data::<f32>()?,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_ultralytics_ir_names_and_quoted_indices() {
        let metadata: IrMetadata = serde_yaml_ng::from_str(include_str!(
            "../Programa/best_openvino_model/metadata.yaml"
        ))
        .unwrap();
        let names = metadata.names.into_classes().unwrap();
        assert_eq!(names[&8], "Serenata do Amor");
        let quoted: IrMetadata =
            serde_yaml_ng::from_str(r#"{"names": {"0": "Café", "1": "Milho"}}"#).unwrap();
        assert_eq!(quoted.names.into_classes().unwrap()[&0], "Café");
    }
}
