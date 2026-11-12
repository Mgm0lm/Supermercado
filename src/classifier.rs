use crate::config::{Config, Result};
use image::{RgbImage, imageops};
use ort::{session::Session, value::Tensor};
use std::{
    collections::BTreeMap, env, fs, os::windows::process::CommandExt, path::Path, process::Command,
};

pub type Prediction = (String, f32);

pub struct ProductClassifier {
    session: Session,
    width: u32,
    height: u32,
    labels: BTreeMap<usize, String>,
}

impl ProductClassifier {
    pub fn new(base: &Path, path: &Path) -> Result<Self> {
        let extension = path.extension().ok_or("Informe um modelo .h5 ou .onnx.")?;
        ort::init_from(base.join("onnxruntime.dll"))?.commit();
        let mut builder = Session::builder()?
            .with_intra_threads(2)
            .map_err(|e| e.to_string())?;
        let session = if extension.eq_ignore_ascii_case("h5") {
            let python = env::var_os("CLASSIFIER_PYTHON").unwrap_or_else(|| "python".into());
            let output = Command::new(python)
                .arg(base.join("export_classifier.py"))
                .arg(path)
                .arg("-")
                .creation_flags(0x08000000) // CREATE_NO_WINDOW
                .output()
                .map_err(|e| format!("Falha ao executar o conversor Keras: {e}"))?;
            if !output.status.success() {
                return Err(format!(
                    "Falha ao converter {}: {}",
                    path.display(),
                    String::from_utf8_lossy(&output.stderr)
                )
                .into());
            }
            builder.commit_from_memory(&output.stdout)?
        } else if extension.eq_ignore_ascii_case("onnx") {
            builder.commit_from_file(path)?
        } else {
            return Err("O classificador deve ser um arquivo .h5 ou .onnx.".into());
        };
        let [input] = session.inputs() else {
            return Err("Esperada uma entrada de imagem no classificador.".into());
        };
        let [1, height, width, 3] = &**input
            .dtype()
            .tensor_shape()
            .ok_or("Entrada sem dimensões")?
        else {
            return Err("Esperada entrada fixa [1, altura, largura, 3] no classificador.".into());
        };
        let (width, height) = (u32::try_from(*width)?, u32::try_from(*height)?);
        if width == 0 || height == 0 {
            return Err("Dimensões vazias no classificador.".into());
        }
        let [output] = session.outputs() else {
            return Err("Esperada uma saída de probabilidades no classificador.".into());
        };
        let [1, count] = &**output.dtype().tensor_shape().ok_or("Saída sem dimensões")? else {
            return Err("Esperada uma saída [1, classes] no classificador.".into());
        };
        let labels_path = path.with_file_name("labels.txt");
        let contents = fs::read_to_string(&labels_path)?;
        let mut labels = BTreeMap::new();
        for line in contents
            .trim_start_matches('\u{feff}')
            .lines()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            let (index, label) = line
                .split_once(char::is_whitespace)
                .ok_or("Esperado índice e nome em labels.txt.")?;
            let label = match label.trim() {
                "Miojo" => "Nissin Miojo",
                "Coca-Cola" => "Coca 600ml",
                name => name,
            };
            if labels.insert(index.parse()?, label.to_owned()).is_some() {
                return Err("Índice repetido em labels.txt.".into());
            }
        }
        if i64::try_from(labels.len())? != *count || !labels.keys().copied().eq(0..labels.len()) {
            return Err(format!(
                "Os índices de {} devem corresponder às saídas do modelo.",
                labels_path.display()
            )
            .into());
        }
        Ok(Self {
            session,
            width,
            height,
            labels,
        })
    }

    pub fn predict(
        &mut self,
        image: &RgbImage,
        bounds: [f32; 4],
        threshold: f32,
        cfg: &Config,
    ) -> Result<Option<Prediction>> {
        let [left, top, right, bottom] = bounds.map(|value| value as u32);
        let width = right.min(image.width()).saturating_sub(left);
        let height = bottom.min(image.height()).saturating_sub(top);
        if width == 0 || height == 0 {
            return Ok(None);
        }
        let crop = imageops::crop_imm(image, left, top, width, height).to_image();
        let resized = imageops::resize(
            &crop,
            self.width,
            self.height,
            imageops::FilterType::Triangle,
        );
        let data: Vec<_> = resized
            .as_raw()
            .iter()
            .map(|&v| f32::from(v) / 127.5 - 1.0)
            .collect();
        let input = Tensor::from_array((
            [1, self.height as usize, self.width as usize, 3],
            data.into_boxed_slice(),
        ))?;
        let outputs = self.session.run(ort::inputs![input])?;
        let (_, probabilities) = outputs[0].try_extract_tensor::<f32>()?;
        if !probabilities.iter().all(|p| p.is_finite()) {
            return Ok(None);
        }
        let best = probabilities
            .iter()
            .enumerate()
            .rev()
            .max_by(|a, b| a.1.total_cmp(b.1));
        Ok(best.and_then(|(index, &probability)| {
            let label = self.labels.get(&index)?;
            (cfg.products.contains_key(label)
                && !label.eq_ignore_ascii_case("nada")
                && (threshold..=1.0).contains(&probability))
            .then(|| (label.clone(), probability))
        }))
    }
}
