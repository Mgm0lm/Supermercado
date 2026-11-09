use crate::config::{Config, Result};
use image::{Rgb, RgbImage, imageops};
use ort::{session::Session, value::Tensor};
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Debug)]
pub struct Detection {
    pub class: String,
    pub confidence: f32,
    pub bounds: [f32; 4],
}

pub struct Detector {
    session: Session,
    side: usize,
    names: BTreeMap<usize, String>,
}

impl Detector {
    pub fn new(base: &Path, config: &Config) -> Result<Self> {
        ort::init_from(base.join("onnxruntime.dll"))?.commit();
        let session = Session::builder()?
            .with_intra_threads(4)
            .map_err(|err| err.to_string())?
            .commit_from_file(&config.model_path)?;
        let inputs = session.inputs();
        let Some(shape) = inputs
            .first()
            .and_then(|input| input.dtype().tensor_shape())
        else {
            return Err("Entrada YOLO ausente".into());
        };
        if inputs.len() != 1
            || shape.len() != 4
            || shape[0] != 1
            || shape[1] != 3
            || shape[2] != shape[3]
            || !(32..=1280).contains(&shape[2])
        {
            return Err(format!(
                "Esperada entrada YOLO quadrada e fixa em NCHW, recebida {shape:?}"
            )
            .into());
        }
        let side = shape[2] as usize;
        let metadata = session.metadata()?;
        let description = metadata
            .custom("names")
            .ok_or("Modelo ONNX sem nomes de classes")?;
        // O exportador Ultralytics grava um dicionário Python com chaves inteiras.
        let names = description
            .trim_matches(['{', '}'])
            .split(',')
            .map(|pair| {
                let (index, name) = pair.split_once(':').ok_or("Metadados YOLO inválidos")?;
                Ok((
                    index.trim().parse()?,
                    name.trim().trim_matches(['\'', '"']).to_owned(),
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        drop(metadata);
        if !names
            .values()
            .any(|name| config.products.contains_key(name))
        {
            return Err("Nenhuma classe do ONNX está cadastrada no catálogo".into());
        }
        Ok(Self {
            session,
            side,
            names,
        })
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
        let mut input = vec![0.0; 3 * plane];
        for (i, pixel) in padded.pixels().enumerate() {
            input[i] = f32::from(pixel[0]) / 255.0;
            input[plane + i] = f32::from(pixel[1]) / 255.0;
            input[2 * plane + i] = f32::from(pixel[2]) / 255.0;
        }
        let input = Tensor::from_array(([1, 3, self.side, self.side], input.into_boxed_slice()))?;
        let outputs = self.session.run(ort::inputs![input])?;
        let (shape, values) = outputs[0].try_extract_tensor::<f32>()?;
        if shape.len() != 3 || shape[0] != 1 || shape[2] != 6 {
            return Err(
                format!("Esperada saída YOLO26 end-to-end [1, N, 6], recebida {shape:?}").into(),
            );
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
    }
}
