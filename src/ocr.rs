use crate::config::{Config, Result};
use image::{
    RgbImage,
    imageops::{self, FilterType},
};
use ort::{session::Session, value::Tensor};
use std::{collections::VecDeque, path::Path};

pub struct Reader {
    detect: Session,
    recognize: Session,
    alphabet: Vec<String>,
    max_side: u32,
    detector_side: u32,
    min_score: f32,
}

fn session(model: &Path, threads: usize) -> Result<Session> {
    Ok(Session::builder()?
        .with_intra_threads(threads)
        .map_err(|err| err.to_string())?
        .commit_from_file(model)?)
}

impl Reader {
    pub fn new(base: &Path, cfg: &Config) -> Result<Self> {
        ort::init_from(base.join("onnxruntime.dll"))?.commit();
        let directory = base.join("modelo/ocr");
        let detect = session(&directory.join("det.onnx"), cfg.ocr_threads)?;
        let recognize = session(&directory.join("rec.onnx"), cfg.ocr_threads)?;
        let dictionary = recognize
            .metadata()?
            .custom("character")
            .ok_or("Modelo OCR sem alfabeto")?;
        let alphabet = std::iter::once(String::new())
            .chain(dictionary.lines().map(str::to_owned))
            .chain(std::iter::once(" ".into()))
            .collect();
        Ok(Self {
            detect,
            recognize,
            alphabet,
            max_side: cfg.ocr_max_side,
            detector_side: cfg.ocr_det_side_len,
            min_score: cfg.ocr_min_text_score,
        })
    }

    pub fn read(&mut self, image: &RgbImage, bounds: Option<[f32; 4]>, pad: f32) -> Result<String> {
        let (width, height) = image.dimensions();
        let (x, y, w, h) = if let Some([x1, y1, x2, y2]) = bounds {
            let (dx, dy) = (((x2 - x1) * pad).floor(), ((y2 - y1) * pad).floor());
            let (x, y) = ((x1 - dx).max(0.0) as u32, (y1 - dy).max(0.0) as u32);
            let w = ((x2 + dx).min(width as f32) as u32).saturating_sub(x);
            let h = ((y2 + dy).min(height as f32) as u32).saturating_sub(y);
            if w >= 8 && h >= 8 {
                (x, y, w, h)
            } else {
                (0, 0, width, height)
            }
        } else {
            (0, 0, width, height)
        };
        let crop = imageops::crop_imm(image, x, y, w, h).to_image();
        let scale = (self.max_side as f32 / w.max(h) as f32).min(1.0);
        let crop = if scale < 1.0 {
            imageops::resize(
                &crop,
                ((w as f32 * scale) as u32).max(1),
                ((h as f32 * scale) as u32).max(1),
                FilterType::Triangle,
            )
        } else {
            crop
        };
        let boxes = self.boxes(&crop)?;
        let mut text = Vec::new();
        for line in boxes {
            let crop = line.extract(&crop);
            if let Some(word) = self.word(&crop)? {
                text.push(word);
            }
        }
        Ok(text.join(" "))
    }

    fn boxes(&mut self, image: &RgbImage) -> Result<Vec<Region>> {
        let (w, h) = image.dimensions();
        let scale = (self.detector_side as f32 / w.max(h) as f32).min(1.0);
        let dimension = |n| (((n as f32 * scale / 32.0).round() as u32) * 32).max(32);
        let (rw, rh) = (dimension(w), dimension(h));
        let input = imageops::resize(image, rw, rh, FilterType::Triangle);
        let plane = rw as usize * rh as usize;
        let mut data = vec![0.0; 3 * plane];
        for (i, rgb) in input.pixels().enumerate() {
            for (channel, (mean, std)) in [(0.5, 0.5); 3].into_iter().enumerate() {
                data[channel * plane + i] = (f32::from(rgb[2 - channel]) / 255.0 - mean) / std;
            }
        }
        let tensor =
            Tensor::from_array(([1, 3, rh as usize, rw as usize], data.into_boxed_slice()))?;
        let outputs = self.detect.run(ort::inputs![tensor])?;
        let (shape, probabilities) = outputs[0].try_extract_tensor::<f32>()?;
        if shape.len() != 4 || shape[0] != 1 || shape[1] != 1 || shape[2] < 1 || shape[3] < 1 {
            return Err(format!("Saída inválida do detector de texto: {shape:?}").into());
        }
        let width = shape[3] as usize;
        let height = shape[2] as usize;
        let mut active: Vec<bool> = probabilities.iter().map(|p| *p >= 0.3).collect();
        let mut result = Vec::new();
        let mut queue = VecDeque::new();
        let mut points = Vec::new();
        for i in 0..active.len() {
            if !active[i] {
                continue;
            }
            active[i] = false;
            queue.push_back(i);
            points.clear();
            let mut sum = 0.0;
            while let Some(p) = queue.pop_front() {
                sum += probabilities[p];
                let x = p % width;
                let y = p / width;
                points.push([
                    x as f32 * w as f32 / width as f32,
                    y as f32 * h as f32 / height as f32,
                ]);
                for yy in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                    for xx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                        let neighbor = yy * width + xx;
                        if active[neighbor] {
                            active[neighbor] = false;
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
            if points.len() >= 10 && sum / points.len() as f32 >= 0.5 {
                if let Some(region) = Region::from_component(&points) {
                    result.push(region);
                }
            }
        }
        result.sort_by(|a, b| {
            a.center[1]
                .total_cmp(&b.center[1])
                .then(a.center[0].total_cmp(&b.center[0]))
        });
        result.truncate(32);
        Ok(result)
    }

    fn word(&mut self, image: &RgbImage) -> Result<Option<String>> {
        let (w, h) = image.dimensions();
        let height = 48usize;
        let content = ((height as f32 * w as f32 / h as f32).ceil() as usize).clamp(1, 960);
        let width = 320.max(content);
        let input = imageops::resize(image, content as u32, height as u32, FilterType::Triangle);
        let plane = width * height;
        let mut data = vec![0.0; 3 * plane];
        for (x, y, rgb) in input.enumerate_pixels() {
            let i = y as usize * width + x as usize;
            for channel in 0..3 {
                data[channel * plane + i] = f32::from(rgb[2 - channel]) / 127.5 - 1.0;
            }
        }
        let tensor = Tensor::from_array(([1, 3, height, width], data.into_boxed_slice()))?;
        let outputs = self.recognize.run(ort::inputs![tensor])?;
        let (shape, probabilities) = outputs[0].try_extract_tensor::<f32>()?;
        let classes = self.alphabet.len();
        if shape.len() != 3 || shape[0] != 1 || shape[2] as usize != classes {
            return Err(format!(
                "Saída incompatível com o alfabeto do OCR: {shape:?}, {classes} símbolos"
            )
            .into());
        }
        let mut text = String::new();
        let mut previous = 0;
        let mut confidence = 0.0;
        let mut letters = 0;
        for step in probabilities.chunks_exact(classes) {
            let (index, probability) = step
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap();
            if index != 0 && index != previous {
                text.push_str(&self.alphabet[index]);
                confidence += *probability;
                letters += 1;
            }
            previous = index;
        }
        Ok((letters > 0 && confidence / letters as f32 >= self.min_score).then_some(text))
    }
}

struct Region {
    center: [f32; 2],
    axis: [f32; 2],
    width: f32,
    height: f32,
}

impl Region {
    fn from_component(points: &[[f32; 2]]) -> Option<Self> {
        let count = points.len() as f32;
        let mean = points
            .iter()
            .fold([0.0, 0.0], |[x, y], p| [x + p[0] / count, y + p[1] / count]);
        let [xx, xy, yy] = points.iter().fold([0.0, 0.0, 0.0], |[xx, xy, yy], p| {
            let (x, y) = (p[0] - mean[0], p[1] - mean[1]);
            [xx + x * x, xy + x * y, yy + y * y]
        });
        let angle = (2.0 * xy).atan2(xx - yy) / 2.0;
        let (sin, cos) = angle.sin_cos();
        let (mut min_x, mut max_x, mut min_y, mut max_y) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for &[x, y] in points {
            let u = cos * x + sin * y;
            let v = -sin * x + cos * y;
            min_x = min_x.min(u);
            max_x = max_x.max(u);
            min_y = min_y.min(v);
            max_y = max_y.max(v);
        }
        let width = max_x - min_x + 1.0;
        let height = max_y - min_y + 1.0;
        if width < 7.0 || height < 4.0 {
            return None;
        }
        let (u, v) = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0);
        let margin = 0.55 * width * height / (width + height);
        Some(Self {
            center: [cos * u - sin * v, sin * u + cos * v],
            axis: [cos, sin],
            width: width + 2.0 * margin,
            height: height + 2.0 * margin,
        })
    }

    fn extract(&self, image: &RgbImage) -> RgbImage {
        let [cos, sin] = self.axis;
        let [cx, cy] = self.center;
        RgbImage::from_fn(
            self.width.ceil() as u32,
            self.height.ceil() as u32,
            |x, y| {
                let (u, v) = (x as f32 - self.width / 2.0, y as f32 - self.height / 2.0);
                let sx = (cx + cos * u - sin * v)
                    .round()
                    .clamp(0.0, image.width() as f32 - 1.0);
                let sy = (cy + sin * u + cos * v)
                    .round()
                    .clamp(0.0, image.height() as f32 - 1.0);
                *image.get_pixel(sx as u32, sy as u32)
            },
        )
    }
}
