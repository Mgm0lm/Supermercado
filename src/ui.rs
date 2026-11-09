use crate::{
    checkout::{self, Checkout, Keywords, Score},
    config::{Config, Result},
    workers::{self, Found, Frame, Gesture, OcrRequest, Workers},
};
use eframe::egui::{
    self, Align, Align2, Color32, FontId, Frame as Card, Layout, Margin, Rect, RichText, Stroke,
    StrokeKind, Ui, Vec2,
};
use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};
use windows::Win32::System::Diagnostics::Debug::Beep;

const BLUE: Color32 = Color32::from_rgb(37, 99, 235);
const GREEN: Color32 = Color32::from_rgb(15, 138, 76);
const INK: Color32 = Color32::from_rgb(31, 36, 48);
const GREY: Color32 = Color32::from_rgb(107, 114, 128);
const BORDER: Color32 = Color32::from_rgb(227, 231, 238);
const CARD: Color32 = Color32::from_rgb(244, 246, 249);
const VIDEO: Color32 = Color32::from_rgb(16, 19, 24);
const RED: Color32 = Color32::from_rgb(239, 68, 68);
const CONNECTIONS: [(usize, usize); 21] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
];

pub struct App {
    cfg: Config,
    workers: Workers,
    checkout: Checkout,
    debug: bool,
    desync: bool,
    started: Instant,
    threshold: f32,
    keywords: Keywords,
    camera_epoch: u64,
    detection_id: u64,
    gesture_id: u64,
    ocr_token: u64,
    texture: Option<(u64, egui::TextureHandle)>,
    previous_frame: Option<(u64, Instant)>,
    previous_detection: Option<Instant>,
    camera_fps: f32,
    detection_fps: f32,
    status: String,
    gesture_status: String,
    scores: BTreeMap<String, Score>,
    toast: Option<(String, f64)>,
}

impl App {
    pub fn new(context: &egui::Context, base: &Path, cfg: Config, debug: bool) -> Result<Self> {
        let keywords = match fs::read(base.join("ocr_keywords.json")) {
            Ok(bytes) => serde_json::from_slice(&bytes)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                serde_json::from_str(include_str!("../Programa/ocr_keywords.json"))?
            }
            Err(error) => return Err(error.into()),
        };
        let mut style = egui::Style::default();
        style.visuals = egui::Visuals::light();
        style.visuals.override_text_color = Some(INK);
        style.spacing.item_spacing = Vec2::new(8.0, 9.0);
        style.spacing.button_padding = Vec2::new(10.0, 6.0);
        context.set_style(style);
        let workers = workers::start(&cfg, base);
        let threshold = cfg
            .conf_threshold_default
            .clamp(cfg.conf_threshold_min, cfg.conf_threshold_max);
        Ok(Self {
            desync: cfg.desync_bboxes,
            cfg,
            workers,
            checkout: Checkout::default(),
            debug,
            started: Instant::now(),
            threshold,
            keywords,
            camera_epoch: u64::MAX,
            detection_id: 0,
            gesture_id: 0,
            ocr_token: 1,
            texture: None,
            previous_frame: None,
            previous_detection: None,
            camera_fps: 0.0,
            detection_fps: 0.0,
            status: "Aguardando produto...".into(),
            gesture_status: "Mostre as duas mãos".into(),
            scores: BTreeMap::new(),
            toast: None,
        })
    }

    fn beep(&self) {
        if self.cfg.enable_sound && self.cfg.sound_volume > 0.0 {
            let frequency = self.cfg.sound_freq_hz.clamp(37, 32767);
            let duration = self.cfg.sound_duration_ms;
            std::thread::spawn(move || unsafe {
                let _ = Beep(frequency, duration);
            });
        }
    }

    fn reset_ocr(&mut self) {
        self.ocr_token += 1;
        let mut workers = self.workers.lock().unwrap();
        workers.ocr_request = None;
        workers.ocr = None;
    }

    fn finish(&mut self, now: f64, from_gesture: bool) {
        self.toast = Some((
            match self.checkout.finish(&self.cfg) {
                Some(total) => {
                    self.beep();
                    self.reset_ocr();
                    if from_gesture {
                        self.gesture_status = "Compra finalizada!".into();
                    }
                    format!("Compra finalizada\nTotal: {}\nVolte sempre", money(total))
                }
                None => {
                    if from_gesture {
                        self.gesture_status = "Carrinho vazio: nada a finalizar".into();
                    }
                    "Carrinho vazio\nInicie a compra".into()
                }
            },
            now + self.cfg.toast_seconds,
        ));
    }

    fn update_product(&mut self, found: &Arc<Found>, text: &str, now: f64) {
        self.detection_id = found.frame.id;
        self.detection_fps = fps(
            self.previous_detection.replace(found.finished),
            found.finished,
            self.detection_fps,
        );
        let valid = found
            .detections
            .iter()
            .filter(|d| d.confidence >= self.threshold && self.cfg.products.contains_key(&d.class));
        let has_product = valid.clone().next().is_some();
        let best = valid
            .filter(|d| !self.checkout.blocked.contains_key(&d.class))
            .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
            .map(|d| d.bounds);
        let text = if self.cfg.enable_ocr && self.cfg.ocr_use_box_crop && !has_product {
            self.reset_ocr();
            ""
        } else {
            text
        };
        self.scores = checkout::scores(
            &self.cfg,
            &found.detections,
            found.frame.pixels.dimensions(),
            self.threshold,
            text,
            &self.keywords,
        );
        if self
            .checkout
            .product(now, &self.cfg, self.threshold, &self.scores)
        {
            self.beep();
            self.reset_ocr();
        }
        let cooldown = now < self.checkout.cooldown_until;
        self.status = if cooldown {
            "Aguardando... (evitando leitura duplicada)".into()
        } else if let Some((class, _)) = &self.checkout.candidate {
            let product = &self.cfg.products[class].display_name;
            if self.debug {
                format!("Lendo: {product}  [{:.0} pts]", self.scores[class].total())
            } else {
                format!("Lendo: {product}")
            }
        } else {
            "Aguardando produto...".into()
        };
        if self.cfg.enable_ocr && !cooldown {
            if best.is_some() || (!self.cfg.ocr_use_box_crop && !has_product) {
                let bounds = if self.cfg.ocr_use_box_crop {
                    best
                } else {
                    None
                };
                let mut workers = self.workers.lock().unwrap();
                if workers.camera_epoch == self.camera_epoch && workers.frame.is_some() {
                    workers.ocr_request = Some(OcrRequest {
                        frame: found.frame.clone(),
                        bounds,
                        token: self.ocr_token,
                    });
                }
            }
        }
    }

    fn update_gesture(&mut self, gesture: &Gesture, now: f64) {
        self.gesture_id = gesture.frame.id;
        self.gesture_status = if now < self.checkout.gesture_cooldown_until {
            "Aguardando...".into()
        } else if gesture.confidence >= self.cfg.gesture_min_confidence {
            format!("Segure o gesto... ({:.0}%)", gesture.confidence * 100.0)
        } else if gesture.hands.is_empty() {
            "Mostre as duas mãos".into()
        } else {
            "Gesto não reconhecido".into()
        };
        if self.checkout.gesture(now, &self.cfg, gesture.confidence) {
            self.finish(now, true);
        }
    }

    fn camera_card(&mut self, ui: &mut Ui, now: f64) {
        card(ui, |ui| {
            ui.label(RichText::new("Câmera").strong().size(14.0));
            let (cameras, mut camera, mut gesture, gesture_error) = {
                let state = self.workers.lock().unwrap();
                (
                    state.cameras.clone(),
                    state.camera.clone(),
                    state.gesture_enabled,
                    state.gesture_error.clone(),
                )
            };
            let initial = camera.clone();
            ui.horizontal(|ui| {
                let current = cameras
                    .iter()
                    .find(|(id, _)| Some(id) == camera.as_ref())
                    .map_or("Câmera 0 (padrão)", |(_, name)| name);
                egui::ComboBox::from_id_salt("camera")
                    .selected_text(current)
                    .width((ui.available_width() - 55.0).max(100.0))
                    .show_ui(ui, |ui| {
                        for (id, label) in &cameras {
                            ui.selectable_value(&mut camera, Some(id.clone()), label);
                        }
                    });
                if button(ui, "↻", BLUE)
                    .on_hover_text("Atualizar câmeras")
                    .clicked()
                {
                    self.workers.lock().unwrap().refresh = true;
                }
            });
            if initial != camera {
                self.workers.lock().unwrap().select_camera(camera);
            }
            if ui
                .checkbox(&mut gesture, "Finalizar compra com gesto (2 mãos)")
                .changed()
            {
                let mut state = self.workers.lock().unwrap();
                state.gesture_enabled = gesture;
                state.gesture_epoch += 1;
                state.gesture = None;
                state.gesture_error.clear();
                self.checkout.gesture_start = None;
                self.gesture_status = "Mostre as duas mãos".into();
            }
            if gesture {
                ui.label(
                    RichText::new(if gesture_error.is_empty() {
                        &self.gesture_status
                    } else {
                        &gesture_error
                    })
                    .size(11.0)
                    .color(GREY),
                );
                let progress = progress(
                    self.checkout.gesture_start,
                    now,
                    self.cfg.gesture_hold_seconds,
                );
                ui.add(
                    egui::ProgressBar::new(progress)
                        .fill(BLUE)
                        .desired_height(8.0),
                );
            }
            ui.checkbox(&mut self.desync, "Vídeo fluido (caixas desincronizadas)");
        });
    }

    fn threshold_card(&mut self, ui: &mut Ui) {
        card(ui, |ui| {
            ui.label(
                RichText::new("Precisão mínima para detecção")
                    .strong()
                    .size(14.0),
            );
            let size = ui.available_width();
            let mut delta = 0.0;
            ui.horizontal(|ui| {
                if button(ui, "−", BLUE).clicked() {
                    delta -= self.cfg.conf_threshold_step;
                }
                ui.allocate_ui_with_layout(
                    Vec2::new((size - 100.0).max(28.0), 32.0),
                    Layout::top_down(Align::Center),
                    |ui| {
                        ui.label(
                            RichText::new(format!("{:.0}%", self.threshold * 100.0))
                                .size(18.0)
                                .strong(),
                        );
                    },
                );
                if button(ui, "+", BLUE).clicked() {
                    delta += self.cfg.conf_threshold_step;
                }
            });
            if delta != 0.0 {
                self.threshold = (self.threshold + delta)
                    .clamp(self.cfg.conf_threshold_min, self.cfg.conf_threshold_max);
                self.workers.lock().unwrap().threshold = self.threshold;
                self.checkout.candidate = None;
                self.reset_ocr();
            }
        });
    }

    fn debug_card(&self, ui: &mut Ui, text: &str) {
        card(ui, |ui| {
            ui.label(
                RichText::new(format!(
                    "DEBUG · mínimo para aceitar {:.1} pts",
                    self.threshold * self.cfg.score_yolo_max_points
                ))
                .monospace(),
            );
            egui::Grid::new("scores")
                .spacing(Vec2::new(9.0, 4.0))
                .show(ui, |ui| {
                    for label in ["Classe", "YOLO", "Centro", "OCR", "Total"] {
                        ui.label(RichText::new(label).monospace().strong().size(11.0));
                    }
                    ui.end_row();
                    let mut rows: Vec<_> = self.scores.iter().collect();
                    rows.sort_by(|a, b| b.1.total().total_cmp(&a.1.total()));
                    for (name, s) in rows.into_iter().take(5) {
                        let mark = if self.checkout.blocked.contains_key(name) {
                            "× "
                        } else {
                            ""
                        };
                        ui.label(
                            RichText::new(format!("{mark}{name}"))
                                .monospace()
                                .size(11.0),
                        );
                        for v in [s.yolo, s.center, s.ocr, s.total()] {
                            ui.label(RichText::new(format!("{v:.1}")).monospace().size(11.0));
                        }
                        ui.end_row();
                    }
                });
            ui.label(RichText::new(text).monospace().size(11.0));
        });
    }

    fn cart_card(&mut self, ui: &mut Ui, now: f64) {
        card(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Carrinho").strong().size(14.0));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let label = egui::Button::new(RichText::new("Nova compra").size(12.0))
                        .fill(Color32::WHITE)
                        .stroke(Stroke::new(1.0_f32, BORDER))
                        .corner_radius(8);
                    if ui.add(label).clicked() {
                        self.finish(now, false);
                    }
                });
            });
            let mut remove = None;
            let height = (ui.available_height() - 50.0).max(75.0);
            Card::new()
                .fill(Color32::WHITE)
                .stroke(Stroke::new(1.0_f32, BORDER))
                .corner_radius(10)
                .inner_margin(8)
                .show(ui, |ui| {
                    ui.set_min_height(height);
                    egui::ScrollArea::vertical()
                        .max_height(height)
                        .show(ui, |ui| {
                            egui::Grid::new("cart")
                                .min_col_width(30.0)
                                .spacing(Vec2::new(9.0, 9.0))
                                .striped(true)
                                .show(ui, |ui| {
                                    for heading in ["Produto", "Qtd", "Preço", "Subtotal", ""] {
                                        ui.label(RichText::new(heading).strong().size(12.0));
                                    }
                                    ui.end_row();
                                    let name_width = (ui.available_width() - 235.0).max(90.0);
                                    for (index, (name, qty)) in
                                        self.checkout.cart.iter().enumerate()
                                    {
                                        let item = &self.cfg.products[name];
                                        ui.allocate_ui(Vec2::new(name_width, 20.0), |ui| {
                                            ui.add(egui::Label::new(&item.display_name).wrap());
                                        });
                                        ui.label(qty.to_string());
                                        ui.label(money(item.price));
                                        ui.label(money(item.price * f64::from(*qty)));
                                        if button(ui, "Remover", RED).clicked() {
                                            remove = Some(index);
                                        }
                                        ui.end_row();
                                    }
                                });
                        });
                });
            if let Some(index) = remove {
                self.checkout.cart.remove(index);
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(
                    RichText::new(format!("Total: {}", money(self.checkout.total(&self.cfg))))
                        .color(GREEN)
                        .size(22.0)
                        .strong(),
                );
            });
        });
    }

    fn video(
        &mut self,
        ui: &mut Ui,
        now: f64,
        frame: Option<&Frame>,
        found: Option<&Found>,
        gesture: Option<&Gesture>,
        error: &str,
    ) {
        ui.horizontal(|ui| {
            let title = if self.checkout.candidate.is_some() {
                "Aguarde..."
            } else if self.checkout.cart.is_empty() {
                "Inicie a compra"
            } else {
                "Passe o próximo produto"
            };
            ui.label(RichText::new(title).size(42.0));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(
                    RichText::new("Pressione ESC para sair")
                        .size(13.0)
                        .color(GREY),
                );
            });
        });
        let (rect, _) = ui.allocate_exact_size(
            Vec2::new(
                ui.available_width(),
                (ui.available_height() - 56.0).max(100.0),
            ),
            egui::Sense::hover(),
        );
        ui.painter().rect_filled(rect, 14, VIDEO);
        if let Some(frame) = frame {
            if self.texture.as_ref().is_none_or(|(id, _)| *id != frame.id) {
                let pixels = egui::ColorImage::from_rgb(
                    [
                        frame.pixels.width() as usize,
                        frame.pixels.height() as usize,
                    ],
                    frame.pixels.as_raw(),
                );
                if let Some((id, texture)) = &mut self.texture {
                    texture.set(pixels, egui::TextureOptions::LINEAR);
                    *id = frame.id;
                } else {
                    self.texture = Some((
                        frame.id,
                        ui.ctx()
                            .load_texture("camera", pixels, egui::TextureOptions::LINEAR),
                    ));
                }
            }
            let (w, h) = (frame.pixels.width() as f32, frame.pixels.height() as f32);
            let scale = (rect.width() / w).min(rect.height() / h);
            let image = Rect::from_center_size(rect.center(), Vec2::new(w * scale, h * scale));
            let texture = &self.texture.as_ref().unwrap().1;
            egui::Image::new((texture.id(), image.size()))
                .corner_radius(14)
                .paint_at(ui, image);
            let point = |x, y| image.left_top() + Vec2::new(x * scale, y * scale);
            let painter = ui.painter().with_clip_rect(image);
            if let Some(found) = found {
                for detection in found
                    .detections
                    .iter()
                    .filter(|d| d.confidence >= self.threshold)
                {
                    let [x1, y1, x2, y2] = detection.bounds;
                    let active = self
                        .checkout
                        .candidate
                        .as_ref()
                        .is_some_and(|(name, _)| name == &detection.class);
                    let color = if active {
                        Color32::LIGHT_GREEN
                    } else {
                        Color32::LIGHT_BLUE
                    };
                    let box_rect = Rect::from_two_pos(point(x1, y1), point(x2, y2));
                    painter.rect_stroke(
                        box_rect,
                        0,
                        Stroke::new(2.0_f32, color),
                        StrokeKind::Inside,
                    );
                    let label = format!("{} {:.0}%", detection.class, detection.confidence * 100.0);
                    painter.text(
                        box_rect.left_top() + Vec2::new(3.0, 3.0),
                        Align2::LEFT_TOP,
                        label,
                        FontId::proportional(16.0),
                        color,
                    );
                }
            }
            if let Some(gesture) = gesture {
                let color = if gesture.confidence >= self.cfg.gesture_min_confidence {
                    Color32::LIGHT_GREEN
                } else {
                    Color32::LIGHT_GRAY
                };
                for hand in &gesture.hands {
                    let points = hand.points.map(|[x, y, _]| point(x * w, y * h));
                    for (a, b) in CONNECTIONS {
                        painter.line_segment([points[a], points[b]], Stroke::new(2.0_f32, color));
                    }
                    for p in points {
                        painter.circle_filled(p, 3.0, color);
                    }
                }
            }
            if self.cfg.show_fps {
                painter.text(
                    image.left_top() + Vec2::splat(12.0),
                    Align2::LEFT_TOP,
                    format!(
                        "Câmera {:.0} FPS  |  Detecção {:.0} FPS",
                        self.camera_fps, self.detection_fps
                    ),
                    FontId::monospace(13.0),
                    Color32::WHITE,
                );
            }
        } else {
            let message = if error.is_empty() {
                "Aguardando vídeo..."
            } else {
                "Sem vídeo"
            };
            ui.painter().text(
                rect.center(),
                Align2::CENTER_CENTER,
                message,
                FontId::proportional(17.0),
                Color32::WHITE,
            );
        }
        ui.label(
            RichText::new(if error.is_empty() {
                &self.status
            } else {
                error
            })
            .color(BLUE)
            .strong()
            .size(14.0),
        );
        let bar = progress(
            self.checkout.candidate.as_ref().map(|(_, start)| *start),
            now,
            self.cfg.min_detection_seconds,
        );
        ui.add(egui::ProgressBar::new(bar).fill(BLUE).show_percentage());
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        let now = self.started.elapsed().as_secs_f64();
        let (epoch, frame, found, gesture, ocr, error, ocr_error) = {
            let state = self.workers.lock().unwrap();
            let error = if !state.camera_error.is_empty() {
                &state.camera_error
            } else {
                &state.detector_status
            };
            (
                state.camera_epoch,
                state.frame.clone(),
                state.found.clone(),
                state.gesture.clone(),
                state.ocr.clone(),
                error.clone(),
                state.ocr_error.clone(),
            )
        };
        let new_frame = frame
            .as_ref()
            .filter(|f| f.captured.elapsed() < Duration::from_secs(2));
        let new_found = found
            .as_ref()
            .filter(|f| f.frame.captured.elapsed() < Duration::from_secs(2));
        let gesture = gesture
            .as_ref()
            .filter(|g| g.frame.captured.elapsed() < Duration::from_secs(1));
        if epoch != self.camera_epoch || new_found.is_none() {
            self.camera_epoch = epoch;
            self.checkout.candidate = None;
            self.scores.clear();
            self.status = "Aguardando produto...".into();
        }
        if gesture.is_none() {
            self.checkout.gesture_start = None;
        }
        if let Some(f) = new_frame.filter(|f| self.previous_frame.is_none_or(|(id, _)| id != f.id))
        {
            self.camera_fps = fps(
                self.previous_frame
                    .replace((f.id, f.captured))
                    .map(|(_, time)| time),
                f.captured,
                self.camera_fps,
            );
        }
        let fresh_ocr = ocr.as_ref().filter(|o| {
            o.token == self.ocr_token
                && o.finished.elapsed().as_secs_f64() <= self.cfg.ocr_result_ttl_seconds
        });
        if let Some(found) = new_found.filter(|f| f.frame.id != self.detection_id) {
            let text = fresh_ocr.map_or("", |o| o.text.as_str());
            self.update_product(found, text, now);
        }
        if let Some(gesture) = gesture.filter(|g| g.frame.id != self.gesture_id) {
            self.update_gesture(gesture, now);
        }
        let debug = if !self.cfg.enable_ocr {
            "OCR desativado".into()
        } else if !ocr_error.is_empty() {
            ocr_error
        } else if let Some(ocr) = fresh_ocr {
            format!("OCR ({} ms): {}", ocr.duration.as_millis(), ocr.text)
        } else {
            "OCR: aguardando leitura...".into()
        };
        let width = ctx.content_rect().width() * (1.0 - self.cfg.camera_panel_ratio);
        egui::SidePanel::right("controls")
            .resizable(false)
            .exact_width(width)
            .frame(Card::new().fill(Color32::WHITE).inner_margin(Margin {
                left: 8,
                right: 16,
                top: 16,
                bottom: 16,
            }))
            .show(ctx, |ui| {
                self.camera_card(ui, now);
                self.threshold_card(ui);
                if self.debug {
                    self.debug_card(ui, &debug);
                }
                self.cart_card(ui, now);
            });
        egui::CentralPanel::default()
            .frame(Card::new().fill(Color32::WHITE).inner_margin(Margin {
                left: 16,
                right: 8,
                top: 16,
                bottom: 16,
            }))
            .show(ctx, |ui| {
                let frame = if self.desync {
                    new_frame.map(AsRef::as_ref)
                } else {
                    new_found
                        .map(|f| f.frame.as_ref())
                        .or(new_frame.map(AsRef::as_ref))
                };
                self.video(
                    ui,
                    now,
                    frame,
                    new_found.map(AsRef::as_ref),
                    gesture.map(AsRef::as_ref),
                    &error,
                );
            });
        if let Some((message, _)) = self.toast.as_ref().filter(|(_, until)| now < *until) {
            egui::Area::new("toast".into())
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .order(egui::Order::Foreground)
                .interactable(false)
                .show(ctx, |ui| {
                    Card::new()
                        .fill(Color32::WHITE)
                        .stroke(Stroke::new(5.0_f32, Color32::BLACK))
                        .corner_radius(28)
                        .inner_margin(Margin::symmetric(75, 45))
                        .show(ui, |ui| {
                            ui.vertical_centered(|ui| {
                                for line in message.lines() {
                                    ui.label(
                                        RichText::new(line)
                                            .strong()
                                            .color(Color32::BLACK)
                                            .size(46.0),
                                    );
                                }
                            });
                        });
                });
        }
        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.workers.lock().unwrap().stop = true;
    }
}

fn button(ui: &mut Ui, label: &str, color: Color32) -> egui::Response {
    ui.add(
        egui::Button::new(RichText::new(label).color(Color32::WHITE).strong())
            .fill(color)
            .corner_radius(9),
    )
}

fn card(ui: &mut Ui, render: impl FnOnce(&mut Ui)) {
    Card::new()
        .fill(CARD)
        .stroke(Stroke::new(1.0_f32, BORDER))
        .corner_radius(14)
        .inner_margin(14)
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            render(ui);
        });
}

fn money(total: f64) -> String {
    format!("R$ {total:.2}").replace('.', ",")
}

fn progress(start: Option<f64>, now: f64, seconds: f64) -> f32 {
    start.map_or(0.0, |start| {
        ((now - start) / seconds).clamp(0.0, 1.0) as f32
    })
}

fn fps(previous: Option<Instant>, current: Instant, smooth: f32) -> f32 {
    previous.map_or(0.0, |time| {
        let rate = 1.0 / current.duration_since(time).as_secs_f32().max(0.001);
        if smooth == 0.0 {
            rate
        } else {
            smooth * 0.85 + rate * 0.15
        }
    })
}
