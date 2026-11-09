use crate::{
    checkout::{self, Checkout, Keywords, Score},
    config::{Config, Result},
    workers::{self, Found, Frame, Gesture, OcrRequest, Workers},
};
use eframe::egui::{
    self, Align, Align2, Color32, CornerRadius, FontFamily, FontId, Frame as Card, Layout, Rect,
    RichText, Stroke, StrokeKind, Ui, Vec2, pos2,
};
use std::{
    cell::Cell,
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    rc::Rc,
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
const CONTROL_BORDER: Color32 = Color32::from_rgb(208, 213, 221);
const GRID: Color32 = Color32::from_rgb(238, 241, 245);
const HEADER_INK: Color32 = Color32::from_rgb(75, 85, 99);

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
    fullscreen_attempt: u8,
    fullscreen_probe: Option<Rc<Cell<bool>>>,
}

impl App {
    pub fn new(
        context: &egui::Context,
        base: &Path,
        cfg: Config,
        debug: bool,
        fullscreen_probe: Option<Rc<Cell<bool>>>,
    ) -> Result<Self> {
        let mut keywords =
            checkout::parse_keywords(include_bytes!("../Programa/ocr_keywords.json"))?;
        match fs::read(base.join("ocr_keywords.json")) {
            Ok(bytes) => match checkout::parse_keywords(&bytes) {
                Ok(overrides) => keywords.extend(overrides),
                Err(error) => {
                    eprintln!("Falha ao ler ocr_keywords.json. Usando os padrões: {error}")
                }
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => eprintln!("Falha ao abrir ocr_keywords.json. Usando os padrões: {error}"),
        }
        let mut style = egui::Style::default();
        style.visuals = egui::Visuals::light();
        style.visuals.widgets.inactive.fg_stroke.color = INK;
        style.visuals.widgets.noninteractive.fg_stroke.color = INK;
        style.spacing.item_spacing = Vec2::splat(6.0);
        style.spacing.button_padding = Vec2::new(8.0, 4.0);
        context.set_style(style);
        set_fonts(context);
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
            fullscreen_attempt: 0,
            fullscreen_probe,
        })
    }

    fn check_window(&mut self, ctx: &egui::Context) {
        let (fullscreen, dimensions) = ctx.input(|input| {
            let viewport = input.viewport();
            (
                viewport.fullscreen == Some(true),
                viewport.inner_rect.zip(viewport.monitor_size),
            )
        });
        if let Some((rect, monitor)) = dimensions {
            if fullscreen && (rect.size() - monitor).abs().max_elem() <= 2.0 {
                if let Some(probe) = self.fullscreen_probe.take() {
                    println!(
                        "Tela inteira: {:.0} × {:.0} pontos, monitor {:.0} × {:.0}.",
                        rect.width(),
                        rect.height(),
                        monitor.x,
                        monitor.y
                    );
                    probe.set(true);
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                return;
            }
            if self.fullscreen_attempt < 2
                && self.started.elapsed()
                    >= Duration::from_millis(200 * (u64::from(self.fullscreen_attempt) + 1))
            {
                let reenter = self.fullscreen_attempt != 0 || !fullscreen;
                ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(reenter));
                self.fullscreen_attempt += 1;
            }
        }
        if self.fullscreen_probe.is_some() && self.started.elapsed() > Duration::from_secs(5) {
            eprintln!("Área da janela: {dimensions:?}");
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
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
            label_row(ui, "Câmera", 14.0, 19.0);
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
                let width = (ui.available_width() - 42.0).max(0.0);
                ui.scope(|ui| {
                    ui.spacing_mut().interact_size.y = 33.0;
                    {
                        let widget = &mut ui.visuals_mut().widgets.inactive;
                        widget.weak_bg_fill = Color32::WHITE;
                        widget.bg_fill = Color32::WHITE;
                        widget.bg_stroke = Stroke::new(1.0_f32, CONTROL_BORDER);
                        widget.corner_radius = CornerRadius::same(8);
                    }
                    egui::ComboBox::from_id_salt("camera")
                        .selected_text(RichText::new(current).size(13.0))
                        .width(width)
                        .show_ui(ui, |ui| {
                            for (id, label) in &cameras {
                                ui.selectable_value(&mut camera, Some(id.clone()), label);
                            }
                        });
                });
                if ui
                    .add_sized([36.0, 35.0], blue_button("↻", 14.0))
                    .on_hover_text("Atualizar câmeras")
                    .clicked()
                {
                    self.workers.lock().unwrap().refresh = true;
                }
            });
            if initial != camera {
                self.workers.lock().unwrap().select_camera(camera);
            }
            if checkbox(ui, &mut gesture, "Finalizar compra com gesto (2 mãos)") {
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
                let value = progress(
                    self.checkout.gesture_start,
                    now,
                    self.cfg.gesture_hold_seconds,
                );
                progress_bar(ui, value, 8.0, false);
            }
            checkbox(
                ui,
                &mut self.desync,
                "Vídeo fluido (caixas desincronizadas)",
            );
        });
    }

    fn threshold_card(&mut self, ui: &mut Ui) {
        card(ui, |ui| {
            label_row(ui, "Precisão mínima para detecção", 14.0, 19.0);
            let (row, _) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 32.0), egui::Sense::hover());
            let minus = Rect::from_min_size(row.left_top(), Vec2::new(44.0, 32.0));
            let plus =
                Rect::from_min_size(pos2(row.right() - 44.0, row.top()), Vec2::new(44.0, 32.0));
            ui.painter().text(
                row.center(),
                Align2::CENTER_CENTER,
                format!("{:.0}%", self.threshold * 100.0),
                font(18.0, "bold"),
                INK,
            );
            let mut delta = 0.0;
            if ui.place(minus, blue_button("-", 18.0)).clicked() {
                delta -= self.cfg.conf_threshold_step;
            }
            if ui.place(plus, blue_button("+", 18.0)).clicked() {
                delta += self.cfg.conf_threshold_step;
            }
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
            let (row, _) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 30.0), egui::Sense::hover());
            ui.painter().text(
                row.left_center(),
                Align2::LEFT_CENTER,
                "Carrinho",
                font(14.0, "semibold"),
                INK,
            );
            let new_purchase =
                Rect::from_min_size(pos2(row.right() - 98.0, row.top()), Vec2::new(98.0, 30.0));
            if ui
                .place(
                    new_purchase,
                    egui::Button::new(RichText::new("Nova compra").font(font(12.0, "semibold")))
                        .fill(Color32::WHITE)
                        .stroke(Stroke::new(1.0_f32, CONTROL_BORDER))
                        .corner_radius(10),
                )
                .clicked()
            {
                self.finish(now, false);
            }
            let width = ui.available_width();
            let height = (ui.available_height() - 36.0).max(35.0);
            let mut removed = None;
            Card::new()
                .fill(Color32::WHITE)
                .stroke(Stroke::new(1.0_f32, BORDER))
                .corner_radius(14)
                .show(ui, |ui| {
                    ui.set_width((width - 2.0).max(0.0));
                    ui.set_height((height - 2.0).max(0.0));
                    ui.spacing_mut().item_spacing.y = 0.0;
                    let columns = cart_columns(ui.available_width());
                    let (header, _) = ui.allocate_exact_size(
                        Vec2::new(ui.available_width(), 32.0),
                        egui::Sense::hover(),
                    );
                    ui.painter().rect_filled(
                        header,
                        CornerRadius {
                            nw: 13,
                            ne: 13,
                            sw: 0,
                            se: 0,
                        },
                        CARD,
                    );
                    for (column, title) in ["Produto", "Qtd", "Preço", "Subtotal"]
                        .into_iter()
                        .enumerate()
                    {
                        let cell = cart_cell(header, &columns, column);
                        ui.painter().text(
                            cell.center(),
                            Align2::CENTER_CENTER,
                            title,
                            font(12.0, "semibold"),
                            HEADER_INK,
                        );
                    }
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            for (index, (name, qty)) in self.checkout.cart.iter().enumerate() {
                                let item = &self.cfg.products[name];
                                let (row, _) = ui.allocate_exact_size(
                                    Vec2::new(header.width(), 30.0),
                                    egui::Sense::hover(),
                                );
                                let values = [
                                    item.display_name.clone(),
                                    qty.to_string(),
                                    money(item.price),
                                    money(item.price * f64::from(*qty)),
                                ];
                                for (column, value) in values.iter().enumerate() {
                                    let cell = cart_cell(row, &columns, column);
                                    let mut job = egui::text::LayoutJob::simple_singleline(
                                        value.clone(),
                                        FontId::proportional(13.0),
                                        Color32::BLACK,
                                    );
                                    job.wrap.max_width = (cell.width() - 6.0).max(0.0);
                                    job.wrap.max_rows = 1;
                                    job.wrap.break_anywhere = true;
                                    let text = ui.painter().layout_job(job);
                                    let origin =
                                        cell.left_center() + Vec2::new(3.0, -text.size().y / 2.0);
                                    ui.painter().with_clip_rect(cell).galley(
                                        origin,
                                        text,
                                        Color32::BLACK,
                                    );
                                    ui.painter().line_segment(
                                        [cell.right_top(), cell.right_bottom()],
                                        Stroke::new(1.0_f32, GRID),
                                    );
                                }
                                let remove = cart_cell(row, &columns, 4).shrink(0.5);
                                if ui
                                    .place(
                                        remove,
                                        egui::Button::new(
                                            RichText::new("Remover")
                                                .size(12.0)
                                                .color(Color32::WHITE),
                                        )
                                        .fill(RED)
                                        .corner_radius(10),
                                    )
                                    .clicked()
                                {
                                    removed = Some(index);
                                }
                                ui.painter().line_segment(
                                    [row.left_bottom(), row.right_bottom()],
                                    Stroke::new(1.0_f32, GRID),
                                );
                            }
                        });
                });
            if let Some(index) = removed {
                self.checkout.cart.remove(index);
            }
            let (total, _) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 30.0), egui::Sense::hover());
            let amount = if self.checkout.cart.is_empty() {
                "R$ 0,00".into()
            } else {
                money(self.checkout.total(&self.cfg))
            };
            ui.painter().text(
                total.right_center(),
                Align2::RIGHT_CENTER,
                format!("Total: {amount}"),
                font(22.0, "bold"),
                GREEN,
            );
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
        ui.spacing_mut().item_spacing.y = 10.0;
        let title = if self.checkout.candidate.is_some() {
            "Aguarde..."
        } else if self.checkout.cart.is_empty() {
            "Inicie a compra"
        } else {
            "Passe o próximo produto"
        };
        let size = (ui.ctx().content_rect().height() * 0.055).floor().max(40.0);
        let (header, _) = ui.allocate_exact_size(
            Vec2::new(ui.available_width(), size + 16.0),
            egui::Sense::hover(),
        );
        let hint = ui.painter().layout_no_wrap(
            "Pressione ESC para sair".into(),
            FontId::proportional(13.0),
            GREY,
        );
        let title_font = FontId::proportional(size);
        let title_color = Color32::from_rgb(17, 17, 17);
        let title = ui
            .painter()
            .layout_job(egui::text::LayoutJob::simple_singleline(
                title.to_owned(),
                title_font,
                title_color,
            ));
        let clip = Rect::from_min_max(
            header.min,
            pos2(header.right() - hint.size().x - 10.0, header.bottom()),
        );
        ui.painter().with_clip_rect(clip).galley(
            header.left_center() - Vec2::new(0.0, title.size().y / 2.0),
            title,
            title_color,
        );
        ui.painter().galley(
            header.right_center() - Vec2::new(hint.size().x, hint.size().y / 2.0),
            hint,
            GREY,
        );
        let (rect, _) = ui.allocate_exact_size(
            Vec2::new(
                ui.available_width(),
                (ui.available_height() - 55.0).max(0.0),
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
                .corner_radius(if (image.size() - rect.size()).abs().max_elem() <= 1.0 {
                    14
                } else {
                    0
                })
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
                        Color32::from_rgb(76, 138, 17)
                    } else {
                        Color32::from_rgb(235, 99, 37)
                    };
                    let box_rect = Rect::from_two_pos(point(x1, y1), point(x2, y2));
                    let thickness = if active { 3.0 * scale } else { 2.0 * scale };
                    painter.rect_stroke(
                        box_rect,
                        0,
                        Stroke::new(thickness, color),
                        StrokeKind::Inside,
                    );
                    let label = format!("{} {:.0}%", detection.class, detection.confidence * 100.0);
                    let text = painter.layout_no_wrap(
                        label,
                        font((17.0 * scale).max(10.0), "semibold"),
                        Color32::WHITE,
                    );
                    let origin = pos2(
                        box_rect.left(),
                        (box_rect.top() - text.size().y - 5.0).max(image.top()),
                    );
                    painter.rect_filled(
                        Rect::from_min_size(origin, text.size() + Vec2::new(8.0, 5.0)),
                        0,
                        color,
                    );
                    painter.galley(origin + Vec2::new(4.0, 1.0), text, Color32::WHITE);
                }
            }
            if let Some(gesture) = gesture {
                let color = if self.checkout.gesture_start.is_some() {
                    Color32::from_rgb(76, 138, 17)
                } else {
                    Color32::from_gray(200)
                };
                for hand in &gesture.hands {
                    for [x, y, _] in hand.points {
                        painter.circle_filled(point(x * w, y * h), (3.0 * scale).max(1.0), color);
                    }
                }
            }
            if self.cfg.show_fps {
                let text = format!(
                    "cam {:4.1} fps  |  IA {:4.1} fps",
                    self.camera_fps, self.detection_fps
                );
                let color = Color32::from_gray(220);
                let text =
                    painter.layout_no_wrap(text, FontId::monospace((13.0 * scale).max(8.0)), color);
                let origin = image.left_top() + Vec2::new(4.0 * scale, 2.0 * scale);
                painter.rect_filled(
                    Rect::from_min_size(origin, text.size() + Vec2::new(12.0, 6.0)),
                    0,
                    Color32::from_black_alpha(89),
                );
                painter.galley(origin + Vec2::new(6.0, 2.0), text, color);
            }
        }
        let (status, _) =
            ui.allocate_exact_size(Vec2::new(ui.available_width(), 19.0), egui::Sense::hover());
        ui.painter().with_clip_rect(status).text(
            status.left_center(),
            Align2::LEFT_CENTER,
            if error.is_empty() {
                &self.status
            } else {
                error
            },
            font(14.0, "semibold"),
            BLUE,
        );
        let bar = progress(
            self.checkout.candidate.as_ref().map(|(_, start)| *start),
            now,
            self.cfg.min_detection_seconds,
        );
        progress_bar(ui, bar, 16.0, true);
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.check_window(ctx);
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
        egui::CentralPanel::default()
            .frame(Card::new().fill(Color32::WHITE).inner_margin(16))
            .show(ctx, |ui| {
                let size = ui.available_size();
                let width = (size.x - 16.0).max(0.0);
                let left = (width * self.cfg.camera_panel_ratio).round();
                ui.spacing_mut().item_spacing.x = 16.0;
                ui.horizontal_top(|ui| {
                    ui.allocate_ui_with_layout(
                        Vec2::new(left, size.y),
                        Layout::top_down(Align::Min),
                        |ui| {
                            ui.set_width(left);
                            ui.set_height(size.y);
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
                        },
                    );
                    let right = width - left;
                    ui.allocate_ui_with_layout(
                        Vec2::new(right, size.y),
                        Layout::top_down(Align::Min),
                        |ui| {
                            ui.set_width(right);
                            ui.set_height(size.y);
                            ui.spacing_mut().item_spacing.y = 14.0;
                            self.camera_card(ui, now);
                            self.threshold_card(ui);
                            if self.debug {
                                self.debug_card(ui, &debug);
                            }
                            self.cart_card(ui, now);
                        },
                    );
                });
            });
        if let Some((message, _)) = self.toast.as_ref().filter(|(_, until)| now < *until) {
            paint_toast(ctx, message);
        }
        ctx.request_repaint_after(Duration::from_millis(16));
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.workers.lock().unwrap().stop = true;
    }
}

fn set_fonts(ctx: &egui::Context) {
    let mut definitions = egui::FontDefinitions::default();
    let directory = std::env::var_os("WINDIR")
        .map(PathBuf::from)
        .map(|p| p.join("Fonts"));
    let regular = definitions.families[&FontFamily::Proportional].clone();
    for (name, filename, family) in [
        ("regular", "segoeui.ttf", FontFamily::Proportional),
        (
            "semibold",
            "seguisb.ttf",
            FontFamily::Name("semibold".into()),
        ),
        ("bold", "segoeuib.ttf", FontFamily::Name("bold".into())),
        ("mono", "consola.ttf", FontFamily::Monospace),
    ] {
        let list = definitions
            .families
            .entry(family)
            .or_insert_with(|| regular.clone());
        if let Some(data) = directory
            .as_ref()
            .and_then(|dir| fs::read(dir.join(filename)).ok())
        {
            definitions
                .font_data
                .insert(name.into(), Arc::new(egui::FontData::from_owned(data)));
            list.insert(0, name.into());
        }
    }
    ctx.set_fonts(definitions);
}

fn font(size: f32, weight: &'static str) -> FontId {
    FontId::new(size, FontFamily::Name(weight.into()))
}

fn blue_button(label: &str, size: f32) -> egui::Button<'_> {
    egui::Button::new(
        RichText::new(label)
            .font(font(size, "semibold"))
            .color(Color32::WHITE),
    )
    .fill(BLUE)
    .corner_radius(10)
}

fn label_row(ui: &mut Ui, label: &str, size: f32, height: f32) {
    let (rect, _) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), height),
        egui::Sense::hover(),
    );
    ui.painter().text(
        rect.left_center(),
        Align2::LEFT_CENTER,
        label,
        font(size, "semibold"),
        INK,
    );
}

fn checkbox(ui: &mut Ui, value: &mut bool, label: &str) -> bool {
    let (rect, mut response) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 25.0), egui::Sense::click());
    if response.clicked() {
        *value = !*value;
        response.mark_changed();
    }
    response.widget_info(|| {
        egui::WidgetInfo::selected(egui::WidgetType::Checkbox, ui.is_enabled(), *value, label)
    });
    let mark = Rect::from_center_size(rect.left_center() + Vec2::new(11.0, 0.0), Vec2::splat(14.0));
    ui.painter().rect(
        mark,
        0,
        Color32::WHITE,
        Stroke::new(1.0_f32, CONTROL_BORDER),
        StrokeKind::Inside,
    );
    if *value {
        let origin = mark.left_top();
        ui.painter().add(egui::Shape::line(
            vec![
                origin + Vec2::new(3.0, 7.0),
                origin + Vec2::new(6.0, 10.0),
                origin + Vec2::new(11.0, 3.0),
            ],
            Stroke::new(1.6_f32, INK),
        ));
    }
    ui.painter().text(
        rect.left_center() + Vec2::new(24.0, 0.0),
        Align2::LEFT_CENTER,
        label,
        FontId::proportional(13.0),
        INK,
    );
    response.clicked()
}

fn paint_toast(ctx: &egui::Context, message: &str) {
    let screen = ctx.content_rect();
    let painter = ctx.layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new("toast"),
    ));
    let mut size = (screen.height() * 0.09).floor().max(48.0);
    let (lines, width) = loop {
        let lines: Vec<_> = message
            .lines()
            .map(|line| painter.layout_no_wrap(line.to_owned(), font(size, "bold"), Color32::BLACK))
            .collect();
        let text_width = lines.iter().map(|text| text.size().x).fold(0.0, f32::max);
        let width = (text_width + 3.2 * size + 10.0).max(screen.width() * 0.62);
        if width <= screen.width() * 0.94 || size <= 28.0 {
            break (lines, width);
        }
        size = (size - 6.0).max(28.0);
    };
    let padding = size * 0.7 + 5.0;
    let height = lines.iter().map(|text| text.size().y).sum::<f32>() + 2.0 * padding;
    let rect = Rect::from_center_size(screen.center(), Vec2::new(width, height));
    painter.rect(
        rect,
        28,
        Color32::WHITE,
        Stroke::new(5.0_f32, Color32::BLACK),
        StrokeKind::Inside,
    );
    let mut y = rect.top() + padding;
    for line in lines {
        let x = rect.center().x - line.size().x / 2.0;
        let line_height = line.size().y;
        painter.galley(pos2(x, y), line, Color32::BLACK);
        y += line_height;
    }
}

fn card(ui: &mut Ui, render: impl FnOnce(&mut Ui)) {
    Card::new()
        .fill(CARD)
        .stroke(Stroke::new(1.0_f32, BORDER))
        .corner_radius(14)
        .inner_margin(14)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing = Vec2::splat(6.0);
            render(ui);
        });
}

fn cart_columns(width: f32) -> [f32; 5] {
    let fixed = ((width - 115.0) / 4.0).clamp(0.0, 100.0);
    [width - 4.0 * fixed, fixed, fixed, fixed, fixed]
}

fn cart_cell(row: Rect, widths: &[f32; 5], column: usize) -> Rect {
    let x = row.left() + widths[..column].iter().sum::<f32>();
    Rect::from_min_size(pos2(x, row.top()), Vec2::new(widths[column], row.height()))
}

fn progress_bar(ui: &mut Ui, value: f32, height: f32, show_percent: bool) {
    let (rect, _) = ui.allocate_exact_size(
        Vec2::new(ui.available_width(), height),
        egui::Sense::hover(),
    );
    ui.painter().rect(
        rect,
        8,
        GRID,
        Stroke::new(1.0_f32, CONTROL_BORDER),
        StrokeKind::Inside,
    );
    if value > 0.0 {
        let inner = rect.shrink(1.0);
        let fill = Rect::from_min_size(inner.min, Vec2::new(inner.width() * value, inner.height()));
        ui.painter().rect_filled(fill, 7, BLUE);
    }
    if show_percent {
        ui.painter().text(
            rect.center(),
            Align2::CENTER_CENTER,
            format!("{:.0}%", value * 100.0),
            FontId::proportional(10.0),
            Color32::BLACK,
        );
    }
}

fn money(total: f64) -> String {
    format!("R$ {total:.2}")
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
