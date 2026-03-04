#[cfg(all(feature = "camera-v4l", feature = "display"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fercuda_vision_lite::camera::{
        discover_cameras, gray_to_argb32, spawn_highspeed_poller, yuyv_to_argb32, CaptureEncoding,
        PollerConfig,
    };
    use minifb::{Key, Window, WindowOptions};
    use std::sync::mpsc::TryRecvError;
    use std::time::Instant;
    use v4l::format::FourCC;

    let args: Vec<String> = std::env::args().collect();
    let mut dev_idx = 0usize;
    let mut req_w = 1280u32;
    let mut req_h = 720u32;
    let mut target_capture_fps = 120u32;
    let mut target_window_fps = 120u32;
    let mut fourcc = *b"YUYV";

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--device" => {
                i += 1;
                dev_idx = args.get(i).ok_or("missing --device value")?.parse()?;
            }
            "--width" => {
                i += 1;
                req_w = args.get(i).ok_or("missing --width value")?.parse()?;
            }
            "--height" => {
                i += 1;
                req_h = args.get(i).ok_or("missing --height value")?.parse()?;
            }
            "--capture-fps" => {
                i += 1;
                target_capture_fps = args
                    .get(i)
                    .ok_or("missing --capture-fps value")?
                    .parse()?;
            }
            "--window-fps" => {
                i += 1;
                target_window_fps = args
                    .get(i)
                    .ok_or("missing --window-fps value")?
                    .parse()?;
            }
            "--fourcc" => {
                i += 1;
                let s = args.get(i).ok_or("missing --fourcc value")?;
                let b = s.as_bytes();
                if b.len() != 4 {
                    return Err("fourcc must be exactly 4 chars (e.g. YUYV, MJPG, GREY)".into());
                }
                fourcc = [b[0], b[1], b[2], b[3]];
            }
            "--help" | "-h" => {
                println!("Usage: cargo run --example live_camera_view --features \"camera-v4l display parallel cv\" -- [--device N] [--width W] [--height H] [--capture-fps N] [--window-fps N] [--fourcc YUYV|MJPG|GREY]");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    let cams = discover_cameras(16);
    if cams.is_empty() {
        return Err("no /dev/video* cameras found".into());
    }
    println!("discovered cameras:");
    for c in &cams {
        println!("  {}: {} ({})", c.index, c.path, c.name);
    }

    let poll_cfg = PollerConfig {
        device_index: dev_idx,
        width: req_w,
        height: req_h,
        fourcc: FourCC::new(&fourcc),
        target_fps: target_capture_fps,
        queue_depth: 4,
        poll_timeout_ms: 2,
    };

    let (rx, _poller_thread) = spawn_highspeed_poller(poll_cfg)?;
    let first = rx.recv().map_err(|_| "poller failed before first frame")?;
    let enc = first.encoding;
    let fmt_w = first.width;
    let fmt_h = first.height;
    let fourcc_name = String::from_utf8_lossy(&fourcc).to_string();

    println!(
        "opened /dev/video{} format={} {}x{} capture_fps={} window_fps={}",
        dev_idx, fourcc_name, fmt_w, fmt_h, target_capture_fps, target_window_fps
    );

    let width = fmt_w as usize;
    let height = fmt_h as usize;
    let pixel_count = width * height;

    let mut window = Window::new(
        "feRcuda live_camera_view",
        width,
        height,
        WindowOptions::default(),
    )?;
    window.set_target_fps(target_window_fps as usize);

    let mut argb = vec![0u32; pixel_count];

    match enc {
        CaptureEncoding::Yuyv => yuyv_to_argb32(fmt_w, fmt_h, &first.data, &mut argb)?,
        CaptureEncoding::Gray => gray_to_argb32(fmt_w, fmt_h, &first.data, &mut argb)?,
        CaptureEncoding::Mjpg => {
            #[cfg(feature = "cv")]
            {
                let (_w, _h) =
                    fercuda_vision_lite::camera::mjpeg_to_argb32(fmt_w, fmt_h, &first.data, &mut argb)?;
            }
        }
        CaptureEncoding::Unknown => {}
    }

    let mut cam_frames = 0u64;
    let mut win_frames = 0u64;
    let mut last_stats_t = Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        loop {
            match rx.try_recv() {
                Ok(pkt) => {
                    match pkt.encoding {
                        CaptureEncoding::Yuyv => {
                            yuyv_to_argb32(pkt.width, pkt.height, &pkt.data, &mut argb)?;
                        }
                        CaptureEncoding::Gray => {
                            gray_to_argb32(pkt.width, pkt.height, &pkt.data, &mut argb)?;
                        }
                        CaptureEncoding::Mjpg => {
                            #[cfg(feature = "cv")]
                            {
                                let (_w, _h) = fercuda_vision_lite::camera::mjpeg_to_argb32(
                                    pkt.width,
                                    pkt.height,
                                    &pkt.data,
                                    &mut argb,
                                )?;
                            }
                        }
                        CaptureEncoding::Unknown => {}
                    }
                    cam_frames += 1;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }

        window.update_with_buffer(&argb, width, height)?;
        win_frames += 1;

        let dt = last_stats_t.elapsed().as_secs_f32();
        if dt >= 1.0 {
            let cam_fps = cam_frames as f32 / dt;
            let win_fps = win_frames as f32 / dt;
            cam_frames = 0;
            win_frames = 0;
            last_stats_t = Instant::now();
            window.set_title(&format!(
                "feRcuda live_camera_view | dev={} {}x{} {} | cam:{:.1} fps | win:{:.1} fps",
                dev_idx, fmt_w, fmt_h, fourcc_name, cam_fps, win_fps
            ));
        }
    }

    Ok(())
}

#[cfg(not(all(feature = "camera-v4l", feature = "display")))]
fn main() {
    println!("Enable features: camera-v4l display (and optionally parallel cv)");
}
