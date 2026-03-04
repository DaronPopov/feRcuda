#[cfg(feature = "camera-v4l")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fercuda_vision_lite::camera::{discover_cameras, spawn_highspeed_poller, PollerConfig};
    use std::time::{Duration, Instant};
    use v4l::format::FourCC;

    #[derive(Clone, Copy)]
    struct Args {
        device: usize,
        width: u32,
        height: u32,
        fps: u32,
        seconds: f32,
        queue_depth: usize,
        fourcc: [u8; 4],
    }

    fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
        let mut a = Args {
            device: 0,
            width: 1280,
            height: 720,
            fps: 120,
            seconds: 3.0,
            queue_depth: 4,
            fourcc: *b"YUYV",
        };
        let args: Vec<String> = std::env::args().collect();
        let mut i = 1usize;
        while i < args.len() {
            match args[i].as_str() {
                "--device" => {
                    i += 1;
                    a.device = args.get(i).ok_or("missing --device value")?.parse()?;
                }
                "--width" => {
                    i += 1;
                    a.width = args.get(i).ok_or("missing --width value")?.parse()?;
                }
                "--height" => {
                    i += 1;
                    a.height = args.get(i).ok_or("missing --height value")?.parse()?;
                }
                "--fps" => {
                    i += 1;
                    a.fps = args.get(i).ok_or("missing --fps value")?.parse()?;
                }
                "--seconds" => {
                    i += 1;
                    a.seconds = args.get(i).ok_or("missing --seconds value")?.parse()?;
                }
                "--queue-depth" => {
                    i += 1;
                    a.queue_depth = args.get(i).ok_or("missing --queue-depth value")?.parse()?;
                }
                "--fourcc" => {
                    i += 1;
                    let s = args.get(i).ok_or("missing --fourcc value")?;
                    let b = s.as_bytes();
                    if b.len() != 4 {
                        return Err("fourcc must be exactly 4 chars (e.g. YUYV, MJPG, GREY)".into());
                    }
                    a.fourcc = [b[0], b[1], b[2], b[3]];
                }
                "--help" | "-h" => {
                    println!(
                        "Usage: cargo run --example highspeed_poller --features \"camera-v4l\" -- \\
                         [--device N] [--width W] [--height H] [--fps N] [--seconds S] [--queue-depth N] [--fourcc YUYV|MJPG|GREY]"
                    );
                    std::process::exit(0);
                }
                _ => {}
            }
            i += 1;
        }
        Ok(a)
    }

    let args = parse_args()?;

    let cams = discover_cameras(16);
    if cams.is_empty() {
        return Err("no /dev/video* cameras found".into());
    }
    println!("cameras:");
    for c in &cams {
        println!("  {}: {} ({})", c.index, c.path, c.name);
    }

    let cfg = PollerConfig {
        device_index: args.device,
        width: args.width,
        height: args.height,
        fourcc: FourCC::new(&args.fourcc),
        target_fps: args.fps,
        queue_depth: args.queue_depth.max(1),
        poll_timeout_ms: 2,
    };

    let (rx, _poller_thread) = spawn_highspeed_poller(cfg)?;
    let first = rx.recv_timeout(Duration::from_secs(2)).map_err(|_| {
        "poller failed to produce first frame (check camera permissions/format/fps capability)"
    })?;

    println!(
        "polling /dev/video{} {}x{} fourcc={} target_fps={} test_s={:.2}",
        args.device,
        first.width,
        first.height,
        String::from_utf8_lossy(&args.fourcc),
        args.fps,
        args.seconds
    );

    let start = Instant::now();
    let mut last_t = start;
    let mut frame_count: u64 = 1;
    let mut inter_us: Vec<f64> = Vec::new();
    let mut last_seq = first.sequence;
    let mut seq_gaps: u64 = 0;

    while start.elapsed().as_secs_f32() < args.seconds {
        if let Ok(pkt) = rx.recv_timeout(Duration::from_millis(100)) {
            let now = Instant::now();
            inter_us.push((now - last_t).as_secs_f64() * 1e6);
            last_t = now;
            frame_count += 1;
            if pkt.sequence > last_seq + 1 {
                seq_gaps += pkt.sequence - (last_seq + 1);
            }
            last_seq = pkt.sequence;
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let achieved_fps = frame_count as f64 / elapsed;

    inter_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p = |v: &Vec<f64>, q: f64| -> f64 {
        if v.is_empty() {
            0.0
        } else {
            let idx = ((v.len() as f64 - 1.0) * q).round() as usize;
            v[idx.min(v.len() - 1)]
        }
    };

    println!(
        "frames={} elapsed_s={:.3} achieved_fps={:.2}",
        frame_count, elapsed, achieved_fps
    );
    println!(
        "inter_frame_us p50={:.1} p95={:.1} p99={:.1} max={:.1}",
        p(&inter_us, 0.50),
        p(&inter_us, 0.95),
        p(&inter_us, 0.99),
        inter_us.last().copied().unwrap_or(0.0)
    );
    println!("sequence_gaps={} (dropped before app read)", seq_gaps);

    Ok(())
}

#[cfg(not(feature = "camera-v4l"))]
fn main() {
    println!("Enable feature: camera-v4l");
}
