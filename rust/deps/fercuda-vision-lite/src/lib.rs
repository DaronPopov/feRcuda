use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[cfg(feature = "cv")]
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    #[error("invalid frame shape: width={width} height={height} bytes={bytes}")]
    InvalidFrameShape { width: u32, height: u32, bytes: usize },
    #[error("parse error: {0}")]
    Parse(String),
    #[cfg(feature = "sensor-serial")]
    #[error("serial error: {0}")]
    Serial(#[from] serialport::Error),
    #[cfg(feature = "display")]
    #[error("display error: {0}")]
    Display(#[from] minifb::Error),
}

#[derive(Debug, Clone)]
pub struct SensorSample {
    pub ts_millis: u64,
    pub values: Vec<f32>,
}

pub trait SensorSource {
    fn poll(&mut self) -> Result<SensorSample, VisionError>;
}

pub fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(feature = "sensor-serial")]
pub mod sensor {
    use super::{now_millis, SensorSample, SensorSource, VisionError};
    use std::io::Read;
    use std::time::Duration;

    pub struct SerialLineSensor {
        port: Box<dyn serialport::SerialPort>,
        buf: Vec<u8>,
    }

    impl SerialLineSensor {
        pub fn open(path: &str, baud: u32) -> Result<Self, VisionError> {
            let port = serialport::new(path, baud)
                .timeout(Duration::from_millis(50))
                .open()?;
            Ok(Self {
                port,
                buf: vec![0u8; 1024],
            })
        }
    }

    impl SensorSource for SerialLineSensor {
        fn poll(&mut self) -> Result<SensorSample, VisionError> {
            let n = self.port.read(&mut self.buf)?;
            let line = std::str::from_utf8(&self.buf[..n]).map_err(|e| VisionError::Parse(e.to_string()))?;
            let values = line
                .trim()
                .split(',')
                .filter(|s| !s.is_empty())
                .map(|s| s.trim().parse::<f32>().map_err(|e| VisionError::Parse(e.to_string())))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(SensorSample {
                ts_millis: now_millis(),
                values,
            })
        }
    }
}

#[cfg(feature = "cv")]
pub mod cv {
    use super::VisionError;
    use image::{imageops, GrayImage, Luma};
    use imageproc::edges;

    #[derive(Debug, Clone)]
    pub struct FrameGray {
        pub width: u32,
        pub height: u32,
        pub data: Vec<u8>,
    }

    impl FrameGray {
        pub fn from_luma8(width: u32, height: u32, data: Vec<u8>) -> Result<Self, VisionError> {
            let expected = (width as usize) * (height as usize);
            if data.len() != expected {
                return Err(VisionError::InvalidFrameShape {
                    width,
                    height,
                    bytes: data.len(),
                });
            }
            Ok(Self { width, height, data })
        }

        pub fn to_image(&self) -> Result<GrayImage, VisionError> {
            GrayImage::from_raw(self.width, self.height, self.data.clone()).ok_or(
                VisionError::InvalidFrameShape {
                    width: self.width,
                    height: self.height,
                    bytes: self.data.len(),
                },
            )
        }

        pub fn threshold_binary(&self, thr: u8) -> Result<Self, VisionError> {
            let mut img = self.to_image()?;
            for p in img.pixels_mut() {
                p.0[0] = if p.0[0] >= thr { 255 } else { 0 };
            }
            Self::from_luma8(self.width, self.height, img.into_raw())
        }

        pub fn canny_edges(&self, low: f32, high: f32) -> Result<Self, VisionError> {
            let img = self.to_image()?;
            let out = edges::canny(&img, low, high);
            Self::from_luma8(self.width, self.height, out.into_raw())
        }

        pub fn resize_nearest(&self, new_w: u32, new_h: u32) -> Result<Self, VisionError> {
            let img = self.to_image()?;
            let resized = imageops::resize(&img, new_w, new_h, imageops::FilterType::Nearest);
            Self::from_luma8(new_w, new_h, resized.into_raw())
        }

        pub fn mean_intensity(&self) -> f32 {
            let sum: u64 = self.data.iter().map(|v| *v as u64).sum();
            (sum as f32) / (self.data.len() as f32)
        }

        pub fn constant(width: u32, height: u32, v: u8) -> Self {
            let mut img = GrayImage::new(width, height);
            for y in 0..height {
                for x in 0..width {
                    img.put_pixel(x, y, Luma([v]));
                }
            }
            Self {
                width,
                height,
                data: img.into_raw(),
            }
        }
    }
}

#[cfg(feature = "camera-v4l")]
pub mod camera {
    #[cfg(feature = "cv")]
    use image::ImageReader;
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;
    use super::VisionError;
    use std::path::Path;
    use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
    use std::thread;
    use std::time::Duration;
    use v4l::format::FourCC;
    use v4l::io::traits::CaptureStream;
    use v4l::buffer::Type;
    use v4l::io::mmap::Stream;
    use v4l::prelude::*;
    use v4l::video::capture::Parameters as CaptureParameters;
    use v4l::video::Capture;

    #[derive(Debug, Clone)]
    pub struct CameraInfo {
        pub index: usize,
        pub path: String,
        pub name: String,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum CaptureEncoding {
        Yuyv,
        Gray,
        Mjpg,
        Unknown,
    }

    pub fn discover_cameras(max_devices: usize) -> Vec<CameraInfo> {
        let mut out = Vec::new();
        for i in 0..max_devices {
            if let Ok(dev) = Device::new(i) {
                let name = dev
                    .query_caps()
                    .map(|c| c.card)
                    .unwrap_or_else(|_| "unknown".to_string());
                out.push(CameraInfo {
                    index: i,
                    path: format!("/dev/video{i}"),
                    name,
                });
            }
        }
        out
    }

    #[derive(Debug, Clone)]
    pub struct RawFramePacket {
        pub width: u32,
        pub height: u32,
        pub encoding: CaptureEncoding,
        pub data: Vec<u8>,
        pub sequence: u64,
    }

    #[derive(Debug, Clone)]
    pub struct PollerConfig {
        pub device_index: usize,
        pub width: u32,
        pub height: u32,
        pub fourcc: FourCC,
        pub target_fps: u32,
        pub queue_depth: usize,
        pub poll_timeout_ms: i32,
    }

    impl Default for PollerConfig {
        fn default() -> Self {
            Self {
                device_index: 0,
                width: 1280,
                height: 720,
                fourcc: FourCC::new(b"YUYV"),
                target_fps: 120,
                queue_depth: 3,
                poll_timeout_ms: 2,
            }
        }
    }

    fn try_send_latest(tx: &SyncSender<RawFramePacket>, pkt: RawFramePacket) -> bool {
        match tx.try_send(pkt) {
            Ok(_) => true,
            Err(TrySendError::Full(_)) => true,
            Err(TrySendError::Disconnected(_)) => false,
        }
    }

    pub fn spawn_highspeed_poller(
        cfg: PollerConfig,
    ) -> Result<(Receiver<RawFramePacket>, thread::JoinHandle<()>), VisionError> {
        let (tx, rx) = sync_channel::<RawFramePacket>(cfg.queue_depth.max(1));
        let cam_cfg = cfg.clone();
        let handle = thread::spawn(move || {
            let (dev, fmt) = match open_device_with_format(
                cam_cfg.device_index,
                cam_cfg.width,
                cam_cfg.height,
                cam_cfg.fourcc,
            ) {
                Ok(v) => v,
                Err(_) => return,
            };

            let _ = dev.set_params(&CaptureParameters::with_fps(cam_cfg.target_fps));
            let enc = encoding_from_fourcc(fmt.fourcc);

            let mut stream = match Stream::with_buffers(&dev, Type::VideoCapture, 6) {
                Ok(s) => s,
                Err(_) => return,
            };
            stream.set_timeout(Duration::from_millis(cam_cfg.poll_timeout_ms.max(1) as u64));
            let h = stream.handle();

            let mut seq: u64 = 0;
            loop {
                match h.poll(libc::POLLIN, cam_cfg.poll_timeout_ms) {
                    Ok(0) => continue,
                    Ok(_) => {}
                    Err(_) => {
                        thread::sleep(Duration::from_millis(1));
                        continue;
                    }
                }

                let mut latest = match CaptureStream::next(&mut stream) {
                    Ok((bytes, _meta)) => bytes.to_vec(),
                    Err(_) => continue,
                };

                // Drain immediately available frames so consumer always gets freshest frame.
                while let Ok(ready) = h.poll(libc::POLLIN, 0) {
                    if ready == 0 {
                        break;
                    }
                    if let Ok((bytes, _meta)) = CaptureStream::next(&mut stream) {
                        latest = bytes.to_vec();
                    } else {
                        break;
                    }
                }

                seq = seq.wrapping_add(1);
                let pkt = RawFramePacket {
                    width: fmt.width,
                    height: fmt.height,
                    encoding: enc,
                    data: latest,
                    sequence: seq,
                };
                if !try_send_latest(&tx, pkt) {
                    break;
                }
            }
        });
        Ok((rx, handle))
    }

    pub fn encoding_from_fourcc(fourcc: FourCC) -> CaptureEncoding {
        match fourcc.str().unwrap_or("????") {
            "YUYV" => CaptureEncoding::Yuyv,
            "GREY" => CaptureEncoding::Gray,
            "MJPG" => CaptureEncoding::Mjpg,
            _ => CaptureEncoding::Unknown,
        }
    }

    pub fn open_device_with_format(
        index: usize,
        req_w: u32,
        req_h: u32,
        req_fourcc: FourCC,
    ) -> Result<(Device, v4l::Format), VisionError> {
        let dev = Device::new(index)?;
        let mut fmt = dev.format()?;
        fmt.width = req_w;
        fmt.height = req_h;
        fmt.fourcc = req_fourcc;
        let fmt = dev.set_format(&fmt)?;
        Ok((dev, fmt))
    }

    pub fn gray_to_argb32(width: u32, height: u32, src: &[u8], dst: &mut [u32]) -> Result<(), VisionError> {
        let n = (width as usize) * (height as usize);
        if src.len() < n || dst.len() < n {
            return Err(VisionError::InvalidFrameShape {
                width,
                height,
                bytes: src.len(),
            });
        }
        #[cfg(feature = "parallel")]
        {
            dst.par_iter_mut().zip(src.par_iter()).for_each(|(d, y)| {
                let yy = *y as u32;
                *d = 0xFF00_0000 | (yy << 16) | (yy << 8) | yy;
            });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..n {
                let yy = src[i] as u32;
                dst[i] = 0xFF00_0000 | (yy << 16) | (yy << 8) | yy;
            }
        }
        Ok(())
    }

    pub fn yuyv_to_argb32(width: u32, height: u32, src: &[u8], dst: &mut [u32]) -> Result<(), VisionError> {
        let n = (width as usize) * (height as usize);
        let need = n * 2;
        if src.len() < need || dst.len() < n {
            return Err(VisionError::InvalidFrameShape {
                width,
                height,
                bytes: src.len(),
            });
        }

        #[inline]
        fn clamp_u8(v: i32) -> u8 {
            if v < 0 {
                0
            } else if v > 255 {
                255
            } else {
                v as u8
            }
        }

        #[inline]
        fn yuv_to_rgb(y: i32, u: i32, v: i32) -> (u8, u8, u8) {
            let c = y - 16;
            let d = u - 128;
            let e = v - 128;
            let r = (298 * c + 409 * e + 128) >> 8;
            let g = (298 * c - 100 * d - 208 * e + 128) >> 8;
            let b = (298 * c + 516 * d + 128) >> 8;
            (clamp_u8(r), clamp_u8(g), clamp_u8(b))
        }

        #[cfg(feature = "parallel")]
        {
            dst.par_chunks_mut(2)
                .zip(src.par_chunks_exact(4))
                .for_each(|(out2, yuv)| {
                    let y0 = yuv[0] as i32;
                    let u = yuv[1] as i32;
                    let y1 = yuv[2] as i32;
                    let v = yuv[3] as i32;
                    let (r0, g0, b0) = yuv_to_rgb(y0, u, v);
                    let (r1, g1, b1) = yuv_to_rgb(y1, u, v);
                    out2[0] = 0xFF00_0000 | ((r0 as u32) << 16) | ((g0 as u32) << 8) | (b0 as u32);
                    if out2.len() > 1 {
                        out2[1] = 0xFF00_0000 | ((r1 as u32) << 16) | ((g1 as u32) << 8) | (b1 as u32);
                    }
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..(n / 2) {
                let off = i * 4;
                let y0 = src[off] as i32;
                let u = src[off + 1] as i32;
                let y1 = src[off + 2] as i32;
                let v = src[off + 3] as i32;
                let (r0, g0, b0) = yuv_to_rgb(y0, u, v);
                let (r1, g1, b1) = yuv_to_rgb(y1, u, v);
                dst[i * 2] = 0xFF00_0000 | ((r0 as u32) << 16) | ((g0 as u32) << 8) | (b0 as u32);
                dst[i * 2 + 1] = 0xFF00_0000 | ((r1 as u32) << 16) | ((g1 as u32) << 8) | (b1 as u32);
            }
        }
        Ok(())
    }

    #[cfg(feature = "cv")]
    pub fn mjpeg_to_argb32(_width: u32, _height: u32, src: &[u8], dst: &mut [u32]) -> Result<(u32, u32), VisionError> {
        let img = ImageReader::new(std::io::Cursor::new(src))
            .with_guessed_format()?
            .decode()?
            .to_rgb8();
        let (w, h) = img.dimensions();
        let n = (w as usize) * (h as usize);
        if dst.len() < n {
            return Err(VisionError::InvalidFrameShape {
                width: w,
                height: h,
                bytes: src.len(),
            });
        }
        let raw = img.into_raw();
        #[cfg(feature = "parallel")]
        {
            dst[..n]
                .par_iter_mut()
                .zip(raw.par_chunks_exact(3))
                .for_each(|(d, rgb)| {
                    *d = 0xFF00_0000 | ((rgb[0] as u32) << 16) | ((rgb[1] as u32) << 8) | (rgb[2] as u32);
                });
        }
        #[cfg(not(feature = "parallel"))]
        {
            for (i, rgb) in raw.chunks_exact(3).enumerate() {
                dst[i] = 0xFF00_0000 | ((rgb[0] as u32) << 16) | ((rgb[1] as u32) << 8) | (rgb[2] as u32);
            }
        }
        Ok((w, h))
    }

    pub fn is_video_device(index: usize) -> bool {
        Path::new(&format!("/dev/video{index}")).exists()
    }

    #[cfg(feature = "cv")]
    #[allow(dead_code)]
    pub fn capture_one_gray(dev_index: usize) -> Result<super::cv::FrameGray, VisionError> {
        use super::cv::FrameGray;
        let dev = Device::new(dev_index)?;
        let fmt = dev.format()?;
        let mut stream = Stream::with_buffers(&dev, Type::VideoCapture, 2)?;
        let (bytes, _meta) = stream.next()?;
        let raw = bytes.to_vec();

        // Best-effort: treat incoming bytes as grayscale if size matches.
        let expected = (fmt.width as usize) * (fmt.height as usize);
        let data = if raw.len() >= expected {
            raw[..expected].to_vec()
        } else {
            return Err(VisionError::InvalidFrameShape {
                width: fmt.width,
                height: fmt.height,
                bytes: raw.len(),
            });
        };
        FrameGray::from_luma8(fmt.width, fmt.height, data)
    }
}
