use fercuda_ffi::vision::cv::FrameGray;

#[test]
fn vision_cv_pipeline_smoke() {
    let frame = FrameGray::constant(32, 32, 64);
    let thr = frame.threshold_binary(32).expect("threshold");
    assert_eq!(thr.width, 32);
    assert_eq!(thr.height, 32);
    assert!(thr.mean_intensity() > 200.0);

    let edges = thr.canny_edges(10.0, 20.0).expect("canny");
    assert_eq!(edges.width, 32);
    assert_eq!(edges.height, 32);

    let small = edges.resize_nearest(16, 16).expect("resize");
    assert_eq!(small.width, 16);
    assert_eq!(small.height, 16);
}
