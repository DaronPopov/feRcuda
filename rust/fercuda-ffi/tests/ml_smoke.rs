use fercuda_ffi::ml;

#[test]
fn gaussian_noise_is_deterministic() {
    let a = ml::gaussian_noise_f32(16, 0.0, 1.0, 7).expect("noise a");
    let b = ml::gaussian_noise_f32(16, 0.0, 1.0, 7).expect("noise b");
    assert_eq!(a, b);
    assert!(a.iter().all(|v| v.is_finite()));
}

#[test]
fn safetensors_api_is_available() {
    let empty: &[u8] = &[];
    let parsed = ml::safetensors::SafeTensors::deserialize(empty);
    assert!(parsed.is_err());
}
