use fercuda_ffi::ml_adapter::{
    decode_bf16_le_bytes, decode_f16_le_bytes, decode_f32_le_bytes, dequantize_q4_affine_packed,
    dequantize_q4_symmetric_packed, dequantize_q8_affine_u8, dequantize_q8_symmetric_i8,
    AdapterError,
};

#[test]
fn decode_f32_le_bytes_roundtrip() {
    let bytes = [
        0x00, 0x00, 0x80, 0x3f, // 1.0
        0x00, 0x00, 0x00, 0x40, // 2.0
        0x00, 0x00, 0x40, 0x40, // 3.0
    ];
    let out = decode_f32_le_bytes(&bytes).expect("decode");
    assert_eq!(out, vec![1.0, 2.0, 3.0]);
}

#[test]
fn decode_f32_le_bytes_rejects_bad_len() {
    let err = decode_f32_le_bytes(&[1, 2, 3]).expect_err("must fail");
    match err {
        AdapterError::InvalidF32Bytes { len } => assert_eq!(len, 3),
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn decode_f16_le_bytes_roundtrip() {
    // f16(1.0)=0x3C00, f16(2.0)=0x4000
    let bytes = [0x00, 0x3c, 0x00, 0x40];
    let out = decode_f16_le_bytes(&bytes).expect("decode f16");
    assert_eq!(out.len(), 2);
    assert!((out[0] - 1.0).abs() < 1e-3);
    assert!((out[1] - 2.0).abs() < 1e-3);
}

#[test]
fn decode_bf16_le_bytes_roundtrip() {
    // bf16(1.0)=0x3F80, bf16(2.0)=0x4000
    let bytes = [0x80, 0x3f, 0x00, 0x40];
    let out = decode_bf16_le_bytes(&bytes).expect("decode bf16");
    assert_eq!(out, vec![1.0, 2.0]);
}

#[test]
fn dequant_q8_paths() {
    let sym = dequantize_q8_symmetric_i8(&[255, 0, 1], 0.5);
    assert_eq!(sym, vec![-0.5, 0.0, 0.5]);

    let aff = dequantize_q8_affine_u8(&[127, 128, 129], 0.25, 128);
    assert_eq!(aff, vec![-0.25, 0.0, 0.25]);
}

#[test]
fn dequant_q4_paths() {
    // low nibble then high nibble: 0xF1 -> [1, -1] (symmetric)
    let sym = dequantize_q4_symmetric_packed(&[0xF1], 2, 1.0).expect("q4 sym");
    assert_eq!(sym, vec![1.0, -1.0]);

    // affine with zp=8: 0x98 -> [0, 1]
    let aff = dequantize_q4_affine_packed(&[0x98], 2, 1.0, 8).expect("q4 affine");
    assert_eq!(aff, vec![0.0, 1.0]);
}
