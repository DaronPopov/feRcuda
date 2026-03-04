use fercuda_ffi::{BufferDType, BufferDesc};

#[test]
fn dtype_elem_sizes_are_correct() {
    assert_eq!(BufferDType::F32.elem_size(), 4);
    assert_eq!(BufferDType::F16.elem_size(), 2);
    assert_eq!(BufferDType::BF16.elem_size(), 2);
    assert_eq!(BufferDType::I8.elem_size(), 1);
    assert_eq!(BufferDType::U8.elem_size(), 1);
    assert_eq!(BufferDType::I16.elem_size(), 2);
    assert_eq!(BufferDType::U16.elem_size(), 2);
    assert_eq!(BufferDType::I32.elem_size(), 4);
    assert_eq!(BufferDType::U32.elem_size(), 4);
    assert_eq!(BufferDType::I64.elem_size(), 8);
    assert_eq!(BufferDType::U64.elem_size(), 8);
    assert_eq!(BufferDType::F64.elem_size(), 8);
}

#[test]
fn buffer_desc_byte_len_matches_dtype_and_shape() {
    let d = BufferDesc::new(BufferDType::F16, 2, [4, 8, 0, 0], false, 0);
    assert_eq!(d.elem_count(), 32);
    assert_eq!(d.byte_len(), 64);

    let q = BufferDesc::new(BufferDType::I8, 1, [256, 0, 0, 0], true, 7);
    assert_eq!(q.elem_count(), 256);
    assert_eq!(q.byte_len(), 256);
}
