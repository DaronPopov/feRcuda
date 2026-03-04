use fercuda_ffi::math;

#[test]
fn reexported_math_aliases_work() {
    let v: math::Vec3f = math::nalgebra::SVector::<f32, 3>::new(1.0, 2.0, 3.0);
    let m: math::Mat3f = math::nalgebra::SMatrix::<f32, 3, 3>::identity();
    let y = m * v;
    assert_eq!(y[0], 1.0);
    assert_eq!(y[1], 2.0);
    assert_eq!(y[2], 3.0);
}

#[test]
fn kalman_step_runs_and_returns_finite_state() {
    let x = math::nalgebra::SVector::<f32, 2>::new(0.0, 0.0);
    let p = math::nalgebra::SMatrix::<f32, 2, 2>::identity();
    let dt = 0.01f32;
    let f = math::nalgebra::SMatrix::<f32, 2, 2>::new(1.0, dt, 0.0, 1.0);
    let b = math::nalgebra::SVector::<f32, 2>::new(0.5 * dt * dt, dt);
    let q = math::nalgebra::SMatrix::<f32, 2, 2>::new(1e-4, 0.0, 0.0, 1e-3);
    let z = 0.2f32;
    let r = 0.05f32;

    let (x_next, p_next) = math::kalman_step_2x1(x, p, f, b, 0.1, q, z, r);

    assert!(x_next[0].is_finite());
    assert!(x_next[1].is_finite());
    assert!(p_next[(0, 0)].is_finite());
    assert!(p_next[(1, 1)].is_finite());
    assert!(p_next[(0, 0)] >= 0.0);
    assert!(p_next[(1, 1)] >= 0.0);
}
