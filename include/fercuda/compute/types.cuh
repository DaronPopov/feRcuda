#pragma once
/*
 * feRcuda :: types.cuh
 *
 * Core abstract types.
 *
 *  DFloat<RM>   — float32 wrapper that enforces a fixed IEEE 754 rounding mode
 *                 on every arithmetic op via PTX intrinsics. No compiler
 *                 can silently fuse or reorder these.
 *
 *  Tensor<T,N>  — N-dimensional typed view. Owns no memory; just a (ptr, shape)
 *                 pair usable on both host and device.
 *
 *  Shape<N>     — compile-time or runtime N-dim extent.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <cassert>

namespace fer {

// ─── Rounding Modes ────────────────────────────────────────────────────────────

enum class Round : uint8_t {
    NEAREST  = 0,   // round-to-nearest-even  (.rn)  — default IEEE
    ZERO     = 1,   // round-toward-zero       (.rz)
    POS_INF  = 2,   // round-toward-+inf       (.rp)
    NEG_INF  = 3,   // round-toward--inf       (.rm)
};

// ─── DFloat<RM> ───────────────────────────────────────────────────────────────
//
// Every binary op calls the matching PTX intrinsic so rounding is locked
// regardless of compiler flags. Determinism is structural, not flag-dependent.
//
template<Round RM = Round::NEAREST>
struct DFloat {
    float v;

    __host__ __device__ DFloat() : v(0.f) {}
    __host__ __device__ explicit DFloat(float x) : v(x) {}

    // Implicit conversion out (read-only)
    __host__ __device__ operator float() const { return v; }

    // ── arithmetic ──────────────────────────────────────────────────────────
    __device__ DFloat operator+(DFloat o) const { return DFloat(_add(v, o.v)); }
    __device__ DFloat operator-(DFloat o) const { return DFloat(_sub(v, o.v)); }
    __device__ DFloat operator*(DFloat o) const { return DFloat(_mul(v, o.v)); }
    __device__ DFloat operator/(DFloat o) const { return DFloat(_div(v, o.v)); }

    __device__ DFloat& operator+=(DFloat o) { v = _add(v, o.v); return *this; }
    __device__ DFloat& operator-=(DFloat o) { v = _sub(v, o.v); return *this; }
    __device__ DFloat& operator*=(DFloat o) { v = _mul(v, o.v); return *this; }

    // Fused multiply-add with locked rounding: out = a*b + c
    __device__ static DFloat fma(DFloat a, DFloat b, DFloat c) {
        return DFloat(_fma(a.v, b.v, c.v));
    }

    // ── comparisons ─────────────────────────────────────────────────────────
    __host__ __device__ bool operator==(DFloat o) const { return v == o.v; }
    __host__ __device__ bool operator< (DFloat o) const { return v <  o.v; }
    __host__ __device__ bool operator> (DFloat o) const { return v >  o.v; }
    __host__ __device__ bool operator<=(DFloat o) const { return v <= o.v; }
    __host__ __device__ bool operator>=(DFloat o) const { return v >= o.v; }

private:
    // PTX intrinsic dispatch — resolved at compile time via if-constexpr
    __device__ static float _add(float a, float b) {
        if constexpr (RM == Round::NEAREST) return __fadd_rn(a, b);
        if constexpr (RM == Round::ZERO)    return __fadd_rz(a, b);
        if constexpr (RM == Round::POS_INF) return __fadd_ru(a, b);
        if constexpr (RM == Round::NEG_INF) return __fadd_rd(a, b);
    }
    __device__ static float _sub(float a, float b) {
        // sub via negation — keeps intrinsic parity
        if constexpr (RM == Round::NEAREST) return __fadd_rn(a, -b);
        if constexpr (RM == Round::ZERO)    return __fadd_rz(a, -b);
        if constexpr (RM == Round::POS_INF) return __fadd_ru(a, -b);
        if constexpr (RM == Round::NEG_INF) return __fadd_rd(a, -b);
    }
    __device__ static float _mul(float a, float b) {
        if constexpr (RM == Round::NEAREST) return __fmul_rn(a, b);
        if constexpr (RM == Round::ZERO)    return __fmul_rz(a, b);
        if constexpr (RM == Round::POS_INF) return __fmul_ru(a, b);
        if constexpr (RM == Round::NEG_INF) return __fmul_rd(a, b);
    }
    __device__ static float _div(float a, float b) {
        // IEEE division — no fast reciprocal
        if constexpr (RM == Round::NEAREST) return __fdiv_rn(a, b);
        if constexpr (RM == Round::ZERO)    return __fdiv_rz(a, b);
        if constexpr (RM == Round::POS_INF) return __fdiv_ru(a, b);
        if constexpr (RM == Round::NEG_INF) return __fdiv_rd(a, b);
    }
    __device__ static float _fma(float a, float b, float c) {
        if constexpr (RM == Round::NEAREST) return __fmaf_rn(a, b, c);
        if constexpr (RM == Round::ZERO)    return __fmaf_rz(a, b, c);
        if constexpr (RM == Round::POS_INF) return __fmaf_ru(a, b, c);
        if constexpr (RM == Round::NEG_INF) return __fmaf_rd(a, b, c);
    }
};

// Convenience aliases
using F32  = DFloat<Round::NEAREST>;  // canonical deterministic float
using F32z = DFloat<Round::ZERO>;
using F32p = DFloat<Round::POS_INF>;
using F32m = DFloat<Round::NEG_INF>;

// ─── Shape<N> ─────────────────────────────────────────────────────────────────

template<int N>
struct Shape {
    uint32_t dims[N];

    __host__ __device__ Shape() { for (int i = 0; i < N; i++) dims[i] = 0; }

    template<typename... Args>
    __host__ __device__ Shape(Args... args) : dims{static_cast<uint32_t>(args)...} {
        static_assert(sizeof...(args) == N, "Shape dimension mismatch");
    }

    __host__ __device__ uint32_t operator[](int i) const { return dims[i]; }
    __host__ __device__ uint32_t& operator[](int i)     { return dims[i]; }

    __host__ __device__ size_t numel() const {
        size_t n = 1;
        for (int i = 0; i < N; i++) n *= dims[i];
        return n;
    }

    __host__ __device__ bool operator==(const Shape& o) const {
        for (int i = 0; i < N; i++) if (dims[i] != o.dims[i]) return false;
        return true;
    }
};

// ─── Tensor<T, N> ─────────────────────────────────────────────────────────────
//
// Non-owning typed view. Safe to copy into kernels by value.
//
template<typename T, int N>
struct Tensor {
    T*      data;
    Shape<N> shape;

    __host__ __device__ Tensor() : data(nullptr) {}
    __host__ __device__ Tensor(T* ptr, Shape<N> sh) : data(ptr), shape(sh) {}

    __host__ __device__ size_t numel() const { return shape.numel(); }
    __host__ __device__ size_t nbytes() const { return numel() * sizeof(T); }

    // 1-D element access
    __host__ __device__ T& operator[](size_t i)       { return data[i]; }
    __host__ __device__ T  operator[](size_t i) const { return data[i]; }

    // 2-D element access (row-major)
    __host__ __device__ T& at(uint32_t r, uint32_t c) {
        static_assert(N == 2, "at(r,c) requires 2-D tensor");
        return data[r * shape[1] + c];
    }
    __host__ __device__ T at(uint32_t r, uint32_t c) const {
        static_assert(N == 2, "at(r,c) requires 2-D tensor");
        return data[r * shape[1] + c];
    }

    // Row slice → 1-D Tensor (no alloc)
    __host__ __device__ Tensor<T, 1> row(uint32_t r) const {
        static_assert(N == 2, "row() requires 2-D tensor");
        return Tensor<T, 1>(data + r * shape[1], Shape<1>(shape[1]));
    }

    __host__ __device__ bool valid() const { return data != nullptr; }
};

// Shorthand aliases
template<typename T> using Tensor1D = Tensor<T, 1>;
template<typename T> using Tensor2D = Tensor<T, 2>;
template<typename T> using Tensor3D = Tensor<T, 3>;

// Common concrete types
using FTensor1D = Tensor1D<F32>;
using FTensor2D = Tensor2D<F32>;

} // namespace fer
