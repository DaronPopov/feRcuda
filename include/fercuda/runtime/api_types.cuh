#pragma once

#include <array>
#include <cstdint>

#include "fercuda/runtime/regime_manager.cuh"

namespace fer::runtime {

enum class BufferDType : uint8_t {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    I8 = 3,
    U8 = 4,
    I16 = 5,
    U16 = 6,
    I32 = 7,
    U32 = 8,
    I64 = 9,
    U64 = 10,
    F64 = 11,
};

struct BufferDesc {
    BufferDType dtype = BufferDType::F32;
    uint8_t rank = 0;
    std::array<uint32_t, 4> dims{};
    bool immutable = false;
    uint32_t tag = 0;
};

struct MatmulRequest {
    uint64_t a = 0;
    uint64_t b = 0;
    uint64_t out = 0;
    uint32_t memory_regime = MEMORY_REGIME_AUTO;
};

struct LayerNormRequest {
    uint64_t x = 0;
    uint64_t out = 0;
    float eps = 1e-6f;
    uint32_t memory_regime = MEMORY_REGIME_AUTO;
};

} // namespace fer::runtime
