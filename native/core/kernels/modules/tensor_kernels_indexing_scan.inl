// F32 Gather (index along middle axis)
__global__ void k_gather_f32(const float* in, const int32_t* indices, float* out,
                             size_t outer, size_t input_dim_size, size_t idx_dim_size, size_t inner) {
    size_t total = outer * idx_dim_size * inner;
    for (unsigned int elem = blockIdx.x * blockDim.x + threadIdx.x; elem < total; elem += blockDim.x * gridDim.x) {
        size_t o = elem / (idx_dim_size * inner);
        size_t k = (elem / inner) % idx_dim_size;
        size_t i = elem % inner;

        int32_t idx = indices[o * idx_dim_size * inner + k * inner + i];
        if (idx < 0) idx += (int32_t)input_dim_size;
        if (idx < 0) idx = 0;
        if (idx >= (int32_t)input_dim_size) idx = (int32_t)input_dim_size - 1;

        out[elem] = in[o * input_dim_size * inner + (size_t)idx * inner + i];
    }
}

extern "C" void ptx_tensor_gather_f32(float* in, int32_t* indices, float* out,
                                       size_t outer, size_t input_dim_size, size_t idx_dim_size, size_t inner,
                                       cudaStream_t stream) {
    size_t total = outer * idx_dim_size * inner;
    if (total == 0) return;
    k_gather_f32<<<PTX_GRID_SIZE(total), PTX_BLOCK_SIZE, 0, stream>>>(in, indices, out, outer, input_dim_size, idx_dim_size, inner);
}

// F32 CumSum (inclusive prefix sum along middle axis)
// Tensor is [outer, dim_size, inner]; each thread handles one (outer, inner) lane.
__global__ void k_cumsum_f32(const float* in, float* out, size_t outer, size_t dim_size, size_t inner) {
    size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_lanes = outer * inner;
    if (lane >= total_lanes) return;

    size_t o = lane / inner;
    size_t i = lane % inner;
    size_t base = o * dim_size * inner + i;

    float acc = 0.0f;
    for (size_t k = 0; k < dim_size; k++) {
        size_t idx = base + k * inner;
        acc += in[idx];
        out[idx] = acc;
    }
}

extern "C" void ptx_tensor_cumsum_f32(float* in, float* out, size_t outer, size_t dim_size, size_t inner, cudaStream_t stream) {
    size_t total_lanes = outer * inner;
    if (total_lanes == 0) return;
    k_cumsum_f32<<<PTX_GRID_SIZE(total_lanes), PTX_BLOCK_SIZE, 0, stream>>>(in, out, outer, dim_size, inner);
}

// Cumulative product
__global__ void k_cumprod_f32(const float* in, float* out, size_t outer, size_t dim_size, size_t inner) {
    size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_lanes = outer * inner;
    if (lane >= total_lanes) return;
    size_t o = lane / inner;
    size_t i = lane % inner;
    size_t base = o * dim_size * inner + i;
    float acc = 1.0f;
    for (size_t k = 0; k < dim_size; k++) {
        size_t idx = base + k * inner;
        acc *= in[idx];
        out[idx] = acc;
    }
}

extern "C" void ptx_tensor_cumprod_f32(float* in, float* out, size_t outer, size_t dim_size, size_t inner, cudaStream_t stream) {
    size_t total_lanes = outer * inner;
    if (total_lanes == 0) return;
    k_cumprod_f32<<<PTX_GRID_SIZE(total_lanes), PTX_BLOCK_SIZE, 0, stream>>>(in, out, outer, dim_size, inner);
}

// Cumulative max
__global__ void k_cummax_f32(const float* in, float* out, size_t outer, size_t dim_size, size_t inner) {
    size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_lanes = outer * inner;
    if (lane >= total_lanes) return;
    size_t o = lane / inner;
    size_t i = lane % inner;
    size_t base = o * dim_size * inner + i;
    float acc = -INFINITY;
    for (size_t k = 0; k < dim_size; k++) {
        size_t idx = base + k * inner;
        if (in[idx] > acc) acc = in[idx];
        out[idx] = acc;
    }
}

extern "C" void ptx_tensor_cummax_f32(float* in, float* out, size_t outer, size_t dim_size, size_t inner, cudaStream_t stream) {
    size_t total_lanes = outer * inner;
    if (total_lanes == 0) return;
    k_cummax_f32<<<PTX_GRID_SIZE(total_lanes), PTX_BLOCK_SIZE, 0, stream>>>(in, out, outer, dim_size, inner);
}

// Cumulative min
__global__ void k_cummin_f32(const float* in, float* out, size_t outer, size_t dim_size, size_t inner) {
    size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_lanes = outer * inner;
    if (lane >= total_lanes) return;
    size_t o = lane / inner;
    size_t i = lane % inner;
    size_t base = o * dim_size * inner + i;
    float acc = INFINITY;
    for (size_t k = 0; k < dim_size; k++) {
        size_t idx = base + k * inner;
        if (in[idx] < acc) acc = in[idx];
        out[idx] = acc;
    }
}

extern "C" void ptx_tensor_cummin_f32(float* in, float* out, size_t outer, size_t dim_size, size_t inner, cudaStream_t stream) {
    size_t total_lanes = outer * inner;
    if (total_lanes == 0) return;
    k_cummin_f32<<<PTX_GRID_SIZE(total_lanes), PTX_BLOCK_SIZE, 0, stream>>>(in, out, outer, dim_size, inner);
}

// ============================================================================
// Cast kernels: f32 <-> i32
// ============================================================================

__global__ void k_cast_f32_to_i32(const float* __restrict__ in, int* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (int)in[idx]; }
}

__global__ void k_cast_i32_to_f32(const int* __restrict__ in, float* __restrict__ out, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { out[idx] = (float)in[idx]; }
}

extern "C" void ptx_tensor_cast_f32_to_i32(float* in, int* out, size_t n, cudaStream_t stream) {
    k_cast_f32_to_i32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_cast_i32_to_f32(int* in, float* out, size_t n, cudaStream_t stream) {
    k_cast_i32_to_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

// ============================================================================
// Random number generation (xorshift128+ PRNG)
// ============================================================================

__global__ void k_rand_f32(float* __restrict__ out, size_t n, unsigned long long seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Per-thread state from seed + idx
        unsigned long long s0 = seed ^ (idx * 6364136223846793005ULL + 1442695040888963407ULL);
        unsigned long long s1 = s0 * 6364136223846793005ULL + 1442695040888963407ULL;
        // xorshift128+
        s1 ^= s0;
        s0 = (s0 << 55 | s0 >> 9) ^ s1 ^ (s1 << 14);
        s1 = s1 << 36 | s1 >> 28;
        unsigned long long result = s0 + s1;
        // Convert to float in [0, 1)
        out[idx] = (float)(result >> 40) / 16777216.0f;
    }
}

__global__ void k_randn_f32(float* __restrict__ out, size_t n, unsigned long long seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Generate two uniform values using different seeds
        unsigned long long s0 = seed ^ (idx * 6364136223846793005ULL + 1442695040888963407ULL);
        unsigned long long s1 = s0 * 6364136223846793005ULL + 1442695040888963407ULL;
        s1 ^= s0;
        s0 = (s0 << 55 | s0 >> 9) ^ s1 ^ (s1 << 14);
        s1 = s1 << 36 | s1 >> 28;
        float u1 = (float)((s0 + s1) >> 40) / 16777216.0f;
        // Second round
        s1 ^= s0;
        s0 = (s0 << 55 | s0 >> 9) ^ s1 ^ (s1 << 14);
        s1 = s1 << 36 | s1 >> 28;
        float u2 = (float)((s0 + s1) >> 40) / 16777216.0f;
        // Box-Muller transform
        u1 = fmaxf(u1, 1e-7f); // avoid log(0)
        out[idx] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    }
}

extern "C" void ptx_tensor_rand_f32(float* out, size_t n, unsigned long long seed, cudaStream_t stream) {
    k_rand_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(out, n, seed);
}

extern "C" void ptx_tensor_randn_f32(float* out, size_t n, unsigned long long seed, cudaStream_t stream) {
    k_randn_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(out, n, seed);
}

// F32 TopK selection along middle axis
// Tensor is [outer, dim_size, inner]; each thread handles one (outer, inner) lane,
// selecting k extreme values via repeated scan.
__global__ void k_topk_f32(const float* in, float* values_out, int32_t* indices_out,
                           size_t outer, size_t dim_size, size_t inner, size_t k, int largest) {
    size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_lanes = outer * inner;
    if (lane >= total_lanes) return;

    size_t o = lane / inner;
    size_t i = lane % inner;
    size_t in_base  = o * dim_size * inner + i;
    size_t out_base = o * k * inner + i;
    size_t actual_k = k < dim_size ? k : dim_size;

    for (size_t ki = 0; ki < actual_k; ki++) {
        float best_val;
        int32_t best_idx = -1;
        for (size_t d = 0; d < dim_size; d++) {
            float val = in[in_base + d * inner];
            bool skip = false;
            for (size_t prev = 0; prev < ki; prev++) {
                if (indices_out[out_base + prev * inner] == (int32_t)d) { skip = true; break; }
            }
            if (skip) continue;
            if (best_idx < 0) {
                best_val = val; best_idx = (int32_t)d;
            } else {
                bool better = largest ? (val > best_val) : (val < best_val);
                bool tie = (val == best_val && (int32_t)d < best_idx);
                if (better || tie) { best_val = val; best_idx = (int32_t)d; }
            }
        }
        values_out [out_base + ki * inner] = best_val;
        if (indices_out) indices_out[out_base + ki * inner] = best_idx;
    }
}

extern "C" void ptx_tensor_topk_f32(float* in, float* values_out, int32_t* indices_out,
                                     size_t outer, size_t dim_size, size_t inner,
                                     size_t k, int largest, cudaStream_t stream) {
    size_t total_lanes = outer * inner;
    if (total_lanes == 0 || k == 0) return;
    k_topk_f32<<<PTX_GRID_SIZE(total_lanes), PTX_BLOCK_SIZE, 0, stream>>>(
        in, values_out, indices_out, outer, dim_size, inner, k, largest);
}

// F32 Index-Select (select slices along a dimension using a 1D index array)
// Input shape: [left_size, src_dim_size, right_size]
// Output shape: [left_size, ids_dim_size, right_size]
// out[l][k][r] = input[l][ids[k]][r]
__global__ void k_index_select_f32(const float* input, const int32_t* ids, float* output,
                                    size_t left_size, size_t src_dim_size, size_t ids_dim_size, size_t right_size) {
    size_t total = left_size * ids_dim_size * right_size;
    for (unsigned int elem = blockIdx.x * blockDim.x + threadIdx.x; elem < total; elem += blockDim.x * gridDim.x) {
        size_t l = elem / (ids_dim_size * right_size);
        size_t k = (elem / right_size) % ids_dim_size;
        size_t r = elem % right_size;

        int32_t idx = ids[k];
        if (idx < 0) idx += (int32_t)src_dim_size;
        if (idx < 0) idx = 0;
        if (idx >= (int32_t)src_dim_size) idx = (int32_t)src_dim_size - 1;

        output[elem] = input[l * src_dim_size * right_size + (size_t)idx * right_size + r];
    }
}

extern "C" void ptx_tensor_index_select_f32(float* input, int32_t* ids, float* output,
                                              size_t left_size, size_t src_dim_size, size_t ids_dim_size,
                                              size_t right_size, cudaStream_t stream) {
    size_t total = left_size * ids_dim_size * right_size;
    if (total == 0) return;
    k_index_select_f32<<<PTX_GRID_SIZE(total), PTX_BLOCK_SIZE, 0, stream>>>(
        input, ids, output, left_size, src_dim_size, ids_dim_size, right_size);
}

// F32 Scatter-Add (atomically accumulate src values into output at index positions)
// src shape: [left_size, src_dim_size, right_size]
// output shape: [left_size, dst_dim_size, right_size] (must be zero-initialized)
// output[l][ids[k]][r] += src[l][k][r]
__global__ void k_scatter_add_f32(const int32_t* ids, const float* src, float* output,
                                   size_t left_size, size_t src_dim_size, size_t dst_dim_size, size_t right_size) {
    size_t total = left_size * src_dim_size * right_size;
    for (unsigned int elem = blockIdx.x * blockDim.x + threadIdx.x; elem < total; elem += blockDim.x * gridDim.x) {
        size_t l = elem / (src_dim_size * right_size);
        size_t k = (elem / right_size) % src_dim_size;
        size_t r = elem % right_size;

        int32_t idx = ids[k];
        if (idx < 0) idx += (int32_t)dst_dim_size;
        if (idx >= 0 && idx < (int32_t)dst_dim_size) {
            atomicAdd(&output[l * dst_dim_size * right_size + (size_t)idx * right_size + r], src[elem]);
        }
    }
}

extern "C" void ptx_tensor_scatter_add_f32(int32_t* ids, float* src, float* output,
                                             size_t left_size, size_t src_dim_size, size_t dst_dim_size,
                                             size_t right_size, cudaStream_t stream) {
    size_t total = left_size * src_dim_size * right_size;
    if (total == 0) return;
    k_scatter_add_f32<<<PTX_GRID_SIZE(total), PTX_BLOCK_SIZE, 0, stream>>>(
        ids, src, output, left_size, src_dim_size, dst_dim_size, right_size);
}

// F32 Argsort (bitonic sort network — one block per row, shared memory)
// Input: [nrows, ncols], Output: [nrows, ncols] of uint32 indices
// ncols_pad must be the next power of 2 >= ncols.
__global__ void k_argsort_asc_f32(const float* input, unsigned int* output,
                                   int ncols, int ncols_pad) {
    int row = blockIdx.x;
    const float* x = input + row * ncols;
    extern __shared__ int shmem[];

    for (int col = threadIdx.x; col < ncols_pad; col += blockDim.x) {
        shmem[col] = col;
    }
    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int col = threadIdx.x; col < ncols_pad; col += blockDim.x) {
                int ixj = col ^ j;
                if (ixj > col) {
                    if ((col & k) == 0) {
                        if (shmem[col] >= ncols ||
                            (shmem[ixj] < ncols && x[shmem[col]] > x[shmem[ixj]])) {
                            int tmp = shmem[col]; shmem[col] = shmem[ixj]; shmem[ixj] = tmp;
                        }
                    } else {
                        if (shmem[ixj] >= ncols ||
                            (shmem[col] < ncols && x[shmem[col]] < x[shmem[ixj]])) {
                            int tmp = shmem[col]; shmem[col] = shmem[ixj]; shmem[ixj] = tmp;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        output[row * ncols + col] = (unsigned int)shmem[col];
    }
}

__global__ void k_argsort_desc_f32(const float* input, unsigned int* output,
                                    int ncols, int ncols_pad) {
    int row = blockIdx.x;
    const float* x = input + row * ncols;
    extern __shared__ int shmem[];

    for (int col = threadIdx.x; col < ncols_pad; col += blockDim.x) {
        shmem[col] = col;
    }
    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int col = threadIdx.x; col < ncols_pad; col += blockDim.x) {
                int ixj = col ^ j;
                if (ixj > col) {
                    if ((col & k) == 0) {
                        if (shmem[col] >= ncols ||
                            (shmem[ixj] < ncols && x[shmem[col]] < x[shmem[ixj]])) {
                            int tmp = shmem[col]; shmem[col] = shmem[ixj]; shmem[ixj] = tmp;
                        }
                    } else {
                        if (shmem[ixj] >= ncols ||
                            (shmem[col] < ncols && x[shmem[col]] > x[shmem[ixj]])) {
                            int tmp = shmem[col]; shmem[col] = shmem[ixj]; shmem[ixj] = tmp;
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        output[row * ncols + col] = (unsigned int)shmem[col];
    }
}

extern "C" void ptx_tensor_argsort_f32(float* input, unsigned int* output,
                                         size_t nrows, size_t ncols, int ascending,
                                         cudaStream_t stream) {
    if (nrows == 0 || ncols == 0) return;
    // Round ncols up to next power of 2
    int ncols_pad = 1;
    while (ncols_pad < (int)ncols) ncols_pad *= 2;
    int block = (ncols_pad < PTX_BLOCK_SIZE) ? ncols_pad : PTX_BLOCK_SIZE;
    size_t smem = ncols_pad * sizeof(int);
    if (ascending) {
        k_argsort_asc_f32<<<(int)nrows, block, smem, stream>>>(input, output, (int)ncols, ncols_pad);
    } else {
        k_argsort_desc_f32<<<(int)nrows, block, smem, stream>>>(input, output, (int)ncols, ncols_pad);
    }
}

// F32 Softmax
extern "C" void ptx_tensor_softmax_f32(float* in, float* out, size_t batch, size_t dim, cudaStream_t stream) {
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE + 1) * sizeof(float);
    k_softmax_f32<<<batch, PTX_BLOCK_SIZE, smem, stream>>>(in, out, batch, dim);
}

extern "C" void ptx_tensor_log_softmax_f32(float* in, float* out, size_t batch, size_t dim, cudaStream_t stream) {
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE + 1) * sizeof(float);
    k_log_softmax_f32<<<batch, PTX_BLOCK_SIZE, smem, stream>>>(in, out, batch, dim);
}

// F64 API
extern "C" void ptx_tensor_add_f64(double* a, double* b, double* out, size_t n, cudaStream_t stream) {
    k_add_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_sub_f64(double* a, double* b, double* out, size_t n, cudaStream_t stream) {
    k_sub_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_mul_f64(double* a, double* b, double* out, size_t n, cudaStream_t stream) {
    k_mul_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_div_f64(double* a, double* b, double* out, size_t n, cudaStream_t stream) {
    k_div_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_neg_f64(double* in, double* out, size_t n, cudaStream_t stream) {
    k_neg_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_exp_f64(double* in, double* out, size_t n, cudaStream_t stream) {
    k_exp_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_log_f64(double* in, double* out, size_t n, cudaStream_t stream) {
    k_log_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_sqrt_f64(double* in, double* out, size_t n, cudaStream_t stream) {
    k_sqrt_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_tanh_f64(double* in, double* out, size_t n, cudaStream_t stream) {
    k_tanh_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_relu_f64(double* in, double* out, size_t n, cudaStream_t stream) {
    k_relu_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_gelu_f64(double* in, double* out, size_t n, cudaStream_t stream) {
    k_gelu_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_sigmoid_f64(double* in, double* out, size_t n, cudaStream_t stream) {
    k_sigmoid_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_reduce_sum_f64(double* in, double* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream) {
    size_t num_outputs = outer * inner;
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE) * sizeof(double);
    k_reduce_sum_f64<<<num_outputs, PTX_BLOCK_SIZE, smem, stream>>>(in, out, outer, reduce, inner);
}

extern "C" void ptx_tensor_reduce_max_f64(double* in, double* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream) {
    size_t num_outputs = outer * inner;
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE) * sizeof(double);
    k_reduce_max_f64<<<num_outputs, PTX_BLOCK_SIZE, smem, stream>>>(in, out, outer, reduce, inner);
}

extern "C" void ptx_tensor_reduce_min_f64(double* in, double* out, size_t outer, size_t reduce, size_t inner, cudaStream_t stream) {
    size_t num_outputs = outer * inner;
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE) * sizeof(double);
    k_reduce_min_f64<<<num_outputs, PTX_BLOCK_SIZE, smem, stream>>>(in, out, outer, reduce, inner);
}

extern "C" void ptx_tensor_softmax_f64(double* in, double* out, size_t batch, size_t dim, cudaStream_t stream) {
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE + 1) * sizeof(double);
    k_softmax_f64<<<batch, PTX_BLOCK_SIZE, smem, stream>>>(in, out, batch, dim);
}

extern "C" void ptx_tensor_affine_f64(double* in, double* out, size_t n, double mul, double add, cudaStream_t stream) {
    k_affine_f64<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n, mul, add);
}

// F16 API
extern "C" void ptx_tensor_add_f16(__half* a, __half* b, __half* out, size_t n, cudaStream_t stream) {
    k_add_f16<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_sub_f16(__half* a, __half* b, __half* out, size_t n, cudaStream_t stream) {
    k_sub_f16<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_mul_f16(__half* a, __half* b, __half* out, size_t n, cudaStream_t stream) {
    k_mul_f16<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_div_f16(__half* a, __half* b, __half* out, size_t n, cudaStream_t stream) {
    k_div_f16<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(a, b, out, n);
}

extern "C" void ptx_tensor_relu_f16(__half* in, __half* out, size_t n, cudaStream_t stream) {
    k_relu_f16<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_gelu_f16(__half* in, __half* out, size_t n, cudaStream_t stream) {
    k_gelu_f16<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_sigmoid_f16(__half* in, __half* out, size_t n, cudaStream_t stream) {
    k_sigmoid_f16<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_softmax_f16(__half* in, __half* out, size_t batch, size_t dim, cudaStream_t stream) {
    int smem = (PTX_BLOCK_SIZE / PTX_WARP_SIZE + 1) * sizeof(float);
    k_softmax_f16<<<batch, PTX_BLOCK_SIZE, smem, stream>>>(in, out, batch, dim);
}

extern "C" void ptx_tensor_cast_f32_to_f16(float* in, __half* out, size_t n, cudaStream_t stream) {
    k_cast_f32_to_f16<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

extern "C" void ptx_tensor_cast_f16_to_f32(__half* in, float* out, size_t n, cudaStream_t stream) {
    k_cast_f16_to_f32<<<PTX_GRID_SIZE(n), PTX_BLOCK_SIZE, 0, stream>>>(in, out, n);
}

