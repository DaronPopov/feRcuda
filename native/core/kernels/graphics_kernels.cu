/*
 * Ferrite-OS Graphics Kernels — GPU Software Rasterization
 *
 * Pure CUDA compute kernels for real-time 3D rendering.
 * No OpenGL, Vulkan, or GPU driver graphics stack — just math on tensors.
 *
 * Pipeline: vertices → MVP transform → perspective divide → viewport →
 *           triangle rasterization (edge function + atomic depth) →
 *           per-pixel shading → packed RGBA for present()
 */

#include "gpu/graphics_ops.h"
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define GFX_BLOCK_SIZE 256
#define GFX_GRID_SIZE(n) (((n) + GFX_BLOCK_SIZE - 1) / GFX_BLOCK_SIZE)

/* Encode float depth [0,1] to uint32 for atomicMin depth testing. */
__device__ __forceinline__ uint32_t depth_to_uint(float d) {
    /* Clamp and scale to full uint32 range. */
    d = fminf(fmaxf(d, 0.0f), 1.0f);
    return __float_as_uint(d);
}

__device__ __forceinline__ float uint_to_depth(uint32_t u) {
    return __uint_as_float(u);
}

/* ========================================================================== */
/* Framebuffer clear kernels                                                  */
/* ========================================================================== */

__global__ void k_gfx_clear_color(float* __restrict__ color_buf,
                                   float r, float g, float b, float a,
                                   uint32_t num_pixels) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        uint32_t base = idx * 4;
        color_buf[base + 0] = r;
        color_buf[base + 1] = g;
        color_buf[base + 2] = b;
        color_buf[base + 3] = a;
    }
}

extern "C" void ptx_gfx_clear_color(float* color_buf,
                                      float r, float g, float b, float a,
                                      uint32_t num_pixels,
                                      cudaStream_t stream) {
    if (num_pixels == 0) return;
    k_gfx_clear_color<<<GFX_GRID_SIZE(num_pixels), GFX_BLOCK_SIZE, 0, stream>>>(
        color_buf, r, g, b, a, num_pixels);
}

__global__ void k_gfx_clear_depth(float* __restrict__ depth_buf,
                                   float value,
                                   uint32_t num_pixels) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        depth_buf[idx] = value;
    }
}

extern "C" void ptx_gfx_clear_depth(float* depth_buf, float value,
                                      uint32_t num_pixels,
                                      cudaStream_t stream) {
    if (num_pixels == 0) return;
    k_gfx_clear_depth<<<GFX_GRID_SIZE(num_pixels), GFX_BLOCK_SIZE, 0, stream>>>(
        depth_buf, value, num_pixels);
}

/* ========================================================================== */
/* Vertex transform: position × MVP → clip space                             */
/* ========================================================================== */

__global__ void k_gfx_vertex_transform(const float* __restrict__ pos_in,
                                        float* __restrict__ pos_out,
                                        const float* __restrict__ mvp,
                                        uint32_t num_vertices) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vertices) return;

    /* Load vertex position (x, y, z), implicitly w=1. */
    float x = pos_in[idx * 3 + 0];
    float y = pos_in[idx * 3 + 1];
    float z = pos_in[idx * 3 + 2];

    /* Column-major 4x4 matrix multiply: out = mvp * [x, y, z, 1]^T
     * mvp layout: col0=[0..3], col1=[4..7], col2=[8..11], col3=[12..15] */
    float ox = mvp[0]*x + mvp[4]*y + mvp[ 8]*z + mvp[12];
    float oy = mvp[1]*x + mvp[5]*y + mvp[ 9]*z + mvp[13];
    float oz = mvp[2]*x + mvp[6]*y + mvp[10]*z + mvp[14];
    float ow = mvp[3]*x + mvp[7]*y + mvp[11]*z + mvp[15];

    pos_out[idx * 4 + 0] = ox;
    pos_out[idx * 4 + 1] = oy;
    pos_out[idx * 4 + 2] = oz;
    pos_out[idx * 4 + 3] = ow;
}

extern "C" void ptx_gfx_vertex_transform(const float* positions_in,
                                           float* positions_out,
                                           const float* mvp,
                                           uint32_t num_vertices,
                                           cudaStream_t stream) {
    if (num_vertices == 0) return;
    k_gfx_vertex_transform<<<GFX_GRID_SIZE(num_vertices), GFX_BLOCK_SIZE, 0, stream>>>(
        positions_in, positions_out, mvp, num_vertices);
}

/* ========================================================================== */
/* Perspective divide + viewport transform                                    */
/* ========================================================================== */

__global__ void k_gfx_viewport_transform(const float* __restrict__ clip,
                                          float* __restrict__ screen_out,
                                          uint32_t vp_w, uint32_t vp_h,
                                          uint32_t num_vertices) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vertices) return;

    float cx = clip[idx * 4 + 0];
    float cy = clip[idx * 4 + 1];
    float cz = clip[idx * 4 + 2];
    float cw = clip[idx * 4 + 3];

    /* Perspective divide → NDC [-1, 1] */
    float inv_w = (fabsf(cw) > 1e-7f) ? (1.0f / cw) : 0.0f;
    float nx = cx * inv_w;
    float ny = cy * inv_w;
    float nz = cz * inv_w;

    /* NDC → screen coordinates.
     * X: [-1,1] → [0, vp_w]
     * Y: [-1,1] → [vp_h, 0]  (flip Y — screen Y goes down)
     * Z: [-1,1] → [0, 1]     (depth for depth buffer)
     */
    float sx = (nx * 0.5f + 0.5f) * (float)vp_w;
    float sy = (1.0f - (ny * 0.5f + 0.5f)) * (float)vp_h;
    float sz = nz * 0.5f + 0.5f;

    screen_out[idx * 3 + 0] = sx;
    screen_out[idx * 3 + 1] = sy;
    screen_out[idx * 3 + 2] = sz;
}

extern "C" void ptx_gfx_viewport_transform(const float* clip_positions,
                                             float* screen_out,
                                             uint32_t viewport_width,
                                             uint32_t viewport_height,
                                             uint32_t num_vertices,
                                             cudaStream_t stream) {
    if (num_vertices == 0) return;
    k_gfx_viewport_transform<<<GFX_GRID_SIZE(num_vertices), GFX_BLOCK_SIZE, 0, stream>>>(
        clip_positions, screen_out, viewport_width, viewport_height, num_vertices);
}

/* ========================================================================== */
/* Triangle rasterization                                                     */
/* ========================================================================== */

/*
 * One thread per triangle. For each triangle:
 *   1. Compute bounding box (clamped to framebuffer)
 *   2. For each pixel in the bbox, compute edge function (barycentric)
 *   3. If inside, interpolate depth & color
 *   4. atomicMin on integer depth buffer for depth test
 *   5. If won, write color
 *
 * This is a simple scanline rasterizer. For large triangles a tile-based
 * approach would be faster, but this is correct and portable.
 */

__device__ __forceinline__ float edge_function(float ax, float ay,
                                                float bx, float by,
                                                float px, float py) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

__global__ void k_gfx_rasterize_triangles(
    const float* __restrict__ screen_pos,  /* [V*3] (sx, sy, depth) */
    const float* __restrict__ vtx_colors,  /* [V*3] (r, g, b) or NULL */
    const uint32_t* __restrict__ indices,  /* [T*3] */
    float* __restrict__ color_buf,         /* [H*W*4] */
    uint32_t* __restrict__ depth_buf,      /* [H*W] */
    uint32_t fb_w, uint32_t fb_h,
    uint32_t num_triangles)
{
    uint32_t tri = blockIdx.x * blockDim.x + threadIdx.x;
    if (tri >= num_triangles) return;

    /* Load vertex indices */
    uint32_t i0 = indices[tri * 3 + 0];
    uint32_t i1 = indices[tri * 3 + 1];
    uint32_t i2 = indices[tri * 3 + 2];

    /* Load screen positions */
    float x0 = screen_pos[i0 * 3 + 0], y0 = screen_pos[i0 * 3 + 1], z0 = screen_pos[i0 * 3 + 2];
    float x1 = screen_pos[i1 * 3 + 0], y1 = screen_pos[i1 * 3 + 1], z1 = screen_pos[i1 * 3 + 2];
    float x2 = screen_pos[i2 * 3 + 0], y2 = screen_pos[i2 * 3 + 1], z2 = screen_pos[i2 * 3 + 2];

    /* Triangle area (2x) via edge function */
    float area = edge_function(x0, y0, x1, y1, x2, y2);
    if (fabsf(area) < 1e-6f) return;  /* degenerate triangle */
    float inv_area = 1.0f / area;

    /* Load vertex colors (default white if NULL) */
    float r0 = 1.0f, g0 = 1.0f, b0 = 1.0f;
    float r1 = 1.0f, g1 = 1.0f, b1 = 1.0f;
    float r2 = 1.0f, g2 = 1.0f, b2 = 1.0f;
    if (vtx_colors != NULL) {
        r0 = vtx_colors[i0*3+0]; g0 = vtx_colors[i0*3+1]; b0 = vtx_colors[i0*3+2];
        r1 = vtx_colors[i1*3+0]; g1 = vtx_colors[i1*3+1]; b1 = vtx_colors[i1*3+2];
        r2 = vtx_colors[i2*3+0]; g2 = vtx_colors[i2*3+1]; b2 = vtx_colors[i2*3+2];
    }

    /* Bounding box, clamped to framebuffer */
    int min_x = max(0, (int)floorf(fminf(fminf(x0, x1), x2)));
    int max_x = min((int)fb_w - 1, (int)ceilf(fmaxf(fmaxf(x0, x1), x2)));
    int min_y = max(0, (int)floorf(fminf(fminf(y0, y1), y2)));
    int max_y = min((int)fb_h - 1, (int)ceilf(fmaxf(fmaxf(y0, y1), y2)));

    /* Rasterize: scan the bounding box */
    for (int py = min_y; py <= max_y; py++) {
        for (int px = min_x; px <= max_x; px++) {
            float fpx = (float)px + 0.5f;
            float fpy = (float)py + 0.5f;

            /* Barycentric coordinates */
            float w0 = edge_function(x1, y1, x2, y2, fpx, fpy) * inv_area;
            float w1 = edge_function(x2, y2, x0, y0, fpx, fpy) * inv_area;
            float w2 = edge_function(x0, y0, x1, y1, fpx, fpy) * inv_area;

            /* Inside test (handle both CW and CCW winding) */
            bool inside;
            if (area > 0.0f) {
                inside = (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f);
            } else {
                inside = (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);
                w0 = -w0; w1 = -w1; w2 = -w2;
            }
            if (!inside) continue;

            /* Interpolate depth */
            float depth = w0 * z0 + w1 * z1 + w2 * z2;
            uint32_t idepth = depth_to_uint(depth);

            /* Atomic depth test: only write if closer */
            uint32_t pixel_idx = (uint32_t)py * fb_w + (uint32_t)px;
            uint32_t old_depth = atomicMin(&depth_buf[pixel_idx], idepth);
            if (idepth > old_depth) continue;  /* failed depth test */

            /* Interpolate color */
            float r = w0 * r0 + w1 * r1 + w2 * r2;
            float g = w0 * g0 + w1 * g1 + w2 * g2;
            float b = w0 * b0 + w1 * b1 + w2 * b2;

            /* Write to color buffer (no blending, opaque) */
            uint32_t cidx = pixel_idx * 4;
            color_buf[cidx + 0] = r;
            color_buf[cidx + 1] = g;
            color_buf[cidx + 2] = b;
            color_buf[cidx + 3] = 1.0f;
        }
    }
}

extern "C" void ptx_gfx_rasterize_triangles(
    const float* screen_positions,
    const float* vertex_colors,
    const uint32_t* indices,
    float* color_buf,
    uint32_t* depth_buf,
    uint32_t fb_width, uint32_t fb_height,
    uint32_t num_triangles,
    cudaStream_t stream)
{
    if (num_triangles == 0) return;
    k_gfx_rasterize_triangles<<<GFX_GRID_SIZE(num_triangles), GFX_BLOCK_SIZE, 0, stream>>>(
        screen_positions, vertex_colors, indices,
        color_buf, depth_buf,
        fb_width, fb_height, num_triangles);
}

/* ========================================================================== */
/* Normal rasterization (writes interpolated normals for shading)             */
/* ========================================================================== */

__global__ void k_gfx_rasterize_normals(
    const float* __restrict__ screen_pos,
    const float* __restrict__ vtx_normals,
    const uint32_t* __restrict__ indices,
    float* __restrict__ normal_buf,
    const uint32_t* __restrict__ depth_buf,
    uint32_t fb_w, uint32_t fb_h,
    uint32_t num_triangles)
{
    uint32_t tri = blockIdx.x * blockDim.x + threadIdx.x;
    if (tri >= num_triangles) return;

    uint32_t i0 = indices[tri * 3 + 0];
    uint32_t i1 = indices[tri * 3 + 1];
    uint32_t i2 = indices[tri * 3 + 2];

    float x0 = screen_pos[i0*3+0], y0 = screen_pos[i0*3+1], z0 = screen_pos[i0*3+2];
    float x1 = screen_pos[i1*3+0], y1 = screen_pos[i1*3+1], z1 = screen_pos[i1*3+2];
    float x2 = screen_pos[i2*3+0], y2 = screen_pos[i2*3+1], z2 = screen_pos[i2*3+2];

    float area = edge_function(x0, y0, x1, y1, x2, y2);
    if (fabsf(area) < 1e-6f) return;
    float inv_area = 1.0f / area;

    float nx0 = vtx_normals[i0*3+0], ny0 = vtx_normals[i0*3+1], nz0 = vtx_normals[i0*3+2];
    float nx1 = vtx_normals[i1*3+0], ny1 = vtx_normals[i1*3+1], nz1 = vtx_normals[i1*3+2];
    float nx2 = vtx_normals[i2*3+0], ny2 = vtx_normals[i2*3+1], nz2 = vtx_normals[i2*3+2];

    int min_x = max(0, (int)floorf(fminf(fminf(x0, x1), x2)));
    int max_x = min((int)fb_w - 1, (int)ceilf(fmaxf(fmaxf(x0, x1), x2)));
    int min_y = max(0, (int)floorf(fminf(fminf(y0, y1), y2)));
    int max_y = min((int)fb_h - 1, (int)ceilf(fmaxf(fmaxf(y0, y1), y2)));

    for (int py = min_y; py <= max_y; py++) {
        for (int px = min_x; px <= max_x; px++) {
            float fpx = (float)px + 0.5f;
            float fpy = (float)py + 0.5f;

            float w0 = edge_function(x1, y1, x2, y2, fpx, fpy) * inv_area;
            float w1 = edge_function(x2, y2, x0, y0, fpx, fpy) * inv_area;
            float w2 = edge_function(x0, y0, x1, y1, fpx, fpy) * inv_area;

            bool inside;
            if (area > 0.0f) {
                inside = (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f);
            } else {
                inside = (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);
                w0 = -w0; w1 = -w1; w2 = -w2;
            }
            if (!inside) continue;

            /* Check this pixel matches the depth buffer (was written by rasterize_triangles) */
            float depth = w0 * z0 + w1 * z1 + w2 * z2;
            uint32_t idepth = depth_to_uint(depth);
            uint32_t pixel_idx = (uint32_t)py * fb_w + (uint32_t)px;

            if (depth_buf[pixel_idx] != idepth) continue;

            /* Interpolate and write normal */
            uint32_t nidx = pixel_idx * 3;
            normal_buf[nidx + 0] = w0 * nx0 + w1 * nx1 + w2 * nx2;
            normal_buf[nidx + 1] = w0 * ny0 + w1 * ny1 + w2 * ny2;
            normal_buf[nidx + 2] = w0 * nz0 + w1 * nz1 + w2 * nz2;
        }
    }
}

extern "C" void ptx_gfx_rasterize_normals(
    const float* screen_positions,
    const float* vertex_normals,
    const uint32_t* indices,
    float* normal_buf,
    const uint32_t* depth_buf,
    uint32_t fb_width, uint32_t fb_height,
    uint32_t num_triangles,
    cudaStream_t stream)
{
    if (num_triangles == 0) return;
    k_gfx_rasterize_normals<<<GFX_GRID_SIZE(num_triangles), GFX_BLOCK_SIZE, 0, stream>>>(
        screen_positions, vertex_normals, indices,
        normal_buf, depth_buf,
        fb_width, fb_height, num_triangles);
}

/* ========================================================================== */
/* Blinn-Phong shading                                                        */
/* ========================================================================== */

__global__ void k_gfx_shade_phong(
    const float* __restrict__ normal_buf,
    float* __restrict__ color_buf,
    const uint32_t* __restrict__ depth_buf,
    const float* __restrict__ light_dir,
    const float* __restrict__ light_color,
    const float* __restrict__ ambient,
    uint32_t num_pixels)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    /* Skip background pixels (depth == max) */
    if (depth_buf[idx] == 0x7F800000u) return;  /* +inf as uint */

    float ld_x = light_dir[0], ld_y = light_dir[1], ld_z = light_dir[2];
    float lc_r = light_color[0], lc_g = light_color[1], lc_b = light_color[2];
    float am_r = ambient[0], am_g = ambient[1], am_b = ambient[2];

    float nx, ny, nz;
    if (normal_buf != NULL) {
        nx = normal_buf[idx * 3 + 0];
        ny = normal_buf[idx * 3 + 1];
        nz = normal_buf[idx * 3 + 2];
        /* Normalize */
        float len = sqrtf(nx*nx + ny*ny + nz*nz);
        if (len > 1e-7f) {
            float inv_len = 1.0f / len;
            nx *= inv_len; ny *= inv_len; nz *= inv_len;
        }
    } else {
        /* Default normal: facing camera (0, 0, 1) */
        nx = 0.0f; ny = 0.0f; nz = 1.0f;
    }

    /* Lambertian diffuse: max(0, N . L) */
    float ndotl = fmaxf(0.0f, nx * ld_x + ny * ld_y + nz * ld_z);

    /* Read base color */
    uint32_t cidx = idx * 4;
    float cr = color_buf[cidx + 0];
    float cg = color_buf[cidx + 1];
    float cb = color_buf[cidx + 2];

    /* Apply lighting: color * (ambient + diffuse * light_color) */
    color_buf[cidx + 0] = fminf(1.0f, cr * (am_r + ndotl * lc_r));
    color_buf[cidx + 1] = fminf(1.0f, cg * (am_g + ndotl * lc_g));
    color_buf[cidx + 2] = fminf(1.0f, cb * (am_b + ndotl * lc_b));
    /* Alpha unchanged */
}

extern "C" void ptx_gfx_shade_phong(
    const float* normal_buf,
    float* color_buf,
    const uint32_t* depth_buf,
    const float* light_dir,
    const float* light_color,
    const float* ambient,
    uint32_t num_pixels,
    cudaStream_t stream)
{
    if (num_pixels == 0) return;
    k_gfx_shade_phong<<<GFX_GRID_SIZE(num_pixels), GFX_BLOCK_SIZE, 0, stream>>>(
        normal_buf, color_buf, depth_buf,
        light_dir, light_color, ambient,
        num_pixels);
}

/* ========================================================================== */
/* Framebuffer → packed RGBA for presentation                                 */
/* ========================================================================== */

__global__ void k_gfx_framebuffer_to_rgba(const float* __restrict__ color_buf,
                                           uint32_t* __restrict__ rgba_out,
                                           uint32_t num_pixels) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    uint32_t cidx = idx * 4;
    uint32_t r = (uint32_t)(fminf(fmaxf(color_buf[cidx + 0], 0.0f), 1.0f) * 255.0f + 0.5f);
    uint32_t g = (uint32_t)(fminf(fmaxf(color_buf[cidx + 1], 0.0f), 1.0f) * 255.0f + 0.5f);
    uint32_t b = (uint32_t)(fminf(fmaxf(color_buf[cidx + 2], 0.0f), 1.0f) * 255.0f + 0.5f);
    uint32_t a = (uint32_t)(fminf(fmaxf(color_buf[cidx + 3], 0.0f), 1.0f) * 255.0f + 0.5f);

    /* Pack as 0xRRGGBBAA (ferrite-window format) */
    rgba_out[idx] = (r << 24) | (g << 16) | (b << 8) | a;
}

extern "C" void ptx_gfx_framebuffer_to_rgba(const float* color_buf,
                                              uint32_t* rgba_out,
                                              uint32_t num_pixels,
                                              cudaStream_t stream) {
    if (num_pixels == 0) return;
    k_gfx_framebuffer_to_rgba<<<GFX_GRID_SIZE(num_pixels), GFX_BLOCK_SIZE, 0, stream>>>(
        color_buf, rgba_out, num_pixels);
}
