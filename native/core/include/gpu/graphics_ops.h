/*
 * Ferrite-OS Graphics Operations - GPU Software Rasterization
 * CUDA kernels for vertex processing, triangle rasterization, shading,
 * and framebuffer management. All rendering runs as pure compute — no
 * OpenGL, Vulkan, or GPU driver graphics stack required.
 */

#ifndef FERRITE_GRAPHICS_OPS_H
#define FERRITE_GRAPHICS_OPS_H

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
/* Framebuffer Operations                                                     */
/* ========================================================================== */

/*
 * Clear a color framebuffer to a solid RGBA color.
 *   color_buf  — [H * W * 4] floats (RGBA per pixel, row-major)
 *   r, g, b, a — clear color components [0..1]
 *   num_pixels — W * H
 */
void ptx_gfx_clear_color(
    float* color_buf,
    float r, float g, float b, float a,
    uint32_t num_pixels,
    cudaStream_t stream
);

/*
 * Clear a depth buffer to a given value (typically 1.0 = far plane).
 *   depth_buf  — [H * W] floats
 *   value      — clear depth (1.0 for far, 0.0 for near)
 *   num_pixels — W * H
 */
void ptx_gfx_clear_depth(
    float* depth_buf,
    float value,
    uint32_t num_pixels,
    cudaStream_t stream
);

/* ========================================================================== */
/* Vertex Processing                                                          */
/* ========================================================================== */

/*
 * Transform vertices by a 4x4 matrix (model-view-projection).
 *   positions_in  — [N * 3] floats  (x, y, z per vertex)
 *   positions_out — [N * 4] floats  (x, y, z, w clip-space output)
 *   mvp           — [16] floats     (4x4 column-major matrix)
 *   num_vertices  — N
 */
void ptx_gfx_vertex_transform(
    const float* positions_in,
    float* positions_out,
    const float* mvp,
    uint32_t num_vertices,
    cudaStream_t stream
);

/*
 * Perspective divide + viewport transform: clip-space → screen-space.
 *   clip_positions  — [N * 4] floats (x, y, z, w from vertex transform)
 *   screen_out      — [N * 3] floats (screen_x, screen_y, depth)
 *   viewport_width  — framebuffer width in pixels
 *   viewport_height — framebuffer height in pixels
 *   num_vertices    — N
 */
void ptx_gfx_viewport_transform(
    const float* clip_positions,
    float* screen_out,
    uint32_t viewport_width,
    uint32_t viewport_height,
    uint32_t num_vertices,
    cudaStream_t stream
);

/* ========================================================================== */
/* Triangle Rasterization                                                     */
/* ========================================================================== */

/*
 * Rasterize indexed triangles into a color + depth framebuffer.
 * One thread per triangle — scans the bounding box, tests edge functions,
 * performs atomic depth test, and writes interpolated color.
 *
 *   screen_positions — [V * 3] floats (screen_x, screen_y, depth) per vertex
 *   vertex_colors    — [V * 3] floats (r, g, b) per vertex (NULL for white)
 *   indices          — [T * 3] uint32 (3 vertex indices per triangle)
 *   color_buf        — [H * W * 4] floats (RGBA framebuffer, output)
 *   depth_buf        — [H * W] uint32 (integer depth for atomicMin)
 *   fb_width         — framebuffer width
 *   fb_height        — framebuffer height
 *   num_triangles    — T
 */
void ptx_gfx_rasterize_triangles(
    const float* screen_positions,
    const float* vertex_colors,
    const uint32_t* indices,
    float* color_buf,
    uint32_t* depth_buf,
    uint32_t fb_width,
    uint32_t fb_height,
    uint32_t num_triangles,
    cudaStream_t stream
);

/* ========================================================================== */
/* Shading                                                                    */
/* ========================================================================== */

/*
 * Apply per-pixel Blinn-Phong shading to a framebuffer.
 * Modifies color_buf in-place using normals and light parameters.
 *
 *   normal_buf    — [H * W * 3] floats (interpolated surface normals, or NULL to skip)
 *   color_buf     — [H * W * 4] floats (RGBA, modified in-place)
 *   depth_buf     — [H * W] uint32     (integer depth, pixels with max depth are skipped)
 *   light_dir     — [3] floats         (normalized light direction, world space)
 *   light_color   — [3] floats         (light RGB intensity)
 *   ambient       — [3] floats         (ambient RGB)
 *   num_pixels    — W * H
 */
void ptx_gfx_shade_phong(
    const float* normal_buf,
    float* color_buf,
    const uint32_t* depth_buf,
    const float* light_dir,
    const float* light_color,
    const float* ambient,
    uint32_t num_pixels,
    cudaStream_t stream
);

/* ========================================================================== */
/* Framebuffer Conversion                                                     */
/* ========================================================================== */

/*
 * Convert float RGBA framebuffer [H*W*4] to packed 0xRRGGBBAA u32 for
 * presentation via ferrite-window.
 *   color_buf  — [H * W * 4] floats (RGBA, each in [0..1])
 *   rgba_out   — [H * W] uint32     (packed 0xRRGGBBAA)
 *   num_pixels — W * H
 */
void ptx_gfx_framebuffer_to_rgba(
    const float* color_buf,
    uint32_t* rgba_out,
    uint32_t num_pixels,
    cudaStream_t stream
);

/*
 * Generate a normal buffer from triangle mesh via rasterization.
 * Same structure as rasterize_triangles but writes interpolated normals.
 *
 *   screen_positions — [V * 3] floats (screen_x, screen_y, depth)
 *   vertex_normals   — [V * 3] floats (nx, ny, nz per vertex)
 *   indices          — [T * 3] uint32
 *   normal_buf       — [H * W * 3] floats (output)
 *   depth_buf        — [H * W] uint32 (read, used to match pixels)
 *   fb_width, fb_height, num_triangles
 */
void ptx_gfx_rasterize_normals(
    const float* screen_positions,
    const float* vertex_normals,
    const uint32_t* indices,
    float* normal_buf,
    const uint32_t* depth_buf,
    uint32_t fb_width,
    uint32_t fb_height,
    uint32_t num_triangles,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif /* FERRITE_GRAPHICS_OPS_H */
