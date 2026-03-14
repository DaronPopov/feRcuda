// ============================================================================
// Kernel Configuration
// ============================================================================

#define PTX_BLOCK_SIZE 256
#define PTX_WARP_SIZE 32

// Calculate grid size for N elements
#define PTX_GRID_SIZE(n) (((n) + PTX_BLOCK_SIZE - 1) / PTX_BLOCK_SIZE)

// Math constants
#define PTX_SQRT_2_PI 2.5066282746310002f
#define PTX_SQRT_2_OVER_PI 0.7978845608028654f
#define PTX_GELU_COEF 0.044715f
#define PTX_SELU_ALPHA 1.6732632423543772f
#define PTX_SELU_LAMBDA 1.0507009873554805f

