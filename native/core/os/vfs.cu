/*
 * PTX-OS Virtual Filesystem
 * GPU-resident tensor filesystem with POSIX-like API
 */

#include "gpu/gpu_hot_runtime.h"
#include "ptx_debug.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// VFS Internal Helpers
// ============================================================================

static inline uint64_t vfs_get_timestamp() {
#ifdef _WIN32
    LARGE_INTEGER counter, freq;
    QueryPerformanceCounter(&counter);
    QueryPerformanceFrequency(&freq);
    return (counter.QuadPart * 1000000) / freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
#endif
}

// Get dtype size in bytes
static size_t vfs_dtype_size(int dtype) {
    switch (dtype) {
        case 0: return 4;  // f32
        case 1: return 2;  // f16
        case 2: return 4;  // i32
        case 3: return 1;  // i8
        default: return 4;
    }
}

// Calculate tensor size in bytes
static size_t vfs_tensor_size(int* shape, int dims, int dtype) {
    size_t elements = 1;
    for (int i = 0; i < dims; i++) {
        elements *= shape[i];
    }
    return elements * vfs_dtype_size(dtype);
}

static void* vfs_alloc_device(VFSState* vfs, size_t size) {
    if (vfs && vfs->runtime) {
        return gpu_hot_alloc(vfs->runtime, size);
    }
    return NULL;
}

static void vfs_free_device(VFSState* vfs, void* ptr) {
    if (!ptr) return;
    if (vfs && vfs->runtime && gpu_hot_owns_ptr(vfs->runtime, ptr)) {
        gpu_hot_free(vfs->runtime, ptr);
        return;
    }
    ptx_strict_free_violation("VFS", ptr);
}

// Find free inode
static VFSInode* vfs_alloc_inode(VFSState* vfs) {
    for (uint32_t i = 0; i < VFS_MAX_NODES; i++) {
        if (!vfs->inodes[i].active) {
            VFSInode* inode = &vfs->inodes[i];
            memset(inode, 0, sizeof(VFSInode));
            inode->inode_id = i;
            inode->active = true;
            inode->created_at = vfs_get_timestamp();
            inode->modified_at = inode->created_at;
            inode->accessed_at = inode->created_at;
            vfs->num_inodes++;
            return inode;
        }
    }
    return NULL;
}

// Free inode and its resources
static void vfs_free_inode(VFSState* vfs, VFSInode* inode) {
    if (!inode || !inode->active) return;

    // Free GPU data
    if (inode->data) {
        vfs_free_device(vfs, inode->data);
        inode->data = NULL;
    }

    // Remove from parent's child list
    if (inode->parent) {
        VFSInode* prev = NULL;
        VFSInode* curr = inode->parent->first_child;
        while (curr) {
            if (curr == inode) {
                if (prev) {
                    prev->next_sibling = curr->next_sibling;
                } else {
                    inode->parent->first_child = curr->next_sibling;
                }
                break;
            }
            prev = curr;
            curr = curr->next_sibling;
        }
    }

    inode->active = false;
    vfs->num_inodes--;
}

// Parse path and find parent directory
static VFSInode* vfs_resolve_parent(VFSState* vfs, const char* path, char* basename) {
    if (!path || path[0] != '/') return NULL;

    // Find last component
    const char* last_slash = strrchr(path, '/');
    if (!last_slash) return NULL;

    // Extract basename
    if (basename) {
        strncpy(basename, last_slash + 1, GPU_HOT_MAX_NAME_LEN - 1);
    }

    // Handle root directory
    if (last_slash == path) {
        return vfs->root;
    }

    // Extract parent path
    size_t parent_len = last_slash - path;
    char parent_path[VFS_MAX_PATH_LEN];
    strncpy(parent_path, path, parent_len);
    parent_path[parent_len] = '\0';

    return vfs_lookup(vfs, parent_path);
}

// Find file handle by fd
static VFSFileHandle* vfs_get_handle(VFSState* vfs, int fd) {
    for (uint32_t i = 0; i < VFS_MAX_OPEN_FILES; i++) {
        if (vfs->open_files[i].is_open && vfs->open_files[i].fd == (uint32_t)fd) {
            return &vfs->open_files[i];
        }
    }
    return NULL;
}

// ============================================================================
// VFS Initialization
// ============================================================================

VFSState* vfs_init(GPUHotRuntime* runtime) {
    printf("[VFS] Initializing Virtual Filesystem...\n");

    VFSState* vfs = (VFSState*)malloc(sizeof(VFSState));
    if (!vfs) {
        printf("[VFS] ERROR: Failed to allocate VFS state\n");
        return NULL;
    }

    memset(vfs, 0, sizeof(VFSState));
    vfs->runtime = runtime;

    // Create root directory
    vfs->root = vfs_alloc_inode(vfs);
    if (!vfs->root) {
        printf("[VFS] ERROR: Failed to create root inode\n");
        free(vfs);
        return NULL;
    }

    vfs->root->type = VFS_NODE_DIRECTORY;
    strcpy(vfs->root->name, "/");
    strcpy(vfs->root->path, "/");
    vfs->root->mode = 0755;

    vfs->next_fd = 3;  // 0, 1, 2 reserved for stdin/stdout/stderr

    printf("[VFS] [OK] VFS initialized with root at /\n");
    printf("[VFS] Max nodes: %d\n", VFS_MAX_NODES);
    printf("[VFS] Max open files: %d\n", VFS_MAX_OPEN_FILES);

    return vfs;
}

void vfs_shutdown(VFSState* vfs) {
    if (!vfs) return;

    printf("[VFS] Shutting down...\n");

    // Close all open files
    for (uint32_t i = 0; i < VFS_MAX_OPEN_FILES; i++) {
        if (vfs->open_files[i].is_open) {
            vfs->open_files[i].is_open = false;
        }
    }

    // Free all inodes
    for (uint32_t i = 0; i < VFS_MAX_NODES; i++) {
        if (vfs->inodes[i].active && vfs->inodes[i].data) {
            vfs_free_device(vfs, vfs->inodes[i].data);
        }
    }

    free(vfs);
    printf("[VFS] [OK] Shutdown complete\n");
}

// ============================================================================
// Path Operations
// ============================================================================

VFSInode* vfs_lookup(VFSState* vfs, const char* path) {
    if (!vfs || !path) return NULL;

    // Handle root
    if (strcmp(path, "/") == 0) {
        return vfs->root;
    }

    // Tokenize path
    char path_copy[VFS_MAX_PATH_LEN];
    strncpy(path_copy, path, VFS_MAX_PATH_LEN - 1);

    VFSInode* current = vfs->root;
    char* token = strtok(path_copy + 1, "/");  // Skip leading /

    while (token && current) {
        VFSInode* found = NULL;
        VFSInode* child = current->first_child;

        while (child) {
            if (strcmp(child->name, token) == 0) {
                found = child;
                break;
            }
            child = child->next_sibling;
        }

        if (!found) return NULL;

        current = found;
        token = strtok(NULL, "/");
    }

    if (current) {
        current->accessed_at = vfs_get_timestamp();
    }

    return current;
}

int vfs_mkdir(VFSState* vfs, const char* path, uint32_t mode) {
    if (!vfs || !path) return -1;

    // Check if already exists
    if (vfs_lookup(vfs, path)) {
        printf("[VFS] ERROR: Path already exists: %s\n", path);
        return -1;
    }

    // Resolve parent
    char name[GPU_HOT_MAX_NAME_LEN];
    VFSInode* parent = vfs_resolve_parent(vfs, path, name);
    if (!parent || parent->type != VFS_NODE_DIRECTORY) {
        printf("[VFS] ERROR: Parent directory not found: %s\n", path);
        return -1;
    }

    // Create new directory inode
    VFSInode* dir = vfs_alloc_inode(vfs);
    if (!dir) {
        printf("[VFS] ERROR: No free inodes\n");
        return -1;
    }

    dir->type = VFS_NODE_DIRECTORY;
    strncpy(dir->name, name, GPU_HOT_MAX_NAME_LEN - 1);
    strncpy(dir->path, path, VFS_MAX_PATH_LEN - 1);
    dir->mode = mode;
    dir->parent = parent;

    // Add to parent's children
    dir->next_sibling = parent->first_child;
    parent->first_child = dir;
    parent->modified_at = vfs_get_timestamp();

    printf("[VFS] Created directory: %s\n", path);
    return 0;
}

int vfs_rmdir(VFSState* vfs, const char* path) {
    if (!vfs || !path) return -1;

    VFSInode* dir = vfs_lookup(vfs, path);
    if (!dir) {
        printf("[VFS] ERROR: Directory not found: %s\n", path);
        return -1;
    }

    if (dir->type != VFS_NODE_DIRECTORY) {
        printf("[VFS] ERROR: Not a directory: %s\n", path);
        return -1;
    }

    if (dir->first_child) {
        printf("[VFS] ERROR: Directory not empty: %s\n", path);
        return -1;
    }

    if (dir == vfs->root) {
        printf("[VFS] ERROR: Cannot remove root directory\n");
        return -1;
    }

    vfs_free_inode(vfs, dir);
    printf("[VFS] Removed directory: %s\n", path);
    return 0;
}

int vfs_unlink(VFSState* vfs, const char* path) {
    if (!vfs || !path) return -1;

    VFSInode* inode = vfs_lookup(vfs, path);
    if (!inode) {
        printf("[VFS] ERROR: File not found: %s\n", path);
        return -1;
    }

    if (inode->type == VFS_NODE_DIRECTORY) {
        printf("[VFS] ERROR: Cannot unlink directory (use rmdir): %s\n", path);
        return -1;
    }

    // Check if file is open
    for (uint32_t i = 0; i < VFS_MAX_OPEN_FILES; i++) {
        if (vfs->open_files[i].is_open && vfs->open_files[i].inode == inode) {
            printf("[VFS] ERROR: File is open: %s\n", path);
            return -1;
        }
    }

    vfs_free_inode(vfs, inode);
    printf("[VFS] Removed file: %s\n", path);
    return 0;
}

// ============================================================================
// File Operations
// ============================================================================

int vfs_open(VFSState* vfs, const char* path, uint32_t flags) {
    if (!vfs || !path) return -1;

    VFSInode* inode = vfs_lookup(vfs, path);

    // Create if not exists and O_CREAT
    if (!inode && (flags & VFS_O_CREAT)) {
        char name[GPU_HOT_MAX_NAME_LEN];
        VFSInode* parent = vfs_resolve_parent(vfs, path, name);
        if (!parent) {
            printf("[VFS] ERROR: Parent directory not found: %s\n", path);
            return -1;
        }

        inode = vfs_alloc_inode(vfs);
        if (!inode) {
            printf("[VFS] ERROR: No free inodes\n");
            return -1;
        }

        inode->type = VFS_NODE_FILE;
        strncpy(inode->name, name, GPU_HOT_MAX_NAME_LEN - 1);
        strncpy(inode->path, path, VFS_MAX_PATH_LEN - 1);
        inode->mode = 0644;
        inode->parent = parent;

        inode->next_sibling = parent->first_child;
        parent->first_child = inode;

        printf("[VFS] Created file: %s\n", path);
    }

    if (!inode) {
        printf("[VFS] ERROR: File not found: %s\n", path);
        return -1;
    }

    // Find free file handle
    VFSFileHandle* handle = NULL;
    for (uint32_t i = 0; i < VFS_MAX_OPEN_FILES; i++) {
        if (!vfs->open_files[i].is_open) {
            handle = &vfs->open_files[i];
            break;
        }
    }

    if (!handle) {
        printf("[VFS] ERROR: Too many open files\n");
        return -1;
    }

    // Truncate if O_TRUNC
    if (flags & VFS_O_TRUNC) {
        if (inode->data) {
            vfs_free_device(vfs, inode->data);
            inode->data = NULL;
            inode->size = 0;
        }
    }

    handle->fd = vfs->next_fd++;
    handle->inode = inode;
    handle->offset = (flags & VFS_O_APPEND) ? inode->size : 0;
    handle->flags = flags;
    handle->is_open = true;

    vfs->num_open_files++;
    vfs->opens++;
    inode->accessed_at = vfs_get_timestamp();

    return handle->fd;
}

int vfs_close(VFSState* vfs, int fd) {
    if (!vfs) return -1;

    VFSFileHandle* handle = vfs_get_handle(vfs, fd);
    if (!handle) {
        printf("[VFS] ERROR: Invalid file descriptor: %d\n", fd);
        return -1;
    }

    handle->is_open = false;
    vfs->num_open_files--;
    vfs->closes++;

    return 0;
}

ssize_t vfs_read(VFSState* vfs, int fd, void* buf, size_t count) {
    if (!vfs || !buf) return -1;

    VFSFileHandle* handle = vfs_get_handle(vfs, fd);
    if (!handle) {
        printf("[VFS] ERROR: Invalid file descriptor: %d\n", fd);
        return -1;
    }

    if (!(handle->flags & VFS_O_RDONLY) && !(handle->flags & VFS_O_RDWR)) {
        printf("[VFS] ERROR: File not opened for reading\n");
        return -1;
    }

    VFSInode* inode = handle->inode;
    if (!inode->data || handle->offset >= inode->size) {
        return 0;  // EOF
    }

    size_t available = inode->size - handle->offset;
    size_t to_read = (count < available) ? count : available;

    // Copy from GPU to host buffer
    cudaError_t err = cudaMemcpy(buf, (char*)inode->data + handle->offset,
                                  to_read, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("[VFS] ERROR: Read failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    handle->offset += to_read;
    inode->accessed_at = vfs_get_timestamp();
    vfs->reads++;

    return to_read;
}

ssize_t vfs_write(VFSState* vfs, int fd, const void* buf, size_t count) {
    if (!vfs || !buf || count == 0) return -1;

    VFSFileHandle* handle = vfs_get_handle(vfs, fd);
    if (!handle) {
        printf("[VFS] ERROR: Invalid file descriptor: %d\n", fd);
        return -1;
    }

    if (!(handle->flags & VFS_O_WRONLY) && !(handle->flags & VFS_O_RDWR)) {
        printf("[VFS] ERROR: File not opened for writing\n");
        return -1;
    }

    VFSInode* inode = handle->inode;
    size_t new_size = handle->offset + count;

    // Grow file if needed
    if (new_size > inode->size || !inode->data) {
        void* new_data = vfs_alloc_device(vfs, new_size);
        if (!new_data) {
            printf("[VFS] ERROR: Failed to allocate GPU memory\n");
            return -1;
        }

        // Copy existing data
        if (inode->data && inode->size > 0) {
            cudaMemcpy(new_data, inode->data, inode->size, cudaMemcpyDeviceToDevice);
            vfs_free_device(vfs, inode->data);
        }

        inode->data = new_data;
        inode->size = new_size;
    }

    // Write to GPU
    cudaError_t err = cudaMemcpy((char*)inode->data + handle->offset,
                                  buf, count, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("[VFS] ERROR: Write failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    handle->offset += count;
    inode->modified_at = vfs_get_timestamp();
    vfs->writes++;

    return count;
}

int vfs_seek(VFSState* vfs, int fd, size_t offset, int whence) {
    if (!vfs) return -1;

    VFSFileHandle* handle = vfs_get_handle(vfs, fd);
    if (!handle) return -1;

    VFSInode* inode = handle->inode;

    switch (whence) {
        case 0:  // SEEK_SET
            handle->offset = offset;
            break;
        case 1:  // SEEK_CUR
            handle->offset += offset;
            break;
        case 2:  // SEEK_END
            handle->offset = inode->size + offset;
            break;
        default:
            return -1;
    }

    return handle->offset;
}

// ============================================================================
// Tensor Operations
// ============================================================================

int vfs_create_tensor(VFSState* vfs, const char* path, int* shape, int dims, int dtype) {
    if (!vfs || !path || !shape || dims <= 0 || dims > 8) return -1;

    // Check if already exists
    if (vfs_lookup(vfs, path)) {
        printf("[VFS] ERROR: Path already exists: %s\n", path);
        return -1;
    }

    // Resolve parent
    char name[GPU_HOT_MAX_NAME_LEN];
    VFSInode* parent = vfs_resolve_parent(vfs, path, name);
    if (!parent || parent->type != VFS_NODE_DIRECTORY) {
        printf("[VFS] ERROR: Parent directory not found: %s\n", path);
        return -1;
    }

    // Calculate size
    size_t tensor_bytes = vfs_tensor_size(shape, dims, dtype);

    // Allocate GPU memory
    void* gpu_data = vfs_alloc_device(vfs, tensor_bytes);
    if (!gpu_data) {
        printf("[VFS] ERROR: Failed to allocate tensor memory\n");
        return -1;
    }

    // Zero-initialize
    cudaMemset(gpu_data, 0, tensor_bytes);

    // Create tensor inode
    VFSInode* tensor = vfs_alloc_inode(vfs);
    if (!tensor) {
        vfs_free_device(vfs, gpu_data);
        printf("[VFS] ERROR: No free inodes\n");
        return -1;
    }

    tensor->type = VFS_NODE_TENSOR;
    strncpy(tensor->name, name, GPU_HOT_MAX_NAME_LEN - 1);
    strncpy(tensor->path, path, VFS_MAX_PATH_LEN - 1);
    tensor->mode = 0644;
    tensor->parent = parent;
    tensor->data = gpu_data;
    tensor->size = tensor_bytes;
    tensor->dims = dims;
    tensor->dtype = dtype;
    memcpy(tensor->shape, shape, dims * sizeof(int));

    // Add to parent's children
    tensor->next_sibling = parent->first_child;
    parent->first_child = tensor;
    parent->modified_at = vfs_get_timestamp();

    printf("[VFS] Created tensor: %s (", path);
    for (int i = 0; i < dims; i++) {
        printf("%d%s", shape[i], i < dims - 1 ? "x" : "");
    }
    printf(") = %.2f MB\n", tensor_bytes / (1024.0 * 1024.0));

    return 0;
}

void* vfs_mmap_tensor(VFSState* vfs, const char* path) {
    if (!vfs || !path) return NULL;

    VFSInode* inode = vfs_lookup(vfs, path);
    if (!inode || inode->type != VFS_NODE_TENSOR) {
        printf("[VFS] ERROR: Tensor not found: %s\n", path);
        return NULL;
    }

    inode->accessed_at = vfs_get_timestamp();
    return inode->data;
}

int vfs_sync_tensor(VFSState* vfs, const char* path) {
    if (!vfs || !path) return -1;

    VFSInode* inode = vfs_lookup(vfs, path);
    if (!inode || inode->type != VFS_NODE_TENSOR) {
        return -1;
    }

    // Ensure GPU operations are complete
    cudaDeviceSynchronize();
    inode->modified_at = vfs_get_timestamp();

    return 0;
}

// ============================================================================
// Directory Listing
// ============================================================================

int vfs_readdir(VFSState* vfs, const char* path, char** names, int max_entries) {
    if (!vfs || !path || !names) return -1;

    VFSInode* dir = vfs_lookup(vfs, path);
    if (!dir || dir->type != VFS_NODE_DIRECTORY) {
        return -1;
    }

    int count = 0;
    VFSInode* child = dir->first_child;

    while (child && count < max_entries) {
        if (names[count]) {
            strncpy(names[count], child->name, GPU_HOT_MAX_NAME_LEN - 1);
        }
        count++;
        child = child->next_sibling;
    }

    return count;
}

int vfs_stat(VFSState* vfs, const char* path, VFSInode* stat_out) {
    if (!vfs || !path || !stat_out) return -1;

    VFSInode* inode = vfs_lookup(vfs, path);
    if (!inode) {
        return -1;
    }

    memcpy(stat_out, inode, sizeof(VFSInode));
    return 0;
}

// ============================================================================
// VFS Debug
// ============================================================================

static void vfs_print_tree_recursive(VFSInode* node, int depth) {
    if (!node) return;

    for (int i = 0; i < depth; i++) printf("  ");

    const char* type_str = "???";
    switch (node->type) {
        case VFS_NODE_FILE: type_str = "FILE"; break;
        case VFS_NODE_DIRECTORY: type_str = "DIR"; break;
        case VFS_NODE_TENSOR: type_str = "TENSOR"; break;
        case VFS_NODE_STREAM: type_str = "STREAM"; break;
        case VFS_NODE_DEVICE: type_str = "DEV"; break;
    }

    printf("%s [%s]", node->name, type_str);

    if (node->type == VFS_NODE_TENSOR) {
        printf(" (");
        for (int i = 0; i < node->dims; i++) {
            printf("%d%s", node->shape[i], i < node->dims - 1 ? "x" : "");
        }
        printf(")");
    }

    if (node->size > 0) {
        printf(" %.2f KB", node->size / 1024.0);
    }

    printf("\n");

    VFSInode* child = node->first_child;
    while (child) {
        vfs_print_tree_recursive(child, depth + 1);
        child = child->next_sibling;
    }
}

void vfs_print_tree(VFSState* vfs) {
    if (!vfs) return;

    printf("\n========== VFS Tree ==========\n");
    vfs_print_tree_recursive(vfs->root, 0);
    printf("==============================\n\n");
}

void vfs_print_stats(VFSState* vfs) {
    if (!vfs) return;

    printf("\n========== VFS Statistics ==========\n");
    printf("Total inodes:  %u / %d\n", vfs->num_inodes, VFS_MAX_NODES);
    printf("Open files:    %u / %d\n", vfs->num_open_files, VFS_MAX_OPEN_FILES);
    printf("Reads:         %lu\n", vfs->reads);
    printf("Writes:        %lu\n", vfs->writes);
    printf("Opens:         %lu\n", vfs->opens);
    printf("Closes:        %lu\n", vfs->closes);
    printf("====================================\n\n");
}
