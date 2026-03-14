#ifndef PTX_DEBUG_H
#define PTX_DEBUG_H

#include <stdio.h>
#include <stdlib.h>

#if defined(PTX_STRICT_FREE) || !defined(NDEBUG)
#define PTX_STRICT_FREE_ABORT 1
#else
#define PTX_STRICT_FREE_ABORT 0
#endif

static inline void ptx_strict_free_violation(const char* tag, const void* ptr) {
    fprintf(stderr, "[%s] ERROR: Free violation for pointer %p\n", tag, ptr);
#if PTX_STRICT_FREE_ABORT
    abort();
#endif
}

#endif // PTX_DEBUG_H
