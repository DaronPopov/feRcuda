#include "fercuda/alloc/memory.cuh"

#include <cstdio>

using namespace fer ; int main() {
    bool all = true ; PoolConfig cfg{} ; cfg.mutable_bytes = 8ULL << 20 ; cfg.immutable_bytes = 8ULL << 20 ; cfg.regime = MemoryRegime::CUSTOM_POOL ; ElasticPool pool(0, cfg) ; void* a = nullptr ; void* b = nullptr ; void* d = nullptr ; try {
        a = pool.alloc_bytes(3ULL << 20, Tier::MUTABLE) ; b = pool.alloc_bytes(3ULL << 20, Tier::MUTABLE) ; bool t1 = (a != nullptr && b != nullptr) ; std::printf("[%s] initial mutable allocations\n", t1 ? "PASS" : "FAIL") ; all = all && t1 ; } catch (...) {
        std::printf("[FAIL] initial mutable allocations\n") ; return 1 ; }

    bool exhausted = false ; try {
        (void)pool.alloc_bytes(3ULL << 20, Tier::MUTABLE) ; } catch (...) {
        exhausted = true ; }
    std::printf("[%s] mutable slab exhaustion detected\n", exhausted ? "PASS" : "FAIL") ; all = all && exhausted ; pool.free(a) ; try {
        d = pool.alloc_bytes(3ULL << 20, Tier::MUTABLE) ; bool t2 = (d == a) ; std::printf("[%s] reclaimed block reused after free\n", t2 ? "PASS" : "FAIL") ; all = all && t2 ; } catch (...) {
        std::printf("[FAIL] reclaimed block reused after free\n") ; return 1 ; }

    pool.free(b) ; pool.free(d) ; auto s = pool.stats() ; bool t3 = (s.mutable_used == 0) ; std::printf("[%s] mutable live usage returns to zero\n", t3 ? "PASS" : "FAIL") ; all = all && t3 ; std::printf("\n%s\n", all ? "POOL RECLAIM TEST PASSED" : "POOL RECLAIM TEST FAILED") ; return all ? 0 : 1 ; }
