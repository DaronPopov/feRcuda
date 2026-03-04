#include "fercuda/alloc/memory.cuh"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

using namespace fer ; struct LiveAlloc {
    void* ptr = nullptr ; size_t bytes = 0 ; } ; struct LatencySummary {
    double mean_us = 0.0 ; double p50_us = 0.0 ; double p95_us = 0.0 ; double p99_us = 0.0 ; double max_us = 0.0 ; } ; struct ProfileResult {
    bool ok = false ; int alloc_ok = 0 ; int alloc_fail = 0 ; int frees = 0 ; size_t post_churn_used = 0 ; size_t post_big_used = 0 ; LatencySummary alloc_lat{} ; LatencySummary free_lat{} ; } ; static LatencySummary summarize(std::vector<double> v) {
    LatencySummary s{} ; if (v.empty()) return s ; std::sort(v.begin(), v.end()) ; const auto pick = [&](double q) {
        size_t idx = static_cast<size_t>(q * static_cast<double>(v.size() - 1)) ; return v[idx] ; } ; s.mean_us = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()) ; s.p50_us = pick(0.50) ; s.p95_us = pick(0.95) ; s.p99_us = pick(0.99) ; s.max_us = v.back() ; return s ; }

static bool free_random(std::mt19937& rng,
                        ElasticPool& pool,
                        std::vector<LiveAlloc>& live,
                        size_t* live_bytes,
                        int* frees,
                        std::vector<double>* free_lat_us) {
    if (live.empty()) return false ; std::uniform_int_distribution<size_t> idx_dist(0, live.size() - 1) ; size_t idx = idx_dist(rng) ; auto t0 = std::chrono::steady_clock::now() ; pool.free(live[idx].ptr) ; auto t1 = std::chrono::steady_clock::now() ; free_lat_us->push_back(std::chrono::duration<double, std::micro>(t1 - t0).count()) ; *live_bytes -= live[idx].bytes ; live[idx] = live.back() ; live.pop_back() ; ++(*frees) ; return true ; }

static ProfileResult run_profile(const char* name,
                                 MemoryRegime regime,
                                 uint32_t seed,
                                 int iters,
                                 bool high_pressure) {
    constexpr size_t kMiB = 1ULL << 20 ; constexpr size_t kMutableCap = 64ULL * kMiB ; PoolConfig cfg{} ; cfg.mutable_bytes = kMutableCap ; cfg.immutable_bytes = 8ULL * kMiB ; cfg.regime = regime ; ElasticPool pool(0, cfg) ; std::mt19937 rng(seed) ; std::uniform_int_distribution<int> op_dist(0, 99) ; std::uniform_int_distribution<int> size_kib_dist(4, 512) ; std::vector<LiveAlloc> live ; live.reserve(4096) ; size_t live_bytes = 0 ; ProfileResult r{} ; std::vector<double> alloc_lat_us ; std::vector<double> free_lat_us ; alloc_lat_us.reserve(static_cast<size_t>(iters)) ; free_lat_us.reserve(static_cast<size_t>(iters)) ; const size_t pressure_low = (kMutableCap * 85) / 100 ; const size_t pressure_high = (kMutableCap * 95) / 100 ; for (int i = 0 ; i < iters ; ++i) {
        bool do_alloc = false ; if (live.empty()) {
            do_alloc = true ; } else if (!high_pressure) {
            do_alloc = (op_dist(rng) < 60) ; } else {
            if (live_bytes < pressure_low) do_alloc = true ; else if (live_bytes > pressure_high) do_alloc = false ; else do_alloc = (op_dist(rng) < 55) ; }

        if (do_alloc) {
            size_t req = static_cast<size_t>(size_kib_dist(rng)) * 1024ULL ; try {
                auto t0 = std::chrono::steady_clock::now() ; void* p = pool.alloc_bytes(req, Tier::MUTABLE) ; auto t1 = std::chrono::steady_clock::now() ; alloc_lat_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count()) ; live.push_back({p, req}) ; live_bytes += req ; ++r.alloc_ok ; } catch (...) {
                ++r.alloc_fail ; free_random(rng, pool, live, &live_bytes, &r.frees, &free_lat_us) ; }
        } else {
            free_random(rng, pool, live, &live_bytes, &r.frees, &free_lat_us) ; }
    }

    for (const auto& a : live) {
        auto t0 = std::chrono::steady_clock::now() ; pool.free(a.ptr) ; auto t1 = std::chrono::steady_clock::now() ; free_lat_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count()) ; }
    live.clear() ; live_bytes = 0 ; auto s0 = pool.stats() ; bool t1 = (s0.mutable_used == 0) ; void* big = nullptr ; bool t2 = true ; try {
        auto t0 = std::chrono::steady_clock::now() ; big = pool.alloc_bytes(48ULL * kMiB, Tier::MUTABLE) ; auto t1b = std::chrono::steady_clock::now() ; alloc_lat_us.push_back(std::chrono::duration<double, std::micro>(t1b - t0).count()) ; } catch (...) {
        t2 = false ; }
    if (big) {
        auto t0 = std::chrono::steady_clock::now() ; pool.free(big) ; auto t1b = std::chrono::steady_clock::now() ; free_lat_us.push_back(std::chrono::duration<double, std::micro>(t1b - t0).count()) ; }

    auto s1 = pool.stats() ; bool t3 = (s1.mutable_used == 0) ; bool t4 = (r.alloc_ok > 0 && r.frees > 0) ; r.ok = t1 && t2 && t3 && t4 ; r.post_churn_used = s0.mutable_used ; r.post_big_used = s1.mutable_used ; r.alloc_lat = summarize(alloc_lat_us) ; r.free_lat = summarize(free_lat_us) ; std::printf("[%s] %s\n", r.ok ? "PASS" : "FAIL", name) ; std::printf("      alloc_ok=%d alloc_fail=%d frees=%d\n", r.alloc_ok, r.alloc_fail, r.frees) ; std::printf("      post_churn_used=%zu post_big_used=%zu\n", r.post_churn_used, r.post_big_used) ; std::printf("      alloc_us mean=%.2f p50=%.2f p95=%.2f p99=%.2f max=%.2f\n",
                r.alloc_lat.mean_us, r.alloc_lat.p50_us, r.alloc_lat.p95_us, r.alloc_lat.p99_us, r.alloc_lat.max_us) ; std::printf("      free_us  mean=%.2f p50=%.2f p95=%.2f p99=%.2f max=%.2f\n",
                r.free_lat.mean_us, r.free_lat.p50_us, r.free_lat.p95_us, r.free_lat.p99_us, r.free_lat.max_us) ; return r ; }

int main() {
    bool all = true ; std::printf("=== CUSTOM_POOL ===\n") ; ProfileResult cp_bal = run_profile("custom balanced", MemoryRegime::CUSTOM_POOL, 0x11112222u, 10000, false) ; ProfileResult cp_hi0 = run_profile("custom high-pressure #1", MemoryRegime::CUSTOM_POOL, 0x22223333u, 20000, true) ; ProfileResult cp_hi1 = run_profile("custom high-pressure #2", MemoryRegime::CUSTOM_POOL, 0x33334444u, 20000, true) ; ProfileResult cp_hi2 = run_profile("custom high-pressure #3", MemoryRegime::CUSTOM_POOL, 0x44445555u, 20000, true) ; all = all && cp_bal.ok && cp_hi0.ok && cp_hi1.ok && cp_hi2.ok ; std::printf("=== CUDA_MALLOC ===\n") ; ProfileResult cm_bal = run_profile("cuda_malloc balanced", MemoryRegime::CUDA_MALLOC, 0x11112222u, 10000, false) ; ProfileResult cm_hi0 = run_profile("cuda_malloc high-pressure #1", MemoryRegime::CUDA_MALLOC, 0x22223333u, 20000, true) ; ProfileResult cm_hi1 = run_profile("cuda_malloc high-pressure #2", MemoryRegime::CUDA_MALLOC, 0x33334444u, 20000, true) ; ProfileResult cm_hi2 = run_profile("cuda_malloc high-pressure #3", MemoryRegime::CUDA_MALLOC, 0x44445555u, 20000, true) ; all = all && cm_bal.ok && cm_hi0.ok && cm_hi1.ok && cm_hi2.ok ; std::printf("\n%s\n", all ? "POOL FRAGMENTATION A/B TEST PASSED" : "POOL FRAGMENTATION A/B TEST FAILED") ; return all ? 0 : 1 ; }
