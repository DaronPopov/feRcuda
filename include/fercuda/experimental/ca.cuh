#pragma once
/*
 * feRcuda :: ca.cuh
 *
 * Cellular-automata inter-thread memory token bus.
 *
 * CONCEPT
 * ───────
 * Each thread in a block is a "cell" carrying:
 *   pressure  — accumulated need; grows each step while token-less, decays on receipt
 *   rank      — static priority baked in at launch (higher = stronger pull)
 *   tokens    — memory-access tokens currently held (0 .. max_tokens)
 *   flags     — structural bits: warp_leader, block_leader
 *
 * One call to CABus::step() runs one CA generation over shared memory:
 *
 *   Phase 0  Pressure update — token-less cells that need a token accumulate pressure.
 *   Phase 1  Intent — each token-holding cell signals which neighbor it wants
 *            to give to, based on an "effective pressure" score that blends
 *            raw pressure with rank (rank_weight controls the blend).
 *   Phase 2  Apply — non-conflicting transfers execute; tokens are conserved.
 *            Pressure decays when a token arrives.
 *
 * ALIGNMENT GATING
 * ────────────────
 * The boundary between thread i and thread i+1 has an alignment level:
 *   level = __builtin_ctz(i + 1)   (number of trailing zeros, 0..5+)
 *   level 0 = every adjacent pair        (most permeable)
 *   level 5 = only warp boundaries       (most restrictive)
 *
 * align_gate is the MINIMUM level required for a boundary to be passable.
 *   align_gate=0  → all boundaries open (free flow)
 *   align_gate=5  → tokens can only cross at warp boundaries (warp leaders relay)
 *   align_gate=3  → tokens cross at multiples of 8, not freely within
 *
 * RULE ENCODING  (uint32_t, Wolfram-inspired)
 * ────────────────────────────────────────────
 *   bits  3:0   pressure_thresh  (×16 → 0, 16, 32 … 240)
 *               min effective-pressure delta to trigger a transfer
 *   bits  7:4   rank_weight      (0 = rank ignored; 15 = rank dominant)
 *               effective_pressure = pressure + rank * rank_weight / 15
 *   bits 11:8   align_gate       (0-5; see above)
 *   bits 15:12  pressure_decay   (×16 subtracted from pressure on receipt)
 *   bits 19:16  max_tokens_m1    (max tokens per cell = field + 1)
 *   bits 31:20  reserved
 *
 * PRESET RULES
 *   RULE_FREE_FLOW     0x0001'0031   threshold=1×16, rank_weight=3, gate=0, decay=1×16, max=2
 *   RULE_WARP_LOCAL    0x0000'5011   threshold=1×16, rank_weight=1, gate=5, decay=0,    max=1
 *   RULE_RANK_DOMINANT 0x0001'00F1   threshold=1×16, rank_weight=15,gate=0, decay=1×16, max=2
 *   RULE_PRESSURE_PRI  0x0001'0001   threshold=0,    rank_weight=0, gate=0, decay=1×16, max=1
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace fer::ca {

// ─── Cell State (4 bytes, 4-byte aligned) ─────────────────────────────────────

struct alignas(4) CellState {
    uint8_t pressure;   // 0-255: accumulated need
    uint8_t rank;       // 0-255: static priority
    uint8_t tokens;     // 0-max: tokens held
    uint8_t flags;      // bit0=warp_leader, bit1=block_leader

    __host__ __device__ bool is_warp_leader()  const { return (flags & 0x01) != 0; }
    __host__ __device__ bool is_block_leader() const { return (flags & 0x02) != 0; }
};

// ─── Rule Parameters ──────────────────────────────────────────────────────────

struct RuleParams {
    uint8_t pressure_thresh;   // effective-pressure delta to allow transfer
    uint8_t rank_weight;       // 0-15: rank influence
    uint8_t align_gate;        // min boundary level to pass
    uint8_t pressure_decay;    // pressure subtracted on token receipt
    uint8_t max_tokens;        // max tokens per cell

    static constexpr __host__ __device__ RuleParams decode(uint32_t r) {
        return RuleParams{
            /* pressure_thresh */ uint8_t(( r        & 0xF) << 4),
            /* rank_weight     */ uint8_t(((r >>  4) & 0xF)),
            /* align_gate      */ uint8_t(((r >>  8) & 0xF)),
            /* pressure_decay  */ uint8_t(((r >> 12) & 0xF) << 4),
            /* max_tokens      */ uint8_t( ((r >> 16) & 0xF) + 1),
        };
    }

    // Effective pressure: blends raw pressure with rank-based bonus
    __device__ uint16_t effective(const CellState& c) const {
        return uint16_t(c.pressure) + uint16_t(c.rank) * rank_weight / 15u;
    }

    // Is the boundary between thread i and i+1 passable under this rule?
    // alignment level = number of trailing zeros in (i+1), computed bit-by-bit
    // so it works in both __host__ and __device__ contexts.
    __host__ __device__ bool boundary_passable(int i) const {
        if (align_gate == 0) return true;
        unsigned v = (unsigned)(i + 1);
        int level = 0;
        while (level < 6 && (v & 1u) == 0) { v >>= 1; level++; }
        return level >= (int)align_gate;
    }
};

// ─── Preset Rule IDs ──────────────────────────────────────────────────────────
//
// Decode: thresh=(bits3:0)×16, rw=bits7:4, gate=bits11:8, decay=(bits15:12)×16, max=bits19:16+1
//

//  Rank flow: thresh=0, rank_weight=15, gate=0, decay=0, max=1
//  Tokens flow immediately toward the highest-rank neighbor. No threshold barrier.
//  Best for rank-ordered arbitration and directional flow demos.
static constexpr uint32_t RULE_RANK_FLOW     = 0x0000'00F0u;

//  Free flow: thresh=16, rank_weight=3, gate=0, decay=16, max=2
//  Moderate rank influence; tokens require pressure buildup before moving.
static constexpr uint32_t RULE_FREE_FLOW     = 0x0001'0031u;

//  Warp-local: thresh=16, rank_weight=1, gate=5, decay=0, max=1
//  Tokens can only cross at warp boundaries (every 32 threads).
//  Within a warp, tokens are pinned unless the intra-warp boundary is at level>=5.
static constexpr uint32_t RULE_WARP_LOCAL    = 0x0000'5011u;

//  Rank dominant: thresh=16, rank_weight=15, gate=0, decay=16, max=2
//  Rank strongly determines who wins; pressure still required to start movement.
static constexpr uint32_t RULE_RANK_DOMINANT = 0x0001'00F1u;

//  Gated stride-2: thresh=0, rank_weight=15, gate=1, decay=0, max=1
//  Tokens can only cross even-numbered boundaries (level>=1).
//  Odd-thread boundaries are impassable — tokens skip over odd positions.
static constexpr uint32_t RULE_STRIDE2_GATE  = 0x0000'01F0u;

// ─── CA Bus ───────────────────────────────────────────────────────────────────
//
// Declare in shared memory inside a kernel:
//   __shared__ fer::ca::CABus<128, fer::ca::RULE_FREE_FLOW> bus;
//
// All THREADS threads must call init() then step() simultaneously.
// THREADS must equal blockDim.x (or be a leading contiguous subset).
//
template<int THREADS, uint32_t RULE = RULE_FREE_FLOW>
struct CABus {

    CellState cur[THREADS];     // current cell states
    int8_t    intent[THREADS];  // per-thread transfer intent: +1=give-right, -1=give-left, 0=hold

    // ── init ──────────────────────────────────────────────────────────────────
    // Call once before first step(). All threads must participate.
    __device__ void init(uint8_t rank, uint8_t initial_tokens = 0) {
        const int tid = threadIdx.x;
        if (tid < THREADS) {
            cur[tid].pressure = 0;
            cur[tid].rank     = rank;
            cur[tid].tokens   = initial_tokens;
            cur[tid].flags    = 0;
            if ((tid & 31) == 0) cur[tid].flags |= 0x01;   // warp leader
            if (tid == 0)        cur[tid].flags |= 0x02;   // block leader
            intent[tid]       = 0;
        }
        __syncthreads();
    }

    // ── step ──────────────────────────────────────────────────────────────────
    // One CA generation.  Pass needs_token=true for threads that want a token.
    // Token-less threads that need a token accumulate pressure each step.
    //
    // Transfer rule (per adjacent pair i | i+1):
    //   Score(src → dst) = effective(dst) - effective(src) - pressure_thresh
    //   If score > 0 and src has tokens and dst has room and boundary passable:
    //     src intends to give dst.
    //   Conflicts (both cells trying to give away simultaneously) are dropped.
    //   Tokens are clamped to [0, max_tokens]; excess is silently absorbed
    //   (models natural attrition at saturated cells).
    //
    __device__ void step(bool needs_token) {
        constexpr RuleParams rp = RuleParams::decode(RULE);
        const int tid = threadIdx.x;
        if (tid >= THREADS) return;

        // ── Phase 0: pressure accumulation ────────────────────────────────────
        if (needs_token && cur[tid].tokens == 0) {
            if (cur[tid].pressure < 255) cur[tid].pressure++;
        }
        __syncthreads();

        // ── Phase 1: compute transfer intent ──────────────────────────────────
        intent[tid] = 0;
        if (cur[tid].tokens > 0) {
            const uint16_t self_eff = rp.effective(cur[tid]);

            // Prefer right neighbor first (arbitrary tie-break; right = higher tid)
            if (tid + 1 < THREADS && rp.boundary_passable(tid)) {
                const uint16_t right_eff = rp.effective(cur[tid + 1]);
                if (cur[tid + 1].tokens < rp.max_tokens &&
                    right_eff > uint16_t(self_eff + rp.pressure_thresh)) {
                    intent[tid] = +1;
                }
            }
            // Left neighbor (only if right wasn't chosen)
            if (intent[tid] == 0 && tid > 0 && rp.boundary_passable(tid - 1)) {
                const uint16_t left_eff = rp.effective(cur[tid - 1]);
                if (cur[tid - 1].tokens < rp.max_tokens &&
                    left_eff > uint16_t(self_eff + rp.pressure_thresh)) {
                    intent[tid] = -1;
                }
            }
        }
        __syncthreads();

        // ── Phase 2: resolve and apply ────────────────────────────────────────
        // Transfer i→i+1 fires when:
        //   intent[i] == +1  AND  intent[i+1] != -1  (no counter-move conflict)
        // Transfer i+1→i fires when:
        //   intent[i+1] == -1  AND  intent[i] != +1
        // This prevents simultaneous swap (which would be a no-op but wastes cycles).

        const int8_t my_intent = intent[tid];

        const bool give_right = (my_intent == +1) &&
                                (tid + 1 < THREADS) &&
                                (intent[tid + 1] != -1);

        const bool give_left  = (my_intent == -1) &&
                                (tid > 0) &&
                                (intent[tid - 1] != +1);

        const bool recv_from_left  = (tid > 0) &&
                                     (intent[tid - 1] == +1) &&
                                     (my_intent != -1);   // not simultaneously giving left

        const bool recv_from_right = (tid + 1 < THREADS) &&
                                     (intent[tid + 1] == -1) &&
                                     (my_intent != +1);   // not simultaneously giving right

        int delta = 0;
        if (give_right)     delta--;
        if (give_left)      delta--;
        if (recv_from_left) delta++;
        if (recv_from_right)delta++;

        int new_tok = int(cur[tid].tokens) + delta;
        new_tok = max(0, min(int(rp.max_tokens), new_tok));
        cur[tid].tokens = uint8_t(new_tok);

        // Pressure decay on receipt (saturating subtract)
        if (recv_from_left || recv_from_right) {
            cur[tid].pressure = uint8_t(
                int(cur[tid].pressure) >= int(rp.pressure_decay)
                    ? int(cur[tid].pressure) - int(rp.pressure_decay)
                    : 0);
        }
        __syncthreads();
    }

    // ── Accessors ─────────────────────────────────────────────────────────────
    __device__ uint8_t my_tokens()   const { return cur[threadIdx.x].tokens;   }
    __device__ uint8_t my_pressure() const { return cur[threadIdx.x].pressure; }
    __device__ bool    has_token()   const { return my_tokens() > 0; }

    // Manually inject pressure (e.g. from an external priority signal)
    __device__ void boost_pressure(uint8_t amount) {
        const int tid = threadIdx.x;
        if (tid < THREADS) {
            uint16_t p = uint16_t(cur[tid].pressure) + amount;
            cur[tid].pressure = uint8_t(p > 255u ? 255u : p);
        }
    }

    // Inject tokens at a specific cell (only that thread executes safely)
    __device__ void deposit(int cell_idx, uint8_t count = 1) {
        constexpr RuleParams rp = RuleParams::decode(RULE);
        const int tid = threadIdx.x;
        if (tid == cell_idx && cell_idx < THREADS) {
            int t = int(cur[cell_idx].tokens) + count;
            cur[cell_idx].tokens = uint8_t(t > rp.max_tokens ? rp.max_tokens : t);
        }
    }

    // Snapshot one cell's state (call from any thread after __syncthreads)
    __device__ CellState snapshot(int cell_idx) const {
        return cur[cell_idx];
    }
};

} // namespace fer::ca
