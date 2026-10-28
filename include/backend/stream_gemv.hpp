// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — Streaming GEMV with software multi-buffering (decode hot path)
// (stream_gemv.hpp)
//
// During autoregressive decode every weight matrix is read exactly ONCE per
// token and never reused, so the GEMV (M=1) is purely DRAM-bandwidth bound.
// There is nothing to "block for reuse"; the only job is to keep the memory
// system saturated while the cores do their (tiny) amount of math.
//
// Cell/BE mapping. The SPE hid DRAM latency with explicit DMA into a local
// store (double/triple buffering). x86 has no local store and no DMA engine,
// but the same pipeline can be expressed with software prefetch:
//
//     Cell/BE                        x86 (this file)
//     ------------------------       -------------------------------------
//     buffer being computed          group g   (4 rows)       -> L1 / L2
//     buffer being DMA'd in          group g+1                -> L2 (T1 hint)
//     third buffer (triple)          group g+2                -> L3 (T2 hint)
//     DMA tag wait                   (none — OoO + prefetch are non-blocking)
//     NTA / bypass                   prefetchnta for the L1 stage: weights are
//                                    used once, don't evict x / accumulators
//
// Where this helps (and where it doesn't) — see the benchmark:
//   * The HW L2 streamer does not cross 4 KiB page boundaries and tracks a
//     limited number of streams; SW prefetch is not bound by either.
//   * It cannot beat the DRAM roofline. If a thread already saturates the
//     memory controller, extra prefetch buys nothing. Measure (bench tool).
//   * Biggest wins in this change are NOT the prefetches but: (1) no per-token
//     weight transpose copy, and (2) 8 independent FMA chains instead of 1.
//
// Tuning knobs (runtime, no rebuild):
//   VLRF_STREAM_CFG="B,L,D"   B = 1|2|3 buffering depth
//                           L = 0 none | 1 prefetcht0 | 2 prefetchnta  (L1 stage)
//                           D = L1-stage distance in cache lines
//   VLRF_STREAM_CFG=off       disable this kernel (fall back to the old GEMV)
// =============================================================================
#include "simd_gemm.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace vlrf {
namespace simd {

#if VLRF_X86

// ---- AVX2 + FMA instantiation ----------------------------------------------
#define VLRF_SG_NAME  gemv_t_stream_avx2
#define VLRF_SG_ATTR  VLRF_TARGET_AVX2_FMA
#define VLRF_SG_FMA(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#include "stream_gemv_kernel.inc"
#undef VLRF_SG_NAME
#undef VLRF_SG_ATTR
#undef VLRF_SG_FMA

// ---- AVX1 instantiation (no FMA: Sandy/Ivy Bridge) -------------------------
#define VLRF_SG_NAME  gemv_t_stream_avx1
#define VLRF_SG_ATTR
#define VLRF_SG_FMA(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#include "stream_gemv_kernel.inc"
#undef VLRF_SG_NAME
#undef VLRF_SG_ATTR
#undef VLRF_SG_FMA

// ---- float16-weight kernels (F16C) -----------------------------------------
// Separate attributes: these need the f16c ISA extension on top of avx / avx2+fma.
#if defined(__GNUC__) || defined(__clang__)
#  define VLRF_TARGET_AVX2_FMA_F16C __attribute__((target("avx2,fma,f16c")))
#  define VLRF_TARGET_AVX_F16C      __attribute__((target("avx,f16c")))
#else
#  define VLRF_TARGET_AVX2_FMA_F16C
#  define VLRF_TARGET_AVX_F16C
#endif

#define VLRF_SG16_NAME gemv_t_stream_f16_avx2
#define VLRF_SG16_ATTR VLRF_TARGET_AVX2_FMA_F16C
#define VLRF_SG_FMA(a, b, c) _mm256_fmadd_ps((a), (b), (c))
#include "stream_gemv_f16_kernel.inc"
#undef VLRF_SG16_NAME
#undef VLRF_SG16_ATTR
#undef VLRF_SG_FMA

#define VLRF_SG16_NAME gemv_t_stream_f16_avx1
#define VLRF_SG16_ATTR VLRF_TARGET_AVX_F16C
#define VLRF_SG_FMA(a, b, c) _mm256_add_ps(_mm256_mul_ps((a), (b)), (c))
#include "stream_gemv_f16_kernel.inc"
#undef VLRF_SG16_NAME
#undef VLRF_SG16_ATTR
#undef VLRF_SG_FMA

inline bool cpu_has_f16c() {
#  if defined(_MSC_VER)
    int info[4]; __cpuid(info, 1);
    return (info[2] & (1 << 29)) != 0;          // ECX bit 29 = F16C
#  else
    return __builtin_cpu_supports("f16c");
#  endif
}

#endif // VLRF_X86

struct StreamCfg {
    bool enabled   {true};
    int  buffering {3};    // 1 = single, 2 = double, 3 = triple
    int  l1_hint   {1};    // 0 none, 1 T0, 2 NTA
    int  l1_lines  {8};    // L1-stage distance, cache lines (16 floats each)
};

inline StreamCfg const& stream_cfg() {
    static StreamCfg cfg = [] {
        StreamCfg c;
        if (const char* e = std::getenv("VLRF_STREAM_CFG")) {
            if (std::strcmp(e, "off") == 0) {
                c.enabled = false;
            } else {
                int b = c.buffering, l = c.l1_hint, d = c.l1_lines;
                if (std::sscanf(e, "%d,%d,%d", &b, &l, &d) >= 1) {
                    c.buffering = b < 1 ? 1 : (b > 3 ? 3 : b);
                    c.l1_hint   = l < 0 ? 0 : (l > 2 ? 2 : l);
                    c.l1_lines  = d < 1 ? 1 : (d > 64 ? 64 : d);
                }
            }
        }
        return c;
    }();
    return cfg;
}

using StreamFn = void (*)(float const*, float const*, float*, int, int,
                          size_t, float, int);

#if VLRF_X86
inline StreamFn select_stream_kernel(bool avx2, StreamCfg const& c) {
    static const StreamFn t2[3][3] = {
        { gemv_t_stream_avx2<1,0>, gemv_t_stream_avx2<1,1>, gemv_t_stream_avx2<1,2> },
        { gemv_t_stream_avx2<2,0>, gemv_t_stream_avx2<2,1>, gemv_t_stream_avx2<2,2> },
        { gemv_t_stream_avx2<3,0>, gemv_t_stream_avx2<3,1>, gemv_t_stream_avx2<3,2> } };
    static const StreamFn t1[3][3] = {
        { gemv_t_stream_avx1<1,0>, gemv_t_stream_avx1<1,1>, gemv_t_stream_avx1<1,2> },
        { gemv_t_stream_avx1<2,0>, gemv_t_stream_avx1<2,1>, gemv_t_stream_avx1<2,2> },
        { gemv_t_stream_avx1<3,0>, gemv_t_stream_avx1<3,1>, gemv_t_stream_avx1<3,2> } };
    return (avx2 ? t2 : t1)[c.buffering - 1][c.l1_hint];
}
#endif

// y[0..N) += alpha * (W[N,K] · x[K])      W row stride = ldw floats.
// `W` is the weight tensor AS STORED ([out, in]); no transpose is needed.
// Single-threaded over the given slice — the caller (ops::gemm) splits N across
// the thread pool, so each thread streams its own contiguous slab of W.
// Returns false if no AVX path is available (caller falls back).
inline bool stream_gemv_t(float const* x, float const* W, float* y,
                          int K, int N, size_t ldw, float alpha)
{
#if VLRF_X86
    StreamCfg const& c = stream_cfg();
    if (!c.enabled) return false;
    auto const& f = cpu_features();
    const bool avx2 = f.avx2 && f.fma;
    if (!avx2 && !f.avx1) return false;
    select_stream_kernel(avx2, c)(x, W, y, K, N, ldw, alpha, c.l1_lines * 16);
    return true;
#else
    (void)x; (void)W; (void)y; (void)K; (void)N; (void)ldw; (void)alpha;
    return false;
#endif
}

// Same contract as stream_gemv_t, but W is IEEE-754 half (uint16_t).
// Returns false when F16C / AVX is unavailable so the caller can fall back to
// a f32 copy of the weights.
inline bool stream_gemv_t_f16(float const* x, uint16_t const* W, float* y,
                              int K, int N, size_t ldw, float alpha)
{
#if VLRF_X86
    auto const& f = cpu_features();
    if (!f.avx1 || !cpu_has_f16c()) return false;
    if (f.avx2 && f.fma) gemv_t_stream_f16_avx2(x, W, y, K, N, ldw, alpha);
    else                 gemv_t_stream_f16_avx1(x, W, y, K, N, ldw, alpha);
    return true;
#else
    (void)x; (void)W; (void)y; (void)K; (void)N; (void)ldw; (void)alpha;
    return false;
#endif
}

} // namespace simd
} // namespace vlrf
