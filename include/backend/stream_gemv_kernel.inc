// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF — Streaming GEMV micro-kernel  (stream_gemv_kernel.inc)
// Body of the streaming GEMV micro-kernel.
//
// NOT a standalone header. Included (twice) by stream_gemv.hpp with:
//     VLRF_SG_NAME   name of the generated function
//     VLRF_SG_ATTR   function attribute (target("avx2,fma") or empty)
//     VLRF_SG_FMA    fused multiply-add macro  (a*b + c)
//
// Why a .inc and not a template: GCC/Clang need `target("avx2,fma")` on the
// AVX2 function and MUST NOT have it on the AVX1 one (otherwise -ffp-contract
// may legally emit FMA inside the "AVX1" kernel and it would SIGILL on
// Ivy/Sandy Bridge). A template cannot carry the attribute per-instantiation.
//
// Computes   y[n] += alpha * dot(W[n, 0..K), x[0..K))      for n in [0, N)
// W is the weight matrix exactly as stored in the model: [N, K] row-major,
// row stride `ldw` floats.  (== "x @ W^T", no transpose copy needed.)
//
// Template parameters
//   BUF  1 = single buffering : only the short-range L1 stage (below)
//        2 = double buffering : + "next group" window streamed into L2
//        3 = triple buffering : + "group after next" window streamed into L3
//   L1H  0 = no L1 stage, 1 = prefetcht0, 2 = prefetchnta
//
// Terminology (Cell/BE -> x86):
//   * A "group" is 4 consecutive weight rows (4*K floats). It is the unit being
//     computed on — the "compute buffer".
//   * While group g is being computed, the cache lines of group g+1 are
//     prefetched into L2 (hint T1) — the "DMA-in-flight buffer".
//   * Group g+2 is prefetched into L3 (hint T2) — the third buffer.
//   Prefetches are issued *interleaved* with compute (one line per row per
//   16-float step), so the memory system always sees a smooth stream instead
//   of a burst that would overflow the line-fill / superqueue buffers.
// =============================================================================
template <int BUF, int L1H>
VLRF_SG_ATTR
inline void VLRF_SG_NAME(float const* __restrict x, float const* __restrict W,
                       float* __restrict y, int K, int N, size_t ldw,
                       float alpha, int d1_floats)
{
    const int K16 = K & ~15;
    const size_t g1 = 4 * ldw;   // distance (floats) to next group
    const size_t g2 = 8 * ldw;   // distance to group after next

    int n = 0;
    for (; n + 4 <= N; n += 4) {
        float const* w0 = W + (size_t)n * ldw;
        float const* w1 = w0 + ldw;
        float const* w2 = w1 + ldw;
        float const* w3 = w2 + ldw;

        // Prefetch base pointers for the look-ahead windows. These may point
        // past the end of the slice on the last groups: prefetch never faults,
        // and the arithmetic is done on uintptr_t to avoid pointer-overflow UB.
        auto adv = [](float const* p, size_t floats) -> const char* {
            return reinterpret_cast<const char*>(
                reinterpret_cast<uintptr_t>(p) + floats * sizeof(float));
        };
        const char* a1_0 = adv(w0, g1); const char* a1_1 = adv(w1, g1);
        const char* a1_2 = adv(w2, g1); const char* a1_3 = adv(w3, g1);
        const char* a2_0 = adv(w0, g2); const char* a2_1 = adv(w1, g2);
        const char* a2_2 = adv(w2, g2); const char* a2_3 = adv(w3, g2);
        (void)a1_0; (void)a1_1; (void)a1_2; (void)a1_3;
        (void)a2_0; (void)a2_1; (void)a2_2; (void)a2_3;

        // 4 rows x 2 accumulators = 8 independent FMA chains: enough to cover
        // FMA latency (4c) x 2 ports. (The old kernel had ONE chain per row.)
        __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
        __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
        __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();

        int k = 0;
        for (; k < K16; k += 16) {
            // ---- stage 3: group g+2 -> L3 ---------------------------------
            if constexpr (BUF >= 3) {
                _mm_prefetch(a2_0 + (size_t)k * 4, _MM_HINT_T2);
                _mm_prefetch(a2_1 + (size_t)k * 4, _MM_HINT_T2);
                _mm_prefetch(a2_2 + (size_t)k * 4, _MM_HINT_T2);
                _mm_prefetch(a2_3 + (size_t)k * 4, _MM_HINT_T2);
            }
            // ---- stage 2: group g+1 -> L2 ---------------------------------
            if constexpr (BUF >= 2) {
                _mm_prefetch(a1_0 + (size_t)k * 4, _MM_HINT_T1);
                _mm_prefetch(a1_1 + (size_t)k * 4, _MM_HINT_T1);
                _mm_prefetch(a1_2 + (size_t)k * 4, _MM_HINT_T1);
                _mm_prefetch(a1_3 + (size_t)k * 4, _MM_HINT_T1);
            }
            // ---- stage 1: short range, into L1 ----------------------------
            // Lands in the next row's early offsets when k+d1 >= K, which is
            // a cheap L1 hit; the cross-group case is covered by stage 2.
            if constexpr (L1H != 0) {
                _mm_prefetch(reinterpret_cast<const char*>(w0 + k + d1_floats),
                             (L1H == 1) ? _MM_HINT_T0 : _MM_HINT_NTA);
                _mm_prefetch(reinterpret_cast<const char*>(w1 + k + d1_floats),
                             (L1H == 1) ? _MM_HINT_T0 : _MM_HINT_NTA);
                _mm_prefetch(reinterpret_cast<const char*>(w2 + k + d1_floats),
                             (L1H == 1) ? _MM_HINT_T0 : _MM_HINT_NTA);
                _mm_prefetch(reinterpret_cast<const char*>(w3 + k + d1_floats),
                             (L1H == 1) ? _MM_HINT_T0 : _MM_HINT_NTA);
            }

            const __m256 x0 = _mm256_loadu_ps(x + k);
            const __m256 x1 = _mm256_loadu_ps(x + k + 8);
            c00 = VLRF_SG_FMA(x0, _mm256_loadu_ps(w0 + k),     c00);
            c01 = VLRF_SG_FMA(x1, _mm256_loadu_ps(w0 + k + 8), c01);
            c10 = VLRF_SG_FMA(x0, _mm256_loadu_ps(w1 + k),     c10);
            c11 = VLRF_SG_FMA(x1, _mm256_loadu_ps(w1 + k + 8), c11);
            c20 = VLRF_SG_FMA(x0, _mm256_loadu_ps(w2 + k),     c20);
            c21 = VLRF_SG_FMA(x1, _mm256_loadu_ps(w2 + k + 8), c21);
            c30 = VLRF_SG_FMA(x0, _mm256_loadu_ps(w3 + k),     c30);
            c31 = VLRF_SG_FMA(x1, _mm256_loadu_ps(w3 + k + 8), c31);
        }
        __m256 s0 = _mm256_add_ps(c00, c01);
        __m256 s1 = _mm256_add_ps(c10, c11);
        __m256 s2 = _mm256_add_ps(c20, c21);
        __m256 s3 = _mm256_add_ps(c30, c31);

        if (k + 8 <= K) {                       // one leftover 8-wide step
            const __m256 x0 = _mm256_loadu_ps(x + k);
            s0 = VLRF_SG_FMA(x0, _mm256_loadu_ps(w0 + k), s0);
            s1 = VLRF_SG_FMA(x0, _mm256_loadu_ps(w1 + k), s1);
            s2 = VLRF_SG_FMA(x0, _mm256_loadu_ps(w2 + k), s2);
            s3 = VLRF_SG_FMA(x0, _mm256_loadu_ps(w3 + k), s3);
            k += 8;
        }

        // 4-way horizontal reduction: lanes -> [y0 y1 y2 y3]
        __m256 t0 = _mm256_hadd_ps(s0, s1);
        __m256 t1 = _mm256_hadd_ps(s2, s3);
        __m256 t2 = _mm256_hadd_ps(t0, t1);
        __m128 r  = _mm_add_ps(_mm256_castps256_ps128(t2),
                               _mm256_extractf128_ps(t2, 1));

        if (k < K) {                            // scalar tail (< 8 elements)
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, r);
            for (; k < K; ++k) {
                const float xv = x[k];
                tmp[0] += xv * w0[k]; tmp[1] += xv * w1[k];
                tmp[2] += xv * w2[k]; tmp[3] += xv * w3[k];
            }
            r = _mm_load_ps(tmp);
        }
        __m128 yo = _mm_loadu_ps(y + n);
        yo = _mm_add_ps(yo, _mm_mul_ps(r, _mm_set1_ps(alpha)));
        _mm_storeu_ps(y + n, yo);
    }

    // Leftover rows (N % 4): one row at a time, 4 accumulators to hide latency.
    for (; n < N; ++n) {
        float const* w = W + (size_t)n * ldw;
        __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
        __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
        int k = 0;
        const int K32 = K & ~31;
        for (; k < K32; k += 32) {
            c0 = VLRF_SG_FMA(_mm256_loadu_ps(x + k),      _mm256_loadu_ps(w + k),      c0);
            c1 = VLRF_SG_FMA(_mm256_loadu_ps(x + k + 8),  _mm256_loadu_ps(w + k + 8),  c1);
            c2 = VLRF_SG_FMA(_mm256_loadu_ps(x + k + 16), _mm256_loadu_ps(w + k + 16), c2);
            c3 = VLRF_SG_FMA(_mm256_loadu_ps(x + k + 24), _mm256_loadu_ps(w + k + 24), c3);
        }
        for (; k + 8 <= K; k += 8)
            c0 = VLRF_SG_FMA(_mm256_loadu_ps(x + k), _mm256_loadu_ps(w + k), c0);
        __m256 s = _mm256_add_ps(_mm256_add_ps(c0, c1), _mm256_add_ps(c2, c3));
        __m128 h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
        h = _mm_hadd_ps(h, h);
        h = _mm_hadd_ps(h, h);
        float sum = _mm_cvtss_f32(h);
        for (; k < K; ++k) sum += x[k] * w[k];
        y[n] += alpha * sum;
    }
}
