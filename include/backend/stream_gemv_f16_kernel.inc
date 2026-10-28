// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF — Streaming GEMV, f16 weights  (stream_gemv_f16_kernel.inc)
// GEMV with WEIGHTS KEPT AS FLOAT16 IN RAM.
//
// Included twice by stream_gemv.hpp (AVX2+FMA / AVX1+F16C), parameterised by
//     VLRF_SG16_NAME, VLRF_SG16_ATTR, VLRF_SG_FMA
//
// y[n] += alpha * dot(W16[n, 0..K), x[0..K))       W16: uint16_t (IEEE half)
//
// Decode is DRAM-bandwidth bound, so the lever that actually moves the needle
// is bytes per token, not latency tricks: f16 weights stream half the bytes of
// f32. Conversion is one vcvtph2ps per 8 weights (F16C, Ivy Bridge+ / Zen1+),
// which is hidden entirely behind the memory stall. Accumulation stays f32, so
// numerics match what you get today by dequantizing f16 -> f32 at load time.
//
// Same double-buffer window as the f32 kernel (next group -> L2, short range
// -> L1), one prefetch per 64B line per row.
// =============================================================================
VLRF_SG16_ATTR
inline void VLRF_SG16_NAME(float const* __restrict x, uint16_t const* __restrict W,
                         float* __restrict y, int K, int N, size_t ldw, float alpha)
{
#define VLRF_SG16_LD(p) _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)))
    const int K16 = K & ~15;
    int n = 0;
    for (; n + 4 <= N; n += 4) {
        uint16_t const* w0 = W + (size_t)n * ldw;
        uint16_t const* w1 = w0 + ldw;
        uint16_t const* w2 = w1 + ldw;
        uint16_t const* w3 = w2 + ldw;
        __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
        __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
        __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
        __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
        int k = 0;
        // Main loop: 32 halves/row/iteration == exactly one 64B cache line per
        // row, so one prefetch per row per stage with no duplicate lines.
        //   stage 2 : next group (4 rows ahead)  -> L2
        //   stage 1 : short range                -> L1
        const char* nx0 = reinterpret_cast<const char*>(
            reinterpret_cast<uintptr_t>(w0) + 4 * ldw * sizeof(uint16_t));
        const char* nx1 = nx0 + ldw * sizeof(uint16_t);
        const char* nx2 = nx1 + ldw * sizeof(uint16_t);
        const char* nx3 = nx2 + ldw * sizeof(uint16_t);
        for (; k + 32 <= K; k += 32) {
            _mm_prefetch(nx0 + (size_t)k * 2, _MM_HINT_T1);
            _mm_prefetch(nx1 + (size_t)k * 2, _MM_HINT_T1);
            _mm_prefetch(nx2 + (size_t)k * 2, _MM_HINT_T1);
            _mm_prefetch(nx3 + (size_t)k * 2, _MM_HINT_T1);
            _mm_prefetch(reinterpret_cast<const char*>(w0 + k + 512), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(w1 + k + 512), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(w2 + k + 512), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(w3 + k + 512), _MM_HINT_T0);
            for (int u = 0; u < 32; u += 16) {
                const __m256 x0 = _mm256_loadu_ps(x + k + u);
                const __m256 x1 = _mm256_loadu_ps(x + k + u + 8);
                c00 = VLRF_SG_FMA(x0, VLRF_SG16_LD(w0 + k + u),     c00);
                c01 = VLRF_SG_FMA(x1, VLRF_SG16_LD(w0 + k + u + 8), c01);
                c10 = VLRF_SG_FMA(x0, VLRF_SG16_LD(w1 + k + u),     c10);
                c11 = VLRF_SG_FMA(x1, VLRF_SG16_LD(w1 + k + u + 8), c11);
                c20 = VLRF_SG_FMA(x0, VLRF_SG16_LD(w2 + k + u),     c20);
                c21 = VLRF_SG_FMA(x1, VLRF_SG16_LD(w2 + k + u + 8), c21);
                c30 = VLRF_SG_FMA(x0, VLRF_SG16_LD(w3 + k + u),     c30);
                c31 = VLRF_SG_FMA(x1, VLRF_SG16_LD(w3 + k + u + 8), c31);
            }
        }
        for (; k < K16; k += 16) {   // 16-element remainder (no prefetch)
            const __m256 x0 = _mm256_loadu_ps(x + k);
            const __m256 x1 = _mm256_loadu_ps(x + k + 8);
            c00 = VLRF_SG_FMA(x0, VLRF_SG16_LD(w0 + k),     c00);
            c01 = VLRF_SG_FMA(x1, VLRF_SG16_LD(w0 + k + 8), c01);
            c10 = VLRF_SG_FMA(x0, VLRF_SG16_LD(w1 + k),     c10);
            c11 = VLRF_SG_FMA(x1, VLRF_SG16_LD(w1 + k + 8), c11);
            c20 = VLRF_SG_FMA(x0, VLRF_SG16_LD(w2 + k),     c20);
            c21 = VLRF_SG_FMA(x1, VLRF_SG16_LD(w2 + k + 8), c21);
            c30 = VLRF_SG_FMA(x0, VLRF_SG16_LD(w3 + k),     c30);
            c31 = VLRF_SG_FMA(x1, VLRF_SG16_LD(w3 + k + 8), c31);
        }
        __m256 s0 = _mm256_add_ps(c00, c01), s1 = _mm256_add_ps(c10, c11);
        __m256 s2 = _mm256_add_ps(c20, c21), s3 = _mm256_add_ps(c30, c31);
        if (k + 8 <= K) {
            const __m256 x0 = _mm256_loadu_ps(x + k);
            s0 = VLRF_SG_FMA(x0, VLRF_SG16_LD(w0 + k), s0);
            s1 = VLRF_SG_FMA(x0, VLRF_SG16_LD(w1 + k), s1);
            s2 = VLRF_SG_FMA(x0, VLRF_SG16_LD(w2 + k), s2);
            s3 = VLRF_SG_FMA(x0, VLRF_SG16_LD(w3 + k), s3);
            k += 8;
        }
        __m256 t0 = _mm256_hadd_ps(s0, s1);
        __m256 t1 = _mm256_hadd_ps(s2, s3);
        __m256 t2 = _mm256_hadd_ps(t0, t1);
        __m128 r  = _mm_add_ps(_mm256_castps256_ps128(t2), _mm256_extractf128_ps(t2, 1));
        if (k < K) {
            alignas(16) float tmp[4];
            _mm_store_ps(tmp, r);
            for (; k < K; ++k) {
                const float xv = x[k];
                tmp[0] += xv * _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(w0[k])));
                tmp[1] += xv * _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(w1[k])));
                tmp[2] += xv * _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(w2[k])));
                tmp[3] += xv * _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(w3[k])));
            }
            r = _mm_load_ps(tmp);
        }
        __m128 yo = _mm_loadu_ps(y + n);
        _mm_storeu_ps(y + n, _mm_add_ps(yo, _mm_mul_ps(r, _mm_set1_ps(alpha))));
    }
    for (; n < N; ++n) {
        uint16_t const* w = W + (size_t)n * ldw;
        __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
        int k = 0;
        for (; k + 16 <= K; k += 16) {
            c0 = VLRF_SG_FMA(_mm256_loadu_ps(x + k),     VLRF_SG16_LD(w + k),     c0);
            c1 = VLRF_SG_FMA(_mm256_loadu_ps(x + k + 8), VLRF_SG16_LD(w + k + 8), c1);
        }
        for (; k + 8 <= K; k += 8)
            c0 = VLRF_SG_FMA(_mm256_loadu_ps(x + k), VLRF_SG16_LD(w + k), c0);
        __m256 s = _mm256_add_ps(c0, c1);
        __m128 h = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
        h = _mm_hadd_ps(h, h);
        h = _mm_hadd_ps(h, h);
        float sum = _mm_cvtss_f32(h);
        for (; k < K; ++k)
            sum += x[k] * _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(w[k])));
        y[n] += alpha * sum;
    }
#undef VLRF_SG16_LD
}
