// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — SIMD GEMM Kernels  (simd_gemm.hpp)
//
// Three tiers, selected at compile time and runtime:
//
//   1. ONEMKL   (VLRF_USE_ONEMKL defined)  — Intel oneMKL sgemm, fastest on Intel CPUs
//   2. AVX2/FMA (VLRF_USE_AVX2 or auto-detected) — hand-written 8-wide SIMD kernel
//   3. SCALAR   (always available)        — portable fallback, used by ops.hpp today
//
// Runtime CPU detection means a binary built with AVX2 support still runs
// (falling back to scalar) on CPUs that don't have it — no crash.
//
// Key design choices for the AVX2 kernel:
//   - B is pre-transposed into a packed micro-panel so memory access in the
//     inner loop is fully sequential (fixes the strided-access bottleneck
//     that caused ~30% CPU utilisation in the scalar version).
//   - Micro-kernel computes an 8x8 block of C per inner iteration.
//   - FMA (fused multiply-add) used throughout: 1 instruction instead of 2.
// =============================================================================
#include "tensor.hpp"
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define VLRF_X86 1
#  include <immintrin.h>
#else
#  define VLRF_X86 0
#endif

// -----------------------------------------------------------------------------
// Per-function target attribute for AVX2/FMA functions.
//
// Problem: a single translation unit may be compiled with a baseline flag
// (e.g. -mavx, or no /arch flag on MSVC) for portability, while still
// containing AVX2+FMA-specific functions that are only CALLED when the
// runtime CPU check confirms support. GCC/Clang refuse to compile FMA
// intrinsics unless the FMA target is enabled for that function, even if
// the function is never reached on non-FMA hardware (confirmed by testing:
// -mno-fma + _mm256_fmadd_ps => hard compile error, not a warning).
//
// Fix: tag AVX2/FMA-specific functions with __attribute__((target(...)))
// on GCC/Clang, which compiles JUST that function with the extra ISA
// extensions enabled, independent of the file's baseline -march/-mavx flag.
// MSVC has no equivalent attribute, but doesn't need one: /arch:AVX simply
// fails to define the FMA intrinsics as usable (they're unconditionally
// declared in immintrin.h but most MSVC versions don't reject calling them
// at the chosen /arch level the way GCC does) — and since our MSVC builds
// always select a single global /arch via VLRF_TARGET_ARCH, both AVX2 and
// AVX1 functions compile fine in the same TU on MSVC without attributes.
#if defined(__GNUC__) || defined(__clang__)
#  define VLRF_TARGET_AVX2_FMA __attribute__((target("avx2,fma")))
#else
#  define VLRF_TARGET_AVX2_FMA
#endif

#ifdef VLRF_USE_ONEMKL
#  include <mkl.h>
#endif

namespace vlrf {
namespace simd {

// -----------------------------------------------------------------------------
// Runtime CPU feature detection
// -----------------------------------------------------------------------------
struct CpuFeatures {
    bool avx1    {false};   // Ivy Bridge (2012) and later — 256-bit float ops, no FMA
    bool avx2    {false};   // Haswell (2013) and later — adds integer YMM + gather
    bool fma     {false};   // Haswell and later (fused multiply-add, 3-operand)
    bool avx512f {false};

    static CpuFeatures detect() {
        CpuFeatures f;
#if VLRF_X86
#  if defined(_MSC_VER)
        int info[4];
        __cpuid(info, 0);
        int n_ids = info[0];
        if (n_ids >= 7) {
            __cpuidex(info, 7, 0);
            f.avx2    = (info[1] & (1 << 5))  != 0;
            f.avx512f = (info[1] & (1 << 16)) != 0;
        }
        __cpuid(info, 1);
        f.avx1 = (info[2] & (1 << 28)) != 0;   // ECX bit 28 = AVX
        f.fma  = (info[2] & (1 << 12)) != 0;   // ECX bit 12 = FMA
#  else
        f.avx1    = __builtin_cpu_supports("avx");
        f.avx2    = __builtin_cpu_supports("avx2");
        f.fma     = __builtin_cpu_supports("fma");
        f.avx512f = __builtin_cpu_supports("avx512f");
#  endif
#endif
        return f;
    }
};

inline CpuFeatures const& cpu_features() {
    static CpuFeatures f = CpuFeatures::detect();
    return f;
}

inline const char* active_kernel_name() {
#ifdef VLRF_USE_ONEMKL
    return "oneMKL sgemm";
#else
    auto const& f = cpu_features();
    if (f.avx512f) return "AVX-512 (compiled AVX2, runtime AVX-512 unused - enable VLRF_USE_AVX512)";
    if (f.avx2 && f.fma) return "AVX2+FMA (hand-written 8-wide micro-kernel)";
    if (f.avx1) return "AVX1 (hand-written 8-wide micro-kernel, no FMA - e.g. Ivy Bridge/Sandy Bridge)";
    return "Scalar fallback (no AVX detected)";
#endif
}

// -----------------------------------------------------------------------------
// Packed B panel — transposes + packs B into contiguous micro-panels.
// This is THE fix for the cache-miss bottleneck: instead of jumping by
// `ldb` floats for every multiply (cache miss every time when ldb is large),
// we pre-arrange B so the inner loop reads sequentially.
// -----------------------------------------------------------------------------
struct PackedB {
    std::vector<float> data;
    int K {0}, N {0};
    static constexpr int NR = 8;   // micro-panel width (matches AVX2 8xf32)

    // Pack B [K, N] (row-major) or B^T [N, K] into [K, N] panel-major layout:
    // panel p covers columns [p*NR, p*NR+NR), stored as K rows of NR floats each.
    void pack(float const* B, int K_, int N_, int ldb, bool transB) {
        K = K_; N = N_;
        int n_panels = (N + NR - 1) / NR;
        data.assign((size_t)n_panels * NR * K, 0.f);

        for (int p = 0; p < n_panels; ++p) {
            int col0 = p * NR;
            int cols = std::min(NR, N - col0);
            float* dst = data.data() + (size_t)p * NR * K;
            for (int k = 0; k < K; ++k) {
                for (int c = 0; c < cols; ++c) {
                    float v = transB ? B[(col0 + c) * ldb + k]   // B^T[col,k] = B[k,col] transposed src
                                     : B[k * ldb + (col0 + c)];
                    dst[k * NR + c] = v;
                }
            }
        }
    }

    float const* panel(int p) const { return data.data() + (size_t)p * NR * K; }
};

// -----------------------------------------------------------------------------
// AVX2 + FMA micro-kernel: computes an 8-wide row of C for one row of A
// against one packed panel of B (8 columns), accumulating over K.
// -----------------------------------------------------------------------------
#if VLRF_X86
VLRF_TARGET_AVX2_FMA
inline void microkernel_avx2_1x8(float const* a_row, float const* b_panel,
                                  int K, float* c_row, float alpha) {
    __m256 acc = _mm256_setzero_ps();
    // Iterate ALL K rows — each row contributes one FMA against the 8-wide
    // column block. (Previous off-by-one here dropped the last row of K.)
    for (int k = 0; k < K; ++k) {
        __m256 bv = _mm256_loadu_ps(b_panel + (size_t)k * 8);
        __m256 av = _mm256_set1_ps(a_row[k]);
        acc = _mm256_fmadd_ps(av, bv, acc);
    }
    __m256 alpha_v = _mm256_set1_ps(alpha);
    acc = _mm256_mul_ps(acc, alpha_v);

    __m256 c_old = _mm256_loadu_ps(c_row);
    __m256 c_new = _mm256_add_ps(c_old, acc);
    _mm256_storeu_ps(c_row, c_new);
}
#endif

// -----------------------------------------------------------------------------
// AVX1 micro-kernel: identical math to the AVX2/FMA version, but uses
// separate multiply + add instead of fused multiply-add (FMA was added in
// Haswell/2013; Ivy Bridge/Sandy Bridge 2011-2012 have AVX1 only).
// One extra instruction per step, same throughput per element on these
// older CPUs since they don't have a wider FMA pipe to exploit anyway.
// -----------------------------------------------------------------------------
#if VLRF_X86
inline void microkernel_avx1_1x8(float const* a_row, float const* b_panel,
                                  int K, float* c_row, float alpha) {
    __m256 acc = _mm256_setzero_ps();
    for (int k = 0; k < K; ++k) {
        __m256 bv  = _mm256_loadu_ps(b_panel + (size_t)k * 8);
        __m256 av  = _mm256_set1_ps(a_row[k]);
        __m256 prod = _mm256_mul_ps(av, bv);
        acc = _mm256_add_ps(acc, prod);
    }
    __m256 alpha_v = _mm256_set1_ps(alpha);
    acc = _mm256_mul_ps(acc, alpha_v);

    __m256 c_old = _mm256_loadu_ps(c_row);
    __m256 c_new = _mm256_add_ps(c_old, acc);
    _mm256_storeu_ps(c_row, c_new);
}
#endif

// -----------------------------------------------------------------------------
// AVX1 GEMV fast path for M=1 — same structure as the AVX2 version,
// mul+add instead of fmadd. See gemv_avx2_row for full design rationale.
// -----------------------------------------------------------------------------
inline void gemv_avx1_row(float const* a, float const* B, float* c,
                          int K, int N, int ldb, float alpha, bool transB) {
#if VLRF_X86
    int N8 = (N / 8) * 8;

    if (!transB) {
        constexpr int NBLOCK = 64;
        for (int n0 = 0; n0 < N8; n0 += NBLOCK) {
            int n1 = std::min(n0 + NBLOCK, N8);
            int nb = (n1 - n0) / 8;
            __m256 acc[NBLOCK/8];
            for (int i = 0; i < nb; ++i) acc[i] = _mm256_setzero_ps();

            for (int k = 0; k < K; ++k) {
                __m256 av = _mm256_set1_ps(a[k]);
                float const* brow = B + (size_t)k * ldb + n0;
                for (int i = 0; i < nb; ++i) {
                    __m256 bv   = _mm256_loadu_ps(brow + i*8);
                    __m256 prod = _mm256_mul_ps(av, bv);
                    acc[i] = _mm256_add_ps(acc[i], prod);
                }
            }
            __m256 alpha_v = _mm256_set1_ps(alpha);
            for (int i = 0; i < nb; ++i) {
                __m256 result = _mm256_mul_ps(acc[i], alpha_v);
                __m256 c_old  = _mm256_loadu_ps(c + n0 + i*8);
                _mm256_storeu_ps(c + n0 + i*8, _mm256_add_ps(c_old, result));
            }
        }
        for (int n = N8; n < N; ++n) {
            float s = 0.f;
            for (int k = 0; k < K; ++k) s += a[k] * B[k*ldb+n];
            c[n] += alpha * s;
        }
    } else {
        for (int n = 0; n < N; ++n) {
            float const* brow = B + (size_t)n * ldb;
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            int K8 = (K/8)*8;
            for (; k < K8; k += 8) {
                __m256 av   = _mm256_loadu_ps(a + k);
                __m256 bv   = _mm256_loadu_ps(brow + k);
                __m256 prod = _mm256_mul_ps(av, bv);
                acc = _mm256_add_ps(acc, prod);
            }
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 sum128 = _mm_add_ps(lo, hi);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float s = _mm_cvtss_f32(sum128);
            for (; k < K; ++k) s += a[k] * brow[k];
            c[n] += alpha * s;
        }
    }
#else
    (void)a;(void)B;(void)c;(void)K;(void)N;(void)ldb;(void)alpha;(void)transB;
#endif
}

// -----------------------------------------------------------------------------
// AVX2/FMA GEMV fast path for M=1 (the dominant shape during autoregressive
// decode: one token times a [K,N] weight matrix).
//
// Critical optimisation: NO packing step. B is read in its natural
// row-major layout. For each k, we broadcast a[k] and FMA against 8
// contiguous floats of B's row k — this is fully sequential, cache-friendly
// access with zero pre-processing overhead.
//
// Packing made sense for the prefill (M>>1) case because the packed panel
// is reused across many rows of A. For M=1 there's only one row, so the
// pack cost (which scales with K*N) is pure waste — confirmed by profiling
// to cost MORE than the GEMM itself for typical Granite-3B shapes.
// -----------------------------------------------------------------------------
VLRF_TARGET_AVX2_FMA
inline void gemv_avx2_row(float const* a, float const* B, float* c,
                          int K, int N, int ldb, float alpha, bool transB) {
#if VLRF_X86
    int N8 = (N / 8) * 8;

    if (!transB) {
        // B is [K, N] row-major: B[k,n] at B[k*ldb+n]. Inner 8 contiguous
        // floats of row k map directly to one AVX2 register — ideal layout.
        // Process N in blocks of up to 64 columns (8 AVX2 registers) at a
        // time on the stack — avoids heap allocation, keeps accumulators
        // resident in registers/L1 across the K-loop.
        constexpr int NBLOCK = 64;   // 8 registers x 8 floats
        for (int n0 = 0; n0 < N8; n0 += NBLOCK) {
            int n1 = std::min(n0 + NBLOCK, N8);
            int nb = (n1 - n0) / 8;
            __m256 acc[NBLOCK/8];
            for (int i = 0; i < nb; ++i) acc[i] = _mm256_setzero_ps();

            for (int k = 0; k < K; ++k) {
                __m256 av = _mm256_set1_ps(a[k]);
                float const* brow = B + (size_t)k * ldb + n0;
                for (int i = 0; i < nb; ++i) {
                    __m256 bv = _mm256_loadu_ps(brow + i*8);
                    acc[i] = _mm256_fmadd_ps(av, bv, acc[i]);
                }
            }
            __m256 alpha_v = _mm256_set1_ps(alpha);
            for (int i = 0; i < nb; ++i) {
                __m256 result = _mm256_mul_ps(acc[i], alpha_v);
                __m256 c_old  = _mm256_loadu_ps(c + n0 + i*8);
                _mm256_storeu_ps(c + n0 + i*8, _mm256_add_ps(c_old, result));
            }
        }
        // Tail (N not multiple of 8)
        for (int n = N8; n < N; ++n) {
            float s = 0.f;
            for (int k = 0; k < K; ++k) s += a[k] * B[k*ldb+n];
            c[n] += alpha * s;
        }
    } else {
        // B is [N, K] (transB): row n of B has K contiguous floats.
        // Each output n is an independent dot-product of length K.
        // Vectorise the K-reduction itself (horizontal sum at the end).
        for (int n = 0; n < N; ++n) {
            float const* brow = B + (size_t)n * ldb;  // ldb == K here
            __m256 acc = _mm256_setzero_ps();
            int k = 0;
            int K8 = (K/8)*8;
            for (; k < K8; k += 8) {
                __m256 av = _mm256_loadu_ps(a + k);
                __m256 bv = _mm256_loadu_ps(brow + k);
                acc = _mm256_fmadd_ps(av, bv, acc);
            }
            // Horizontal sum of acc
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 sum128 = _mm_add_ps(lo, hi);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float s = _mm_cvtss_f32(sum128);
            for (; k < K; ++k) s += a[k] * brow[k];   // tail
            c[n] += alpha * s;
        }
    }
#else
    (void)a;(void)B;(void)c;(void)K;(void)N;(void)ldb;(void)alpha;(void)transB;
#endif
}

// -----------------------------------------------------------------------------
// AVX2/FMA GEMM: C[M,N] += alpha * A[M,K] * op(B)
// A is read directly (row-major, supports transA via stride change).
// B is packed once per call into cache-friendly panels.
// -----------------------------------------------------------------------------
VLRF_TARGET_AVX2_FMA
inline bool gemm_avx2(float const* A, float const* B, float* C,
                      int M, int K, int N,
                      int lda, int ldb, int ldc,
                      float alpha, bool transA, bool transB) {
#if !VLRF_X86
    (void)A;(void)B;(void)C;(void)M;(void)K;(void)N;
    (void)lda;(void)ldb;(void)ldc;(void)alpha;(void)transA;(void)transB;
    return false;  // not on x86, caller should fall back
#else
    auto const& feat = cpu_features();
    if (!feat.avx2 || !feat.fma) return false;

    constexpr int NR = PackedB::NR;
    int n_panels = (N + NR - 1) / NR;

    PackedB packed;
    packed.pack(B, K, N, ldb, transB);

    // Buffer for one row of A (handles transA without re-striding per element)
    std::vector<float> a_row_buf;
    if (transA) a_row_buf.resize(K);

    for (int i = 0; i < M; ++i) {
        float const* a_row;
        if (transA) {
            for (int k = 0; k < K; ++k) a_row_buf[k] = A[k * lda + i];
            a_row = a_row_buf.data();
        } else {
            a_row = A + (size_t)i * lda;
        }

        float* c_row = C + (size_t)i * ldc;

        for (int p = 0; p < n_panels; ++p) {
            int col0 = p * NR;
            int cols = std::min(NR, N - col0);
            float const* b_panel = packed.panel(p);

            if (cols == NR) {
                microkernel_avx2_1x8(a_row, b_panel, K, c_row + col0, alpha);
            } else {
                // Tail: scalar fallback for the last partial panel
                for (int c = 0; c < cols; ++c) {
                    float acc = 0.f;
                    for (int k = 0; k < K; ++k)
                        acc += a_row[k] * b_panel[k * NR + c];
                    c_row[col0 + c] += alpha * acc;
                }
            }
        }
    }
    return true;
#endif
}

// -----------------------------------------------------------------------------
// AVX1 GEMM: same structure as gemm_avx2 but uses the AVX1 micro-kernel
// (no FMA). For prefill / batch shapes (M > 1) on Sandy Bridge/Ivy Bridge.
// -----------------------------------------------------------------------------
inline bool gemm_avx1(float const* A, float const* B, float* C,
                      int M, int K, int N,
                      int lda, int ldb, int ldc,
                      float alpha, bool transA, bool transB) {
#if !VLRF_X86
    (void)A;(void)B;(void)C;(void)M;(void)K;(void)N;
    (void)lda;(void)ldb;(void)ldc;(void)alpha;(void)transA;(void)transB;
    return false;
#else
    auto const& feat = cpu_features();
    if (!feat.avx1) return false;

    constexpr int NR = PackedB::NR;
    int n_panels = (N + NR - 1) / NR;

    PackedB packed;
    packed.pack(B, K, N, ldb, transB);

    std::vector<float> a_row_buf;
    if (transA) a_row_buf.resize(K);

    for (int i = 0; i < M; ++i) {
        float const* a_row;
        if (transA) {
            for (int k = 0; k < K; ++k) a_row_buf[k] = A[k * lda + i];
            a_row = a_row_buf.data();
        } else {
            a_row = A + (size_t)i * lda;
        }

        float* c_row = C + (size_t)i * ldc;

        for (int p = 0; p < n_panels; ++p) {
            int col0 = p * NR;
            int cols = std::min(NR, N - col0);
            float const* b_panel = packed.panel(p);

            if (cols == NR) {
                microkernel_avx1_1x8(a_row, b_panel, K, c_row + col0, alpha);
            } else {
                for (int c = 0; c < cols; ++c) {
                    float acc = 0.f;
                    for (int k = 0; k < K; ++k)
                        acc += a_row[k] * b_panel[k * NR + c];
                    c_row[col0 + c] += alpha * acc;
                }
            }
        }
    }
    return true;
#endif
}

// -----------------------------------------------------------------------------
// oneMKL backend (compile-time opt-in via -DVLRF_USE_ONEMKL)
// -----------------------------------------------------------------------------
#ifdef VLRF_USE_ONEMKL
inline bool gemm_onemkl(float const* A, float const* B, float* C,
                        int M, int K, int N,
                        int lda, int ldb, int ldc,
                        float alpha, float beta,
                        bool transA, bool transB) {
    CBLAS_TRANSPOSE ta = transA ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE tb = transB ? CblasTrans : CblasNoTrans;
    cblas_sgemm(CblasRowMajor, ta, tb,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    return true;
}
#endif

} // namespace simd
} // namespace vlrf
