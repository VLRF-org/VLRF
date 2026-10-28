// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — Math Ops  (ops.hpp)
// Pure-C++ float32 kernels.  Each function is independently replaceable by
// an accelerated backend (oneMKL / SYCL) without changing caller code.
// =============================================================================
#include "tensor.hpp"
#include "thread_pool.hpp"
#include "simd_gemm.hpp"
#include "stream_gemv.hpp"
#include <cmath>
#include <algorithm>
#include <cassert>
#include <thread>
#include <vector>
#include <functional>
#include <mutex>
#include <atomic>

#ifdef VLRF_USE_ONEMKL
#  include <oneapi/mkl.hpp>
#endif

namespace vlrf {
namespace ops {

constexpr int TILE = 32;

// -----------------------------------------------------------------------------
// Threading helpers
// -----------------------------------------------------------------------------
inline int hw_threads() {
    static int n = static_cast<int>(std::thread::hardware_concurrency());
    return std::max(1, n);
}

// Parallel-for over [0, N) in chunks.
// Uses a persistent thread pool (see thread_pool.hpp) instead of spawning
// fresh OS threads on every call — critical for decode-heavy workloads
// where this is invoked hundreds of times per generated token.
inline void parallel_for(int64_t N, std::function<void(int64_t, int64_t)> fn,
                          int max_threads = 0) {
    if (N <= 0) return;
    if (max_threads <= 0) max_threads = hw_threads();
    global_thread_pool().run(N, max_threads, fn);
}

// -----------------------------------------------------------------------------
// GEMM  — C = alpha * A * B + beta * C
//   A : [M, K]   B : [K, N]   C : [M, N]
//   All row-major.
// -----------------------------------------------------------------------------
void gemm(Tensor const& A, Tensor const& B, Tensor& C,
          float alpha = 1.f, float beta = 0.f,
          bool transA = false, bool transB = false);

// Convenience: C = A * B  (no transpose)
inline Tensor matmul(Tensor const& A, Tensor const& B) {
    assert(A.ndim() == 2 && B.ndim() == 2);
    assert(A.dim(1) == B.dim(0));
    Tensor C({A.dim(0), B.dim(1)});
    gemm(A, B, C, 1.f, 0.f, false, false);
    return C;
}

// Batched matmul: [B,M,K] x [B,K,N] -> [B,M,N]
Tensor bmm(Tensor const& A, Tensor const& B);

// -----------------------------------------------------------------------------
// Element-wise / activation
// -----------------------------------------------------------------------------
void add_inplace(Tensor& dst, Tensor const& src);           // dst += src
void add_bias_inplace(Tensor& dst, Tensor const& bias);     // dst[*, j] += bias[j]
void scale_inplace(Tensor& t, float s);                     // t *= s
void gelu_inplace(Tensor& t);                               // GELU activation
void silu_inplace(Tensor& t);                               // SiLU / Swish
void relu_inplace(Tensor& t);
void multiply_inplace(Tensor& dst, Tensor const& src);      // dst *= src (element)

// -----------------------------------------------------------------------------
// Normalization
// -----------------------------------------------------------------------------
// Layer-norm: operates on last dimension
//   x shape: [*, D]   weight/bias: [D]
void layer_norm(Tensor const& x, Tensor const& w, Tensor const& b,
                Tensor& out, float eps = 1e-5f);

// RMS-norm (no mean subtraction, used in LLaMA-style models)
void rms_norm(Tensor const& x, Tensor const& w,
              Tensor& out, float eps = 1e-6f);

// -----------------------------------------------------------------------------
// Attention helpers
// -----------------------------------------------------------------------------
// in-place softmax over last dimension of a 2-D or 3-D tensor
void softmax_inplace(Tensor& t);

// Causal mask: fill upper triangle (above diagonal) with -inf
void apply_causal_mask(Tensor& scores);   // [T, T] or [H, T, T]

// Scaled dot-product attention (single head or batch of heads)
//   Q,K,V : [T, D_head] or [H, T, D_head]
//   returns [T, D_head] or [H, T, D_head]
Tensor sdp_attention(Tensor const& Q, Tensor const& K, Tensor const& V,
                     bool causal = true);

// -----------------------------------------------------------------------------
// Embedding
// -----------------------------------------------------------------------------
// rows from weight table: out[i] = table[indices[i]]
Tensor embedding_lookup(Tensor const& table, std::vector<int> const& indices);

// Positional encoding (sinusoidal, additive)
void add_sinusoidal_pe(Tensor& x, int offset = 0);  // x: [T, D]

// -----------------------------------------------------------------------------
// Reshape / transpose helpers
// -----------------------------------------------------------------------------
// Transpose last two dims of a 3-D tensor [A, B, C] -> [A, C, B]
Tensor transpose_last2(Tensor const& t);

// Contiguous copy (force a new allocation)
Tensor contiguous(Tensor const& t);

// -----------------------------------------------------------------------------
// Implementations (header-only for portability; compile once via VLRF_IMPL)
// -----------------------------------------------------------------------------

// ---- GEMM: dispatches to the fastest available kernel ---------------------
//
// Priority order:
//   1. oneMKL sgemm        (compile with -DVLRF_USE_ONEMKL, fastest on Intel CPUs)
//   2. AVX2+FMA hand-kernel (auto-detected at runtime, ~4-8x scalar)
//   3. Scalar tiled loop   (always works, portable fallback)
//
// The AVX2 path packs B into cache-friendly panels once per call, fixing the
// strided-access cache-miss bottleneck that limited the scalar kernel to
// ~30% CPU utilisation. Multithreading splits work by row across hardware
// threads in all three paths.
inline void gemm_scalar_fallback(Tensor const& A, Tensor const& B, Tensor& C,
                                 float alpha, bool transA, bool transB,
                                 int M, int K, int N, int lda, int ldb) {
    float const* pA = A.data_ptr();
    float const* pB = B.data_ptr();
    float*       pC = C.data_ptr();

    parallel_for(M, [&](int64_t r0, int64_t r1) {
        for (int64_t i = r0; i < r1; ++i) {
            for (int jj = 0; jj < N; jj += TILE) {
                int jend = std::min(jj + TILE, N);
                float acc[TILE] = {};
                int tlen = jend - jj;
                for (int k = 0; k < K; ++k) {
                    float a = transA ? pA[k * lda + i] : pA[i * lda + k];
                    for (int j = 0; j < tlen; ++j) {
                        float b = transB ? pB[(jj + j) * ldb + k]
                                         : pB[k * ldb + (jj + j)];
                        acc[j] += a * b;
                    }
                }
                for (int j = 0; j < tlen; ++j)
                    pC[i * N + jj + j] += alpha * acc[j];
            }
        }
    });
}

inline void gemm(Tensor const& A, Tensor const& B, Tensor& C,
                 float alpha, float beta,
                 bool transA, bool transB)
{
    const int M = transA ? A.dim(1) : A.dim(0);
    const int K = transA ? A.dim(0) : A.dim(1);
    const int N = transB ? B.dim(0) : B.dim(1);

    assert(C.dim(0) == M && C.dim(1) == N);

    float*       pC = C.data_ptr();
    const int lda = A.dim(1);
    const int ldb = B.dim(1);
    const int ldc = N;

    // Apply beta scaling to C first (all kernels then accumulate alpha*A*B)
    if (beta == 0.f) {
        std::fill(pC, pC + (int64_t)M * N, 0.f);
    } else if (beta != 1.f) {
        for (int64_t i = 0; i < (int64_t)M * N; ++i) pC[i] *= beta;
    }
    if (alpha == 0.f) return;

#ifdef VLRF_USE_ONEMKL
    // oneMKL handles transpose, alpha and beta natively in one call.
    // Reset C to zero first since we already applied beta above;
    // pass beta=1.0 here so oneMKL just adds onto what we have.
    simd::gemm_onemkl(A.data_ptr(), B.data_ptr(), pC,
                      M, K, N, lda, ldb, ldc, alpha, 1.f, transA, transB);
    return;
#else
    // Pick the best available SIMD tier at runtime. Both AVX2+FMA and plain
    // AVX1 paths exist; AVX1 covers Sandy Bridge/Ivy Bridge (2011-2012,
    // e.g. 3rd-gen Intel Core laptops) which have 256-bit float ops but no FMA.
    auto const& feat = simd::cpu_features();
    bool use_avx2 = feat.avx2 && feat.fma;
    bool use_avx1 = !use_avx2 && feat.avx1;

    // M=1 fast path: autoregressive decode does one token at a time against
    // huge weight matrices. Packing B costs more than the GEMM itself here
    // (profiled: packing 2560x2560 = ~7ms vs ~2.6ms total GEMM time), so we
    // skip packing entirely and read B in its natural layout.
    if ((use_avx2 || use_avx1) && M == 1 && K >= 8 && N >= 8) {
        float const* pA = A.data_ptr();
        float const* pB = B.data_ptr();

        std::vector<float> a_row_buf;
        float const* a_row;
        if (transA) {
            a_row_buf.resize(K);
            for (int k = 0; k < K; ++k) a_row_buf[k] = pA[k * lda];
            a_row = a_row_buf.data();
        } else {
            a_row = pA;
        }

        auto run_gemv = [&](float const* a_in, float const* b_in,
                            float* c_out, int n_len, int ld, bool tB) {
            if (use_avx2) simd::gemv_avx2_row(a_in, b_in, c_out, K, n_len, ld, alpha, tB);
            else          simd::gemv_avx1_row(a_in, b_in, c_out, K, n_len, ld, alpha, tB);
        };

        // transB (weights stored [N,K], the model-file layout) goes to the
        // streaming kernel in stream_gemv.hpp: 4 rows x 2 accumulators and a
        // double/triple-buffered software-prefetch window (see that header).
        // Parallelise across the N (output) dimension — each thread computes
        // an independent contiguous slice of output columns. No write
        // contention since slices don't overlap.
        if (N >= 256) {
            parallel_for(N, [&](int64_t n0, int64_t n1) {
                int nlen = (int)(n1 - n0);
                if (!transB)
                    run_gemv(a_row, pB + n0, pC + n0, nlen, ldb, false);
                else if (!simd::stream_gemv_t(a_row, pB + (size_t)n0 * ldb, pC + n0,
                                               K, nlen, (size_t)ldb, alpha))
                    run_gemv(a_row, pB + (size_t)n0 * ldb, pC + n0, nlen, ldb, true);
            });
        } else {
            if (!(transB && simd::stream_gemv_t(a_row, pB, pC, K, N, (size_t)ldb, alpha)))
                run_gemv(a_row, pB, pC, N, ldb, transB);
        }
        return;
    }

    if ((use_avx2 || use_avx1) && M >= 4 && K >= 8 && N >= 8) {
        float const* pA = A.data_ptr();
        float const* pB = B.data_ptr();

        // Pack B once (shared across all row-blocks processed by every thread)
        simd::PackedB packed;
        packed.pack(pB, K, N, ldb, transB);

        parallel_for(M, [&](int64_t r0, int64_t r1) {
            std::vector<float> a_row_buf;
            if (transA) a_row_buf.resize(K);

            constexpr int NR = simd::PackedB::NR;
            int n_panels = (N + NR - 1) / NR;

            for (int64_t i = r0; i < r1; ++i) {
                float const* a_row;
                if (transA) {
                    for (int k = 0; k < K; ++k) a_row_buf[k] = pA[k * lda + i];
                    a_row = a_row_buf.data();
                } else {
                    a_row = pA + (size_t)i * lda;
                }
                float* c_row = pC + (size_t)i * ldc;

                for (int p = 0; p < n_panels; ++p) {
                    int col0 = p * NR;
                    int cols = std::min(NR, N - col0);
                    float const* b_panel = packed.panel(p);

                    if (cols == NR) {
#if VLRF_X86
                        if (use_avx2)
                            simd::microkernel_avx2_1x8(a_row, b_panel, K,
                                                       c_row + col0, alpha);
                        else
                            simd::microkernel_avx1_1x8(a_row, b_panel, K,
                                                       c_row + col0, alpha);
#endif
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
        });
        return;
    }

    // Fallback: scalar tiled loop (small matrices, no AVX support, or ARM/other arch)
    gemm_scalar_fallback(A, B, C, alpha, transA, transB, M, K, N, lda, ldb);
#endif
}

inline Tensor bmm(Tensor const& A, Tensor const& B) {
    assert(A.ndim() == 3 && B.ndim() == 3);
    assert(A.dim(0) == B.dim(0));
    assert(A.dim(2) == B.dim(1));
    int batch = A.dim(0), M = A.dim(1), K = A.dim(2), N = B.dim(2);
    Tensor C({batch, M, N});
    for (int b = 0; b < batch; ++b) {
        // wrap sub-matrices as 2-D views sharing the storage
        Tensor Av = Tensor::from_ptr(
            const_cast<float*>(A.data_ptr()) + (int64_t)b * M * K, {M, K});
        Tensor Bv = Tensor::from_ptr(
            const_cast<float*>(B.data_ptr()) + (int64_t)b * K * N, {K, N});
        Tensor Cv = Tensor::from_ptr(C.data_ptr() + (int64_t)b * M * N, {M, N});
        gemm(Av, Bv, Cv, 1.f, 0.f, false, false);
    }
    return C;
}

// -----------------------------------------------------------------------------
// Elementwise ops — parallelised across the FLAT element range, not rows.
// During autoregressive decode there is only 1 row (1 token), so any
// row-based parallel_for collapses to single-threaded by construction.
// FFN hidden dims (d_ff, often 4-8x d_model — e.g. 10240 for Granite-3B)
// are large enough to split across cores even for a single token, and
// these run on EVERY layer EVERY token, so leaving them single-threaded
// means one core does this work while the others sit idle between GEMMs.
//
// PARALLEL_ELEMENTWISE_MIN gates the threading: below this size the thread
// dispatch overhead would exceed the work itself, so we just run serially.
// -----------------------------------------------------------------------------
constexpr int64_t PARALLEL_ELEMENTWISE_MIN = 4096;

inline void add_inplace(Tensor& dst, Tensor const& src) {
    assert(dst.numel() == src.numel());
    float*       d = dst.data_ptr();
    float const* s = src.data_ptr();
    int64_t n = dst.numel();
    if (n < PARALLEL_ELEMENTWISE_MIN) {
        for (int64_t i = 0; i < n; ++i) d[i] += s[i];
        return;
    }
    parallel_for(n, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) d[i] += s[i];
    });
}

inline void add_bias_inplace(Tensor& dst, Tensor const& bias) {
    // dst: [T, D]  or  [*, D],  bias: [D]
    int64_t D = bias.numel();
    int64_t rows = dst.numel() / D;
    float*       d = dst.data_ptr();
    float const* b = bias.data_ptr();
    int64_t n = dst.numel();
    if (n < PARALLEL_ELEMENTWISE_MIN) {
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t j = 0; j < D; ++j)
                d[r * D + j] += b[j];
        return;
    }
    // Parallelise over flat index; bias index = flat % D works since
    // each row is exactly D elements (row-major contiguous).
    parallel_for(n, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) d[i] += b[i % D];
    });
}

inline void scale_inplace(Tensor& t, float s) {
    float* d = t.data_ptr();
    int64_t n = t.numel();
    if (n < PARALLEL_ELEMENTWISE_MIN) {
        for (int64_t i = 0; i < n; ++i) d[i] *= s;
        return;
    }
    parallel_for(n, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) d[i] *= s;
    });
}

inline void gelu_inplace(Tensor& t) {
    // Gaussian Error Linear Unit (tanh approximation).
    // std::tanh is a scalar libm call with no SIMD path — this is the most
    // expensive elementwise op in the model, parallelising it matters most.
    constexpr float c1 = 0.7978845608028654f;   // sqrt(2/pi)
    constexpr float c2 = 0.044715f;
    float* d = t.data_ptr();
    int64_t n = t.numel();
    auto body = [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) {
            float x = d[i];
            float inner = c1 * (x + c2 * x * x * x);
            d[i] = 0.5f * x * (1.f + std::tanh(inner));
        }
    };
    if (n < PARALLEL_ELEMENTWISE_MIN) { body(0, n); return; }
    parallel_for(n, body);
}

inline void silu_inplace(Tensor& t) {
    // std::exp, like tanh, is a scalar libm call — same rationale as gelu.
    float* d = t.data_ptr();
    int64_t n = t.numel();
    auto body = [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i)
            d[i] = d[i] / (1.f + std::exp(-d[i]));
    };
    if (n < PARALLEL_ELEMENTWISE_MIN) { body(0, n); return; }
    parallel_for(n, body);
}

inline void relu_inplace(Tensor& t) {
    float* d = t.data_ptr();
    int64_t n = t.numel();
    auto body = [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) d[i] = std::max(0.f, d[i]);
    };
    if (n < PARALLEL_ELEMENTWISE_MIN) { body(0, n); return; }
    parallel_for(n, body);
}

inline void multiply_inplace(Tensor& dst, Tensor const& src) {
    assert(dst.numel() == src.numel());
    float*       d = dst.data_ptr();
    float const* s = src.data_ptr();
    int64_t n = dst.numel();
    auto body = [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) d[i] *= s[i];
    };
    if (n < PARALLEL_ELEMENTWISE_MIN) { body(0, n); return; }
    parallel_for(n, body);
}

inline void layer_norm(Tensor const& x, Tensor const& w, Tensor const& b,
                       Tensor& out, float eps)
{
    int64_t D    = x.dim(x.ndim() - 1);
    int64_t rows = x.numel() / D;
    assert(w.numel() == D && b.numel() == D);
    if (out.numel() != x.numel()) out = like(x);

    float const* px = x.data_ptr();
    float const* pw = w.data_ptr();
    float const* pb = b.data_ptr();
    float*       po = out.data_ptr();

    parallel_for(rows, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            float const* row = px + r * D;
            float*       dst = po + r * D;
            // mean
            float mean = 0.f;
            for (int64_t j = 0; j < D; ++j) mean += row[j];
            mean /= D;
            // var
            float var = 0.f;
            for (int64_t j = 0; j < D; ++j) {
                float diff = row[j] - mean;
                var += diff * diff;
            }
            var = var / D + eps;
            float inv_std = 1.f / std::sqrt(var);
            for (int64_t j = 0; j < D; ++j)
                dst[j] = (row[j] - mean) * inv_std * pw[j] + pb[j];
        }
    });
}

inline void rms_norm(Tensor const& x, Tensor const& w, Tensor& out, float eps) {
    int64_t D    = x.dim(x.ndim() - 1);
    int64_t rows = x.numel() / D;
    assert(w.numel() == D);
    if (out.numel() != x.numel()) out = like(x);

    float const* px = x.data_ptr();
    float const* pw = w.data_ptr();
    float*       po = out.data_ptr();

    parallel_for(rows, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            float const* row = px + r * D;
            float*       dst = po + r * D;
            float rms = 0.f;
            for (int64_t j = 0; j < D; ++j) rms += row[j] * row[j];
            rms = std::sqrt(rms / D + eps);
            float inv = 1.f / rms;
            for (int64_t j = 0; j < D; ++j) dst[j] = row[j] * inv * pw[j];
        }
    });
}

inline void softmax_inplace(Tensor& t) {
    // Used inside attention for the [T, ctx_len] score matrix. During
    // decode T=1, so rows=1 and the row-level parallel_for below would
    // collapse to one thread; ctx_len can be large (up to max_seq_len),
    // so we parallelise the inner exp/sum reduction per row instead.
    //
    // NOTE: this function may itself be called from inside a parallel_for
    // job (e.g. attention.hpp parallelises across heads, and calls this
    // per head). The thread pool's reentrancy guard (thread_pool.hpp)
    // makes any parallel_for() called from such a context run inline on
    // the calling thread rather than submit+wait on the pool, so nesting
    // here is now safe by construction — no deadlock, just serial
    // execution for the nested call specifically.
    int64_t D    = t.dim(t.ndim() - 1);
    int64_t rows = t.numel() / D;
    float*  p    = t.data_ptr();

    auto softmax_row = [&](float* row) {
        float mx = *std::max_element(row, row + D);
        if (D < PARALLEL_ELEMENTWISE_MIN) {
            float sum = 0.f;
            for (int64_t j = 0; j < D; ++j) { row[j] = std::exp(row[j]-mx); sum += row[j]; }
            float inv = 1.f / (sum + 1e-9f);
            for (int64_t j = 0; j < D; ++j) row[j] *= inv;
            return;
        }
        // Parallel exp + per-slice partial sums (one slot per slice,
        // written by exactly one thread — no lock needed), then a tiny
        // serial reduce, then parallel scale.
        int n_jobs = hw_threads();
        std::vector<float> partial(n_jobs, 0.f);
        std::atomic<int> slice_idx{0};
        parallel_for(D, [&](int64_t j0, int64_t j1) {
            int my_slot = slice_idx.fetch_add(1, std::memory_order_relaxed);
            float local = 0.f;
            for (int64_t j = j0; j < j1; ++j) {
                row[j] = std::exp(row[j] - mx);
                local += row[j];
            }
            partial[my_slot % n_jobs] = local;
        });
        float sum = 0.f;
        for (float v : partial) sum += v;
        float inv = 1.f / (sum + 1e-9f);
        parallel_for(D, [&](int64_t j0, int64_t j1) {
            for (int64_t j = j0; j < j1; ++j) row[j] *= inv;
        });
    };

    if (rows == 1) {
        softmax_row(p);
        return;
    }
    parallel_for(rows, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) softmax_row(p + r * D);
    });
}

inline void apply_causal_mask(Tensor& scores) {
    // scores: [T, T]
    assert(scores.ndim() == 2 && scores.dim(0) == scores.dim(1));
    int T = scores.dim(0);
    float* p = scores.data_ptr();
    for (int i = 0; i < T; ++i)
        for (int j = i + 1; j < T; ++j)
            p[i * T + j] = -1e30f;
}

inline Tensor sdp_attention(Tensor const& Q, Tensor const& K, Tensor const& V,
                             bool causal)
{
    // Q,K,V: [T, Dh]  or [H, T, Dh]
    bool batched = Q.ndim() == 3;
    if (!batched) {
        int T  = Q.dim(0);
        int Dh = Q.dim(1);
        float scale = 1.f / std::sqrt((float)Dh);

        // scores = Q * K^T  [T, T]
        Tensor scores({T, T});
        Tensor Kt = transpose_last2(K);   // [Dh, T]
        gemm(Q, Kt, scores, scale, 0.f, false, false);
        if (causal) apply_causal_mask(scores);
        softmax_inplace(scores);

        // out = scores * V  [T, Dh]
        Tensor out({T, Dh});
        gemm(scores, V, out, 1.f, 0.f, false, false);
        return out;
    } else {
        int H  = Q.dim(0);
        int T  = Q.dim(1);
        int Dh = Q.dim(2);
        float scale = 1.f / std::sqrt((float)Dh);
        Tensor out({H, T, Dh});
        for (int h = 0; h < H; ++h) {
            Tensor Qh = Tensor::from_ptr(
                const_cast<float*>(Q.data_ptr()) + (int64_t)h * T * Dh, {T, Dh});
            Tensor Kh = Tensor::from_ptr(
                const_cast<float*>(K.data_ptr()) + (int64_t)h * T * Dh, {T, Dh});
            Tensor Vh = Tensor::from_ptr(
                const_cast<float*>(V.data_ptr()) + (int64_t)h * T * Dh, {T, Dh});
            Tensor Kht = transpose_last2(Kh);
            Tensor scores({T, T});
            gemm(Qh, Kht, scores, scale, 0.f, false, false);
            if (causal) apply_causal_mask(scores);
            softmax_inplace(scores);
            Tensor outh = Tensor::from_ptr(out.data_ptr() + (int64_t)h * T * Dh, {T, Dh});
            gemm(scores, Vh, outh, 1.f, 0.f, false, false);
        }
        return out;
    }
}

inline Tensor embedding_lookup(Tensor const& table, std::vector<int> const& indices) {
    int T = static_cast<int>(indices.size());
    int D = table.dim(1);
    Tensor out({T, D});
    float const* pt = table.data_ptr();
    float*       po = out.data_ptr();
    for (int i = 0; i < T; ++i)
        std::copy(pt + (int64_t)indices[i] * D,
                  pt + (int64_t)indices[i] * D + D,
                  po + (int64_t)i * D);
    return out;
}

inline void add_sinusoidal_pe(Tensor& x, int offset) {
    int T = x.dim(0);
    int D = x.dim(1);
    float* p = x.data_ptr();
    for (int t = 0; t < T; ++t) {
        int pos = t + offset;
        for (int i = 0; i < D; i += 2) {
            float div = std::pow(10000.f, (float)i / D);
            p[t * D + i]     += std::sin(pos / div);
            if (i + 1 < D)
                p[t * D + i + 1] += std::cos(pos / div);
        }
    }
}

inline Tensor transpose_last2(Tensor const& t) {
    if (t.ndim() == 2) {
        int R = t.dim(0), C = t.dim(1);
        Tensor out({C, R});
        float const* s = t.data_ptr();
        float*       d = out.data_ptr();
        for (int r = 0; r < R; ++r)
            for (int c = 0; c < C; ++c)
                d[c * R + r] = s[r * C + c];
        return out;
    } else {
        assert(t.ndim() == 3);
        int A = t.dim(0), B = t.dim(1), C = t.dim(2);
        Tensor out({A, C, B});
        float const* s = t.data_ptr();
        float*       d = out.data_ptr();
        for (int a = 0; a < A; ++a)
            for (int b = 0; b < B; ++b)
                for (int c = 0; c < C; ++c)
                    d[a * C * B + c * B + b] = s[a * B * C + b * C + c];
        return out;
    }
}

inline Tensor contiguous(Tensor const& t) {
    Tensor out(t.shape);
    std::copy(t.data_ptr(), t.data_ptr() + t.numel(), out.data_ptr());
    return out;
}

} // namespace ops
} // namespace vlrf
