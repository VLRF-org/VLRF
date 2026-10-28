// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — compute routing  (compute.hpp)
//
// Thin wrappers around ops::* that route through the active GPU backend
// (vlrf::backend::active_backend()) when one is selected, and call ops::*
// directly otherwise.
//
// Deliberate scope: only ops that IBackend actually implements are routed
// here — gemm, layer_norm, softmax_inplace, gelu_inplace, add_inplace.
// add_bias_inplace, transpose_last2, and bmm have no IBackend equivalent
// and always run on CPU regardless of the active backend; extending
// IBackend for those is a separate decision, not folded into this pass.
//
// IMPORTANT — CPU is intentionally excluded from GPU routing:
// when BackendRegistry's active backend is BackendType::CPU (the default,
// and what auto_select() falls back to if no GPU is available), every
// function below calls ops::* directly rather than going through
// active_backend(). This is deliberate, not an oversight: IBackend's CPU
// path (CpuBackend) is functionally identical to calling ops:: directly,
// but adds a virtual-call indirection and (for gemm/layer_norm/etc.) an
// is_available() branch for no benefit. Routing is only meaningful once a
// GPU backend (OpenCL/SYCL) is active, so that's the only case this checks.
// =============================================================================
#include "tensor.hpp"
#include "ops.hpp"
#include "backend.hpp"

namespace vlrf {
namespace compute {

// True once a non-CPU backend has been selected via BackendRegistry
// (try_opencl(), try_sycl(), or auto_select() successfully picking one).
// Cheap: a virtual call + enum compare, safe to call per-op.
inline bool gpu_active() {
    return vlrf::backend::active_backend().type() != vlrf::backend::BackendType::CPU;
}

inline void gemm(Tensor const& A, Tensor const& B, Tensor& C,
                  float alpha = 1.f, float beta = 0.f,
                  bool transA = false, bool transB = false) {
    if (gpu_active()) {
        vlrf::backend::active_backend().gemm(A, B, C, alpha, beta, transA, transB);
    } else {
        ops::gemm(A, B, C, alpha, beta, transA, transB);
    }
}

// Convenience: C = A * B (no transpose), matching ops::matmul's contract.
inline Tensor matmul(Tensor const& A, Tensor const& B) {
    Tensor C({A.dim(0), B.dim(1)});
    gemm(A, B, C, 1.f, 0.f, false, false);
    return C;
}

// Linear layer exactly as weights are stored in the model file:
//   out[M,N] = x[M,K] * W[N,K]^T
// Passes transB=true so the weight matrix is read IN PLACE. The previous
// pattern, matmul(x, ops::transpose_last2(W)), materialised a full
// transposed copy of W (single-threaded, strided writes, fresh zeroed
// allocation) on EVERY call — i.e. every weight matrix, every token.
inline Tensor linear(Tensor const& x, Tensor const& W) {
    Tensor C({x.dim(0), W.dim(0)});
    gemm(x, W, C, 1.f, 0.f, false, true);
    return C;
}

inline void layer_norm(Tensor const& x, Tensor const& w, Tensor const& b,
                        Tensor& out, float eps) {
    if (gpu_active()) {
        vlrf::backend::active_backend().layer_norm(x, w, b, out, eps);
    } else {
        ops::layer_norm(x, w, b, out, eps);
    }
}

inline void softmax_inplace(Tensor& t) {
    if (gpu_active()) {
        vlrf::backend::active_backend().softmax_inplace(t);
    } else {
        ops::softmax_inplace(t);
    }
}

inline void gelu_inplace(Tensor& t) {
    if (gpu_active()) {
        vlrf::backend::active_backend().gelu_inplace(t);
    } else {
        ops::gelu_inplace(t);
    }
}

inline void add_inplace(Tensor& dst, Tensor const& src) {
    if (gpu_active()) {
        vlrf::backend::active_backend().add_inplace(dst, src);
    } else {
        ops::add_inplace(dst, src);
    }
}

} // namespace compute
} // namespace vlrf
