// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — Backend Abstraction  (backend.hpp)
// Separates compute backend from model logic.
// CPU backend uses the ops.hpp kernels (see ibackend.hpp).
// GPU backends provide the same interface over OpenCL/SYCL.
// =============================================================================
#include "ibackend.hpp"
#include <sstream>
#include <cmath>

#ifndef VLRF_BACKEND_HPP
#  define VLRF_BACKEND_HPP
#endif

// NOTE: these GPU backend headers must be included here, at true global
// scope (before `namespace vlrf { namespace backend {` opens below), even
// though their contents live in vlrf::backend. Each of them pulls in its own
// system headers (<CL/cl.h>, <sycl/sycl.hpp>, <optional>, <unordered_map>,
// ...); if this include happened while already nested inside
// namespace vlrf::backend, those system headers' own internal declarations
// would be textually nested inside vlrf::backend as well, which some standard
// library implementations don't tolerate. See ibackend.hpp's header comment.
#if defined(VLRF_USE_OPENCL)
#  include "opencl_kernel_manager.hpp"
#endif

#if defined(VLRF_USE_SYCL)
#  include "sycl_backend.hpp"
#endif

namespace vlrf {
namespace backend {

// -----------------------------------------------------------------------------
// OpenCLBackend  — wraps KernelManager + dispatches real OpenCL kernels
// -----------------------------------------------------------------------------
class OpenCLBackend : public IBackend {
    bool available_{false};
    std::string info_{"OpenCL: not initialised"};
    CpuBackend  fallback_;
#if defined(VLRF_USE_OPENCL)
    // mutable: gemm_kernel()/softmax_kernel()/attention_kernel() lazily
    // build+cache on first use, called from OpenCLBackend's const dispatch_*
    // methods (mirrors the same lazy-cache-from-const-method pattern used by
    // SYCLBackend::allocations_).
    mutable KernelManager kernel_manager_;
#endif

public:
    BackendType type()  const override { return BackendType::OPENCL; }
    bool is_available() const override { return available_; }
    std::string device_info() const override { return info_; }

    std::string capabilities() const override {
        std::ostringstream oss;
        oss << "OpenCL backend; " << (available_ ? "ready" : "not initialised");
        // Per-op dispatch, matching the dispatch_* methods below: each op
        // routes to a real OpenCL kernel for the shapes it supports and
        // silently falls back to CPU per-call otherwise, not just when the
        // backend as a whole is unavailable.
#if defined(VLRF_USE_OPENCL)
        const std::string dev = available_ ? device_kind() : std::string("CPU");
        oss << "\n  gemm            : " << (available_ ? (dev + " (OpenCL kernel, 2D, any transA/transB combination — else CPU fallback)") : "CPU (fallback - backend not initialised)")
            << "\n  layer_norm      : " << (available_ ? (dev + " (OpenCL kernel)") : "CPU (fallback - backend not initialised)")
            << "\n  softmax_inplace : " << (available_ ? (dev + " (OpenCL kernel)") : "CPU (fallback - backend not initialised)")
            << "\n  gelu_inplace    : " << (available_ ? (dev + " (OpenCL kernel)") : "CPU (fallback - backend not initialised)")
            << "\n  add_inplace     : " << (available_ ? (dev + " (OpenCL kernel)") : "CPU (fallback - backend not initialised)")
            << "\n  attention (fused): " << (available_ ? (dev + " (fused flash_attention kernel: QK^T + online softmax + weighted-V in one launch)") : "CPU (fallback - backend not initialised)");
#else
        oss << "\n  (compiled without VLRF_USE_OPENCL — all ops CPU)";
#endif
        return oss.str();
    }

    // Human-readable classification of the currently selected OpenCL
    // device's type (GPU / CPU / accelerator / device), shared by
    // device_info() and capabilities() so both report the same thing.
    std::string device_kind() const {
#if defined(VLRF_USE_OPENCL)
        cl_device_type t = kernel_manager_.device_type();
        if (t & CL_DEVICE_TYPE_GPU)         return "GPU";
        if (t & CL_DEVICE_TYPE_CPU)         return "CPU";
        if (t & CL_DEVICE_TYPE_ACCELERATOR) return "accelerator";
        return "device";
#else
        return "CPU";
#endif
    }

    bool try_init() {
#if defined(VLRF_USE_OPENCL)
        if (!kernel_manager_.init()) {
            available_ = false;
            info_ = "OpenCL initialization failed";
            return false;
        }

        cl_device_id device = nullptr;
        size_t size = 0;
        clGetCommandQueueInfo(kernel_manager_.queue(), CL_QUEUE_DEVICE, sizeof(device), &device, &size);
        if (device) {
            char name[256] = {0};
            char vendor[256] = {0};
            clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
            clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
            const std::string kind = device_kind();
            info_ = std::string("OpenCL: ") + name + " (" + vendor + ") ["
                  + kind + "] (gemm/softmax/layer_norm/gelu/add/attention on "
                  + kind + ")";
            available_ = true;
        } else {
            info_ = "OpenCL: device information unavailable";
            available_ = false;
        }
#else
        available_ = false;
        info_ = "OpenCL support not compiled in (define VLRF_USE_OPENCL)";
#endif
        return available_;
    }

    Tensor to_device(Tensor const& t) const override { return t; }
    Tensor to_host(Tensor const& t)   const override { return t; }

    void gemm(Tensor const& A, Tensor const& B, Tensor& C,
              float alpha, float beta, bool tA, bool tB) const override {
#if defined(VLRF_USE_OPENCL)
        if (available_ && dispatch_gemm(A, B, C, alpha, beta, tA, tB)) return;
#endif
        fallback_.gemm(A, B, C, alpha, beta, tA, tB);
    }
    void layer_norm(Tensor const& x, Tensor const& w, Tensor const& b,
                    Tensor& out, float eps) const override {
#if defined(VLRF_USE_OPENCL)
        if (available_ && dispatch_layer_norm(x, w, b, out, eps)) return;
#endif
        fallback_.layer_norm(x, w, b, out, eps);
    }
    void softmax_inplace(Tensor& t) const override {
#if defined(VLRF_USE_OPENCL)
        if (available_ && dispatch_softmax(t)) return;
#endif
        fallback_.softmax_inplace(t);
    }
    void gelu_inplace(Tensor& t)    const override {
#if defined(VLRF_USE_OPENCL)
        if (available_ && dispatch_gelu(t)) return;
#endif
        fallback_.gelu_inplace(t);
    }
    void add_inplace(Tensor& dst, Tensor const& src) const override {
#if defined(VLRF_USE_OPENCL)
        if (available_ && dispatch_add_inplace(dst, src)) return;
#endif
        fallback_.add_inplace(dst, src);
    }

    // Fused scaled-dot-product attention. Not part of IBackend (ops.hpp's
    // sdp_attention is a free function, not a virtual method), but exposed
    // here so callers that already hold an OpenCLBackend can opt into the
    // fused GPU path directly; falls back to ops::sdp_attention on CPU.
    // Q, K, V: [n_h, T_or_ctx_len, d_h] batched (3D) layout — matches the
    // per-head view layout ops::sdp_attention's batched path builds via
    // Tensor::from_ptr.
    Tensor sdp_attention(Tensor const& Q, Tensor const& K, Tensor const& V, bool causal) const override {
#if defined(VLRF_USE_OPENCL)
        if (available_) {
            Tensor out;
            if (dispatch_attention(Q, K, V, causal, out)) return out;
        }
#endif
        return ops::sdp_attention(Q, K, V, causal);
    }

private:
#if defined(VLRF_USE_OPENCL)
    // Real dispatch through the OpenCL kernels built by KernelManager.
    // Returns false (falls back to CPU) on any setup failure or shape the
    // kernel doesn't support, rather than throwing, matching the pattern
    // used by the SYCL backend.
    //
    // NOTE: the compiled `gemm` kernel (opencl_kernels.hpp) is
    // C = alpha*op(A)*op(B) + beta*C for row-major, tightly-packed A/B/C —
    // transA/transB select one of four precompiled program variants
    // (NN/NT/TN/TT), each built lazily on first use. Non-contiguous strides
    // (views) still fall back to CPU rather than silently computing the
    // wrong thing, since the kernel assumes tightly-packed rows.
    bool dispatch_gemm(Tensor const& A, Tensor const& B, Tensor& C,
                        float alpha, float beta, bool tA, bool tB) const {
        if (A.ndim() != 2 || B.ndim() != 2 || C.ndim() != 2) return false;

        int M = tA ? A.dim(1) : A.dim(0);
        int K = tA ? A.dim(0) : A.dim(1);
        int N = tB ? B.dim(0) : B.dim(1);
        int K_b = tB ? B.dim(1) : B.dim(0);
        if (K_b != K || C.dim(0) != M || C.dim(1) != N) return false;

        cl_kernel k = kernel_manager_.gemm_kernel(tA, tB);
        cl_context ctx = kernel_manager_.context();
        cl_command_queue q = kernel_manager_.queue();
        if (!k || !ctx || !q) return false;

        cl_int err = CL_SUCCESS;
        cl_mem bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      A.numel() * sizeof(float), const_cast<float*>(A.data_ptr()), &err);
        cl_mem bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      B.numel() * sizeof(float), const_cast<float*>(B.data_ptr()), &err);
        cl_mem bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      C.numel() * sizeof(float), C.data_ptr(), &err);
        if (!bufA || !bufB || !bufC) {
            if (bufA) clReleaseMemObject(bufA);
            if (bufB) clReleaseMemObject(bufB);
            if (bufC) clReleaseMemObject(bufC);
            return false;
        }

        // Kernel signature: gemm(M, N, K, alpha, beta, A, B, C)
        clSetKernelArg(k, 0, sizeof(int), &M);
        clSetKernelArg(k, 1, sizeof(int), &N);
        clSetKernelArg(k, 2, sizeof(int), &K);
        clSetKernelArg(k, 3, sizeof(float), &alpha);
        clSetKernelArg(k, 4, sizeof(float), &beta);
        clSetKernelArg(k, 5, sizeof(cl_mem), &bufA);
        clSetKernelArg(k, 6, sizeof(cl_mem), &bufB);
        clSetKernelArg(k, 7, sizeof(cl_mem), &bufC);

        // Work-group size is fixed at (RTS, RTS) = (8, 8) via
        // reqd_work_group_size in the kernel source (see TS/WPT/RTS in
        // opencl_kernels.hpp); each thread covers a 4x4 output block, so
        // global size must be a multiple of TS=32 along each dim.
        constexpr size_t kTile = 32;
        constexpr size_t kThreadsPerDim = 8; // TS / WPT
        size_t local[2] = { kThreadsPerDim, kThreadsPerDim };
        size_t global[2] = {
            ((static_cast<size_t>(M) + kTile - 1) / kTile) * kThreadsPerDim,
            ((static_cast<size_t>(N) + kTile - 1) / kTile) * kThreadsPerDim
        };
        err = clEnqueueNDRangeKernel(q, k, 2, nullptr, global, local, 0, nullptr, nullptr);
        bool ok = (err == CL_SUCCESS);
        if (ok) {
            err = clEnqueueReadBuffer(q, bufC, CL_TRUE, 0, C.numel() * sizeof(float), C.data_ptr(), 0, nullptr, nullptr);
            ok = (err == CL_SUCCESS);
        }

        clReleaseMemObject(bufA);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufC);
        return ok;
    }

    bool dispatch_softmax(Tensor& t) const {
        cl_kernel k = kernel_manager_.softmax_kernel();
        cl_context ctx = kernel_manager_.context();
        cl_command_queue q = kernel_manager_.queue();
        if (!k || !ctx || !q) return false;
        if (t.ndim() < 1) return false;

        int64_t D = t.dim(t.ndim() - 1);
        int64_t rows = t.numel() / D;

        cl_int err = CL_SUCCESS;
        cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     t.numel() * sizeof(float), t.data_ptr(), &err);
        if (!buf) return false;

        int r = static_cast<int>(rows);
        int c = static_cast<int>(D);
        // Kernel signature: softmax_inplace(x, rows, cols)
        clSetKernelArg(k, 0, sizeof(cl_mem), &buf);
        clSetKernelArg(k, 1, sizeof(int), &r);
        clSetKernelArg(k, 2, sizeof(int), &c);

        // One work-group (256 threads) per row — matches layer_norm's
        // launch geometry and the kernel's get_group_id(0)-based row
        // indexing (softmax_inplace's block reduction needs the full
        // 256-thread group cooperating on a single row, not one thread
        // per row).
        size_t local = 256;
        size_t global = static_cast<size_t>(rows) * 256;
        err = clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        bool ok = (err == CL_SUCCESS);
        if (ok) {
            err = clEnqueueReadBuffer(q, buf, CL_TRUE, 0, t.numel() * sizeof(float), t.data_ptr(), 0, nullptr, nullptr);
            ok = (err == CL_SUCCESS);
        }
        clReleaseMemObject(buf);
        return ok;
    }

    // layer_norm: matches ops::layer_norm — normalises over the last
    // dimension, one work-group per row.
    bool dispatch_layer_norm(Tensor const& x, Tensor const& w, Tensor const& b,
                              Tensor& out, float eps) const {
        cl_kernel k = kernel_manager_.layer_norm_kernel();
        cl_context ctx = kernel_manager_.context();
        cl_command_queue q = kernel_manager_.queue();
        if (!k || !ctx || !q) return false;
        if (x.ndim() < 1) return false;

        int64_t D = x.dim(x.ndim() - 1);
        int64_t rows = x.numel() / D;
        if (w.numel() != D || b.numel() != D) return false;
        if (out.numel() != x.numel()) out = vlrf::like(x);

        cl_int err = CL_SUCCESS;
        cl_mem bufX = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      x.numel() * sizeof(float), const_cast<float*>(x.data_ptr()), &err);
        cl_mem bufW = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      w.numel() * sizeof(float), const_cast<float*>(w.data_ptr()), &err);
        cl_mem bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      b.numel() * sizeof(float), const_cast<float*>(b.data_ptr()), &err);
        cl_mem bufOut = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                        out.numel() * sizeof(float), nullptr, &err);
        if (!bufX || !bufW || !bufB || !bufOut) {
            if (bufX) clReleaseMemObject(bufX);
            if (bufW) clReleaseMemObject(bufW);
            if (bufB) clReleaseMemObject(bufB);
            if (bufOut) clReleaseMemObject(bufOut);
            return false;
        }

        int r = static_cast<int>(rows);
        int c = static_cast<int>(D);
        // Kernel signature: layer_norm(x, w, b, out, rows, cols, eps)
        clSetKernelArg(k, 0, sizeof(cl_mem), &bufX);
        clSetKernelArg(k, 1, sizeof(cl_mem), &bufW);
        clSetKernelArg(k, 2, sizeof(cl_mem), &bufB);
        clSetKernelArg(k, 3, sizeof(cl_mem), &bufOut);
        clSetKernelArg(k, 4, sizeof(int), &r);
        clSetKernelArg(k, 5, sizeof(int), &c);
        clSetKernelArg(k, 6, sizeof(float), &eps);

        // One work-group (256 threads) per row.
        size_t local = 256;
        size_t global = static_cast<size_t>(rows) * 256;
        cl_int enq_err = clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        bool ok = (enq_err == CL_SUCCESS);
        if (ok) {
            enq_err = clEnqueueReadBuffer(q, bufOut, CL_TRUE, 0, out.numel() * sizeof(float), out.data_ptr(), 0, nullptr, nullptr);
            ok = (enq_err == CL_SUCCESS);
        }
        clReleaseMemObject(bufX);
        clReleaseMemObject(bufW);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufOut);
        return ok;
    }

    bool dispatch_gelu(Tensor& t) const {
        cl_kernel k = kernel_manager_.gelu_kernel();
        cl_context ctx = kernel_manager_.context();
        cl_command_queue q = kernel_manager_.queue();
        if (!k || !ctx || !q) return false;

        int64_t n = t.numel();
        if (n == 0) return true;

        cl_int err = CL_SUCCESS;
        cl_mem buf = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     n * sizeof(float), t.data_ptr(), &err);
        if (!buf) return false;

        int ni = static_cast<int>(n);
        clSetKernelArg(k, 0, sizeof(cl_mem), &buf);
        clSetKernelArg(k, 1, sizeof(int), &ni);

        size_t local = 256;
        size_t global = ((static_cast<size_t>(n) + local - 1) / local) * local;
        err = clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        bool ok = (err == CL_SUCCESS);
        if (ok) {
            err = clEnqueueReadBuffer(q, buf, CL_TRUE, 0, n * sizeof(float), t.data_ptr(), 0, nullptr, nullptr);
            ok = (err == CL_SUCCESS);
        }
        clReleaseMemObject(buf);
        return ok;
    }

    bool dispatch_add_inplace(Tensor& dst, Tensor const& src) const {
        if (dst.numel() != src.numel()) return false;
        cl_kernel k = kernel_manager_.add_inplace_kernel();
        cl_context ctx = kernel_manager_.context();
        cl_command_queue q = kernel_manager_.queue();
        if (!k || !ctx || !q) return false;

        int64_t n = dst.numel();
        if (n == 0) return true;

        cl_int err = CL_SUCCESS;
        cl_mem bufDst = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        n * sizeof(float), dst.data_ptr(), &err);
        cl_mem bufSrc = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        n * sizeof(float), const_cast<float*>(src.data_ptr()), &err);
        if (!bufDst || !bufSrc) {
            if (bufDst) clReleaseMemObject(bufDst);
            if (bufSrc) clReleaseMemObject(bufSrc);
            return false;
        }

        int ni = static_cast<int>(n);
        clSetKernelArg(k, 0, sizeof(cl_mem), &bufDst);
        clSetKernelArg(k, 1, sizeof(cl_mem), &bufSrc);
        clSetKernelArg(k, 2, sizeof(int), &ni);

        size_t local = 256;
        size_t global = ((static_cast<size_t>(n) + local - 1) / local) * local;
        err = clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        bool ok = (err == CL_SUCCESS);
        if (ok) {
            err = clEnqueueReadBuffer(q, bufDst, CL_TRUE, 0, n * sizeof(float), dst.data_ptr(), 0, nullptr, nullptr);
            ok = (err == CL_SUCCESS);
        }
        clReleaseMemObject(bufDst);
        clReleaseMemObject(bufSrc);
        return ok;
    }

    // Fused scaled-dot-product attention. Q, K, V must be 3D [n_h, T_or_ctx_len, d_h]
    // (batched/per-head layout — see sdp_attention's public wrapper above).
    // d_h must fit within the kernel's MAX_DH (256); larger head dims fall
    // back to CPU. Writes attention OUTPUT directly (not intermediate
    // scores), matching ops::sdp_attention's batched-path result shape
    // [n_h, T, d_h].
    bool dispatch_attention(Tensor const& Q, Tensor const& K, Tensor const& V,
                             bool causal, Tensor& out) const {
        if (Q.ndim() != 3 || K.ndim() != 3 || V.ndim() != 3) return false;
        int n_h = Q.dim(0);
        int T = Q.dim(1);
        int d_h = Q.dim(2);
        int ctx_len = K.dim(1);
        if (K.dim(0) != n_h || V.dim(0) != n_h) return false;
        if (K.dim(2) != d_h || V.dim(2) != d_h || V.dim(1) != ctx_len) return false;
        constexpr int kMaxDh = 256;
        if (d_h <= 0 || d_h > kMaxDh) return false;

        cl_kernel k = kernel_manager_.flash_attention_kernel();
        cl_context ctx = kernel_manager_.context();
        cl_command_queue q = kernel_manager_.queue();
        if (!k || !ctx || !q) return false;

        out = Tensor({n_h, T, d_h});

        cl_int err = CL_SUCCESS;
        cl_mem bufQ = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      Q.numel() * sizeof(float), const_cast<float*>(Q.data_ptr()), &err);
        cl_mem bufK = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      K.numel() * sizeof(float), const_cast<float*>(K.data_ptr()), &err);
        cl_mem bufV = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      V.numel() * sizeof(float), const_cast<float*>(V.data_ptr()), &err);
        cl_mem bufOut = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                        out.numel() * sizeof(float), nullptr, &err);
        if (!bufQ || !bufK || !bufV || !bufOut) {
            if (bufQ) clReleaseMemObject(bufQ);
            if (bufK) clReleaseMemObject(bufK);
            if (bufV) clReleaseMemObject(bufV);
            if (bufOut) clReleaseMemObject(bufOut);
            return false;
        }

        float scale = 1.0f / std::sqrt(static_cast<float>(d_h));
        int causal_i = causal ? 1 : 0;
        // Kernel signature: flash_attention(n_h, T, ctx_len, d_h, scale, causal, Q, K, V, out)
        clSetKernelArg(k, 0, sizeof(int), &n_h);
        clSetKernelArg(k, 1, sizeof(int), &T);
        clSetKernelArg(k, 2, sizeof(int), &ctx_len);
        clSetKernelArg(k, 3, sizeof(int), &d_h);
        clSetKernelArg(k, 4, sizeof(float), &scale);
        clSetKernelArg(k, 5, sizeof(int), &causal_i);
        clSetKernelArg(k, 6, sizeof(cl_mem), &bufQ);
        clSetKernelArg(k, 7, sizeof(cl_mem), &bufK);
        clSetKernelArg(k, 8, sizeof(cl_mem), &bufV);
        clSetKernelArg(k, 9, sizeof(cl_mem), &bufOut);

        // One work-group of KV_TILE(64) threads per (head, query-row).
        constexpr size_t kKvTile = 64;
        size_t local[2] = { kKvTile, 1 };
        size_t global[2] = { static_cast<size_t>(n_h) * kKvTile, static_cast<size_t>(T) };
        err = clEnqueueNDRangeKernel(q, k, 2, nullptr, global, local, 0, nullptr, nullptr);
        bool ok = (err == CL_SUCCESS);
        if (ok) {
            err = clEnqueueReadBuffer(q, bufOut, CL_TRUE, 0, out.numel() * sizeof(float), out.data_ptr(), 0, nullptr, nullptr);
            ok = (err == CL_SUCCESS);
        }
        clReleaseMemObject(bufQ);
        clReleaseMemObject(bufK);
        clReleaseMemObject(bufV);
        clReleaseMemObject(bufOut);
        return ok;
    }
#endif // VLRF_USE_OPENCL
};

#if !defined(VLRF_USE_SYCL)
// Minimal SYCLBackend stub when SYCL support is not compiled in, so
// references to SYCLBackend (e.g. in try_sycl()) still compile. Implements
// IBackend so it can be assigned to the registry's active_ pointer.
class SYCLBackend : public IBackend {
    bool available_{false};
    std::string info_{"SYCL: not initialised"};
    CpuBackend fallback_;
public:
    BackendType type()   const override { return BackendType::SYCL; }
    bool is_available()  const override { return available_; }
    std::string device_info() const override { return info_; }

    bool try_init() { available_ = false; info_ = "SYCL support not compiled in (define VLRF_USE_SYCL)"; return false; }

    // Transfers
    Tensor to_device(Tensor const& t) const override { return fallback_.to_device(t); }
    Tensor to_host(Tensor const& t)   const override { return fallback_.to_host(t); }

    // Core ops delegate to CPU fallback
    void gemm(Tensor const& A, Tensor const& B, Tensor& C,
              float alpha, float beta, bool tA, bool tB) const override {
        fallback_.gemm(A,B,C,alpha,beta,tA,tB);
    }
    void layer_norm(Tensor const& x, Tensor const& w, Tensor const& b,
                    Tensor& out, float eps) const override {
        fallback_.layer_norm(x,w,b,out,eps);
    }
    void softmax_inplace(Tensor& t) const override { fallback_.softmax_inplace(t); }
    void gelu_inplace(Tensor& t)    const override { fallback_.gelu_inplace(t); }
    void add_inplace(Tensor& dst, Tensor const& src) const override { fallback_.add_inplace(dst, src); }

    // No SYCL toolchain compiled in — always the CPU flash-attention path.
    Tensor sdp_attention(Tensor const& Q, Tensor const& K, Tensor const& V, bool causal) const override {
        return ops::sdp_attention(Q, K, V, causal);
    }

    // Optional capabilities
    std::vector<std::string> available_devices() const override { return {}; }
    std::string capabilities() const override { return {}; }
    bool supports_usm() const override { return false; }
    void synchronize() const override {}
};
#endif

// -----------------------------------------------------------------------------
// BackendRegistry  — singleton selector
// -----------------------------------------------------------------------------
class BackendRegistry {
    std::shared_ptr<IBackend> active_;
public:
    static BackendRegistry& instance() {
        static BackendRegistry reg;
        return reg;
    }

    BackendRegistry() : active_(std::make_shared<CpuBackend>()) {}

    IBackend& get() { return *active_; }

    void set_cpu() {
        active_ = std::make_shared<CpuBackend>();
    }

    bool try_opencl() {
        auto b = std::make_shared<OpenCLBackend>();
        if (b->try_init()) { active_ = b; return true; }
        return false;
    }

    bool try_sycl() {
        auto b = std::make_shared<SYCLBackend>();
        if (b->try_init()) { active_ = b; return true; }
        return false;
    }

    // Auto-select best available backend
    void auto_select() {
        if (try_sycl())   return;
        if (try_opencl()) return;
        set_cpu();
    }
};

inline IBackend& active_backend() {
    return BackendRegistry::instance().get();
}

} // namespace backend
} // namespace vlrf
