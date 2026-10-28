// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — Backend interface + CPU backend  (ibackend.hpp)
// Split out of backend.hpp so that GPU backend headers (sycl_backend.hpp,
// opencl_kernel_manager.hpp) can be #include'd with IBackend/CpuBackend
// already fully defined, WITHOUT needing to be textually nested inside
// namespace vlrf::backend themselves. A header that opens its own
// `namespace vlrf { namespace backend { ... } }` and is #include'd from a
// point already inside that same namespace still works for the C++-level
// declarations (namespaces merge), but any *system* header included inside
// that nesting (e.g. <optional>, <unordered_map>) ends up textually inside
// vlrf::backend too, which is fragile and breaks with some standard library
// implementations. Keeping this split means every backend header's own
// #include block runs at true global scope.
// =============================================================================
#include "tensor.hpp"
#include "ops.hpp"
#include <cstdio>
#include <string>
#include <memory>
#include <vector>

namespace vlrf {
namespace backend {

// -----------------------------------------------------------------------------
// Backend type enum
// -----------------------------------------------------------------------------
enum class BackendType { CPU, OPENCL, SYCL };

inline const char* backend_name(BackendType t) {
    switch (t) {
        case BackendType::CPU:    return "cpu";
        case BackendType::OPENCL: return "opencl";
        case BackendType::SYCL:   return "sycl";
    }
    return "unknown";
}

// -----------------------------------------------------------------------------
// IBackend  — interface every backend must implement
// -----------------------------------------------------------------------------
struct IBackend {
    virtual ~IBackend() = default;

    virtual BackendType type() const = 0;
    virtual std::string device_info() const = 0;
    virtual bool        is_available() const = 0;
    virtual std::vector<std::string> available_devices() const { return {}; }
    virtual std::string capabilities() const { return {}; }
    virtual bool supports_usm() const { return false; }
    virtual void synchronize() const {}

    // Transfer
    virtual Tensor to_device(Tensor const& host_t) const = 0;
    virtual Tensor to_host(Tensor const& dev_t)   const = 0;

    // Core ops (backends can override with accelerated versions)
    virtual void gemm(Tensor const& A, Tensor const& B, Tensor& C,
                      float alpha, float beta,
                      bool transA, bool transB) const = 0;

    virtual void layer_norm(Tensor const& x, Tensor const& w, Tensor const& b,
                            Tensor& out, float eps) const = 0;

    virtual void softmax_inplace(Tensor& t) const = 0;
    virtual void gelu_inplace(Tensor& t)   const = 0;
    virtual void add_inplace(Tensor& dst, Tensor const& src) const = 0;

    // Fused scaled-dot-product (flash) attention. Q, K, V: [n_h, T_or_ctx_len,
    // d_h] batched (3D) layout — see AttentionBlock::sdp_heads. Default
    // implementation just calls the CPU flash-attention path in ops.hpp, so
    // backends that don't have a fused kernel (or fail to dispatch one) need
    // not override this at all. GPU backends override it to attempt a real
    // on-device fused kernel first, falling back to this same CPU path on
    // any unsupported shape or dispatch failure.
    virtual Tensor sdp_attention(Tensor const& Q, Tensor const& K, Tensor const& V,
                                  bool causal) const {
        return ops::sdp_attention(Q, K, V, causal);
    }
};

// -----------------------------------------------------------------------------
// CpuBackend  — wraps ops.hpp inline kernels
// -----------------------------------------------------------------------------
class CpuBackend : public IBackend {
public:
    BackendType type()   const override { return BackendType::CPU; }
    bool is_available()  const override { return true; }

    std::string device_info() const override {
        char buf[128];
        snprintf(buf, sizeof(buf), "CPU (%d hardware threads)", ops::hw_threads());
        return buf;
    }

    std::string capabilities() const override {
        // Every op below runs the same hand-written/oneMKL GEMM kernel
        // reported by simd::active_kernel_name() — there's no per-op split
        // to report here the way OpenCLBackend/SYCLBackend have, since CPU
        // is the fallback path everything else falls back TO.
        std::string s = "CPU backend; gemm kernel: ";
        s += simd::active_kernel_name();
        return s;
    }

    // CPU tensors are already on "device" — these are identity ops
    Tensor to_device(Tensor const& t) const override { return t; }
    Tensor to_host(Tensor const& t)   const override { return t; }

    void gemm(Tensor const& A, Tensor const& B, Tensor& C,
              float alpha, float beta, bool tA, bool tB) const override {
        ops::gemm(A, B, C, alpha, beta, tA, tB);
    }

    void layer_norm(Tensor const& x, Tensor const& w, Tensor const& b,
                    Tensor& out, float eps) const override {
        ops::layer_norm(x, w, b, out, eps);
    }

    void softmax_inplace(Tensor& t) const override { ops::softmax_inplace(t); }
    void gelu_inplace(Tensor& t)    const override { ops::gelu_inplace(t); }
    void add_inplace(Tensor& dst, Tensor const& src) const override {
        ops::add_inplace(dst, src);
    }
};

} // namespace backend
} // namespace vlrf
