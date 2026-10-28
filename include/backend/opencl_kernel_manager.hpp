// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — OpenCL kernel manager  (opencl_kernel_manager.hpp)
// Handles platform/device selection, runtime kernel decryption, program build, and queue creation.
// =============================================================================

#if defined(VLRF_USE_OPENCL)

#include "opencl_kernels.hpp"
#include <CL/cl.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

namespace vlrf {
namespace backend {

class KernelManager {
public:
    KernelManager() = default;
    ~KernelManager();

    bool init();
    cl_program build_program(const std::string& source, const char* build_options = nullptr) const;
    cl_context context() const { return context_; }
    cl_command_queue queue() const { return queue_; }
    cl_device_id device() const;
    // Actual type of the selected device (CL_DEVICE_TYPE_GPU / _CPU /
    // _ACCELERATOR / etc.) — set by init(), useful for reporting to the
    // user whether OpenCL is actually running on a GPU or fell back to
    // whatever OpenCL device was available.
    cl_device_type device_type() const { return device_type_; }

    // Compiled kernel handles, built lazily on first use and cached for the
    // lifetime of this KernelManager. Returns nullptr if init() wasn't
    // called successfully or the build failed; callers should treat that as
    // "fall back to CPU", not as an error to propagate.
    //
    // NOTE on kernel signatures — these must stay in sync with the actual
    // __kernel argument lists in opencl_kernels.hpp:
    //   gemm(M, N, K, alpha, beta, A, B, C)
    //     — row-major, tightly packed (no separate lda/ldb/ldc); transA/transB
    //       select which of four compiled program variants (NN/NT/TN/TT) is
    //       used — see gemm_kernel(transA, transB) below. C = alpha*op(A)*op(B)
    //       + beta*C.
    //   softmax_inplace(x, rows, cols)
    //   scaled_dot_attention(n_h, T, ctx_len, d_h, scale, causal, Q, K, V, scores)
    //     — legacy QK^T-only kernel, kept for compatibility; not on the hot
    //       dispatch path any more (see flash_attention below).
    //   flash_attention(n_h, T, ctx_len, d_h, scale, causal, Q, K, V, out)
    //     — fused attention: writes attention OUTPUT directly (not scores).
    //   layer_norm(x, w, b, out, rows, cols, eps)
    //   gelu_inplace(x, n)
    //   add_inplace(dst, src, n)
    cl_kernel gemm_kernel(bool transA = false, bool transB = false);
    cl_kernel softmax_kernel();
    cl_kernel attention_kernel();
    cl_kernel flash_attention_kernel();
    cl_kernel layer_norm_kernel();
    cl_kernel gelu_kernel();
    cl_kernel add_inplace_kernel();

private:
    cl_kernel get_or_build(cl_program& prog_slot, cl_kernel& kernel_slot,
                            std::string const& source, const char* entry_point,
                            const char* build_options = nullptr);

    cl_platform_id platform_{nullptr};
    cl_device_id device_{nullptr};
    cl_device_type device_type_{0};
    cl_context context_{nullptr};
    cl_command_queue queue_{nullptr};
    bool initialised_{false};

    // GEMM has four program variants depending on transA/transB, each built
    // (and cached) lazily the first time that combination is requested —
    // most models only ever hit one or two of the four in practice.
    struct GemmVariant {
        cl_program program{nullptr};
        cl_kernel  kernel{nullptr};
    };
    GemmVariant gemm_variants_[2][2]; // [transA][transB]

    cl_program softmax_program_{nullptr};
    cl_kernel  softmax_kernel_{nullptr};
    cl_program attention_program_{nullptr};
    cl_kernel  attention_kernel_{nullptr};
    cl_program flash_attention_program_{nullptr};
    cl_kernel  flash_attention_kernel_{nullptr};
    cl_program layer_norm_program_{nullptr};
    cl_kernel  layer_norm_kernel_{nullptr};
    cl_program gelu_program_{nullptr};
    cl_kernel  gelu_kernel_{nullptr};
    cl_program add_inplace_program_{nullptr};
    cl_kernel  add_inplace_kernel_{nullptr};
};

inline KernelManager::~KernelManager() {
    for (int a = 0; a < 2; ++a) {
        for (int b = 0; b < 2; ++b) {
            if (gemm_variants_[a][b].kernel)  clReleaseKernel(gemm_variants_[a][b].kernel);
            if (gemm_variants_[a][b].program) clReleaseProgram(gemm_variants_[a][b].program);
        }
    }
    if (softmax_kernel_) clReleaseKernel(softmax_kernel_);
    if (softmax_program_) clReleaseProgram(softmax_program_);
    if (attention_kernel_) clReleaseKernel(attention_kernel_);
    if (attention_program_) clReleaseProgram(attention_program_);
    if (flash_attention_kernel_) clReleaseKernel(flash_attention_kernel_);
    if (flash_attention_program_) clReleaseProgram(flash_attention_program_);
    if (layer_norm_kernel_) clReleaseKernel(layer_norm_kernel_);
    if (layer_norm_program_) clReleaseProgram(layer_norm_program_);
    if (gelu_kernel_) clReleaseKernel(gelu_kernel_);
    if (gelu_program_) clReleaseProgram(gelu_program_);
    if (add_inplace_kernel_) clReleaseKernel(add_inplace_kernel_);
    if (add_inplace_program_) clReleaseProgram(add_inplace_program_);
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
    if (device_) clReleaseDevice(device_);
}

inline bool KernelManager::init() {
    cl_uint num_platforms = 0;
    if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        return false;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    if (clGetPlatformIDs(num_platforms, platforms.data(), nullptr) != CL_SUCCESS) {
        return false;
    }

    // Prefer a real GPU device (the common/fast case), but don't treat the
    // absence of one as "no OpenCL available" — plenty of valid OpenCL
    // setups only expose CL_DEVICE_TYPE_CPU (e.g. pocl) or
    // CL_DEVICE_TYPE_ACCELERATOR (e.g. some FPGA/Xeon Phi ICDs), and those
    // are still usable, just slower than a GPU. Two passes: first look for
    // CL_DEVICE_TYPE_GPU across all platforms, then fall back to
    // CL_DEVICE_TYPE_ALL (any device type) across all platforms.
    for (cl_device_type wanted : {(cl_device_type)CL_DEVICE_TYPE_GPU, (cl_device_type)CL_DEVICE_TYPE_ALL}) {
        for (auto platform : platforms) {
            cl_uint num_devices = 0;
            if (clGetDeviceIDs(platform, wanted, 0, nullptr, &num_devices) != CL_SUCCESS || num_devices == 0) {
                continue;
            }
            std::vector<cl_device_id> devices(num_devices);
            if (clGetDeviceIDs(platform, wanted, num_devices, devices.data(), nullptr) != CL_SUCCESS) {
                continue;
            }
            platform_ = platform;
            device_ = devices[0];
            break;
        }
        if (device_) break;
    }

    if (!device_) {
        return false;
    }

    // Record the device's REAL type (not which search pass found it —
    // CL_DEVICE_TYPE_ALL is a bitmask matching everything, so it must not
    // be mistaken for the device's actual type when reporting GPU vs
    // CPU/accelerator to the user).
    clGetDeviceInfo(device_, CL_DEVICE_TYPE, sizeof(device_type_), &device_type_, nullptr);

    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, nullptr);
    if (!context_) {
        return false;
    }

    queue_ = clCreateCommandQueue(context_, device_, 0, nullptr);
    // NOTE: clCreateCommandQueue is deprecated since OpenCL 2.0 in favor of
    // clCreateCommandQueueWithProperties. Kept as clCreateCommandQueue here
    // deliberately for now — it remains functional on all OpenCL 1.x/2.x/3.0
    // ICDs and avoids taking on an extra CL_TARGET_OPENCL_VERSION build
    // requirement across the whole project just for this one call. Revisit
    // if the build starts requiring strict OpenCL 3.0 headers.
    if (!queue_) {
        clReleaseContext(context_);
        context_ = nullptr;
        return false;
    }

    initialised_ = true;
    return true;
}

inline cl_program KernelManager::build_program(const std::string& source, const char* build_options) const {
    if (!initialised_) {
        throw std::runtime_error("KernelManager is not initialised");
    }

    const char* src_ptr = source.c_str();
    size_t src_len = source.size();
    cl_int err = 0;
    cl_program program = clCreateProgramWithSource(context_, 1, &src_ptr, &src_len, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL program from source");
    }

    err = clBuildProgram(program, 1, &device_, build_options, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::string build_log(log_size, '\0');
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, log_size, &build_log[0], nullptr);
        clReleaseProgram(program);
        throw std::runtime_error("OpenCL build failed: " + build_log);
    }
    return program;
}

inline cl_device_id KernelManager::device() const {
    return device_;
}

inline cl_kernel KernelManager::get_or_build(cl_program& prog_slot, cl_kernel& kernel_slot,
                                              std::string const& source, const char* entry_point,
                                              const char* build_options) {
    if (kernel_slot) return kernel_slot;
    if (!initialised_) return nullptr;

    try {
        prog_slot = build_program(source, build_options);
    } catch (std::exception const&) {
        // Build failure (bad device, driver rejects the source, etc.) — the
        // caller falls back to CPU rather than seeing an exception, matching
        // how the SYCL backend degrades gracefully on setup failure.
        prog_slot = nullptr;
        return nullptr;
    }

    cl_int err = CL_SUCCESS;
    cl_kernel k = clCreateKernel(prog_slot, entry_point, &err);
    if (err != CL_SUCCESS || !k) {
        clReleaseProgram(prog_slot);
        prog_slot = nullptr;
        return nullptr;
    }
    kernel_slot = k;
    return kernel_slot;
}

inline cl_kernel KernelManager::gemm_kernel(bool transA, bool transB) {
    GemmVariant& v = gemm_variants_[transA ? 1 : 0][transB ? 1 : 0];
    if (v.kernel) return v.kernel;
    const char* opts =
        (transA && transB)  ? "-DTRANSPOSE_A=1 -DTRANSPOSE_B=1" :
        (transA && !transB) ? "-DTRANSPOSE_A=1 -DTRANSPOSE_B=0" :
        (!transA && transB) ? "-DTRANSPOSE_A=0 -DTRANSPOSE_B=1" :
                               "-DTRANSPOSE_A=0 -DTRANSPOSE_B=0";
    return get_or_build(v.program, v.kernel, opencl::gemm_kernel_source(), "gemm", opts);
}

inline cl_kernel KernelManager::softmax_kernel() {
    return get_or_build(softmax_program_, softmax_kernel_, opencl::softmax_kernel_source(), "softmax_inplace");
}

inline cl_kernel KernelManager::attention_kernel() {
    return get_or_build(attention_program_, attention_kernel_, opencl::attention_kernel_source(), "scaled_dot_attention");
}

inline cl_kernel KernelManager::flash_attention_kernel() {
    return get_or_build(flash_attention_program_, flash_attention_kernel_,
                         opencl::flash_attention_kernel_source(), "flash_attention");
}

inline cl_kernel KernelManager::layer_norm_kernel() {
    return get_or_build(layer_norm_program_, layer_norm_kernel_, opencl::layer_norm_kernel_source(), "layer_norm");
}

inline cl_kernel KernelManager::gelu_kernel() {
    return get_or_build(gelu_program_, gelu_kernel_, opencl::gelu_kernel_source(), "gelu_inplace");
}

inline cl_kernel KernelManager::add_inplace_kernel() {
    return get_or_build(add_inplace_program_, add_inplace_kernel_, opencl::add_inplace_kernel_source(), "add_inplace");
}

} // namespace backend
} // namespace vlrf

#endif // defined(VLRF_USE_OPENCL)
