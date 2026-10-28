// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — SYCL 2020 backend  (sycl_backend.hpp)
// Implements a portable heterogeneous backend using SYCL 2020 APIs with
// device discovery, queue management, event synchronization, USM allocation,
// and a set of basic accelerated kernels. The backend is designed to work
// with Intel oneAPI DPC++, AdaptiveCpp (hipSYCL), and other SYCL 2020
// implementations where the compiler exposure is available.
// =============================================================================

#include "tensor.hpp"
#include "ops.hpp"
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(VLRF_USE_SYCL)
#  include <sycl/sycl.hpp>
#endif

namespace vlrf {
namespace backend {

struct IBackend;
enum class BackendType;
class CpuBackend;

#if defined(VLRF_USE_SYCL)

namespace {

inline std::string sanitize(std::string value) {
    if (value.empty()) return "unknown";
    value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
        return c == '\n' || c == '\r' || c == '\t';
    }), value.end());
    return value;
}

} // namespace

// detail (not anonymous): AllocKey/AllocKeyHash are used as the type of a
// SYCLBackend member (allocations_ below), and SYCLBackend has external
// linkage. A member whose type has internal (anonymous-namespace) linkage
// while the enclosing class has external linkage is well-formed but flagged
// by -Wsubobject-linkage since it can cause ODR surprises across
// translation units; a named namespace avoids the warning and the risk.
namespace detail {

// Identifies a distinct device-buffer slot: same underlying TensorStorage
// but a different offset (e.g. a row-slice view) must NOT collide with the
// buffer used for the parent tensor or another slice.
struct AllocKey {
    TensorStorage* storage{nullptr};
    int64_t offset{0};

    bool operator==(AllocKey const& o) const {
        return storage == o.storage && offset == o.offset;
    }
};

struct AllocKeyHash {
    size_t operator()(AllocKey const& k) const {
        size_t h1 = std::hash<void const*>()(k.storage);
        size_t h2 = std::hash<int64_t>()(k.offset);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};

} // namespace detail

class SYCLBackend : public IBackend {
public:
    struct DeviceDescriptor {
        std::string name;
        std::string vendor;
        std::string driver_version;
        bool is_cpu{false};
        bool is_gpu{false};
        bool is_accelerator{false};
        std::string type_string;
    };

    struct USMAllocation {
        sycl::queue* queue{nullptr};
        void* ptr{nullptr};
        std::size_t bytes{0};
        int64_t elements{0};  // element count this buffer was sized for

        USMAllocation() = default;
        USMAllocation(sycl::queue* q, std::size_t n) : queue(q), bytes(n) {}
        ~USMAllocation() {
            if (ptr && queue) {
                try {
                    sycl::free(ptr, *queue);
                } catch (...) {
                }
            }
        }

        USMAllocation(USMAllocation const&) = delete;
        USMAllocation& operator=(USMAllocation const&) = delete;
        USMAllocation(USMAllocation&&) = delete;
        USMAllocation& operator=(USMAllocation&&) = delete;
    };

    SYCLBackend() = default;

    BackendType type() const override { return BackendType::SYCL; }
    bool is_available() const override { return available_; }
    std::string device_info() const override { return info_; }

    std::vector<std::string> available_devices() const override {
        std::vector<std::string> out;
        out.reserve(devices_.size());
        for (auto const& d : devices_) out.push_back(d.name);
        return out;
    }

    std::string capabilities() const override {
        std::ostringstream oss;
        oss << "SYCL backend; queue=" << (queue_ ? "ready" : "not-ready")
            << "; usm=" << (supports_usm() ? "yes" : "no");
        if (!devices_.empty()) {
            oss << "; devices=" << devices_.size();
        }
        // Per-op dispatch: every op below has a real SYCL kernel (see
        // gemm/layer_norm/softmax_inplace/gelu_inplace/add_inplace further
        // down this file) — unlike OpenCLBackend, there's no partial CPU
        // fallback by op here, only a whole-backend fallback to CpuBackend
        // when queue_ isn't ready (see is_available()). ops with no
        // IBackend method at all (bias-add, RMSNorm, SiLU/ReLU, transpose,
        // ...) are outside any backend's control and always run on CPU
        // regardless — see include/model/compute.hpp's header comment for
        // the full list.
        //
        // The reported device kind reflects whatever SYCL actually selected
        // (see select_preferred_device()) — that may be a CPU or
        // accelerator device, not necessarily a GPU, so this must not be
        // hardcoded to "GPU".
        const std::string dev = queue_ ? selected_device_type(queue_->get_device()) : std::string("cpu");
        const std::string dev_upper = dev.empty() ? dev : (std::string(1, static_cast<char>(std::toupper(dev[0]))) + dev.substr(1));
        oss << "\n  gemm            : " << (queue_ ? (dev_upper + " (SYCL kernel)") : "CPU (fallback — no active queue)")
            << "\n  layer_norm      : " << (queue_ ? (dev_upper + " (SYCL kernel)") : "CPU (fallback — no active queue)")
            << "\n  softmax_inplace : " << (queue_ ? (dev_upper + " (SYCL kernel)") : "CPU (fallback — no active queue)")
            << "\n  gelu_inplace    : " << (queue_ ? (dev_upper + " (SYCL kernel)") : "CPU (fallback — no active queue)")
            << "\n  add_inplace     : " << (queue_ ? (dev_upper + " (SYCL kernel)") : "CPU (fallback — no active queue)")
            << "\n  attention (fused): " << (queue_ ? (dev_upper + " (fused flash_attention kernel: QK^T + online softmax + weighted-V in one launch)") : "CPU (fallback — no active queue)");
        return oss.str();
    }

    bool supports_usm() const override { return true; }

    void synchronize() const override {
        if (!queue_) return;
        // sycl::event::wait() is NOT const in the SYCL 2020 spec (unlike,
        // say, std::future::wait()), so it can't be called through a
        // `const sycl::event&` — hence the non-const `auto&` here rather
        // than `auto const&`. pending_events_ is itself `mutable`, so this
        // is fine to do from synchronize()'s const method body.
        for (auto& e : pending_events_) {
            e.wait();
        }
        pending_events_.clear();
    }

    bool try_init() {
        try {
            std::vector<sycl::device> all_devices;
            try {
                all_devices = sycl::device::get_devices(sycl::info::device_type::all);
            } catch (std::exception const&) {
                all_devices.clear();
            }

            if (all_devices.empty()) {
                available_ = false;
                info_ = "SYCL: no devices detected";
                return false;
            }

            devices_.clear();
            for (auto const& d : all_devices) {
                DeviceDescriptor dd;
                dd.name = sanitize(d.get_info<sycl::info::device::name>());
                dd.vendor = sanitize(d.get_info<sycl::info::device::vendor>());
                dd.driver_version = sanitize(d.get_info<sycl::info::device::driver_version>());
                dd.is_cpu = d.is_cpu();
                dd.is_gpu = d.is_gpu();
                dd.is_accelerator = d.is_accelerator();
                dd.type_string = dd.is_cpu ? "cpu" : (dd.is_gpu ? "gpu" : (dd.is_accelerator ? "accelerator" : "other"));
                devices_.push_back(dd);
            }

            auto best = select_preferred_device(all_devices);
            queue_.emplace(best, sycl::property_list{sycl::property::queue::in_order{}});
            available_ = true;
            info_ = std::string("SYCL: ") + selected_device_name(best) + " [" + selected_device_type(best) + "]";
            return true;
        } catch (std::exception const& ex) {
            available_ = false;
            info_ = std::string("SYCL initialization failed: ") + ex.what();
            return false;
        } catch (...) {
            available_ = false;
            info_ = "SYCL initialization failed: unknown error";
            return false;
        }
    }

    // NOTE on caching: allocations are cached by the *host* tensor's storage
    // identity (host_t.storage), not by any newly-constructed device-side
    // Tensor. Tensor(Shape) always allocates a fresh TensorStorage, so keying
    // off a freshly-built tensor would never hit the cache and would leak a
    // new USM buffer on every call. Keying off the host tensor means repeat
    // calls for the same logical tensor (e.g. persistent weights) reuse one
    // USM buffer instead of leaking.
    Tensor to_device(Tensor const& host_t) const override {
        if (host_t.empty() || !queue_) return host_t;
        auto* alloc = ensure_allocation_for(host_t);
        if (!alloc || !alloc->ptr) return host_t;

        Tensor dev(host_t.shape);
        if (dev.storage->data) {
            delete[] dev.storage->data;
        }
        dev.storage->data = static_cast<float*>(alloc->ptr);
        dev.storage->owned = false;
        copy_to_device(host_t, alloc);
        return dev;
    }

    Tensor to_host(Tensor const& dev_t) const override {
        if (dev_t.empty() || !queue_) return dev_t;
        // dev_t is device-resident: its storage->data already points at a
        // USM allocation. Copy straight out of it rather than going through
        // the allocation cache (which is keyed by host storage identity).
        Tensor host(dev_t.shape);
        std::size_t bytes = static_cast<std::size_t>(dev_t.numel()) * sizeof(float);
        if (bytes > 0) {
            auto e = queue_->memcpy(host.data_ptr(), dev_t.data_ptr(), bytes);
            pending_events_.push_back(e);
            e.wait();
        }
        return host;
    }

    void gemm(Tensor const& A, Tensor const& B, Tensor& C,
              float alpha, float beta, bool transA, bool transB) const override {
        if (!queue_ || !available_) {
            fallback_.gemm(A, B, C, alpha, beta, transA, transB);
            return;
        }
        if (A.ndim() != 2 || B.ndim() != 2 || C.ndim() != 2) {
            fallback_.gemm(A, B, C, alpha, beta, transA, transB);
            return;
        }

        // A and B are read-only inputs, staged in via the storage-keyed
        // cache. C is read (for the `beta * C` term) *and* written, so it
        // also needs its current contents copied in, not just a bare
        // allocation.
        auto* a_alloc = ensure_allocation_for(A);
        auto* b_alloc = ensure_allocation_for(B);
        auto* c_alloc = ensure_allocation_for(C);
        if (!a_alloc || !b_alloc || !c_alloc || !a_alloc->ptr || !b_alloc->ptr || !c_alloc->ptr) {
            fallback_.gemm(A, B, C, alpha, beta, transA, transB);
            return;
        }
        copy_to_device(A, a_alloc);
        copy_to_device(B, b_alloc);
        copy_to_device(C, c_alloc);

        int M = transA ? A.dim(1) : A.dim(0);
        int K = transA ? A.dim(0) : A.dim(1);
        int N = transB ? B.dim(0) : B.dim(1);
        auto* pA = static_cast<float*>(a_alloc->ptr);
        auto* pB = static_cast<float*>(b_alloc->ptr);
        auto* pC = static_cast<float*>(c_alloc->ptr);
        int lda = A.dim(1);
        int ldb = B.dim(1);
        int ldc = C.dim(1);

        auto e = queue_->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(static_cast<std::size_t>(M), static_cast<std::size_t>(N)), [=](sycl::id<2> it) {
                int row = it[0];
                int col = it[1];
                float acc = 0.f;
                for (int k = 0; k < K; ++k) {
                    float a_val = transA ? pA[k * lda + row] : pA[row * lda + k];
                    float b_val = transB ? pB[col * ldb + k] : pB[k * ldb + col];
                    acc += a_val * b_val;
                }
                pC[row * ldc + col] = beta * pC[row * ldc + col] + alpha * acc;
            });
        });
        pending_events_.push_back(e);
        synchronize();
        copy_to_host(C, c_alloc);
    }

    void layer_norm(Tensor const& x, Tensor const& w, Tensor const& b,
                    Tensor& out, float eps) const override {
        if (!queue_ || !available_) {
            fallback_.layer_norm(x, w, b, out, eps);
            return;
        }
        if (x.ndim() < 2 || w.ndim() != 1 || b.ndim() != 1) {
            fallback_.layer_norm(x, w, b, out, eps);
            return;
        }

        // x, w, b are read-only inputs: stage their contents in.
        // out is write-only from the kernel's perspective — it needs a
        // correctly-sized device buffer (cached against its own storage
        // identity) but nothing needs to be copied in first.
        auto* x_alloc = ensure_allocation_for(x);
        auto* w_alloc = ensure_allocation_for(w);
        auto* b_alloc = ensure_allocation_for(b);
        auto* o_alloc = ensure_output_allocation(out);
        if (!x_alloc || !w_alloc || !b_alloc || !o_alloc || !x_alloc->ptr || !w_alloc->ptr || !b_alloc->ptr || !o_alloc->ptr) {
            fallback_.layer_norm(x, w, b, out, eps);
            return;
        }
        copy_to_device(x, x_alloc);
        copy_to_device(w, w_alloc);
        copy_to_device(b, b_alloc);

        int64_t D = x.dim(x.ndim() - 1);
        int64_t rows = x.numel() / D;
        auto* px = static_cast<float*>(x_alloc->ptr);
        auto* pw = static_cast<float*>(w_alloc->ptr);
        auto* pb = static_cast<float*>(b_alloc->ptr);
        auto* po = static_cast<float*>(o_alloc->ptr);

        auto e = queue_->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<2>(static_cast<std::size_t>(rows), static_cast<std::size_t>(D)), [=](sycl::id<2> it) {
                int64_t r = it[0];
                int64_t j = it[1];
                float mean = 0.f;
                for (int64_t k = 0; k < D; ++k) mean += px[r * D + k];
                mean /= static_cast<float>(D);
                float var = 0.f;
                for (int64_t k = 0; k < D; ++k) {
                    float diff = px[r * D + k] - mean;
                    var += diff * diff;
                }
                var = var / static_cast<float>(D) + eps;
                float inv_std = 1.f / sycl::sqrt(var);
                po[r * D + j] = (px[r * D + j] - mean) * inv_std * pw[j] + pb[j];
            });
        });
        pending_events_.push_back(e);
        synchronize();
        copy_to_host(out, o_alloc);
    }

    void softmax_inplace(Tensor& t) const override {
        if (!queue_ || !available_) {
            fallback_.softmax_inplace(t);
            return;
        }
        // In-place op: read current contents in, run the kernel on the same
        // USM buffer, copy the result back into the same host tensor `t`.
        auto* alloc = ensure_allocation_for(t);
        if (!alloc || !alloc->ptr) {
            fallback_.softmax_inplace(t);
            return;
        }
        copy_to_device(t, alloc);
        int64_t D = t.dim(t.ndim() - 1);
        int64_t rows = t.numel() / D;
        auto* p = static_cast<float*>(alloc->ptr);
        // One work-item per ROW, not per (row, column). The previous 2D
        // launch assigned one work-item per element, but every work-item
        // in a row independently recomputed the *entire* row's max/sum and
        // then all D of them wrote to the same D output locations with no
        // synchronization — that's O(D) redundant work-items all racing to
        // write the same addresses (undefined behavior under the SYCL/OpenCL
        // memory model, not just wasted work, even though in practice they'd
        // typically compute matching values).
        auto e = queue_->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(static_cast<std::size_t>(rows)), [=](sycl::id<1> it) {
                int64_t r = it[0];
                float mx = -1e30f;
                for (int64_t k = 0; k < D; ++k) {
                    float v = p[r * D + k];
                    if (v > mx) mx = v;
                }
                float sum = 0.f;
                for (int64_t k = 0; k < D; ++k) {
                    float v = sycl::exp(p[r * D + k] - mx);
                    p[r * D + k] = v;
                    sum += v;
                }
                float inv = 1.f / (sum + 1e-9f);
                for (int64_t k = 0; k < D; ++k) p[r * D + k] *= inv;
            });
        });
        pending_events_.push_back(e);
        synchronize();
        copy_to_host(t, alloc);
    }

    void gelu_inplace(Tensor& t) const override {
        if (!queue_ || !available_) {
            fallback_.gelu_inplace(t);
            return;
        }
        auto* alloc = ensure_allocation_for(t);
        if (!alloc || !alloc->ptr) {
            fallback_.gelu_inplace(t);
            return;
        }
        copy_to_device(t, alloc);
        auto* p = static_cast<float*>(alloc->ptr);
        int64_t n = t.numel();
        auto e = queue_->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(static_cast<std::size_t>(n)), [=](sycl::id<1> it) {
                int64_t i = it[0];
                float x = p[i];
                constexpr float c1 = 0.7978845608028654f;
                constexpr float c2 = 0.044715f;
                float inner = c1 * (x + c2 * x * x * x);
                p[i] = 0.5f * x * (1.f + sycl::tanh(inner));
            });
        });
        pending_events_.push_back(e);
        synchronize();
        copy_to_host(t, alloc);
    }

    void add_inplace(Tensor& dst, Tensor const& src) const override {
        if (!queue_ || !available_) {
            fallback_.add_inplace(dst, src);
            return;
        }
        auto* d_alloc = ensure_allocation_for(dst);
        auto* s_alloc = ensure_allocation_for(src);
        if (!d_alloc || !s_alloc || !d_alloc->ptr || !s_alloc->ptr) {
            fallback_.add_inplace(dst, src);
            return;
        }
        copy_to_device(dst, d_alloc);
        copy_to_device(src, s_alloc);
        int64_t n = dst.numel();
        auto* d = static_cast<float*>(d_alloc->ptr);
        auto* s = static_cast<float*>(s_alloc->ptr);
        auto e = queue_->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(static_cast<std::size_t>(n)), [=](sycl::id<1> it) {
                int64_t i = it[0];
                d[i] += s[i];
            });
        });
        pending_events_.push_back(e);
        synchronize();
        copy_to_host(dst, d_alloc);
    }

    // Fused scaled-dot-product attention. Not part of IBackend (like
    // OpenCLBackend::sdp_attention, ops.hpp's sdp_attention is a free
    // function, not a virtual method), but exposed here so callers that
    // already hold a SYCLBackend can opt into the fused GPU path directly;
    // falls back to ops::sdp_attention on CPU / on any dispatch failure.
    // Q, K, V: [n_h, T_or_ctx_len, d_h] batched (3D) layout — matches the
    // per-head view layout ops::sdp_attention's batched path builds via
    // Tensor::from_ptr, and matches what AttentionBlock::sdp_heads passes
    // to OpenCLBackend's equivalent method.
    Tensor sdp_attention(Tensor const& Q, Tensor const& K, Tensor const& V, bool causal) const override {
        if (queue_ && available_) {
            try {
                Tensor out;
                if (dispatch_attention(Q, K, V, causal, out)) return out;
            } catch (std::exception const&) {
                // Fall through to CPU. A dispatch-time exception (e.g. the
                // SYCL runtime rejecting the launch on this particular
                // device) must not propagate out of an op entry point that
                // every caller assumes is noexcept-equivalent -- same
                // contract as gemm/layer_norm/etc above.
            } catch (...) {
            }
        }
        return ops::sdp_attention(Q, K, V, causal);
    }

private:
    mutable std::optional<sycl::queue> queue_;
    mutable std::vector<sycl::event> pending_events_;
    std::vector<DeviceDescriptor> devices_;
    bool available_{false};
    std::string info_{"SYCL: not initialised"};
    mutable CpuBackend fallback_;

    // Real dispatch through a fused SYCL flash-attention kernel. Returns
    // false (falls back to CPU via the public sdp_attention() wrapper) on
    // any shape the kernel doesn't support, matching the pattern used by
    // gemm/layer_norm/softmax_inplace/gelu_inplace/add_inplace above and by
    // OpenCLBackend::dispatch_attention.
    //
    // Algorithm mirrors OpenCLBackend::dispatch_attention's flash_attention
    // OpenCL kernel exactly (same scale, same causal-mask condition
    // `ki > qi + ctx_len - T`, same online-softmax rescaling), so results
    // agree with both ops::sdp_attention and AttentionBlock::flash_attn_head
    // up to float rounding. One work-group of kKvTile work-items per (head,
    // query-row); the group cooperatively streams over K/V in kKvTile-row
    // tiles, keeping only a running (max, sum, d_h-wide accumulator) in
    // SYCL local memory — the full [T, ctx_len] score matrix is never
    // materialized. d_h must fit in local memory (kMaxDh); larger head
    // dims fall back to CPU.
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
        if (n_h <= 0 || T <= 0 || ctx_len <= 0) return false;

        out = Tensor({n_h, T, d_h});

        auto* q_alloc = ensure_allocation_for(Q);
        auto* k_alloc = ensure_allocation_for(K);
        auto* v_alloc = ensure_allocation_for(V);
        auto* o_alloc = ensure_output_allocation(out);
        if (!q_alloc || !k_alloc || !v_alloc || !o_alloc ||
            !q_alloc->ptr || !k_alloc->ptr || !v_alloc->ptr || !o_alloc->ptr) {
            return false;
        }
        copy_to_device(Q, q_alloc);
        copy_to_device(K, k_alloc);
        copy_to_device(V, v_alloc);

        auto* pQ = static_cast<float const*>(q_alloc->ptr);
        auto* pK = static_cast<float const*>(k_alloc->ptr);
        auto* pV = static_cast<float const*>(v_alloc->ptr);
        auto* pO = static_cast<float*>(o_alloc->ptr);

        constexpr int kKvTile = 64;
        float scale = 1.0f / std::sqrt(static_cast<float>(d_h));
        int causal_i = causal ? 1 : 0;

        try {
            auto e = queue_->submit([&](sycl::handler& h) {
                // Local (work-group-shared) scratch: kKvTile-wide score/
                // reduction buffers, plus a d_h-wide running output
                // accumulator — same roles as the OpenCL kernel's
                // __local scores_tile/red_buf/acc_out.
                sycl::local_accessor<float, 1> scores_tile(sycl::range<1>(kKvTile), h);
                sycl::local_accessor<float, 1> red_buf(sycl::range<1>(kKvTile), h);
                sycl::local_accessor<float, 1> acc_out(sycl::range<1>(kMaxDh), h);
                sycl::local_accessor<float, 1> run_max(sycl::range<1>(1), h);
                sycl::local_accessor<float, 1> run_sum(sycl::range<1>(1), h);

                sycl::range<2> local{static_cast<std::size_t>(kKvTile), 1};
                sycl::range<2> global{static_cast<std::size_t>(n_h) * kKvTile,
                                       static_cast<std::size_t>(T)};

                h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> it) {
                    int hgroup = static_cast<int>(it.get_group(0));
                    int qi     = static_cast<int>(it.get_group(1));
                    int lid    = static_cast<int>(it.get_local_id(0));
                    if (hgroup >= n_h || qi >= T) return;

                    int qbase  = hgroup * T * d_h + qi * d_h;
                    int kvbase = hgroup * ctx_len * d_h;

                    for (int d = lid; d < d_h; d += kKvTile) acc_out[d] = 0.0f;
                    if (lid == 0) { run_max[0] = -std::numeric_limits<float>::infinity(); run_sum[0] = 0.0f; }
                    it.barrier(sycl::access::fence_space::local_space);

                    int n_tiles = (ctx_len + kKvTile - 1) / kKvTile;
                    for (int t = 0; t < n_tiles; ++t) {
                        int kv_off = t * kKvTile;
                        int ki = kv_off + lid;

                        float score = -std::numeric_limits<float>::infinity();
                        if (ki < ctx_len) {
                            bool masked = causal_i && (ki > qi + ctx_len - T);
                            if (!masked) {
                                float acc = 0.0f;
                                int kval = kvbase + ki * d_h;
                                for (int d = 0; d < d_h; ++d) acc += pQ[qbase + d] * pK[kval + d];
                                score = acc * scale;
                            }
                        }
                        scores_tile[lid] = score;
                        it.barrier(sycl::access::fence_space::local_space);

                        // Block-max reduction across the tile.
                        red_buf[lid] = score;
                        it.barrier(sycl::access::fence_space::local_space);
                        for (int stride = kKvTile / 2; stride > 0; stride >>= 1) {
                            if (lid < stride) {
                                float other = red_buf[lid + stride];
                                if (other > red_buf[lid]) red_buf[lid] = other;
                            }
                            it.barrier(sycl::access::fence_space::local_space);
                        }
                        float tile_max = red_buf[0];
                        it.barrier(sycl::access::fence_space::local_space);

                        float new_max = sycl::max(run_max[0], tile_max);
                        float correction = (run_max[0] == -std::numeric_limits<float>::infinity())
                                            ? 0.0f : sycl::exp(run_max[0] - new_max);
                        float p = (scores_tile[lid] == -std::numeric_limits<float>::infinity())
                                  ? 0.0f : sycl::exp(scores_tile[lid] - new_max);

                        // Block-sum reduction of the exp'd scores (softmax
                        // denominator contribution from this tile).
                        red_buf[lid] = p;
                        it.barrier(sycl::access::fence_space::local_space);
                        for (int stride = kKvTile / 2; stride > 0; stride >>= 1) {
                            if (lid < stride) red_buf[lid] += red_buf[lid + stride];
                            it.barrier(sycl::access::fence_space::local_space);
                        }
                        float tile_sum = red_buf[0];
                        it.barrier(sycl::access::fence_space::local_space);

                        // Rescale the running output accumulator and add
                        // this tile's weighted-V contribution.
                        for (int d = lid; d < d_h; d += kKvTile) {
                            float pv = 0.0f;
                            for (int j = 0; j < kKvTile; ++j) {
                                int kj = kv_off + j;
                                if (kj >= ctx_len) break;
                                float pj = (scores_tile[j] == -std::numeric_limits<float>::infinity())
                                           ? 0.0f : sycl::exp(scores_tile[j] - new_max);
                                pv += pj * pV[kvbase + kj * d_h + d];
                            }
                            acc_out[d] = acc_out[d] * correction + pv;
                        }
                        it.barrier(sycl::access::fence_space::local_space);

                        if (lid == 0) {
                            run_sum[0] = run_sum[0] * correction + tile_sum;
                            run_max[0] = new_max;
                        }
                        it.barrier(sycl::access::fence_space::local_space);
                    }

                    float inv_sum = 1.0f / (run_sum[0] + 1e-9f);
                    for (int d = lid; d < d_h; d += kKvTile) {
                        pO[qbase + d] = acc_out[d] * inv_sum;
                    }
                });
            });
            pending_events_.push_back(e);
            synchronize();
        } catch (std::exception const&) {
            return false;
        }

        copy_to_host(out, o_alloc);
        return true;
    }

    static sycl::device select_preferred_device(std::vector<sycl::device> const& devices) {
        if (devices.empty()) throw std::runtime_error("no SYCL devices");
        for (auto const& d : devices) {
            if (d.is_gpu()) return d;
        }
        for (auto const& d : devices) {
            if (d.is_accelerator()) return d;
        }
        return devices.front();
    }

    static std::string selected_device_name(sycl::device const& d) {
        return sanitize(d.get_info<sycl::info::device::name>());
    }

    static std::string selected_device_type(sycl::device const& d) {
        if (d.is_gpu()) return "gpu";
        if (d.is_accelerator()) return "accelerator";
        if (d.is_cpu()) return "cpu";
        return "other";
    }

    // Returns the cached USM allocation for tensor `t`, creating one if none
    // exists yet. Returns nullptr (never throws) so call sites can fall back
    // to the CPU path on allocation failure instead of propagating an
    // exception through op entry points that don't expect one.
    //
    // Keyed by (TensorStorage*, offset) rather than just TensorStorage*:
    // Tensor supports views (row_slice/view/reshape) that share the same
    // underlying storage but a different offset and a different (usually
    // smaller) element count — e.g. a positional-embedding row-slice into a
    // larger weight tensor. Keying by storage alone would let two differently
    // sized/positioned views collide on one cache entry sized for whichever
    // view happened to populate it first, silently corrupting whichever view
    // reads or writes second.
    //
    // The weak_ptr means the cache does NOT keep short-lived activation
    // tensors alive — once the last shared_ptr<TensorStorage> elsewhere goes
    // away, the storage is freed normally and this entry becomes prunable.
    // Persistent tensors (model weights, held for the process lifetime) keep
    // their entry reused across every call, which is the common case we
    // actually want to accelerate.
    USMAllocation* ensure_allocation_for(Tensor const& t) const {
        if (!queue_ || !t.storage || t.numel() <= 0) return nullptr;
        prune_dead_entries();

        detail::AllocKey key{t.storage.get(), t.offset};
        auto it = allocations_.find(key);
        if (it != allocations_.end() && !it->second.first.expired()
            && it->second.second->elements >= t.numel()) {
            return it->second.second.get();
        }

        auto alloc = std::make_unique<USMAllocation>(const_cast<sycl::queue*>(&(*queue_)),
                                                       static_cast<std::size_t>(t.numel()) * sizeof(float));
        alloc->ptr = sycl::malloc_shared<float>(static_cast<std::size_t>(t.numel()), *const_cast<sycl::queue*>(&(*queue_)));
        if (!alloc->ptr) {
            return nullptr;
        }
        alloc->elements = t.numel();

        auto* raw = alloc.get();
        allocations_[key] = {std::weak_ptr<TensorStorage>(t.storage), std::move(alloc)};
        return raw;
    }

    // Sweeps entries whose TensorStorage has already been destroyed
    // elsewhere. Cheap linear scan; called before each lookup/insert so the
    // map doesn't grow without bound across a long generation run. For very
    // large models this could be made amortized (e.g. only sweep every N
    // calls), but correctness comes first here.
    void prune_dead_entries() const {
        for (auto it = allocations_.begin(); it != allocations_.end(); ) {
            if (it->second.first.expired()) it = allocations_.erase(it);
            else ++it;
        }
    }

    // Scratch/output tensors (e.g. `out` in layer_norm, `C` in gemm) are not
    // being copied *from* — there's no meaningful host payload to stage in
    // yet — but they still need a device buffer of the right size, cached
    // against their own (storage, offset) identity so repeated calls with
    // the same output tensor reuse it.
    USMAllocation* ensure_output_allocation(Tensor& out) const {
        if (out.empty()) return nullptr;
        return ensure_allocation_for(out);
    }

    void copy_to_device(Tensor const& host_t, USMAllocation* alloc) const {
        if (!alloc || !alloc->ptr || !queue_) return;
        auto const* src = host_t.data_ptr();
        auto* dst = static_cast<float*>(alloc->ptr);
        std::size_t bytes = static_cast<std::size_t>(host_t.numel()) * sizeof(float);
        if (bytes == 0) return;
        auto e = queue_->memcpy(dst, src, bytes);
        pending_events_.push_back(e);
        e.wait();
    }

    // Writes device data (from `alloc`) back into `host_t`'s own buffer.
    // Takes host_t by non-const reference deliberately — this copies INTO
    // host_t, so it needs the non-const data_ptr() overload. (A const Tensor&
    // here would silently resolve to the const data_ptr() overload
    // returning `float const*`, which cannot be used as a memcpy
    // destination — that was the bug this signature fixes.)
    void copy_to_host(Tensor& host_t, USMAllocation* alloc) const {
        if (!alloc || !alloc->ptr || !queue_) return;
        auto* dst = host_t.data_ptr();
        auto const* src = static_cast<float*>(alloc->ptr);
        std::size_t bytes = static_cast<std::size_t>(host_t.numel()) * sizeof(float);
        if (bytes == 0) return;
        auto e = queue_->memcpy(dst, src, bytes);
        pending_events_.push_back(e);
        e.wait();
    }

    // Keyed by (TensorStorage*, offset) — see AllocKey above. Each entry
    // pairs a weak_ptr (to detect when the storage has died) with the owned
    // USM allocation. See ensure_allocation_for()/prune_dead_entries().
    mutable std::unordered_map<detail::AllocKey, std::pair<std::weak_ptr<TensorStorage>, std::unique_ptr<USMAllocation>>, detail::AllocKeyHash> allocations_;
};

#else

class SYCLBackend : public IBackend {
public:
    BackendType type() const override { return BackendType::SYCL; }
    bool is_available() const override { return false; }
    std::string device_info() const override { return "SYCL support not compiled in (define VLRF_USE_SYCL)"; }
    std::vector<std::string> available_devices() const override { return {}; }
    std::string capabilities() const override { return "SYCL backend disabled"; }
    bool supports_usm() const override { return false; }
    void synchronize() const override {}

    bool try_init() { return false; }
    Tensor to_device(Tensor const& t) const override { return t; }
    Tensor to_host(Tensor const& t) const override { return t; }
    void gemm(Tensor const& A, Tensor const& B, Tensor& C, float alpha, float beta, bool tA, bool tB) const override {
        fallback_.gemm(A, B, C, alpha, beta, tA, tB);
    }
    void layer_norm(Tensor const& x, Tensor const& w, Tensor const& b, Tensor& out, float eps) const override {
        fallback_.layer_norm(x, w, b, out, eps);
    }
    void softmax_inplace(Tensor& t) const override { fallback_.softmax_inplace(t); }
    void gelu_inplace(Tensor& t) const override { fallback_.gelu_inplace(t); }
    void add_inplace(Tensor& dst, Tensor const& src) const override { fallback_.add_inplace(dst, src); }

    // No SYCL toolchain compiled in — always the CPU flash-attention path.
    Tensor sdp_attention(Tensor const& Q, Tensor const& K, Tensor const& V, bool causal) const override {
        return ops::sdp_attention(Q, K, V, causal);
    }

private:
    mutable CpuBackend fallback_;
};

#endif

} // namespace backend
} // namespace vlrf
