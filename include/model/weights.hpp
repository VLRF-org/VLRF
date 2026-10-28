// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — WeightMap  (weights.hpp, v2: mmap + streaming support)
//
// Three loading strategies, auto-selected by available RAM:
//
//   MMAP   (RAM >= 60% of model size)
//     Memory-map the file. Tensor data pointers point directly into the
//     mapped region — zero copy, zero allocation. OS pages in only what
//     is accessed and evicts cold pages under memory pressure.
//     Float16 tensors are dequantised into small per-tensor allocations.
//
//   STREAM (RAM < 60% of model size)
//     One transformer layer at a time is resident in RAM.
//     Only the current layer's weights are allocated; the previous layer
//     is freed before the next is loaded. KV-cache stays allocated.
//     transformer.hpp calls WeightMap::load_layer(l) / free_layer(l)
//     automatically around each layer's forward pass.
//
//   FULL   (legacy / testing)
//     Read entire file into heap allocations. Always works but needs full RAM.
//
// Format v1 (float32): version=1, no dtype byte per tensor
// Format v2 (f16/f32): version=2, dtype byte per tensor
// =============================================================================
#include "tensor.hpp"
#include "mmap_file.hpp"
#include <unordered_map>
#include <string>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <memory>
#include <cstdio>

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#elif defined(__linux__)
#  include <sys/sysinfo.h>
#endif

namespace vlrf {

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
static constexpr uint32_t NFW_MAGIC     = 0x4E465746;
static constexpr uint32_t NFW_VERSION1  = 1;
static constexpr uint32_t NFW_VERSION2  = 2;
static constexpr uint8_t  NFW_DTYPE_F32 = 0;
static constexpr uint8_t  NFW_DTYPE_F16 = 1;

enum class LoadMode {
    AUTO,    // auto-detect based on available RAM
    MMAP,    // memory-map the file (recommended)
    STREAM,  // stream one layer at a time (minimum RAM)
    FULL     // load everything into heap (legacy / testing)
};

// -----------------------------------------------------------------------------
// Float16 → Float32 conversion  (pure software, no hardware FP16 needed)
// -----------------------------------------------------------------------------
namespace detail {

inline float f16_to_f32(uint16_t h) noexcept {
    uint32_t sign     = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa = h & 0x03FFu;
    uint32_t result;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            exponent = 1;
            while (!(mantissa & 0x0400u)) { mantissa <<= 1; exponent--; }
            mantissa &= 0x03FFu;
            result = sign | ((exponent + (127u - 15u)) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31u) {
        result = sign | 0x7F800000u | (mantissa << 13);
    } else {
        result = sign | ((exponent + (127u - 15u)) << 23) | (mantissa << 13);
    }
    float f; std::memcpy(&f, &result, 4); return f;
}

inline void convert_f16_block(const uint16_t* src, float* dst, int64_t n) noexcept {
    for (int64_t i = 0; i < n; ++i) dst[i] = f16_to_f32(src[i]);
}

} // namespace detail

// -----------------------------------------------------------------------------
// Available physical RAM query
// -----------------------------------------------------------------------------
inline int64_t available_ram_bytes() {
#ifdef _WIN32
    MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms))
        return static_cast<int64_t>(ms.ullAvailPhys);
    return 4LL * 1024 * 1024 * 1024;
#elif defined(__linux__)
    struct sysinfo si;
    if (sysinfo(&si) == 0)
        return static_cast<int64_t>(si.freeram + si.bufferram) *
               static_cast<int64_t>(si.mem_unit);
    return 4LL * 1024 * 1024 * 1024;
#else
    return 4LL * 1024 * 1024 * 1024;
#endif
}

// -----------------------------------------------------------------------------
// TensorDescriptor — metadata only, no data allocation
// -----------------------------------------------------------------------------
struct TensorDescriptor {
    std::string name;
    Shape       shape;
    uint8_t     dtype       {NFW_DTYPE_F32};
    int64_t     data_offset {0};   // byte offset in file where data starts
    int64_t     numel       {0};
    int         layer_idx   {-1};  // -1 = global (embed / ln_final / lm_head)
};

// -----------------------------------------------------------------------------
// WeightMap
// -----------------------------------------------------------------------------
class WeightMap {
public:
    using Map = std::unordered_map<std::string, Tensor>;
    Map      tensors;
    LoadMode load_mode   {LoadMode::AUTO};
    int      streamed_layer {-1};

    WeightMap() = default;
    ~WeightMap() = default;
    WeightMap(WeightMap&&) = default;
    WeightMap& operator=(WeightMap&&) = default;

    // ---- Tensor access ------------------------------------------------------
    Tensor& get(std::string const& name) {
        auto it = tensors.find(name);
        if (it != tensors.end()) return it->second;

        // Streaming mode: auto-load the needed layer
        if (active_mode_ == LoadMode::STREAM) {
            int l = layer_from_name(name);
            if (l >= 0 && l != streamed_layer)
                load_layer_internal(l);
            it = tensors.find(name);
            if (it != tensors.end()) return it->second;
        }
        throw std::runtime_error("WeightMap: missing tensor '" + name + "'");
    }

    Tensor const& get(std::string const& name) const {
        auto it = tensors.find(name);
        if (it != tensors.end()) return it->second;
        throw std::runtime_error("WeightMap: missing tensor '" + name + "'");
    }

    bool has(std::string const& name) const {
        // Check resident tensors first
        if (tensors.count(name)) return true;
        // In stream mode, check descriptors
        if (active_mode_ == LoadMode::STREAM) {
            for (auto const& d : descriptors_)
                if (d.name == name) return true;
        }
        return false;
    }

    int64_t total_params() const {
        int64_t n = 0;
        for (auto const& d : descriptors_) n += d.numel;
        return n;
    }

    bool is_streaming() const { return active_mode_ == LoadMode::STREAM; }
    bool is_mmap()      const { return active_mode_ == LoadMode::MMAP;   }

    // Explicit streaming control (called by transformer.hpp)
    void load_layer(int l) {
        if (active_mode_ == LoadMode::STREAM) load_layer_internal(l);
    }

    void free_layer(int l) {
        if (active_mode_ != LoadMode::STREAM) return;
        if (streamed_layer != l) return;
        std::string pfx = "layers." + std::to_string(l) + ".";
        for (auto it = tensors.begin(); it != tensors.end(); ) {
            if (it->first.compare(0, pfx.size(), pfx) == 0)
                it = tensors.erase(it);
            else
                ++it;
        }
        streamed_layer = -1;
    }

    // ---- Save (always v1 float32) -------------------------------------------
    void save(std::string const& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error(
            "WeightMap: cannot write '" + path + "'");

        uint32_t n = (uint32_t)tensors.size(), rsv = 0;
        f.write((char const*)&NFW_MAGIC,    4);
        f.write((char const*)&NFW_VERSION1, 4);
        f.write((char const*)&n,            4);
        f.write((char const*)&rsv,          4);

        for (auto const& kv : tensors) {
            auto const& nm = kv.first;
            auto const& t  = kv.second;
            uint16_t nl = (uint16_t)nm.size();
            f.write((char const*)&nl, 2); f.write(nm.data(), nl);
            uint8_t nd = (uint8_t)t.ndim();
            f.write((char const*)&nd, 1);
            for (int i = 0; i < t.ndim(); ++i) {
                int32_t d = t.dim(i); f.write((char const*)&d, 4);
            }
            int64_t ne = t.numel();
            f.write((char const*)&ne, 8);
            f.write((char const*)t.data_ptr(), ne * 4);
        }
    }

    // ---- Load ---------------------------------------------------------------
    void load(std::string const& path, LoadMode mode = LoadMode::AUTO) {
        file_path_ = path;
        tensors.clear();
        descriptors_.clear();
        mmap_.reset();
        streamed_layer = -1;

        parse_header(path);

        // Estimate model size
        int64_t model_bytes = 0;
        for (auto const& d : descriptors_)
            model_bytes += d.numel * (d.dtype == NFW_DTYPE_F16 ? 2 : 4);

        int64_t avail = available_ram_bytes();

        if (mode == LoadMode::AUTO) {
            // Use mmap if available RAM >= 60% of model size (OS handles paging)
            // Use streaming if memory is tight
            if (avail >= (int64_t)(model_bytes * 0.6))
                mode = LoadMode::MMAP;
            else
                mode = LoadMode::STREAM;
        }
        active_mode_ = mode;

        printf("[WeightMap] model=%.1f GB  avail_ram=%.1f GB  strategy=%s\n",
               model_bytes / 1e9, avail / 1e9,
               mode == LoadMode::MMAP   ? "mmap"   :
               mode == LoadMode::STREAM ? "stream" : "full");

        switch (mode) {
            case LoadMode::MMAP:   load_mmap(path);               break;
            case LoadMode::STREAM: load_globals_streaming(path);  break;
            default:               load_full(path);               break;
        }
    }

    // Random-init (defined in transformer.hpp after ModelConfig is complete)
    static WeightMap random_init(struct ModelConfig const& cfg);

private:
    std::string                     file_path_;
    std::vector<TensorDescriptor>   descriptors_;
    std::unique_ptr<MmapFile>       mmap_;
    uint32_t                        file_version_ {NFW_VERSION1};
    LoadMode                        active_mode_  {LoadMode::FULL};

    // ---- Header parser -------------------------------------------------------
    void parse_header(std::string const& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error(
            "WeightMap: cannot open '" + path + "'");

        uint32_t magic, version, n, reserved;
        f.read((char*)&magic,    4);
        f.read((char*)&version,  4);
        f.read((char*)&n,        4);
        f.read((char*)&reserved, 4);

        if (magic != NFW_MAGIC) {
            char buf[64]; snprintf(buf,sizeof(buf),"0x%08X",magic);
            throw std::runtime_error(
                std::string("WeightMap: bad magic ") + buf +
                " (expected 0x4E465746) in '" + path + "'");
        }
        if (version != NFW_VERSION1 && version != NFW_VERSION2)
            throw std::runtime_error(
                "WeightMap: unsupported version " + std::to_string(version) +
                " in '" + path + "'");

        file_version_ = version;
        descriptors_.reserve(n);

        for (uint32_t i = 0; i < n; ++i) {
            TensorDescriptor desc;
            uint16_t nl; f.read((char*)&nl, 2);
            desc.name.resize(nl); f.read(desc.name.data(), nl);

            uint8_t nd; f.read((char*)&nd, 1);
            desc.shape.ndim = nd;
            for (int d = 0; d < nd; ++d) {
                int32_t dim; f.read((char*)&dim, 4);
                desc.shape[d] = dim;
            }

            int64_t ne; f.read((char*)&ne, 8);
            desc.numel = ne;
            desc.dtype = NFW_DTYPE_F32;
            if (version == NFW_VERSION2)
                f.read((char*)&desc.dtype, 1);

            desc.data_offset = (int64_t)f.tellg();
            desc.layer_idx   = layer_from_name(desc.name);

            int64_t bytes = ne * (desc.dtype == NFW_DTYPE_F16 ? 2 : 4);
            f.seekg(bytes, std::ios::cur);
            descriptors_.push_back(std::move(desc));
        }
    }

    // ---- MMAP strategy ------------------------------------------------------
    void load_mmap(std::string const& path) {
        mmap_ = std::make_unique<MmapFile>(path);
        for (auto const& desc : descriptors_)
            tensors[desc.name] = make_tensor_mmap(desc);
    }

    Tensor make_tensor_mmap(TensorDescriptor const& desc) {
        if (desc.dtype == NFW_DTYPE_F32) {
            // Zero-copy: point into mmap region
            float* ptr = const_cast<float*>(
                mmap_->ptr_at<float>((size_t)desc.data_offset));
            return Tensor::from_ptr(ptr, desc.shape);
        } else {
            // f16 → allocate f32 buffer and convert
            Tensor t(desc.shape);
            const uint16_t* src = mmap_->ptr_at<uint16_t>((size_t)desc.data_offset);
            detail::convert_f16_block(src, t.data_ptr(), desc.numel);
            return t;
        }
    }

    // ---- FULL strategy (heap) -----------------------------------------------
    void load_full(std::string const& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error(
            "WeightMap: cannot open '" + path + "'");
        for (auto const& desc : descriptors_)
            tensors[desc.name] = read_tensor(f, desc);
    }

    // ---- STREAM strategy ----------------------------------------------------
    void load_globals_streaming(std::string const& path) {
        // Load only tensors with layer_idx == -1 (embed, ln_final, lm_head, etc.)
        for (auto const& desc : descriptors_) {
            if (desc.layer_idx < 0)
                tensors[desc.name] = read_tensor_from_file(desc);
        }
        printf("[WeightMap] streaming: %zu global tensors resident, "
               "layers loaded on demand\n", tensors.size());
    }

    void load_layer_internal(int l) {
        if (streamed_layer >= 0 && streamed_layer != l)
            free_layer(streamed_layer);

        for (auto const& desc : descriptors_) {
            if (desc.layer_idx == l)
                tensors[desc.name] = read_tensor_from_file(desc);
        }
        streamed_layer = l;
    }

    // ---- Read one tensor from file ------------------------------------------
    Tensor read_tensor_from_file(TensorDescriptor const& desc) const {
        std::ifstream f(file_path_, std::ios::binary);
        if (!f) throw std::runtime_error(
            "WeightMap: cannot open '" + file_path_ + "'");
        f.seekg(desc.data_offset);
        return read_tensor(f, desc);
    }

    static Tensor read_tensor(std::ifstream& f,
                               TensorDescriptor const& desc) {
        // Always seek to the exact offset — this makes load_full safe regardless
        // of the order descriptors are processed (unordered_map iteration order).
        f.seekg((std::streamoff)desc.data_offset);
        if (!f) throw std::runtime_error(
            "WeightMap: seekg failed for '" + desc.name + "'");

        Tensor t(desc.shape);
        float* dst = t.data_ptr();
        if (desc.dtype == NFW_DTYPE_F16) {
            std::vector<uint16_t> buf((size_t)desc.numel);
            f.read((char*)buf.data(), desc.numel * 2);
            if (!f) throw std::runtime_error(
                "WeightMap: read error on '" + desc.name + "'");
            detail::convert_f16_block(buf.data(), dst, desc.numel);
        } else {
            f.read((char*)dst, desc.numel * 4);
            if (!f) throw std::runtime_error(
                "WeightMap: read error on '" + desc.name + "'");
        }
        return t;
    }

    // ---- Helpers ------------------------------------------------------------
    static int layer_from_name(std::string const& name) {
        if (name.size() > 7 && name.compare(0, 7, "layers.") == 0) {
            size_t dot = name.find('.', 7);
            if (dot != std::string::npos) {
                try { return std::stoi(name.substr(7, dot - 7)); }
                catch (...) {}
            }
        }
        return -1;
    }
};

} // namespace vlrf
