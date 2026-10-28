// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — Tensor  (tensor.hpp)
// A flat, float32, row-major N-dimensional tensor with lightweight view semantics.
// No heap allocation on copy of a *view*; ownership lives in TensorStorage.
// =============================================================================
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>
#include <initializer_list>
#include <algorithm>
#include <numeric>

namespace vlrf {

// -----------------------------------------------------------------------------
// Shape
// -----------------------------------------------------------------------------
struct Shape {
    static constexpr int MAX_DIMS = 5;
    int32_t d[MAX_DIMS]{0};
    int      ndim{0};

    Shape() = default;
    Shape(std::initializer_list<int32_t> il) {
        ndim = static_cast<int>(il.size());
        assert(ndim <= MAX_DIMS);
        int i = 0;
        for (auto v : il) d[i++] = v;
    }
    explicit Shape(std::vector<int32_t> const& v) {
        ndim = static_cast<int>(v.size());
        assert(ndim <= MAX_DIMS);
        for (int i = 0; i < ndim; ++i) d[i] = v[i];
    }

    int32_t  operator[](int i) const { return d[i]; }
    int32_t& operator[](int i)       { return d[i]; }

    int64_t numel() const {
        if (ndim == 0) return 0;
        int64_t n = 1;
        for (int i = 0; i < ndim; ++i) n *= d[i];
        return n;
    }
    bool operator==(Shape const& o) const {
        if (ndim != o.ndim) return false;
        for (int i = 0; i < ndim; ++i) if (d[i] != o.d[i]) return false;
        return true;
    }
    bool operator!=(Shape const& o) const { return !(*this == o); }

    std::string str() const {
        std::string s = "[";
        for (int i = 0; i < ndim; ++i) {
            s += std::to_string(d[i]);
            if (i + 1 < ndim) s += ",";
        }
        return s + "]";
    }
};

// -----------------------------------------------------------------------------
// TensorStorage  — reference-counted backing buffer
// -----------------------------------------------------------------------------
struct TensorStorage {
    float*  data   {nullptr};
    int64_t numel  {0};
    bool    owned  {true};   // if false, data is external (mmap / weight blob)

    TensorStorage() = default;
    explicit TensorStorage(int64_t n) : numel(n), owned(true) {
        data = new float[n]();   // zero-initialised
    }
    // Wrap external pointer (no ownership)
    TensorStorage(float* ptr, int64_t n) : data(ptr), numel(n), owned(false) {}

    ~TensorStorage() { if (owned && data) { delete[] data; data = nullptr; } }

    // Non-copyable, movable
    TensorStorage(TensorStorage const&)            = delete;
    TensorStorage& operator=(TensorStorage const&) = delete;
    TensorStorage(TensorStorage&& o) noexcept
        : data(o.data), numel(o.numel), owned(o.owned)
    { o.data = nullptr; o.owned = false; }
};

// -----------------------------------------------------------------------------
// Tensor  — lightweight handle (shared_ptr to storage + shape + offset)
// -----------------------------------------------------------------------------
class Tensor {
public:
    std::shared_ptr<TensorStorage> storage;
    Shape    shape;
    int64_t  offset{0};   // element offset into storage

    Tensor() = default;

    // Allocate new tensor
    explicit Tensor(Shape s)
        : storage(std::make_shared<TensorStorage>(s.numel()))
        , shape(s), offset(0) {}

    // Allocate via initializer list
    Tensor(std::initializer_list<int32_t> dims)
        : Tensor(Shape(dims)) {}

    // Wrap external float buffer (no copy, no ownership)
    static Tensor from_ptr(float* ptr, Shape s) {
        Tensor t;
        t.storage = std::make_shared<TensorStorage>(ptr, s.numel());
        t.shape   = s;
        t.offset  = 0;
        return t;
    }

    float*       data_ptr()       { return storage->data + offset; }
    float const* data_ptr() const { return storage->data + offset; }

    int64_t numel()  const { return shape.numel(); }
    bool    empty()  const { return !storage || shape.numel() == 0; }
    int     ndim()   const { return shape.ndim; }
    int32_t dim(int i) const { return shape[i]; }

    // Element access (flat)
    float& operator[](int64_t i)       { return data_ptr()[i]; }
    float  operator[](int64_t i) const { return data_ptr()[i]; }

    // 2-D convenience
    float& at(int r, int c)       { return data_ptr()[r * shape[1] + c]; }
    float  at(int r, int c) const { return data_ptr()[r * shape[1] + c]; }

    // 3-D convenience
    float& at(int a, int b, int c) {
        return data_ptr()[(int64_t)a * shape[1] * shape[2] + (int64_t)b * shape[2] + c];
    }
    float at(int a, int b, int c) const {
        return data_ptr()[(int64_t)a * shape[1] * shape[2] + (int64_t)b * shape[2] + c];
    }

    void zero()  { std::fill(data_ptr(), data_ptr() + numel(), 0.f); }
    void fill(float v) { std::fill(data_ptr(), data_ptr() + numel(), v); }

    // Reshape (in-place, same numel required)
    Tensor& reshape(Shape s) {
        assert(s.numel() == numel());
        shape = s;
        return *this;
    }

    // Return a view with different shape (same storage)
    Tensor view(Shape s) const {
        assert(s.numel() == numel());
        Tensor t;
        t.storage = storage;
        t.shape   = s;
        t.offset  = offset;
        return t;
    }

    // Slice along dim-0 (returns view)
    Tensor row_slice(int32_t start, int32_t len) const {
        assert(shape.ndim >= 1);
        int64_t row_stride = 1;
        for (int i = 1; i < shape.ndim; ++i) row_stride *= shape[i];
        Tensor t;
        t.storage = storage;
        t.offset  = offset + (int64_t)start * row_stride;
        t.shape   = shape;
        t.shape[0]= len;
        return t;
    }

    std::string shape_str() const { return shape.str(); }
};

// -----------------------------------------------------------------------------
// Tensor factory helpers
// -----------------------------------------------------------------------------
inline Tensor zeros(Shape s)           { return Tensor(s); }
inline Tensor zeros(std::initializer_list<int32_t> il) { return Tensor(Shape(il)); }

inline Tensor ones(Shape s) {
    Tensor t(s);
    t.fill(1.f);
    return t;
}

inline Tensor like(Tensor const& ref) { return Tensor(ref.shape); }

} // namespace vlrf
