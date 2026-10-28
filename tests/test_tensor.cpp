// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF Tests — Tensor core  (test_tensor.cpp)
// =============================================================================
#include "tensor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace vlrf;

static int g_pass = 0, g_fail = 0;
#define CHECK(cond) do { \
    if(!(cond)){std::cerr<<"FAIL ["<<__LINE__<<"]: "#cond"\n";++g_fail;} \
    else ++g_pass; } while(0)
#define CHECK_EQ(a,b,tol) do { \
    float _a=(float)(a),_b=(float)(b); \
    if(std::fabs(_a-_b)>(tol)){std::cerr<<"FAIL ["<<__LINE__<<"]: |"#a"-"#b"|="<<std::fabs(_a-_b)<<" ("<<_a<<" vs "<<_b<<")\n";++g_fail;} \
    else ++g_pass; }while(0)

void test_default_construct() {
    Tensor t;
    CHECK(t.empty());
    CHECK(t.numel() == 0);
}

void test_shape_numel() {
    Tensor t({3, 4, 2});
    CHECK(t.ndim() == 3);
    CHECK(t.dim(0) == 3);
    CHECK(t.dim(1) == 4);
    CHECK(t.dim(2) == 2);
    CHECK(t.numel() == 24);
}

void test_zero_init() {
    Tensor t({5, 5});
    for (int i = 0; i < 25; ++i) CHECK_EQ(t[i], 0.f, 1e-9f);
}

void test_fill() {
    Tensor t({4});
    t.fill(7.f);
    for (int i = 0; i < 4; ++i) CHECK_EQ(t[i], 7.f, 1e-9f);
}

void test_flat_index_write() {
    Tensor t({2, 3});
    for (int i = 0; i < 6; ++i) t[i] = (float)i * 2.f;
    CHECK_EQ(t[0], 0.f, 1e-9f);
    CHECK_EQ(t[5], 10.f, 1e-9f);
}

void test_2d_at() {
    Tensor t({3, 3});
    t.at(1, 2) = 42.f;
    CHECK_EQ(t[1*3+2], 42.f, 1e-9f);
    CHECK_EQ(t.at(1, 2), 42.f, 1e-9f);
}

void test_3d_at() {
    Tensor t({2, 3, 4});
    t.at(1, 2, 3) = 99.f;
    int flat = 1*(3*4) + 2*4 + 3;
    CHECK_EQ(t[flat], 99.f, 1e-9f);
}

void test_view_shared_storage() {
    Tensor t({6});
    for (int i = 0; i < 6; ++i) t[i] = (float)i;
    Tensor v = t.view({2, 3});
    CHECK(v.dim(0) == 2 && v.dim(1) == 3);
    CHECK(v.data_ptr() == t.data_ptr());  // same buffer
    v[0] = 999.f;
    CHECK_EQ(t[0], 999.f, 1e-9f);  // mutation visible via t
}

void test_row_slice_2d() {
    Tensor t({5, 4});
    for (int i = 0; i < 20; ++i) t[i] = (float)i;
    // Slice rows [2, 4) → 2 rows
    Tensor s = t.row_slice(2, 2);
    CHECK(s.dim(0) == 2 && s.dim(1) == 4);
    CHECK_EQ(s.at(0, 0), 8.f,  1e-9f);  // t[2,0]
    CHECK_EQ(s.at(1, 3), 15.f, 1e-9f);  // t[3,3]
}

void test_from_ptr() {
    float buf[4] = {1, 2, 3, 4};
    Tensor t = Tensor::from_ptr(buf, {2, 2});
    CHECK(t.data_ptr() == buf);  // no copy
    CHECK_EQ(t.at(1, 0), 3.f, 1e-9f);
    buf[2] = 77.f;
    CHECK_EQ(t.at(1, 0), 77.f, 1e-9f);
}

void test_zeros_factory() {
    Tensor t = zeros({3, 3});
    for (int i = 0; i < 9; ++i) CHECK_EQ(t[i], 0.f, 1e-9f);
}

void test_ones_factory() {
    Tensor t = ones({3});
    for (int i = 0; i < 3; ++i) CHECK_EQ(t[i], 1.f, 1e-9f);
}

void test_like_factory() {
    Tensor a({2, 5});
    a.fill(3.f);
    Tensor b = like(a);
    CHECK(b.shape == a.shape);
    CHECK(b.data_ptr() != a.data_ptr());  // independent allocation
}

void test_shape_comparison() {
    Shape s1{2, 3};
    Shape s2{2, 3};
    Shape s3{3, 2};
    CHECK(s1 == s2);
    CHECK(s1 != s3);
}

void test_reshape_inplace() {
    Tensor t({12});
    for (int i = 0; i < 12; ++i) t[i] = (float)i;
    t.reshape({3, 4});
    CHECK(t.dim(0) == 3 && t.dim(1) == 4);
    CHECK_EQ(t.at(2, 3), 11.f, 1e-9f);
}

// Verify copy-on-write is NOT happening: shared storage is intentional
void test_shared_storage_independence() {
    Tensor a({4});
    a.fill(1.f);
    Tensor b = a.view({4});  // view shares storage
    b[0] = 5.f;
    CHECK_EQ(a[0], 5.f, 1e-9f);  // shared, mutation visible
}

int main() {
    std::cout << "=== VLRF Tensor Tests ===\n";
    test_default_construct();
    test_shape_numel();
    test_zero_init();
    test_fill();
    test_flat_index_write();
    test_2d_at();
    test_3d_at();
    test_view_shared_storage();
    test_row_slice_2d();
    test_from_ptr();
    test_zeros_factory();
    test_ones_factory();
    test_like_factory();
    test_shape_comparison();
    test_reshape_inplace();
    test_shared_storage_independence();
    std::cout << "Results: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
