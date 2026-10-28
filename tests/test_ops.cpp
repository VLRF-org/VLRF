// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
// =============================================================================
// VLRF Tests — Tensor & Ops  (test_ops.cpp)
// =============================================================================
#include "tensor.hpp"
#include "ops.hpp"
#include <cmath>
#include <iostream>
#include <cassert>

using namespace vlrf;

static int g_pass = 0, g_fail = 0;

#define EXPECT(cond) do { \
    if (!(cond)) { \
        std::cerr << "FAIL [" << __LINE__ << "]: " #cond "\n"; \
        ++g_fail; \
    } else { \
        ++g_pass; \
    } \
} while(0)

#define EXPECT_NEAR(a, b, tol) do { \
    float _a = (a), _b = (b); \
    if (std::abs(_a - _b) > (tol)) { \
        std::cerr << "FAIL [" << __LINE__ << "]: |" #a " - " #b "| = " \
                  << std::abs(_a-_b) << " > " << (tol) << "  (" << _a << " vs " << _b << ")\n"; \
        ++g_fail; \
    } else { ++g_pass; } \
} while(0)

// -----------------------------------------------------------------------------
// Tensor basic
// -----------------------------------------------------------------------------
void test_tensor_create() {
    Tensor t({2, 3});
    EXPECT(t.dim(0) == 2 && t.dim(1) == 3);
    EXPECT(t.numel() == 6);
    // should be zero-initialised
    for (int i = 0; i < 6; ++i) EXPECT(t[i] == 0.f);
}

void test_tensor_fill() {
    Tensor t({4});
    t.fill(3.14f);
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(t[i], 3.14f, 1e-6f);
}

void test_tensor_view() {
    Tensor t({6});
    for (int i = 0; i < 6; ++i) t[i] = (float)i;
    Tensor v = t.view({2, 3});
    EXPECT(v.at(1, 2) == 5.f);   // row1, col2
    v.at(0, 0) = 99.f;
    EXPECT(t[0] == 99.f);  // same storage
}

void test_tensor_row_slice() {
    Tensor t({4, 2});
    for (int i = 0; i < 8; ++i) t[i] = (float)i;
    Tensor s = t.row_slice(1, 2);  // rows [1,2]
    EXPECT(s.dim(0) == 2 && s.dim(1) == 2);
    EXPECT(s.at(0, 0) == 2.f);
    EXPECT(s.at(1, 1) == 5.f);   // t[2,2]: row2=[4,5] → col1=5
}

// -----------------------------------------------------------------------------
// Ops
// -----------------------------------------------------------------------------
void test_gemm_identity() {
    // A = [[1,0],[0,1]]   B = [[2,3],[4,5]]   C = A*B = B
    Tensor A({2,2}), B({2,2}), C({2,2});
    A.at(0,0) = 1; A.at(1,1) = 1;
    B.at(0,0) = 2; B.at(0,1) = 3;
    B.at(1,0) = 4; B.at(1,1) = 5;
    ops::gemm(A, B, C, 1.f, 0.f, false, false);
    EXPECT_NEAR(C.at(0,0), 2.f, 1e-5f);
    EXPECT_NEAR(C.at(0,1), 3.f, 1e-5f);
    EXPECT_NEAR(C.at(1,0), 4.f, 1e-5f);
    EXPECT_NEAR(C.at(1,1), 5.f, 1e-5f);
}

void test_gemm_2x3_3x2() {
    // [2,3] x [3,2]
    Tensor A({2,3}), B({3,2}), C({2,2});
    // A = [[1,2,3],[4,5,6]]
    A[0]=1; A[1]=2; A[2]=3; A[3]=4; A[4]=5; A[5]=6;
    // B = [[1,2],[3,4],[5,6]]
    B[0]=1; B[1]=2; B[2]=3; B[3]=4; B[4]=5; B[5]=6;
    ops::gemm(A, B, C, 1.f, 0.f, false, false);
    // C[0,0] = 1+6+15 = 22, C[0,1] = 2+8+18 = 28
    // C[1,0] = 4+15+30 = 49, C[1,1] = 8+20+36 = 64
    EXPECT_NEAR(C.at(0,0), 22.f, 1e-4f);
    EXPECT_NEAR(C.at(0,1), 28.f, 1e-4f);
    EXPECT_NEAR(C.at(1,0), 49.f, 1e-4f);
    EXPECT_NEAR(C.at(1,1), 64.f, 1e-4f);
}

void test_softmax() {
    Tensor t({1, 4});
    t[0]=1.f; t[1]=2.f; t[2]=3.f; t[3]=4.f;
    ops::softmax_inplace(t);
    float sum = 0;
    for (int i = 0; i < 4; ++i) { EXPECT(t[i] > 0); sum += t[i]; }
    EXPECT_NEAR(sum, 1.f, 1e-5f);
    // Largest input → largest probability
    EXPECT(t[3] > t[2] && t[2] > t[1] && t[1] > t[0]);
}

void test_layer_norm() {
    int D = 8;
    Tensor x({1, D});
    for (int i = 0; i < D; ++i) x[i] = (float)(i - D/2);
    Tensor w({D}); w.fill(1.f);
    Tensor b({D}); b.fill(0.f);
    Tensor out({1, D});
    ops::layer_norm(x, w, b, out, 1e-5f);

    // Output should have mean ≈ 0 and std ≈ 1
    float mean = 0, var = 0;
    for (int i = 0; i < D; ++i) mean += out[i];
    mean /= D;
    for (int i = 0; i < D; ++i) { float d = out[i]-mean; var += d*d; }
    var /= D;
    EXPECT_NEAR(mean, 0.f, 1e-4f);
    EXPECT_NEAR(std::sqrt(var), 1.f, 1e-3f);
}

void test_gelu() {
    Tensor t({3});
    t[0] = -2.f; t[1] = 0.f; t[2] = 2.f;
    ops::gelu_inplace(t);
    EXPECT(t[0] < 0.f);          // GELU(-2) ≈ -0.045
    EXPECT_NEAR(t[1], 0.f, 1e-6f); // GELU(0) = 0
    EXPECT(t[2] > 1.9f);           // GELU(2) ≈ 1.96
}

void test_transpose() {
    Tensor t({2, 3});
    for (int i = 0; i < 6; ++i) t[i] = (float)i;
    Tensor tr = ops::transpose_last2(t);
    EXPECT(tr.dim(0) == 3 && tr.dim(1) == 2);
    // t[r,c] = r*3+c  → tr[c,r] = r*3+c
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c)
            EXPECT_NEAR(t.at(r,c), tr.at(c,r), 1e-6f);
}

void test_add_inplace() {
    Tensor a({4}); a.fill(1.f);
    Tensor b({4}); b.fill(2.f);
    ops::add_inplace(a, b);
    for (int i = 0; i < 4; ++i) EXPECT_NEAR(a[i], 3.f, 1e-6f);
}

void test_embedding_lookup() {
    int V=5, D=3;
    Tensor table({V, D});
    for (int v=0; v<V; ++v)
        for (int d=0; d<D; ++d)
            table.at(v,d) = (float)(v*10 + d);
    std::vector<int> ids = {2, 0, 4};
    Tensor out = ops::embedding_lookup(table, ids);
    EXPECT(out.dim(0)==3 && out.dim(1)==3);
    EXPECT_NEAR(out.at(0,0), 20.f, 1e-5f);  // vocab[2][0]
    EXPECT_NEAR(out.at(1,2),  2.f, 1e-5f);  // vocab[0][2]
    EXPECT_NEAR(out.at(2,1), 41.f, 1e-5f);  // vocab[4][1]
}

void test_sinusoidal_pe() {
    Tensor x({4, 16});
    x.zero();
    ops::add_sinusoidal_pe(x, 0);
    // Position 0 sine part = sin(0) = 0 for even dims
    EXPECT_NEAR(x.at(0, 0), 0.f, 1e-5f);   // sin(0/div) = 0
    EXPECT_NEAR(x.at(0, 1), 1.f, 1e-5f);   // cos(0/div) = 1
    // Position 1 sine part should be non-zero
    EXPECT(std::abs(x.at(1, 0)) > 0.f);
}

// -----------------------------------------------------------------------------
// Attention smoke test
// -----------------------------------------------------------------------------
void test_sdp_attention_shape() {
    int T=4, Dh=8;
    Tensor Q({T, Dh}), K({T, Dh}), V({T, Dh});
    Q.fill(0.1f); K.fill(0.1f); V.fill(0.5f);
    Tensor out = ops::sdp_attention(Q, K, V, true);
    EXPECT(out.dim(0) == T && out.dim(1) == Dh);
    // With uniform Q,K output should be close to V
    for (int i = 0; i < T*Dh; ++i)
        EXPECT_NEAR(out[i], 0.5f, 0.1f);
}

void test_causal_mask() {
    Tensor s({4, 4});
    s.fill(0.f);
    ops::apply_causal_mask(s);
    // Upper triangle should be -inf
    for (int i=0; i<4; ++i)
        for (int j=i+1; j<4; ++j)
            EXPECT(s.at(i,j) < -1e29f);
    // Diagonal and below should be 0
    for (int i=0; i<4; ++i)
        for (int j=0; j<=i; ++j)
            EXPECT_NEAR(s.at(i,j), 0.f, 1e-6f);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main() {
    std::cout << "=== VLRF Tensor/Ops Tests ===\n";

    test_tensor_create();
    test_tensor_fill();
    test_tensor_view();
    test_tensor_row_slice();
    test_gemm_identity();
    test_gemm_2x3_3x2();
    test_softmax();
    test_layer_norm();
    test_gelu();
    test_transpose();
    test_add_inplace();
    test_embedding_lookup();
    test_sinusoidal_pe();
    test_sdp_attention_shape();
    test_causal_mask();

    std::cout << "\nResults: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail > 0 ? 1 : 0;
}
