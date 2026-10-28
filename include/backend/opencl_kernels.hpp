// SPDX-License-Identifier: EUPL-1.2
// Copyright (C) 2026 Aleksander "Kowalski"
#pragma once
// =============================================================================
// VLRF — OpenCL kernel sources  (opencl_kernels.hpp)
//
// NOTE: kernel sources below are stored as XOR-"obfuscated" byte arrays and
// decoded at runtime. This is NOT encryption and provides no real
// protection — it's a single-byte XOR with a hardcoded key, and the
// plaintext (*_SOURCE_PLAIN below) is compiled into the same binary right
// next to the "encrypted" arrays, so anyone with the binary already has the
// plaintext with zero effort. Treat this purely as light obfuscation against
// a casual `strings` scan, not as a security boundary for the kernel source.
//
// Kernel inventory (mirrors ops.hpp's CPU surface — see backend.hpp's
// dispatch_* methods for what actually wires each of these up):
//   gemm                   — tiled + register-blocked (4x4/thread) GEMM,
//                            C = alpha*op(A)*op(B) + beta*C, with optional
//                            transposed A/B selected via compile-time
//                            defines (-DTRANSPOSE_A=1 / -DTRANSPOSE_B=1) so
//                            the hot inner loop has no per-element branch on
//                            tA/tB. KernelManager builds (and caches) up to
//                            four program variants (NN/NT/TN/TT) lazily,
//                            only the ones actually requested.
//   softmax_inplace        — unchanged algorithm (block-parallel max/sum
//                            reduction), still correct and reasonably fast.
//   scaled_dot_attention   — legacy: QK^T-only kernel (scores out), kept for
//                            ABI/back-compat since KernelManager exposed it
//                            before. Superseded by flash_attention below for
//                            the actual dispatch path.
//   flash_attention        — fused QK^T + online-softmax + weighted-V
//                            kernel, one work-group per (head, query row);
//                            avoids materialising the full [T, ctx_len]
//                            score matrix in global memory the way
//                            sdp_attention's gemm->softmax->gemm CPU path
//                            does.
//   layer_norm             — one work-group per row, block reduction for
//                            mean/var (matches ops::layer_norm semantics).
//   gelu_inplace           — tanh-approximation GELU, matches
//                            ops::gelu_inplace's constants exactly.
//   add_inplace            — simple elementwise a[i] += b[i].
// =============================================================================

#include <array>
#include <cstddef>
#include <string>

namespace vlrf {
namespace backend {
namespace opencl {

static constexpr unsigned char XOR_KEY = 0xA5u;

constexpr unsigned char xor_char(unsigned char value) {
    return static_cast<unsigned char>(value ^ XOR_KEY);
}

template <size_t N>
inline std::array<unsigned char, N> xor_encrypt(const char (&text)[N]) {
    std::array<unsigned char, N> result{};
    for (size_t i = 0; i < N; ++i) {
        result[i] = xor_char(static_cast<unsigned char>(text[i]));
    }
    return result;
}

template <size_t N>
inline std::string decrypt_kernel(const std::array<unsigned char, N>& data) {
    std::string out;
    out.reserve(N - 1);
    for (size_t i = 0; i < N - 1; ++i) {
        out.push_back(static_cast<char>(xor_char(data[i])));
    }
    return out;
}

// -----------------------------------------------------------------------------
// GEMM — 32x32 output tile per work-group, 8x8 threads, each thread
// computes a 4x4 register-blocked sub-tile. Local memory tiles are
// padded ([TS][TS+1]) to avoid local-memory bank conflicts. TRANSPOSE_A /
// TRANSPOSE_B are set via build options rather than kernel args, so the
// inner product loop has no per-iteration branch.
//
// Semantics match ops::gemm: C = alpha * op(A) * op(B) + beta * C, where
// op(A) is A^T when TRANSPOSE_A is defined (A stored [K, M], row-major),
// and likewise op(B) is B^T when TRANSPOSE_B is defined (B stored [N, K]).
// -----------------------------------------------------------------------------
static constexpr char GEMM_SOURCE_PLAIN[] =
    "#ifndef TRANSPOSE_A\n"
    "#define TRANSPOSE_A 0\n"
    "#endif\n"
    "#ifndef TRANSPOSE_B\n"
    "#define TRANSPOSE_B 0\n"
    "#endif\n"
    "#define TS 32\n"          // output tile size (both dims)
    "#define WPT 4\n"          // work per thread, per dim (4x4 register block)
    "#define RTS (TS / WPT)\n" // threads per work-group dim (8)
    "__kernel __attribute__((reqd_work_group_size(RTS, RTS, 1))) void gemm(\n"
    "    const int M,\n"
    "    const int N,\n"
    "    const int K,\n"
    "    const float alpha,\n"
    "    const float beta,\n"
    "    __global const float* A,\n"
    "    __global const float* B,\n"
    "    __global float* C) {\n"
    "    __local float tile_a[TS][TS + 1];\n"
    "    __local float tile_b[TS][TS + 1];\n"
    "    const int lrow = get_local_id(0);\n"
    "    const int lcol = get_local_id(1);\n"
    "    const int group_row = get_group_id(0) * TS;\n"
    "    const int group_col = get_group_id(1) * TS;\n"
    "    float acc[WPT][WPT];\n"
    "    for (int i = 0; i < WPT; ++i)\n"
    "        for (int j = 0; j < WPT; ++j)\n"
    "            acc[i][j] = 0.0f;\n"
    "    const int tiles = (K + TS - 1) / TS;\n"
    "    for (int t = 0; t < tiles; ++t) {\n"
    "        const int tile_off = t * TS;\n"
    "        for (int i = 0; i < WPT; ++i) {\n"
    "            const int dst_row = lrow + i * RTS;\n"
    "            for (int j = 0; j < WPT; ++j) {\n"
    "                const int dst_col = lcol + j * RTS;\n"
    "                const int g_row = group_row + dst_row;\n"
    "                const int a_k = tile_off + dst_col;\n"
    "#if TRANSPOSE_A\n"
    "                tile_a[dst_row][dst_col] = (g_row < M && a_k < K) ? A[a_k * M + g_row] : 0.0f;\n"
    "#else\n"
    "                tile_a[dst_row][dst_col] = (g_row < M && a_k < K) ? A[g_row * K + a_k] : 0.0f;\n"
    "#endif\n"
    "                const int g_col = group_col + dst_col;\n"
    "                const int b_k = tile_off + dst_row;\n"
    "#if TRANSPOSE_B\n"
    "                tile_b[dst_row][dst_col] = (g_col < N && b_k < K) ? B[g_col * K + b_k] : 0.0f;\n"
    "#else\n"
    "                tile_b[dst_row][dst_col] = (g_col < N && b_k < K) ? B[b_k * N + g_col] : 0.0f;\n"
    "#endif\n"
    "            }\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        const int limit = min(TS, K - tile_off);\n"
    "        for (int kk = 0; kk < limit; ++kk) {\n"
    "            float a_reg[WPT];\n"
    "            float b_reg[WPT];\n"
    "            for (int i = 0; i < WPT; ++i) a_reg[i] = tile_a[lrow + i * RTS][kk];\n"
    "            for (int j = 0; j < WPT; ++j) b_reg[j] = tile_b[kk][lcol + j * RTS];\n"
    "            for (int i = 0; i < WPT; ++i)\n"
    "                for (int j = 0; j < WPT; ++j)\n"
    "                    acc[i][j] += a_reg[i] * b_reg[j];\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    for (int i = 0; i < WPT; ++i) {\n"
    "        const int row = group_row + lrow + i * RTS;\n"
    "        if (row >= M) continue;\n"
    "        for (int j = 0; j < WPT; ++j) {\n"
    "            const int col = group_col + lcol + j * RTS;\n"
    "            if (col >= N) continue;\n"
    "            const int idx = row * N + col;\n"
    "            C[idx] = alpha * acc[i][j] + beta * C[idx];\n"
    "        }\n"
    "    }\n"
    "}\n";

// -----------------------------------------------------------------------------
// softmax_inplace — block-level max/sum reduction over a row, one
// 256-thread work-group per row (rid = get_group_id(0), NOT
// get_global_id(0) — the caller launches rows*256 total work-items so
// each work-group's 256 threads cooperatively reduce exactly one row;
// using get_global_id(0) here would make every thread think it owns a
// distinct row while still sharing one __local reduction buffer across
// the whole group, corrupting results / reading uninitialised __local
// memory whenever rows != 256).
// -----------------------------------------------------------------------------
static constexpr char SOFTMAX_SOURCE_PLAIN[] =
    "__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void softmax_inplace(\n"
    "    __global float* x,\n"
    "    const int rows,\n"
    "    const int cols) {\n"
    "    const int rid = get_group_id(0);\n"
    "    const int lid = get_local_id(0);\n"
    "    if (rid >= rows) return;\n"
    "    const int base = rid * cols;\n"
    "    __local float shared_max[256];\n"
    "    __local float shared_sum[256];\n"
    "    float max_val = -INFINITY;\n"
    "    for (int c = lid; c < cols; c += 256) {\n"
    "        const float v = x[base + c];\n"
    "        if (v > max_val) max_val = v;\n"
    "    }\n"
    "    shared_max[lid] = max_val;\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    if (lid == 0) {\n"
    "        for (int i = 1; i < 256; ++i) {\n"
    "            if (shared_max[i] > max_val) max_val = shared_max[i];\n"
    "        }\n"
    "        shared_max[0] = max_val;\n"
    "    }\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    max_val = shared_max[0];\n"
    "    float sum = 0.0f;\n"
    "    for (int c = lid; c < cols; c += 256) {\n"
    "        const float v = exp(x[base + c] - max_val);\n"
    "        x[base + c] = v;\n"
    "        sum += v;\n"
    "    }\n"
    "    shared_sum[lid] = sum;\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    if (lid == 0) {\n"
    "        for (int i = 1; i < 256; ++i) sum += shared_sum[i];\n"
    "        shared_sum[0] = sum;\n"
    "    }\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    sum = shared_sum[0];\n"
    "    for (int c = lid; c < cols; c += 256) {\n"
    "        x[base + c] = x[base + c] / sum;\n"
    "    }\n"
    "}\n";

// -----------------------------------------------------------------------------
// scaled_dot_attention — legacy QK^T-only kernel, kept for source/ABI
// compatibility with anything referencing KernelManager::attention_kernel()
// pre-upgrade. Not used by the new dispatch_attention path (see
// flash_attention below), which fuses the softmax and the V matmul too.
// -----------------------------------------------------------------------------
static constexpr char ATTENTION_SOURCE_PLAIN[] =
    "__kernel void scaled_dot_attention(\n"
    "    const int n_h,\n"
    "    const int T,\n"
    "    const int ctx_len,\n"
    "    const int d_h,\n"
    "    const float scale,\n"
    "    const int causal,\n"
    "    __global const float* Q,\n"
    "    __global const float* K,\n"
    "    __global const float* V,\n"
    "    __global float* scores) {\n"
    "    const int h = get_global_id(0);\n"
    "    const int qi = get_global_id(1);\n"
    "    if (h >= n_h || qi >= T) return;\n"
    "    const int qbase = h * T * d_h + qi * d_h;\n"
    "    const int kbase = h * ctx_len * d_h;\n"
    "    for (int ki = 0; ki < ctx_len; ++ki) {\n"
    "        float acc = 0.0f;\n"
    "        const int kval = kbase + ki * d_h;\n"
    "        int d = 0;\n"
    "        for (; d + 3 < d_h; d += 4) {\n"
    "            const float4 qv = vload4(0, &Q[qbase + d]);\n"
    "            const float4 kv = vload4(0, &K[kval + d]);\n"
    "            acc += dot(qv, kv);\n"
    "        }\n"
    "        for (; d < d_h; ++d) {\n"
    "            acc += Q[qbase + d] * K[kval + d];\n"
    "        }\n"
    "        if (causal && ki > qi + ctx_len - T) {\n"
    "            acc = -INFINITY;\n"
    "        }\n"
    "        scores[h * T * ctx_len + qi * ctx_len + ki] = acc * scale;\n"
    "    }\n"
    "}\n";

// -----------------------------------------------------------------------------
// flash_attention — fused scaled-dot-product attention: one work-group
// per (head, query-row). Streams over K/V in tiles, maintaining an
// online (running) softmax max/sum so the full [T, ctx_len] score row is
// never materialised in global memory — only a d_h-sized accumulator per
// work-group lives in local memory. This mirrors ops::sdp_attention's
// math exactly (same scale, same causal-mask condition
// `ki > qi + ctx_len - T`) but fuses all three CPU-side steps
// (Q*K^T, softmax, scores*V) into a single kernel launch.
//
// Layout: Q is [n_h, T, d_h], K/V are [n_h, ctx_len, d_h], out is
// [n_h, T, d_h] — all row-major, matching how dispatch_attention lays out
// the per-head views it gets from ops::sdp_attention's batched path.
//
// Work-group size is (KV_TILE, 1, 1); each work-group cooperatively loads
// KV_TILE key/value rows at a time and every thread in the group
// accumulates the full output row redundantly (d_h is typically small —
// 64/128 — so this trades a little redundant compute for avoiding any
// cross-thread output reduction). d_h must be <= MAX_DH.
// -----------------------------------------------------------------------------
static constexpr char FLASH_ATTENTION_SOURCE_PLAIN[] =
    "#define KV_TILE 64\n"
    "#define MAX_DH 256\n"
    "__kernel __attribute__((reqd_work_group_size(KV_TILE, 1, 1))) void flash_attention(\n"
    "    const int n_h,\n"
    "    const int T,\n"
    "    const int ctx_len,\n"
    "    const int d_h,\n"
    "    const float scale,\n"
    "    const int causal,\n"
    "    __global const float* Q,\n"
    "    __global const float* K,\n"
    "    __global const float* V,\n"
    "    __global float* out) {\n"
    "    const int h  = get_group_id(0);\n"
    "    const int qi = get_group_id(1);\n"
    "    const int lid = get_local_id(0);\n"
    "    if (h >= n_h || qi >= T) return;\n"
    "    __local float scores_tile[KV_TILE];\n"
    "    __local float red_buf[KV_TILE];\n"
    "    __local float run_max;\n"
    "    __local float run_sum;\n"
    "    __local float acc_out[MAX_DH];\n"
    "    const int qbase = h * T * d_h + qi * d_h;\n"
    "    const int kvbase = h * ctx_len * d_h;\n"
    "    for (int d = lid; d < d_h; d += KV_TILE) acc_out[d] = 0.0f;\n"
    "    if (lid == 0) { run_max = -INFINITY; run_sum = 0.0f; }\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    const int n_tiles = (ctx_len + KV_TILE - 1) / KV_TILE;\n"
    "    for (int t = 0; t < n_tiles; ++t) {\n"
    "        const int kv_off = t * KV_TILE;\n"
    "        const int ki = kv_off + lid;\n"
    "        float score = -INFINITY;\n"
    "        if (ki < ctx_len) {\n"
    "            const bool masked = causal && (ki > qi + ctx_len - T);\n"
    "            if (!masked) {\n"
    "                float acc = 0.0f;\n"
    "                const int kval = kvbase + ki * d_h;\n"
    "                int d = 0;\n"
    "                for (; d + 3 < d_h; d += 4) {\n"
    "                    const float4 qv = vload4(0, &Q[qbase + d]);\n"
    "                    const float4 kvv = vload4(0, &K[kval + d]);\n"
    "                    acc += dot(qv, kvv);\n"
    "                }\n"
    "                for (; d < d_h; ++d) acc += Q[qbase + d] * K[kval + d];\n"
    "                score = acc * scale;\n"
    "            }\n"
    "        }\n"
    "        scores_tile[lid] = score;\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        red_buf[lid] = score;\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        for (int stride = KV_TILE / 2; stride > 0; stride >>= 1) {\n"
    "            if (lid < stride) {\n"
    "                float other = red_buf[lid + stride];\n"
    "                if (other > red_buf[lid]) red_buf[lid] = other;\n"
    "            }\n"
    "            barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        }\n"
    "        const float tile_max = red_buf[0];\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        const float new_max = max(run_max, tile_max);\n"
    "        const float correction = (run_max == -INFINITY) ? 0.0f : exp(run_max - new_max);\n"
    "        float p = (scores_tile[lid] == -INFINITY) ? 0.0f : exp(scores_tile[lid] - new_max);\n"
    "        red_buf[lid] = p;\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        for (int stride = KV_TILE / 2; stride > 0; stride >>= 1) {\n"
    "            if (lid < stride) red_buf[lid] += red_buf[lid + stride];\n"
    "            barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        }\n"
    "        const float tile_sum = red_buf[0];\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        for (int d = lid; d < d_h; d += KV_TILE) {\n"
    "            float pv = 0.0f;\n"
    "            for (int j = 0; j < KV_TILE; ++j) {\n"
    "                const int kj = kv_off + j;\n"
    "                if (kj >= ctx_len) break;\n"
    "                const float pj = (scores_tile[j] == -INFINITY) ? 0.0f : exp(scores_tile[j] - new_max);\n"
    "                pv += pj * V[kvbase + kj * d_h + d];\n"
    "            }\n"
    "            acc_out[d] = acc_out[d] * correction + pv;\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "        if (lid == 0) {\n"
    "            run_sum = run_sum * correction + tile_sum;\n"
    "            run_max = new_max;\n"
    "        }\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    const float inv_sum = 1.0f / (run_sum + 1e-9f);\n"
    "    for (int d = lid; d < d_h; d += KV_TILE) {\n"
    "        out[qbase + d] = acc_out[d] * inv_sum;\n"
    "    }\n"
    "}\n";

// -----------------------------------------------------------------------------
// layer_norm — one work-group (256 threads) per row; block reduction for
// mean and variance. Matches ops::layer_norm exactly: mean over D, biased
// variance (/D, not /(D-1)), eps added before the sqrt.
// -----------------------------------------------------------------------------
static constexpr char LAYER_NORM_SOURCE_PLAIN[] =
    "__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void layer_norm(\n"
    "    __global const float* x,\n"
    "    __global const float* w,\n"
    "    __global const float* b,\n"
    "    __global float* out,\n"
    "    const int rows,\n"
    "    const int cols,\n"
    "    const float eps) {\n"
    "    const int rid = get_group_id(0);\n"
    "    const int lid = get_local_id(0);\n"
    "    if (rid >= rows) return;\n"
    "    const int base = rid * cols;\n"
    "    __local float red_buf[256];\n"
    "    float local_sum = 0.0f;\n"
    "    for (int c = lid; c < cols; c += 256) local_sum += x[base + c];\n"
    "    red_buf[lid] = local_sum;\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    for (int stride = 128; stride > 0; stride >>= 1) {\n"
    "        if (lid < stride) red_buf[lid] += red_buf[lid + stride];\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    const float mean = red_buf[0] / cols;\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    float local_var = 0.0f;\n"
    "    for (int c = lid; c < cols; c += 256) {\n"
    "        const float diff = x[base + c] - mean;\n"
    "        local_var += diff * diff;\n"
    "    }\n"
    "    red_buf[lid] = local_var;\n"
    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    for (int stride = 128; stride > 0; stride >>= 1) {\n"
    "        if (lid < stride) red_buf[lid] += red_buf[lid + stride];\n"
    "        barrier(CLK_LOCAL_MEM_FENCE);\n"
    "    }\n"
    "    const float var = red_buf[0] / cols + eps;\n"
    "    const float inv_std = rsqrt(var);\n"
    "    for (int c = lid; c < cols; c += 256) {\n"
    "        out[base + c] = (x[base + c] - mean) * inv_std * w[c] + b[c];\n"
    "    }\n"
    "}\n";

// -----------------------------------------------------------------------------
// gelu_inplace — tanh-approximation GELU, constants match ops::gelu_inplace
// bit-for-bit (sqrt(2/pi) and 0.044715f).
// -----------------------------------------------------------------------------
static constexpr char GELU_SOURCE_PLAIN[] =
    "__kernel void gelu_inplace(__global float* x, const int n) {\n"
    "    const int i = get_global_id(0);\n"
    "    if (i >= n) return;\n"
    "    const float c1 = 0.7978845608028654f;\n"
    "    const float c2 = 0.044715f;\n"
    "    const float v = x[i];\n"
    "    const float inner = c1 * (v + c2 * v * v * v);\n"
    "    x[i] = 0.5f * v * (1.0f + tanh(inner));\n"
    "}\n";

// -----------------------------------------------------------------------------
// add_inplace — dst[i] += src[i], matches ops::add_inplace.
// -----------------------------------------------------------------------------
static constexpr char ADD_INPLACE_SOURCE_PLAIN[] =
    "__kernel void add_inplace(__global float* dst, __global const float* src, const int n) {\n"
    "    const int i = get_global_id(0);\n"
    "    if (i >= n) return;\n"
    "    dst[i] += src[i];\n"
    "}\n";

static const auto GEMM_SOURCE_ENC             = xor_encrypt(GEMM_SOURCE_PLAIN);
static const auto SOFTMAX_SOURCE_ENC          = xor_encrypt(SOFTMAX_SOURCE_PLAIN);
static const auto ATTENTION_SOURCE_ENC        = xor_encrypt(ATTENTION_SOURCE_PLAIN);
static const auto FLASH_ATTENTION_SOURCE_ENC  = xor_encrypt(FLASH_ATTENTION_SOURCE_PLAIN);
static const auto LAYER_NORM_SOURCE_ENC       = xor_encrypt(LAYER_NORM_SOURCE_PLAIN);
static const auto GELU_SOURCE_ENC             = xor_encrypt(GELU_SOURCE_PLAIN);
static const auto ADD_INPLACE_SOURCE_ENC      = xor_encrypt(ADD_INPLACE_SOURCE_PLAIN);

inline std::string gemm_kernel_source() {
    return decrypt_kernel(GEMM_SOURCE_ENC);
}

inline std::string softmax_kernel_source() {
    return decrypt_kernel(SOFTMAX_SOURCE_ENC);
}

inline std::string attention_kernel_source() {
    return decrypt_kernel(ATTENTION_SOURCE_ENC);
}

inline std::string flash_attention_kernel_source() {
    return decrypt_kernel(FLASH_ATTENTION_SOURCE_ENC);
}

inline std::string layer_norm_kernel_source() {
    return decrypt_kernel(LAYER_NORM_SOURCE_ENC);
}

inline std::string gelu_kernel_source() {
    return decrypt_kernel(GELU_SOURCE_ENC);
}

inline std::string add_inplace_kernel_source() {
    return decrypt_kernel(ADD_INPLACE_SOURCE_ENC);
}

} // namespace opencl
} // namespace backend
} // namespace vlrf
