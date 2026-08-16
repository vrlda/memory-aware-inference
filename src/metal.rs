//! Minimal Metal device probe.
//!
//! This is intentionally smaller than the eventual tensor backend. It proves
//! device discovery and exposes limits the memory planner must respect.

#[cfg(target_os = "macos")]
use crate::cache::KvCache;
#[cfg(target_os = "macos")]
use crate::model::{GgufTensorView, StagedQwenLayer, TensorView};
#[cfg(target_os = "macos")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(target_os = "macos")]
use std::sync::Arc;
#[cfg(target_os = "macos")]
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalDeviceInfo {
    pub name: String,
    pub registry_id: u64,
    pub recommended_max_working_set_bytes: u64,
    pub max_buffer_bytes: u64,
    pub has_unified_memory: bool,
}

#[cfg(target_os = "macos")]
pub struct AttentionDecodeInput<'a> {
    pub query: &'a [f32],
    pub key_cache: &'a [f32],
    pub value_cache: &'a [f32],
    pub new_keys: &'a [f32],
    pub new_values: &'a [f32],
    pub query_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub cached_tokens: usize,
    pub cache_capacity_tokens: usize,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, PartialEq)]
pub struct ChainedAttentionOutput {
    pub projected: Vec<f32>,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
pub struct ChainedAttentionConfig {
    pub query_heads: usize,
    pub key_value_heads: usize,
    pub head_dim: usize,
    pub cached_tokens: usize,
    pub cache_capacity_tokens: usize,
    pub position: usize,
    pub rope_theta: f32,
    pub epsilon: f32,
}

#[cfg(target_os = "macos")]
pub struct ChainedAttentionTensorRequest<'a, 'b> {
    pub q_tensor: &'a TensorView<'b>,
    pub k_tensor: &'a TensorView<'b>,
    pub v_tensor: &'a TensorView<'b>,
    pub o_tensor: &'a TensorView<'b>,
    pub q_norm_bytes: &'a [u8],
    pub k_norm_bytes: &'a [u8],
    pub input: &'a [f32],
    pub key_cache: &'a [f32],
    pub value_cache: &'a [f32],
    pub config: ChainedAttentionConfig,
}

#[cfg(target_os = "macos")]
pub struct ChainedAttentionBufferRequest<'a> {
    pub q: (&'a Bf16Weight, usize, usize),
    pub k: (&'a Bf16Weight, usize, usize),
    pub v: (&'a Bf16Weight, usize, usize),
    pub o: (&'a Bf16Weight, usize, usize),
    pub q_norm_bytes: &'a [u8],
    pub k_norm_bytes: &'a [u8],
    pub input: &'a [f32],
    pub key_cache: &'a [f32],
    pub value_cache: &'a [f32],
    pub config: ChainedAttentionConfig,
}

#[cfg(target_os = "macos")]
struct AttentionBufferRefs<'a> {
    query: &'a metal::BufferRef,
    key_cache: &'a metal::BufferRef,
    value_cache: &'a metal::BufferRef,
    new_keys: &'a metal::BufferRef,
    new_values: &'a metal::BufferRef,
    output: &'a metal::BufferRef,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct NormParams {
    heads: usize,
    head_dim: usize,
    epsilon: f32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
struct RopeParams {
    heads: usize,
    head_dim: usize,
    position: usize,
    theta: f32,
}

#[cfg(target_os = "macos")]
const RMS_NORM_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void si_rms_norm(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& length [[buffer(3)]],
    constant float& epsilon [[buffer(4)]],
    uint3 thread_id [[thread_position_in_grid]]) {
    if (thread_id.x != 0) {
        return;
    }
    float sum = 0.0f;
    for (uint index = 0; index < length; ++index) {
        sum += input[index] * input[index];
    }
    float inverse_rms = rsqrt(sum / static_cast<float>(length) + epsilon);
    for (uint index = 0; index < length; ++index) {
        output[index] = input[index] * inverse_rms * weight[index];
    }
}

kernel void si_rms_norm_bf16_heads(
    device const float* input [[buffer(0)]],
    device const ushort* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    uint3 thread_id [[thread_position_in_grid]]) {
    uint head = thread_id.x;
    if (head >= heads) {
        return;
    }
    uint base = head * head_dim;
    float sum = 0.0f;
    for (uint index = 0; index < head_dim; ++index) {
        float value = input[base + index];
        sum += value * value;
    }
    float inverse_rms = rsqrt(sum / static_cast<float>(head_dim) + epsilon);
    for (uint index = 0; index < head_dim; ++index) {
        uint bits = uint(weight[index]) << 16;
        output[base + index] = input[base + index] * inverse_rms * as_type<float>(bits);
    }
}

kernel void si_rms_norm_gated(
    device const float* input [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& heads [[buffer(4)]],
    constant uint& head_dim [[buffer(5)]],
    constant float& epsilon [[buffer(6)]],
    uint3 thread_id [[thread_position_in_grid]]) {
    uint head = thread_id.x;
    if (head >= heads) {
        return;
    }
    uint base = head * head_dim;
    float sum = 0.0f;
    for (uint index = 0u; index < head_dim; ++index) {
        float value = input[base + index];
        sum += value * value;
    }
    float inverse_rms = rsqrt(sum / static_cast<float>(head_dim) + epsilon);
    for (uint index = 0u; index < head_dim; ++index) {
        float gate_value = gate[base + index];
        float silu = gate_value / (1.0f + exp(-gate_value));
        output[base + index] = input[base + index] * inverse_rms * weight[index] * silu;
    }
}

kernel void si_rms_norm_heads(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& heads [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    constant float& epsilon [[buffer(5)]],
    uint3 thread_id [[thread_position_in_grid]]) {
    uint head = thread_id.x;
    if (head >= heads) {
        return;
    }
    uint base = head * head_dim;
    float sum = 0.0f;
    for (uint index = 0u; index < head_dim; ++index) {
        float value = input[base + index];
        sum += value * value;
    }
    float inverse_rms = rsqrt(sum / static_cast<float>(head_dim) + epsilon);
    for (uint index = 0u; index < head_dim; ++index) {
        output[base + index] = input[base + index] * inverse_rms * (1.0f + weight[index]);
    }
}

kernel void si_bf16_matvec(
    device const ushort* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    uint row = threadgroup_id.x;
    if (row >= rows) {
        return;
    }
    float sum = 0.0f;
    if ((columns & 3u) == 0u) {
        uint vector_columns = columns / 4u;
        device const ushort4* vector_weights = reinterpret_cast<device const ushort4*>(weights);
        device const float4* vector_input = reinterpret_cast<device const float4*>(input);
        for (uint index = thread_id.x; index < vector_columns; index += threads_per_group.x) {
            uint4 bits = uint4(vector_weights[row * vector_columns + index]) << 16;
            sum += dot(as_type<float4>(bits), vector_input[index]);
        }
    } else {
        for (uint column = thread_id.x; column < columns; column += threads_per_group.x) {
            uint bits = uint(weights[row * columns + column]) << 16;
            sum += as_type<float>(bits) * input[column];
        }
    }
    float simd_total = simd_sum(sum);
    threadgroup float partials[32];
    if (simd_lane == 0) {
        partials[simdgroup_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_id.x == 0) {
        float total = 0.0f;
        uint simdgroups = (threads_per_group.x + 31u) / 32u;
        for (uint index = 0; index < simdgroups; ++index) {
            total += partials[index];
        }
        output[row] = total;
    }
}

kernel void si_f32_matvec(
    device const float* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    uint row = threadgroup_id.x;
    if (row >= rows) {
        return;
    }
    float sum = 0.0f;
    if ((columns & 3u) == 0u) {
        uint vector_columns = columns / 4u;
        device const float4* vector_weights = reinterpret_cast<device const float4*>(weights);
        device const float4* vector_input = reinterpret_cast<device const float4*>(input);
        for (uint index = thread_id.x; index < vector_columns; index += threads_per_group.x) {
            sum += dot(vector_weights[row * vector_columns + index], vector_input[index]);
        }
    } else {
        for (uint column = thread_id.x; column < columns; column += threads_per_group.x) {
            sum += weights[row * columns + column] * input[column];
        }
    }
    float simd_total = simd_sum(sum);
    threadgroup float partials[32];
    if (simd_lane == 0) {
        partials[simdgroup_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_id.x == 0) {
        float total = 0.0f;
        uint simdgroups = (threads_per_group.x + 31u) / 32u;
        for (uint index = 0; index < simdgroups; ++index) {
            total += partials[index];
        }
        output[row] = total;
    }
}

// Exact GGML Q4_K matvec. The 144-byte block is decoded in registers while
// the input vector is consumed, so no expanded FP32/BF16 weight allocation is
// needed on the device.
kernel void si_q4_k_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    uint row = threadgroup_id.x;
    if (row >= rows) {
        return;
    }
    uint blocks_per_row = columns / 256u;
    float sum = 0.0f;
    for (uint column = thread_id.x; column < columns; column += threads_per_group.x) {
        uint local = column & 255u;
        uint group = local / 64u;
        uint within_group = local & 63u;
        uint quant_byte = within_group & 31u;
        uint block = column / 256u;
        uint block_base = (row * blocks_per_row + block) * 144u;
        ushort d_bits = ushort(weights[block_base]) |
            (ushort(weights[block_base + 1u]) << 8);
        ushort min_bits = ushort(weights[block_base + 2u]) |
            (ushort(weights[block_base + 3u]) << 8);
        float d = float(as_type<half>(d_bits));
        float min = float(as_type<half>(min_bits));
        uint scale_index = group * 2u + (within_group >= 32u ? 1u : 0u);
        uint scale_byte = block_base + 4u + scale_index;
        uint scale;
        uint minimum;
        if (scale_index < 4u) {
            scale = uint(weights[scale_byte] & 0x3fu);
            minimum = uint(weights[block_base + 8u + scale_index] & 0x3fu);
        } else {
            uint packed_scale = block_base + 8u + scale_index;
            scale = uint(weights[packed_scale] & 0x0fu) |
                (uint(weights[block_base + scale_index]) >> 6u << 4u);
            minimum = uint(weights[packed_scale] >> 4u) |
                (uint(weights[block_base + 4u + scale_index]) >> 6u << 4u);
        }
        uchar packed = weights[block_base + 16u + group * 32u + quant_byte];
        uint quant = within_group >= 32u ? uint(packed >> 4u) : uint(packed & 0x0fu);
        float value = d * float(scale) * float(quant) - min * float(minimum);
        sum += value * input[column];
    }
    float simd_total = simd_sum(sum);
    threadgroup float partials[32];
    if (simd_lane == 0) {
        partials[simdgroup_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_id.x == 0) {
        float total = 0.0f;
        uint simdgroups = (threads_per_group.x + 31u) / 32u;
        for (uint index = 0; index < simdgroups; ++index) {
            total += partials[index];
        }
        output[row] = total;
    }
}

// Q4_K row-blocked variant. Four SIMD groups share one threadgroup, with one
// SIMD group owning one output row. This removes the cross-SIMD partial buffer
// and the barrier from the ordinary one-row dispatch while preserving the
// exact decode and FP32 accumulation order within each row.
kernel void si_q4_k_matvec_rows4(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    uint row = threadgroup_id.x * 4u + simdgroup_id;
    if (row >= rows) {
        return;
    }
    uint blocks_per_row = columns / 256u;
    float sum = 0.0f;
    for (uint column = simd_lane; column < columns; column += 32u) {
        uint local = column & 255u;
        uint group = local / 64u;
        uint within_group = local & 63u;
        uint quant_byte = within_group & 31u;
        uint block = column / 256u;
        uint block_base = (row * blocks_per_row + block) * 144u;
        ushort d_bits = ushort(weights[block_base]) |
            (ushort(weights[block_base + 1u]) << 8);
        ushort min_bits = ushort(weights[block_base + 2u]) |
            (ushort(weights[block_base + 3u]) << 8);
        float d = float(as_type<half>(d_bits));
        float min = float(as_type<half>(min_bits));
        uint scale_index = group * 2u + (within_group >= 32u ? 1u : 0u);
        uint scale;
        uint minimum;
        if (scale_index < 4u) {
            scale = uint(weights[block_base + 4u + scale_index] & 0x3fu);
            minimum = uint(weights[block_base + 8u + scale_index] & 0x3fu);
        } else {
            uint packed_scale = block_base + 8u + scale_index;
            scale = uint(weights[packed_scale] & 0x0fu) |
                (uint(weights[block_base + scale_index]) >> 6u << 4u);
            minimum = uint(weights[packed_scale] >> 4u) |
                (uint(weights[block_base + 4u + scale_index]) >> 6u << 4u);
        }
        uchar packed = weights[block_base + 16u + group * 32u + quant_byte];
        uint quant = within_group >= 32u ? uint(packed >> 4u) : uint(packed & 0x0fu);
        float value = d * float(scale) * float(quant) - min * float(minimum);
        sum += value * input[column];
    }
    float total = simd_sum(sum);
    if (simd_lane == 0u) {
        output[row] = total;
    }
}

// Four rows per threadgroup, four SIMD groups per row. This is the high
// occupancy variant: 512 threads perform the same per-row reduction as the
// ordinary kernel while amortizing dispatch and input-cache overhead across
// four output rows.
kernel void si_q4_k_matvec_rows4x128(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    uint local_row = simdgroup_id / 4u;
    uint row = threadgroup_id.x * 4u + local_row;
    uint local_simd = simdgroup_id & 3u;
    float sum = 0.0f;
    if (row < rows) {
        uint blocks_per_row = columns / 256u;
        for (uint column = local_simd * 32u + simd_lane;
             column < columns;
             column += 128u) {
            uint local = column & 255u;
            uint group = local / 64u;
            uint within_group = local & 63u;
            uint quant_byte = within_group & 31u;
            uint block = column / 256u;
            uint block_base = (row * blocks_per_row + block) * 144u;
            ushort d_bits = ushort(weights[block_base]) |
                (ushort(weights[block_base + 1u]) << 8u);
            ushort min_bits = ushort(weights[block_base + 2u]) |
                (ushort(weights[block_base + 3u]) << 8u);
            float d = float(as_type<half>(d_bits));
            float min = float(as_type<half>(min_bits));
            uint scale_index = group * 2u + (within_group >= 32u ? 1u : 0u);
            uint scale;
            uint minimum;
            if (scale_index < 4u) {
                scale = uint(weights[block_base + 4u + scale_index] & 0x3fu);
                minimum = uint(weights[block_base + 8u + scale_index] & 0x3fu);
            } else {
                uint packed_scale = block_base + 8u + scale_index;
                scale = uint(weights[packed_scale] & 0x0fu) |
                    (uint(weights[block_base + scale_index]) >> 6u << 4u);
                minimum = uint(weights[packed_scale] >> 4u) |
                    (uint(weights[block_base + 4u + scale_index]) >> 6u << 4u);
            }
            uchar packed = weights[block_base + 16u + group * 32u + quant_byte];
            uint quant = within_group >= 32u ? uint(packed >> 4u) : uint(packed & 0x0fu);
            float value = d * float(scale) * float(quant) - min * float(minimum);
            sum += value * input[column];
        }
    }
    float total = simd_sum(sum);
    threadgroup float partials[4][4];
    if (simd_lane == 0u) {
        partials[local_row][local_simd] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0u && local_simd == 0u && row < rows) {
        output[row] = partials[local_row][0] + partials[local_row][1]
            + partials[local_row][2] + partials[local_row][3];
    }
}

// Exact Q4_K fused SwiGLU input projections. Both matrices are decoded in
// registers during one dispatch; only the activated product is written out.
kernel void si_q4_k_fused_gate_up(
    device const uchar* gate_weights [[buffer(0)]],
    device const uchar* up_weights [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& columns [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    uint row = threadgroup_id.x;
    if (row >= rows) {
        return;
    }
    uint blocks_per_row = columns / 256u;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (uint column = thread_id.x; column < columns; column += threads_per_group.x) {
        uint local = column & 255u;
        uint group = local / 64u;
        uint within_group = local & 63u;
        uint quant_byte = within_group & 31u;
        uint block = column / 256u;
        uint gate_base = (row * blocks_per_row + block) * 144u;
        uint up_base = gate_base;
        ushort gate_d_bits = ushort(gate_weights[gate_base]) |
            (ushort(gate_weights[gate_base + 1u]) << 8);
        ushort gate_min_bits = ushort(gate_weights[gate_base + 2u]) |
            (ushort(gate_weights[gate_base + 3u]) << 8);
        ushort up_d_bits = ushort(up_weights[up_base]) |
            (ushort(up_weights[up_base + 1u]) << 8);
        ushort up_min_bits = ushort(up_weights[up_base + 2u]) |
            (ushort(up_weights[up_base + 3u]) << 8);
        float gate_d = float(as_type<half>(gate_d_bits));
        float gate_min = float(as_type<half>(gate_min_bits));
        float up_d = float(as_type<half>(up_d_bits));
        float up_min = float(as_type<half>(up_min_bits));
        uint scale_index = group * 2u + (within_group >= 32u ? 1u : 0u);
        uint gate_scale;
        uint gate_minimum;
        uint up_scale;
        uint up_minimum;
        if (scale_index < 4u) {
            gate_scale = uint(gate_weights[gate_base + 4u + scale_index] & 0x3fu);
            gate_minimum = uint(gate_weights[gate_base + 8u + scale_index] & 0x3fu);
            up_scale = uint(up_weights[up_base + 4u + scale_index] & 0x3fu);
            up_minimum = uint(up_weights[up_base + 8u + scale_index] & 0x3fu);
        } else {
            uint gate_packed = gate_base + 8u + scale_index;
            uint up_packed = up_base + 8u + scale_index;
            gate_scale = uint(gate_weights[gate_packed] & 0x0fu) |
                (uint(gate_weights[gate_base + scale_index]) >> 6u << 4u);
            gate_minimum = uint(gate_weights[gate_packed] >> 4u) |
                (uint(gate_weights[gate_base + 4u + scale_index]) >> 6u << 4u);
            up_scale = uint(up_weights[up_packed] & 0x0fu) |
                (uint(up_weights[up_base + scale_index]) >> 6u << 4u);
            up_minimum = uint(up_weights[up_packed] >> 4u) |
                (uint(up_weights[up_base + 4u + scale_index]) >> 6u << 4u);
        }
        uchar gate_packed = gate_weights[gate_base + 16u + group * 32u + quant_byte];
        uchar up_packed = up_weights[up_base + 16u + group * 32u + quant_byte];
        uint gate_quant = within_group >= 32u ? uint(gate_packed >> 4u) : uint(gate_packed & 0x0fu);
        uint up_quant = within_group >= 32u ? uint(up_packed >> 4u) : uint(up_packed & 0x0fu);
        float gate_value = gate_d * float(gate_scale) * float(gate_quant) - gate_min * float(gate_minimum);
        float up_value = up_d * float(up_scale) * float(up_quant) - up_min * float(up_minimum);
        gate_sum += gate_value * input[column];
        up_sum += up_value * input[column];
    }
    float gate_simd_total = simd_sum(gate_sum);
    float up_simd_total = simd_sum(up_sum);
    threadgroup float gate_partials[32];
    threadgroup float up_partials[32];
    if (simd_lane == 0) {
        gate_partials[simdgroup_id] = gate_simd_total;
        up_partials[simdgroup_id] = up_simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_id.x == 0) {
        float gate_total = 0.0f;
        float up_total = 0.0f;
        uint simdgroups = (threads_per_group.x + 31u) / 32u;
        for (uint index = 0; index < simdgroups; ++index) {
            gate_total += gate_partials[index];
            up_total += up_partials[index];
        }
        output[row] = gate_total / (1.0f + exp(-gate_total)) * up_total;
    }
}

kernel void si_q4_k_embedding_row(
    device const uchar* weights [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint3 thread_id [[thread_position_in_grid]]) {
    uint column = thread_id.x;
    if (column >= length) {
        return;
    }
    uint local = column & 255u;
    uint group = local / 64u;
    uint within_group = local & 63u;
    uint quant_byte = within_group & 31u;
    uint block_base = (column / 256u) * 144u;
    ushort d_bits = ushort(weights[block_base]) |
        (ushort(weights[block_base + 1u]) << 8);
    ushort min_bits = ushort(weights[block_base + 2u]) |
        (ushort(weights[block_base + 3u]) << 8);
    float d = float(as_type<half>(d_bits));
    float min = float(as_type<half>(min_bits));
    uint scale_index = group * 2u + (within_group >= 32u ? 1u : 0u);
    uint scale;
    uint minimum;
    if (scale_index < 4u) {
        scale = uint(weights[block_base + 4u + scale_index] & 0x3fu);
        minimum = uint(weights[block_base + 8u + scale_index] & 0x3fu);
    } else {
        uint packed_scale = block_base + 8u + scale_index;
        scale = uint(weights[packed_scale] & 0x0fu) |
            (uint(weights[block_base + scale_index]) >> 6u << 4u);
        minimum = uint(weights[packed_scale] >> 4u) |
            (uint(weights[block_base + 4u + scale_index]) >> 6u << 4u);
    }
    uchar packed = weights[block_base + 16u + group * 32u + quant_byte];
    uint quant = within_group >= 32u ? uint(packed >> 4u) : uint(packed & 0x0fu);
    output[column] = d * float(scale) * float(quant) - min * float(minimum);
}

// Correctness-first recurrent Gated DeltaNet update. One thread owns one
// head, so state updates are disjoint; the state layout is
// [head][key_dim][value_dim].
kernel void si_gated_delta_step(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    device const float* beta [[buffer(4)]],
    device float* state [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    constant uint& key_dim [[buffer(8)]],
    constant uint& value_dim [[buffer(9)]],
    uint3 thread_id [[thread_position_in_grid]]) {
    uint head = thread_id.x;
    if (head >= heads || value_dim > 256u) {
        return;
    }
    float memory[256];
    float delta[256];
    uint key_base = head * key_dim;
    uint value_base = head * value_dim;
    uint state_base = head * key_dim * value_dim;
    float decay = exp(gate[head]);
    for (uint key_index = 0u; key_index < key_dim; ++key_index) {
        uint row_base = state_base + key_index * value_dim;
        for (uint value_index = 0u; value_index < value_dim; ++value_index) {
            state[row_base + value_index] *= decay;
        }
    }
    for (uint value_index = 0u; value_index < value_dim; ++value_index) {
        memory[value_index] = 0.0f;
    }
    for (uint key_index = 0u; key_index < key_dim; ++key_index) {
        uint row_base = state_base + key_index * value_dim;
        float key_value = key[key_base + key_index];
        for (uint value_index = 0u; value_index < value_dim; ++value_index) {
            memory[value_index] += state[row_base + value_index] * key_value;
        }
    }
    for (uint value_index = 0u; value_index < value_dim; ++value_index) {
        delta[value_index] = (value[value_base + value_index] - memory[value_index]) * beta[head];
    }
    for (uint key_index = 0u; key_index < key_dim; ++key_index) {
        uint row_base = state_base + key_index * value_dim;
        float key_value = key[key_base + key_index];
        for (uint value_index = 0u; value_index < value_dim; ++value_index) {
            state[row_base + value_index] += key_value * delta[value_index];
        }
    }
    for (uint value_index = 0u; value_index < value_dim; ++value_index) {
        float result = 0.0f;
        for (uint key_index = 0u; key_index < key_dim; ++key_index) {
            result += state[state_base + key_index * value_dim + value_index]
                * query[key_base + key_index];
        }
        output[value_base + value_index] = result;
    }
}

// Parallel Gated DeltaNet update. One thread owns one [head][value] column;
// it performs the same decay, memory read, delta update, and output dot
// product as si_gated_delta_step, but exposes value_dim-way parallelism per
// head instead of serializing the whole recurrent matrix on one thread.
kernel void si_gated_delta_step_parallel(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const float* gate [[buffer(3)]],
    device const float* beta [[buffer(4)]],
    device float* state [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    constant uint& key_dim [[buffer(8)]],
    constant uint& value_dim [[buffer(9)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]]) {
    uint head = threadgroup_id.x;
    uint value_index = thread_id.x;
    if (head >= heads || value_index >= value_dim) {
        return;
    }
    threadgroup float key_query_dot;
    if (value_index == 0u) {
        float dot = 0.0f;
        uint key_base = head * key_dim;
        for (uint key_index = 0u; key_index < key_dim; ++key_index) {
            dot += key[key_base + key_index] * query[key_base + key_index];
        }
        key_query_dot = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint key_base = head * key_dim;
    uint value_base = head * value_dim;
    uint state_base = head * key_dim * value_dim + value_index;
    float decay = exp(gate[head]);
    float memory = 0.0f;
    float query_sum = 0.0f;
    for (uint key_index = 0u; key_index < key_dim; ++key_index) {
        uint index = state_base + key_index * value_dim;
        float old_value = state[index] * decay;
        state[index] = old_value;
        memory += old_value * key[key_base + key_index];
        query_sum += old_value * query[key_base + key_index];
    }
    float delta = (value[value_base + value_index] - memory) * beta[head];
    for (uint key_index = 0u; key_index < key_dim; ++key_index) {
        uint index = state_base + key_index * value_dim;
        state[index] += key[key_base + key_index] * delta;
    }
    output[value_base + value_index] = query_sum + delta * key_query_dot;
}

// Exact single-token depthwise causal convolution update used by Qwen3.5/3.6
// Gated DeltaNet. State is [channel][kernel_size - 1] in chronological order;
// weights are [channel][kernel_size].
kernel void si_causal_conv1d_step(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* state [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& channels [[buffer(4)]],
    constant uint& kernel_size [[buffer(5)]],
    uint3 thread_id [[thread_position_in_grid]]) {
    uint channel = thread_id.x;
    if (channel >= channels || kernel_size < 2u) {
        return;
    }
    uint state_base = channel * (kernel_size - 1u);
    uint weight_base = channel * kernel_size;
    float value = input[channel] * weights[weight_base + kernel_size - 1u];
    for (uint tap = 0u; tap + 1u < kernel_size; ++tap) {
        value += state[state_base + tap] * weights[weight_base + tap];
    }
    output[channel] = value / (1.0f + exp(-value));
    for (uint tap = 0u; tap + 1u < kernel_size - 1u; ++tap) {
        state[state_base + tap] = state[state_base + tap + 1u];
    }
    state[state_base + kernel_size - 2u] = input[channel];
}

kernel void si_q5_k_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    uint row = threadgroup_id.x;
    if (row >= rows) {
        return;
    }
    uint blocks_per_row = columns / 256u;
    float sum = 0.0f;
    for (uint column = thread_id.x; column < columns; column += threads_per_group.x) {
        uint local = column & 255u;
        uint group = local / 64u;
        uint within_group = local & 63u;
        uint quant_byte = within_group & 31u;
        uint block = column / 256u;
        uint block_base = (row * blocks_per_row + block) * 176u;
        ushort d_bits = ushort(weights[block_base]) |
            (ushort(weights[block_base + 1u]) << 8);
        ushort min_bits = ushort(weights[block_base + 2u]) |
            (ushort(weights[block_base + 3u]) << 8);
        float d = float(as_type<half>(d_bits));
        float min = float(as_type<half>(min_bits));
        uint scale_index = group * 2u + (within_group >= 32u ? 1u : 0u);
        uint scale;
        uint minimum;
        if (scale_index < 4u) {
            scale = uint(weights[block_base + 4u + scale_index] & 0x3fu);
            minimum = uint(weights[block_base + 8u + scale_index] & 0x3fu);
        } else {
            uint packed_scale = block_base + 8u + scale_index;
            scale = uint(weights[packed_scale] & 0x0fu) |
                (uint(weights[block_base + scale_index]) >> 6u << 4u);
            minimum = uint(weights[packed_scale] >> 4u) |
                (uint(weights[block_base + 4u + scale_index]) >> 6u << 4u);
        }
        uchar packed = weights[block_base + 48u + group * 32u + quant_byte];
        uint high_mask = within_group >= 32u ? (2u << (group * 2u)) : (1u << (group * 2u));
        uint quant = within_group >= 32u ? uint(packed >> 4u) : uint(packed & 0x0fu);
        if ((uint(weights[block_base + 16u + quant_byte]) & high_mask) != 0u) {
            quant += 16u;
        }
        float value = d * float(scale) * float(quant) - min * float(minimum);
        sum += value * input[column];
    }
    float simd_total = simd_sum(sum);
    threadgroup float partials[32];
    if (simd_lane == 0) {
        partials[simdgroup_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_id.x == 0) {
        float total = 0.0f;
        uint simdgroups = (threads_per_group.x + 31u) / 32u;
        for (uint index = 0; index < simdgroups; ++index) {
            total += partials[index];
        }
        output[row] = total;
    }
}

// Q5_K row-blocked variant; one SIMD group computes one output row.
kernel void si_q5_k_matvec_rows4(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    uint row = threadgroup_id.x * 4u + simdgroup_id;
    if (row >= rows) {
        return;
    }
    uint blocks_per_row = columns / 256u;
    float sum = 0.0f;
    for (uint column = simd_lane; column < columns; column += 32u) {
        uint local = column & 255u;
        uint group = local / 64u;
        uint within_group = local & 63u;
        uint quant_byte = within_group & 31u;
        uint block = column / 256u;
        uint block_base = (row * blocks_per_row + block) * 176u;
        ushort d_bits = ushort(weights[block_base]) |
            (ushort(weights[block_base + 1u]) << 8);
        ushort min_bits = ushort(weights[block_base + 2u]) |
            (ushort(weights[block_base + 3u]) << 8);
        float d = float(as_type<half>(d_bits));
        float min = float(as_type<half>(min_bits));
        uint scale_index = group * 2u + (within_group >= 32u ? 1u : 0u);
        uint scale;
        uint minimum;
        if (scale_index < 4u) {
            scale = uint(weights[block_base + 4u + scale_index] & 0x3fu);
            minimum = uint(weights[block_base + 8u + scale_index] & 0x3fu);
        } else {
            uint packed_scale = block_base + 8u + scale_index;
            scale = uint(weights[packed_scale] & 0x0fu) |
                (uint(weights[block_base + scale_index]) >> 6u << 4u);
            minimum = uint(weights[packed_scale] >> 4u) |
                (uint(weights[block_base + 4u + scale_index]) >> 6u << 4u);
        }
        uchar packed = weights[block_base + 48u + group * 32u + quant_byte];
        uint high_mask = within_group >= 32u ? (2u << (group * 2u)) : (1u << (group * 2u));
        uint quant = within_group >= 32u ? uint(packed >> 4u) : uint(packed & 0x0fu);
        if ((uint(weights[block_base + 16u + quant_byte]) & high_mask) != 0u) {
            quant += 16u;
        }
        float value = d * float(scale) * float(quant) - min * float(minimum);
        sum += value * input[column];
    }
    float total = simd_sum(sum);
    if (simd_lane == 0u) {
        output[row] = total;
    }
}

kernel void si_q6_k_matvec(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    uint row = threadgroup_id.x;
    if (row >= rows) {
        return;
    }
    uint blocks_per_row = columns / 256u;
    float sum = 0.0f;
    for (uint column = thread_id.x; column < columns; column += threads_per_group.x) {
        uint local = column & 255u;
        uint block_half = local / 128u;
        uint within_half = local & 127u;
        uint quartet = within_half / 32u;
        uint index = within_half & 31u;
        uint block = column / 256u;
        uint block_base = (row * blocks_per_row + block) * 210u;
        uint ql_base = block_base + block_half * 64u;
        uint qh_base = block_base + 128u + block_half * 32u;
        uchar packed = weights[ql_base + (quartet & 1u) * 32u + index];
        uint nibble = quartet >= 2u ? uint(packed >> 4u) : uint(packed & 0x0fu);
        uint high_bits = (uint(weights[qh_base + index]) >> (quartet * 2u)) & 0x03u;
        int quant = int(nibble | (high_bits << 4u)) - 32;
        uint scale_index = (index / 16u) + quartet * 2u;
        int scale = int(weights[block_base + 192u + block_half * 8u + scale_index]);
        if (scale > 127) {
            scale -= 256;
        }
        ushort d_bits = ushort(weights[block_base + 208u]) |
            (ushort(weights[block_base + 209u]) << 8);
        float d = float(as_type<half>(d_bits));
        sum += d * float(scale) * float(quant) * input[column];
    }
    float simd_total = simd_sum(sum);
    threadgroup float partials[32];
    if (simd_lane == 0) {
        partials[simdgroup_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_id.x == 0) {
        float total = 0.0f;
        uint simdgroups = (threads_per_group.x + 31u) / 32u;
        for (uint index = 0; index < simdgroups; ++index) {
            total += partials[index];
        }
        output[row] = total;
    }
}

// Q6_K row-blocked variant. Four SIMD groups share one threadgroup; each
// group performs one row reduction, avoiding a threadgroup partial array and
// barrier while keeping the quantized bytes and FP32 math unchanged.
kernel void si_q6_k_matvec_rows4(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    uint row = threadgroup_id.x * 4u + simdgroup_id;
    if (row >= rows) {
        return;
    }
    uint blocks_per_row = columns / 256u;
    float sum = 0.0f;
    for (uint column = simd_lane; column < columns; column += 32u) {
        uint local = column & 255u;
        uint block_half = local / 128u;
        uint within_half = local & 127u;
        uint quartet = within_half / 32u;
        uint index = within_half & 31u;
        uint block = column / 256u;
        uint block_base = (row * blocks_per_row + block) * 210u;
        uint ql_base = block_base + block_half * 64u;
        uint qh_base = block_base + 128u + block_half * 32u;
        uchar packed = weights[ql_base + (quartet & 1u) * 32u + index];
        uint nibble = quartet >= 2u ? uint(packed >> 4u) : uint(packed & 0x0fu);
        uint high_bits = (uint(weights[qh_base + index]) >> (quartet * 2u)) & 0x03u;
        int quant = int(nibble | (high_bits << 4u)) - 32;
        uint scale_index = (index / 16u) + quartet * 2u;
        int scale = int(weights[block_base + 192u + block_half * 8u + scale_index]);
        if (scale > 127) {
            scale -= 256;
        }
        ushort d_bits = ushort(weights[block_base + 208u]) |
            (ushort(weights[block_base + 209u]) << 8);
        float d = float(as_type<half>(d_bits));
        sum += d * float(scale) * float(quant) * input[column];
    }
    float total = simd_sum(sum);
    if (simd_lane == 0u) {
        output[row] = total;
    }
}

kernel void si_q6_k_matvec_rows4x128(
    device const uchar* weights [[buffer(0)]],
    device const float* input [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    uint local_row = simdgroup_id / 4u;
    uint row = threadgroup_id.x * 4u + local_row;
    uint local_simd = simdgroup_id & 3u;
    float sum = 0.0f;
    if (row < rows) {
        uint blocks_per_row = columns / 256u;
        for (uint column = local_simd * 32u + simd_lane;
             column < columns;
             column += 128u) {
            uint local = column & 255u;
            uint block_half = local / 128u;
            uint within_half = local & 127u;
            uint quartet = within_half / 32u;
            uint index = within_half & 31u;
            uint block = column / 256u;
            uint block_base = (row * blocks_per_row + block) * 210u;
            uint ql_base = block_base + block_half * 64u;
            uint qh_base = block_base + 128u + block_half * 32u;
            uchar packed = weights[ql_base + (quartet & 1u) * 32u + index];
            uint nibble = quartet >= 2u ? uint(packed >> 4u) : uint(packed & 0x0fu);
            uint high_bits = (uint(weights[qh_base + index]) >> (quartet * 2u)) & 0x03u;
            int quant = int(nibble | (high_bits << 4u)) - 32;
            uint scale_index = (index / 16u) + quartet * 2u;
            int scale = int(weights[block_base + 192u + block_half * 8u + scale_index]);
            if (scale > 127) scale -= 256;
            ushort d_bits = ushort(weights[block_base + 208u]) |
                (ushort(weights[block_base + 209u]) << 8u);
            float d = float(as_type<half>(d_bits));
            sum += d * float(scale) * float(quant) * input[column];
        }
    }
    float total = simd_sum(sum);
    threadgroup float partials[4][4];
    if (simd_lane == 0u) {
        partials[local_row][local_simd] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane == 0u && local_simd == 0u && row < rows) {
        output[row] = partials[local_row][0] + partials[local_row][1]
            + partials[local_row][2] + partials[local_row][3];
    }
}

// Batched quantized matvecs used by the GGUF verifier.  One threadgroup still
// owns one output row, but each decoded weight value is reused across up to
// eight candidate input vectors before the next quant block is fetched.
inline float si_q4_k_value(device const uchar* weights, uint block_base, uint local) {
    uint group = local / 64u;
    uint within_group = local & 63u;
    uint quant_byte = within_group & 31u;
    ushort d_bits = ushort(weights[block_base]) |
        (ushort(weights[block_base + 1u]) << 8);
    ushort min_bits = ushort(weights[block_base + 2u]) |
        (ushort(weights[block_base + 3u]) << 8);
    float d = float(as_type<half>(d_bits));
    float min = float(as_type<half>(min_bits));
    uint scale_index = group * 2u + (within_group >= 32u ? 1u : 0u);
    uint scale;
    uint minimum;
    if (scale_index < 4u) {
        scale = uint(weights[block_base + 4u + scale_index] & 0x3fu);
        minimum = uint(weights[block_base + 8u + scale_index] & 0x3fu);
    } else {
        uint packed_scale = block_base + 8u + scale_index;
        scale = uint(weights[packed_scale] & 0x0fu) |
            (uint(weights[block_base + scale_index]) >> 6u << 4u);
        minimum = uint(weights[packed_scale] >> 4u) |
            (uint(weights[block_base + 4u + scale_index]) >> 6u << 4u);
    }
    uchar packed = weights[block_base + 16u + group * 32u + quant_byte];
    uint quant = within_group >= 32u ? uint(packed >> 4u) : uint(packed & 0x0fu);
    return d * float(scale) * float(quant) - min * float(minimum);
}

kernel void si_q4_k_matmul_many(
    device const uchar* weights [[buffer(0)]],
    device const float* inputs [[buffer(1)]],
    device float* outputs [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    uint row = threadgroup_id.x;
    if (row >= rows || batch == 0u || batch > 8u) {
        return;
    }
    float sums[8];
    for (uint candidate = 0u; candidate < 8u; ++candidate) {
        sums[candidate] = 0.0f;
    }
    uint blocks_per_row = columns / 256u;
    for (uint column = thread_id.x; column < columns; column += threads_per_group.x) {
        uint block = column / 256u;
        float value = si_q4_k_value(weights, (row * blocks_per_row + block) * 144u, column & 255u);
        for (uint candidate = 0u; candidate < batch; ++candidate) {
            sums[candidate] += value * inputs[candidate * columns + column];
        }
    }
    threadgroup float partials[8][32];
    for (uint candidate = 0u; candidate < batch; ++candidate) {
        float total = simd_sum(sums[candidate]);
        if (simd_lane == 0u) {
            partials[candidate][simdgroup_id] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_id.x == 0u) {
        uint simdgroups = (threads_per_group.x + 31u) / 32u;
        for (uint candidate = 0u; candidate < batch; ++candidate) {
            float total = 0.0f;
            for (uint index = 0u; index < simdgroups; ++index) {
                total += partials[candidate][index];
            }
            outputs[candidate * rows + row] = total;
        }
    }
}

inline float si_q5_k_value(device const uchar* weights, uint block_base, uint local) {
    uint group = local / 64u;
    uint within_group = local & 63u;
    uint quant_byte = within_group & 31u;
    ushort d_bits = ushort(weights[block_base]) |
        (ushort(weights[block_base + 1u]) << 8);
    ushort min_bits = ushort(weights[block_base + 2u]) |
        (ushort(weights[block_base + 3u]) << 8);
    float d = float(as_type<half>(d_bits));
    float min = float(as_type<half>(min_bits));
    uint scale_index = group * 2u + (within_group >= 32u ? 1u : 0u);
    uint scale;
    uint minimum;
    if (scale_index < 4u) {
        scale = uint(weights[block_base + 4u + scale_index] & 0x3fu);
        minimum = uint(weights[block_base + 8u + scale_index] & 0x3fu);
    } else {
        uint packed_scale = block_base + 8u + scale_index;
        scale = uint(weights[packed_scale] & 0x0fu) |
            (uint(weights[block_base + scale_index]) >> 6u << 4u);
        minimum = uint(weights[packed_scale] >> 4u) |
            (uint(weights[block_base + 4u + scale_index]) >> 6u << 4u);
    }
    uchar packed = weights[block_base + 48u + group * 32u + quant_byte];
    uint high_mask = within_group >= 32u ? (2u << (group * 2u)) : (1u << (group * 2u));
    uint quant = within_group >= 32u ? uint(packed >> 4u) : uint(packed & 0x0fu);
    if ((uint(weights[block_base + 16u + quant_byte]) & high_mask) != 0u) {
        quant += 16u;
    }
    return d * float(scale) * float(quant) - min * float(minimum);
}

kernel void si_q5_k_matmul_many(
    device const uchar* weights [[buffer(0)]],
    device const float* inputs [[buffer(1)]],
    device float* outputs [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    uint row = threadgroup_id.x;
    if (row >= rows || batch == 0u || batch > 8u) {
        return;
    }
    float sums[8];
    for (uint candidate = 0u; candidate < 8u; ++candidate) sums[candidate] = 0.0f;
    uint blocks_per_row = columns / 256u;
    for (uint column = thread_id.x; column < columns; column += threads_per_group.x) {
        float value = si_q5_k_value(weights, (row * blocks_per_row + column / 256u) * 176u, column & 255u);
        for (uint candidate = 0u; candidate < batch; ++candidate) {
            sums[candidate] += value * inputs[candidate * columns + column];
        }
    }
    threadgroup float partials[8][32];
    for (uint candidate = 0u; candidate < batch; ++candidate) {
        float total = simd_sum(sums[candidate]);
        if (simd_lane == 0u) partials[candidate][simdgroup_id] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_id.x == 0u) {
        uint simdgroups = (threads_per_group.x + 31u) / 32u;
        for (uint candidate = 0u; candidate < batch; ++candidate) {
            float total = 0.0f;
            for (uint index = 0u; index < simdgroups; ++index) total += partials[candidate][index];
            outputs[candidate * rows + row] = total;
        }
    }
}

inline float si_q6_k_value(device const uchar* weights, uint block_base, uint local) {
    uint block_half = local / 128u;
    uint within_half = local & 127u;
    uint quartet = within_half / 32u;
    uint index = within_half & 31u;
    uint ql_base = block_base + block_half * 64u;
    uint qh_base = block_base + 128u + block_half * 32u;
    uchar packed = weights[ql_base + (quartet & 1u) * 32u + index];
    uint nibble = quartet >= 2u ? uint(packed >> 4u) : uint(packed & 0x0fu);
    uint high_bits = (uint(weights[qh_base + index]) >> (quartet * 2u)) & 0x03u;
    int quant = int(nibble | (high_bits << 4u)) - 32;
    uint scale_index = (index / 16u) + quartet * 2u;
    int scale = int(weights[block_base + 192u + block_half * 8u + scale_index]);
    if (scale > 127) scale -= 256;
    ushort d_bits = ushort(weights[block_base + 208u]) |
        (ushort(weights[block_base + 209u]) << 8);
    return float(as_type<half>(d_bits)) * float(scale) * float(quant);
}

kernel void si_q6_k_matmul_many(
    device const uchar* weights [[buffer(0)]],
    device const float* inputs [[buffer(1)]],
    device float* outputs [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    uint row = threadgroup_id.x;
    if (row >= rows || batch == 0u || batch > 8u) {
        return;
    }
    float sums[8];
    for (uint candidate = 0u; candidate < 8u; ++candidate) sums[candidate] = 0.0f;
    uint blocks_per_row = columns / 256u;
    for (uint column = thread_id.x; column < columns; column += threads_per_group.x) {
        float value = si_q6_k_value(weights, (row * blocks_per_row + column / 256u) * 210u, column & 255u);
        for (uint candidate = 0u; candidate < batch; ++candidate) {
            sums[candidate] += value * inputs[candidate * columns + column];
        }
    }
    threadgroup float partials[8][32];
    for (uint candidate = 0u; candidate < batch; ++candidate) {
        float total = simd_sum(sums[candidate]);
        if (simd_lane == 0u) partials[candidate][simdgroup_id] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_id.x == 0u) {
        uint simdgroups = (threads_per_group.x + 31u) / 32u;
        for (uint candidate = 0u; candidate < batch; ++candidate) {
            float total = 0.0f;
            for (uint index = 0u; index < simdgroups; ++index) total += partials[candidate][index];
            outputs[candidate * rows + row] = total;
        }
    }
}

// Exact row-aligned invariant-bit-packed BF16 matvec. Each tile stores a
// four-byte header (invariant mask and constants) followed by LSB-first
// variable-bit payloads. Values are reconstructed in registers immediately
// before multiplication, so no decompressed BF16 weight buffer is allocated.
kernel void si_bf16_bitpack_matvec(
    device const uchar* packed [[buffer(0)]],
    device const uint* offsets [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    constant uint& tile_rows [[buffer(6)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    uint row = threadgroup_id.x;
    if (row >= rows || tile_rows == 0u) {
        return;
    }
    uint tile = row / tile_rows;
    uint local_row = row - tile * tile_rows;
    uint tile_offset = offsets[tile];
    ushort invariant_mask = ushort(packed[tile_offset])
        | (ushort(packed[tile_offset + 1u]) << 8);
    ushort constants = ushort(packed[tile_offset + 2u])
        | (ushort(packed[tile_offset + 3u]) << 8);
    uint variable_bits = 16u - popcount(uint(invariant_mask));
    uint payload_offset = tile_offset + 4u;
    uint tile_end = offsets[tile + 1u];
    float sum = 0.0f;
    for (uint column = thread_id.x; column < columns; column += threads_per_group.x) {
        uint value_index = local_row * columns + column;
        uint cursor = value_index * variable_bits;
        uint byte_index = cursor >> 3u;
        uint payload_index = payload_offset + byte_index;
        uint packed_value = 0u;
        if (payload_index < tile_end) {
            packed_value = uint(packed[payload_index]);
        }
        if (payload_index + 1u < tile_end) {
            packed_value |= uint(packed[payload_index + 1u]) << 8u;
        }
        if (payload_index + 2u < tile_end) {
            packed_value |= uint(packed[payload_index + 2u]) << 16u;
        }
        uint variable_value = (packed_value >> (cursor & 7u))
            & ((1u << variable_bits) - 1u);
        ushort value = constants;
        uint variable_index = 0u;
        for (uint bit = 0u; bit < 16u; ++bit) {
            ushort mask = ushort(1u << bit);
            if ((invariant_mask & mask) != 0u) {
                continue;
            }
            if ((variable_value & (1u << variable_index)) != 0u) {
                value |= mask;
            }
            variable_index += 1u;
        }
        uint bits = uint(value) << 16;
        sum += as_type<float>(bits) * input[column];
    }
    float simd_total = simd_sum(sum);
    threadgroup float partials[32];
    if (simd_lane == 0u) {
        partials[simdgroup_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_id.x == 0u) {
        float total = 0.0f;
        uint simdgroups = (threads_per_group.x + 31u) / 32u;
        for (uint index = 0u; index < simdgroups; ++index) {
            total += partials[index];
        }
        output[row] = total;
    }
}

// Exact batched matrix-vector primitive. Inputs are batch-major [K, columns]
// and outputs are batch-major [K, rows]. Each SIMD group owns one output row,
// so every BF16 weight is loaded once and reused across up to eight candidates.
kernel void si_bf16_matmul_many(
    device const ushort* weights [[buffer(0)]],
    device const float* inputs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& columns [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& batch [[buffer(5)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    uint row = threadgroup_id.x;
    if (row >= rows || simdgroup_id != 0u) {
        return;
    }
    float sums[8];
    for (uint candidate = 0; candidate < 8u; ++candidate) {
        sums[candidate] = 0.0f;
    }
    for (uint column = simd_lane; column < columns; column += 32u) {
        uint bits = uint(weights[row * columns + column]) << 16;
        float weight = as_type<float>(bits);
        for (uint candidate = 0; candidate < 8u; ++candidate) {
            if (candidate < batch) {
                sums[candidate] += weight * inputs[candidate * columns + column];
            }
        }
    }
    for (uint candidate = 0; candidate < 8u; ++candidate) {
        if (candidate < batch) {
            float total = simd_sum(sums[candidate]);
            if (simd_lane == 0) {
                output[candidate * rows + row] = total;
            }
        }
    }
}

// Exact batched QKV projection. One SIMD group owns one output row across
// the concatenated Q/K/V matrices and reuses that row's BF16 weights for up to
// eight candidate hidden states.
kernel void si_bf16_fused_qkv_many(
    device const ushort* q_weights [[buffer(0)]],
    device const ushort* k_weights [[buffer(1)]],
    device const ushort* v_weights [[buffer(2)]],
    device const float* inputs [[buffer(3)]],
    device float* q_output [[buffer(4)]],
    device float* k_output [[buffer(5)]],
    device float* v_output [[buffer(6)]],
    constant uint& columns [[buffer(7)]],
    constant uint& q_rows [[buffer(8)]],
    constant uint& k_rows [[buffer(9)]],
    constant uint& v_rows [[buffer(10)]],
    constant uint& batch [[buffer(11)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
    uint row = threadgroup_id.x;
    uint total_rows = q_rows + k_rows + v_rows;
    if (row >= total_rows) {
        return;
    }
    device const ushort* weights;
    device float* output;
    uint local_row;
    uint output_rows;
    if (row < q_rows) {
        weights = q_weights;
        output = q_output;
        local_row = row;
        output_rows = q_rows;
    } else if (row < q_rows + k_rows) {
        weights = k_weights;
        output = k_output;
        local_row = row - q_rows;
        output_rows = k_rows;
    } else {
        weights = v_weights;
        output = v_output;
        local_row = row - q_rows - k_rows;
        output_rows = v_rows;
    }
    float sums[8];
    for (uint candidate = 0; candidate < 8u; ++candidate) {
        sums[candidate] = 0.0f;
    }
    device const ushort* row_weights = weights + local_row * columns;
    for (uint column = simd_lane; column < columns; column += 32u) {
        uint bits = uint(row_weights[column]) << 16;
        float weight = as_type<float>(bits);
        for (uint candidate = 0; candidate < 8u; ++candidate) {
            if (candidate < batch) {
                sums[candidate] += weight * inputs[candidate * columns + column];
            }
        }
    }
    for (uint candidate = 0; candidate < 8u; ++candidate) {
        if (candidate < batch) {
            float total = simd_sum(sums[candidate]);
            if (simd_lane == 0) {
                output[candidate * output_rows + local_row] = total;
            }
        }
    }
}

// Exact batched gate/up projection. The layout matches qkv_many: inputs and
// each output are batch-major so the CPU elementwise path needs no transpose.
kernel void si_bf16_fused_gate_up_many(
    device const ushort* gate_weights [[buffer(0)]],
    device const ushort* up_weights [[buffer(1)]],
    device const float* inputs [[buffer(2)]],
    device float* gate_output [[buffer(3)]],
    device float* up_output [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    constant uint& gate_rows [[buffer(6)]],
    constant uint& up_rows [[buffer(7)]],
    constant uint& batch [[buffer(8)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]) {
    uint row = threadgroup_id.x;
    uint total_rows = gate_rows + up_rows;
    if (row >= total_rows) {
        return;
    }
    device const ushort* weights;
    device float* output;
    uint local_row;
    uint output_rows;
    if (row < gate_rows) {
        weights = gate_weights;
        output = gate_output;
        local_row = row;
        output_rows = gate_rows;
    } else {
        weights = up_weights;
        output = up_output;
        local_row = row - gate_rows;
        output_rows = up_rows;
    }
    float sums[8];
    for (uint candidate = 0; candidate < 8u; ++candidate) {
        sums[candidate] = 0.0f;
    }
    device const ushort* row_weights = weights + local_row * columns;
    for (uint column = simd_lane; column < columns; column += 32u) {
        uint bits = uint(row_weights[column]) << 16;
        float weight = as_type<float>(bits);
        for (uint candidate = 0; candidate < 8u; ++candidate) {
            if (candidate < batch) {
                sums[candidate] += weight * inputs[candidate * columns + column];
            }
        }
    }
    for (uint candidate = 0; candidate < 8u; ++candidate) {
        if (candidate < batch) {
            float total = simd_sum(sums[candidate]);
            if (simd_lane == 0) {
                output[candidate * output_rows + local_row] = total;
            }
        }
    }
}

kernel void si_bf16_fused_qkv(
    device const ushort* q_weights [[buffer(0)]],
    device const ushort* k_weights [[buffer(1)]],
    device const ushort* v_weights [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device float* q_output [[buffer(4)]],
    device float* k_output [[buffer(5)]],
    device float* v_output [[buffer(6)]],
    constant uint& columns [[buffer(7)]],
    constant uint& q_rows [[buffer(8)]],
    constant uint& k_rows [[buffer(9)]],
    constant uint& v_rows [[buffer(10)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    uint row = threadgroup_id.x;
    uint total_rows = q_rows + k_rows + v_rows;
    if (row >= total_rows) {
        return;
    }
    device const ushort* weights;
    device float* output;
    uint local_row;
    if (row < q_rows) {
        weights = q_weights;
        output = q_output;
        local_row = row;
    } else if (row < q_rows + k_rows) {
        weights = k_weights;
        output = k_output;
        local_row = row - q_rows;
    } else {
        weights = v_weights;
        output = v_output;
        local_row = row - q_rows - k_rows;
    }
    float sum = 0.0f;
    if ((columns & 3u) == 0u) {
        uint vector_columns = columns / 4u;
        device const ushort4* vector_weights = reinterpret_cast<device const ushort4*>(weights + local_row * columns);
        device const float4* vector_input = reinterpret_cast<device const float4*>(input);
        for (uint index = thread_id.x; index < vector_columns; index += threads_per_group.x) {
            uint4 bits = uint4(vector_weights[index]) << 16;
            sum += dot(as_type<float4>(bits), vector_input[index]);
        }
    } else {
        device const ushort* row_weights = weights + local_row * columns;
        for (uint column = thread_id.x; column < columns; column += threads_per_group.x) {
            uint bits = uint(row_weights[column]) << 16;
            sum += as_type<float>(bits) * input[column];
        }
    }
    float simd_total = simd_sum(sum);
    threadgroup float partials[32];
    if (simd_lane == 0) {
        partials[simdgroup_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_id.x == 0) {
        float total = 0.0f;
        uint simdgroups = (threads_per_group.x + 31u) / 32u;
        for (uint index = 0; index < simdgroups; ++index) {
            total += partials[index];
        }
        output[local_row] = total;
    }
}

kernel void si_bf16_fused_gate_up(
    device const ushort* gate_weights [[buffer(0)]],
    device const ushort* up_weights [[buffer(1)]],
    device const float* input [[buffer(2)]],
    device float* gate_output [[buffer(3)]],
    device float* up_output [[buffer(4)]],
    constant uint& columns [[buffer(5)]],
    constant uint& gate_rows [[buffer(6)]],
    constant uint& up_rows [[buffer(7)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint3 thread_id [[thread_position_in_threadgroup]],
    uint3 threads_per_group [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    uint row = threadgroup_id.x;
    uint total_rows = gate_rows + up_rows;
    if (row >= total_rows) {
        return;
    }
    device const ushort* weights;
    device float* output;
    uint local_row;
    if (row < gate_rows) {
        weights = gate_weights;
        output = gate_output;
        local_row = row;
    } else {
        weights = up_weights;
        output = up_output;
        local_row = row - gate_rows;
    }
    float sum = 0.0f;
    if ((columns & 3u) == 0u) {
        uint vector_columns = columns / 4u;
        device const ushort4* vector_weights = reinterpret_cast<device const ushort4*>(weights + local_row * columns);
        device const float4* vector_input = reinterpret_cast<device const float4*>(input);
        for (uint index = thread_id.x; index < vector_columns; index += threads_per_group.x) {
            uint4 bits = uint4(vector_weights[index]) << 16;
            sum += dot(as_type<float4>(bits), vector_input[index]);
        }
    } else {
        device const ushort* row_weights = weights + local_row * columns;
        for (uint column = thread_id.x; column < columns; column += threads_per_group.x) {
            uint bits = uint(row_weights[column]) << 16;
            sum += as_type<float>(bits) * input[column];
        }
    }
    float simd_total = simd_sum(sum);
    threadgroup float partials[32];
    if (simd_lane == 0) {
        partials[simdgroup_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_id.x == 0) {
        float total = 0.0f;
        uint simdgroups = (threads_per_group.x + 31u) / 32u;
        for (uint index = 0; index < simdgroups; ++index) {
            total += partials[index];
        }
        output[local_row] = total;
    }
}

kernel void si_bf16_embedding_row(
    device const ushort* weights [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& length [[buffer(2)]],
    uint3 thread_id [[thread_position_in_grid]]) {
    uint index = thread_id.x;
    if (index >= length) {
        return;
    }
    uint bits = uint(weights[index]) << 16;
    output[index] = as_type<float>(bits);
}

kernel void si_rope(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& heads [[buffer(2)]],
    constant uint& head_dim [[buffer(3)]],
    constant uint& position [[buffer(4)]],
    constant float& theta [[buffer(5)]],
    uint3 thread_id [[thread_position_in_grid]]) {
    uint pairs_per_head = head_dim / 2;
    uint pair = thread_id.x;
    if (pair >= heads * pairs_per_head) {
        return;
    }
    uint head = pair / pairs_per_head;
    uint pair_in_head = pair % pairs_per_head;
    float exponent = (2.0f * static_cast<float>(pair_in_head)) / static_cast<float>(head_dim);
    float angle = static_cast<float>(position) * pow(theta, -exponent);
    float sine = sin(angle);
    float cosine = cos(angle);
    uint offset = head * head_dim + pair_in_head * 2;
    float first = input[offset];
    float second = input[offset + 1];
    output[offset] = first * cosine - second * sine;
    output[offset + 1] = first * sine + second * cosine;
}

kernel void si_attention_decode(
    device const float* query [[buffer(0)]],
    device const float* key_cache [[buffer(1)]],
    device const float* value_cache [[buffer(2)]],
    device const float* new_keys [[buffer(3)]],
    device const float* new_values [[buffer(4)]],
    device float* output [[buffer(5)]],
    constant uint& query_heads [[buffer(6)]],
    constant uint& key_value_heads [[buffer(7)]],
    constant uint& head_dim [[buffer(8)]],
    constant uint& cached_tokens [[buffer(9)]],
    constant uint& cache_capacity_tokens [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint3 thread_id [[thread_position_in_grid]]) {
    uint query_head = thread_id.x;
    uint group_size = query_heads / key_value_heads;
    uint key_value_head = query_head / group_size;
    uint total_tokens = cached_tokens + 1;
    float max_score = -INFINITY;
    for (uint token = 0; token < total_tokens; ++token) {
        float score = 0.0f;
        for (uint dimension = 0; dimension < head_dim; ++dimension) {
            uint query_index = query_head * head_dim + dimension;
            uint key_index;
            float key_value;
            if (token < cached_tokens) {
                key_index = (key_value_head * cache_capacity_tokens + token) * head_dim + dimension;
                key_value = key_cache[key_index];
            } else {
                key_index = key_value_head * head_dim + dimension;
                key_value = new_keys[key_index];
            }
            score += query[query_index] * key_value;
        }
        max_score = max(max_score, score * scale);
    }
    float normalizer = 0.0f;
    for (uint token = 0; token < total_tokens; ++token) {
        float score = 0.0f;
        for (uint dimension = 0; dimension < head_dim; ++dimension) {
            uint query_index = query_head * head_dim + dimension;
            uint key_index;
            float key_value;
            if (token < cached_tokens) {
                key_index = (key_value_head * cache_capacity_tokens + token) * head_dim + dimension;
                key_value = key_cache[key_index];
            } else {
                key_index = key_value_head * head_dim + dimension;
                key_value = new_keys[key_index];
            }
            score += query[query_index] * key_value;
        }
        normalizer += exp(score * scale - max_score);
    }
    for (uint dimension = 0; dimension < head_dim; ++dimension) {
        float result = 0.0f;
        for (uint token = 0; token < total_tokens; ++token) {
            float score = 0.0f;
            for (uint key_dimension = 0; key_dimension < head_dim; ++key_dimension) {
                uint query_index = query_head * head_dim + key_dimension;
                uint key_index;
                float key_value;
                if (token < cached_tokens) {
                    key_index = (key_value_head * cache_capacity_tokens + token) * head_dim + key_dimension;
                    key_value = key_cache[key_index];
                } else {
                    key_index = key_value_head * head_dim + key_dimension;
                    key_value = new_keys[key_index];
                }
                score += query[query_index] * key_value;
            }
            float weight = exp(score * scale - max_score) / normalizer;
            uint value_index;
            float value;
            if (token < cached_tokens) {
                value_index = (key_value_head * cache_capacity_tokens + token) * head_dim + dimension;
                value = value_cache[value_index];
            } else {
                value_index = key_value_head * head_dim + dimension;
                value = new_values[value_index];
            }
            result += weight * value;
        }
        output[query_head * head_dim + dimension] = result;
    }
}
"#;

#[cfg(target_os = "macos")]
pub fn probe() -> Result<MetalDeviceInfo, String> {
    let device = metal::Device::system_default().ok_or("no Metal device found")?;
    Ok(MetalDeviceInfo {
        name: device.name().to_owned(),
        registry_id: device.registry_id(),
        recommended_max_working_set_bytes: device.recommended_max_working_set_size(),
        max_buffer_bytes: device.max_buffer_length(),
        has_unified_memory: device.has_unified_memory(),
    })
}

#[cfg(target_os = "macos")]
pub struct MetalContext {
    device: metal::Device,
    queue: metal::CommandQueue,
    upload_queue: metal::CommandQueue,
    command_stats: Arc<CommandStats>,
    peak_allocated_bytes: AtomicU64,
    peak_active_weight_bytes: AtomicU64,
    resident_weight_bytes: AtomicU64,
    persistent_weight_bytes: AtomicU64,
    peak_kv_bytes: AtomicU64,
    peak_scratch_bytes: AtomicU64,
    rms_norm_pipeline: metal::ComputePipelineState,
    rms_norm_bf16_pipeline: metal::ComputePipelineState,
    bf16_matvec_pipeline: metal::ComputePipelineState,
    f32_matvec_pipeline: metal::ComputePipelineState,
    q4_k_matvec_pipeline: metal::ComputePipelineState,
    q4_k_matvec_rows4_pipeline: metal::ComputePipelineState,
    q4_k_matvec_rows4x128_pipeline: metal::ComputePipelineState,
    q4_k_matmul_many_pipeline: metal::ComputePipelineState,
    q4_k_fused_gate_up_pipeline: metal::ComputePipelineState,
    q4_k_embedding_pipeline: metal::ComputePipelineState,
    q5_k_matvec_pipeline: metal::ComputePipelineState,
    q5_k_matvec_rows4_pipeline: metal::ComputePipelineState,
    q5_k_matmul_many_pipeline: metal::ComputePipelineState,
    q6_k_matvec_pipeline: metal::ComputePipelineState,
    q6_k_matvec_rows4_pipeline: metal::ComputePipelineState,
    q6_k_matvec_rows4x128_pipeline: metal::ComputePipelineState,
    q6_k_matmul_many_pipeline: metal::ComputePipelineState,
    gated_delta_pipeline: metal::ComputePipelineState,
    gated_delta_parallel_pipeline: metal::ComputePipelineState,
    causal_conv_pipeline: metal::ComputePipelineState,
    rms_norm_gated_pipeline: metal::ComputePipelineState,
    rms_norm_heads_pipeline: metal::ComputePipelineState,
    bf16_bitpack_pipeline: Option<metal::ComputePipelineState>,
    batched_matmul_pipeline: Option<metal::ComputePipelineState>,
    batched_fused_qkv_pipeline: Option<metal::ComputePipelineState>,
    batched_fused_gate_up_pipeline: Option<metal::ComputePipelineState>,
    bf16_fused_qkv_pipeline: metal::ComputePipelineState,
    bf16_fused_gate_up_pipeline: metal::ComputePipelineState,
    bf16_embedding_pipeline: metal::ComputePipelineState,
    rope_pipeline: metal::ComputePipelineState,
    attention_decode_pipeline: metal::ComputePipelineState,
}

#[cfg(target_os = "macos")]
pub struct Bf16Weight {
    buffer: metal::Buffer,
    offset: u64,
    bytes: u64,
    persistent: bool,
    mapped: bool,
}

/// A quantized matrix retained in private Metal storage. The GGUF bytes are
/// copied once during runtime construction; inference then reuses this buffer
/// instead of rebinding the file-backed mapping for every token.
#[cfg(target_os = "macos")]
#[derive(Clone)]
pub struct QuantWeight {
    pub(crate) buffer: metal::Buffer,
    pub(crate) offset: u64,
    pub(crate) bytes: u64,
    pub(crate) ggml_type: u32,
    pub(crate) mapped: bool,
}

/// A packed decoder layer uploaded to transient private Metal storage. The
/// backing buffer is shared by all tensor views in the layer, so promotion
/// requires one blit instead of one upload per projection.
#[cfg(target_os = "macos")]
pub struct PrivateStagedQwenLayer {
    buffer: metal::Buffer,
    ranges: std::collections::BTreeMap<String, (usize, usize)>,
}

#[cfg(target_os = "macos")]
impl PrivateStagedQwenLayer {
    pub fn quant_weight(&self, name: &str, ggml_type: u32) -> Option<QuantWeight> {
        let (start, end) = self.ranges.get(name).copied()?;
        Some(QuantWeight {
            buffer: self.buffer.clone(),
            offset: start as u64,
            bytes: (end - start) as u64,
            ggml_type,
            mapped: false,
        })
    }
}

#[cfg(target_os = "macos")]
impl QuantWeight {
    pub fn byte_len(&self) -> u64 {
        self.bytes
    }

    pub fn ggml_type(&self) -> u32 {
        self.ggml_type
    }
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy)]
enum BatchedFusedProjectionKind {
    Qkv,
    GateUp,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Default)]
struct CommandStats {
    submitted: AtomicU64,
    async_submitted: AtomicU64,
    waited: AtomicU64,
    wait_nanos: AtomicU64,
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CommandStatsSnapshot {
    pub submitted: u64,
    pub async_submitted: u64,
    pub waited: u64,
    pub wait_nanos: u64,
}

#[cfg(target_os = "macos")]
impl CommandStatsSnapshot {
    pub fn delta_since(self, before: Self) -> Self {
        Self {
            submitted: self.submitted.saturating_sub(before.submitted),
            async_submitted: self.async_submitted.saturating_sub(before.async_submitted),
            waited: self.waited.saturating_sub(before.waited),
            wait_nanos: self.wait_nanos.saturating_sub(before.wait_nanos),
        }
    }
}

#[cfg(target_os = "macos")]
impl CommandStats {
    fn record_submission(&self) {
        self.submitted.fetch_add(1, Ordering::Relaxed);
    }

    fn record_async_submission(&self) {
        self.record_submission();
        self.async_submitted.fetch_add(1, Ordering::Relaxed);
    }

    fn record_wait(&self, duration: Duration) {
        self.waited.fetch_add(1, Ordering::Relaxed);
        self.wait_nanos.fetch_add(
            duration.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }

    fn snapshot(&self) -> CommandStatsSnapshot {
        CommandStatsSnapshot {
            submitted: self.submitted.load(Ordering::Relaxed),
            async_submitted: self.async_submitted.load(Ordering::Relaxed),
            waited: self.waited.load(Ordering::Relaxed),
            wait_nanos: self.wait_nanos.load(Ordering::Relaxed),
        }
    }
}

#[cfg(target_os = "macos")]
pub struct PendingMatvec {
    command_buffer: metal::CommandBuffer,
    _input: metal::Buffer,
    _weights: Vec<metal::Buffer>,
    _keepalive: Vec<Vec<u8>>,
    outputs: Vec<(metal::Buffer, usize)>,
    stats: Arc<CommandStats>,
}

#[cfg(target_os = "macos")]
impl PendingMatvec {
    pub fn status(&self) -> metal::MTLCommandBufferStatus {
        self.command_buffer.status()
    }

    pub fn wait(self) -> Result<Vec<Vec<f32>>, String> {
        let started = Instant::now();
        self.command_buffer.wait_until_completed();
        self.stats.record_wait(started.elapsed());
        if self.command_buffer.status() != metal::MTLCommandBufferStatus::Completed {
            return Err(format!(
                "Metal command failed: {:?}",
                self.command_buffer.status()
            ));
        }
        Ok(self
            .outputs
            .iter()
            .map(|(output, rows)| unsafe {
                // SAFETY: command buffer completed and output contains rows f32s.
                std::slice::from_raw_parts(output.contents() as *const f32, *rows).to_vec()
            })
            .collect())
    }

    fn with_keepalive(mut self, keepalive: Vec<Vec<u8>>) -> Self {
        self._keepalive = keepalive;
        self
    }
}

#[cfg(target_os = "macos")]
impl Bf16Weight {
    pub fn byte_len(&self) -> u64 {
        self.bytes
    }
}

#[cfg(target_os = "macos")]
fn validate_matvec_tensor_batch(
    tensors: &[&TensorView<'_>],
    input_len: usize,
) -> Result<Vec<(usize, usize)>, String> {
    if tensors.is_empty() {
        return Err("batched BF16 tensor list must be non-empty".into());
    }
    let mut shapes = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        if tensor.info.dtype != "BF16" || tensor.info.shape.len() != 2 {
            return Err(format!(
                "tensor {} must be a rank-2 BF16 matrix",
                tensor.info.name
            ));
        }
        let rows = tensor.info.shape[0];
        let columns = tensor.info.shape[1];
        let expected_bytes = rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or("BF16 matrix dimensions overflow")?;
        if rows == 0 || columns == 0 || tensor.bytes.len() != expected_bytes {
            return Err(format!(
                "tensor {} has invalid matrix dimensions or byte length",
                tensor.info.name
            ));
        }
        if input_len != columns {
            return Err(format!(
                "tensor {} input length {} does not match columns {}",
                tensor.info.name, input_len, columns
            ));
        }
        shapes.push((rows, columns));
    }
    Ok(shapes)
}

#[cfg(target_os = "macos")]
fn validate_matvec_tensor_many(
    tensors: &[&TensorView<'_>],
    batch: usize,
    input_len: usize,
) -> Result<Vec<(usize, usize)>, String> {
    if !(1..=8).contains(&batch) {
        return Err("batched BF16 tensor list supports between one and eight candidates".into());
    }
    if input_len == 0 || !input_len.is_multiple_of(batch) {
        return Err("batched BF16 input length must be a non-zero batch multiple".into());
    }
    validate_matvec_tensor_batch(tensors, input_len / batch)
}

#[cfg(target_os = "macos")]
fn validate_matmul_many_shape(
    rows: usize,
    columns: usize,
    batch: usize,
    weight_bytes: usize,
    input_len: usize,
) -> Result<(), String> {
    let expected_weight_bytes = rows
        .checked_mul(columns)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or("batched BF16 matrix dimensions overflow")?;
    let expected_input_len = batch
        .checked_mul(columns)
        .ok_or("batched BF16 input dimensions overflow")?;
    if rows == 0
        || columns == 0
        || batch == 0
        || batch > 8
        || weight_bytes != expected_weight_bytes
        || input_len != expected_input_len
    {
        return Err("batched BF16 matrix dimensions or byte length are invalid".into());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn validate_fused_projection_shapes(
    shapes: &[(usize, usize)],
    input_len: usize,
) -> Result<(), String> {
    if !(2..=3).contains(&shapes.len()) {
        return Err("fused projections require two or three matrices".into());
    }
    for (rows, columns) in shapes {
        let _ = rows
            .checked_mul(*columns)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or("fused projection dimensions overflow")?;
        if *rows == 0 || *columns == 0 || *columns != input_len {
            return Err("fused projection dimensions or input length are invalid".into());
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn validate_fused_projection_many_shapes(
    shapes: &[(usize, usize)],
    batch: usize,
    input_len: usize,
) -> Result<(), String> {
    validate_fused_projection_shapes(shapes, input_len)
        .map_err(|error| format!("invalid batched fused projection shapes: {error}"))?;
    if !(1..=8).contains(&batch) {
        return Err("batched fused projections support between one and eight candidates".into());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn validate_bf16_weight_bytes(weight_bytes: &[u8]) -> Result<(), String> {
    if weight_bytes.is_empty() || !weight_bytes.len().is_multiple_of(2) {
        return Err("BF16 weight bytes must be non-empty and 2-byte aligned".into());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn aligned_weight_range(
    backing_ptr: usize,
    tensor_ptr: usize,
    tensor_len: usize,
    page_size: usize,
) -> Result<(usize, usize, usize), String> {
    if page_size == 0 || !page_size.is_power_of_two() {
        return Err("Metal backing page size must be a non-zero power of two".into());
    }
    if tensor_ptr < backing_ptr {
        return Err("tensor bytes are outside their backing mapping".into());
    }
    let tensor_end = tensor_ptr
        .checked_add(tensor_len)
        .ok_or("tensor byte range overflows")?;
    let aligned_base = tensor_ptr & !(page_size - 1);
    let end_offset = tensor_end
        .checked_sub(aligned_base)
        .ok_or("tensor range is before aligned backing base")?;
    let aligned_length = end_offset
        .checked_add(page_size - 1)
        .map(|value| value & !(page_size - 1))
        .ok_or("aligned tensor range overflows")?;
    if aligned_length == 0 {
        return Err("aligned tensor range is empty".into());
    }
    Ok((aligned_base, tensor_ptr - aligned_base, aligned_length))
}

#[cfg(target_os = "macos")]
fn qk_rows4_enabled(setting: Option<&str>) -> bool {
    setting == Some("1")
}

#[cfg(target_os = "macos")]
fn qk_rows4x128_enabled() -> bool {
    std::env::var("SI_QK_ROWS4X128").ok().as_deref() == Some("1")
}

#[cfg(target_os = "macos")]
fn qk_rows4x128_for_type(ggml_type: u32) -> bool {
    qk_rows4x128_enabled()
        && (ggml_type == crate::quant::GGML_TYPE_Q4_K
            || (ggml_type == crate::quant::GGML_TYPE_Q6_K
                && std::env::var("SI_QK_ROWS4X128_Q6").ok().as_deref() == Some("1")))
}

#[cfg(target_os = "macos")]
impl MetalContext {
    pub fn new() -> Result<Self, String> {
        let device = metal::Device::system_default().ok_or("no Metal device found")?;
        let options = metal::CompileOptions::new();
        let library = device.new_library_with_source(RMS_NORM_SHADER, &options)?;
        let function = library.get_function("si_rms_norm", None)?;
        let rms_norm_pipeline = device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_rms_norm_bf16_heads", None)?;
        let rms_norm_bf16_pipeline = device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_bf16_matvec", None)?;
        let bf16_matvec_pipeline = device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_f32_matvec", None)?;
        let f32_matvec_pipeline = device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_q4_k_matvec", None)?;
        let q4_k_matvec_pipeline = device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_q4_k_matvec_rows4", None)?;
        let q4_k_matvec_rows4_pipeline =
            device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_q4_k_matvec_rows4x128", None)?;
        let q4_k_matvec_rows4x128_pipeline =
            device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_q4_k_matmul_many", None)?;
        let q4_k_matmul_many_pipeline =
            device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_q4_k_fused_gate_up", None)?;
        let q4_k_fused_gate_up_pipeline =
            device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_q4_k_embedding_row", None)?;
        let q4_k_embedding_pipeline = device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_q5_k_matvec", None)?;
        let q5_k_matvec_pipeline = device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_q5_k_matvec_rows4", None)?;
        let q5_k_matvec_rows4_pipeline =
            device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_q5_k_matmul_many", None)?;
        let q5_k_matmul_many_pipeline =
            device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_q6_k_matvec", None)?;
        let q6_k_matvec_pipeline = device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_q6_k_matvec_rows4", None)?;
        let q6_k_matvec_rows4_pipeline =
            device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_q6_k_matvec_rows4x128", None)?;
        let q6_k_matvec_rows4x128_pipeline =
            device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_q6_k_matmul_many", None)?;
        let q6_k_matmul_many_pipeline =
            device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_gated_delta_step", None)?;
        let gated_delta_pipeline = device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_gated_delta_step_parallel", None)?;
        let gated_delta_parallel_pipeline =
            device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_causal_conv1d_step", None)?;
        let causal_conv_pipeline = device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_rms_norm_gated", None)?;
        let rms_norm_gated_pipeline = device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_rms_norm_heads", None)?;
        let rms_norm_heads_pipeline = device.new_compute_pipeline_state_with_function(&function)?;
        let bf16_bitpack_pipeline = if cfg!(test) || std::env::var_os("SI_LOSSLESS_GPU").is_some() {
            let function = library.get_function("si_bf16_bitpack_matvec", None)?;
            Some(device.new_compute_pipeline_state_with_function(&function)?)
        } else {
            None
        };
        let batched_matmul_pipeline = if cfg!(test)
            || std::env::var_os("SI_VERIFY_MANY").is_some()
            || std::env::var_os("SI_LOOKAHEAD").is_some()
        {
            let function = library.get_function("si_bf16_matmul_many", None)?;
            Some(device.new_compute_pipeline_state_with_function(&function)?)
        } else {
            None
        };
        let (batched_fused_qkv_pipeline, batched_fused_gate_up_pipeline) = if cfg!(test)
            || std::env::var_os("SI_VERIFY_MANY").is_some()
            || std::env::var_os("SI_LOOKAHEAD").is_some()
        {
            let function = library.get_function("si_bf16_fused_qkv_many", None)?;
            let qkv = device.new_compute_pipeline_state_with_function(&function)?;
            let function = library.get_function("si_bf16_fused_gate_up_many", None)?;
            let gate_up = device.new_compute_pipeline_state_with_function(&function)?;
            (Some(qkv), Some(gate_up))
        } else {
            (None, None)
        };
        let function = library.get_function("si_bf16_fused_qkv", None)?;
        let bf16_fused_qkv_pipeline = device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_bf16_fused_gate_up", None)?;
        let bf16_fused_gate_up_pipeline =
            device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_bf16_embedding_row", None)?;
        let bf16_embedding_pipeline = device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_rope", None)?;
        let rope_pipeline = device.new_compute_pipeline_state_with_function(&function)?;
        let function = library.get_function("si_attention_decode", None)?;
        let attention_decode_pipeline =
            device.new_compute_pipeline_state_with_function(&function)?;
        Ok(Self {
            queue: device.new_command_queue(),
            upload_queue: device.new_command_queue(),
            device,
            command_stats: Arc::new(CommandStats::default()),
            peak_allocated_bytes: AtomicU64::new(0),
            peak_active_weight_bytes: AtomicU64::new(0),
            resident_weight_bytes: AtomicU64::new(0),
            persistent_weight_bytes: AtomicU64::new(0),
            peak_kv_bytes: AtomicU64::new(0),
            peak_scratch_bytes: AtomicU64::new(0),
            rms_norm_pipeline,
            rms_norm_bf16_pipeline,
            bf16_matvec_pipeline,
            f32_matvec_pipeline,
            q4_k_matvec_pipeline,
            q4_k_matvec_rows4_pipeline,
            q4_k_matvec_rows4x128_pipeline,
            q4_k_matmul_many_pipeline,
            q4_k_fused_gate_up_pipeline,
            q4_k_embedding_pipeline,
            q5_k_matvec_pipeline,
            q5_k_matvec_rows4_pipeline,
            q5_k_matmul_many_pipeline,
            q6_k_matvec_pipeline,
            q6_k_matvec_rows4_pipeline,
            q6_k_matvec_rows4x128_pipeline,
            q6_k_matmul_many_pipeline,
            gated_delta_pipeline,
            gated_delta_parallel_pipeline,
            causal_conv_pipeline,
            rms_norm_gated_pipeline,
            rms_norm_heads_pipeline,
            bf16_bitpack_pipeline,
            batched_matmul_pipeline,
            batched_fused_qkv_pipeline,
            batched_fused_gate_up_pipeline,
            bf16_fused_qkv_pipeline,
            bf16_fused_gate_up_pipeline,
            bf16_embedding_pipeline,
            rope_pipeline,
            attention_decode_pipeline,
        })
    }

    pub fn current_allocated_bytes(&self) -> u64 {
        self.device.current_allocated_size()
    }

    pub fn command_stats(&self) -> CommandStatsSnapshot {
        self.command_stats.snapshot()
    }

    fn commit_and_wait(
        &self,
        command_buffer: &metal::CommandBufferRef,
        operation: &str,
    ) -> Result<(), String> {
        command_buffer.commit();
        self.command_stats.record_submission();
        let started = Instant::now();
        command_buffer.wait_until_completed();
        self.command_stats.record_wait(started.elapsed());
        if command_buffer.status() != metal::MTLCommandBufferStatus::Completed {
            return Err(format!(
                "Metal {operation} failed: {:?}",
                command_buffer.status()
            ));
        }
        Ok(())
    }

    pub fn peak_allocated_bytes(&self) -> u64 {
        self.peak_allocated_bytes.load(Ordering::Relaxed)
    }

    pub fn peak_active_weight_bytes(&self) -> u64 {
        self.peak_active_weight_bytes.load(Ordering::Relaxed)
    }

    pub fn peak_kv_bytes(&self) -> u64 {
        self.peak_kv_bytes.load(Ordering::Relaxed)
    }

    pub fn peak_scratch_bytes(&self) -> u64 {
        self.peak_scratch_bytes.load(Ordering::Relaxed)
    }

    /// Start a new measured repetition while retaining persistent allocations.
    /// Resident weights remain part of the next peak; transient KV/scratch
    /// counters are measured afresh.
    pub fn reset_peaks(&self) {
        self.peak_allocated_bytes
            .store(self.current_allocated_bytes(), Ordering::Relaxed);
        self.peak_active_weight_bytes.store(
            self.resident_weight_bytes
                .load(Ordering::Relaxed)
                .max(self.persistent_weight_bytes.load(Ordering::Relaxed)),
            Ordering::Relaxed,
        );
        self.peak_kv_bytes.store(0, Ordering::Relaxed);
        self.peak_scratch_bytes.store(0, Ordering::Relaxed);
    }

    /// Sample the device allocator at an explicit runtime boundary. This is
    /// intentionally outside `record_profile`, because querying Metal's
    /// allocator for every small kernel would put telemetry on the hot path.
    pub fn sample_allocated(&self) {
        let current = self.current_allocated_bytes();
        self.peak_allocated_bytes
            .fetch_max(current, Ordering::Relaxed);
    }

    pub fn note_resident_weight_bytes(&self, bytes: u64) {
        self.resident_weight_bytes
            .fetch_max(bytes, Ordering::Relaxed);
        self.persistent_weight_bytes
            .fetch_max(bytes, Ordering::Relaxed);
        self.peak_active_weight_bytes
            .fetch_max(bytes, Ordering::Relaxed);
    }

    pub fn note_persistent_weight_bytes(&self, bytes: u64) {
        self.persistent_weight_bytes
            .fetch_add(bytes, Ordering::Relaxed);
        self.peak_active_weight_bytes.fetch_max(
            self.persistent_weight_bytes.load(Ordering::Relaxed),
            Ordering::Relaxed,
        );
    }

    fn record_profile(&self, active_weight_bytes: u64, kv_bytes: u64, scratch_bytes: u64) {
        self.peak_kv_bytes.fetch_max(kv_bytes, Ordering::Relaxed);
        self.peak_scratch_bytes
            .fetch_max(scratch_bytes, Ordering::Relaxed);
        let resident_weight_bytes = self.resident_weight_bytes.load(Ordering::Relaxed);
        let persistent_weight_bytes = self.persistent_weight_bytes.load(Ordering::Relaxed);
        let estimated_weight_bytes = if resident_weight_bytes > 0 {
            resident_weight_bytes
        } else {
            persistent_weight_bytes.saturating_add(active_weight_bytes)
        };
        self.peak_active_weight_bytes
            .fetch_max(estimated_weight_bytes, Ordering::Relaxed);
        let estimated_allocated_bytes = estimated_weight_bytes
            .saturating_add(kv_bytes)
            .saturating_add(scratch_bytes);
        self.peak_allocated_bytes
            .fetch_max(estimated_allocated_bytes, Ordering::Relaxed);
    }

    fn record_mapped_profile(&self, active_weight_bytes: u64, kv_bytes: u64, scratch_bytes: u64) {
        self.peak_kv_bytes.fetch_max(kv_bytes, Ordering::Relaxed);
        self.peak_scratch_bytes
            .fetch_max(scratch_bytes, Ordering::Relaxed);
        let resident_weight_bytes = self.resident_weight_bytes.load(Ordering::Relaxed);
        let persistent_weight_bytes = self.persistent_weight_bytes.load(Ordering::Relaxed);
        let estimated_logical_weight_bytes = if resident_weight_bytes > 0 {
            resident_weight_bytes
        } else {
            persistent_weight_bytes.saturating_add(active_weight_bytes)
        };
        self.peak_active_weight_bytes
            .fetch_max(estimated_logical_weight_bytes, Ordering::Relaxed);
        let allocated_weight_bytes = if resident_weight_bytes > 0 {
            resident_weight_bytes
        } else {
            persistent_weight_bytes
        };
        self.peak_allocated_bytes.fetch_max(
            allocated_weight_bytes
                .saturating_add(kv_bytes)
                .saturating_add(scratch_bytes),
            Ordering::Relaxed,
        );
    }

    /// Upload immutable BF16 bytes once and retain the GPU-visible buffer for
    /// subsequent matvecs. The caller owns the residency budget by retaining
    /// or dropping the returned handle.
    pub fn upload_bf16_weight(&self, weight_bytes: &[u8]) -> Result<Bf16Weight, String> {
        if weight_bytes.is_empty() || !weight_bytes.len().is_multiple_of(2) {
            return Err("BF16 weight bytes must be non-empty and 2-byte aligned".into());
        }
        let buffer = self.device.new_buffer_with_data(
            weight_bytes.as_ptr() as *const std::ffi::c_void,
            weight_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let bytes = weight_bytes.len() as u64;
        self.record_profile(bytes, 0, 0);
        Ok(Bf16Weight {
            buffer,
            offset: 0,
            bytes,
            persistent: false,
            mapped: false,
        })
    }

    /// Upload an immutable resident weight through a shared staging buffer and
    /// retain it in private GPU storage. Private storage avoids routing every
    /// matvec read through the CPU/GPU-coherent shared-memory path.
    pub fn upload_bf16_weight_private(&self, weight_bytes: &[u8]) -> Result<Bf16Weight, String> {
        if weight_bytes.is_empty() || !weight_bytes.len().is_multiple_of(2) {
            return Err("BF16 weight bytes must be non-empty and 2-byte aligned".into());
        }
        let staging = self.device.new_buffer_with_data(
            weight_bytes.as_ptr() as *const std::ffi::c_void,
            weight_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let buffer = self.device.new_buffer(
            weight_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModePrivate,
        );
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_blit_command_encoder();
        encoder.copy_from_buffer(&staging, 0, &buffer, 0, weight_bytes.len() as u64);
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "weight upload")?;
        let bytes = weight_bytes.len() as u64;
        self.record_profile(bytes, 0, 0);
        Ok(Bf16Weight {
            buffer,
            offset: 0,
            bytes,
            persistent: true,
            mapped: false,
        })
    }

    /// Upload one GGML quantized matrix to private Metal storage. The source
    /// bytes may be file-backed and unaligned; the shared staging buffer is
    /// released after the blit completes, leaving only the private copy alive.
    pub fn upload_quant_weight_private(
        &self,
        weight_bytes: &[u8],
        ggml_type: u32,
    ) -> Result<QuantWeight, String> {
        if weight_bytes.is_empty()
            || !matches!(
                ggml_type,
                crate::quant::GGML_TYPE_Q4_K
                    | crate::quant::GGML_TYPE_Q5_K
                    | crate::quant::GGML_TYPE_Q6_K
            )
        {
            return Err("unsupported or empty quantized weight".into());
        }
        let staging = self.device.new_buffer_with_data(
            weight_bytes.as_ptr() as *const std::ffi::c_void,
            weight_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let buffer = self.device.new_buffer(
            weight_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModePrivate,
        );
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_blit_command_encoder();
        encoder.copy_from_buffer(&staging, 0, &buffer, 0, weight_bytes.len() as u64);
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "quantized weight upload")?;
        let bytes = weight_bytes.len() as u64;
        self.note_persistent_weight_bytes(bytes);
        self.peak_allocated_bytes
            .fetch_max(bytes, Ordering::Relaxed);
        Ok(QuantWeight {
            buffer,
            offset: 0,
            bytes,
            ggml_type,
            mapped: false,
        })
    }

    /// Bind a quantized GGUF tensor directly to its file-backed mapping.
    ///
    /// This creates only a Metal view; it does not copy the payload. Keeping
    /// the view alive lets the runtime reuse the same resource identity on
    /// every token while the model store owns the underlying mapping.
    pub fn bind_quant_weight_mapped(
        &self,
        weight_bytes: &[u8],
        ggml_type: u32,
    ) -> Result<QuantWeight, String> {
        if weight_bytes.is_empty()
            || !matches!(
                ggml_type,
                crate::quant::GGML_TYPE_Q4_K
                    | crate::quant::GGML_TYPE_Q5_K
                    | crate::quant::GGML_TYPE_Q6_K
            )
        {
            return Err("unsupported or empty mapped quantized weight".into());
        }
        let buffer = self.device.new_buffer_with_bytes_no_copy(
            weight_bytes.as_ptr() as *const std::ffi::c_void,
            weight_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
            None,
        );
        if buffer.length() < weight_bytes.len() as u64 {
            return Err("Metal could not bind mapped quantized weight".into());
        }
        Ok(QuantWeight {
            buffer,
            offset: 0,
            bytes: weight_bytes.len() as u64,
            ggml_type,
            mapped: true,
        })
    }

    /// Promote one packed staged layer into a single transient private Metal
    /// buffer. The caller controls its lifetime by retaining or dropping the
    /// returned handle; no model-wide copy is created.
    pub fn upload_staged_qwen_layer_private(
        &self,
        staged: &StagedQwenLayer,
    ) -> Result<PrivateStagedQwenLayer, String> {
        let bytes = staged.packed_bytes();
        if bytes.is_empty() {
            return Err("cannot promote an empty staged Qwen layer".into());
        }
        let shared = self.device.new_buffer_with_data(
            bytes.as_ptr() as *const std::ffi::c_void,
            bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let buffer = self.device.new_buffer(
            bytes.len() as u64,
            metal::MTLResourceOptions::StorageModePrivate,
        );
        let command_buffer = self.upload_queue.new_command_buffer();
        let encoder = command_buffer.new_blit_command_encoder();
        encoder.copy_from_buffer(&shared, 0, &buffer, 0, bytes.len() as u64);
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "packed Qwen layer private upload")?;
        let ranges = staged
            .ranges()
            .map(|(name, start, end)| (name.to_owned(), (start, end)))
            .collect();
        Ok(PrivateStagedQwenLayer { buffer, ranges })
    }

    /// Bind one packed staged layer through a single shared-memory Metal
    /// buffer. The caller must retain `staged` for at least as long as the
    /// returned buffer is used; this removes one MTLBuffer creation per
    /// projection without copying the packed bytes.
    pub fn bind_staged_qwen_layer_shared(
        &self,
        staged: &StagedQwenLayer,
    ) -> Result<metal::Buffer, String> {
        let bytes = staged.packed_bytes();
        if bytes.is_empty() {
            return Err("cannot bind an empty staged Qwen layer".into());
        }
        let buffer = self.device.new_buffer_with_bytes_no_copy(
            bytes.as_ptr() as *const std::ffi::c_void,
            bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
            None,
        );
        if buffer.length() < bytes.len() as u64 {
            return Err("Metal could not bind packed staged Qwen bytes".into());
        }
        Ok(buffer)
    }

    /// Run one-vector RMSNorm. This is a correctness/profiling kernel, not the
    /// final parallel implementation; it proves shader compilation, buffer
    /// binding, command submission, and shared-buffer readback.
    pub fn rms_norm(
        &self,
        input: &[f32],
        weight: &[f32],
        epsilon: f32,
    ) -> Result<Vec<f32>, String> {
        if input.is_empty() || input.len() != weight.len() {
            return Err("RMSNorm input and weight must have equal non-zero length".into());
        }
        let byte_length = std::mem::size_of_val(input) as u64;
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            byte_length,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let weight_buffer = self.device.new_buffer_with_data(
            weight.as_ptr() as *const std::ffi::c_void,
            byte_length,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self
            .device
            .new_buffer(byte_length, metal::MTLResourceOptions::StorageModeShared);
        self.record_profile(0, 0, byte_length.saturating_mul(3));
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.rms_norm_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        let length = input.len() as u32;
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&length) as u64,
            &length as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of_val(&epsilon) as u64,
            &epsilon as *const f32 as *const std::ffi::c_void,
        );
        encoder.dispatch_threads(
            metal::MTLSize::new(1, 1, 1),
            metal::MTLSize::new(
                1,
                1,
                self.rms_norm_pipeline
                    .max_total_threads_per_threadgroup()
                    .max(1),
            ),
        );
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "command")?;
        // SAFETY: command buffer completed, output buffer is shared storage and
        // contains exactly input.len() initialized f32 values.
        let output = unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, input.len())
        };
        Ok(output.to_vec())
    }

    /// Apply RMSNorm to one or more contiguous vectors with a shared BF16
    /// scale vector. This keeps source norm weights immutable and widens them
    /// inside the Metal kernel.
    pub fn rms_norm_bf16_heads(
        &self,
        input: &[f32],
        weight_bytes: &[u8],
        heads: usize,
        head_dim: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, String> {
        let input_len = heads
            .checked_mul(head_dim)
            .ok_or("BF16 RMSNorm dimensions overflow")?;
        if heads == 0
            || head_dim == 0
            || input.len() != input_len
            || weight_bytes.len() != head_dim * 2
        {
            return Err("BF16 RMSNorm dimensions or byte length are invalid".into());
        }
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let weight_buffer = self.device.new_buffer_with_data(
            weight_bytes.as_ptr() as *const std::ffi::c_void,
            weight_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.record_profile(
            weight_bytes.len() as u64,
            0,
            (std::mem::size_of_val(input) as u64).saturating_mul(2),
        );
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.rms_norm_bf16_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        let heads = u32::try_from(heads).map_err(|_| "RMSNorm head count exceeds Metal limits")?;
        let head_dim =
            u32::try_from(head_dim).map_err(|_| "RMSNorm dimension exceeds Metal limits")?;
        encoder.set_bytes(3, 4, &heads as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &head_dim as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &epsilon as *const f32 as *const std::ffi::c_void);
        let threads = self
            .rms_norm_bf16_pipeline
            .max_total_threads_per_threadgroup()
            .max(1);
        encoder.dispatch_threads(
            metal::MTLSize::new(heads as u64, 1, 1),
            metal::MTLSize::new(threads.min(heads as u64), 1, 1),
        );
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "command")?;
        // SAFETY: command buffer completed and output contains input.len() f32s.
        let output = unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, input.len())
        };
        Ok(output.to_vec())
    }

    /// Apply per-head FP32 RMSNorm followed by a SiLU gate without creating a
    /// temporary normalized tensor on the CPU.
    pub fn rms_norm_gated(
        &self,
        input: &[f32],
        gate: &[f32],
        weight: &[f32],
        heads: usize,
        head_dim: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, String> {
        let input_len = heads
            .checked_mul(head_dim)
            .ok_or("Gated RMSNorm dimensions overflow")?;
        if heads == 0
            || head_dim == 0
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || input.len() != input_len
            || gate.len() != input_len
            || weight.len() != head_dim
        {
            return Err("Gated RMSNorm dimensions are invalid".into());
        }
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let gate_buffer = self.device.new_buffer_with_data(
            gate.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(gate) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let weight_buffer = self.device.new_buffer_with_bytes_no_copy(
            weight.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(weight) as u64,
            metal::MTLResourceOptions::StorageModeShared,
            None,
        );
        if weight_buffer.length() < std::mem::size_of_val(weight) as u64 {
            return Err("Metal could not bind mapped gated RMSNorm weights".into());
        }
        let output_buffer = self.device.new_buffer(
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.record_mapped_profile(
            std::mem::size_of_val(weight) as u64,
            0,
            std::mem::size_of_val(input) as u64 * 3,
        );
        let heads_u32 =
            u32::try_from(heads).map_err(|_| "Gated RMSNorm heads exceed Metal limits")?;
        let head_dim_u32 =
            u32::try_from(head_dim).map_err(|_| "Gated RMSNorm dimension exceeds limits")?;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.rms_norm_gated_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&gate_buffer), 0);
        encoder.set_buffer(2, Some(&weight_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.set_bytes(4, 4, &heads_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &head_dim_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(6, 4, &epsilon as *const f32 as *const std::ffi::c_void);
        let threads = self
            .rms_norm_gated_pipeline
            .max_total_threads_per_threadgroup()
            .max(1)
            .min(heads as u64);
        encoder.dispatch_threads(
            metal::MTLSize::new(heads as u64, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "gated RMSNorm command")?;
        // SAFETY: command buffer completed and output contains input.len() f32s.
        Ok(unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, input.len()).to_vec()
        })
    }

    /// Apply Qwen3.5's centered per-head FP32 RMSNorm, using `(1 + weight)`
    /// as the learned scale.
    pub fn rms_norm_heads(
        &self,
        input: &[f32],
        weight: &[f32],
        heads: usize,
        head_dim: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, String> {
        let input_len = heads
            .checked_mul(head_dim)
            .ok_or("head RMSNorm dimensions overflow")?;
        if heads == 0
            || head_dim == 0
            || !epsilon.is_finite()
            || epsilon <= 0.0
            || input.len() != input_len
            || weight.len() != head_dim
        {
            return Err("head RMSNorm dimensions are invalid".into());
        }
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let weight_buffer = self.device.new_buffer_with_bytes_no_copy(
            weight.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(weight) as u64,
            metal::MTLResourceOptions::StorageModeShared,
            None,
        );
        if weight_buffer.length() < std::mem::size_of_val(weight) as u64 {
            return Err("Metal could not bind mapped head RMSNorm weights".into());
        }
        let output_buffer = self.device.new_buffer(
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.record_mapped_profile(
            std::mem::size_of_val(weight) as u64,
            0,
            std::mem::size_of_val(input) as u64 * 2,
        );
        let heads_u32 = u32::try_from(heads).map_err(|_| "RMSNorm heads exceed Metal limits")?;
        let head_dim_u32 =
            u32::try_from(head_dim).map_err(|_| "RMSNorm dimension exceeds Metal limits")?;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.rms_norm_heads_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, 4, &heads_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &head_dim_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &epsilon as *const f32 as *const std::ffi::c_void);
        let threads = self
            .rms_norm_heads_pipeline
            .max_total_threads_per_threadgroup()
            .max(1)
            .min(heads as u64);
        encoder.dispatch_threads(
            metal::MTLSize::new(heads as u64, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "head RMSNorm command")?;
        // SAFETY: command buffer completed and output contains input.len() f32s.
        Ok(unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, input.len()).to_vec()
        })
    }

    pub fn rms_norm_bf16_tensor(
        &self,
        input: &[f32],
        tensor: &TensorView<'_>,
        epsilon: f32,
    ) -> Result<Vec<f32>, String> {
        if tensor.info.dtype != "BF16" || tensor.info.shape.len() != 1 {
            return Err("RMSNorm weight must be a rank-1 BF16 tensor".into());
        }
        self.rms_norm_bf16_heads(input, tensor.bytes, 1, tensor.info.shape[0], epsilon)
    }

    pub fn rms_norm_bf16_heads_tensor(
        &self,
        input: &[f32],
        tensor: &TensorView<'_>,
        heads: usize,
        head_dim: usize,
        epsilon: f32,
    ) -> Result<Vec<f32>, String> {
        if tensor.info.dtype != "BF16"
            || tensor.info.shape.len() != 1
            || tensor.info.shape[0] != head_dim
        {
            return Err("per-head RMSNorm weight has an invalid shape or dtype".into());
        }
        self.rms_norm_bf16_heads(input, tensor.bytes, heads, head_dim, epsilon)
    }

    /// Multiply a row-major BF16 matrix by an FP32 vector. Weight bytes stay
    /// BF16 until the Metal kernel widens each value for FP32 accumulation.
    pub fn bf16_matvec(
        &self,
        weight_bytes: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let expected_bytes = rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or("BF16 matrix dimensions overflow")?;
        if rows == 0
            || columns == 0
            || input.len() != columns
            || weight_bytes.len() != expected_bytes
        {
            return Err("BF16 matvec dimensions or byte length are invalid".into());
        }
        let weight = self.map_bf16_weight(weight_bytes)?;
        self.bf16_matvec_buffer(&weight, rows, columns, input)
    }

    /// Multiply a row-major F32 matrix by an FP32 vector without creating a
    /// second weight representation.
    pub fn f32_matvec(
        &self,
        weight_values: &[f32],
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let expected_values = rows
            .checked_mul(columns)
            .ok_or("F32 matvec dimensions overflow")?;
        if rows == 0
            || columns == 0
            || input.len() != columns
            || weight_values.len() != expected_values
        {
            return Err("F32 matvec dimensions or byte length are invalid".into());
        }
        let weight_buffer = self.device.new_buffer_with_bytes_no_copy(
            weight_values.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(weight_values) as u64,
            metal::MTLResourceOptions::StorageModeShared,
            None,
        );
        if weight_buffer.length() < std::mem::size_of_val(weight_values) as u64 {
            return Err("Metal could not bind the mapped F32 weight values".into());
        }
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64 * rows as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.record_mapped_profile(
            std::mem::size_of_val(weight_values) as u64,
            0,
            std::mem::size_of_val(input) as u64 + std::mem::size_of::<f32>() as u64 * rows as u64,
        );
        let columns_u32 =
            u32::try_from(columns).map_err(|_| "F32 matvec columns exceed Metal limits")?;
        let rows_u32 = u32::try_from(rows).map_err(|_| "F32 matvec rows exceed Metal limits")?;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.f32_matvec_pipeline);
        encoder.set_buffer(0, Some(&weight_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, 4, &columns_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &rows_u32 as *const u32 as *const std::ffi::c_void);
        let threads = self
            .f32_matvec_pipeline
            .max_total_threads_per_threadgroup()
            .clamp(1, 128);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(rows as u64, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "F32 matvec command")?;
        // SAFETY: the command buffer completed and output contains `rows`
        // contiguous FP32 values.
        Ok(unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, rows).to_vec()
        })
    }

    pub fn f32_matvec_bytes(
        &self,
        weight_bytes: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let expected_bytes = rows
            .checked_mul(columns)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .ok_or("F32 matvec byte length overflows")?;
        if weight_bytes.len() != expected_bytes {
            return Err("F32 matvec byte length is invalid".into());
        }
        // SAFETY: the GGUF F32 payload is 4-byte aligned and the byte length
        // is validated above; no bytes are reinterpreted outside the slice.
        let (prefix, values, suffix) = unsafe { weight_bytes.align_to::<f32>() };
        if !prefix.is_empty() || !suffix.is_empty() {
            return Err("F32 matvec bytes are not 4-byte aligned".into());
        }
        self.f32_matvec(values, rows, columns, input)
    }

    pub fn f32_matvec_tensor_rows(
        &self,
        tensor: &GgufTensorView<'_>,
        row_start: usize,
        row_count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if tensor.info.ggml_type != 0 || tensor.info.shape.len() != 2 {
            return Err("F32 matvec requires a rank-2 F32 tensor".into());
        }
        let columns = tensor.info.shape[0];
        let rows = tensor.info.shape[1];
        if row_count == 0 || row_start >= rows || row_count > rows - row_start {
            return Err("F32 row range is invalid".into());
        }
        if input.len() != columns {
            return Err("F32 tensor columns or input length is invalid".into());
        }
        let row_bytes = columns
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or("F32 row byte length overflows")?;
        let byte_start = row_start
            .checked_mul(row_bytes)
            .ok_or("F32 row byte range overflows")?;
        let byte_len = row_count
            .checked_mul(row_bytes)
            .ok_or("F32 row byte length overflows")?;
        let byte_end = byte_start
            .checked_add(byte_len)
            .ok_or("F32 row byte range overflows")?;
        if byte_end > tensor.bytes.len() {
            return Err("F32 tensor payload is shorter than its shape".into());
        }
        self.f32_matvec_bytes(
            &tensor.bytes[byte_start..byte_end],
            row_count,
            columns,
            input,
        )
    }

    /// Apply one exact recurrent Gated DeltaNet update per head.
    ///
    /// The recurrent state is copied into a shared Metal buffer, updated
    /// in-place by the kernel, and returned alongside the head outputs. This
    /// keeps the public primitive stateless while allowing the runtime to own
    /// and persist the returned state between tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_step(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        gate: &[f32],
        beta: &[f32],
        state: &[f32],
        heads: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        if heads == 0 || key_dim == 0 || value_dim == 0 || value_dim > 256 {
            return Err("Gated DeltaNet dimensions are invalid".into());
        }
        let head_key_values = heads
            .checked_mul(key_dim)
            .ok_or("Gated DeltaNet query dimensions overflow")?;
        let state_values = head_key_values
            .checked_mul(value_dim)
            .ok_or("Gated DeltaNet state dimensions overflow")?;
        if query.len() != head_key_values
            || key.len() != head_key_values
            || value.len() != heads * value_dim
            || gate.len() != heads
            || beta.len() != heads
            || state.len() != state_values
        {
            return Err("Gated DeltaNet input or state dimensions are invalid".into());
        }

        let query_buffer = self.device.new_buffer_with_data(
            query.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(query) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let key_buffer = self.device.new_buffer_with_data(
            key.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(key) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let value_buffer = self.device.new_buffer_with_data(
            value.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(value) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let gate_buffer = self.device.new_buffer_with_data(
            gate.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(gate) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let beta_buffer = self.device.new_buffer_with_data(
            beta.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(beta) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let state_buffer = self.device.new_buffer_with_data(
            state.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(state) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            std::mem::size_of_val(value) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let state_bytes = std::mem::size_of_val(state) as u64;
        let scratch_bytes = std::mem::size_of_val(query) as u64
            + std::mem::size_of_val(key) as u64
            + std::mem::size_of_val(value) as u64
            + std::mem::size_of_val(gate) as u64
            + std::mem::size_of_val(beta) as u64
            + std::mem::size_of_val(value) as u64;
        self.record_profile(0, state_bytes, scratch_bytes);

        let heads_u32 =
            u32::try_from(heads).map_err(|_| "Gated DeltaNet heads exceed Metal limits")?;
        let key_dim_u32 =
            u32::try_from(key_dim).map_err(|_| "Gated DeltaNet key dimension exceeds limits")?;
        let value_dim_u32 = u32::try_from(value_dim)
            .map_err(|_| "Gated DeltaNet value dimension exceeds Metal limits")?;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let parallel = std::env::var("SI_GDN_PARALLEL").ok().as_deref() == Some("1")
            && (value_dim as u64)
                <= self
                    .gated_delta_parallel_pipeline
                    .max_total_threads_per_threadgroup();
        let pipeline = if parallel {
            &self.gated_delta_parallel_pipeline
        } else {
            &self.gated_delta_pipeline
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&query_buffer), 0);
        encoder.set_buffer(1, Some(&key_buffer), 0);
        encoder.set_buffer(2, Some(&value_buffer), 0);
        encoder.set_buffer(3, Some(&gate_buffer), 0);
        encoder.set_buffer(4, Some(&beta_buffer), 0);
        encoder.set_buffer(5, Some(&state_buffer), 0);
        encoder.set_buffer(6, Some(&output_buffer), 0);
        encoder.set_bytes(7, 4, &heads_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(8, 4, &key_dim_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(
            9,
            4,
            &value_dim_u32 as *const u32 as *const std::ffi::c_void,
        );
        if parallel {
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(heads as u64, 1, 1),
                metal::MTLSize::new(value_dim as u64, 1, 1),
            );
        } else {
            let threads = self
                .gated_delta_pipeline
                .max_total_threads_per_threadgroup()
                .max(1)
                .min(heads as u64);
            encoder.dispatch_threads(
                metal::MTLSize::new(heads as u64, 1, 1),
                metal::MTLSize::new(threads, 1, 1),
            );
        }
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "Gated DeltaNet step command")?;

        // SAFETY: the command buffer completed and both shared buffers contain
        // exactly the validated number of FP32 values.
        let output = unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, value.len()).to_vec()
        };
        let updated_state = unsafe {
            std::slice::from_raw_parts(state_buffer.contents() as *const f32, state.len()).to_vec()
        };
        Ok((output, updated_state))
    }

    /// Apply one exact depthwise causal convolution update with SiLU.
    ///
    /// `weights` is row-major `[channel][kernel]`, and `state` is returned in
    /// its updated chronological `[channel][kernel - 1]` layout.
    pub fn causal_conv1d_step(
        &self,
        input: &[f32],
        state: &[f32],
        weights: &[f32],
        channels: usize,
        kernel_size: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        if channels == 0 || kernel_size < 2 || input.len() != channels {
            return Err("causal convolution dimensions are invalid".into());
        }
        let expected_state = channels
            .checked_mul(kernel_size - 1)
            .ok_or("causal convolution state dimensions overflow")?;
        let expected_weights = channels
            .checked_mul(kernel_size)
            .ok_or("causal convolution weight dimensions overflow")?;
        if state.len() != expected_state || weights.len() != expected_weights {
            return Err("causal convolution input, state, or weight lengths are invalid".into());
        }

        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let weight_buffer = self.device.new_buffer_with_bytes_no_copy(
            weights.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(weights) as u64,
            metal::MTLResourceOptions::StorageModeShared,
            None,
        );
        if weight_buffer.length() < std::mem::size_of_val(weights) as u64 {
            return Err("Metal could not bind mapped causal convolution weights".into());
        }
        let state_buffer = self.device.new_buffer_with_data(
            state.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(state) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let state_bytes = std::mem::size_of_val(state) as u64;
        self.record_mapped_profile(
            std::mem::size_of_val(weights) as u64,
            state_bytes,
            std::mem::size_of_val(input) as u64 + std::mem::size_of_val(input) as u64,
        );

        let channels_u32 =
            u32::try_from(channels).map_err(|_| "causal convolution channels exceed limits")?;
        let kernel_size_u32 = u32::try_from(kernel_size)
            .map_err(|_| "causal convolution kernel size exceeds limits")?;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.causal_conv_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&state_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.set_bytes(4, 4, &channels_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(
            5,
            4,
            &kernel_size_u32 as *const u32 as *const std::ffi::c_void,
        );
        let threads = self
            .causal_conv_pipeline
            .max_total_threads_per_threadgroup()
            .max(1)
            .min(channels as u64);
        encoder.dispatch_threads(
            metal::MTLSize::new(channels as u64, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "causal convolution command")?;

        // SAFETY: the command buffer completed and shared buffers contain the
        // validated output and updated state lengths.
        let output = unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, input.len()).to_vec()
        };
        let updated_state = unsafe {
            std::slice::from_raw_parts(state_buffer.contents() as *const f32, state.len()).to_vec()
        };
        Ok((output, updated_state))
    }

    /// Multiply a row-major GGML Q4_K matrix by an FP32 vector. Each 256-value
    /// block is decoded directly in the Metal kernel and accumulated in FP32;
    /// the original quantized bytes remain the only weight representation.
    pub fn q4_k_matvec(
        &self,
        weight_bytes: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if qk_rows4x128_enabled() {
            self.quant_k_matvec_dispatch(
                &self.q4_k_matvec_rows4x128_pipeline,
                weight_bytes,
                rows,
                columns,
                input,
                crate::quant::Q4_K_BLOCK_BYTES,
                "Q4_K",
                4,
            )
        } else if qk_rows4_enabled(std::env::var("SI_QK_ROWS4").ok().as_deref()) {
            self.quant_k_matvec_dispatch(
                &self.q4_k_matvec_rows4_pipeline,
                weight_bytes,
                rows,
                columns,
                input,
                crate::quant::Q4_K_BLOCK_BYTES,
                "Q4_K",
                4,
            )
        } else {
            self.quant_k_matvec(
                &self.q4_k_matvec_pipeline,
                weight_bytes,
                rows,
                columns,
                input,
                crate::quant::Q4_K_BLOCK_BYTES,
                "Q4_K",
            )
        }
    }

    /// Compute `SiLU(gate_weights * input) * (up_weights * input)` directly
    /// from two row-major Q4_K matrices without materializing either
    /// projection on the CPU or issuing two independent matvec dispatches.
    pub fn q4_k_fused_gate_up(
        &self,
        gate_weights: &[u8],
        up_weights: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if rows == 0 || columns == 0 || !columns.is_multiple_of(256) || input.len() != columns {
            return Err("Q4_K fused gate/up dimensions are invalid".into());
        }
        let blocks = rows
            .checked_mul(columns / 256)
            .ok_or("Q4_K fused gate/up block count overflows")?;
        let expected_bytes = blocks
            .checked_mul(crate::quant::Q4_K_BLOCK_BYTES)
            .ok_or("Q4_K fused gate/up byte length overflows")?;
        if gate_weights.len() != expected_bytes || up_weights.len() != expected_bytes {
            return Err("Q4_K fused gate/up byte length is invalid".into());
        }
        let gate_buffer = self.device.new_buffer_with_bytes_no_copy(
            gate_weights.as_ptr() as *const std::ffi::c_void,
            gate_weights.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
            None,
        );
        let up_buffer = self.device.new_buffer_with_bytes_no_copy(
            up_weights.as_ptr() as *const std::ffi::c_void,
            up_weights.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
            None,
        );
        if gate_buffer.length() < gate_weights.len() as u64
            || up_buffer.length() < up_weights.len() as u64
        {
            return Err("Metal could not bind mapped Q4_K fused gate/up weights".into());
        }
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64 * rows as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.record_mapped_profile(
            (gate_weights.len() as u64).saturating_add(up_weights.len() as u64),
            0,
            std::mem::size_of_val(input) as u64 + std::mem::size_of::<f32>() as u64 * rows as u64,
        );
        let columns_u32 =
            u32::try_from(columns).map_err(|_| "Q4_K fused gate/up columns exceed limits")?;
        let rows_u32 = u32::try_from(rows).map_err(|_| "Q4_K fused gate/up rows exceed limits")?;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.q4_k_fused_gate_up_pipeline);
        encoder.set_buffer(0, Some(&gate_buffer), 0);
        encoder.set_buffer(1, Some(&up_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.set_bytes(4, 4, &columns_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &rows_u32 as *const u32 as *const std::ffi::c_void);
        let threads = self
            .q4_k_fused_gate_up_pipeline
            .max_total_threads_per_threadgroup()
            .clamp(
                1,
                if std::env::var("SI_QK_FUSED_ROWS4X128").ok().as_deref() == Some("1") {
                    512
                } else {
                    128
                },
            );
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(rows as u64, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "Q4_K fused gate/up command")?;
        // SAFETY: the command buffer completed and output contains rows f32s.
        Ok(unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, rows).to_vec()
        })
    }

    /// Fused Q4_K gate/up projection using two retained private matrices.
    pub fn q4_k_fused_gate_up_weights(
        &self,
        gate: &QuantWeight,
        up: &QuantWeight,
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if gate.ggml_type != crate::quant::GGML_TYPE_Q4_K
            || up.ggml_type != crate::quant::GGML_TYPE_Q4_K
            || gate.bytes != up.bytes
            || rows == 0
            || columns == 0
            || !columns.is_multiple_of(256)
            || input.len() != columns
        {
            return Err("retained Q4_K fused gate/up dimensions are invalid".into());
        }
        let expected_bytes = rows
            .checked_mul(columns / crate::quant::Q4_K_BLOCK_ELEMENTS)
            .and_then(|blocks| blocks.checked_mul(crate::quant::Q4_K_BLOCK_BYTES))
            .ok_or("retained Q4_K fused gate/up byte length overflows")?;
        if gate.bytes as usize != expected_bytes {
            return Err("retained Q4_K fused gate/up byte length is invalid".into());
        }
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64 * rows as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.record_profile(
            0,
            0,
            std::mem::size_of_val(input) as u64 + std::mem::size_of::<f32>() as u64 * rows as u64,
        );
        let columns_u32 =
            u32::try_from(columns).map_err(|_| "retained gate/up columns exceed limits")?;
        let rows_u32 = u32::try_from(rows).map_err(|_| "retained gate/up rows exceed limits")?;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.q4_k_fused_gate_up_pipeline);
        encoder.set_buffer(0, Some(&gate.buffer), gate.offset);
        encoder.set_buffer(1, Some(&up.buffer), up.offset);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.set_bytes(4, 4, &columns_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &rows_u32 as *const u32 as *const std::ffi::c_void);
        let threads = self
            .q4_k_fused_gate_up_pipeline
            .max_total_threads_per_threadgroup()
            .clamp(
                1,
                if std::env::var("SI_QK_FUSED_ROWS4X128").ok().as_deref() == Some("1") {
                    512
                } else {
                    128
                },
            );
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(rows as u64, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "retained Q4_K fused gate/up command")?;
        // SAFETY: the command buffer completed and output contains `rows`
        // contiguous FP32 values.
        Ok(unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, rows).to_vec()
        })
    }

    /// Encode the Q4 gate/up projection and the following quantized down
    /// projection in one command buffer. The intermediate SwiGLU vector stays
    /// in shared GPU-visible storage; the CPU only waits once for the final
    /// residual contribution.
    #[allow(clippy::too_many_arguments)]
    pub fn gguf_quant_fused_mlp_weights(
        &self,
        gate: &QuantWeight,
        up: &QuantWeight,
        down: &QuantWeight,
        gate_rows: usize,
        columns: usize,
        down_rows: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if gate.ggml_type != crate::quant::GGML_TYPE_Q4_K
            || up.ggml_type != crate::quant::GGML_TYPE_Q4_K
            || gate.bytes != up.bytes
            || gate_rows == 0
            || columns == 0
            || down_rows == 0
            || !columns.is_multiple_of(256)
            || input.len() != columns
        {
            return Err("fused quantized MLP dimensions are invalid".into());
        }
        let gate_bytes = gate_rows
            .checked_mul(columns / crate::quant::Q4_K_BLOCK_ELEMENTS)
            .and_then(|blocks| blocks.checked_mul(crate::quant::Q4_K_BLOCK_BYTES))
            .ok_or("fused quantized MLP gate byte length overflows")?;
        if gate.bytes as usize != gate_bytes {
            return Err("fused quantized MLP gate byte length is invalid".into());
        }
        let down_columns = gate_rows;
        let down_block_bytes = match down.ggml_type {
            crate::quant::GGML_TYPE_Q4_K => crate::quant::Q4_K_BLOCK_BYTES,
            crate::quant::GGML_TYPE_Q5_K => crate::quant::Q5_K_BLOCK_BYTES,
            crate::quant::GGML_TYPE_Q6_K => crate::quant::Q6_K_BLOCK_BYTES,
            _ => return Err("unsupported fused quantized MLP down type".into()),
        };
        if !down_columns.is_multiple_of(256) {
            return Err("fused quantized MLP down columns are invalid".into());
        }
        let down_bytes = down_rows
            .checked_mul(down_columns / 256)
            .and_then(|blocks| blocks.checked_mul(down_block_bytes))
            .ok_or("fused quantized MLP down byte length overflows")?;
        if down.bytes as usize != down_bytes {
            return Err("fused quantized MLP down byte length is invalid".into());
        }
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let gate_up_buffer = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64 * gate_rows as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64 * down_rows as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.record_profile(
            0,
            0,
            std::mem::size_of_val(input) as u64
                + std::mem::size_of::<f32>() as u64 * (gate_rows + down_rows) as u64,
        );
        let columns_u32 =
            u32::try_from(columns).map_err(|_| "fused quantized MLP columns exceed limits")?;
        let gate_rows_u32 =
            u32::try_from(gate_rows).map_err(|_| "fused quantized MLP gate rows exceed limits")?;
        let down_rows_u32 =
            u32::try_from(down_rows).map_err(|_| "fused quantized MLP down rows exceed limits")?;
        let command_buffer = self.queue.new_command_buffer();
        let gate_encoder = command_buffer.new_compute_command_encoder();
        gate_encoder.set_compute_pipeline_state(&self.q4_k_fused_gate_up_pipeline);
        gate_encoder.set_buffer(0, Some(&gate.buffer), gate.offset);
        gate_encoder.set_buffer(1, Some(&up.buffer), up.offset);
        gate_encoder.set_buffer(2, Some(&input_buffer), 0);
        gate_encoder.set_buffer(3, Some(&gate_up_buffer), 0);
        gate_encoder.set_bytes(4, 4, &columns_u32 as *const u32 as *const std::ffi::c_void);
        gate_encoder.set_bytes(
            5,
            4,
            &gate_rows_u32 as *const u32 as *const std::ffi::c_void,
        );
        let gate_threads = self
            .q4_k_fused_gate_up_pipeline
            .max_total_threads_per_threadgroup()
            .clamp(
                1,
                if std::env::var("SI_QK_FUSED_ROWS4X128").ok().as_deref() == Some("1") {
                    512
                } else {
                    128
                },
            );
        gate_encoder.dispatch_thread_groups(
            metal::MTLSize::new(gate_rows as u64, 1, 1),
            metal::MTLSize::new(gate_threads, 1, 1),
        );
        gate_encoder.end_encoding();

        let down_rows4 = qk_rows4_enabled(std::env::var("SI_QK_ROWS4").ok().as_deref());
        let (down_pipeline, down_label) = match down.ggml_type {
            crate::quant::GGML_TYPE_Q4_K => (
                if down_rows4 {
                    &self.q4_k_matvec_rows4_pipeline
                } else {
                    &self.q4_k_matvec_pipeline
                },
                "Q4_K",
            ),
            crate::quant::GGML_TYPE_Q5_K => (
                if down_rows4 {
                    &self.q5_k_matvec_rows4_pipeline
                } else {
                    &self.q5_k_matvec_pipeline
                },
                "Q5_K",
            ),
            crate::quant::GGML_TYPE_Q6_K => (
                if down_rows4 {
                    &self.q6_k_matvec_rows4_pipeline
                } else {
                    &self.q6_k_matvec_pipeline
                },
                "Q6_K",
            ),
            _ => unreachable!(),
        };
        let down_encoder = command_buffer.new_compute_command_encoder();
        down_encoder.set_compute_pipeline_state(down_pipeline);
        down_encoder.set_buffer(0, Some(&down.buffer), down.offset);
        down_encoder.set_buffer(1, Some(&gate_up_buffer), 0);
        down_encoder.set_buffer(2, Some(&output_buffer), 0);
        down_encoder.set_bytes(
            3,
            4,
            &gate_rows_u32 as *const u32 as *const std::ffi::c_void,
        );
        down_encoder.set_bytes(
            4,
            4,
            &down_rows_u32 as *const u32 as *const std::ffi::c_void,
        );
        let down_threads = down_pipeline
            .max_total_threads_per_threadgroup()
            .clamp(1, 128);
        down_encoder.dispatch_thread_groups(
            metal::MTLSize::new(
                if down_rows4 {
                    down_rows.div_ceil(4)
                } else {
                    down_rows
                } as u64,
                1,
                1,
            ),
            metal::MTLSize::new(down_threads, 1, 1),
        );
        down_encoder.end_encoding();
        self.commit_and_wait(command_buffer, &format!("fused MLP {down_label} command"))?;
        Ok(unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, down_rows).to_vec()
        })
    }

    pub fn q4_k_embedding_row(&self, row_bytes: &[u8], columns: usize) -> Result<Vec<f32>, String> {
        if columns == 0 || !columns.is_multiple_of(256) {
            return Err("Q4_K embedding dimension is invalid".into());
        }
        let expected_bytes = columns / 256 * crate::quant::Q4_K_BLOCK_BYTES;
        if row_bytes.len() != expected_bytes {
            return Err("Q4_K embedding row byte length is invalid".into());
        }
        let weight_buffer = self.device.new_buffer_with_bytes_no_copy(
            row_bytes.as_ptr() as *const std::ffi::c_void,
            row_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
            None,
        );
        if weight_buffer.length() < row_bytes.len() as u64 {
            return Err("Metal could not bind the mapped Q4_K embedding row".into());
        }
        let output_buffer = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64 * columns as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.record_mapped_profile(
            row_bytes.len() as u64,
            0,
            std::mem::size_of::<f32>() as u64 * columns as u64,
        );
        let columns =
            u32::try_from(columns).map_err(|_| "embedding dimension exceeds Metal limits")?;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.q4_k_embedding_pipeline);
        encoder.set_buffer(0, Some(&weight_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(2, 4, &columns as *const u32 as *const std::ffi::c_void);
        let threads = self
            .q4_k_embedding_pipeline
            .max_total_threads_per_threadgroup()
            .max(1);
        encoder.dispatch_threads(
            metal::MTLSize::new(u64::from(columns), 1, 1),
            metal::MTLSize::new(threads.min(u64::from(columns)), 1, 1),
        );
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "Q4_K embedding command")?;
        // SAFETY: the command buffer completed and output contains columns f32s.
        Ok(unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, columns as usize)
                .to_vec()
        })
    }

    pub fn q4_k_embedding_tensor(
        &self,
        tensor: &GgufTensorView<'_>,
        token_id: usize,
    ) -> Result<Vec<f32>, String> {
        if tensor.info.ggml_type != crate::quant::GGML_TYPE_Q4_K || tensor.info.shape.len() != 2 {
            return Err("Q4_K embedding requires a rank-2 Q4_K tensor".into());
        }
        let columns = tensor.info.shape[0];
        let rows = tensor.info.shape[1];
        if token_id >= rows || !columns.is_multiple_of(256) {
            return Err("Q4_K embedding token id or dimension is invalid".into());
        }
        let row_bytes = columns / 256 * crate::quant::Q4_K_BLOCK_BYTES;
        let row_start = token_id
            .checked_mul(row_bytes)
            .ok_or("Q4_K embedding row offset overflows")?;
        let row_end = row_start
            .checked_add(row_bytes)
            .ok_or("Q4_K embedding row end overflows")?;
        let row = tensor
            .bytes
            .get(row_start..row_end)
            .ok_or("Q4_K embedding tensor payload is shorter than its shape")?;
        self.q4_k_embedding_row(row, columns)
    }

    #[allow(clippy::too_many_arguments)]
    fn quant_k_matvec(
        &self,
        pipeline: &metal::ComputePipelineState,
        weight_bytes: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
        block_bytes: usize,
        label: &str,
    ) -> Result<Vec<f32>, String> {
        self.quant_k_matvec_dispatch(
            pipeline,
            weight_bytes,
            rows,
            columns,
            input,
            block_bytes,
            label,
            1,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn quant_k_matvec_dispatch(
        &self,
        pipeline: &metal::ComputePipelineState,
        weight_bytes: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
        block_bytes: usize,
        label: &str,
        rows_per_group: usize,
    ) -> Result<Vec<f32>, String> {
        if rows == 0 || columns == 0 || !columns.is_multiple_of(256) || input.len() != columns {
            return Err(format!("{label} matvec dimensions are invalid"));
        }
        let blocks = rows
            .checked_mul(columns / 256)
            .ok_or_else(|| format!("{label} matvec block count overflows"))?;
        let expected_bytes = blocks
            .checked_mul(block_bytes)
            .ok_or_else(|| format!("{label} matvec byte length overflows"))?;
        if weight_bytes.len() != expected_bytes {
            return Err(format!("{label} matvec byte length is invalid"));
        }
        let weight_buffer = self.device.new_buffer_with_bytes_no_copy(
            weight_bytes.as_ptr() as *const std::ffi::c_void,
            weight_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
            None,
        );
        if weight_buffer.length() < weight_bytes.len() as u64 {
            return Err(format!(
                "Metal could not bind the mapped {label} weight bytes"
            ));
        }
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64 * rows as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.record_mapped_profile(
            weight_bytes.len() as u64,
            0,
            std::mem::size_of_val(input) as u64 + std::mem::size_of::<f32>() as u64 * rows as u64,
        );
        let columns_u32 =
            u32::try_from(columns).map_err(|_| format!("{label} columns exceed Metal limits"))?;
        let rows_u32 =
            u32::try_from(rows).map_err(|_| format!("{label} rows exceed Metal limits"))?;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&weight_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, 4, &columns_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &rows_u32 as *const u32 as *const std::ffi::c_void);
        let rows_per_group = rows_per_group.max(1);
        let thread_cap = if rows_per_group == 4 && qk_rows4x128_enabled() {
            512
        } else {
            128
        };
        let threads = pipeline
            .max_total_threads_per_threadgroup()
            .clamp(1, thread_cap);
        let groups = rows.div_ceil(rows_per_group);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(groups as u64, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, &format!("{label} matvec command"))?;
        // SAFETY: the command buffer completed and output contains `rows`
        // contiguous FP32 values.
        Ok(unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, rows).to_vec()
        })
    }

    /// Execute a matvec against a previously uploaded private quantized
    /// matrix. This keeps the kernel identical to the mapped path while
    /// removing repeated zero-copy buffer creation and file-backed page
    /// binding for retained layers.
    pub fn gguf_quant_matvec_weight(
        &self,
        weight: &QuantWeight,
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let rows4x = qk_rows4x128_for_type(weight.ggml_type);
        let rows4 = rows4x
            || qk_rows4_enabled(std::env::var("SI_QK_ROWS4").ok().as_deref())
                && matches!(
                    weight.ggml_type,
                    crate::quant::GGML_TYPE_Q4_K
                        | crate::quant::GGML_TYPE_Q5_K
                        | crate::quant::GGML_TYPE_Q6_K
                );
        let (pipeline, block_bytes, label) = match weight.ggml_type {
            crate::quant::GGML_TYPE_Q4_K => (
                if rows4x {
                    &self.q4_k_matvec_rows4x128_pipeline
                } else if rows4 {
                    &self.q4_k_matvec_rows4_pipeline
                } else {
                    &self.q4_k_matvec_pipeline
                },
                crate::quant::Q4_K_BLOCK_BYTES,
                "Q4_K",
            ),
            crate::quant::GGML_TYPE_Q5_K => (
                if rows4 {
                    &self.q5_k_matvec_rows4_pipeline
                } else {
                    &self.q5_k_matvec_pipeline
                },
                crate::quant::Q5_K_BLOCK_BYTES,
                "Q5_K",
            ),
            crate::quant::GGML_TYPE_Q6_K => (
                if rows4x {
                    &self.q6_k_matvec_rows4x128_pipeline
                } else if rows4 {
                    &self.q6_k_matvec_rows4_pipeline
                } else {
                    &self.q6_k_matvec_pipeline
                },
                crate::quant::Q6_K_BLOCK_BYTES,
                "Q6_K",
            ),
            _ => return Err("unsupported retained quantized weight type".into()),
        };
        if rows == 0 || columns == 0 || !columns.is_multiple_of(256) || input.len() != columns {
            return Err(format!("{label} retained matvec dimensions are invalid"));
        }
        let expected_bytes = rows
            .checked_mul(columns / 256)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or_else(|| format!("{label} retained matvec byte length overflows"))?;
        if expected_bytes as u64 != weight.bytes {
            return Err(format!("{label} retained matvec byte length is invalid"));
        }
        self.quant_k_matvec_buffer_dispatch(
            pipeline,
            &weight.buffer,
            weight.offset,
            weight.bytes,
            rows,
            columns,
            input,
            label,
            true,
            if rows4 { 4 } else { 1 },
        )
    }

    /// Execute a quantized GGUF matrix from an anonymous/staged byte slice.
    /// The kernel is identical to the mmap path; only the backing allocation
    /// changes, which lets callers bound unified-memory file-cache pressure.
    pub fn gguf_quant_matvec_bytes(
        &self,
        ggml_type: u32,
        weight_bytes: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let rows4x = qk_rows4x128_for_type(ggml_type);
        let rows4 = rows4x
            || qk_rows4_enabled(std::env::var("SI_QK_ROWS4").ok().as_deref())
                && matches!(
                    ggml_type,
                    crate::quant::GGML_TYPE_Q4_K
                        | crate::quant::GGML_TYPE_Q5_K
                        | crate::quant::GGML_TYPE_Q6_K
                );
        let (pipeline, block_bytes, label) = match ggml_type {
            crate::quant::GGML_TYPE_Q4_K => (
                if rows4x {
                    &self.q4_k_matvec_rows4x128_pipeline
                } else if rows4 {
                    &self.q4_k_matvec_rows4_pipeline
                } else {
                    &self.q4_k_matvec_pipeline
                },
                crate::quant::Q4_K_BLOCK_BYTES,
                "Q4_K",
            ),
            crate::quant::GGML_TYPE_Q5_K => (
                if rows4 {
                    &self.q5_k_matvec_rows4_pipeline
                } else {
                    &self.q5_k_matvec_pipeline
                },
                crate::quant::Q5_K_BLOCK_BYTES,
                "Q5_K",
            ),
            crate::quant::GGML_TYPE_Q6_K => (
                if rows4x {
                    &self.q6_k_matvec_rows4x128_pipeline
                } else if rows4 {
                    &self.q6_k_matvec_rows4_pipeline
                } else {
                    &self.q6_k_matvec_pipeline
                },
                crate::quant::Q6_K_BLOCK_BYTES,
                "Q6_K",
            ),
            _ => return Err("unsupported staged quantized weight type".into()),
        };
        self.quant_k_matvec_dispatch(
            pipeline,
            weight_bytes,
            rows,
            columns,
            input,
            block_bytes,
            label,
            if rows4 { 4 } else { 1 },
        )
    }

    /// Execute one quantized matrix against several candidate vectors in one
    /// dispatch. Inputs and outputs are batch-major. The byte slice remains
    /// unchanged, so this is usable for both mmap-backed and staged GGUF
    /// tensors.
    pub fn gguf_quant_matmul_many_bytes(
        &self,
        ggml_type: u32,
        weight_bytes: &[u8],
        rows: usize,
        columns: usize,
        batch: usize,
        inputs: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        if batch == 0 || batch > 8 || rows == 0 || columns == 0 || !columns.is_multiple_of(256) {
            return Err("batched quantized matvec dimensions are invalid".into());
        }
        if inputs.len() != batch.saturating_mul(columns) {
            return Err("batched quantized input length is invalid".into());
        }
        let (pipeline, block_bytes) = match ggml_type {
            crate::quant::GGML_TYPE_Q4_K => (
                &self.q4_k_matmul_many_pipeline,
                crate::quant::Q4_K_BLOCK_BYTES,
            ),
            crate::quant::GGML_TYPE_Q5_K => (
                &self.q5_k_matmul_many_pipeline,
                crate::quant::Q5_K_BLOCK_BYTES,
            ),
            crate::quant::GGML_TYPE_Q6_K => (
                &self.q6_k_matmul_many_pipeline,
                crate::quant::Q6_K_BLOCK_BYTES,
            ),
            _ => return Err("unsupported batched quantized weight type".into()),
        };
        let expected_bytes = rows
            .checked_mul(columns / 256)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or("batched quantized weight byte length overflows")?;
        if weight_bytes.len() != expected_bytes {
            return Err("batched quantized weight byte length is invalid".into());
        }
        let input_bytes = std::mem::size_of_val(inputs) as u64;
        let output_elements = batch
            .checked_mul(rows)
            .ok_or("batched quantized output dimensions overflow")?;
        let output_bytes = output_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or("batched quantized output byte length overflows")?;
        let weight_buffer = self.device.new_buffer_with_bytes_no_copy(
            weight_bytes.as_ptr() as *const std::ffi::c_void,
            weight_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
            None,
        );
        let input_buffer = self.device.new_buffer_with_data(
            inputs.as_ptr() as *const std::ffi::c_void,
            input_bytes,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            output_bytes as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.record_mapped_profile(
            weight_bytes.len() as u64,
            0,
            input_bytes + output_bytes as u64,
        );
        let columns_u32 = u32::try_from(columns).map_err(|_| "batched columns exceed limits")?;
        let rows_u32 = u32::try_from(rows).map_err(|_| "batched rows exceed limits")?;
        let batch_u32 = u32::try_from(batch).map_err(|_| "batched count exceeds limits")?;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&weight_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, 4, &columns_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &rows_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &batch_u32 as *const u32 as *const std::ffi::c_void);
        let threads = pipeline.max_total_threads_per_threadgroup().clamp(1, 128);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(rows as u64, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "batched quantized matvec command")?;
        // SAFETY: the command completed and output contains batch * rows f32s.
        let flat = unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, output_elements)
        };
        Ok(flat
            .chunks_exact(rows)
            .map(|candidate| candidate.to_vec())
            .collect())
    }

    /// Batched quantized matvec against a private retained matrix.
    pub fn gguf_quant_matmul_many_weight(
        &self,
        weight: &QuantWeight,
        rows: usize,
        columns: usize,
        batch: usize,
        inputs: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        if batch == 0 || batch > 8 || rows == 0 || columns == 0 || !columns.is_multiple_of(256) {
            return Err("batched retained quantized dimensions are invalid".into());
        }
        if inputs.len() != batch.saturating_mul(columns) {
            return Err("batched retained quantized input length is invalid".into());
        }
        let (pipeline, block_bytes) = match weight.ggml_type {
            crate::quant::GGML_TYPE_Q4_K => (
                &self.q4_k_matmul_many_pipeline,
                crate::quant::Q4_K_BLOCK_BYTES,
            ),
            crate::quant::GGML_TYPE_Q5_K => (
                &self.q5_k_matmul_many_pipeline,
                crate::quant::Q5_K_BLOCK_BYTES,
            ),
            crate::quant::GGML_TYPE_Q6_K => (
                &self.q6_k_matmul_many_pipeline,
                crate::quant::Q6_K_BLOCK_BYTES,
            ),
            _ => return Err("unsupported retained batched quantized type".into()),
        };
        let expected_bytes = rows
            .checked_mul(columns / 256)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or("retained batched weight byte length overflows")?;
        if weight.bytes != expected_bytes as u64 {
            return Err("retained batched weight byte length is invalid".into());
        }
        let input_bytes = std::mem::size_of_val(inputs) as u64;
        let output_elements = batch
            .checked_mul(rows)
            .ok_or("retained batched output dimensions overflow")?;
        let output_bytes = output_elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or("retained batched output byte length overflows")?;
        let input_buffer = self.device.new_buffer_with_data(
            inputs.as_ptr() as *const std::ffi::c_void,
            input_bytes,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            output_bytes as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.record_profile(0, 0, input_bytes + output_bytes as u64);
        let columns_u32 = u32::try_from(columns).map_err(|_| "retained columns exceed limits")?;
        let rows_u32 = u32::try_from(rows).map_err(|_| "retained rows exceed limits")?;
        let batch_u32 = u32::try_from(batch).map_err(|_| "retained batch exceeds limits")?;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&weight.buffer), weight.offset);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, 4, &columns_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &rows_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &batch_u32 as *const u32 as *const std::ffi::c_void);
        let threads = pipeline.max_total_threads_per_threadgroup().clamp(1, 128);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(rows as u64, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "retained batched quantized matvec command")?;
        // SAFETY: the command completed and output contains batch * rows f32s.
        let flat = unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, output_elements)
        };
        Ok(flat
            .chunks_exact(rows)
            .map(|candidate| candidate.to_vec())
            .collect())
    }

    #[allow(clippy::too_many_arguments)]
    fn quant_k_matvec_buffer_dispatch(
        &self,
        pipeline: &metal::ComputePipelineState,
        weight_buffer: &metal::BufferRef,
        weight_offset: u64,
        weight_bytes: u64,
        rows: usize,
        columns: usize,
        input: &[f32],
        label: &str,
        persistent: bool,
        rows_per_group: usize,
    ) -> Result<Vec<f32>, String> {
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64 * rows as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let scratch_bytes =
            std::mem::size_of_val(input) as u64 + std::mem::size_of::<f32>() as u64 * rows as u64;
        if persistent {
            self.record_profile(0, 0, scratch_bytes);
        } else {
            self.record_mapped_profile(weight_bytes, 0, scratch_bytes);
        }
        let columns_u32 =
            u32::try_from(columns).map_err(|_| format!("{label} columns exceed Metal limits"))?;
        let rows_u32 =
            u32::try_from(rows).map_err(|_| format!("{label} rows exceed Metal limits"))?;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(weight_buffer), weight_offset);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, 4, &columns_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &rows_u32 as *const u32 as *const std::ffi::c_void);
        let rows_per_group = rows_per_group.max(1);
        let thread_cap = if rows_per_group == 4 && qk_rows4x128_enabled() {
            512
        } else {
            128
        };
        let threads = pipeline
            .max_total_threads_per_threadgroup()
            .clamp(1, thread_cap);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(rows.div_ceil(rows_per_group) as u64, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, &format!("{label} retained matvec command"))?;
        // SAFETY: the command buffer completed and output contains `rows`
        // contiguous FP32 values.
        Ok(unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, rows).to_vec()
        })
    }

    pub fn q5_k_matvec(
        &self,
        weight_bytes: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.quant_k_matvec(
            &self.q5_k_matvec_pipeline,
            weight_bytes,
            rows,
            columns,
            input,
            crate::quant::Q5_K_BLOCK_BYTES,
            "Q5_K",
        )
    }

    pub fn q6_k_matvec(
        &self,
        weight_bytes: &[u8],
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.quant_k_matvec(
            &self.q6_k_matvec_pipeline,
            weight_bytes,
            rows,
            columns,
            input,
            crate::quant::Q6_K_BLOCK_BYTES,
            "Q6_K",
        )
    }

    /// Multiply a GGUF Q4_K matrix while honoring GGUF's `[columns, rows]`
    /// shape convention. Row ranges are intentionally explicit so callers can
    /// stream a large matrix through a bounded fast-memory window.
    pub fn q4_k_matvec_tensor_rows(
        &self,
        tensor: &GgufTensorView<'_>,
        row_start: usize,
        row_count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if tensor.info.ggml_type != crate::quant::GGML_TYPE_Q4_K || tensor.info.shape.len() != 2 {
            return Err("Q4_K matvec requires a rank-2 Q4_K tensor".into());
        }
        let columns = tensor.info.shape[0];
        let rows = tensor.info.shape[1];
        if row_count == 0 || row_start >= rows || row_count > rows - row_start {
            return Err("Q4_K row range is invalid".into());
        }
        if !columns.is_multiple_of(crate::quant::Q4_K_BLOCK_ELEMENTS) || input.len() != columns {
            return Err("Q4_K tensor columns or input length is invalid".into());
        }
        let row_bytes =
            columns / crate::quant::Q4_K_BLOCK_ELEMENTS * crate::quant::Q4_K_BLOCK_BYTES;
        let byte_start = row_start
            .checked_mul(row_bytes)
            .ok_or("Q4_K row byte range overflows")?;
        let byte_len = row_count
            .checked_mul(row_bytes)
            .ok_or("Q4_K row byte length overflows")?;
        let byte_end = byte_start
            .checked_add(byte_len)
            .ok_or("Q4_K row byte range overflows")?;
        if byte_end > tensor.bytes.len() {
            return Err("Q4_K tensor payload is shorter than its shape".into());
        }
        self.q4_k_matvec(
            &tensor.bytes[byte_start..byte_end],
            row_count,
            columns,
            input,
        )
    }

    pub fn q4_k_matvec_tensor(
        &self,
        tensor: &GgufTensorView<'_>,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if tensor.info.shape.len() != 2 {
            return Err("Q4_K matvec requires a rank-2 tensor".into());
        }
        self.q4_k_matvec_tensor_rows(tensor, 0, tensor.info.shape[1], input)
    }

    #[allow(clippy::too_many_arguments)]
    fn quant_k_matvec_tensor_rows(
        &self,
        tensor: &GgufTensorView<'_>,
        row_start: usize,
        row_count: usize,
        input: &[f32],
        ggml_type: u32,
        block_bytes: usize,
        label: &str,
    ) -> Result<Vec<f32>, String> {
        if tensor.info.ggml_type != ggml_type || tensor.info.shape.len() != 2 {
            return Err(format!("{label} matvec requires a rank-2 {label} tensor"));
        }
        let columns = tensor.info.shape[0];
        let rows = tensor.info.shape[1];
        if row_count == 0 || row_start >= rows || row_count > rows - row_start {
            return Err(format!("{label} row range is invalid"));
        }
        if !columns.is_multiple_of(256) || input.len() != columns {
            return Err(format!("{label} tensor columns or input length is invalid"));
        }
        let row_bytes = columns / 256 * block_bytes;
        let byte_start = row_start
            .checked_mul(row_bytes)
            .ok_or_else(|| format!("{label} row byte range overflows"))?;
        let byte_len = row_count
            .checked_mul(row_bytes)
            .ok_or_else(|| format!("{label} row byte length overflows"))?;
        let byte_end = byte_start
            .checked_add(byte_len)
            .ok_or_else(|| format!("{label} row byte range overflows"))?;
        if byte_end > tensor.bytes.len() {
            return Err(format!("{label} tensor payload is shorter than its shape"));
        }
        let rows4x = qk_rows4x128_for_type(ggml_type);
        let rows4 = rows4x || qk_rows4_enabled(std::env::var("SI_QK_ROWS4").ok().as_deref());
        match ggml_type {
            crate::quant::GGML_TYPE_Q5_K => self.quant_k_matvec_dispatch(
                if rows4 {
                    &self.q5_k_matvec_rows4_pipeline
                } else {
                    &self.q5_k_matvec_pipeline
                },
                &tensor.bytes[byte_start..byte_end],
                row_count,
                columns,
                input,
                block_bytes,
                label,
                if rows4 { 4 } else { 1 },
            ),
            crate::quant::GGML_TYPE_Q6_K => self.quant_k_matvec_dispatch(
                if rows4x {
                    &self.q6_k_matvec_rows4x128_pipeline
                } else if rows4 {
                    &self.q6_k_matvec_rows4_pipeline
                } else {
                    &self.q6_k_matvec_pipeline
                },
                &tensor.bytes[byte_start..byte_end],
                row_count,
                columns,
                input,
                block_bytes,
                label,
                if rows4 { 4 } else { 1 },
            ),
            _ => Err(format!("unsupported {label} tensor type {ggml_type}")),
        }
    }

    pub fn q5_k_matvec_tensor_rows(
        &self,
        tensor: &GgufTensorView<'_>,
        row_start: usize,
        row_count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.quant_k_matvec_tensor_rows(
            tensor,
            row_start,
            row_count,
            input,
            crate::quant::GGML_TYPE_Q5_K,
            crate::quant::Q5_K_BLOCK_BYTES,
            "Q5_K",
        )
    }

    pub fn q6_k_matvec_tensor_rows(
        &self,
        tensor: &GgufTensorView<'_>,
        row_start: usize,
        row_count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        self.quant_k_matvec_tensor_rows(
            tensor,
            row_start,
            row_count,
            input,
            crate::quant::GGML_TYPE_Q6_K,
            crate::quant::Q6_K_BLOCK_BYTES,
            "Q6_K",
        )
    }

    pub fn gguf_quant_matvec_tensor_rows(
        &self,
        tensor: &GgufTensorView<'_>,
        row_start: usize,
        row_count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        match tensor.info.ggml_type {
            crate::quant::GGML_TYPE_Q4_K => {
                self.q4_k_matvec_tensor_rows(tensor, row_start, row_count, input)
            }
            crate::quant::GGML_TYPE_Q5_K => {
                self.q5_k_matvec_tensor_rows(tensor, row_start, row_count, input)
            }
            crate::quant::GGML_TYPE_Q6_K => {
                self.q6_k_matvec_tensor_rows(tensor, row_start, row_count, input)
            }
            ggml_type => Err(format!(
                "GGUF tensor {} has unsupported quantized type {ggml_type}",
                tensor.info.name
            )),
        }
    }

    /// Submit several independent quantized matvecs against the same input
    /// vector in one command buffer. The projections remain separate outputs,
    /// but CPU/Metal synchronization happens once for the group.
    pub fn gguf_quant_matvec_many_tensors(
        &self,
        tensors: &[&GgufTensorView<'_>],
        input: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        if tensors.is_empty() {
            return Err("quantized matvec group requires at least one tensor".into());
        }
        let columns = tensors[0]
            .info
            .shape
            .first()
            .copied()
            .ok_or("quantized matvec tensor is not a matrix")?;
        if input.len() != columns {
            return Err("quantized matvec group input length is invalid".into());
        }
        let mut weights = Vec::with_capacity(tensors.len());
        let mut outputs = Vec::with_capacity(tensors.len());
        let mut specs = Vec::with_capacity(tensors.len());
        let mut scratch_bytes = std::mem::size_of_val(input) as u64;
        let mut mapped_bytes = 0_u64;
        for tensor in tensors {
            if tensor.info.shape.len() != 2
                || tensor.info.shape[0] != columns
                || !matches!(
                    tensor.info.ggml_type,
                    crate::quant::GGML_TYPE_Q4_K
                        | crate::quant::GGML_TYPE_Q5_K
                        | crate::quant::GGML_TYPE_Q6_K
                )
            {
                return Err("quantized matvec group tensor shapes or types differ".into());
            }
            let rows = tensor.info.shape[1];
            let block_bytes = match tensor.info.ggml_type {
                crate::quant::GGML_TYPE_Q4_K => crate::quant::Q4_K_BLOCK_BYTES,
                crate::quant::GGML_TYPE_Q5_K => crate::quant::Q5_K_BLOCK_BYTES,
                crate::quant::GGML_TYPE_Q6_K => crate::quant::Q6_K_BLOCK_BYTES,
                _ => unreachable!(),
            };
            let expected_bytes = rows
                .checked_mul(columns / 256)
                .and_then(|blocks| blocks.checked_mul(block_bytes))
                .ok_or("quantized matvec group byte length overflows")?;
            if tensor.bytes.len() < expected_bytes {
                return Err("quantized matvec group tensor is truncated".into());
            }
            let weight = self.device.new_buffer_with_bytes_no_copy(
                tensor.bytes.as_ptr() as *const std::ffi::c_void,
                expected_bytes as u64,
                metal::MTLResourceOptions::StorageModeShared,
                None,
            );
            if weight.length() < expected_bytes as u64 {
                return Err("Metal could not bind grouped quantized weights".into());
            }
            let output = self.device.new_buffer(
                std::mem::size_of::<f32>() as u64 * rows as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            mapped_bytes = mapped_bytes.saturating_add(expected_bytes as u64);
            scratch_bytes =
                scratch_bytes.saturating_add(std::mem::size_of::<f32>() as u64 * rows as u64);
            specs.push((tensor.info.ggml_type, rows));
            weights.push(weight);
            outputs.push(output);
        }
        self.record_mapped_profile(mapped_bytes, 0, scratch_bytes);
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = input_buffer;
        let command_buffer = self.queue.new_command_buffer();
        for ((weight, output), (ggml_type, rows)) in weights.iter().zip(&outputs).zip(specs) {
            let pipeline = match ggml_type {
                crate::quant::GGML_TYPE_Q4_K => &self.q4_k_matvec_pipeline,
                crate::quant::GGML_TYPE_Q5_K => &self.q5_k_matvec_pipeline,
                crate::quant::GGML_TYPE_Q6_K => &self.q6_k_matvec_pipeline,
                _ => unreachable!(),
            };
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(weight), 0);
            encoder.set_buffer(1, Some(&input_buffer), 0);
            encoder.set_buffer(2, Some(output), 0);
            let columns_u32 = u32::try_from(columns)
                .map_err(|_| "grouped quantized columns exceed Metal limits")?;
            let rows_u32 =
                u32::try_from(rows).map_err(|_| "grouped quantized rows exceed Metal limits")?;
            encoder.set_bytes(3, 4, &columns_u32 as *const u32 as *const std::ffi::c_void);
            encoder.set_bytes(4, 4, &rows_u32 as *const u32 as *const std::ffi::c_void);
            let threads = pipeline.max_total_threads_per_threadgroup().clamp(1, 128);
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(rows as u64, 1, 1),
                metal::MTLSize::new(threads, 1, 1),
            );
            encoder.end_encoding();
        }
        self.commit_and_wait(command_buffer, "grouped quantized matvec command")?;
        outputs
            .iter()
            .zip(tensors.iter())
            .map(|(output, tensor)| {
                let rows = tensor.info.shape[1];
                // SAFETY: the command buffer completed and output contains
                // exactly `rows` contiguous FP32 values.
                Ok(unsafe {
                    std::slice::from_raw_parts(output.contents() as *const f32, rows).to_vec()
                })
            })
            .collect()
    }

    /// Submit several independent matvecs whose quantized matrices are
    /// already retained in Metal. This is the private-buffer counterpart of
    /// `gguf_quant_matvec_many_tensors`; it keeps full-layer residency from
    /// accidentally turning one Q/K/V group into three command-buffer waits.
    pub fn gguf_quant_matvec_many_weights(
        &self,
        weights: &[(&QuantWeight, usize, usize)],
        input: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        if weights.is_empty() {
            return Err("retained quantized matvec group requires at least one weight".into());
        }
        let columns = weights[0].2;
        if columns == 0 || input.len() != columns {
            return Err("retained quantized matvec group input length is invalid".into());
        }
        let mut outputs = Vec::with_capacity(weights.len());
        let mut specs = Vec::with_capacity(weights.len());
        let mut scratch_bytes = std::mem::size_of_val(input) as u64;
        for (weight, rows, tensor_columns) in weights {
            if *rows == 0 || *tensor_columns != columns || !columns.is_multiple_of(256) {
                return Err("retained quantized matvec group shapes differ".into());
            }
            let (block_bytes, rows4x, rows4) = match weight.ggml_type {
                crate::quant::GGML_TYPE_Q4_K => (
                    crate::quant::Q4_K_BLOCK_BYTES,
                    qk_rows4x128_for_type(weight.ggml_type),
                    qk_rows4_enabled(std::env::var("SI_QK_ROWS4").ok().as_deref()),
                ),
                crate::quant::GGML_TYPE_Q5_K => (
                    crate::quant::Q5_K_BLOCK_BYTES,
                    false,
                    qk_rows4_enabled(std::env::var("SI_QK_ROWS4").ok().as_deref()),
                ),
                crate::quant::GGML_TYPE_Q6_K => (
                    crate::quant::Q6_K_BLOCK_BYTES,
                    qk_rows4x128_for_type(weight.ggml_type),
                    qk_rows4_enabled(std::env::var("SI_QK_ROWS4").ok().as_deref()),
                ),
                _ => return Err("unsupported retained quantized matvec group type".into()),
            };
            let expected_bytes = rows
                .checked_mul(columns / 256)
                .and_then(|blocks| blocks.checked_mul(block_bytes))
                .ok_or("retained quantized matvec group byte length overflows")?;
            if weight.bytes != expected_bytes as u64 {
                return Err("retained quantized matvec group byte length is invalid".into());
            }
            let output = self.device.new_buffer(
                std::mem::size_of::<f32>() as u64 * *rows as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            scratch_bytes =
                scratch_bytes.saturating_add(std::mem::size_of::<f32>() as u64 * *rows as u64);
            specs.push((weight, *rows, rows4x, rows4));
            outputs.push(output);
        }
        self.record_profile(0, 0, scratch_bytes);
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.queue.new_command_buffer();
        for ((weight, rows, rows4x, rows4), output) in specs.iter().zip(&outputs) {
            let pipeline = match weight.ggml_type {
                crate::quant::GGML_TYPE_Q4_K => {
                    if *rows4x {
                        &self.q4_k_matvec_rows4x128_pipeline
                    } else if *rows4 {
                        &self.q4_k_matvec_rows4_pipeline
                    } else {
                        &self.q4_k_matvec_pipeline
                    }
                }
                crate::quant::GGML_TYPE_Q5_K => {
                    if *rows4 {
                        &self.q5_k_matvec_rows4_pipeline
                    } else {
                        &self.q5_k_matvec_pipeline
                    }
                }
                crate::quant::GGML_TYPE_Q6_K => {
                    if *rows4x {
                        &self.q6_k_matvec_rows4x128_pipeline
                    } else if *rows4 {
                        &self.q6_k_matvec_rows4_pipeline
                    } else {
                        &self.q6_k_matvec_pipeline
                    }
                }
                _ => return Err("unsupported retained quantized matvec group type".into()),
            };
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(&weight.buffer), weight.offset);
            encoder.set_buffer(1, Some(&input_buffer), 0);
            encoder.set_buffer(2, Some(output), 0);
            let columns_u32 = u32::try_from(columns)
                .map_err(|_| "retained quantized columns exceed Metal limits")?;
            let rows_u32 =
                u32::try_from(*rows).map_err(|_| "retained quantized rows exceed Metal limits")?;
            encoder.set_bytes(3, 4, &columns_u32 as *const u32 as *const std::ffi::c_void);
            encoder.set_bytes(4, 4, &rows_u32 as *const u32 as *const std::ffi::c_void);
            let rows_per_group = if *rows4 { 4 } else { 1 };
            let thread_cap = if *rows4x { 512 } else { 128 };
            let threads = pipeline
                .max_total_threads_per_threadgroup()
                .clamp(1, thread_cap);
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(rows.div_ceil(rows_per_group) as u64, 1, 1),
                metal::MTLSize::new(threads, 1, 1),
            );
            encoder.end_encoding();
        }
        self.commit_and_wait(command_buffer, "retained grouped quantized matvec command")?;
        outputs
            .iter()
            .zip(weights)
            .map(|(output, (_, rows, _))| {
                // SAFETY: the command buffer completed and output contains
                // exactly `rows` contiguous FP32 values.
                Ok(unsafe {
                    std::slice::from_raw_parts(output.contents() as *const f32, *rows).to_vec()
                })
            })
            .collect()
    }

    /// Submit independent GGUF F32 projections against one input in a single
    /// command buffer. Qwen3.6's Gated DeltaNet alpha and beta projections use
    /// this path to avoid two host waits for two 48-row matrices.
    pub fn f32_matvec_many_gguf_tensors(
        &self,
        tensors: &[&GgufTensorView<'_>],
        input: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        if tensors.is_empty() {
            return Err("GGUF F32 matvec group requires at least one tensor".into());
        }
        let columns = tensors[0]
            .info
            .shape
            .first()
            .copied()
            .ok_or("GGUF F32 matvec tensor is not a matrix")?;
        if input.len() != columns {
            return Err("GGUF F32 matvec group input length is invalid".into());
        }
        let mut weights = Vec::with_capacity(tensors.len());
        let mut outputs = Vec::with_capacity(tensors.len());
        let mut specs = Vec::with_capacity(tensors.len());
        let mut mapped_bytes = 0_u64;
        let mut scratch_bytes = std::mem::size_of_val(input) as u64;
        for tensor in tensors {
            if tensor.info.ggml_type != 0
                || tensor.info.shape.len() != 2
                || tensor.info.shape[0] != columns
            {
                return Err("GGUF F32 matvec group shapes or types differ".into());
            }
            let rows = tensor.info.shape[1];
            let expected_bytes = rows
                .checked_mul(columns)
                .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
                .ok_or("GGUF F32 matvec group byte length overflows")?;
            if tensor.bytes.len() != expected_bytes {
                return Err("GGUF F32 matvec group tensor is truncated".into());
            }
            let weight = self.device.new_buffer_with_bytes_no_copy(
                tensor.bytes.as_ptr() as *const std::ffi::c_void,
                expected_bytes as u64,
                metal::MTLResourceOptions::StorageModeShared,
                None,
            );
            let output = self.device.new_buffer(
                std::mem::size_of::<f32>() as u64 * rows as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            mapped_bytes = mapped_bytes.saturating_add(expected_bytes as u64);
            scratch_bytes =
                scratch_bytes.saturating_add(std::mem::size_of::<f32>() as u64 * rows as u64);
            weights.push(weight);
            outputs.push(output);
            specs.push(rows);
        }
        self.record_mapped_profile(mapped_bytes, 0, scratch_bytes);
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.queue.new_command_buffer();
        let columns_u32 = u32::try_from(columns)
            .map_err(|_| "GGUF F32 matvec group columns exceed Metal limits")?;
        for ((weight, output), rows) in weights.iter().zip(&outputs).zip(&specs) {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.f32_matvec_pipeline);
            encoder.set_buffer(0, Some(weight), 0);
            encoder.set_buffer(1, Some(&input_buffer), 0);
            encoder.set_buffer(2, Some(output), 0);
            encoder.set_bytes(3, 4, &columns_u32 as *const u32 as *const std::ffi::c_void);
            let rows_u32 = u32::try_from(*rows)
                .map_err(|_| "GGUF F32 matvec group rows exceed Metal limits")?;
            encoder.set_bytes(4, 4, &rows_u32 as *const u32 as *const std::ffi::c_void);
            let threads = self
                .f32_matvec_pipeline
                .max_total_threads_per_threadgroup()
                .clamp(1, 128);
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(*rows as u64, 1, 1),
                metal::MTLSize::new(threads, 1, 1),
            );
            encoder.end_encoding();
        }
        self.commit_and_wait(command_buffer, "grouped GGUF F32 matvec command")?;
        outputs
            .iter()
            .zip(&specs)
            .map(|(output, rows)| {
                Ok(unsafe {
                    std::slice::from_raw_parts(output.contents() as *const f32, *rows).to_vec()
                })
            })
            .collect()
    }

    /// Submit several staged quantized projections against one input in one
    /// command buffer. Staged tensors cannot use the mmap-only grouped helper
    /// above, but they still share the input upload, submission, and wait.
    pub fn gguf_quant_matvec_many_bytes(
        &self,
        tensors: &[(u32, &[u8], usize, usize)],
        input: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        if tensors.is_empty() {
            return Err("staged quantized matvec group requires at least one tensor".into());
        }
        let columns = tensors[0].3;
        if columns == 0 || input.len() != columns {
            return Err("staged quantized matvec group input length is invalid".into());
        }
        let rows4x = qk_rows4x128_enabled()
            && tensors
                .iter()
                .all(|(ggml_type, _, _, _)| qk_rows4x128_for_type(*ggml_type));
        let rows4 = rows4x || qk_rows4_enabled(std::env::var("SI_QK_ROWS4").ok().as_deref());
        let mut weights = Vec::with_capacity(tensors.len());
        let mut outputs = Vec::with_capacity(tensors.len());
        let mut specs = Vec::with_capacity(tensors.len());
        let mut mapped_bytes = 0_u64;
        let mut scratch_bytes = std::mem::size_of_val(input) as u64;
        for (ggml_type, bytes, rows, tensor_columns) in tensors {
            if *rows == 0 || *tensor_columns != columns || !columns.is_multiple_of(256) {
                return Err("staged quantized matvec group shapes differ".into());
            }
            let (block_bytes, pipeline) = match *ggml_type {
                crate::quant::GGML_TYPE_Q4_K => (
                    crate::quant::Q4_K_BLOCK_BYTES,
                    if rows4x {
                        &self.q4_k_matvec_rows4x128_pipeline
                    } else if rows4 {
                        &self.q4_k_matvec_rows4_pipeline
                    } else {
                        &self.q4_k_matvec_pipeline
                    },
                ),
                crate::quant::GGML_TYPE_Q5_K => (
                    crate::quant::Q5_K_BLOCK_BYTES,
                    if rows4 {
                        &self.q5_k_matvec_rows4_pipeline
                    } else {
                        &self.q5_k_matvec_pipeline
                    },
                ),
                crate::quant::GGML_TYPE_Q6_K => (
                    crate::quant::Q6_K_BLOCK_BYTES,
                    if rows4x {
                        &self.q6_k_matvec_rows4x128_pipeline
                    } else if rows4 {
                        &self.q6_k_matvec_rows4_pipeline
                    } else {
                        &self.q6_k_matvec_pipeline
                    },
                ),
                _ => return Err("unsupported staged quantized tensor type".into()),
            };
            let expected_bytes = rows
                .checked_mul(columns / 256)
                .and_then(|blocks| blocks.checked_mul(block_bytes))
                .ok_or("staged quantized matvec group byte length overflows")?;
            if bytes.len() != expected_bytes {
                return Err("staged quantized matvec tensor byte length is invalid".into());
            }
            let weight = self.device.new_buffer_with_bytes_no_copy(
                bytes.as_ptr() as *const std::ffi::c_void,
                bytes.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
                None,
            );
            if weight.length() < bytes.len() as u64 {
                return Err("Metal could not bind staged grouped quantized weights".into());
            }
            let output = self.device.new_buffer(
                std::mem::size_of::<f32>() as u64 * *rows as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            mapped_bytes = mapped_bytes.saturating_add(bytes.len() as u64);
            scratch_bytes =
                scratch_bytes.saturating_add(std::mem::size_of::<f32>() as u64 * *rows as u64);
            specs.push((*ggml_type, *rows, pipeline));
            weights.push(weight);
            outputs.push(output);
        }
        self.record_mapped_profile(mapped_bytes, 0, scratch_bytes);
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let command_buffer = self.queue.new_command_buffer();
        let columns_u32 = u32::try_from(columns)
            .map_err(|_| "staged grouped quantized columns exceed Metal limits")?;
        for ((weight, output), (_, rows, pipeline)) in
            weights.iter().zip(outputs.iter()).zip(specs.iter())
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(weight), 0);
            encoder.set_buffer(1, Some(&input_buffer), 0);
            encoder.set_buffer(2, Some(output), 0);
            encoder.set_bytes(3, 4, &columns_u32 as *const u32 as *const std::ffi::c_void);
            let rows_u32 = u32::try_from(*rows)
                .map_err(|_| "staged grouped quantized rows exceed Metal limits")?;
            encoder.set_bytes(4, 4, &rows_u32 as *const u32 as *const std::ffi::c_void);
            let threads = pipeline.max_total_threads_per_threadgroup().clamp(1, 128);
            let groups = if rows4 { (*rows).div_ceil(4) } else { *rows };
            encoder.dispatch_thread_groups(
                metal::MTLSize::new(groups as u64, 1, 1),
                metal::MTLSize::new(threads, 1, 1),
            );
            encoder.end_encoding();
        }
        self.commit_and_wait(command_buffer, "staged grouped quantized matvec command")?;
        outputs
            .iter()
            .zip(tensors.iter())
            .map(|(output, (_, _, rows, _))| {
                Ok(unsafe {
                    std::slice::from_raw_parts(output.contents() as *const f32, *rows).to_vec()
                })
            })
            .collect()
    }

    pub fn gguf_matvec_tensor_rows(
        &self,
        tensor: &GgufTensorView<'_>,
        row_start: usize,
        row_count: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if tensor.info.ggml_type == 0 {
            self.f32_matvec_tensor_rows(tensor, row_start, row_count, input)
        } else {
            self.gguf_quant_matvec_tensor_rows(tensor, row_start, row_count, input)
        }
    }

    /// Multiply an exact row-aligned invariant-bit-packed BF16 matrix by an
    /// FP32 vector. The Metal kernel reconstructs each value in registers,
    /// avoiding a full decompressed weight allocation.
    pub fn bf16_bitpack_matvec(
        &self,
        packed: &[u8],
        offsets: &[u32],
        rows: usize,
        columns: usize,
        tile_rows: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if rows == 0 || columns == 0 || tile_rows == 0 || input.len() != columns {
            return Err("BF16 bitpack matvec dimensions are invalid".into());
        }
        if packed.len() < 4 {
            return Err("BF16 bitpack payload is too short".into());
        }
        let tile_count = rows.div_ceil(tile_rows);
        if offsets.len() != tile_count + 1 {
            return Err("BF16 bitpack offset count does not match matrix tiles".into());
        }
        let packed_len = u32::try_from(packed.len())
            .map_err(|_| "BF16 bitpack payload exceeds 4 GiB offset range")?;
        if offsets[0] != 0
            || offsets.windows(2).any(|pair| pair[0] > pair[1])
            || offsets.last().copied() != Some(packed_len)
        {
            return Err("BF16 bitpack offsets are not monotonic or complete".into());
        }
        let packed_bytes = u64::try_from(packed.len())
            .map_err(|_| "BF16 bitpack payload length exceeds Metal limits")?;
        let offsets_bytes = u64::try_from(std::mem::size_of_val(offsets))
            .map_err(|_| "BF16 bitpack offset length exceeds Metal limits")?;
        let input_bytes = u64::try_from(std::mem::size_of_val(input))
            .map_err(|_| "BF16 bitpack input length exceeds Metal limits")?;
        let output_bytes = u64::try_from(
            rows.checked_mul(std::mem::size_of::<f32>())
                .ok_or("BF16 bitpack output dimensions overflow")?,
        )
        .map_err(|_| "BF16 bitpack output length exceeds Metal limits")?;
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            input_bytes,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let packed_buffer = self.device.new_buffer_with_bytes_no_copy(
            packed.as_ptr() as *const std::ffi::c_void,
            packed_bytes,
            metal::MTLResourceOptions::StorageModeShared,
            None,
        );
        let offsets_buffer = self.device.new_buffer_with_bytes_no_copy(
            offsets.as_ptr() as *const std::ffi::c_void,
            offsets_bytes,
            metal::MTLResourceOptions::StorageModeShared,
            None,
        );
        if packed_buffer.length() < packed_bytes || offsets_buffer.length() < offsets_bytes {
            return Err("Metal could not bind the mapped BF16 bitpack buffers".into());
        }
        let output_buffer = self
            .device
            .new_buffer(output_bytes, metal::MTLResourceOptions::StorageModeShared);
        self.record_profile(packed_bytes + offsets_bytes, 0, input_bytes + output_bytes);
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let pipeline = self
            .bf16_bitpack_pipeline
            .as_ref()
            .ok_or("BF16 bitpack Metal probe is disabled; set SI_LOSSLESS_GPU=1")?;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&packed_buffer), 0);
        encoder.set_buffer(1, Some(&offsets_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        let rows = u32::try_from(rows).map_err(|_| "BF16 bitpack rows exceed Metal limits")?;
        let columns =
            u32::try_from(columns).map_err(|_| "BF16 bitpack columns exceed Metal limits")?;
        let tile_rows =
            u32::try_from(tile_rows).map_err(|_| "BF16 bitpack tile rows exceed Metal limits")?;
        encoder.set_bytes(4, 4, &rows as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &columns as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(6, 4, &tile_rows as *const u32 as *const std::ffi::c_void);
        let threads = pipeline.max_total_threads_per_threadgroup().clamp(1, 128);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(u64::from(rows), 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "BF16 bitpack matvec command")?;
        // SAFETY: the command buffer completed and output contains `rows`
        // contiguous FP32 values.
        Ok(unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, rows as usize)
                .to_vec()
        })
    }

    /// Bind mapped Safetensors bytes directly as a shared Metal buffer. The
    /// command path is synchronous, so the input slice remains alive until the
    /// GPU has finished reading it and the no-copy buffer is released.
    fn map_bf16_weight(&self, weight_bytes: &[u8]) -> Result<Bf16Weight, String> {
        validate_bf16_weight_bytes(weight_bytes)?;
        let buffer = self.device.new_buffer_with_bytes_no_copy(
            weight_bytes.as_ptr() as *const std::ffi::c_void,
            weight_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
            None,
        );
        if buffer.length() < weight_bytes.len() as u64 {
            return Err("Metal could not bind the mapped BF16 weight bytes".into());
        }
        Ok(Bf16Weight {
            buffer,
            offset: 0,
            bytes: weight_bytes.len() as u64,
            persistent: false,
            mapped: true,
        })
    }

    /// Bind one file-backed tensor through an aligned no-copy Metal view. The
    /// view is intentionally scoped to this operation: retaining every tensor
    /// buffer would make Metal count the whole model as resident and defeat
    /// the low-memory streaming target.
    fn map_bf16_tensor_weight(&self, tensor: &TensorView<'_>) -> Result<Bf16Weight, String> {
        self.map_bf16_mapped_weight(tensor.bytes, tensor.backing)
    }

    fn map_bf16_mapped_weight(
        &self,
        weight_bytes: &[u8],
        backing: &[u8],
    ) -> Result<Bf16Weight, String> {
        validate_bf16_weight_bytes(weight_bytes)?;
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size <= 0 {
            return Err("could not determine host page size".into());
        }
        let page_size = page_size as usize;
        let backing_ptr = backing.as_ptr() as usize;
        if !backing_ptr.is_multiple_of(page_size) {
            return self.map_bf16_weight(weight_bytes);
        }
        let tensor_ptr = weight_bytes.as_ptr() as usize;
        let (base_ptr, _, buffer_len) =
            aligned_weight_range(backing_ptr, tensor_ptr, weight_bytes.len(), page_size)?;
        let backing_end = backing_ptr
            .checked_add(backing.len())
            .ok_or("tensor backing range overflows")?;
        let tensor_end = tensor_ptr
            .checked_add(weight_bytes.len())
            .ok_or("tensor range overflows")?;
        let offset = tensor_ptr
            .checked_sub(base_ptr)
            .ok_or("tensor range is before its backing mapping")?;
        if tensor_ptr < backing_ptr || tensor_end > backing_end || base_ptr < backing_ptr {
            return Err("tensor bytes are outside their backing mapping".into());
        }
        let buffer = self.device.new_buffer_with_bytes_no_copy(
            base_ptr as *const std::ffi::c_void,
            buffer_len as u64,
            metal::MTLResourceOptions::StorageModeShared,
            None,
        );
        if buffer.length() < buffer_len as u64 {
            return Err("Metal could not bind the aligned BF16 weight bytes".into());
        }
        Ok(Bf16Weight {
            buffer,
            offset: offset as u64,
            bytes: weight_bytes.len() as u64,
            persistent: false,
            mapped: true,
        })
    }

    pub fn bf16_matvec_buffer(
        &self,
        weight: &Bf16Weight,
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let expected_bytes = rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or("BF16 matrix dimensions overflow")?;
        if rows == 0
            || columns == 0
            || input.len() != columns
            || weight.bytes != expected_bytes as u64
        {
            return Err("BF16 matvec dimensions or byte length are invalid".into());
        }
        self.bf16_matvec_many_buffer(&[(weight, rows, columns)], input)
            .map(|mut outputs| outputs.remove(0))
    }

    pub fn bf16_matvec_buffer_async(
        &self,
        weight: &Bf16Weight,
        rows: usize,
        columns: usize,
        input: &[f32],
    ) -> Result<PendingMatvec, String> {
        let expected_bytes = rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or("BF16 matrix dimensions overflow")?;
        if rows == 0
            || columns == 0
            || input.len() != columns
            || weight.bytes != expected_bytes as u64
        {
            return Err("BF16 matvec dimensions or byte length are invalid".into());
        }
        self.bf16_matvec_many_buffer_async(&[(weight, rows, columns)], input)
    }

    /// Bind several mapped Safetensors matrices and execute them in one
    /// command buffer. The no-copy bindings stay alive until the single wait,
    /// removing per-projection command-buffer synchronization without adding a
    /// resident weight allocation.
    pub fn bf16_matvec_many_tensors(
        &self,
        tensors: &[&TensorView<'_>],
        input: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        let shapes = validate_matvec_tensor_batch(tensors, input.len())?;
        let weights = tensors
            .iter()
            .map(|tensor| self.map_bf16_tensor_weight(tensor))
            .collect::<Result<Vec<_>, _>>()?;
        let matrices = weights
            .iter()
            .zip(shapes)
            .map(|(weight, (rows, columns))| (weight, rows, columns))
            .collect::<Vec<_>>();
        self.bf16_matvec_many_buffer(&matrices, input)
    }

    pub fn bf16_matvec_many_tensors_async(
        &self,
        tensors: &[&TensorView<'_>],
        input: &[f32],
    ) -> Result<PendingMatvec, String> {
        let shapes = validate_matvec_tensor_batch(tensors, input.len())?;
        let weights = tensors
            .iter()
            .map(|tensor| self.map_bf16_tensor_weight(tensor))
            .collect::<Result<Vec<_>, _>>()?;
        let matrices = weights
            .iter()
            .zip(shapes)
            .map(|(weight, (rows, columns))| (weight, rows, columns))
            .collect::<Vec<_>>();
        self.bf16_matvec_many_buffer_async(&matrices, input)
    }

    /// Encode several independent matrix-vector products against one input in
    /// a single command buffer. Q/K/V and gate/up use this to avoid a host
    /// round-trip and command-buffer wait between projections.
    pub fn bf16_matvec_many_buffer(
        &self,
        matrices: &[(&Bf16Weight, usize, usize)],
        input: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        self.bf16_matvec_many_buffer_async(matrices, input)?.wait()
    }

    /// Execute one matrix against several candidate vectors in one dispatch.
    /// Inputs and outputs are batch-major: `[batch * columns]` and
    /// `[batch][rows]`. This is the lossless SI-004 primitive: the streamed
    /// BF16 matrix is bound once, while each row's weights are reused across
    /// up to eight candidate states.
    pub fn bf16_matmul_many_buffer(
        &self,
        weight: &Bf16Weight,
        rows: usize,
        columns: usize,
        batch: usize,
        inputs: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        validate_matmul_many_shape(
            rows,
            columns,
            batch,
            usize::try_from(weight.bytes).map_err(|_| "BF16 weight length exceeds usize")?,
            inputs.len(),
        )?;
        let pipeline = self
            .batched_matmul_pipeline
            .as_ref()
            .ok_or("batched matmul is disabled; set SI_VERIFY_MANY=1")?;
        let input_bytes = std::mem::size_of_val(inputs) as u64;
        let output_elements = batch
            .checked_mul(rows)
            .ok_or("batched matmul output dimensions overflow")?;
        let output_bytes = std::mem::size_of::<f32>()
            .checked_mul(output_elements)
            .ok_or("batched matmul output byte length overflows")?
            as u64;
        let input_buffer = self.device.new_buffer_with_data(
            inputs.as_ptr() as *const std::ffi::c_void,
            input_bytes,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self
            .device
            .new_buffer(output_bytes, metal::MTLResourceOptions::StorageModeShared);
        if weight.mapped {
            self.record_mapped_profile(weight.bytes, 0, input_bytes + output_bytes);
        } else {
            self.record_profile(
                if weight.persistent { 0 } else { weight.bytes },
                0,
                input_bytes + output_bytes,
            );
        }
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_bf16_matmul_many(
            encoder,
            pipeline,
            weight,
            &input_buffer,
            &output_buffer,
            rows,
            columns,
            batch,
        )?;
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "batched matmul command")?;
        // SAFETY: command completed and output contains batch * rows f32s.
        let flat = unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, output_elements)
        };
        Ok(flat
            .chunks_exact(rows)
            .map(|candidate| candidate.to_vec())
            .collect())
    }

    pub fn bf16_matmul_many_tensor(
        &self,
        tensor: &TensorView<'_>,
        batch: usize,
        inputs: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        if tensor.info.dtype != "BF16" || tensor.info.shape.len() != 2 {
            return Err("batched matmul requires a rank-2 BF16 tensor".into());
        }
        let weight = self.map_bf16_tensor_weight(tensor)?;
        self.bf16_matmul_many_buffer(
            &weight,
            tensor.info.shape[0],
            tensor.info.shape[1],
            batch,
            inputs,
        )
    }

    /// Execute Q/K/V projections for several candidate hidden states in one
    /// fused dispatch. Each projection's output is returned batch-major as
    /// `[candidate][row]`, matching `bf16_matmul_many_buffer`.
    pub fn bf16_fused_qkv_many_tensors(
        &self,
        tensors: &[&TensorView<'_>],
        batch: usize,
        inputs: &[f32],
    ) -> Result<Vec<Vec<Vec<f32>>>, String> {
        if tensors.len() != 3 {
            return Err("batched fused QKV requires exactly three tensors".into());
        }
        let shapes = validate_matvec_tensor_many(tensors, batch, inputs.len())?;
        let weights = tensors
            .iter()
            .map(|tensor| self.map_bf16_tensor_weight(tensor))
            .collect::<Result<Vec<_>, _>>()?;
        let matrices = weights
            .iter()
            .zip(shapes)
            .map(|(weight, (rows, columns))| (weight, rows, columns))
            .collect::<Vec<_>>();
        self.bf16_fused_many_buffer(&matrices, batch, inputs, BatchedFusedProjectionKind::Qkv)
    }

    /// Execute gate/up projections for several candidate hidden states in one
    /// fused dispatch. Each projection's output is returned batch-major.
    pub fn bf16_fused_gate_up_many_tensors(
        &self,
        tensors: &[&TensorView<'_>],
        batch: usize,
        inputs: &[f32],
    ) -> Result<Vec<Vec<Vec<f32>>>, String> {
        if tensors.len() != 2 {
            return Err("batched fused gate/up requires exactly two tensors".into());
        }
        let shapes = validate_matvec_tensor_many(tensors, batch, inputs.len())?;
        let weights = tensors
            .iter()
            .map(|tensor| self.map_bf16_tensor_weight(tensor))
            .collect::<Result<Vec<_>, _>>()?;
        let matrices = weights
            .iter()
            .zip(shapes)
            .map(|(weight, (rows, columns))| (weight, rows, columns))
            .collect::<Vec<_>>();
        self.bf16_fused_many_buffer(&matrices, batch, inputs, BatchedFusedProjectionKind::GateUp)
    }

    /// Execute a batched fused projection against already-resident weights.
    /// This remains synchronous so the operation-scoped mapped buffers can be
    /// released immediately after the command completes.
    pub fn bf16_fused_qkv_many_buffer(
        &self,
        matrices: &[(&Bf16Weight, usize, usize)],
        batch: usize,
        inputs: &[f32],
    ) -> Result<Vec<Vec<Vec<f32>>>, String> {
        self.bf16_fused_many_buffer(matrices, batch, inputs, BatchedFusedProjectionKind::Qkv)
    }

    pub fn bf16_fused_gate_up_many_buffer(
        &self,
        matrices: &[(&Bf16Weight, usize, usize)],
        batch: usize,
        inputs: &[f32],
    ) -> Result<Vec<Vec<Vec<f32>>>, String> {
        self.bf16_fused_many_buffer(matrices, batch, inputs, BatchedFusedProjectionKind::GateUp)
    }

    fn bf16_fused_many_buffer(
        &self,
        matrices: &[(&Bf16Weight, usize, usize)],
        batch: usize,
        inputs: &[f32],
        kind: BatchedFusedProjectionKind,
    ) -> Result<Vec<Vec<Vec<f32>>>, String> {
        let expected_matrices = match kind {
            BatchedFusedProjectionKind::Qkv => 3,
            BatchedFusedProjectionKind::GateUp => 2,
        };
        if matrices.len() != expected_matrices {
            return Err(format!(
                "batched fused projection requires exactly {expected_matrices} matrices"
            ));
        }
        if inputs.is_empty() || !inputs.len().is_multiple_of(batch.max(1)) {
            return Err("batched fused projection input length is invalid".into());
        }
        let columns = inputs.len() / batch.max(1);
        let shapes = matrices
            .iter()
            .map(|(_, rows, columns)| (*rows, *columns))
            .collect::<Vec<_>>();
        validate_fused_projection_many_shapes(&shapes, batch, columns)?;
        for (weight, rows, matrix_columns) in matrices {
            let expected_bytes = rows
                .checked_mul(*matrix_columns)
                .and_then(|elements| elements.checked_mul(2))
                .ok_or("batched fused projection weight dimensions overflow")?;
            if weight.bytes != expected_bytes as u64 {
                return Err("batched fused projection weight byte length is invalid".into());
            }
        }
        let pipeline = match kind {
            BatchedFusedProjectionKind::Qkv => self.batched_fused_qkv_pipeline.as_ref(),
            BatchedFusedProjectionKind::GateUp => self.batched_fused_gate_up_pipeline.as_ref(),
        }
        .ok_or("batched fused projection is disabled; set SI_VERIFY_MANY=1")?;
        if pipeline.max_total_threads_per_threadgroup() < 32 {
            return Err("batched fused projection requires one 32-thread SIMD group".into());
        }

        let input_buffer = self.device.new_buffer_with_data(
            inputs.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(inputs) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffers = matrices
            .iter()
            .map(|(_, rows, _)| {
                let elements = batch
                    .checked_mul(*rows)
                    .ok_or("batched fused projection output dimensions overflow")?;
                let bytes = elements
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or("batched fused projection output byte length overflows")?;
                Ok(self
                    .device
                    .new_buffer(bytes as u64, metal::MTLResourceOptions::StorageModeShared))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let scratch_bytes = std::mem::size_of_val(inputs) as u64
            + output_buffers
                .iter()
                .map(|output| output.length())
                .sum::<u64>();
        let mut mapped_weight_bytes = 0_u64;
        let mut active_weight_bytes = 0_u64;
        for (weight, _, _) in matrices {
            if !weight.persistent {
                active_weight_bytes = active_weight_bytes.saturating_add(weight.bytes);
                if weight.mapped {
                    mapped_weight_bytes = mapped_weight_bytes.saturating_add(weight.bytes);
                }
            }
        }
        if mapped_weight_bytes > 0 {
            self.record_mapped_profile(mapped_weight_bytes, 0, scratch_bytes);
        } else {
            self.record_profile(active_weight_bytes, 0, scratch_bytes);
        }

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        for (index, (weight, _, _)) in matrices.iter().enumerate() {
            encoder.set_buffer(index as u64, Some(&weight.buffer), weight.offset);
        }
        let input_index = expected_matrices as u64;
        encoder.set_buffer(input_index, Some(&input_buffer), 0);
        for (index, output) in output_buffers.iter().enumerate() {
            encoder.set_buffer(input_index + 1 + index as u64, Some(output), 0);
        }
        let columns = u32::try_from(columns)
            .map_err(|_| "batched fused projection columns exceed Metal limits")?;
        let rows_u32 = matrices
            .iter()
            .map(|(_, rows, _)| u32::try_from(*rows))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| "batched fused projection rows exceed Metal limits")?;
        let batch_u32 = u32::try_from(batch)
            .map_err(|_| "batched fused projection batch exceeds Metal limits")?;
        match kind {
            BatchedFusedProjectionKind::Qkv => {
                encoder.set_bytes(7, 4, &columns as *const u32 as *const std::ffi::c_void);
                encoder.set_bytes(8, 4, &rows_u32[0] as *const u32 as *const std::ffi::c_void);
                encoder.set_bytes(9, 4, &rows_u32[1] as *const u32 as *const std::ffi::c_void);
                encoder.set_bytes(10, 4, &rows_u32[2] as *const u32 as *const std::ffi::c_void);
                encoder.set_bytes(11, 4, &batch_u32 as *const u32 as *const std::ffi::c_void);
            }
            BatchedFusedProjectionKind::GateUp => {
                encoder.set_bytes(5, 4, &columns as *const u32 as *const std::ffi::c_void);
                encoder.set_bytes(6, 4, &rows_u32[0] as *const u32 as *const std::ffi::c_void);
                encoder.set_bytes(7, 4, &rows_u32[1] as *const u32 as *const std::ffi::c_void);
                encoder.set_bytes(8, 4, &batch_u32 as *const u32 as *const std::ffi::c_void);
            }
        }
        let total_rows = rows_u32.iter().map(|rows| u64::from(*rows)).sum::<u64>();
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(total_rows, 1, 1),
            metal::MTLSize::new(32, 1, 1),
        );
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "batched fused projection command")?;

        output_buffers
            .iter()
            .zip(matrices.iter().map(|(_, rows, _)| *rows))
            .map(|(output, rows)| {
                let elements = batch
                    .checked_mul(rows)
                    .ok_or("batched fused projection output dimensions overflow")?;
                // SAFETY: the command completed and output contains batch * rows f32s.
                let flat = unsafe {
                    std::slice::from_raw_parts(output.contents() as *const f32, elements)
                };
                Ok(flat
                    .chunks_exact(rows)
                    .map(|candidate| candidate.to_vec())
                    .collect::<Vec<_>>())
            })
            .collect()
    }

    /// Submit several independent matrix-vector products without waiting for
    /// the GPU. The returned handle owns the command-buffer resources until
    /// `wait`, so callers may stage the next work item before collecting the
    /// outputs. This is the primitive used by the opt-in async runtime path.
    pub fn bf16_matvec_many_buffer_async(
        &self,
        matrices: &[(&Bf16Weight, usize, usize)],
        input: &[f32],
    ) -> Result<PendingMatvec, String> {
        if matrices.is_empty() {
            return Err("batched BF16 matvec requires at least one matrix".into());
        }
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let mut outputs = Vec::with_capacity(matrices.len());
        let mut mapped_weight_bytes = 0_u64;
        let mut active_weight_bytes = 0_u64;
        let mut scratch_bytes = std::mem::size_of_val(input) as u64;
        for (weight, rows, columns) in matrices {
            let expected_bytes = rows
                .checked_mul(*columns)
                .and_then(|elements| elements.checked_mul(2))
                .ok_or("BF16 matrix dimensions overflow")?;
            if *rows == 0
                || *columns == 0
                || input.len() != *columns
                || weight.bytes != expected_bytes as u64
            {
                return Err("BF16 matvec dimensions or byte length are invalid".into());
            }
            outputs.push(self.device.new_buffer(
                std::mem::size_of::<f32>() as u64 * *rows as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));
            scratch_bytes =
                scratch_bytes.saturating_add(std::mem::size_of::<f32>() as u64 * *rows as u64);
            if !weight.persistent {
                active_weight_bytes = active_weight_bytes.saturating_add(weight.bytes);
                if weight.mapped {
                    mapped_weight_bytes = mapped_weight_bytes.saturating_add(weight.bytes);
                }
            }
        }
        if mapped_weight_bytes > 0 {
            self.record_mapped_profile(mapped_weight_bytes, 0, scratch_bytes);
        } else {
            self.record_profile(active_weight_bytes, 0, scratch_bytes);
        }
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        for ((weight, rows, columns), output) in matrices.iter().zip(&outputs) {
            self.encode_bf16_matvec(encoder, weight, &input_buffer, output, *rows, *columns)?;
        }
        encoder.end_encoding();
        command_buffer.commit();
        self.command_stats.record_async_submission();
        let command_buffer = command_buffer.to_owned();
        let weights = matrices
            .iter()
            .map(|(weight, _, _)| weight.buffer.clone())
            .collect();
        let outputs = outputs
            .into_iter()
            .zip(matrices.iter().map(|(_, rows, _)| *rows))
            .collect();
        Ok(PendingMatvec {
            command_buffer,
            _input: input_buffer,
            _weights: weights,
            _keepalive: Vec::new(),
            outputs,
            stats: Arc::clone(&self.command_stats),
        })
    }

    /// Execute Q/K/V projections with one fused dispatch. The three output
    /// vectors remain separate so the existing CPU attention path can consume
    /// them without a layout conversion.
    pub fn bf16_fused_qkv_tensors(
        &self,
        tensors: &[&TensorView<'_>],
        input: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        if tensors.len() != 3 {
            return Err("fused QKV requires exactly three tensors".into());
        }
        let shapes = validate_matvec_tensor_batch(tensors, input.len())?;
        let weights = tensors
            .iter()
            .map(|tensor| self.map_bf16_tensor_weight(tensor))
            .collect::<Result<Vec<_>, _>>()?;
        let matrices = weights
            .iter()
            .zip(shapes)
            .map(|(weight, (rows, columns))| (weight, rows, columns))
            .collect::<Vec<_>>();
        self.bf16_fused_qkv_buffer(&matrices, input)
    }

    pub fn bf16_fused_qkv_tensors_async(
        &self,
        tensors: &[&TensorView<'_>],
        input: &[f32],
    ) -> Result<PendingMatvec, String> {
        if tensors.len() != 3 {
            return Err("fused QKV requires exactly three tensors".into());
        }
        let shapes = validate_matvec_tensor_batch(tensors, input.len())?;
        let weights = tensors
            .iter()
            .map(|tensor| self.map_bf16_tensor_weight(tensor))
            .collect::<Result<Vec<_>, _>>()?;
        let matrices = weights
            .iter()
            .zip(shapes)
            .map(|(weight, (rows, columns))| (weight, rows, columns))
            .collect::<Vec<_>>();
        self.bf16_fused_qkv_buffer_async(&matrices, input)
    }

    pub fn bf16_fused_qkv_bytes(
        &self,
        matrices: &[(&[u8], usize, usize)],
        input: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        if matrices.len() != 3 {
            return Err("fused QKV requires exactly three matrices".into());
        }
        let weights = matrices
            .iter()
            .map(|(bytes, _, _)| self.map_bf16_weight(bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let buffers = weights
            .iter()
            .zip(matrices)
            .map(|(weight, (_, rows, columns))| (weight, *rows, *columns))
            .collect::<Vec<_>>();
        self.bf16_fused_qkv_buffer(&buffers, input)
    }

    /// Execute gate and up projections with one fused dispatch. The output
    /// vectors are kept separate for the existing SiLU/elementwise path.
    pub fn bf16_fused_gate_up_tensors(
        &self,
        tensors: &[&TensorView<'_>],
        input: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        if tensors.len() != 2 {
            return Err("fused gate/up requires exactly two tensors".into());
        }
        let shapes = validate_matvec_tensor_batch(tensors, input.len())?;
        let weights = tensors
            .iter()
            .map(|tensor| self.map_bf16_tensor_weight(tensor))
            .collect::<Result<Vec<_>, _>>()?;
        let matrices = weights
            .iter()
            .zip(shapes)
            .map(|(weight, (rows, columns))| (weight, rows, columns))
            .collect::<Vec<_>>();
        self.bf16_fused_gate_up_buffer(&matrices, input)
    }

    pub fn bf16_fused_gate_up_tensors_async(
        &self,
        tensors: &[&TensorView<'_>],
        input: &[f32],
    ) -> Result<PendingMatvec, String> {
        if tensors.len() != 2 {
            return Err("fused gate/up requires exactly two tensors".into());
        }
        let shapes = validate_matvec_tensor_batch(tensors, input.len())?;
        let weights = tensors
            .iter()
            .map(|tensor| self.map_bf16_tensor_weight(tensor))
            .collect::<Result<Vec<_>, _>>()?;
        let matrices = weights
            .iter()
            .zip(shapes)
            .map(|(weight, (rows, columns))| (weight, rows, columns))
            .collect::<Vec<_>>();
        self.bf16_fused_gate_up_buffer_async(&matrices, input)
    }

    pub fn bf16_fused_gate_up_bytes(
        &self,
        matrices: &[(&[u8], usize, usize)],
        input: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        if matrices.len() != 2 {
            return Err("fused gate/up requires exactly two matrices".into());
        }
        let weights = matrices
            .iter()
            .map(|(bytes, _, _)| self.map_bf16_weight(bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let buffers = weights
            .iter()
            .zip(matrices)
            .map(|(weight, (_, rows, columns))| (weight, *rows, *columns))
            .collect::<Vec<_>>();
        self.bf16_fused_gate_up_buffer(&buffers, input)
    }

    pub fn bf16_fused_qkv_buffer(
        &self,
        matrices: &[(&Bf16Weight, usize, usize)],
        input: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        if matrices.len() != 3 {
            return Err("fused QKV requires exactly three matrices".into());
        }
        validate_fused_projection_shapes(
            &matrices
                .iter()
                .map(|(_, rows, columns)| (*rows, *columns))
                .collect::<Vec<_>>(),
            input.len(),
        )?;
        self.bf16_fused_qkv_buffer_async(matrices, input)?.wait()
    }

    pub fn bf16_fused_qkv_buffer_async(
        &self,
        matrices: &[(&Bf16Weight, usize, usize)],
        input: &[f32],
    ) -> Result<PendingMatvec, String> {
        if matrices.len() != 3 {
            return Err("fused QKV requires exactly three matrices".into());
        }
        validate_fused_projection_shapes(
            &matrices
                .iter()
                .map(|(_, rows, columns)| (*rows, *columns))
                .collect::<Vec<_>>(),
            input.len(),
        )?;
        self.submit_fused_qkv(matrices, input)
    }

    pub fn bf16_fused_gate_up_buffer(
        &self,
        matrices: &[(&Bf16Weight, usize, usize)],
        input: &[f32],
    ) -> Result<Vec<Vec<f32>>, String> {
        if matrices.len() != 2 {
            return Err("fused gate/up requires exactly two matrices".into());
        }
        validate_fused_projection_shapes(
            &matrices
                .iter()
                .map(|(_, rows, columns)| (*rows, *columns))
                .collect::<Vec<_>>(),
            input.len(),
        )?;
        self.bf16_fused_gate_up_buffer_async(matrices, input)?
            .wait()
    }

    pub fn bf16_fused_gate_up_buffer_async(
        &self,
        matrices: &[(&Bf16Weight, usize, usize)],
        input: &[f32],
    ) -> Result<PendingMatvec, String> {
        if matrices.len() != 2 {
            return Err("fused gate/up requires exactly two matrices".into());
        }
        validate_fused_projection_shapes(
            &matrices
                .iter()
                .map(|(_, rows, columns)| (*rows, *columns))
                .collect::<Vec<_>>(),
            input.len(),
        )?;
        self.submit_fused_gate_up(matrices, input)
    }

    pub fn bf16_fused_qkv_owned_bytes_async(
        &self,
        matrices: Vec<(Vec<u8>, usize, usize)>,
        input: &[f32],
    ) -> Result<PendingMatvec, String> {
        if matrices.len() != 3 {
            return Err("fused QKV requires exactly three matrices".into());
        }
        let refs = matrices
            .iter()
            .map(|(bytes, rows, columns)| (bytes.as_slice(), *rows, *columns))
            .collect::<Vec<_>>();
        let weights = refs
            .iter()
            .map(|(bytes, _, _)| self.map_bf16_weight(bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let buffers = weights
            .iter()
            .zip(&refs)
            .map(|(weight, (_, rows, columns))| (weight, *rows, *columns))
            .collect::<Vec<_>>();
        let keepalive = matrices
            .into_iter()
            .map(|(bytes, _, _)| bytes)
            .collect::<Vec<_>>();
        self.bf16_fused_qkv_buffer_async(&buffers, input)
            .map(|pending| pending.with_keepalive(keepalive))
    }

    pub fn bf16_fused_gate_up_owned_bytes_async(
        &self,
        matrices: Vec<(Vec<u8>, usize, usize)>,
        input: &[f32],
    ) -> Result<PendingMatvec, String> {
        if matrices.len() != 2 {
            return Err("fused gate/up requires exactly two matrices".into());
        }
        let refs = matrices
            .iter()
            .map(|(bytes, rows, columns)| (bytes.as_slice(), *rows, *columns))
            .collect::<Vec<_>>();
        let weights = refs
            .iter()
            .map(|(bytes, _, _)| self.map_bf16_weight(bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let buffers = weights
            .iter()
            .zip(&refs)
            .map(|(weight, (_, rows, columns))| (weight, *rows, *columns))
            .collect::<Vec<_>>();
        let keepalive = matrices
            .into_iter()
            .map(|(bytes, _, _)| bytes)
            .collect::<Vec<_>>();
        self.bf16_fused_gate_up_buffer_async(&buffers, input)
            .map(|pending| pending.with_keepalive(keepalive))
    }

    /// Keep the attention-side projection graph on Metal until the output
    /// projection is complete. QKV, Q/K norms, RoPE, attention, and O-proj
    /// share one command buffer, so the host observes only the final hidden
    /// vector plus the new KV token.
    pub fn chained_qkv_attention_o_tensors(
        &self,
        request: ChainedAttentionTensorRequest<'_, '_>,
    ) -> Result<ChainedAttentionOutput, String> {
        let ChainedAttentionTensorRequest {
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            q_norm_bytes,
            k_norm_bytes,
            input,
            key_cache,
            value_cache,
            config,
        } = request;
        let tensors = [q_tensor, k_tensor, v_tensor, o_tensor];
        if tensors
            .iter()
            .any(|tensor| tensor.info.dtype != "BF16" || tensor.info.shape.len() != 2)
        {
            return Err("chained attention projections require rank-2 BF16 tensors".into());
        }
        let weights = tensors
            .iter()
            .map(|tensor| self.map_bf16_tensor_weight(tensor))
            .collect::<Result<Vec<_>, _>>()?;
        self.chained_qkv_attention_o_buffers(ChainedAttentionBufferRequest {
            q: (&weights[0], q_tensor.info.shape[0], q_tensor.info.shape[1]),
            k: (&weights[1], k_tensor.info.shape[0], k_tensor.info.shape[1]),
            v: (&weights[2], v_tensor.info.shape[0], v_tensor.info.shape[1]),
            o: (&weights[3], o_tensor.info.shape[0], o_tensor.info.shape[1]),
            q_norm_bytes,
            k_norm_bytes,
            input,
            key_cache,
            value_cache,
            config,
        })
    }

    pub fn chained_qkv_attention_o_buffers(
        &self,
        request: ChainedAttentionBufferRequest<'_>,
    ) -> Result<ChainedAttentionOutput, String> {
        let ChainedAttentionBufferRequest {
            q,
            k,
            v,
            o,
            q_norm_bytes,
            k_norm_bytes,
            input,
            key_cache,
            value_cache,
            config,
        } = request;
        let ChainedAttentionConfig {
            query_heads,
            key_value_heads,
            head_dim,
            cached_tokens,
            cache_capacity_tokens,
            position,
            rope_theta,
            epsilon,
        } = config;
        if query_heads == 0
            || key_value_heads == 0
            || head_dim == 0
            || !query_heads.is_multiple_of(key_value_heads)
            || q_norm_bytes.len() != head_dim * 2
            || k_norm_bytes.len() != head_dim * 2
            || cached_tokens >= cache_capacity_tokens
        {
            return Err("chained attention dimensions are invalid".into());
        }
        let q_rows = query_heads
            .checked_mul(head_dim)
            .ok_or("chained Q dimensions overflow")?;
        let kv_rows = key_value_heads
            .checked_mul(head_dim)
            .ok_or("chained KV dimensions overflow")?;
        let cache_elements = kv_rows
            .checked_mul(cache_capacity_tokens)
            .ok_or("chained cache dimensions overflow")?;
        if key_cache.len() != cache_elements || value_cache.len() != cache_elements {
            return Err("chained cache dimensions are invalid".into());
        }
        for (weight, rows, columns) in [q, k, v] {
            let expected_bytes = rows
                .checked_mul(columns)
                .and_then(|elements| elements.checked_mul(2))
                .ok_or("chained projection dimensions overflow")?;
            if rows == 0
                || columns == 0
                || columns != input.len()
                || weight.bytes != expected_bytes as u64
            {
                return Err("chained projection dimensions or byte length are invalid".into());
            }
        }
        let expected_o_bytes =
            o.1.checked_mul(o.2)
                .and_then(|elements| elements.checked_mul(2))
                .ok_or("chained output projection dimensions overflow")?;
        if o.1 == 0
            || o.1 != input.len()
            || o.2 != q_rows
            || o.0.bytes != expected_o_bytes as u64
            || q.1 != q_rows
            || k.1 != kv_rows
            || v.1 != kv_rows
        {
            return Err("chained projection shapes do not match attention dimensions".into());
        }

        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let q_norm_buffer = self.device.new_buffer_with_data(
            q_norm_bytes.as_ptr() as *const std::ffi::c_void,
            q_norm_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let k_norm_buffer = self.device.new_buffer_with_data(
            k_norm_bytes.as_ptr() as *const std::ffi::c_void,
            k_norm_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let key_cache_buffer = self.device.new_buffer_with_data(
            key_cache.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(key_cache) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let value_cache_buffer = self.device.new_buffer_with_data(
            value_cache.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(value_cache) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let q_output = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64 * q_rows as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let k_output = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64 * kv_rows as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let v_output = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64 * kv_rows as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let q_normed = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64 * q_rows as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let k_normed = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64 * kv_rows as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let q_rope = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64 * q_rows as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let k_rope = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64 * kv_rows as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let attended = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64 * q_rows as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let projected = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64 * o.1 as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let mut active_weight_bytes = 0_u64;
        let mut mapped_weight_bytes = 0_u64;
        for (weight, _, _) in [q, k, v, o] {
            if !weight.persistent {
                active_weight_bytes = active_weight_bytes.saturating_add(weight.bytes);
                if weight.mapped {
                    mapped_weight_bytes = mapped_weight_bytes.saturating_add(weight.bytes);
                }
            }
        }
        let scratch_bytes = std::mem::size_of_val(input) as u64
            + q_norm_bytes.len() as u64
            + k_norm_bytes.len() as u64
            + std::mem::size_of_val(key_cache) as u64
            + std::mem::size_of_val(value_cache) as u64
            + q_output.length()
            + k_output.length()
            + v_output.length()
            + q_normed.length()
            + k_normed.length()
            + q_rope.length()
            + k_rope.length()
            + attended.length()
            + projected.length();
        if mapped_weight_bytes > 0 {
            self.record_mapped_profile(mapped_weight_bytes, 0, scratch_bytes);
        } else {
            self.record_profile(active_weight_bytes, 0, scratch_bytes);
        }

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.bf16_fused_qkv_pipeline);
        for (index, (weight, _, _)) in [q, k, v].into_iter().enumerate() {
            encoder.set_buffer(index as u64, Some(&weight.buffer), weight.offset);
        }
        encoder.set_buffer(3, Some(&input_buffer), 0);
        encoder.set_buffer(4, Some(&q_output), 0);
        encoder.set_buffer(5, Some(&k_output), 0);
        encoder.set_buffer(6, Some(&v_output), 0);
        let columns = u32::try_from(q.2).map_err(|_| "chained QKV columns exceed Metal limits")?;
        let q_rows_u32 = u32::try_from(q.1).map_err(|_| "chained Q rows exceed Metal limits")?;
        let k_rows_u32 = u32::try_from(k.1).map_err(|_| "chained K rows exceed Metal limits")?;
        let v_rows_u32 = u32::try_from(v.1).map_err(|_| "chained V rows exceed Metal limits")?;
        encoder.set_bytes(7, 4, &columns as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(8, 4, &q_rows_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(9, 4, &k_rows_u32 as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(10, 4, &v_rows_u32 as *const u32 as *const std::ffi::c_void);
        let qkv_threads = self
            .bf16_fused_qkv_pipeline
            .max_total_threads_per_threadgroup()
            .clamp(1, 128);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new((q_rows_u32 + k_rows_u32 + v_rows_u32) as u64, 1, 1),
            metal::MTLSize::new(qkv_threads, 1, 1),
        );
        self.encode_rms_norm_bf16_heads(
            encoder,
            &q_output,
            &q_norm_buffer,
            &q_normed,
            NormParams {
                heads: query_heads,
                head_dim,
                epsilon,
            },
        )?;
        self.encode_rms_norm_bf16_heads(
            encoder,
            &k_output,
            &k_norm_buffer,
            &k_normed,
            NormParams {
                heads: key_value_heads,
                head_dim,
                epsilon,
            },
        )?;
        self.encode_rope(
            encoder,
            &q_normed,
            &q_rope,
            RopeParams {
                heads: query_heads,
                head_dim,
                position,
                theta: rope_theta,
            },
        )?;
        self.encode_rope(
            encoder,
            &k_normed,
            &k_rope,
            RopeParams {
                heads: key_value_heads,
                head_dim,
                position,
                theta: rope_theta,
            },
        )?;
        self.encode_attention_decode(
            encoder,
            AttentionBufferRefs {
                query: &q_rope,
                key_cache: &key_cache_buffer,
                value_cache: &value_cache_buffer,
                new_keys: &k_rope,
                new_values: &v_output,
                output: &attended,
            },
            config,
        )?;
        self.encode_bf16_matvec(encoder, o.0, &attended, &projected, o.1, o.2)?;
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "chained attention command")?;

        let read = |buffer: &metal::Buffer, length: usize| unsafe {
            // SAFETY: the chained command completed and the shared buffer is initialized.
            std::slice::from_raw_parts(buffer.contents() as *const f32, length).to_vec()
        };
        Ok(ChainedAttentionOutput {
            projected: read(&projected, o.1),
            keys: read(&k_rope, kv_rows),
            values: read(&v_output, kv_rows),
        })
    }

    fn encode_rms_norm_bf16_heads(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &metal::BufferRef,
        weight: &metal::BufferRef,
        output: &metal::BufferRef,
        params: NormParams,
    ) -> Result<(), String> {
        let heads =
            u32::try_from(params.heads).map_err(|_| "RMSNorm head count exceeds Metal limits")?;
        let head_dim =
            u32::try_from(params.head_dim).map_err(|_| "RMSNorm dimension exceeds Metal limits")?;
        encoder.set_compute_pipeline_state(&self.rms_norm_bf16_pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(3, 4, &heads as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &head_dim as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(
            5,
            4,
            &params.epsilon as *const f32 as *const std::ffi::c_void,
        );
        let threads = self
            .rms_norm_bf16_pipeline
            .max_total_threads_per_threadgroup()
            .max(1);
        encoder.dispatch_threads(
            metal::MTLSize::new(heads as u64, 1, 1),
            metal::MTLSize::new(threads.min(heads as u64), 1, 1),
        );
        Ok(())
    }

    fn encode_rope(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        input: &metal::BufferRef,
        output: &metal::BufferRef,
        params: RopeParams,
    ) -> Result<(), String> {
        let heads =
            u32::try_from(params.heads).map_err(|_| "RoPE head count exceeds Metal limits")?;
        let head_dim =
            u32::try_from(params.head_dim).map_err(|_| "RoPE dimension exceeds Metal limits")?;
        let position =
            u32::try_from(params.position).map_err(|_| "RoPE position exceeds Metal limits")?;
        encoder.set_compute_pipeline_state(&self.rope_pipeline);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(2, 4, &heads as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(3, 4, &head_dim as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &position as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &params.theta as *const f32 as *const std::ffi::c_void);
        let pairs = heads as u64 * (head_dim as u64 / 2);
        let threads = self
            .rope_pipeline
            .max_total_threads_per_threadgroup()
            .max(1);
        encoder.dispatch_threads(
            metal::MTLSize::new(pairs, 1, 1),
            metal::MTLSize::new(threads.min(pairs), 1, 1),
        );
        Ok(())
    }

    fn encode_attention_decode(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        buffers: AttentionBufferRefs<'_>,
        config: ChainedAttentionConfig,
    ) -> Result<(), String> {
        let AttentionBufferRefs {
            query,
            key_cache,
            value_cache,
            new_keys,
            new_values,
            output,
        } = buffers;
        let ChainedAttentionConfig {
            query_heads,
            key_value_heads,
            head_dim,
            cached_tokens,
            cache_capacity_tokens,
            ..
        } = config;
        let query_heads =
            u32::try_from(query_heads).map_err(|_| "query head count exceeds Metal limits")?;
        let key_value_heads =
            u32::try_from(key_value_heads).map_err(|_| "KV head count exceeds Metal limits")?;
        let head_dim =
            u32::try_from(head_dim).map_err(|_| "head dimension exceeds Metal limits")?;
        let cached_tokens =
            u32::try_from(cached_tokens).map_err(|_| "cached token count exceeds Metal limits")?;
        let cache_capacity_tokens = u32::try_from(cache_capacity_tokens)
            .map_err(|_| "KV-cache capacity exceeds Metal limits")?;
        let scale = (head_dim as f32).sqrt().recip();
        encoder.set_compute_pipeline_state(&self.attention_decode_pipeline);
        encoder.set_buffer(0, Some(query), 0);
        encoder.set_buffer(1, Some(key_cache), 0);
        encoder.set_buffer(2, Some(value_cache), 0);
        encoder.set_buffer(3, Some(new_keys), 0);
        encoder.set_buffer(4, Some(new_values), 0);
        encoder.set_buffer(5, Some(output), 0);
        encoder.set_bytes(6, 4, &query_heads as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(
            7,
            4,
            &key_value_heads as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(8, 4, &head_dim as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(
            9,
            4,
            &cached_tokens as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            10,
            4,
            &cache_capacity_tokens as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(11, 4, &scale as *const f32 as *const std::ffi::c_void);
        let threads = self
            .attention_decode_pipeline
            .max_total_threads_per_threadgroup()
            .max(1);
        encoder.dispatch_threads(
            metal::MTLSize::new(query_heads as u64, 1, 1),
            metal::MTLSize::new(threads.min(query_heads as u64), 1, 1),
        );
        Ok(())
    }

    fn submit_fused_qkv(
        &self,
        matrices: &[(&Bf16Weight, usize, usize)],
        input: &[f32],
    ) -> Result<PendingMatvec, String> {
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let outputs = matrices
            .iter()
            .map(|(_, rows, _)| {
                self.device.new_buffer(
                    std::mem::size_of::<f32>() as u64 * *rows as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                )
            })
            .collect::<Vec<_>>();
        let mut mapped_weight_bytes = 0_u64;
        let mut active_weight_bytes = 0_u64;
        for (weight, _, _) in matrices {
            if !weight.persistent {
                active_weight_bytes = active_weight_bytes.saturating_add(weight.bytes);
                if weight.mapped {
                    mapped_weight_bytes = mapped_weight_bytes.saturating_add(weight.bytes);
                }
            }
        }
        let scratch_bytes = std::mem::size_of_val(input) as u64
            + outputs.iter().map(|output| output.length()).sum::<u64>();
        if mapped_weight_bytes > 0 {
            self.record_mapped_profile(mapped_weight_bytes, 0, scratch_bytes);
        } else {
            self.record_profile(active_weight_bytes, 0, scratch_bytes);
        }

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.bf16_fused_qkv_pipeline);
        for (index, (weight, _, _)) in matrices.iter().enumerate() {
            encoder.set_buffer(index as u64, Some(&weight.buffer), weight.offset);
        }
        encoder.set_buffer(3, Some(&input_buffer), 0);
        for (index, output) in outputs.iter().enumerate() {
            encoder.set_buffer(4 + index as u64, Some(output), 0);
        }
        let columns =
            u32::try_from(matrices[0].2).map_err(|_| "fused QKV columns exceed Metal limits")?;
        let q_rows =
            u32::try_from(matrices[0].1).map_err(|_| "fused QKV query rows exceed Metal limits")?;
        let k_rows =
            u32::try_from(matrices[1].1).map_err(|_| "fused QKV key rows exceed Metal limits")?;
        let v_rows =
            u32::try_from(matrices[2].1).map_err(|_| "fused QKV value rows exceed Metal limits")?;
        encoder.set_bytes(7, 4, &columns as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(8, 4, &q_rows as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(9, 4, &k_rows as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(10, 4, &v_rows as *const u32 as *const std::ffi::c_void);
        let threads = self
            .bf16_fused_qkv_pipeline
            .max_total_threads_per_threadgroup()
            .clamp(1, 128);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new((q_rows + k_rows + v_rows) as u64, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        self.command_stats.record_async_submission();
        let command_buffer = command_buffer.to_owned();
        let weights = matrices
            .iter()
            .map(|(weight, _, _)| weight.buffer.clone())
            .collect();
        let outputs = outputs
            .into_iter()
            .zip(matrices.iter().map(|(_, rows, _)| *rows))
            .collect();
        Ok(PendingMatvec {
            command_buffer,
            _input: input_buffer,
            _weights: weights,
            _keepalive: Vec::new(),
            outputs,
            stats: Arc::clone(&self.command_stats),
        })
    }

    fn submit_fused_gate_up(
        &self,
        matrices: &[(&Bf16Weight, usize, usize)],
        input: &[f32],
    ) -> Result<PendingMatvec, String> {
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let outputs = matrices
            .iter()
            .map(|(_, rows, _)| {
                self.device.new_buffer(
                    std::mem::size_of::<f32>() as u64 * *rows as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                )
            })
            .collect::<Vec<_>>();
        let mut mapped_weight_bytes = 0_u64;
        let mut active_weight_bytes = 0_u64;
        for (weight, _, _) in matrices {
            if !weight.persistent {
                active_weight_bytes = active_weight_bytes.saturating_add(weight.bytes);
                if weight.mapped {
                    mapped_weight_bytes = mapped_weight_bytes.saturating_add(weight.bytes);
                }
            }
        }
        let scratch_bytes = std::mem::size_of_val(input) as u64
            + outputs.iter().map(|output| output.length()).sum::<u64>();
        if mapped_weight_bytes > 0 {
            self.record_mapped_profile(mapped_weight_bytes, 0, scratch_bytes);
        } else {
            self.record_profile(active_weight_bytes, 0, scratch_bytes);
        }

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.bf16_fused_gate_up_pipeline);
        for (index, (weight, _, _)) in matrices.iter().enumerate() {
            encoder.set_buffer(index as u64, Some(&weight.buffer), weight.offset);
        }
        encoder.set_buffer(2, Some(&input_buffer), 0);
        for (index, output) in outputs.iter().enumerate() {
            encoder.set_buffer(3 + index as u64, Some(output), 0);
        }
        let columns = u32::try_from(matrices[0].2)
            .map_err(|_| "fused gate/up columns exceed Metal limits")?;
        let gate_rows =
            u32::try_from(matrices[0].1).map_err(|_| "fused gate rows exceed Metal limits")?;
        let up_rows =
            u32::try_from(matrices[1].1).map_err(|_| "fused up rows exceed Metal limits")?;
        encoder.set_bytes(5, 4, &columns as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(6, 4, &gate_rows as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(7, 4, &up_rows as *const u32 as *const std::ffi::c_void);
        let threads = self
            .bf16_fused_gate_up_pipeline
            .max_total_threads_per_threadgroup()
            .clamp(1, 128);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new((gate_rows + up_rows) as u64, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        self.command_stats.record_async_submission();
        let command_buffer = command_buffer.to_owned();
        let weights = matrices
            .iter()
            .map(|(weight, _, _)| weight.buffer.clone())
            .collect();
        let outputs = outputs
            .into_iter()
            .zip(matrices.iter().map(|(_, rows, _)| *rows))
            .collect();
        Ok(PendingMatvec {
            command_buffer,
            _input: input_buffer,
            _weights: weights,
            _keepalive: Vec::new(),
            outputs,
            stats: Arc::clone(&self.command_stats),
        })
    }

    fn encode_bf16_matvec(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        weight: &Bf16Weight,
        input: &metal::BufferRef,
        output: &metal::BufferRef,
        rows: usize,
        columns: usize,
    ) -> Result<(), String> {
        encoder.set_compute_pipeline_state(&self.bf16_matvec_pipeline);
        encoder.set_buffer(0, Some(&weight.buffer), weight.offset);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        let columns = u32::try_from(columns).map_err(|_| "matvec columns exceed Metal limits")?;
        let rows = u32::try_from(rows).map_err(|_| "matvec rows exceed Metal limits")?;
        encoder.set_bytes(3, 4, &columns as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &rows as *const u32 as *const std::ffi::c_void);
        let threads = self
            .bf16_matvec_pipeline
            .max_total_threads_per_threadgroup()
            .clamp(1, 128);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(rows as u64, 1, 1),
            metal::MTLSize::new(threads, 1, 1),
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_bf16_matmul_many(
        &self,
        encoder: &metal::ComputeCommandEncoderRef,
        pipeline: &metal::ComputePipelineState,
        weight: &Bf16Weight,
        input: &metal::BufferRef,
        output: &metal::BufferRef,
        rows: usize,
        columns: usize,
        batch: usize,
    ) -> Result<(), String> {
        if !(1..=8).contains(&batch) {
            return Err("batched matmul supports between one and eight inputs".into());
        }
        if pipeline.max_total_threads_per_threadgroup() < 32 {
            return Err("batched matmul requires at least one 32-thread SIMD group".into());
        }
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(&weight.buffer), weight.offset);
        encoder.set_buffer(1, Some(input), 0);
        encoder.set_buffer(2, Some(output), 0);
        let columns =
            u32::try_from(columns).map_err(|_| "batched matmul columns exceed Metal limits")?;
        let rows = u32::try_from(rows).map_err(|_| "batched matmul rows exceed Metal limits")?;
        let batch =
            u32::try_from(batch).map_err(|_| "batched matmul batch exceeds Metal limits")?;
        encoder.set_bytes(3, 4, &columns as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &rows as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &batch as *const u32 as *const std::ffi::c_void);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(u64::from(rows), 1, 1),
            metal::MTLSize::new(32, 1, 1),
        );
        Ok(())
    }

    pub fn bf16_matvec_tensor(
        &self,
        tensor: &TensorView<'_>,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if tensor.info.dtype != "BF16" || tensor.info.shape.len() != 2 {
            return Err("matvec requires a rank-2 BF16 tensor".into());
        }
        let weight = self.map_bf16_tensor_weight(tensor)?;
        self.bf16_matvec_buffer(&weight, tensor.info.shape[0], tensor.info.shape[1], input)
    }

    /// Multiply a matrix in bounded row chunks. This is the first explicit
    /// fast-memory trade-off: logits are assembled in order while no uploaded
    /// weight buffer exceeds `max_rows * columns * 2` bytes.
    pub fn bf16_matvec_tensor_chunked(
        &self,
        tensor: &TensorView<'_>,
        input: &[f32],
        max_rows: usize,
    ) -> Result<Vec<f32>, String> {
        if tensor.info.dtype != "BF16" || tensor.info.shape.len() != 2 || max_rows == 0 {
            return Err("chunked matvec requires a rank-2 BF16 tensor and non-zero row cap".into());
        }
        let rows = tensor.info.shape[0];
        let columns = tensor.info.shape[1];
        let row_bytes = columns
            .checked_mul(2)
            .ok_or("chunked matvec row byte length overflow")?;
        let expected_bytes = rows
            .checked_mul(row_bytes)
            .ok_or("chunked matvec dimensions overflow")?;
        if tensor.bytes.len() != expected_bytes || input.len() != columns {
            return Err("chunked matvec tensor bytes or input length is invalid".into());
        }
        let mut output = Vec::with_capacity(rows);
        for row_start in (0..rows).step_by(max_rows) {
            let row_count = max_rows.min(rows - row_start);
            let byte_start = row_start * row_bytes;
            let byte_end = byte_start + row_count * row_bytes;
            let chunk_bytes = &tensor.bytes[byte_start..byte_end];
            let weight = self.map_bf16_mapped_weight(chunk_bytes, tensor.backing)?;
            let chunk = self.bf16_matvec_buffer(&weight, row_count, columns, input)?;
            output.extend(chunk);
        }
        Ok(output)
    }

    /// Multiply only a contiguous row range of a mapped BF16 tensor. This is
    /// used by exact output-head search after a conservative block bound has
    /// proved that other vocabulary rows cannot win.
    pub fn bf16_matvec_tensor_rows(
        &self,
        tensor: &TensorView<'_>,
        input: &[f32],
        row_start: usize,
        row_count: usize,
    ) -> Result<Vec<f32>, String> {
        if tensor.info.dtype != "BF16"
            || tensor.info.shape.len() != 2
            || row_count == 0
            || row_start
                .checked_add(row_count)
                .is_none_or(|end| end > tensor.info.shape[0])
            || input.len() != tensor.info.shape[1]
        {
            return Err("row-range matvec dimensions are invalid".into());
        }
        let row_bytes = tensor.info.shape[1]
            .checked_mul(2)
            .ok_or("row-range matvec row byte length overflow")?;
        let byte_start = row_start
            .checked_mul(row_bytes)
            .ok_or("row-range matvec byte offset overflow")?;
        let byte_len = row_count
            .checked_mul(row_bytes)
            .ok_or("row-range matvec byte length overflow")?;
        let byte_end = byte_start
            .checked_add(byte_len)
            .ok_or("row-range matvec byte end overflow")?;
        let bytes = tensor
            .bytes
            .get(byte_start..byte_end)
            .ok_or("row-range matvec bytes are shorter than requested range")?;
        let weight = self.map_bf16_mapped_weight(bytes, tensor.backing)?;
        self.bf16_matvec_buffer(&weight, row_count, tensor.info.shape[1], input)
    }

    pub fn upload_bf16_tensor(&self, tensor: &TensorView<'_>) -> Result<Bf16Weight, String> {
        if tensor.info.dtype != "BF16" {
            return Err("resident weight upload requires a BF16 tensor".into());
        }
        self.upload_bf16_weight(tensor.bytes)
    }

    pub fn bf16_matvec_tensor_buffer(
        &self,
        tensor: &TensorView<'_>,
        weight: &Bf16Weight,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if tensor.info.dtype != "BF16" || tensor.info.shape.len() != 2 {
            return Err("matvec requires a rank-2 BF16 tensor".into());
        }
        self.bf16_matvec_buffer(weight, tensor.info.shape[0], tensor.info.shape[1], input)
    }

    pub fn bf16_embedding_tensor(
        &self,
        tensor: &TensorView<'_>,
        token_id: usize,
    ) -> Result<Vec<f32>, String> {
        if tensor.info.dtype != "BF16" || tensor.info.shape.len() != 2 {
            return Err("embedding requires a rank-2 BF16 tensor".into());
        }
        let rows = tensor.info.shape[0];
        let columns = tensor.info.shape[1];
        if token_id >= rows {
            return Err("embedding token id exceeds vocabulary".into());
        }
        let row_bytes = columns
            .checked_mul(2)
            .ok_or("embedding dimension overflow")?;
        let row_start = token_id
            .checked_mul(row_bytes)
            .ok_or("embedding row offset overflow")?;
        let row_end = row_start
            .checked_add(row_bytes)
            .ok_or("embedding row end overflow")?;
        let row = tensor
            .bytes
            .get(row_start..row_end)
            .ok_or("embedding tensor bytes are shorter than its shape")?;
        self.bf16_embedding_row(row, columns)
    }

    pub fn bf16_embedding_tensor_buffer(
        &self,
        tensor: &TensorView<'_>,
        weight: &Bf16Weight,
        token_id: usize,
    ) -> Result<Vec<f32>, String> {
        if tensor.info.dtype != "BF16" || tensor.info.shape.len() != 2 {
            return Err("embedding requires a rank-2 BF16 tensor".into());
        }
        let rows = tensor.info.shape[0];
        let columns = tensor.info.shape[1];
        if token_id >= rows {
            return Err("embedding token id exceeds vocabulary".into());
        }
        let expected_bytes = rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_mul(2))
            .ok_or("embedding dimensions overflow")?;
        let row_offset = token_id
            .checked_mul(columns)
            .and_then(|element| element.checked_mul(2))
            .ok_or("embedding row offset overflow")?;
        if weight.bytes != expected_bytes as u64 {
            return Err("resident embedding weight has an invalid byte length".into());
        }
        let output_buffer = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64 * columns as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.record_profile(
            weight.bytes,
            0,
            std::mem::size_of::<f32>() as u64 * columns as u64,
        );
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.bf16_embedding_pipeline);
        encoder.set_buffer(0, Some(&weight.buffer), weight.offset + row_offset as u64);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        let columns =
            u32::try_from(columns).map_err(|_| "embedding dimension exceeds Metal limits")?;
        encoder.set_bytes(2, 4, &columns as *const u32 as *const std::ffi::c_void);
        let threads = self
            .bf16_embedding_pipeline
            .max_total_threads_per_threadgroup()
            .max(1);
        encoder.dispatch_threads(
            metal::MTLSize::new(columns as u64, 1, 1),
            metal::MTLSize::new(threads.min(columns as u64), 1, 1),
        );
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "command")?;
        // SAFETY: command buffer completed and output contains columns f32s.
        let output = unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, columns as usize)
        };
        Ok(output.to_vec())
    }

    pub fn bf16_embedding_row(&self, row_bytes: &[u8], columns: usize) -> Result<Vec<f32>, String> {
        if columns == 0 || row_bytes.len() != columns * 2 {
            return Err("embedding row dimensions or byte length are invalid".into());
        }
        let weight_buffer = self.device.new_buffer_with_data(
            row_bytes.as_ptr() as *const std::ffi::c_void,
            row_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64 * columns as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.record_profile(
            row_bytes.len() as u64,
            0,
            std::mem::size_of::<f32>() as u64 * columns as u64,
        );
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.bf16_embedding_pipeline);
        encoder.set_buffer(0, Some(&weight_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        let columns = columns as u32;
        encoder.set_bytes(
            2,
            std::mem::size_of_val(&columns) as u64,
            &columns as *const u32 as *const std::ffi::c_void,
        );
        let threads = self
            .bf16_embedding_pipeline
            .max_total_threads_per_threadgroup()
            .max(1);
        encoder.dispatch_threads(
            metal::MTLSize::new(columns as u64, 1, 1),
            metal::MTLSize::new(threads.min(columns as u64), 1, 1),
        );
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "command")?;
        // SAFETY: command buffer completed and output contains columns f32s.
        let output = unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, columns as usize)
        };
        Ok(output.to_vec())
    }

    pub fn rope(
        &self,
        input: &[f32],
        heads: usize,
        head_dim: usize,
        position: usize,
        theta: f32,
    ) -> Result<Vec<f32>, String> {
        if heads == 0
            || head_dim == 0
            || !head_dim.is_multiple_of(2)
            || input.len() != heads * head_dim
        {
            return Err("RoPE dimensions are invalid".into());
        }
        let byte_length = std::mem::size_of_val(input) as u64;
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            byte_length,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self
            .device
            .new_buffer(byte_length, metal::MTLResourceOptions::StorageModeShared);
        self.record_profile(0, 0, byte_length.saturating_mul(2));
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.rope_pipeline);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        let heads = heads as u32;
        let head_dim = head_dim as u32;
        let position = position as u32;
        encoder.set_bytes(2, 4, &heads as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(3, 4, &head_dim as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &position as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, &theta as *const f32 as *const std::ffi::c_void);
        let pairs = heads as u64 * (head_dim as u64 / 2);
        let threads = self
            .rope_pipeline
            .max_total_threads_per_threadgroup()
            .max(1);
        encoder.dispatch_threads(
            metal::MTLSize::new(pairs, 1, 1),
            metal::MTLSize::new(threads.min(pairs), 1, 1),
        );
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "command")?;
        // SAFETY: command buffer completed and output contains input.len() f32s.
        let output = unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, input.len())
        };
        Ok(output.to_vec())
    }

    pub fn attention_decode(&self, input: AttentionDecodeInput<'_>) -> Result<Vec<f32>, String> {
        let AttentionDecodeInput {
            query,
            key_cache,
            value_cache,
            new_keys,
            new_values,
            query_heads,
            key_value_heads,
            head_dim,
            cached_tokens,
            cache_capacity_tokens,
        } = input;
        if query_heads == 0
            || key_value_heads == 0
            || head_dim == 0
            || !query_heads.is_multiple_of(key_value_heads)
        {
            return Err("attention head dimensions are invalid".into());
        }
        let query_len = query_heads
            .checked_mul(head_dim)
            .ok_or("attention query dimensions overflow")?;
        let new_len = key_value_heads
            .checked_mul(head_dim)
            .ok_or("attention key/value dimensions overflow")?;
        if query.len() != query_len || new_keys.len() != new_len || new_values.len() != new_len {
            return Err("attention query or new key/value lengths are invalid".into());
        }
        let cache_stride = key_value_heads
            .checked_mul(head_dim)
            .ok_or("attention cache dimensions overflow")?;
        if cached_tokens > cache_capacity_tokens {
            return Err("attention cached token count exceeds cache capacity".into());
        }
        let cache_elements = cache_stride
            .checked_mul(cache_capacity_tokens)
            .ok_or("attention cache dimensions overflow")?;
        if key_cache.len() != cache_elements || value_cache.len() != cache_elements {
            return Err("attention cache dimensions are invalid".into());
        }
        let empty_cache = [0.0_f32];
        let key_cache_input = if key_cache.is_empty() {
            &empty_cache[..]
        } else {
            key_cache
        };
        let value_cache_input = if value_cache.is_empty() {
            &empty_cache[..]
        } else {
            value_cache
        };
        let query_buffer = self.device.new_buffer_with_data(
            query.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(query) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let key_cache_buffer = self.device.new_buffer_with_data(
            key_cache_input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(key_cache_input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let value_cache_buffer = self.device.new_buffer_with_data(
            value_cache_input.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(value_cache_input) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let new_keys_buffer = self.device.new_buffer_with_data(
            new_keys.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(new_keys) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let new_values_buffer = self.device.new_buffer_with_data(
            new_values.as_ptr() as *const std::ffi::c_void,
            std::mem::size_of_val(new_values) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64 * query_len as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.record_profile(
            0,
            (std::mem::size_of_val(key_cache_input) as u64)
                .saturating_add(std::mem::size_of_val(value_cache_input) as u64),
            std::mem::size_of_val(query) as u64
                + std::mem::size_of_val(new_keys) as u64
                + std::mem::size_of_val(new_values) as u64
                + std::mem::size_of::<f32>() as u64 * query_len as u64,
        );
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.attention_decode_pipeline);
        encoder.set_buffer(0, Some(&query_buffer), 0);
        encoder.set_buffer(1, Some(&key_cache_buffer), 0);
        encoder.set_buffer(2, Some(&value_cache_buffer), 0);
        encoder.set_buffer(3, Some(&new_keys_buffer), 0);
        encoder.set_buffer(4, Some(&new_values_buffer), 0);
        encoder.set_buffer(5, Some(&output_buffer), 0);
        let query_heads =
            u32::try_from(query_heads).map_err(|_| "query head count exceeds Metal limits")?;
        let key_value_heads =
            u32::try_from(key_value_heads).map_err(|_| "KV head count exceeds Metal limits")?;
        let head_dim =
            u32::try_from(head_dim).map_err(|_| "head dimension exceeds Metal limits")?;
        let cached_tokens =
            u32::try_from(cached_tokens).map_err(|_| "cached token count exceeds Metal limits")?;
        let cache_capacity_tokens = u32::try_from(cache_capacity_tokens)
            .map_err(|_| "KV-cache capacity exceeds Metal limits")?;
        let scale = (head_dim as f32).sqrt().recip();
        encoder.set_bytes(6, 4, &query_heads as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(
            7,
            4,
            &key_value_heads as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(8, 4, &head_dim as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(
            9,
            4,
            &cached_tokens as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            10,
            4,
            &cache_capacity_tokens as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(11, 4, &scale as *const f32 as *const std::ffi::c_void);
        let threads = self
            .attention_decode_pipeline
            .max_total_threads_per_threadgroup()
            .max(1);
        encoder.dispatch_threads(
            metal::MTLSize::new(query_heads as u64, 1, 1),
            metal::MTLSize::new(threads.min(query_heads as u64), 1, 1),
        );
        encoder.end_encoding();
        self.commit_and_wait(command_buffer, "command")?;
        // SAFETY: command buffer completed and output contains query_len f32s.
        let output = unsafe {
            std::slice::from_raw_parts(output_buffer.contents() as *const f32, query_len)
        };
        Ok(output.to_vec())
    }

    /// Decode attention directly from a fixed-capacity cache. The cache's
    /// backing storage is passed through unchanged; only the active token
    /// count is sent to the kernel.
    pub fn attention_decode_kv_cache(
        &self,
        query: &[f32],
        cache: &KvCache,
        new_keys: &[f32],
        new_values: &[f32],
        query_heads: usize,
    ) -> Result<Vec<f32>, String> {
        self.attention_decode(AttentionDecodeInput {
            query,
            key_cache: cache.key_storage(),
            value_cache: cache.value_storage(),
            new_keys,
            new_values,
            query_heads,
            key_value_heads: cache.key_value_heads(),
            head_dim: cache.head_dim(),
            cached_tokens: cache.cached_tokens(),
            cache_capacity_tokens: cache.capacity_tokens(),
        })
    }
}

#[cfg(not(target_os = "macos"))]
pub fn probe() -> Result<MetalDeviceInfo, String> {
    Err("Metal backend requires macOS".into())
}

#[cfg(test)]
mod tests {
    #[cfg(target_os = "macos")]
    #[test]
    fn qk_rows4_switch_is_explicitly_opt_in() {
        assert!(!super::qk_rows4_enabled(None));
        assert!(!super::qk_rows4_enabled(Some("0")));
        assert!(super::qk_rows4_enabled(Some("1")));
        assert!(!super::qk_rows4_enabled(Some("true")));
    }

    #[cfg(target_os = "macos")]
    use crate::model::{GgufModelStore, TensorInfo, TensorView};

    #[test]
    fn probe_api_is_callable() {
        let _ = super::probe();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn command_stats_snapshot_tracks_submissions_and_waits() {
        let stats = super::CommandStats::default();

        stats.record_submission();
        stats.record_submission();
        stats.record_async_submission();
        stats.record_wait(std::time::Duration::from_nanos(7));

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.submitted, 3);
        assert_eq!(snapshot.async_submitted, 1);
        assert_eq!(snapshot.waited, 1);
        assert_eq!(snapshot.wait_nanos, 7);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn command_stats_delta_saturates_counters() {
        let before = super::CommandStatsSnapshot {
            submitted: 10,
            async_submitted: 4,
            waited: 8,
            wait_nanos: 100,
        };
        let after = super::CommandStatsSnapshot {
            submitted: 13,
            async_submitted: 5,
            waited: 7,
            wait_nanos: 90,
        };

        let delta = after.delta_since(before);

        assert_eq!(delta.submitted, 3);
        assert_eq!(delta.async_submitted, 1);
        assert_eq!(delta.waited, 0);
        assert_eq!(delta.wait_nanos, 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn streaming_batch_validation_accepts_rank2_bf16_tensors() {
        let info = TensorInfo {
            name: "projection.weight".into(),
            shard: "shard.safetensors".into(),
            dtype: "BF16".into(),
            shape: vec![2, 3],
            data_start: 0,
            data_end: 12,
        };
        let bytes = [0_u8; 12];
        let tensor = TensorView {
            info: &info,
            bytes: &bytes,
            backing: &bytes,
        };

        let shapes = super::validate_matvec_tensor_batch(&[&tensor], 3)
            .expect("valid streamed projection should pass validation");

        assert_eq!(shapes, vec![(2, 3)]);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn streaming_batch_validation_rejects_mismatched_input_columns() {
        let info = TensorInfo {
            name: "projection.weight".into(),
            shard: "shard.safetensors".into(),
            dtype: "BF16".into(),
            shape: vec![2, 4],
            data_start: 0,
            data_end: 16,
        };
        let bytes = [0_u8; 16];
        let tensor = TensorView {
            info: &info,
            bytes: &bytes,
            backing: &bytes,
        };

        let error = super::validate_matvec_tensor_batch(&[&tensor], 3)
            .expect_err("mismatched streamed projection must fail validation");

        assert!(error.contains("input length"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn batched_matmul_validation_accepts_k_up_to_eight() {
        assert!(super::validate_matmul_many_shape(3, 4, 8, 24, 32).is_ok());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn batched_matmul_validation_rejects_more_than_eight_candidates() {
        let error = super::validate_matmul_many_shape(3, 4, 9, 24, 36)
            .expect_err("candidate batch must be bounded for the first kernel");
        assert!(error.contains("invalid"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn fused_projection_validation_accepts_qkv_and_gate_up_shapes() {
        assert!(super::validate_fused_projection_shapes(&[(6, 4), (2, 4), (2, 4)], 4,).is_ok());
        assert!(super::validate_fused_projection_shapes(&[(8, 4), (8, 4)], 4).is_ok());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn fused_projection_validation_rejects_invalid_group_count() {
        let error = super::validate_fused_projection_shapes(&[(4, 4)], 4)
            .expect_err("fused projections require two or three matrices");
        assert!(error.contains("two or three"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn batched_fused_projection_validation_accepts_k_up_to_eight() {
        assert!(
            super::validate_fused_projection_many_shapes(&[(6, 4), (2, 4), (2, 4)], 8, 4,).is_ok()
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn batched_fused_projection_validation_rejects_more_than_eight_candidates() {
        let error = super::validate_fused_projection_many_shapes(&[(6, 4), (2, 4), (2, 4)], 9, 4)
            .expect_err("batched fused projection width must be bounded");
        assert!(error.contains("between one and eight"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn projection_shaders_use_simdgroup_reduction() {
        assert!(super::RMS_NORM_SHADER.contains("simd_sum"));
        assert!(super::RMS_NORM_SHADER.contains("simdgroup_index_in_threadgroup"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rms_norm_kernel_matches_cpu_reference() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let input = [1.0_f32, -2.0, 3.0, -4.0];
        let weight = [1.0_f32, 0.5, 2.0, -1.0];
        let epsilon = 1.0e-5_f32;
        let actual = context
            .rms_norm(&input, &weight, epsilon)
            .expect("RMSNorm kernel should run");
        let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
        let inverse_rms = (mean_square + epsilon).sqrt().recip();
        for (index, value) in actual.iter().enumerate() {
            let expected = input[index] * inverse_rms * weight[index];
            assert!(
                (value - expected).abs() < 1.0e-5,
                "index {index}: {value} != {expected}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn bf16_rms_norm_kernel_matches_cpu_reference() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let input = [1.0_f32, -2.0, 3.0, -4.0, 0.5, 1.5, -2.5, 3.5];
        let weights = [1.0_f32, 0.5, 2.0, -1.0];
        let weight_bytes = weights
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        let epsilon = 1.0e-5_f32;
        let actual = context
            .rms_norm_bf16_heads(&input, &weight_bytes, 2, 4, epsilon)
            .expect("BF16 RMSNorm should run");
        for head in 0..2 {
            let row = &input[head * 4..(head + 1) * 4];
            let mean_square = row.iter().map(|value| value * value).sum::<f32>() / 4.0;
            let inverse_rms = (mean_square + epsilon).sqrt().recip();
            for dimension in 0..4 {
                let index = head * 4 + dimension;
                let expected = row[dimension] * inverse_rms * weights[dimension];
                assert!(
                    (actual[index] - expected).abs() < 1.0e-5,
                    "index {index}: {} != {expected}",
                    actual[index]
                );
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn bf16_matvec_kernel_matches_cpu_reference() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let rows = 3;
        let columns = 4;
        let weights = [
            1.0_f32, 0.5, -1.0, 2.0, -2.0, 1.0, 0.25, 3.0, 0.125, -0.5, 2.0, -1.0,
        ];
        let input = [2.0_f32, -1.0, 0.5, 3.0];
        let weight_bytes = weights
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        let actual = context
            .bf16_matvec(&weight_bytes, rows, columns, &input)
            .expect("BF16 matvec kernel should run");
        let expected = (0..rows)
            .map(|row| {
                (0..columns)
                    .map(|column| weights[row * columns + column] * input[column])
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        for (index, (value, reference)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (value - reference).abs() < 1.0e-5,
                "index {index}: {value} != {reference}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn f32_matvec_kernel_matches_cpu_reference() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let rows = 3;
        let columns = 4;
        let weights = [
            1.0_f32, 0.5, -1.0, 2.0, -2.0, 1.0, 0.25, 3.0, 0.125, -0.5, 2.0, -1.0,
        ];
        let input = [2.0_f32, -1.0, 0.5, 3.0];
        let actual = context
            .f32_matvec(&weights, rows, columns, &input)
            .expect("F32 matvec kernel should run");
        let expected = (0..rows)
            .map(|row| {
                (0..columns)
                    .map(|column| weights[row * columns + column] * input[column])
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        for (index, (value, reference)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (value - reference).abs() < 1.0e-5,
                "index {index}: {value} != {reference}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gated_delta_step_kernel_matches_cpu_reference() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let mut reference =
            crate::qwen35::GatedDeltaState::zeros(1, 2, 1).expect("valid recurrent state");
        let query = [1.0_f32, 0.0];
        let key = [1.0_f32, 0.0];
        let value = [2.0_f32];
        let gate = [0.0_f32];
        let beta = [0.5_f32];
        let (expected_output, expected_state) = {
            let output = reference
                .step(&query, &key, &value, &gate, &beta)
                .expect("CPU recurrent step should run");
            (output, reference.as_slice().to_vec())
        };
        let (actual_output, actual_state) = context
            .gated_delta_step(&query, &key, &value, &gate, &beta, &[0.0, 0.0], 1, 2, 1)
            .expect("Metal recurrent step should run");
        assert!((actual_output[0] - expected_output[0]).abs() < 1.0e-5);
        for (actual, expected) in actual_state.iter().zip(expected_state) {
            assert!((actual - expected).abs() < 1.0e-5);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn causal_conv_kernel_matches_cpu_reference() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let input = [5.0_f32, 6.0];
        let state = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weights = [1.0_f32, 2.0, 3.0, 4.0, -1.0, 0.5, 1.0, 2.0];
        let mut expected_state = state.to_vec();
        let expected_output =
            crate::qwen35::causal_conv1d_step(&input, &mut expected_state, &weights, 4)
                .expect("CPU causal convolution should run");
        let (actual_output, actual_state) = context
            .causal_conv1d_step(&input, &state, &weights, 2, 4)
            .expect("Metal causal convolution should run");
        for (actual, expected) in actual_output.iter().zip(expected_output) {
            assert!((actual - expected).abs() < 1.0e-5);
        }
        for (actual, expected) in actual_state.iter().zip(expected_state) {
            assert!((actual - expected).abs() < 1.0e-5);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn gated_rms_norm_kernel_matches_cpu_reference() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let input = [1.0_f32, 2.0, -1.0, 3.0];
        let gate = [0.0_f32, 1.0, -1.0, 0.5];
        let weight = [1.0_f32, 0.5];
        let expected = crate::qwen35::gated_rms_norm(&input, &gate, &weight, 2, 2, 1.0e-6)
            .expect("CPU gated RMSNorm should run");
        let actual = context
            .rms_norm_gated(&input, &gate, &weight, 2, 2, 1.0e-6)
            .expect("Metal gated RMSNorm should run");
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-5);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn head_rms_norm_kernel_matches_cpu_reference() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let input = [1.0_f32, 2.0, -1.0, 3.0];
        let weight = [0.25_f32, -0.5];
        let expected = crate::qwen35::rms_norm_heads(&input, &weight, 2, 2, 1.0e-6)
            .expect("CPU head RMSNorm should run");
        let actual = context
            .rms_norm_heads(&input, &weight, 2, 2, 1.0e-6)
            .expect("Metal head RMSNorm should run");
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-5);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn q4_k_matvec_kernel_matches_cpu_reference() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let rows = 2;
        let columns = 512;
        let mut weight_bytes = vec![0_u8; rows * 2 * crate::quant::Q4_K_BLOCK_BYTES];
        for row in 0..rows {
            for block_index in 0..2 {
                let block_start = (row * 2 + block_index) * crate::quant::Q4_K_BLOCK_BYTES;
                weight_bytes[block_start..block_start + 2]
                    .copy_from_slice(&0x3c00_u16.to_le_bytes());
                for (index, scale) in weight_bytes[block_start + 4..block_start + 16]
                    .iter_mut()
                    .enumerate()
                {
                    *scale = (index as u8).wrapping_mul(13).wrapping_add(7);
                }
                for (index, quant) in weight_bytes[block_start + 16..block_start + 144]
                    .iter_mut()
                    .enumerate()
                {
                    *quant = if (index + row + block_index) % 3 == 0 {
                        0x10
                    } else {
                        0x21
                    };
                }
            }
        }
        let input = (0..columns)
            .map(|index| ((index % 11) as f32 - 5.0) / 3.0)
            .collect::<Vec<_>>();
        let decoded = crate::quant::dequantize_q4_k(&weight_bytes, rows * columns)
            .expect("synthetic Q4_K weights should decode");
        let expected = (0..rows)
            .map(|row| {
                (0..columns)
                    .map(|column| decoded[row * columns + column] * input[column])
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        let actual = context
            .q4_k_matvec(&weight_bytes, rows, columns, &input)
            .expect("Q4_K matvec kernel should run");
        for (index, (value, reference)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (value - reference).abs() < 1.0e-4,
                "index {index}: {value} != {reference}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn q4_k_fused_gate_up_kernel_matches_two_cpu_matvecs() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let rows = 2;
        let columns = 256;
        let mut gate = vec![0_u8; rows * crate::quant::Q4_K_BLOCK_BYTES];
        let mut up = vec![0_u8; rows * crate::quant::Q4_K_BLOCK_BYTES];
        for row in 0..rows {
            let start = row * crate::quant::Q4_K_BLOCK_BYTES;
            gate[start..start + 2].copy_from_slice(&0x3c00_u16.to_le_bytes());
            up[start..start + 2].copy_from_slice(&0x3800_u16.to_le_bytes());
            gate[start + 4..start + 16].fill(7);
            up[start + 4..start + 16].fill(11);
            gate[start + 16..start + 144].fill(0x21);
            up[start + 16..start + 144].fill(0x12);
        }
        let input = (0..columns)
            .map(|index| ((index % 17) as f32 - 8.0) / 5.0)
            .collect::<Vec<_>>();
        let gate_values = crate::quant::dequantize_q4_k(&gate, rows * columns)
            .expect("gate weights should decode");
        let up_values =
            crate::quant::dequantize_q4_k(&up, rows * columns).expect("up weights should decode");
        let expected = (0..rows)
            .map(|row| {
                let gate_sum = (0..columns)
                    .map(|column| gate_values[row * columns + column] * input[column])
                    .sum::<f32>();
                let up_sum = (0..columns)
                    .map(|column| up_values[row * columns + column] * input[column])
                    .sum::<f32>();
                gate_sum / (1.0 + (-gate_sum).exp()) * up_sum
            })
            .collect::<Vec<_>>();
        let actual = context
            .q4_k_fused_gate_up(&gate, &up, rows, columns, &input)
            .expect("fused Q4_K gate/up kernel should run");
        for (index, (value, reference)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (value - reference).abs() < 1.0e-4,
                "index {index}: {value} != {reference}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn q4_k_embedding_row_kernel_matches_cpu_reference() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let mut row = vec![0_u8; crate::quant::Q4_K_BLOCK_BYTES];
        row[0..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        row[4..16].copy_from_slice(&[7, 19, 31, 43, 55, 67, 79, 91, 103, 115, 127, 139]);
        row[16..].fill(0x21);
        let expected =
            crate::quant::dequantize_q4_k(&row, 256).expect("synthetic Q4_K row should decode");
        let actual = context
            .q4_k_embedding_row(&row, 256)
            .expect("Q4_K embedding kernel should run");
        for (index, (value, reference)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (value - reference).abs() < 1.0e-5,
                "index {index}: {value} != {reference}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn q5_and_q6_k_matvec_kernels_match_cpu_reference() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let input = (0..256)
            .map(|index| ((index % 13) as f32 - 6.0) / 5.0)
            .collect::<Vec<_>>();

        let mut q5 = vec![0_u8; crate::quant::Q5_K_BLOCK_BYTES];
        q5[0..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        q5[4..16].fill(1);
        q5[16..48].fill(0xff);
        q5[48..].fill(0x10);
        let q5_expected = crate::quant::dequantize_q5_k(&q5, 256)
            .expect("synthetic Q5_K weights should decode")
            .iter()
            .zip(&input)
            .map(|(weight, value)| weight * value)
            .sum::<f32>();
        let q5_actual = context
            .q5_k_matvec(&q5, 1, 256, &input)
            .expect("Q5_K matvec kernel should run");
        assert!((q5_actual[0] - q5_expected).abs() < 1.0e-4);

        let mut q6 = vec![0_u8; crate::quant::Q6_K_BLOCK_BYTES];
        q6[208..210].copy_from_slice(&0x3c00_u16.to_le_bytes());
        q6[..128].fill(0x21);
        q6[128..192].fill(0xe4);
        q6[192..208].fill(1);
        let q6_expected = crate::quant::dequantize_q6_k(&q6, 256)
            .expect("synthetic Q6_K weights should decode")
            .iter()
            .zip(&input)
            .map(|(weight, value)| weight * value)
            .sum::<f32>();
        let q6_actual = context
            .q6_k_matvec(&q6, 1, 256, &input)
            .expect("Q6_K matvec kernel should run");
        assert!((q6_actual[0] - q6_expected).abs() < 1.0e-4);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn q4_k_matvec_runs_against_real_gguf_row() {
        let Ok(model_path) = std::env::var("SI_GGUF_MODEL") else {
            return;
        };
        let store = GgufModelStore::open(model_path).expect("GGUF model should open");
        let info = store
            .tensors
            .values()
            .find(|tensor| tensor.is_q4_k() && tensor.shape.len() == 2)
            .expect("GGUF model should contain a rank-2 Q4_K tensor");
        let tensor = store.tensor(&info.name).expect("Q4_K tensor should open");
        let columns = info.shape[0];
        assert!(columns.is_multiple_of(crate::quant::Q4_K_BLOCK_ELEMENTS));
        let row_bytes =
            columns / crate::quant::Q4_K_BLOCK_ELEMENTS * crate::quant::Q4_K_BLOCK_BYTES;
        let row = &tensor.bytes[..row_bytes];
        let input = (0..columns)
            .map(|index| ((index % 17) as f32 - 8.0) / 7.0)
            .collect::<Vec<_>>();
        let decoded =
            crate::quant::dequantize_q4_k(row, columns).expect("GGUF Q4_K row should decode");
        let expected = decoded
            .iter()
            .zip(&input)
            .map(|(weight, value)| weight * value)
            .sum::<f32>();
        let context = super::MetalContext::new().expect("Metal context should compile");
        let actual = context
            .q4_k_matvec_tensor_rows(&tensor, 0, 1, &input)
            .expect("real GGUF Q4_K row should run");
        assert!(
            (actual[0] - expected).abs() < 1.0e-2,
            "actual={} expected={} delta={}",
            actual[0],
            expected,
            actual[0] - expected
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn q5_and_q6_k_matvec_run_against_real_gguf_rows() {
        let Ok(model_path) = std::env::var("SI_GGUF_MODEL") else {
            return;
        };
        let store = GgufModelStore::open(model_path).expect("GGUF model should open");
        let context = super::MetalContext::new().expect("Metal context should compile");
        for ggml_type in [crate::quant::GGML_TYPE_Q5_K, crate::quant::GGML_TYPE_Q6_K] {
            let info = store
                .tensors
                .values()
                .find(|tensor| tensor.ggml_type == ggml_type && tensor.shape.len() == 2)
                .expect("GGUF model should contain the mixed K tensor");
            let tensor = store
                .tensor(&info.name)
                .expect("mixed K tensor should open");
            let columns = info.shape[0];
            assert!(columns.is_multiple_of(crate::quant::Q4_K_BLOCK_ELEMENTS));
            let block_bytes = if ggml_type == crate::quant::GGML_TYPE_Q5_K {
                crate::quant::Q5_K_BLOCK_BYTES
            } else {
                crate::quant::Q6_K_BLOCK_BYTES
            };
            let row_bytes = columns / crate::quant::Q4_K_BLOCK_ELEMENTS * block_bytes;
            let row = &tensor.bytes[..row_bytes];
            let input = (0..columns)
                .map(|index| ((index % 19) as f32 - 9.0) / 8.0)
                .collect::<Vec<_>>();
            let expected = if ggml_type == crate::quant::GGML_TYPE_Q5_K {
                crate::quant::dequantize_q5_k(row, columns)
            } else {
                crate::quant::dequantize_q6_k(row, columns)
            }
            .expect("mixed K row should decode")
            .iter()
            .zip(&input)
            .map(|(weight, value)| weight * value)
            .sum::<f32>();
            let actual = if ggml_type == crate::quant::GGML_TYPE_Q5_K {
                context.q5_k_matvec(row, 1, columns, &input)
            } else {
                context.q6_k_matvec(row, 1, columns, &input)
            }
            .expect("mixed K row should run");
            assert!(
                (actual[0] - expected).abs() < 1.0e-2,
                "type={} tensor={} actual={} expected={} delta={}",
                ggml_type,
                info.name,
                actual[0],
                expected,
                actual[0] - expected
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn f32_matvec_runs_against_real_gguf_row() {
        let Ok(model_path) = std::env::var("SI_GGUF_MODEL") else {
            return;
        };
        let store = GgufModelStore::open(model_path).expect("GGUF model should open");
        let info = store
            .tensors
            .values()
            .find(|tensor| tensor.ggml_type == 0 && tensor.shape == [5120, 48])
            .expect("GGUF model should contain an F32 SSM matrix");
        let tensor = store.tensor(&info.name).expect("F32 tensor should open");
        let columns = info.shape[0];
        let input = (0..columns)
            .map(|index| ((index % 23) as f32 - 11.0) / 10.0)
            .collect::<Vec<_>>();
        let weights =
            unsafe { std::slice::from_raw_parts(tensor.bytes.as_ptr() as *const f32, columns) };
        let expected = weights
            .iter()
            .zip(&input)
            .map(|(weight, value)| weight * value)
            .sum::<f32>();
        let context = super::MetalContext::new().expect("Metal context should compile");
        let actual = context
            .f32_matvec_tensor_rows(&tensor, 0, 1, &input)
            .expect("real F32 row should run");
        assert!((actual[0] - expected).abs() < 1.0e-4);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn q4_k_embedding_runs_against_real_gguf_token() {
        let Ok(model_path) = std::env::var("SI_GGUF_MODEL") else {
            return;
        };
        let store = GgufModelStore::open(model_path).expect("GGUF model should open");
        let info = store
            .tensors
            .get("token_embd.weight")
            .expect("GGUF model should contain token embeddings");
        assert_eq!(info.ggml_type, crate::quant::GGML_TYPE_Q4_K);
        let tensor = store
            .tensor("token_embd.weight")
            .expect("token embedding tensor should open");
        let columns = info.shape[0];
        let row_bytes = columns / 256 * crate::quant::Q4_K_BLOCK_BYTES;
        let expected = crate::quant::dequantize_q4_k(&tensor.bytes[..row_bytes], columns)
            .expect("token embedding row should decode");
        let context = super::MetalContext::new().expect("Metal context should compile");
        let actual = context
            .q4_k_embedding_tensor(&tensor, 0)
            .expect("real Q4_K embedding row should run");
        for (index, (value, reference)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (value - reference).abs() < 1.0e-5,
                "index {index}: {value} != {reference}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn bf16_bitpack_matvec_kernel_matches_bf16_reference() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let rows = 3;
        let columns = 4;
        let tile_rows = 2;
        let weights = [
            1.0_f32, 0.5, -1.0, 2.0, -2.0, 1.0, 0.25, 3.0, 0.125, -0.5, 2.0, -1.0,
        ];
        let input = [2.0_f32, -1.0, 0.5, 3.0];
        let weight_bits = weights
            .iter()
            .map(|value| (value.to_bits() >> 16) as u16)
            .collect::<Vec<_>>();
        let mut packed = Vec::new();
        let mut offsets = Vec::new();
        for row_start in (0..rows).step_by(tile_rows) {
            offsets.push(packed.len() as u32);
            let values =
                &weight_bits[row_start * columns..(row_start + tile_rows).min(rows) * columns];
            let first = values[0];
            let mut invariant = 0_u16;
            let mut constants = 0_u16;
            for bit in 0..16 {
                let mask = 1_u16 << bit;
                let set = first & mask;
                if values.iter().all(|value| value & mask == set) {
                    invariant |= mask;
                    constants |= set;
                }
            }
            let variable_bits = 16 - invariant.count_ones();
            packed.extend_from_slice(&invariant.to_le_bytes());
            packed.extend_from_slice(&constants.to_le_bytes());
            let mut accumulator = 0_u8;
            let mut available = 0_u8;
            for value in values {
                for bit in 0..16 {
                    let mask = 1_u16 << bit;
                    if invariant & mask != 0 {
                        continue;
                    }
                    if value & mask != 0 {
                        accumulator |= 1 << available;
                    }
                    available += 1;
                    if available == 8 {
                        packed.push(accumulator);
                        accumulator = 0;
                        available = 0;
                    }
                }
            }
            if available > 0 {
                packed.push(accumulator);
            }
            assert_eq!(
                packed.len() - offsets.last().copied().unwrap() as usize,
                4 + (values.len() * variable_bits as usize).div_ceil(8)
            );
        }
        offsets.push(packed.len() as u32);
        let actual = context
            .bf16_bitpack_matvec(&packed, &offsets, rows, columns, tile_rows, &input)
            .expect("packed BF16 matvec should run");
        let expected = context
            .bf16_matvec(
                &weight_bits
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>(),
                rows,
                columns,
                &input,
            )
            .expect("reference BF16 matvec should run");
        for (index, (value, reference)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (value - reference).abs() < 1.0e-5,
                "index {index}: {value} != {reference}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn aligned_weight_range_preserves_tensor_offset() {
        let (base, offset, length) = super::aligned_weight_range(0x12_340, 0x12_456, 300, 0x1000)
            .expect("aligned range should fit inside backing mapping");
        assert_eq!(base, 0x12_000);
        assert_eq!(offset, 0x456);
        assert_eq!(length, 0x1000);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn async_matvec_submission_can_be_waited_without_changing_results() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let weights = [1.0_f32, 0.5, -1.0, 2.0, -2.0, 1.0, 0.25, 3.0];
        let input = [2.0_f32, -1.0, 0.5, 3.0];
        let weight_bytes = weights
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        let weight = context
            .upload_bf16_weight(&weight_bytes)
            .expect("weight should upload");

        let pending = context
            .bf16_matvec_buffer_async(&weight, 2, 4, &input)
            .expect("async matvec should submit");
        assert_ne!(pending.status(), metal::MTLCommandBufferStatus::NotEnqueued);
        let actual = pending.wait().expect("async matvec should complete");
        assert_eq!(actual.len(), 1);
        assert_eq!(actual[0].len(), 2);
        assert_eq!(context.command_stats().submitted, 1);
        assert_eq!(context.command_stats().async_submitted, 1);
        assert_eq!(context.command_stats().waited, 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn fused_qkv_kernel_matches_three_independent_matvecs() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let input = [2.0_f32, -1.0, 0.5, 3.0];
        let weights = [
            vec![1.0_f32, 0.5, -1.0, 2.0, -2.0, 1.0, 0.25, 3.0],
            vec![0.125_f32, -0.5, 2.0, -1.0],
            vec![
                0.25_f32, 1.5, -0.75, 2.5, -1.0, 0.5, 1.25, -2.0, 0.75, 0.25, -1.5, 1.0,
            ],
        ];
        let bytes = weights
            .iter()
            .map(|matrix| {
                matrix
                    .iter()
                    .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let buffers = bytes
            .iter()
            .map(|matrix| {
                context
                    .upload_bf16_weight(matrix)
                    .expect("weight should upload")
            })
            .collect::<Vec<_>>();
        let actual = context
            .bf16_fused_qkv_buffer(
                &[
                    (&buffers[0], 2, 4),
                    (&buffers[1], 1, 4),
                    (&buffers[2], 3, 4),
                ],
                &input,
            )
            .expect("fused QKV should run");
        for (matrix, (rows, output)) in weights.iter().zip([2, 1, 3].into_iter().zip(actual)) {
            for row in 0..rows {
                let expected = (0..4)
                    .map(|column| matrix[row * 4 + column] * input[column])
                    .sum::<f32>();
                assert!((output[row] - expected).abs() < 1.0e-5);
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn fused_gate_up_kernel_matches_two_independent_matvecs() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let input = [2.0_f32, -1.0, 0.5, 3.0];
        let weights = [
            vec![1.0_f32, 0.5, -1.0, 2.0, -2.0, 1.0, 0.25, 3.0],
            vec![0.125_f32, -0.5, 2.0, -1.0, 0.25, 1.5, -0.75, 2.5],
        ];
        let bytes = weights
            .iter()
            .map(|matrix| {
                matrix
                    .iter()
                    .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let buffers = bytes
            .iter()
            .map(|matrix| {
                context
                    .upload_bf16_weight(matrix)
                    .expect("weight should upload")
            })
            .collect::<Vec<_>>();
        let actual = context
            .bf16_fused_gate_up_buffer(&[(&buffers[0], 2, 4), (&buffers[1], 2, 4)], &input)
            .expect("fused gate/up should run");
        for (matrix, output) in weights.iter().zip(actual) {
            for row in 0..2 {
                let expected = (0..4)
                    .map(|column| matrix[row * 4 + column] * input[column])
                    .sum::<f32>();
                assert!((output[row] - expected).abs() < 1.0e-5);
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn async_owned_fused_gate_up_keeps_staged_bytes_alive() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let input = [2.0_f32, -1.0, 0.5, 3.0];
        let matrices = [
            vec![1.0_f32, 0.5, -1.0, 2.0, -2.0, 1.0, 0.25, 3.0],
            vec![0.125_f32, -0.5, 2.0, -1.0, 0.25, 1.5, -0.75, 2.5],
        ]
        .into_iter()
        .map(|matrix| {
            (
                matrix
                    .iter()
                    .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
                    .collect::<Vec<_>>(),
                2,
                4,
            )
        })
        .collect::<Vec<_>>();
        let pending = context
            .bf16_fused_gate_up_owned_bytes_async(matrices, &input)
            .expect("owned staged gate/up should submit");
        let actual = pending
            .wait()
            .expect("owned staged gate/up should complete");
        assert_eq!(actual.len(), 2);
        assert_eq!(actual[0].len(), 2);
        assert_eq!(actual[1].len(), 2);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn chained_qkv_attention_o_matches_separate_gpu_operations() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let input = [0.5_f32, -1.0, 2.0, 0.25];
        let matrices = [
            (
                "q",
                vec![
                    1.0_f32, 0.5, -1.0, 2.0, -2.0, 1.0, 0.25, 3.0, 0.75, -0.5, 2.0, -1.0, 1.5,
                    0.25, -0.75, 0.5,
                ],
                vec![4, 4],
            ),
            (
                "k",
                vec![0.25_f32, 1.5, -0.75, 2.5, -1.0, 0.5, 1.25, -2.0],
                vec![2, 4],
            ),
            (
                "v",
                vec![0.5_f32, -1.25, 0.75, 1.0, 1.5, 0.25, -0.5, 2.0],
                vec![2, 4],
            ),
            (
                "o",
                vec![
                    1.0_f32, -0.5, 0.25, 2.0, 0.5, 1.25, -1.0, 0.75, -0.75, 0.5, 2.0, -1.5, 1.5,
                    -0.25, 0.5, 1.0,
                ],
                vec![4, 4],
            ),
        ];
        let bytes = matrices
            .iter()
            .map(|(_, values, _)| {
                values
                    .iter()
                    .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let infos = matrices
            .iter()
            .zip(&bytes)
            .map(|((name, _, shape), bytes)| TensorInfo {
                name: (*name).into(),
                shard: "tiny.safetensors".into(),
                dtype: "BF16".into(),
                shape: shape.clone(),
                data_start: 0,
                data_end: bytes.len(),
            })
            .collect::<Vec<_>>();
        let tensors = infos
            .iter()
            .zip(&bytes)
            .map(|(info, bytes)| TensorView {
                info,
                bytes,
                backing: bytes,
            })
            .collect::<Vec<_>>();
        let q_norm = [1.0_f32, 0.75];
        let k_norm = [0.5_f32, 1.25];
        let q_norm_bytes = q_norm
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        let k_norm_bytes = k_norm
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        let key_cache = vec![0.0_f32; 4];
        let value_cache = vec![0.0_f32; 4];
        let chained = context
            .chained_qkv_attention_o_tensors(super::ChainedAttentionTensorRequest {
                q_tensor: &tensors[0],
                k_tensor: &tensors[1],
                v_tensor: &tensors[2],
                o_tensor: &tensors[3],
                q_norm_bytes: &q_norm_bytes,
                k_norm_bytes: &k_norm_bytes,
                input: &input,
                key_cache: &key_cache,
                value_cache: &value_cache,
                config: super::ChainedAttentionConfig {
                    query_heads: 2,
                    key_value_heads: 1,
                    head_dim: 2,
                    cached_tokens: 0,
                    cache_capacity_tokens: 2,
                    position: 0,
                    rope_theta: 10_000.0,
                    epsilon: 1.0e-5,
                },
            })
            .expect("chained layer should run");

        let qkv = context
            .bf16_fused_qkv_tensors(&[&tensors[0], &tensors[1], &tensors[2]], &input)
            .expect("separate QKV should run");
        let query = context
            .rms_norm_bf16_heads(&qkv[0], &q_norm_bytes, 2, 2, 1.0e-5)
            .expect("separate Q norm should run");
        let keys = context
            .rms_norm_bf16_heads(&qkv[1], &k_norm_bytes, 1, 2, 1.0e-5)
            .expect("separate K norm should run");
        let query = context
            .rope(&query, 2, 2, 0, 10_000.0)
            .expect("separate Q RoPE should run");
        let keys = context
            .rope(&keys, 1, 2, 0, 10_000.0)
            .expect("separate K RoPE should run");
        let attended = context
            .attention_decode(super::AttentionDecodeInput {
                query: &query,
                key_cache: &key_cache,
                value_cache: &value_cache,
                new_keys: &keys,
                new_values: &qkv[2],
                query_heads: 2,
                key_value_heads: 1,
                head_dim: 2,
                cached_tokens: 0,
                cache_capacity_tokens: 2,
            })
            .expect("separate attention should run");
        let projected = context
            .bf16_matvec_tensor(&tensors[3], &attended)
            .expect("separate output projection should run");

        for (actual, expected) in chained.projected.iter().zip(projected) {
            assert!((actual - expected).abs() < 1.0e-4, "{actual} != {expected}");
        }
        for (actual, expected) in chained.keys.iter().zip(keys) {
            assert!((actual - expected).abs() < 1.0e-4, "{actual} != {expected}");
        }
        for (actual, expected) in chained.values.iter().zip(&qkv[2]) {
            assert!((actual - expected).abs() < 1.0e-4, "{actual} != {expected}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn retained_bf16_weight_buffer_matches_upload_per_call_path() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let weights = [1.0_f32, 0.5, -1.0, 2.0, 0.25, 3.0];
        let input = [2.0_f32, -1.0];
        let weight_bytes = weights
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        let retained = context
            .upload_bf16_weight(&weight_bytes)
            .expect("weight should upload");
        let actual = context
            .bf16_matvec_buffer(&retained, 3, 2, &input)
            .expect("retained matvec should run");
        let expected = (0..3)
            .map(|row| {
                (0..2)
                    .map(|column| weights[row * 2 + column] * input[column])
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        for (index, (value, reference)) in actual.iter().zip(expected).enumerate() {
            assert!((value - reference).abs() < 1.0e-5, "index {index}");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn chunked_bf16_matvec_matches_full_reference() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let weights = [1.0_f32, 0.5, -1.0, 2.0, 0.25, 3.0];
        let input = [2.0_f32, -1.0];
        let weight_bytes = weights
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        let header = br#"{"weight":{"dtype":"BF16","shape":[3,2],"data_offsets":[0,12]}}"#;
        let mut shard = (header.len() as u64).to_le_bytes().to_vec();
        shard.extend_from_slice(header);
        shard.extend_from_slice(&weight_bytes);
        let data_start = 8 + header.len();
        let info = crate::model::TensorInfo {
            name: "weight".into(),
            shard: "tiny.safetensors".into(),
            dtype: "BF16".into(),
            shape: vec![3, 2],
            data_start,
            data_end: data_start + weight_bytes.len(),
        };
        let tensor = crate::model::TensorView {
            info: &info,
            bytes: &shard[data_start..],
            backing: &shard,
        };
        let full = context
            .bf16_matvec_tensor(&tensor, &input)
            .expect("full matvec should run");
        let chunked = context
            .bf16_matvec_tensor_chunked(&tensor, &input, 2)
            .expect("chunked matvec should run");
        assert_eq!(full, chunked);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn bf16_matvec_accepts_real_qwen_tensor_when_requested() {
        let Ok(model_dir) = std::env::var("SI_MODEL_DIR") else {
            return;
        };
        let store = crate::model::ModelStore::open(model_dir, false).expect("model should parse");
        let tensor = store
            .tensor("model.layers.0.self_attn.q_proj.weight")
            .expect("Qwen q projection should exist");
        let input = vec![0.0_f32; tensor.info.shape[1]];
        let output = super::MetalContext::new()
            .expect("Metal context should compile")
            .bf16_matvec_tensor(&tensor, &input)
            .expect("real BF16 tensor should upload");
        assert_eq!(output.len(), tensor.info.shape[0]);
        assert!(output.iter().all(|value| *value == 0.0));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn bf16_embedding_row_kernel_matches_cpu_reference() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let values = [1.0_f32, -2.0, 0.5, 3.0];
        let row_bytes = values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect::<Vec<_>>();
        let actual = context
            .bf16_embedding_row(&row_bytes, values.len())
            .expect("embedding row should run");
        for (index, (value, reference)) in actual.iter().zip(values).enumerate() {
            assert!(
                (value - reference).abs() < 1.0e-5,
                "index {index}: {value} != {reference}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn rope_kernel_matches_cpu_reference() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let position = 3;
        let theta = 1_000_000.0_f32;
        let actual = context
            .rope(&input, 1, 4, position, theta)
            .expect("RoPE should run");
        let angle0 = position as f32;
        let angle1 = position as f32 * theta.powf(-0.5);
        let expected = [
            input[0] * angle0.cos() - input[1] * angle0.sin(),
            input[0] * angle0.sin() + input[1] * angle0.cos(),
            input[2] * angle1.cos() - input[3] * angle1.sin(),
            input[2] * angle1.sin() + input[3] * angle1.cos(),
        ];
        for (index, (value, reference)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (value - reference).abs() < 1.0e-5,
                "index {index}: {value} != {reference}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn attention_decode_kernel_matches_cpu_reference() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let query_heads = 4;
        let key_value_heads = 2;
        let head_dim = 2;
        let cached_tokens = 2;
        let query = [
            1.0_f32, 0.0, // q head 0 -> kv head 0
            0.0, 1.0, // q head 1 -> kv head 0
            1.0, 1.0, // q head 2 -> kv head 1
            1.0, -1.0, // q head 3 -> kv head 1
        ];
        let keys = [
            1.0_f32, 0.0, 0.0, 1.0, // token 0, kv heads 0/1
            1.0, 1.0, 1.0, -1.0, // token 1, kv heads 0/1
        ];
        let values = [
            10.0_f32, 0.0, 0.0, 20.0, // token 0, kv heads 0/1
            30.0, 0.0, 0.0, 40.0, // token 1, kv heads 0/1
        ];
        let new_keys = [1.0_f32, -1.0, 1.0, 1.0];
        let new_values = [50.0_f32, 0.0, 0.0, 60.0];
        let actual = context
            .attention_decode(super::AttentionDecodeInput {
                query: &query,
                key_cache: &keys,
                value_cache: &values,
                new_keys: &new_keys,
                new_values: &new_values,
                query_heads,
                key_value_heads,
                head_dim,
                cached_tokens,
                cache_capacity_tokens: cached_tokens,
            })
            .expect("attention decode should run");
        let scale = (head_dim as f32).sqrt().recip();
        let mut expected = Vec::new();
        for query_head in 0..query_heads {
            let kv_head = query_head / (query_heads / key_value_heads);
            let query_row = &query[query_head * head_dim..(query_head + 1) * head_dim];
            let mut scores = Vec::new();
            for token in 0..=cached_tokens {
                let key = if token < cached_tokens {
                    &keys[(kv_head * cached_tokens + token) * head_dim
                        ..(kv_head * cached_tokens + token + 1) * head_dim]
                } else {
                    &new_keys[kv_head * head_dim..(kv_head + 1) * head_dim]
                };
                scores.push(
                    query_row
                        .iter()
                        .zip(key)
                        .map(|(query_value, key_value)| query_value * key_value)
                        .sum::<f32>()
                        * scale,
                );
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights = scores
                .iter()
                .map(|score| (*score - max_score).exp())
                .collect::<Vec<_>>();
            let normalizer = weights.iter().sum::<f32>();
            for dimension in 0..head_dim {
                let mut value = 0.0;
                for token in 0..=cached_tokens {
                    let values_row = if token < cached_tokens {
                        &values[(kv_head * cached_tokens + token) * head_dim
                            ..(kv_head * cached_tokens + token + 1) * head_dim]
                    } else {
                        &new_values[kv_head * head_dim..(kv_head + 1) * head_dim]
                    };
                    value += weights[token] / normalizer * values_row[dimension];
                }
                expected.push(value);
            }
        }
        for (index, (value, reference)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (value - reference).abs() < 1.0e-5,
                "index {index}: {value} != {reference}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn attention_decode_uses_capacity_strided_kv_cache_without_repacking() {
        let context = super::MetalContext::new().expect("Metal context should compile");
        let mut cache = crate::cache::KvCache::new(1, 2, 4).expect("valid cache");
        cache
            .append_token(&[1.0, 0.0], &[10.0, 0.0])
            .expect("first token fits");
        cache
            .append_token(&[0.0, 1.0], &[20.0, 0.0])
            .expect("second token fits");
        let query = [1.0_f32, 1.0];
        let new_keys = [1.0_f32, -1.0];
        let new_values = [30.0_f32, 0.0];
        let direct = context
            .attention_decode(super::AttentionDecodeInput {
                query: &query,
                key_cache: cache.key_storage(),
                value_cache: cache.value_storage(),
                new_keys: &new_keys,
                new_values: &new_values,
                query_heads: 1,
                key_value_heads: 1,
                head_dim: 2,
                cached_tokens: 2,
                cache_capacity_tokens: 4,
            })
            .expect("capacity-strided attention should run");
        let via_cache = context
            .attention_decode_kv_cache(&query, &cache, &new_keys, &new_values, 1)
            .expect("cache adapter should run");
        assert_eq!(direct.len(), via_cache.len());
        for (index, (direct_value, cache_value)) in direct.iter().zip(via_cache).enumerate() {
            assert!(
                (direct_value - cache_value).abs() < 1.0e-6,
                "index {index}: {direct_value} != {cache_value}"
            );
        }
    }
}
