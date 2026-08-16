# SI-002: Direct Metal Runtime

Status: proposed

## Goal

Execute Qwen3-4B-Base directly from Rust and Metal on the M3 Pro, retaining
original BF16 weight bytes and exposing explicit buffer lifetimes to the SI
memory planner.

## Runtime boundaries

Rust owns CLI, model metadata, Safetensors mapping, tokenizer, scheduling,
telemetry, and report serialization. Metal owns tensor kernels and GPU command
submission. Runtime must not silently fall back to a different model,
precision, or backend.

The backend exposes these logical operations:

- load model metadata and tensor descriptors;
- tokenize/de-tokenize one request;
- prefill a token sequence;
- decode one token at a time;
- return logits or selected next-token scores;
- release or yield weight, KV, activation, and scratch leases;
- report device and allocation telemetry.

## Model execution

Implement Qwen3-4B-Base in this order:

1. token embeddings and final projection;
2. RMSNorm;
3. rotary position embeddings;
4. grouped-query attention with causal masking;
5. gated MLP/SwiGLU;
6. residual connections and final normalization;
7. greedy next-token selection.

The first reference profile may keep all tensors resident. It must still use the
same tensor descriptors and leases that later streaming profiles use. The
`metal-streaming` path binds each mapped Safetensors matrix with a Metal
no-copy shared buffer for the duration of its synchronous command, while
`metal-resident` retains all linear and embedding BF16 buffers. An explicit
`--retain-output-head` experiment retains only the tied output head in private
Metal storage; `--retain-layers N` retains a bounded prefix of transformer
linear weights. All profiles exercise one full causal token through all 36
Qwen3-4B blocks and the tied embedding projection; they are correctness
baselines, not quality claims.

The current kernel foundation includes correctness shaders for RMSNorm, row-major
BF16 matrix-vector multiplication, BF16 embedding-row lookup, and RoPE. The
resident path now uses tiled/vectorized BF16 GEMV and executes small
norm/RoPE/attention operations on the CPU to avoid synchronous launch overhead;
the source tensor path remains the correctness reference. The streaming path
uses mapped source bytes without a duplicate weight allocation; its report
separates logical active-weight bytes from Metal allocator bytes. Attention
decode
uses an explicit
GQA cache layout `[kv_head, capacity_token, head_dim]` plus an active-token
count, appends one new K/V row logically, and performs numerically stable
softmax per query head. Unused capacity slots are never read, so the cache can
be reused without repacking or reallocating between decode steps.

## Precision rules

Source weight files remain BF16 and immutable. Kernels may accumulate in FP32
where required by Metal, but any conversion must be explicit in the report.
No silent FP16/INT8 conversion is allowed in the lossless profile. If the M3
Metal toolchain cannot consume BF16 directly, stop with a capability error and
record the limitation rather than changing weights implicitly.

## Kernel and synchronization rules

- Use explicit command queues, command buffers, completion points, and buffer
  lifetimes.
- Avoid per-token heap allocation.
- Keep temporary activation buffers reusable and bounded by declared shapes.
- Make synchronization visible to the scheduler so transfer and execution can
  be overlapped later.
- Keep a small CPU reference path for unit tests and tensor-level diagnostics;
  it is not the performance backend.

## Correctness gates

Before benchmarking speed, the backend must pass tensor-level tests for norms,
RoPE, attention masking, GQA head layout, MLP gating, and logits. End-to-end
results must pass the SI-001 quality suite against the full-resident reference.
