# Super-Inference

Terminal-first research harness for measuring Super-Inference memory/throughput experiments.

## At a glance

Super-Inference explores how to run large local language models under a deliberately
constrained GPU-memory budget while preserving exact greedy behavior from the
untouched target model.

| Area | Status |
| --- | --- |
| Rust runtime | Working |
| Metal kernels | Working |
| Safetensors | Working |
| GGUF / Q4_K and related formats | Working |
| Qwen3-4B | End-to-end |
| Qwen3.6-27B | Active development |
| Research benchmark suite | Reproducible |

The project is an experimental inference engine, not a finished drop-in
replacement for llama.cpp. The research record below documents what works,
what does not, and why.

Research contracts and implementation stages live in [`docs/`](docs/), beginning
with [SI-001](docs/SI-001-benchmark-spec.md), [SI-002](docs/SI-002-direct-metal-runtime-spec.md),
[SI-003](docs/SI-003-memory-planner-spec.md), and
[SI-004](docs/SI-004-throughput-recovery-spec.md).

The initial quality fixture is [`fixtures/quality-v0.json`](fixtures/quality-v0.json),
and the model integrity metadata used for local runs is kept with the local
model artifact rather than committed to this repository.
The frozen SI-001 retain-8 reference and promotion gates are recorded in
[`benchmarks/si-001-baseline-lock-2026-08-10.json`](benchmarks/si-001-baseline-lock-2026-08-10.json).

Model weights are intentionally not included. Download a compatible model
separately and pass its local path to the probes; the benchmark captures and
research record preserve the model revisions, settings, and measured results.

## Current scope

`si-bench` includes a deterministic `mock` backend plus opt-in
`metal-streaming` and `metal-resident` reference profiles. Both load the pinned
model without altering its BF16 weights and execute one causal token path
through Metal; both report logical active weights, sampled Metal allocation, and
process RSS. Streaming binds mapped Safetensors bytes directly; add
`--retain-output-head` to retain only the tied output head as a bounded hot
tensor, or `--retain-layers N` to retain the first N transformer layers.

The primary SI-001 test object was the original BF16 Qwen3-4B-Base artifact;
later experiments also exercised the Qwen3.6-27B Q4_K_M GGUF artifact. Neither
model is distributed by this repository.

Model containers are intentionally dual-format: the existing BF16 Safetensors
runtime remains supported, and `GgufModelStore` now detects and mmap-indexes
GGUF artifacts such as `models/qwen3.6-27b-q4_k_m/Qwen3.6-27B-Q4_K_M.gguf`.
GGUF metadata is normalized into a validated Qwen3.6 configuration, Q4_K
blocks have a tested deterministic decoder, and the matching `tokenizer.json`
sidecar is supported. The Metal backend now also exposes an exact, no-copy
Q4_K/Q5_K/Q6_K and F32 matvec surface, plus a no-copy Q4_K token-embedding
lookup. Exact Qwen3.6 Gated DeltaNet causal-convolution, recurrent-state, and
gated RMSNorm primitives are covered by CPU/Metal tests, and
`si-qwen35-gdn-probe` exercises one real Gated DeltaNet token-mixer path
(quantized projections, F32 gates, recurrent state, gated normalization, and
output projection). The surrounding decoder-block residual/MLP path and full
64-layer DeltaNet/attention runtime are next. `si-qwen35-attn-probe` now
exercises the corresponding quantized full-attention token mixer with partial
RoPE, KV attention, and output projection. The Q4_K SwiGLU gate/up path is
also fused and measured by `si-qwen35-mlp-probe`.
`qwen35_runtime::qwen35_decoder_block` now chains the mixer, both residuals,
Qwen RMSNorms, and MLP for one-token empty-cache block validation; the block
probe exercises both a DeltaNet layer and a full-attention layer.
`Qwen35LayerState` preserves DeltaNet convolution/recurrent state and bounded
full-attention KV state across calls; `si-qwen35-state-probe` exercises that
multi-token path. `Qwen35Runtime` now owns all 64 layer states, runs the full
sequential stack, and can stream the exact output head for a greedy token ID;
`si-qwen35-runtime-probe` measures that end-to-end path.

## Run

```bash
cargo test
cargo run --release --bin si-bench -- --model /models/example.gguf --backend mock --prompt "Explain memory paging" --max-tokens 16 --output text
cargo run --release --bin si-bench -- --model /models/example.gguf --backend mock --prompt "test" --max-tokens 2 --expected-output "mock-0 mock-1" --output json
cargo run --release --bin si-metal-probe
SI_GGUF_MODEL=models/qwen3.6-27b-q4_k_m/Qwen3.6-27B-Q4_K_M.gguf \
  cargo run --release --bin si-gguf-inspect -- \
  --model models/qwen3.6-27b-q4_k_m/Qwen3.6-27B-Q4_K_M.gguf
cargo run --release --bin si-q4-probe -- \
  --model models/qwen3.6-27b-q4_k_m/Qwen3.6-27B-Q4_K_M.gguf \
  --tensor blk.0.attn_gate.weight --repetitions 3
cargo run --release --bin si-qwen35-gdn-probe -- \
  --model models/qwen3.6-27b-q4_k_m/Qwen3.6-27B-Q4_K_M.gguf \
  --layer 0 --repetitions 3
cargo run --release --bin si-qwen35-attn-probe -- \
  --model models/qwen3.6-27b-q4_k_m/Qwen3.6-27B-Q4_K_M.gguf \
  --layer 3 --position 0 --repetitions 3
cargo run --release --bin si-qwen35-mlp-probe -- \
  --model models/qwen3.6-27b-q4_k_m/Qwen3.6-27B-Q4_K_M.gguf \
  --layer 0 --repetitions 3
cargo run --release --bin si-qwen35-block-probe -- \
  --model models/qwen3.6-27b-q4_k_m/Qwen3.6-27B-Q4_K_M.gguf \
  --layer 0 --repetitions 3
cargo run --release --bin si-qwen35-state-probe -- \
  --model models/qwen3.6-27b-q4_k_m/Qwen3.6-27B-Q4_K_M.gguf \
  --layer 3 --tokens 8 --capacity 8
cargo run --release --bin si-qwen35-runtime-probe -- \
  --model models/qwen3.6-27b-q4_k_m/Qwen3.6-27B-Q4_K_M.gguf \
  --token 42 --tokens 1 --capacity 8 --output-head
# Bounded Qwen3.6 residency: cacheable packed staging (depth 3 by default),
# retain an MLP hot set, and overlap layer-0 staging with the output head.
SI_STAGE_PIPELINE=1 SI_RETAIN_MLP_STRIDE=4 SI_RETAIN_MLP_OFFSET=1 \
  cargo run --release --bin si-bench -- \
  --model models/qwen3.6-27b-q4_k_m/Qwen3.6-27B-Q4_K_M.gguf \
  --backend metal-streaming --prompt Hello --max-tokens 2 --context 8 \
  --warmup 0 --repetitions 1 --retain-output-head --output json
# On macOS, compare low-residency streaming with retained-weight residency.
cargo run --release --bin si-bench -- --model models/qwen3-4b-base --backend metal-streaming --prompt "Hello" --max-tokens 1 --context 8 --warmup 0 --output json
cargo run --release --bin si-bench -- --model models/qwen3-4b-base --backend metal-streaming --prompt "Hello" --max-tokens 1 --context 8 --warmup 0 --retain-output-head --output json
cargo run --release --bin si-bench -- --model models/qwen3-4b-base --backend metal-streaming --prompt "Hello" --max-tokens 1 --context 8 --warmup 0 --retain-layers 8 --output json
cargo run --release --bin si-bench -- --model models/qwen3-4b-base --backend metal-streaming --prompt "Hello" --max-tokens 1 --context 8 --chunk-rows 16384 --warmup 0 --output json
cargo run --release --bin si-bench -- --model models/qwen3-4b-base --backend metal-resident --prompt "Hello" --max-tokens 1 --context 8 --warmup 0 --output json
# Add --quality-fixture fixtures/quality-v0.json for the slow opt-in suite.
# Canonical SI-001 resident reference (3 measured repetitions):
cargo run --release --bin si-bench -- --model models/qwen3-4b-base --backend metal-resident --prompt "Explain why memory paging is useful for local model inference." --max-tokens 128 --context 2048 --warmup 1 --repetitions 3 --verify-manifest --quality-fixture fixtures/quality-v0.json --output json

# SI-004 exact batched-projection and target-forward probes
SI_VERIFY_MANY=1 cargo run --release --bin si-matmul-many-probe -- --model models/qwen3-4b-base --verify-manifest
SI_VERIFY_MANY=1 cargo run --release --bin si-verify-many-probe -- --model models/qwen3-4b-base --verify-manifest --prompt Hello --retain-layers 8 --warmup 1 --repetitions 3
```

`si-verify-many-probe` is an SI-004 candidate-forward diagnostic, not a
canonical generated-token benchmark. It compares separate target calls with
the exact K=1/2/4/8 path; `SI_VERIFY_MANY=1` keeps the exploratory Metal
pipeline out of the default SI-001 runtime.

For K=4/K=8, the candidate path also fuses batched QKV and gate/up projections;
the current retain-8 control reaches 6.757/16.622 candidate tok/s while
preserving greedy IDs. K=1/K=2 remain on the slower general batched path.

The first end-to-end lookahead control is also opt-in:

```bash
SI_LOOKAHEAD=1 SI_LOOKAHEAD_WIDTH=8 SI_LOOKAHEAD_ITERS=2 \
  cargo run --release --bin si-bench -- --model models/qwen3-4b-base \
  --backend metal-streaming --prompt Hello --max-tokens 16 \
  --context 2048 --retain-layers 8 --warmup 1 --repetitions 3 --output json
```

This Jacobi control is currently rejected for promotion because its candidate
acceptance is too low; see the SI-004 research artifact for the measured result.
Set `SI_PROFILE_LOOKAHEAD=1` to print accepted tokens per target step. The
optional `SI_LOOKAHEAD_DYNAMIC=1` controller shrinks the next window after a
low-acceptance step and grows it only after a fully accepted window; it is an
economic diagnostic, not part of the canonical path.
same-model partial drafter variant adds `SI_LOOKAHEAD_DRAFT_LAYERS=8`; it is
also exploratory and currently rejected.

For the external-drafter experiment, set `SI_DRAFT_MODEL` to a separate
same-vocabulary model directory. `SI_DRAFT_RESIDENT=1` keeps that drafter's
weights private on Metal, and `SI_DRAFT_RETAIN_LAYERS=N` retains only its
prefix. These controls are exploratory; the current 0.6B BF16 drafter does
not beat the canonical target-only baseline under the memory contract.
`SI_DRAFT_SHARED=1` is the unified-memory variant: it keeps the drafter's
weights resident in shared CPU/GPU-visible storage instead of private Metal
storage. Use it with target retention disabled when testing the combined
working-set ceiling.

`SI_LOOKAHEAD_NGRAM=1` enables a CPU-only repeated-suffix proposal source;
`SI_LOOKAHEAD_NGRAM_ORDER=N` controls its maximum history order. It is an
exact, opt-in control and is currently exploratory.

The shared-weight tree verifier is enabled with `SI_TREE=1` together with
`SI_LOOKAHEAD=1`. `SI_TREE_BRANCHES` (2--4), `SI_TREE_DEPTH` (2--4), and
`SI_TREE_NGRAM_ORDER` control the bounded branch shape; the product is capped
at eight candidate positions so one target traversal can verify the whole
tree. `SI_TREE_MEMORY_MIB` defaults to 2,000 MiB and is hard-capped at 2,048
MiB; both sampled Metal allocation and RSS must remain below that ceiling.
The first exact tree control is recorded in the SI-004 research artifact but
is rejected for promotion because its n-gram branches accept too few suffix
tokens.

`SI_EXACT_HEAD=1` enables the lossless output-head block-bound diagnostic. It
preserves the exact greedy argmax while returning sparse logits for skipped
rows, so it is incompatible with the quality fixture. The first contiguous
vocabulary-block index evaluated 593 of 594 blocks and was rejected; see the
SI-004 benchmark artifact.

The lossless codec probe measures exact block round trips without changing
inference:

```bash
cargo run --release --bin si-lossless-probe -- \
  --model models/qwen3-4b-base --block-kib 64 --verify-manifest
```

The initial zlib and invariant-bit results are recorded in SI-004. The
row-aligned fused Metal decoder was exact but 3.0x–5.1x slower than the
existing mapped BF16 matvec, so it is retained as a rejected diagnostic and
is not integrated into the runtime.

## Resident-layer drafter trace

The retained-prefix trace probe pairs the hidden state after eight resident
layers with exact top-1/top-4 IDs from the untouched target. It is a training
artifact only; speculative tokens still require exact target verification.

```bash
cargo run --release --bin si-resident-drafter-probe -- \
  --model models/qwen3-4b-base --layers 8 --tokens 128 \
  --context 2048 --verify-manifest \
  --trace benchmarks/si-005-resident-drafter-canonical-2026-08-10.jsonl
```

Use `--prompt-file PATH` to collect one header and trace window per non-empty
line, which is useful for a broader held-out sidecar suite. `--prompt` and
`--prompt-file` are mutually exclusive.

The initial pre-normalization sidecar sweep is recorded in
[`si-005-resident-sidecar-sweep`](benchmarks/si-005-resident-sidecar-sweep-2026-08-10.json).
The corrected normalized-target sweep is recorded in
[`si-005-resident-sidecar-normalized-sweep`](benchmarks/si-005-resident-sidecar-normalized-sweep-2026-08-10.json).
Both are proposal-only and are not loaded by the SI runtime by default.

The normalized sidecar can be exercised end-to-end (still opt-in and exact):

```bash
SI_LOOKAHEAD=1 SI_LOOKAHEAD_WIDTH=2 SI_LOOKAHEAD_DRAFT_LAYERS=8 \
SI_DRAFT_SIDECAR=benchmarks/si-005-resident-sidecar-r128-normalized.sisc \
cargo run --release --bin si-bench -- --model models/qwen3-4b-base \
  --backend metal-streaming --prompt Hello --max-tokens 16 --context 2048 \
  --retain-layers 8 --warmup 0 --repetitions 1 --verify-manifest --output json
```

Every sidecar proposal is verified by the untouched target; the first runtime
control is recorded in [`si-005-resident-sidecar-e2e-controls`](benchmarks/si-005-resident-sidecar-e2e-controls-2026-08-10.json).
`SI_DRAFT_SIDECAR_TREE=1` enables the rejected four-branch tree diagnostic;
`SI_DRAFT_SIDECAR_AUTO=1` enables the conservative canonical fallback after a
low-acceptance sidecar step; `SI_DRAFT_SIDECAR_MIN_MARGIN=N` gates K-way
verification on the sidecar's top-1/top-2 score margin. All are exploratory
controls.

An expected-output mismatch exits with status `2`. CLI errors exit with status `1`. JSON output is one object per invocation, designed for JSONL benchmark collection.

## Measurement contract for real backends

Every backend report must retain the same fields:

- model identifier/path and backend;
- prompt/generated token counts and warmup count;
- prefill, decode, and total throughput;
- peak VRAM and RAM in MiB, sampled from real allocator/device telemetry;
- mapped/active weight, KV, and scratch byte counters;
- optional versioned quality summary;
- optional output-equivalence result against a fixed reference.

Compare configurations only with identical model weights, prompt suite, generation settings, context length, and hardware. A later backend should define its numerical equivalence tolerance explicitly; bit-for-bit equality is not guaranteed across hardware/kernel implementations.

`si-metal-probe` confirms the native device and memory limits used by the
direct-Metal backend. The current loader validates Qwen3 configuration and
Safetensors index mappings while memory-mapping shard files without copying
their contents. The tokenizer wrapper, BF16 norm/matvec/embedding kernels,
RoPE, fixed-capacity KV cache, capacity-strided GQA attention decode, and the
first resident end-to-end Qwen3 token path are covered by correctness tests.
The versioned quality fixture has a backend-agnostic loader and deterministic
structured-completion/NLL scorer.
The planner module emits and validates explainable prefetch/execute/evict
traces for bounded sequential layer windows; it is not yet driving Metal
execution.
Both Metal profiles are correctness baselines; streaming is the low-residency
baseline, while resident retains private linear/embedding buffers to expose
the throughput/memory trade-off. Large projections use tiled/vectorized BF16
matvecs with SIMD-group reductions. Streaming binds each Safetensors tensor
through an operation-scoped, page-aligned no-copy view and passes the tensor
offset to Metal; it does not retain a second model copy. The resident
reference executes small norm/RoPE/attention operations
on CPU to avoid hundreds of synchronous tiny Metal launches; this is an
explicit heterogeneous baseline, not a weight transform.

The first mmap streaming capture reaches 3.849 decode tok/s on the canonical
workload with 0 MiB reported Metal allocation and 397 MiB worst RSS. Retaining
only the output head reaches 3.973 decode tok/s at 742 MiB Metal allocation and
394 MiB worst RSS. Both produce the same 128 greedy token IDs as the resident
reference; the hot-head quality capture reports mean NLL 2.475526, perplexity
11.887962, and 9/12 structured cases passed.

Retaining the first eight layers reaches 4.519 decode tok/s at 1,540 MiB
worst Metal allocation and 1,297 MiB worst RSS on the same canonical workload.
The exact canonical rerun reaches 4.607 decode tok/s / 4.624 total tok/s at
1,540 MiB worst Metal allocation and 576 MiB worst RSS; it is preserved in
[`benchmarks/si-001-qwen3-4b-metal-streaming-mmap-retain8-canonical-2026-08-09.json`](benchmarks/si-001-qwen3-4b-metal-streaming-mmap-retain8-canonical-2026-08-09.json).

The streaming scheduler now follows layer execution order for bounded
read-ahead. On macOS, `SI_PREFETCH_MIB=N` enables a two-request background
`F_RDADVISE` queue; it is disabled by default until a measured sweep proves a
throughput gain without increasing the RSS budget. `SI_BATCH_STREAMING=1`
enables an experimental multi-matrix command-buffer path; the default keeps
the sequential path because it is currently the safer memory/throughput point.
`SI_FUSED_PROJECTIONS=1` enables single-dispatch QKV and gate/up kernels;
`SI_STAGE_MIB=N` additionally stages at most one fused projection group in a
bounded worker while the current Metal dispatch runs. Both are opt-in because
the first M3 Pro measurements traded throughput for lower host-side paging;
staging never creates a full second model copy.
Projection matvecs reuse a bounded shared input buffer and contiguous output
arena; the pool returns scratch only after the Metal command completes, so it
removes allocation churn without changing numerical results or weight residency.
Set `SI_SCRATCH_POOL=1` to enable it for an apples-to-apples allocation-overhead
comparison; it is disabled by default until it demonstrates a canonical speedup.
`SI_SERIAL_MATVEC=1` enables an experimental serial-row kernel for very wide
output matrices (at least 65,536 rows); it avoids one reduction threadgroup per
vocabulary row and preserves greedy token IDs. It is disabled by default until
the thermal-noise-sensitive throughput sweep is complete.
`SI_MULTIROW_MATVEC=1` enables a separate wide-matrix kernel that assigns one
SIMD group to each of up to four vocabulary rows per threadgroup. It preserves
the standard BF16-to-FP32 accumulation while reducing threadgroup scheduling
overhead; it remains opt-in pending a controlled canonical A/B sweep.
`SI_ASYNC_METAL=1` enables nonblocking submission handles for projection
dispatches; the runtime can submit a group, queue the next stage, and wait only
when outputs are needed. `SI_PROFILE_METAL=1` prints command submission and
wait totals so overlap can be evaluated instead of inferred from wall time.
`SI_PROFILE_RESOURCES=1` adds per-warmup and per-repetition RSS, page-fault,
context-switch, and Metal-wait deltas to stderr. It is diagnostic-only and
does not change the default benchmark report or execution path; use it when a
run’s page residency or scheduler behavior needs to be explained.
`SI_GPU_QKV_PREP=1` keeps fused QKV, Q/K RMSNorm, and RoPE in one Metal
command buffer while retaining the CPU attention implementation; it is
disabled automatically when staging, row chunking, or full-chain attention is
selected and remains opt-in pending benchmark results.
`SI_CHAIN_ATTENTION=1` is a separate full-GPU attention experiment; its
correctness path is covered, but it is not currently a throughput recommendation
because streamed command graphs still have a higher per-layer cost on the M3 Pro.
`SI_CHAIN_LAYER=1` extends that experiment to a complete transformer-layer
command buffer (including the MLP and residuals); it is lossless on the tested
Qwen3 path but remains opt-in while streamed layers are benchmarked.
`SI_CHAIN_MLP=1` applies the same command-graph idea only to the post-attention
MLP (gate, up, SiLU, down, and residual add). Its correctness path is covered,
but it remains opt-in because the first short throughput A/B did not improve
the streaming reference.

For a lossless storage-layout experiment, build an execution-order cache with
`si-pack-model --model models/qwen3-4b-base --output /tmp/qwen3.sipack`, then
set `SI_PACKED_CACHE=/tmp/qwen3.sipack` for `si-bench`. The cache is a separate
file; original Safetensors remain unchanged.

The canonical resident reference result is preserved in
[`benchmarks/si-001-qwen3-4b-metal-resident-2026-08-08-quality.json`](benchmarks/si-001-qwen3-4b-metal-resident-2026-08-08-quality.json).
On the M3 Pro it reports 10.773 decode tok/s (median of three repetitions),
7,672 MiB worst Metal allocation, 3,055 MiB worst process RSS, and
8,044,544,000 active resident-weight bytes. The quality fixture reports mean
NLL 2.475526, perplexity 11.887962, and 9/12 structured cases passed. The
shorter speed-only capture is kept separately in
[`benchmarks/si-001-qwen3-4b-metal-resident-2026-08-08.json`](benchmarks/si-001-qwen3-4b-metal-resident-2026-08-08.json).
