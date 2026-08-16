# Super-Inference Implementation Plan

This plan turns SI-001 through SI-004 into a sequence of falsifiable milestones.
Each milestone must pass its quality and memory gates before the next planner
feature is added.

## Phase 0: Freeze experiment inputs

- Add model manifest with revision, file sizes, and SHA-256 digests.
- Add versioned quality fixtures and scoring schema.
- Add hardware/profile metadata to benchmark reports.
- Define reference output and logit diagnostic artifact format.

Exit gate: a clean checkout can verify model files and run fixture validation.

## Phase 1: Model and tensor foundations

- Add Safetensors index parsing and memory-mapped tensor descriptors.
- Add Qwen3 configuration validation and shape calculations.
- Add tokenizer loading and deterministic tokenization tests.
- Add CPU tensor reference helpers for small tensor-level tests.

Exit gate: every required tensor is found with expected dtype, shape, and byte
range; malformed or missing shards fail clearly.

## Phase 2: Direct Metal reference backend

- Add Rust Metal device/queue/context setup.
- Implement BF16 buffer upload and explicit FP32 accumulation policy.
- Implement and test RMSNorm, RoPE, GQA attention, SwiGLU, residuals, and LM
  head kernels.
- Implement resident prefill and greedy decode.

Exit gate: resident backend produces stable quality-suite scores and expected
regression diagnostics without memory leaks or unbounded per-token allocation.

Current status: the token path, greedy CLI, fixture scorer, Metal allocation
sampling, process RSS sampling, streaming profile, and retained-weight resident
profile are wired. Streaming row-chunking is the first bounded sublayer
experiment. `--quality-fixture` now runs the versioned likelihood and
structured-completion suite, although it is intentionally slow in this
reference implementation. Repeated-run median/worst aggregation is wired and
the resident reference uses tiled/vectorized BF16 matvecs plus CPU execution of
small norm/RoPE/attention operations to avoid synchronous launch overhead.
The streaming profile now binds mapped Safetensors bytes through Metal no-copy
buffers; an opt-in output-head lease provides the first bounded persistent hot
tensor experiment.
The `--retain-layers N` lease now provides a bounded layer-residency sweep;
eight retained layers reach 4.519 decode tok/s at 1,540 MiB worst Metal
allocation on the canonical workload.
Planner action/trace validation and a sequential layer-window plan are
scaffolded but not yet connected to execution.

## Phase 3: Harness integration and telemetry

- Replace mock-only execution with `mock`/`metal-streaming`/`metal-resident`
  backend selection.
- Add macOS resident-memory and Metal allocation samplers.
- Add quality scoring and regression diagnostics to text and JSON reports.
- Add repeated-run median/worst-case aggregation.

Exit gate: baseline Pareto point is reproducible across three runs and all
required SI-001 report fields are populated from measurements.

Status: resident reference exit gate passed; the captured result is in
`benchmarks/si-001-qwen3-4b-metal-resident-2026-08-08-quality.json`.

## Phase 4: Memory planner

- Introduce tensor leases and operation graph identifiers.
- Implement mmap source storage and resident buffer reuse.
- Implement sequential layer streaming with bounded buffers.
- Add planner trace and peak-memory assertions.

Exit gate: lower-residency profile preserves SI-001 capability and reports a
real measured memory reduction, even if throughput is initially worse.

Status: the mmap streaming exit gate passed on the canonical 128-token run; the
hot-head quality capture is in
`benchmarks/si-001-qwen3-4b-metal-streaming-mmap-hot-head-2026-08-08-quality.json`.

The scheduler now exposes execution-order layer names and an opt-in macOS
`F_RDADVISE` worker bounded to two pending requests. The worker is disabled by
default pending a throughput/memory sweep; the experimental multi-matrix
command-buffer path is likewise opt-in because early M3 Pro runs were slower.
An execution-order `.sipack` cache generator and mmap loader are available as
a separate storage-layout experiment; source Safetensors remain the reference.
Fused QKV and gate/up Metal kernels are available behind
`SI_FUSED_PROJECTIONS=1`. `SI_STAGE_MIB=N` adds a one-group bounded host
staging worker that reads the next fused group while the current dispatch is
running; it is deliberately opt-in because the first M3 Pro run was slower.
`SI_ASYNC_METAL=1` now submits projection command buffers without an immediate
wait and retains their resources until collection. `SI_PROFILE_METAL=1` reports
submission/wait counts and aggregate wait time for overlap measurements.

## Phase 5: Scheduling optimization

- Add asynchronous prefetch and double/triple buffering.
- Benchmark chunk sizes, queue synchronization, and transfer overlap.
- Add sub-layer streaming only where layer streaming cannot meet memory target.
- Evaluate heterogeneous CPU execution using measured cost models.

Exit gate: every optimization has a recorded VRAM/throughput/quality point and
can be disabled independently for bisectable comparisons.

## Phase 6: SI benchmark and research loop

- Add JSONL run collection and Pareto-frontier export.
- Compare resident, mmap, streaming, overlap, and heterogeneous profiles.
- Publish reproducible command lines, traces, and failure cases.
- Use findings to select the next model size and NVIDIA backend requirements.

## Phase 7: Throughput recovery under bounded residency

Follow [`SI-004`](SI-004-throughput-recovery-spec.md) after the current
4.607 decode tok/s / 1,540 MiB canonical low-residency point. Prioritize
structural reductions in streamed weight work over additional isolated kernel
tuning:

- benchmark an MPS matrix-vector oracle for hot shapes;
- implement exact `verify_many(K)` for `K=2`, `4`, and `8` (implemented behind
  `SI_VERIFY_MANY=1`; retain-8 control artifact reaches 8.343 candidate tok/s
  at K=8, but is not yet end-to-end decode);
- layer Lookahead Decoding and, if needed, streaming-aware speculation on the
  same batched verification primitive (initial Jacobi and same-model partial
  draft controls are implemented but rejected for throughput);
- evaluate exact output-head MIPS pruning;
- prototype bit-exact compressed BF16 tile streaming;
- replace contiguous layer retention with profile-guided residency selection;
- evaluate reusable Metal command graphs and CPU/GPU output-head splitting.

Exit gate: every experiment has an isolated feature flag, canonical A/B
artifact, quality/ID result, per-token work telemetry, and a measured point on
the `<=1.60 GiB` Metal target (or an explicitly labelled `<=1.80 GiB`
exploratory point). Promote only changes that beat the canonical 4.607 decode
tok/s baseline without material capability loss.

## Risks and controls

- Metal BF16 support may be incomplete: probe it first and fail explicitly.
- PCIe-style paging assumptions do not transfer to unified memory: keep tier
  definitions logical and report host metrics separately.
- A fast but numerically unstable kernel can hide capability loss: require
  quality gates before throughput claims.
- Streaming every weight for every token may hit a bandwidth floor: measure it
  in Phase 4 before adding scheduler complexity.
