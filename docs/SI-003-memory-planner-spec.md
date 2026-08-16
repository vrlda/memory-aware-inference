# SI-003: SI Memory Planner and Scheduler

Status: proposed

## Goal

Minimize peak fast-memory residency while preserving model capability and
useful throughput. Planner decisions must be explainable in a trace.

## Memory model on M3 Pro

Apple unified memory is physically shared, so the engine reports separate
logical classes rather than claiming separate VRAM and RAM:

- mapped source weights: file-backed Safetensors pages;
- Metal weight buffers: GPU-visible active weights;
- KV cache: live attention state;
- activations: current hidden states and intermediates;
- scratch: reusable kernel workspaces;
- runtime: tokenizer, scheduler, queues, and metadata;
- process resident peak and Metal allocated peak: observed host metrics.

## Residency modes

Implement modes behind one scheduler interface:

1. `resident`: all linear and embedding BF16 tensors uploaded once before
   execution; correctness/throughput baseline.
2. `mmap`: source files mapped without duplicate whole-model CPU copies and
   each operation binds only the matrix it needs through a synchronous no-copy
   Metal buffer.
3. `layer-stream`: only current layer and prefetched next layer occupy Metal
   buffers.
4. `sublayer-stream`: attention and MLP matrices are leased independently;
   row-chunked matvec is the first bounded sublayer experiment.
5. `overlap`: asynchronous prefetch, execution, and eviction with bounded
   double/triple buffers.
6. `heterogeneous`: planner may execute a block on CPU when moving its weights
   costs more than local computation.

Modes 3–6 are optimization experiments. They must never mutate source weights
or silently alter precision.

The first mmap point is captured in
`benchmarks/si-001-qwen3-4b-metal-streaming-mmap-2026-08-08.json`. The optional
hot-output-head point demonstrates a bounded persistent lease without retaining
the rest of the model. `--retain-layers N` extends that lease to a bounded
prefix of transformer layers; the eight-layer point is captured in
`benchmarks/si-001-qwen3-4b-metal-streaming-mmap-retain8-2026-08-08.json`.

## Planner interface

Planner receives an operation graph, tensor descriptors, device limits,
current residency, and measured transfer/compute costs. It returns a sequence
of `prefetch`, `execute`, `retain`, `evict`, `wait`, and `release` actions.

Every action records operation id, tensor ids, bytes, tier, start/end time,
queue, and reason. This trace is required for diagnosing stalls and unexpected
memory growth.

## Invariants

- Peak memory must be sampled, not estimated from planned sizes alone.
- No tensor may be evicted while a command buffer can still read it.
- Prefetch failure is a hard run error, never a silent fallback.
- A profile is valid only if SI-001 capability gates pass.
- Scheduler overhead and transfer time are included in throughput.
- KV and scratch growth must be bounded by declared context and batch limits.

## Optimization order

First establish resident and mmap baselines. Then add sequential layer
streaming, measure its bandwidth floor, add overlap, and only then evaluate
sub-layer and heterogeneous execution. Each step produces a VRAM/throughput
point and a trace artifact.
