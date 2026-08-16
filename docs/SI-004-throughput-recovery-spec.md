# SI-004: Throughput Recovery Under Bounded Residency

Status: in progress (workstreams 0 and 1A)

The proposal that motivated this document called this phase “SI-002”. The
repository already uses SI-002 for the Direct Metal Runtime, so this document
uses SI-004 and keeps the proposal's intent unchanged.

## Why this phase exists

The canonical low-residency point is 4.607 decode tok/s (4.624 total tok/s) at
1,540 MiB peak Metal allocation. That is about 217 ms per generated token.
Reaching 10 tok/s requires about 100 ms per token, so roughly 117 ms/token must
be removed.

The current output-head profile is about 1.3 s for 28 calls, or approximately
46 ms/token. Making the output head free would therefore only raise the
throughput to roughly 5.9 tok/s. The dominant target is repeated dense-model
weight traversal, not another small kernel or buffer-allocation tweak.

Research evidence and controlled measurements live in
[`SI-RESEARCH-RECORD.md`](SI-RESEARCH-RECORD.md). This phase tests structural
ways to amortize or avoid that repeated traversal while preserving the
low-residency contract.

## Hard contract

| Item | Contract |
| --- | --- |
| Model | `Qwen3-4B-Base` |
| Weights | Original BF16 values; source Safetensors remain immutable |
| Decode | Greedy |
| Metal peak target | `<= 1.60 GiB` |
| Exploratory ceiling | `<= 1.80 GiB` (tree-only exploratory ceiling may use `<= 2.00 GiB`, hard-capped at 2,048 MiB) |
| Quality | Identical canonical greedy token IDs for the canonical run; no material capability degradation on the quality suite |
| Current baseline | 4.607 decode tok/s; 4.624 total tok/s |
| Stretch target | Approximately 10 tok/s |

“Exact” here means lossless model capability for the experiment. It does not
require every hardware path to emit bit-for-bit identical floating-point
values, but a promoted path must preserve canonical greedy IDs and pass the
versioned quality gates.

Implementation status: the MPS oracle is measured and rejected as a faster
path. The exact batched projection primitive, end-to-end batched target
forward, and cache-committing `verify_many` seam are implemented and
probe-tested behind `SI_VERIFY_MANY=1`; the canonical path does not initialize
that exploratory pipeline. Fusing QKV and gate/up inside the batched path now
raises the restored retain-8 three-repetition control to 16.622 candidate
tok/s at K=8 (6.757 at K=4), with greedy-ID agreement, but this is not an
end-to-end generated-token result. An initial exact Jacobi/lookahead
scheduler is now implemented behind `SI_LOOKAHEAD=1`, but its K=8 control
accepts only about two tokens per two target passes and is rejected for
promotion. A same-model partial-layer drafter was also tested and accepted
only one token per step. A better external drafter or multi-branch proposal is
required; the K=8 fused path was also run end-to-end on the canonical 128-token
workload and reached only 1.700 decode tok/s at 1,544 MiB, so it is not a new
baseline. Canonical promotion and the remaining workstreams are still pending.

## Measurement protocol

Continue the alternating thermal A/B methodology used by the research record.
Never promote a result from one isolated run. Every experiment must use the
same model revision, prompt, context, generation length, warmup, repetitions,
quality fixture, and hardware as the canonical SI benchmark.

Record these fields for every run:

- decode tok/s;
- total tok/s;
- peak Metal allocation;
- peak and median RSS;
- GPU time/token;
- CPU time/token;
- command buffers/token;
- GPU waits/token;
- weight bytes touched/token;
- model passes per generated token;
- accepted tokens per model pass;
- output-head time/token;
- canonical greedy-ID match and quality-suite result.

The most important new economic metric is **accepted generated tokens per
streamed target-weight pass**. A structural win must reduce target passes per
generated token, not merely move time between queues.

## Explicitly out of scope for this phase

The following have already been implemented or measured and are not the next
primary research direction:

- asynchronous projection submission by itself;
- execution-order packing and read-ahead by itself;
- scratch pools and arenas;
- grouped streaming;
- full, MLP, or attention kernel chaining;
- serial-row and multirow matvec kernels;
- retaining the whole output head.

They may remain as independently switchable controls, but a new experiment
must not be justified as a breakthrough if it only repeats one of these
approaches.

## Workstream 0: MPS matrix oracle (cheap diagnostic)

Before spending time on ordinary GEMV tuning, compare the current SI kernels
with Apple's `MPSMatrixVectorMultiplication` using the same existing
`MTLBuffer` backing wherever possible.

Benchmark the shapes used by:

- Q projection;
- O projection;
- gate/up projection;
- down projection;
- tied output head.

Measure kernel time, end-to-end time, Metal allocation, and output-ID match.

Decision rule:

- If MPS is approximately as fast as SI, stop treating ordinary GEMV as the
  main research target.
- If MPS is materially faster (for example, 24 ms versus 46 ms), fix the
  kernel path before pursuing higher-level scheduling.
- Reject any MPS path that violates the residency ceiling through hidden
  allocations or duplicate model storage.

## Workstream 1: Exact multi-token decoding — highest priority

### Objective

Stop performing one complete streamed model traversal for every generated
token. Add an exact primitive that evaluates several sequential candidate
positions during one target-model traversal:

```rust
verify_many(tokens: &[TokenId], state: &State) -> Verification
```

The desired transformation is:

```text
current:  read W; W × h1; read W; W × h2; read W; W × h3
target:   read W once; W × [h1 h2 h3]
```

This turns the streamed GEMV workload into a small GEMM workload. It is the
highest-priority experiment because it attacks weight bytes touched per
generated token directly.

### 1A. Batched target verification

Implement exact verification for `K = 2`, then `K = 4`, then `K = 8`.

For every streamed matrix tile:

1. load the BF16 tile once;
2. accumulate against all candidate hidden states;
3. release the tile only after all `K` accumulations finish.

Required batched operations:

- RMSNorm;
- QKV projection;
- causal attention over candidate positions;
- temporary KV branch;
- MLP/SwiGLU;
- output verification;
- commit accepted prefix and discard rejected suffix.

Do not implement this as `K` calls to the existing matvec. Build an actual
`[K x hidden] x [hidden x output]` path. `MPSMatrixMultiplication` may be used
as a correctness oracle while a custom Metal kernel is developed, provided it
does not change source weights or violate the memory contract.

Acceptance gates for the batched primitive:

| Batch width | Continue if decode reaches |
| ---: | ---: |
| `K=2` | `>= 5.8 tok/s` |
| `K=4` | `>= 6.8 tok/s` |
| Any | `>= 7.5 tok/s` is a structural win; `>= 9 tok/s` is a major result |

The gate is necessary but not sufficient: canonical IDs, quality, memory, and
per-run telemetry must also pass.

### 1B. Lookahead decoding

Once `verify_many` is correct, add exact Lookahead Decoding without modifying
the target model or requiring a draft model.

Add experimental controls such as:

```text
SI_LOOKAHEAD=1
SI_LOOKAHEAD_WIDTH=2|4|8
SI_LOOKAHEAD_DYNAMIC=1
```

With the dynamic control enabled, the scheduler starts at the configured
maximum window, shrinks to two after zero/one accepted token, halves a
partially accepted window, and grows only after a fully accepted window. This
is a bounded acceptance-based economic heuristic, not a throughput claim; it
exists to avoid repeatedly paying for long low-value drafts while a stronger
proposal model is being developed.

Record:

- target passes per 128 generated tokens;
- accepted tokens per target pass;
- weight bytes per generated token;
- verification time;
- lookahead/Jacobi overhead;
- canonical greedy-ID match.

Promotion criterion: the accepted-token/pass metric must improve materially,
and total throughput must beat the 4.607 tok/s baseline without exceeding the
Metal/RSS ceilings.

The current implementation also exposes an exact shared-weight tree verifier
behind `SI_TREE=1`. Two branches of depth four are flattened into one
eight-position target traversal, with compact branch-local KV caches and
optional bounded Jacobi updates. `SI_TREE_MEMORY_MIB` defaults to 2,000 MiB
and is hard-capped at 2,048 MiB; both sampled Metal allocation and RSS must
remain below it. The first canonical 32-token smoke preserved all target-only
greedy IDs at 1.987 decode tok/s, 1,542 MiB Metal, and 1,606 MiB RSS, but its
n-gram branches accepted too few suffix tokens to beat SI-001. It remains an
exploratory shared-verification seam for a stronger drafter or tree scheduler,
not a promoted path.

## Workstream 2: Streaming-aware speculative decoding

If Lookahead does not create enough useful parallelism, reuse `verify_many` for
speculative decoding. The target remains unchanged and is still verified
exactly; speculation exists only to amortize one streamed BF16 traversal across
multiple accepted tokens.

Initial architecture:

```text
small drafter (CPU/shared memory)
          ↓
      [t1 t2 t3 t4]
          ↓
SI target verify_many(4)
          ↓
one streamed target traversal
          ↓
accept longest valid prefix
```

Keep the drafter off Metal in the first experiment so target residency stays
near the current point. Prefer a same-vocabulary drafter; heterogeneous-vocab
lossless methods are a later branch.

Record:

- draft tok/s;
- mean proposed length;
- mean accepted length;
- acceptance ratio;
- target passes/token;
- draft and verification milliseconds;
- overall tok/s;
- accepted target tokens per streamed target pass.

An accepted-prefix result around 2.2 tokens per target pass would materially
change SI's bandwidth economics. No speculative path is promoted unless it
preserves the target's exact greedy decision under the canonical contract.

## Workstream 3: Exact output-head search

The greedy head only needs:

```text
argmax_i(h · W_i)
```

Build a lossless maximum-inner-product index over the tied output rows. Keep
the original BF16 rows untouched; the index only orders and bounds exact row
evaluation.

Offline layout:

```text
151,936 BF16 rows
        ↓
hierarchical clusters
        ↓
centroid + conservative radius per cluster
        ↓
original rows remain authoritative
```

For cluster centroid `c`, radius `r`, and hidden vector `h`:

```text
upper_bound(cluster) = c·h + r ||h||
```

The runtime must evaluate clusters in descending bound order, seed a strong
lower bound with exact previous-token/top-N candidates, and skip a cluster
only when its conservative bound cannot beat the current best. Add a full-head
fallback whenever pruning becomes unproductive.

Prototype levels:

- 256 clusters;
- 1,024–4,096 clusters.

Record:

- vocabulary rows evaluated (percentage and count);
- cluster-bound overhead;
- head milliseconds/token;
- fallback percentage;
- exact-ID match.

Success target: median evaluation below 30% of vocabulary rows; 5–10% would be
particularly interesting. This is a secondary optimization: even a free head
cannot reach 10 tok/s alone.

The first opt-in contiguous 256-row implementation evaluated 593/594 blocks
and reached 3.481 decode tok/s on the canonical 32-token smoke at 1,540 MiB
Metal and 1,993 MiB RSS, with matching greedy IDs. It is rejected as a
partitioning strategy; a future attempt must use non-contiguous or learned
clusters and retain a full-head fallback.

## Workstream 4: Lossless compressed BF16 streaming

Keep canonical Safetensors untouched. Add a separately verifiable cache:

```text
model.safetensors → SI cache builder → model.sicache
```

The invariant is exact bit recovery:

```text
decode(sicache tensor) == original BF16 bits
```

Do not decompress a whole tensor into a second BF16 model copy. Decode each
compressed tile directly in the Metal multiply path:

```text
compressed tile → registers/threadgroup memory → multiply → discard
```

First codec experiment: GPU-friendly fixed-size blocks with a bit-plane or
invariant/common-bit transform, compact varying bits, and fixed metadata. Sweep
independent 32 KiB, 64 KiB, 128 KiB, and 256 KiB blocks with random tile access.

Benchmark before integrating inference:

- compression ratio;
- decode GB/s;
- compressed-read plus decode GB/s;
- peak temporary decode memory.

Kill the branch if Qwen3 BF16 produces less than 1.2x useful compression. A
1.4x–1.8x result merits integration because it reduces streamed bytes on every
layer and token.

The initial 64 KiB probe is complete: zlib reached 1.27x compression but only
about 0.40 GiB/s CPU decode; invariant-bit packing reached 1.23x and 0.135
GiB/s CPU decode; simple BF16 RLE expanded the data. A row-aligned fused Metal
decoder was then measured with no-copy shared buffers. It preserved BF16 bits
exactly, but its matvec was 3.0x–5.1x slower than the existing mapped BF16
kernel (0.198x–0.299x speedup), so the codec is rejected for runtime
integration. Future compression work needs a different representation or a
more parallel decode path. Artifact:
[`lossless codec probe`](../benchmarks/si-004-lossless-codec-probe-2026-08-10.json).

## Workstream 4A: Resident-layer sidecar proposal

The first eight retained layers are now exposed as a disposable drafter
observation point. `si-resident-drafter-probe` records the hidden state after
that prefix together with exact top-1/top-4 labels from the untouched target.
The sidecar is allowed to be approximate or quantized; only its proposals are
approximate, while target verification remains lossless.

The current trace suite contains 250 labeled positions across canonical,
factual, arithmetic, Rust, memory, and quality prompts. The sidecar trainer
must regress against `model.norm(target_hidden)`, because that is the exact
representation consumed by the target output head. The normalized rank-128
probe reaches 58% held-out target-top-1-in-top-4 coverage and 78% target-top-4
overlap, but remains a proposal-only artifact. The next gate is held-out
coverage and draft latency, before any speculative scheduler changes:

- top-4 target coverage;
- sidecar milliseconds per proposal window;
- accepted tokens per streamed target pass;
- canonical ID and memory gates after exact verification.

The first runtime integration is complete but rejected for promotion: the
79.7 MB normalized sidecar reached 4.174 decode tok/s at K=2 and 2.631 at K=8
on the canonical `Hello` smoke, below the 4.744 tok/s target-only control. A
15.428 tok/s four-layer result was an intentional overfit negative control and
fell to 2.723 tok/s on the canonical paging prompt. This establishes that
resident-prefix compute and cross-prompt proposal quality, rather than the
target verifier alone, are the current blockers.
The first sidecar tree (four branches of depth two) was also exact but slower
than linear drafting because its branch-local prefix work outweighed the
shared target traversal; it is rejected pending a cheaper branch-state path.
The five-family four-layer sidecar reached 9.575 tok/s on its in-distribution
paging control but only 3.306 tok/s on an unseen Japan prompt, so future work
must add an economic fallback/controller before any headline throughput claim.
The full 232-position four-layer sweep reduced held-out target-top-1-in-top-4
coverage to 32.6%, rejecting prefix-depth reduction as a general fix.
An opt-in score-margin gate (`SI_DRAFT_SIDECAR_MIN_MARGIN`) now skips K-way
verification when the sidecar's first proposal is low-confidence; it is a
measurement seam only until a cooled canonical A/B proves that the one-prefix
probe cost is repaid.

The proposal path now avoids advancing the retained-prefix KV cache after the
last candidate, because that state cannot produce another proposal. When a
window is accepted in full, the final accepted token is advanced once before
the next window; rejected or partially accepted windows retain only the useful
prefix. This preserves the cache-position invariant and exact IDs while
removing dead drafter work. It has passed the full Rust test (82 tests) and
Clippy gates;
no canonical throughput promotion is claimed while the host Metal state is
below the SI-001 reference.

Do not promote a sidecar based on training-set coverage or isolated candidate
speed. The target falls back to canonical SI whenever predicted amortization is
not positive.

## Workstream 5: Reusable Metal dispatch graph

Reduce CPU command encoding and wait overhead without chaining transformer
operations or increasing register pressure.

Prototype reusable Metal indirect command buffers and argument buffers:

```text
initialize → encode reusable SI decode graph → execute graph per token
```

Dynamic values belong in small control/argument buffers:

- token position;
- KV offsets;
- tensor offsets;
- sequence length;
- buffer indexes.

Start with one layer or shard. Preserve operation-scoped mmap views and the
current residency model. Stop if persistent resource identity causes Metal
allocation to grow materially. The gate is the same memory point with fewer
CPU submissions and waits—not a full private copy of the model.

Tokio or another Rust async runtime is not sufficient by itself: it can schedule
host work, but Metal command-buffer dependencies, resource lifetimes, and GPU
completion points still require explicit Metal synchronization.

## Workstream 6: Profile-guided residency optimizer

Replace the current contiguous `--retain-layers 8` assumption with an offline
profile and bounded knapsack planner.

For every candidate tensor, measure:

- mapped execution time;
- private execution time;
- private size in MiB;
- benefit = mapped time − private time;
- value = benefit / MiB.

Select independently among `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, and `down_proj` subject to the 1,540 MiB target. Keep
the output head blacklisted initially because whole-head retention has already
shown unstable results.

Promotion requires a controlled A/B win over canonical retain-8 at the same
Metal peak, with unchanged IDs and quality.

## Workstream 7: CPU + GPU exact output-head race

Partition output-head rows between CPU and GPU while both read the same unified
memory backing. Start with:

```text
GPU/CPU: 95/5, 90/10, 80/20, 70/30
```

Merge exact maxima after both partitions complete. Kill immediately if GPU
slowdown is at least as large as the CPU contribution. A 10–20% CPU share that
is effectively free is worth retaining as a secondary optimization.

## Implementation order

1. MPS matrix oracle.
2. Exact `verify_many(K)` for `K=2`, `4`, `8`.
3. Lookahead Decoding.
4. Streaming-aware speculative decoding.
5. Exact output-head MIPS index.
6. Bit-exact compressed BF16 streaming.
7. Profile-guided residency planner.
8. Reusable ICB/argument-buffer graph.
9. CPU/GPU output-head split.

Each step gets its own feature flag, benchmark artifact, quality result, and
rollback path. Do not combine workstreams until their isolated A/B behavior is
known.

## Promotion checklist

- [ ] Canonical command and hardware recorded.
- [ ] Three-repetition alternating A/B sweep complete.
- [ ] Metal peak `<= 1.60 GiB` (or explicitly marked exploratory under
      `1.80 GiB`).
- [ ] RSS peak and median recorded.
- [ ] GPU/CPU time and wait/submission telemetry recorded.
- [ ] Weight bytes/token and model passes/token recorded.
- [ ] Accepted tokens/pass recorded where applicable.
- [ ] Canonical greedy IDs match.
- [ ] Versioned quality suite passes.
- [ ] Result stored as JSON plus a short research-record entry.
- [ ] Feature remains independently disableable.

## Non-goals

- weight quantization, pruning, or lossy low-rank approximation;
- changing Qwen3 weights or tokenizer;
- claiming that lower logical Metal allocation means lower physical memory
  pressure on Apple unified memory;
- optimizing only one thermally favorable run;
- treating a faster but materially degraded completion as exact.
