# Super-Inference research record

Status: measured through 2026-08-09 on an Apple M3 Pro.

This record separates verified measurements from hypotheses and from
experiments that were intentionally rejected. All lossless paths below keep
the original Qwen3-4B-Base BF16 values; no weight quantization, pruning, or
model transformation is used.

## Research contract

The initial thesis is:

> How little fast memory does exact dense-model inference need while retaining
> useful throughput?

“Exact” means no material model-capability degradation. It does not require
bit-for-bit identical hardware outputs, although the current lossless paths
produce identical greedy token IDs in the tested workload.

The first phase intentionally isolates engine/runtime changes:

- original BF16 weights;
- terminal-only Rust harness;
- direct Metal backend for Apple Silicon;
- a versioned quality suite;
- a Pareto measurement of throughput, Metal allocation, and process RSS.

The first target is not a 120B model. The test object is Qwen/Qwen3-4B-Base,
revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`.

## Canonical measurement

Unless noted otherwise:

- prompt: `Explain why memory paging is useful for local model inference.`;
- prompt tokens: 12;
- generation: 128 greedy tokens;
- context: 2048;
- warmup: 1;
- measured repetitions: 3;
- hardware: Apple M3 Pro;
- model: original BF16 Qwen3-4B-Base.

The older SI-001 artifact for the eight-layer streaming point reports
4.519 decode / 4.550 total tok/s. The exact canonical rerun is now stored in
[`retain8 canonical`](../benchmarks/si-001-qwen3-4b-metal-streaming-mmap-retain8-canonical-2026-08-09.json)
and reports 4.607 decode / 4.624 total tok/s.

Throughput varies substantially with sustained system load and frequency. A
single run is therefore not treated as a promotion gate; three repetitions,
median timing, output-ID consistency, and memory peaks are required.

A post-change control rerun with the new batched code disabled measured 2.358
decode / 2.359 total tok/s at the same 1,540 MiB Metal peak, with matching
repetition outputs. It is recorded as a thermally/system-load-affected control,
not as a new baseline or regression claim: [`post-change control`](../benchmarks/si-004-canonical-post-change-2026-08-09.json).

On 2026-08-10 a later direct-dispatch control variant (already after the
historical run) was tested so ordinary single-matvec execution bypassed the
one-item batching wrapper. That three-repetition control measured 3.234 decode /
3.246 total tok/s at the same 1,540 MiB Metal peak, with matching IDs, below the
historical 4.607 reference. It is a post-run experiment, not the recovered
historical source, and is captured separately:
[`restored-path control`](../benchmarks/si-004-canonical-restored-path-2026-08-10.json).

To separate source regression from machine state, a later control executable
(`si_bench-de8895d91c3ac07f`, built at 21:07:33 local) was run with the
identical command on the same machine. It measured 3.078 decode / 3.101 total
tok/s at 1,540 MiB Metal, with matching IDs. Its timestamp is after the first
4.607 run at 17:11:06 local, so it is not a valid source control for that first
run; it only predates a later duplicate 4.607 capture. The control is captured in
[`pre-SI-004 binary control`](../benchmarks/si-004-prechange-binary-control-2026-08-10.json).

The exact historical source boundary is captured in [`canonical source version`](../benchmarks/si-001-canonical-source-version-2026-08-10.json): the release
build completed at 12:05:34Z, before the first post-run SI-004 patch at
13:02:19.450Z. I reconstructed that tree forward from the initial file-creation
patches, replayed 81 historical `cargo fmt` write transitions, applied 303
source/Cargo changes with zero failures, and compared SHA-256 hashes for every
tracked Rust/Cargo file. The reconstructed source matches the current source.

This corrected an earlier mistaken conclusion. The historical default already
used `bf16_matvec_buffer -> bf16_matvec_many_buffer` with a one-matrix slice;
the direct single-matvec dispatch was introduced later at 18:58:42Z and was not
part of the 4.607 run. No Git commit or exact first-run binary hash was
preserved, so source identity is proven but byte-for-byte executable identity
is not.

A clean rebuild from this exact source then measured 1.880 decode / 1.906 total
tok/s at the same 1,540 MiB Metal peak, with matching IDs. That result is not a
new baseline; it demonstrates that the remaining 4.607-versus-current gap is
not explained by a source-tree regression. The historical binary/build
environment is the remaining unrecorded variable, and the control is captured
in the source-version artifact.

## Baselines and Pareto points

| Profile | Decode / total tok/s | Peak Metal | Peak RSS | Result |
| --- | ---: | ---: | ---: | --- |
| `metal-resident` | 10.773 / 10.820 | 7,672 MiB | 3,030 MiB | Full-speed SI reference; all linear and embedding BF16 weights private-resident. |
| Streaming mmap, no retention | 3.849 / 3.858 | 0 MiB reported* | 397 MiB | Lowest measured residency; all projection weights are operation-scoped mapped views. |
| Streaming + private output head | 3.973 / 3.982 | 742 MiB | 394 MiB | Small hot-head residency; stable artifact, but only modest speed gain. |
| Streaming + first 8 layers retained | 4.519 / 4.550 | 1,540 MiB | 1,297 MiB | Best stable recorded low-memory Pareto point. |
| Canonical retain-8 rerun | 4.607 / 4.624 | 1,540 MiB | 576 MiB | Current SI benchmark baseline. |
| Streaming + `SI_MULTIROW_MATVEC=1` + retain-8 | 4.495 / 4.502 | 1,540 MiB | 831 MiB | Latest promising kernel experiment; opt-in pending controlled A/B promotion. |

\*The zero Metal number is the allocator sampler’s reported allocation, not a
claim that the GPU touches no physical pages. The model remains file-backed and
GPU-visible through shared no-copy mappings.

The artifacts for the first five rows are:

- [`metal-resident`](../benchmarks/si-001-qwen3-4b-metal-resident-2026-08-08.json)
- [`streaming-mmap`](../benchmarks/si-001-qwen3-4b-metal-streaming-mmap-2026-08-08.json)
- [`hot-head`](../benchmarks/si-001-qwen3-4b-metal-streaming-mmap-hot-head-2026-08-08.json)
- [`retain8`](../benchmarks/si-001-qwen3-4b-metal-streaming-mmap-retain8-2026-08-08.json)
- [`retain8 canonical`](../benchmarks/si-001-qwen3-4b-metal-streaming-mmap-retain8-canonical-2026-08-09.json)

### External baseline

llama.cpp was built with Metal and tested with a lossless BF16 GGUF
conversion. With all 37 layers offloaded it measured:

- llama-bench generation-only: 11.191 tok/s average;
- canonical completion sanity run: 9.16 decode tok/s;
- device memory: 8,262 MiB;
- canonical maximum RSS: 7,004 MiB;
- structured quality: 11/12, while its continuous-corpus perplexity is not
  directly comparable to SI’s per-case NLL.

This is a useful external speed/memory reference, not an apples-to-apples
comparison with low-residency SI streaming. The full comparison is in
[`llama.cpp baseline`](../benchmarks/si-001-qwen3-4b-llama-cpp-metal-bf16-2026-08-09.json).

### Quality evidence

The versioned SI fixture has 12 likelihood cases, 12 structured-completion
cases, and 4 regression prompts. The full-resident SI reference reports mean
NLL 2.475526, perplexity 11.887962, and 9/12 structured cases passed. The
streaming and hot-head artifacts report the same quality summary and identical
128-token greedy IDs against the resident reference. The multirow kernel also
matched the canonical 128-token IDs; its full quality-suite run is still a
separate promotion task.

## What is implemented in the current engine

The stable low-memory path is built from these pieces:

1. Safetensors shards are memory-mapped instead of copied into a second full
   model representation.
2. Each streamed BF16 matrix is bound through a page-aligned Metal no-copy view
   only for the operation that uses it; the view is released after the command
   completes.
3. The first N layers can be retained as private Metal buffers with
   `--retain-layers N`. Eight retained layers are the current throughput/memory
   reference.
4. BF16 matrix-vector products use vectorized loads, FP32 accumulation, and
   SIMD-group reduction. QKV and gate/up have optional fused kernels.
5. Attention, RoPE, RMSNorm, KV cache, model loading, and telemetry are all
   implemented in the Rust/Metal harness. The low-memory streaming path keeps
   the small control-heavy operations on the CPU and uses Metal for the large
   projection matvecs.
6. Peak logical weights, KV, scratch, Metal allocation, and process RSS are
measured per repetition.
7. SI-004 now includes an exact `si_bf16_matmul_many` Metal primitive. It
shares each BF16 weight load across up to eight batch-major FP32 input vectors
without changing source weights or residency semantics.

By default, the runtime is conservative: no read-ahead, scratch pool, async
projection path, chain graph, serial/multirow experimental kernel, or row
chunking is enabled. Experimental behavior is selected with `SI_*` flags.

The recorded low-memory point is reproducible with:

```text
target/release/si-bench \
  --model models/qwen3-4b-base \
  --backend metal-streaming \
  --prompt "Explain why memory paging is useful for local model inference." \
  --max-tokens 128 --context 2048 --warmup 1 --repetitions 3 \
  --retain-layers 8 --verify-manifest --output json
```

The latest kernel experiment adds `SI_MULTIROW_MATVEC=1` to that exact
command. It does not alter residency or model values.

## Experiment log

### Worked or useful

**Memory-mapped streaming.** This is the primary memory breakthrough. It
reduced the reported Metal residency from the full 7.7 GiB resident profile to
near-zero allocator residency while preserving the same greedy outputs. The
trade-off is weight bandwidth and much lower throughput.

**Bounded private retention.** Keeping the first eight layers private raised
throughput from the no-retention streaming point to the 4.519 decode tok/s
artifact while staying at 1,540 MiB reported Metal allocation. The later
canonical rerun reached 4.607 decode tok/s, 4.624 total tok/s, and 576 MiB RSS
with the same residency plan. This is the current SI baseline.

**Vectorized BF16 matvec.** SIMD-group reductions and vectorized BF16 loads are
the baseline large-matrix kernel. They are materially better than a naive
row-by-row implementation and are correctness-tested against CPU references.

**Multirow SIMD matvec.** Profiling identified the 151,936-row tied output head
as a hotspot. The new opt-in kernel assigns one SIMD group to each of up to four
rows per threadgroup, reducing threadgroup scheduling overhead without changing
the arithmetic. Its correctness test passes and the latest full run reached
4.495 decode tok/s at the same 1,540 MiB Metal peak.

**MPS matrix-vector oracle.** The isolated SI-004 diagnostic matched the SI
BF16 kernel within roughly 1e-6 absolute error after widening source BF16 bytes
to Float32. On the controlled three-run sample, MPS was slower on every tested
projection: SI/MPS milliseconds were 2.229/5.197 (Q), 2.024/5.052 (O),
4.984/11.218 (gate), 4.083/10.632 (up), and 4.101/11.997 (down). The full
151,936-row output head was 380.658 ms for SI versus 845.547 ms for MPS in a
separate one-run diagnostic. The oracle is therefore rejected as a replacement
kernel path. MPS does not support the BF16 data type for this operation on the
current SDK, so its Float32 widening also cannot satisfy the low-residency
production contract. Artifacts:
[`projection oracle`](../benchmarks/si-004-mps-oracle-qwen3-4b-2026-08-09.json),
[`head oracle`](../benchmarks/si-004-mps-oracle-qwen3-4b-output-head-2026-08-09.json).

**Exact batched projection primitive.** The first `verify_many` building block
is correctness-tested against independent matvecs and matched within 1e-6
absolute error. On `model.layers.0.self_attn.q_proj.weight` (4096 x 2560),
the one-dispatch path measured 2.929x faster for K=2, 5.111x for K=4, and
7.955x for K=8 than separate matvec calls. K=1 is slower (0.648x), as
expected from the extra batch bookkeeping. This is a projection
microbenchmark only; no end-to-end token or `verify_many` claim is made yet.
Artifact: [`batched projection primitive`](../benchmarks/si-004-bf16-matmul-many-qwen3-4b-2026-08-09.json).

The restored kernel-only control (same Q projection, one shared runtime,
warmup 1/repetitions 3) measured 4.240x at K=4 and 3.290x at K=8 with
maximum absolute error 1e-6. Artifact:
[`restored batched projection control`](../benchmarks/si-004-bf16-matmul-many-qwen3-4b-2026-08-10.json).

**Exact batched target forward.** `MetalQwen3::forward_tokens_many` now runs a
consecutive candidate sequence against a cloned KV snapshot, leaving the real
cache untouched. Every large projection—including the tied output head—uses
the batch-major kernel; norms, RoPE, and causal attention retain the existing
CPU reference path. In the latest one-run streaming probe, greedy IDs matched
at every batch and the maximum logit difference was 1.7e-5. K=4 reached 3.695
candidate tok/s versus 2.276 tok/s for sequential target calls (1.623x), while
K=8 reached 6.489 versus 2.249 tok/s (2.885x). K=1 is currently slower and
K=2 is near parity, which leaves a clear follow-up target. The probe also
exercised `verify_many`: it accepted one candidate and committed the logical KV
position without rerunning the accepted token. This is still a verification
primitive, not a lookahead scheduler or canonical SI benchmark promotion.
Artifact: [`batched target forward`](../benchmarks/si-004-verify-many-qwen3-4b-2026-08-09.json).

The fairer retain-8 probe on 2026-08-10 reached 6.097 candidate tok/s at K=4
versus 2.622 separate, and 10.153 at K=8 versus 3.069 separate. Greedy IDs
matched and the maximum logit difference was 1.7e-5. K=1 and K=2 remain below
parity, so this is evidence for a K>=4 scheduler experiment—not a canonical
decode result. The probe artifact is
[`retain-8 verify-many control`](../benchmarks/si-004-verify-many-retain8-control-2026-08-10.json).

After restoring the SI-001 source path and reducing the probe to one shared
runtime (so the A/B comparison does not double Metal residency), the required
three-repetition run measured 4.191 candidate tok/s at K=4 and 8.343 at K=8,
versus 2.341 and 2.374 tok/s for separate target calls. Greedy IDs still
matched for every candidate and maximum absolute logit drift stayed at
1.7e-5; the verification smoke accepted one token and committed the expected
KV position. This confirms that the K=4/K=8 primitive survives restoration of
the SI-001 path, but it remains a candidate-forward result rather than an
end-to-end decode promotion. Artifact:
[`restored retain-8 verify-many control`](../benchmarks/si-004-verify-many-retain8-2026-08-10.json).

**Fused batched projections.** The K>=4 candidate path now fuses QKV and
gate/up projections across the candidate batch, so each streamed projection
group uses one dispatch instead of three or two independent batched matmuls.
The one-shared-runtime retain-8 control remains lossless: greedy IDs match and
maximum absolute logit drift is 1.7e-5. K=4 improved from 4.191 to 6.757
candidate tok/s; K=8 improved from 8.343 to 16.622 candidate tok/s. K=1 and
K=2 regress to 1.904 and 2.649 tok/s, so the fused path must stay behind the
wide-batch scheduler. This is still candidate-forward throughput, not an
end-to-end decode promotion. Artifact:
[`fused batched projection control`](../benchmarks/si-004-fused-many-retain8-2026-08-10.json).

The same fused path was then run through the full 128-token canonical
generation workload with K=8 Jacobi lookahead. It preserved every greedy ID
against a target-only reference, but reached only 1.700 decode tok/s at 1,544
MiB Metal, versus the 4.607 tok/s SI-001 baseline. The candidate-forward win
therefore does not qualify as a new generation baseline; it confirms that
proposal quality and scheduler passes—not the fused projection kernel—are the
remaining blocker. Artifact:
[`fused end-to-end canonical control`](../benchmarks/si-004-fused-many-end-to-end-canonical-2026-08-10.json).

A 32-token acceptance trace explains the gap: with width 8 and one Jacobi
iteration, every target pass accepted exactly one candidate, then emitted one
exact correction token. Thus each pass computed eight positions but advanced
the generation by only two tokens. The trace reached 1.437 decode tok/s at
1,544 MiB Metal. Artifact:
[`canonical acceptance trace`](../benchmarks/si-004-canonical-acceptance-trace-2026-08-10.json).

**Reusable verification KV pool.** `verify_many` no longer clones full
capacity-strided KV buffers for every candidate pass. A reusable pool copies
only the active prefix and is recycled after commit/rejection. On the same
canonical 32-token trace this raised K=8/one-iteration lookahead from 1.437 to
1.772 decode tok/s while preserving IDs and 1,544 MiB Metal. Adaptive K=2
reached 1.965 tok/s, but both remain below SI-001 because proposal acceptance
is still one token per pass. Artifact:
[`cache-pool and n-gram controls`](../benchmarks/si-004-cache-pool-ngram-controls-2026-08-10.json).

**CPU n-gram drafter.** An opt-in `SI_LOOKAHEAD_NGRAM=1` proposal source uses
only prompt/generated token history and never changes target execution. The
canonical trace occasionally accepted 2–3 tokens, reaching 1.711 decode
tok/s. It is useful as a low-cost control, but not a baseline candidate; a
multi-branch/tree or learned drafter is still required.

**Jacobi/lookahead scheduler control.** An opt-in `SI_LOOKAHEAD=1` scheduler
now performs bounded exact Jacobi updates, verifies the final K=4/K=8 window,
and commits only the accepted prefix. The K=8, two-iteration, three-run
retain-8 control preserved the canonical greedy IDs, but accepted roughly two
tokens per two target passes and reached only 1.194 decode tok/s at 1,544 MiB
Metal and 2,373 MiB RSS. It is therefore rejected for promotion; the next
lookahead attempt needs a materially better candidate drafter or multi-branch
proposal mechanism. Artifact:
[`Jacobi lookahead control`](../benchmarks/si-004-lookahead-jacobi-retain8-2026-08-10.json).

The scheduler now commits the verifier's exact correction token immediately
when a candidate suffix is rejected, avoiding a second wide pass to rediscover
that token. With the fused K=8 path and one Jacobi iteration this raised the
short retain-8 control to 2.379 decode tok/s at 1,544 MiB Metal, still below
the 4.819 tok/s canonical control. The optimization is retained in the
exploratory scheduler but is not a promotion.

An optional same-model partial-layer drafter (`SI_LOOKAHEAD_DRAFT_LAYERS=8`)
was also wired through the retained SI-001 layers. It preserved target IDs,
but accepted only the first token at every step and reached 1.118 decode tok/s
with 1,544 MiB Metal and 2,167 MiB RSS. This branch is also rejected; the
retained prefix is not a sufficiently accurate drafter for Qwen3-4B.
Artifact: [`partial-layer draft control`](../benchmarks/si-004-partial-draft-retain8-2026-08-10.json).

**External small-model drafter.** `SI_DRAFT_MODEL` now supports a separate
same-vocabulary Qwen3 drafter; exact target verification and cache rollback
preserve the target's greedy IDs. A Qwen3-0.6B BF16 drafter accepted
2/6/2/8 tokens in a sample K=8 run, but target/drafter Metal contention
limited decode to 2.679 tok/s at 1,544 MiB Metal. Making the drafter resident
reduced allocation to 1,137 MiB without target retention and reached 2.910
tok/s; retaining the target's eight layers reached 3.147 tok/s but raised
allocation to 2,677 MiB. This branch is rejected under the current contract;
the next drafter experiment needs CPU/shared-memory or a much smaller
drafter. Artifact:
[`drafter and correction controls`](../benchmarks/si-004-drafter-and-correction-controls-2026-08-10.json).

The next drafter control is `SI_DRAFT_SHARED=1`, which retains the small
drafter in Apple unified shared storage rather than private Metal storage.
This is still target-exact and does not change the target weights; it is
intended to test whether a resident drafter can avoid private-memory and
queue contention while the target runs with no retained layers. It must stay
under the SI tree's 2,000 MiB exploratory ceiling and beat the target-only
baseline before promotion.

The first shared-memory run accepted 2/6/2/8 tokens with width 8. With no
target layers retained it reached 2.115 decode tok/s at 1,137 MiB Metal and
1,727 MiB RSS; retaining four target layers reached 2.033 tok/s at 1,907 MiB
Metal and 1,658 MiB RSS. Both preserved exact target IDs, but neither beat
SI-001. The four-layer point nearly exhausts the relaxed ceiling, so this
variant is rejected for promotion. Artifact:
[`shared drafter controls`](../benchmarks/si-004-shared-drafter-controls-2026-08-10.json).

**Exact output-head block index.** `SI_EXACT_HEAD=1` adds conservative
centroid/radius bounds over the tied BF16 vocabulary rows and evaluates only
blocks whose bound can still win the exact greedy argmax. On Qwen3-4B, the
contiguous 256-row blocks were not a useful partition: 593 of 594 blocks were
still evaluated. The canonical smoke reached 3.481 decode tok/s at 1,540 MiB
Metal and 1,993 MiB RSS with matching greedy IDs, so dispatch overhead and
near-ceiling RSS outweighed the negligible pruning. The feature remains an
opt-in diagnostic; the next index attempt needs learned/non-contiguous
clustering rather than another contiguous-block tweak. Artifact:
[`exact head index controls`](../benchmarks/si-004-exact-head-index-controls-2026-08-10.json).

**Lossless BF16 codec probe.** The new `si-lossless-probe` samples immutable
Safetensors tiles and verifies every decompressed byte. Generic zlib reached
about 1.27x compression at roughly 0.40 GiB/s CPU decode; a GPU-friendly
invariant-bit packer reached about 1.23x but only 0.135 GiB/s CPU decode. A
simple BF16 RLE format expanded the weights. The row-aligned Metal
`si_bf16_bitpack_matvec` probe preserved exact outputs with no-copy shared
buffers, but ran 3.0x–5.1x slower than the mapped BF16 kernel. The codec is
therefore rejected for runtime integration; the next compression attempt
needs a different representation or more parallel decode. Artifact:
[`lossless codec probe`](../benchmarks/si-004-lossless-codec-probe-2026-08-10.json).

The post-probe canonical control remained stable at 4.620 decode / 4.635 total
tok/s, 1,540 MiB peak Metal, and matching repetition outputs. The fused codec
pipeline is opt-in (`SI_LOSSLESS_GPU=1` inside the diagnostic) and is not part
of normal SI initialization. Artifact:
[`post-lossless canonical control`](../benchmarks/si-004-canonical-post-lossless-control-2026-08-10.json).

**Resident-layer drafter trace seam.** The existing partial-layer drafter was
refactored so prompt preparation and candidate-state observation can return the
hidden state after the retained prefix without executing the full vocabulary
head. `si-resident-drafter-probe` now records those layer-8 states alongside
exact top-1/top-4 labels from the untouched target in JSONL. The canonical
128-position control produced 128 records in 29.743 s; this is a trace-only
artifact, not a throughput claim or speculative acceptance result. Artifact:
[`resident drafter trace`](../benchmarks/si-005-resident-drafter-trace-2026-08-10.json).
Four additional short factual, arithmetic, Rust, and memory prompts bring the
trace suite to 192 labeled positions. A 29-prompt quality collection adds 58
complete positions, bringing the training input to 250 labeled positions. The
suite manifest records each file hash and remains a training input only:
[`resident drafter trace suite`](../benchmarks/si-005-resident-drafter-trace-suite-2026-08-10.json).

**Resident-layer low-rank sidecar baseline.** An offline ridge/PCA sidecar was
trained from those traces and evaluated on a held-out split. Rank 32/64/128
placed the exact target top-1 inside four sidecar proposals 21.1%/28.9%/42.1%
of the time; rank 128 reached 68.4% target-top-4 overlap but produced a 41.8
MB artifact and approximately 0.342 ms/holdout scoring on the CPU probe. This
is useful signal, but insufficient proposal quality for a scheduler promotion.
The baseline is rejected for integration and retained as the comparison point
for a nonlinear sidecar. Artifact:
[`resident sidecar sweep`](../benchmarks/si-005-resident-sidecar-sweep-2026-08-10.json).

The first sweep contained a representation bug: it regressed against the
target's pre-normalization hidden state even though the output head consumes
`model.norm(target_hidden)`. After fixing the trainer and adding the quality
traces, rank 32/64/128 reached 42%/56%/58% target-top-1-in-top-4 coverage on
the same held-out split, with rank 128 at 78% target-top-4 overlap and 0.308 ms
per CPU proposal score. This is a real coverage improvement, but it remains
proposal-only and is still rejected for scheduler integration until exact
multi-token verification demonstrates positive end-to-end amortization.
Artifact:
[`normalized resident sidecar sweep`](../benchmarks/si-005-resident-sidecar-normalized-sweep-2026-08-10.json).

**Resident sidecar end-to-end control.** The normalized rank-128 sidecar is
loadable by the runtime as a proposal-only 79.7 MB `.sisc` artifact. On the
canonical `Hello` smoke, K=2 reached 4.174 decode tok/s and K=8 reached 2.631
decode tok/s, versus a 4.744 decode tok/s canonical control; both preserved
exact generated IDs. A four-layer sidecar trained only on the same `Hello`
trace reached 15.428 decode tok/s at K=8, but failed the cross-prompt paging
control at 2.723 decode tok/s, proving that result was overfit. The runtime
seam is retained for broader training and dynamic scheduling, not promoted.
Training four layers on five prompt families improved the canonical paging
control to 9.575 decode tok/s (6.409 total) with exact IDs, but an unseen Japan
prompt fell to 3.306 decode tok/s. The result is therefore an in-distribution
control, not a general benchmark win.
The full 232-position, 29-prompt four-layer sweep reached only 32.6%
held-out target-top-1-in-top-4 coverage and 52.2% target-top-4 overlap, so
prefix-depth reduction is rejected as a general fix. Artifact:
[`full four-layer sidecar report`](../benchmarks/si-005-resident-sidecar-r128-layers4-full.json).
An opt-in four-branch/two-token sidecar tree was also exact but slower at
1.508 decode tok/s for the eight-layer sidecar and 2.028 tok/s for the
four-layer canonical control, so branch fan-out is rejected in its current
form.
Artifact:
[`resident sidecar end-to-end controls`](../benchmarks/si-005-resident-sidecar-e2e-controls-2026-08-10.json).

The sidecar and partial-draft paths were then tightened around their KV-cache
invariant: candidate generation no longer computes the unused hidden state
after the final proposed token. An all-accepted window advances that final
token exactly once before the next window, while shorter windows truncate to
the accepted prefix. The full Rust test suite (82 tests) and Clippy remain
clean, and an 8-token sidecar smoke preserved the canonical IDs. This is an
implementation improvement only; the current Metal performance state is below
the 4.607 tok/s SI-001 reference, so no throughput claim is attached yet.

An opt-in `SI_LOOKAHEAD_DYNAMIC=1` controller now adapts the proposal width
from observed acceptance: low-acceptance windows fall back to K=2, partial
acceptance halves the width, and only fully accepted windows grow it again.
The controller completed a K=2/K=4/K=8 exact-ID smoke without cache errors, but
the current degraded Metal state makes that run diagnostic rather than a
promotion result.

**Exact shared-weight tree verifier.** `SI_TREE=1` now evaluates two
independent depth-four branches as one eight-position target traversal. Each
branch receives a compact prefix-plus-depth KV snapshot; after exact greedy
verification only the best accepted branch is committed. The exploratory
working-set ceiling defaults to 2,000 MiB (hard maximum 2,048 MiB) and checks
both sampled Metal allocation and RSS. The first canonical 32-token smoke
preserved every target-only greedy ID at 1.987 decode tok/s, 1,542 MiB Metal,
and 1,606 MiB RSS; a later order-8/order-1 branch-diversity smoke reached
2.512 decode tok/s at 1,544 MiB Metal and 869 MiB RSS, still as a one-run
exploration. The n-gram branches accepted only short prefixes, so this does
not beat the SI-001 4.607 tok/s baseline. A second Jacobi iteration was
slower and produced a thermal/resource outlier. The implementation is kept as
the shared-verification seam for a stronger drafter or branch scheduler, not
as a promoted throughput path. Artifact:
[`tree verifier controls`](../benchmarks/si-004-tree-verifier-controls-2026-08-10.json).

### Rejected or not promoted

**Private output-head retention.** It looked promising in a short run, but a
long run became unstable: a bounded canonical pass fell to roughly 1.4 decode
tok/s while Metal allocation stayed around 2,282 MiB. It is not a reliable
promotion path.

**Operation-scoped asynchronous projection submission.** `SI_ASYNC_METAL=1`
reached about 3.718 decode tok/s in one canonical attempt, but profiling showed
1,326 submitted and 1,326 waited command buffers. The runtime was still
synchronizing every result, so this is not yet real GPU overlap.

**Scratch pooling.** `SI_SCRATCH_POOL=1` was measured below the direct-allocation
default (about 2.733 versus 3.392 decode tok/s in the paired run) and was
disabled by default. Reusing buffers increased residency pressure rather than
removing the dominant bottleneck.

**Persistent token arena.** It reached about 2.443 decode tok/s and showed no
memory benefit. The implementation was removed.

**Grouped/batched streaming and retained-weight batch fixes.** These reduced
theoretical host launches but measured around 2.6 tok/s in the tested path, so
the changes were reverted from the active design.

**Execution-order packing and read-ahead/prefetch.** Lossless file-layout and
`F_RDADVISE` experiments were implemented as opt-ins, but no reproducible
canonical throughput gain was established. They remain disabled by default.

**GPU QKV preparation, full attention chaining, and full transformer-layer
chaining.** These reduce CPU-visible intermediate results, but on this M3 Pro
the extra temporary buffers, register pressure, and graph setup outweighed the
saved host work. They remain correctness/architecture experiments, not speed
recommendations.

**MLP-only chaining.** `SI_CHAIN_MLP=1` is implemented and correctness-tested:
gate, up, SiLU, down, and residual add share one command buffer. The first A/B
was 2.881 decode tok/s versus 2.742 for its same short default run, which did
not clear a promotion bar. It remains opt-in.

**Serial-row matvec.** `SI_SERIAL_MATVEC=1` removed reductions, but the full
canonical run fell to 2.383 decode tok/s. It is rejected for the current GPU.

**Persistent mapped output-head view.** Retaining a no-copy view did not avoid
Metal residency and fell to about 1.235 decode tok/s in the short test. The
experiment was removed.

## Qwen3.6-27B Q4 bounded-residency experiment

The Qwen3.6-27B Q4_K_M GGUF is a separate engineering target from the SI-001
4B BF16 canonical benchmark. Its artifact is 16,817,244,384 bytes, while the
machine has 19.3 GB unified memory and a recommended Metal working set of
13,639 MiB. Mapping the complete file and retaining a private Metal working
set therefore creates physical-memory pressure; the low private allocation
counter does not mean the full model is resident.

The best verified two-token smoke uses a bounded two-layer staging pipeline,
retains Q4 MLP matrices on alternating layers, and retains the Q6 output head:
about 0.176 decode / 0.179 total tok/s at 4,057 MiB peak private Metal and
3,046 MiB RSS. The artifact is recorded in
[`Qwen3.6 bounded residency`](../benchmarks/si-qwen36-27b-q4-bounded-residency-2026-08-11.json).
This is a real improvement over the initial ~0.11 tok/s mapped path, but it is
not yet usable-speed inference.

An output-head overlap extension stages layer 0 while the private output head
computes, then consumes that staged buffer on the next decode. Flipping the
same 32 retained MLP pairs to odd layers and enabling this overlap reached
0.185 decode / 0.185 total tok/s at the same 4,057 MiB Metal peak and 2,829
MiB RSS. The A/B control with output prefetch disabled reached 0.171 decode /
0.175 total tok/s, with identical generated IDs. This remains a short smoke,
not a canonical 128-token result; details are in
[`Qwen3.6 output prefetch`](../benchmarks/si-qwen36-27b-q4-output-prefetch-2026-08-11.json).

The staging queue was then widened without increasing private Metal
residency. One, two, and three future layer buffers reached approximately
0.185, 0.210, and 0.225 decode tok/s respectively; depth four was flat at
0.227. Depth three is now the default when `SI_STAGE_PIPELINE=1`, with an
explicit `SI_STAGE_DEPTH` override for controls. The clean default rerun
reached 0.229 decode / 0.228 total tok/s at 4,057 MiB Metal and 2,315 MiB RSS,
with exact IDs `[548,108]`. This is the current bounded-residency control,
not yet usable-speed inference. See
[`Qwen3.6 stage-depth sweep`](../benchmarks/si-qwen36-27b-q4-stage-depth3-2026-08-11.json).

**File-cache policy and packed staging.** `F_NOCACHE` caused every staged
tensor read to bypass RAM. Allowing macOS to cache the mapped GGUF raised an
8-token run from 0.229 to 0.304 decode tok/s; the gain persisted at 16 tokens
around 0.290 tok/s with a smaller 2.5 GiB private hot set. The default now
keeps staging cacheable; `SI_STAGE_NOCACHE=1` restores the old diagnostic.
Staged layers are packed into contiguous anonymous runs instead of one
allocation per tensor. The packed 8-token smoke reached 0.321 decode tok/s;
the 16-token validation reached 0.288 decode / 0.290 total tok/s at 2,527 MiB
Metal and 3,746 MiB RSS, with exact IDs preserved. Artifact:
[`Qwen3.6 packed staging`](../benchmarks/si-qwen36-27b-q4-packed-stage-2026-08-11.json).

**Four-GiB residency control.** The low 2.5 GiB Metal figure is a selected
residency profile, not a device limit. Increasing the retained MLP set from
stride four to stride two uses 4,351,856,640 active weight bytes and reports
4,057 MiB peak private Metal (3,786 MiB RSS). The eight-token control reached
0.277 decode / 0.280 total tok/s, so spending the available ~4 GiB does not
by itself recover throughput; staging, dispatch, and quantized-kernel costs
remain the dominant bottleneck. Artifact:
[`Qwen3.6 four-GiB residency`](../benchmarks/si-qwen36-27b-q4-residency-4g-2026-08-11.json).

The exact Qwen3.6 batched verifier primitive is implemented and checked against
sequential execution. With K=4, one sequential candidate window took
9,228 ms versus 26,456 ms for four separate target traversals (2.87x), with
matching greedy IDs and maximum logit difference 0.001047. K=8 did not improve
the end-to-end ratio further. The remaining blocker is proposal quality: the
resident-prefix drafter accepted only 1/4 candidates in the first probes, so
the verifier currently falls back to canonical decoding rather than claiming a
throughput win.

## Findings

1. **Capacity and bandwidth are separate problems.** mmap/paging solves the
   fast-memory capacity problem, but every generated token still touches a
   large fraction of dense weights. Streaming all 8 GB per token would require
   an impractical bandwidth budget at high tok/s.
2. **A small hot set is valuable.** Eight private layers improve throughput
   without returning to full-model residency.
3. **The tied output head is a major bottleneck.** Operation profiling observed
   roughly 1.3 seconds of cumulative wall time across 28 output-head calls in a
   short profile. It has 151,936 rows by 2,560 columns and is the clearest
   kernel-level target.
4. **The runtime is synchronization-heavy.** A profile showed every submitted
   command was also waited on. Tokio or a Rust async façade alone cannot create
   GPU overlap; command-buffer dependencies and output lifetimes must change.
5. **Allocator cleverness is not enough.** Scratch reuse, arenas, and grouping
   do not beat a good kernel/residency decision on this workload.
6. **Dynamic residency, not thermal throttling, is the current stability
   signal.** The machine reported no formal thermal warning, and the canonical
   4.607 run was produced earlier on the same hardware. The new opt-in
   `SI_PROFILE_RESOURCES=1` probe shows why isolated reruns diverge: a short
   slow repetition incurred thousands of macOS major page faults and tens of
   thousands of involuntary context switches, while a subsequent warm
   repetition reached 4.392 decode tok/s with zero major faults. With two
   warmups, three short repetitions reached 4.652 decode tok/s. These counters
   are diagnostic evidence of file-backed weight/Metal residency behavior, not
   proof of a source regression. Promotion must use a fixed preconditioning
   protocol and record page-fault and Metal-wait deltas alongside throughput.

## Current conclusion and next research gate

The engine has already demonstrated the core low-memory result: an unchanged
4B BF16 model can run with 1,540 MiB reported Metal allocation and 4.607 decode
tok/s, instead of requiring the full 7.7 GiB resident profile. The latest
multirow kernel is promising but has not yet beaten the canonical 4.607 decode
baseline in a controlled alternating sweep, so it stays opt-in.

The next research phase is
[`SI-004: Throughput Recovery Under Bounded Residency`](SI-004-throughput-recovery-spec.md).
It supersedes another isolated kernel sweep as the primary direction: first
the MPS matrix-vector oracle is now measured and rejected as a faster path;
the first exact batched projection primitive is now measured and promising;
next extend it through batched target verification so one streamed
target-weight traversal can verify multiple positions. Lookahead,
speculation, exact output-head search, lossless compressed streaming,
profile-guided retention, reusable command graphs, and CPU/GPU head splitting
follow in the order and with the gates defined by SI-004. `SI_MULTIROW_MATVEC=1`
remains an opt-in comparator, not a promoted baseline.
