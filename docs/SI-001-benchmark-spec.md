# SI-001: Capability and Benchmark Contract

Status: resident reference and mmap streaming captures recorded on 2026-08-08

## Objective

Measure whether Super-Inference reduces fast-memory residency while preserving
the useful capabilities of an unchanged dense model. Token identity is a
regression signal, not the definition of exactness.

## Immutable model object

- Model: `Qwen/Qwen3-4B-Base`.
- Revision: `906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- Format: original BF16 Safetensors shards and tokenizer files.
- Local artifact: `models/qwen3-4b-base/<revision>/`.
- No quantization, pruning, weight rewriting, or learned adapters in the
  lossless track.
- A checked-in manifest must record file sizes and SHA-256 digests before a
  result is considered reproducible.

## Canonical workload

- Host profile: Apple M3 Pro, 18 GB unified memory, Metal.
- Batch size: 1.
- Context limit: 2,048 tokens.
- Canonical prompt: `Explain why memory paging is useful for local model inference.`
- Generation: greedy decoding, temperature 0, no sampling, 128 new tokens.
- Warmup: 1 run; measured repetitions: 3; report median and worst peak memory.
- All profiles use the same tokenizer, prompt bytes, generation settings, and
  model revision.

## Quality suite

The initial suite stays small and versioned, but covers different capability
signals:

1. **Likelihood set:** 12 short, fixed, public-domain text excerpts. Report
   mean negative log-likelihood and perplexity.
2. **Structured completion set:** 12 completion-style prompts covering factual
   continuation, arithmetic, code completion, and simple multilingual text.
   Score normalized answer validity, not exact prose.
3. **Regression prompts:** 4 fixed prompts whose token sequences and logits are
   stored for debugging numerical drift. These do not define capability.

The fixture file must contain prompt text, expected answer criteria, tokenizer
revision, and scoring version. No private or user-provided data belongs in the
fixture.

## Capability acceptance

The full-resident BF16 reference is the baseline. For every optimized profile:

- lossless profiles target zero measurable quality delta from baseline;
- experimental profiles may be accepted only with an explicitly recorded
  aggregate quality delta of at most 1% and no suite category regression above
  2%;
- any failure must report the affected category and remain visible in the
  benchmark output;
- token/logit differences are diagnostic and must include max absolute error,
  mean absolute error, and first divergent token position.

## Required report fields

Each run emits model revision, profile, hardware, context, token counts,
prefill/decode/total throughput, peak process resident memory, peak Metal
allocation, mapped weight bytes, active weight bytes, KV bytes, scratch bytes,
quality scores, and regression diagnostics. Reports include repetition count,
median timing, worst and median sampled peaks, and output consistency across
repetitions. JSON output remains one object per run for JSONL collection.

## Non-goals

No claim is made about NVIDIA performance, multi-request serving, long-context
quality, quantized models, KV compression, or model fine-tuning in SI-001.
