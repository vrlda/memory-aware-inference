# SI benchmark captures

These JSON files are immutable captures from the pinned Qwen3-4B-Base BF16
artifact on the Apple M3 Pro. The canonical command is documented in the root
README and SI-001 spec. The quality-bearing capture includes the same measured
workload plus the versioned `quality-v0` suite; the speed-only capture is kept
to make the timing/memory run independently auditable.

Do not compare these captures to a different quantization, model revision,
prompt, context capacity, or sampling policy. Future profiles should emit the
same fields and be compared against the resident reference.

The mmap captures are the first lossless residency experiment. The base point
binds mapped Safetensors bytes directly without retaining a weight buffer; the
hot-head point additionally retains only the tied output head in private Metal
storage. The hot-head `-quality.json` capture records the versioned quality
suite and matches the resident reference scores.

The retain-8 capture keeps the first eight transformer layers in private Metal
buffers; it is the current bounded layer-residency Pareto point.

The `llama-cpp-metal-bf16` capture is the external SI-001 reference. It uses
the official llama.cpp source build with native Metal and a GGUF BF16 container
created from the same pinned Safetensors weights. The conversion changes the
container format only; it is not quantization. Its throughput and memory fields
are intentionally kept separate from the SI runtime captures because the
external tool reports mapped Metal allocations and host RSS differently.
