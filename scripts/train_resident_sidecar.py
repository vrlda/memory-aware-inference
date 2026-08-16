#!/usr/bin/env python3
"""Train and measure a disposable low-rank resident-layer sidecar.

The target model is never changed. The sidecar learns a linear map from the
retained-prefix hidden state to a low-rank approximation of the target's
normalized final hidden state, then scores the untouched vocabulary embedding
through that low-rank basis. It is a proposal-only artifact; exact target
verification remains mandatory.
"""

import argparse
import json
import mmap
import os
import struct
import time
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--ridge", type=float, default=1e-3)
    return parser.parse_args()


def load_trace(paths):
    records = []
    for path in paths:
        with open(path, "r") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("record_type") == "record":
                    if "target_hidden" not in row:
                        raise ValueError(
                            f"{path} lacks target_hidden; regenerate it with the current probe"
                        )
                    records.append(row)
    if len(records) < 8:
        raise ValueError("at least eight trace records are required")
    hidden = np.asarray([row["hidden"] for row in records], dtype=np.float32)
    target_hidden = np.asarray(
        [row["target_hidden"] for row in records], dtype=np.float32
    )
    if hidden.ndim != 2 or target_hidden.shape != hidden.shape:
        raise ValueError("trace hidden-state dimensions are inconsistent")
    return records, hidden, target_hidden


def locate_model_revision(model_dir):
    model_dir = Path(model_dir)
    index_paths = sorted(model_dir.glob("*/model.safetensors.index.json"))
    if not index_paths:
        raise FileNotFoundError(f"no safetensors index under {model_dir}")
    return index_paths[0].parent, index_paths[0]


def load_safetensor(model_dir, tensor_name):
    revision_dir, index_path = locate_model_revision(model_dir)
    with open(index_path, "r") as stream:
        index = json.load(stream)
    shard_name = index["weight_map"][tensor_name]
    shard_path = revision_dir / shard_name
    with open(shard_path, "rb") as stream:
        header_len = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_len))
        entry = header[tensor_name]
        start, end = entry["data_offsets"]
        stream.seek(8 + header_len + start)
        raw = np.frombuffer(stream.read(end - start), dtype="<u2").copy()
    if entry["dtype"] != "BF16":
        raise ValueError(f"{tensor_name} is {entry['dtype']}, expected BF16")
    shape = tuple(entry["shape"])
    if raw.size != int(np.prod(shape)):
        raise ValueError(f"{tensor_name} byte length does not match shape")
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(shape)


def load_config(model_dir):
    revision_dir, _ = locate_model_revision(model_dir)
    with open(revision_dir / "config.json", "r") as stream:
        return json.load(stream)


def rms_norm(hidden, weight, epsilon):
    mean_square = np.mean(hidden * hidden, axis=1, keepdims=True)
    return hidden * (1.0 / np.sqrt(mean_square + epsilon)) * weight


def build_vocab_projection(model_dir, basis, mean_target, chunk_rows=4096):
    revision_dir, index_path = locate_model_revision(model_dir)
    with open(index_path, "r") as stream:
        index = json.load(stream)
    shard_name = index["weight_map"]["model.embed_tokens.weight"]
    shard_path = revision_dir / shard_name
    with open(shard_path, "rb") as stream:
        header_len = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_len))
        entry = header["model.embed_tokens.weight"]
        start, end = entry["data_offsets"]
        shape = tuple(entry["shape"])
        stream_start = 8 + header_len + start
    vocab, hidden_size = shape
    rank = basis.shape[0]
    projection = np.empty((vocab, rank), dtype=np.float32)
    bias = np.empty(vocab, dtype=np.float32)
    with open(shard_path, "rb") as stream:
        stream.seek(stream_start)
        for row_start in range(0, vocab, chunk_rows):
            row_count = min(chunk_rows, vocab - row_start)
            byte_count = row_count * hidden_size * 2
            raw = np.frombuffer(stream.read(byte_count), dtype="<u2").copy()
            rows = (raw.astype(np.uint32) << 16).view(np.float32).reshape(
                row_count, hidden_size
            )
            projection[row_start : row_start + row_count] = rows @ basis.T
            bias[row_start : row_start + row_count] = rows @ mean_target
    return projection, bias


def top_ids(scores, limit):
    indices = np.argpartition(scores, -limit)[-limit:]
    return indices[np.argsort(scores[indices])[::-1]]


def write_binary_sidecar(path, input_mean, input_to_latent, vocab_projection, vocab_bias):
    """Write the inference-only f32 sidecar format consumed by the Rust probe.

    The NPZ retains the training tensors and diagnostics. The compact binary
    contains only the centered input map and vocabulary scorer needed online.
    """
    hidden_size, rank = input_to_latent.shape
    vocab_size = vocab_projection.shape[0]
    with open(path, "wb") as stream:
        stream.write(b"SISCAR01")
        stream.write(struct.pack("<IIII", hidden_size, rank, vocab_size, 1))
        np.asarray(input_mean, dtype="<f4").tofile(stream)
        np.asarray(input_to_latent, dtype="<f4").tofile(stream)
        np.asarray(vocab_projection, dtype="<f4").tofile(stream)
        np.asarray(vocab_bias, dtype="<f4").tofile(stream)


def main():
    args = parse_args()
    if args.rank <= 0:
        raise ValueError("--rank must be positive")
    if not 0.0 < args.holdout < 1.0:
        raise ValueError("--holdout must be between zero and one")
    records, hidden, target_hidden = load_trace(args.trace)
    config = load_config(args.model)
    final_norm = load_safetensor(args.model, "model.norm.weight")
    target_hidden = rms_norm(target_hidden, final_norm, config["rms_norm_eps"])
    order = np.arange(len(records))
    rng = np.random.default_rng(17)
    rng.shuffle(order)
    holdout_count = max(1, int(round(len(order) * args.holdout)))
    test_ids = order[:holdout_count]
    train_ids = order[holdout_count:]
    if train_ids.size < 4:
        raise ValueError("holdout leaves too few training records")
    train_hidden = hidden[train_ids]
    train_target = target_hidden[train_ids]
    input_mean = train_hidden.mean(axis=0)
    target_mean = train_target.mean(axis=0)
    centered_target = train_target - target_mean
    _, _, components = np.linalg.svd(centered_target, full_matrices=False)
    rank = min(args.rank, components.shape[0])
    basis = components[:rank].astype(np.float32)
    latent_target = centered_target @ basis.T
    centered_input = train_hidden - input_mean
    gram = centered_input @ centered_input.T
    gram.flat[:: gram.shape[0] + 1] += args.ridge
    dual = np.linalg.solve(gram, latent_target)
    input_to_latent = (centered_input.T @ dual).astype(np.float32)

    started = time.perf_counter()
    vocab_projection, vocab_bias = build_vocab_projection(
        args.model, basis, target_mean
    )
    build_seconds = time.perf_counter() - started

    test_hidden = hidden[test_ids] - input_mean
    test_latent = test_hidden @ input_to_latent
    score_started = time.perf_counter()
    scores = test_latent @ vocab_projection.T + vocab_bias
    score_seconds = time.perf_counter() - score_started
    top1_hits = 0
    target_top1_in_top4 = 0
    target_top4_overlap = 0
    sidecar_top4 = []
    for row_index, record_id in enumerate(test_ids):
        predicted = top_ids(scores[row_index], 4)
        target_top4 = set(records[record_id]["target_top4"])
        target_top1 = records[record_id]["target_top1"]
        sidecar_top4.append(predicted.tolist())
        top1_hits += int(int(predicted[0]) == target_top1)
        target_top1_in_top4 += int(target_top1 in predicted)
        target_top4_overlap += int(bool(target_top4.intersection(predicted)))

    output_path = Path(args.output)
    binary_path = output_path.with_suffix(".sisc")
    write_binary_sidecar(
        binary_path,
        input_mean,
        input_to_latent,
        vocab_projection,
        vocab_bias,
    )
    np.savez(
        output_path,
        input_mean=input_mean.astype(np.float32),
        input_to_latent=input_to_latent,
        latent_basis=basis,
        target_mean=target_mean.astype(np.float32),
        vocab_projection=vocab_projection.astype(np.float16),
        vocab_bias=vocab_bias.astype(np.float16),
        rank=np.asarray(rank, dtype=np.int32),
    )
    report = {
        "experiment": "resident-layer-low-rank-sidecar",
        "model": str(args.model),
        "trace_files": args.trace,
        "records": len(records),
        "train_records": int(train_ids.size),
        "holdout_records": int(test_ids.size),
        "rank": rank,
        "ridge": args.ridge,
        "target_representation": "model.norm(target_hidden)",
        "sidecar_bytes": int(os.path.getsize(output_path)),
        "binary_path": str(binary_path),
        "binary_bytes": int(os.path.getsize(binary_path)),
        "build_vocab_projection_seconds": build_seconds,
        "score_ms_per_holdout": score_seconds * 1_000.0 / len(test_ids),
        "holdout_sidecar_top1_accuracy": top1_hits / len(test_ids),
        "holdout_target_top1_in_sidecar_top4": target_top1_in_top4 / len(test_ids),
        "holdout_target_top4_overlap": target_top4_overlap / len(test_ids),
        "proposal_only": True,
        "exact_target_unchanged": True,
        "promotion": "coverage_probe_only",
        "sidecar_top4": sidecar_top4,
    }
    with open(args.report, "w") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, separators=(",", ":")))


if __name__ == "__main__":
    main()
