#!/usr/bin/env python
"""Streaming-focused latency and consistency evaluation for ConvTasNet-like models."""

import argparse
import json
import time

import numpy as np
import torch

from asteroid.data import OnlineMixDataset
from asteroid.models.base_models import BaseModel


def evaluate_streaming(
    model,
    sample_rate=8000,
    source_dir="~/data/librimix/LibriSpeech/test-clean",
    n_src=3,
    n_active=3,
    segment_s=3.0,
    num_examples=30,
    seed=123,
    window_ms=1000.0,
    hop_ms=50.0,
    warmup_steps=10,
    timed_steps=50,
    device="cpu",
):
    ds = OnlineMixDataset(
        source_dir=source_dir,
        n_src=n_src,
        sample_rate=sample_rate,
        segment=segment_s,
        min_speakers=n_active,
        max_speakers=n_active,
        num_examples=num_examples,
        seed=seed,
    )

    window_samples = int(window_ms * sample_rate / 1000.0)
    hop_samples = int(hop_ms * sample_rate / 1000.0)
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_ms and hop_ms must resolve to positive sample counts")

    latencies_ms = []
    max_abs_diffs = []
    corr_vals = []

    with torch.no_grad():
        for i in range(len(ds)):
            mixture, _ = ds[i]
            mix = mixture.to(device)
            total = int(mix.shape[0])
            if total < window_samples:
                continue

            full = model(mix.unsqueeze(0)).cpu().squeeze(0).numpy()

            stream_out = np.zeros((n_src, total), dtype=np.float32)
            stream_weight = np.zeros(total, dtype=np.float32)

            starts = list(range(0, total - window_samples + 1, hop_samples))
            if not starts:
                continue

            # Warmup on one representative window.
            warm = mix[:window_samples].unsqueeze(0)
            for _ in range(max(0, warmup_steps)):
                _ = model(warm)
            if device.startswith("cuda"):
                torch.cuda.synchronize()

            step_count = 0
            for s in starts:
                window = mix[s : s + window_samples].unsqueeze(0)

                if step_count < timed_steps:
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    est = model(window)
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    latencies_ms.append((t1 - t0) * 1000.0)
                else:
                    est = model(window)

                est_np = est.cpu().squeeze(0).numpy()
                stream_out[:, s : s + window_samples] += est_np
                stream_weight[s : s + window_samples] += 1.0
                step_count += 1

            valid = stream_weight > 0
            if not np.any(valid):
                continue
            stream_out[:, valid] /= stream_weight[valid][None, :]

            t_len = min(stream_out.shape[1], full.shape[1])
            d = np.max(np.abs(stream_out[:, :t_len] - full[:, :t_len]))
            max_abs_diffs.append(float(d))

            for ch in range(n_src):
                a = stream_out[ch, :t_len]
                b = full[ch, :t_len]
                if np.std(a) > 1e-8 and np.std(b) > 1e-8:
                    corr_vals.append(float(np.corrcoef(a, b)[0, 1]))

    if not latencies_ms:
        raise RuntimeError("No timed streaming steps were collected.")

    avg_latency_ms = float(np.mean(latencies_ms))
    std_latency_ms = float(np.std(latencies_ms))
    p95_latency_ms = float(np.percentile(latencies_ms, 95))

    return {
        "window_ms": float(window_ms),
        "hop_ms": float(hop_ms),
        "window_samples": int(window_samples),
        "hop_samples": int(hop_samples),
        "avg_step_latency_ms": avg_latency_ms,
        "std_step_latency_ms": std_latency_ms,
        "p95_step_latency_ms": p95_latency_ms,
        "rtf": float(avg_latency_ms / float(hop_ms)),
        "mean_full_vs_stream_corr": float(np.mean(corr_vals)) if corr_vals else None,
        "mean_full_vs_stream_max_abs_diff": float(np.mean(max_abs_diffs)) if max_abs_diffs else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path or hub id for BaseModel.from_pretrained")
    parser.add_argument("--n_src", type=int, default=3)
    parser.add_argument("--n_active", type=int, default=3)
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--source_dir", default="~/data/librimix/LibriSpeech/test-clean")
    parser.add_argument("--segment_s", type=float, default=3.0)
    parser.add_argument("--num_examples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--window_ms", type=float, default=1000.0)
    parser.add_argument("--hop_ms", type=float, default=50.0)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--timed_steps", type=int, default=50)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BaseModel.from_pretrained(args.model).eval().to(device)

    out = evaluate_streaming(
        model=model,
        sample_rate=args.sample_rate,
        source_dir=args.source_dir,
        n_src=args.n_src,
        n_active=args.n_active,
        segment_s=args.segment_s,
        num_examples=args.num_examples,
        seed=args.seed,
        window_ms=args.window_ms,
        hop_ms=args.hop_ms,
        warmup_steps=args.warmup_steps,
        timed_steps=args.timed_steps,
        device=device,
    )

    payload = {"device": device, "model": args.model, "metrics": out}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
