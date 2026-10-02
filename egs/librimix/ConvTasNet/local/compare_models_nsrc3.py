#!/usr/bin/env python
"""Compare n_src=3 ConvTasNet checkpoints on 1/2/3-active online mixtures + streaming metrics."""

import argparse
import json
from dataclasses import dataclass

import numpy as np
import torch

from asteroid.data import OnlineMixDataset
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.models.base_models import BaseModel

from streaming_eval import evaluate_streaming


@dataclass
class ModelSpec:
    name: str
    path: str
    n_src: int


def parse_model_spec(text):
    parts = text.split("::")
    if len(parts) != 3:
        raise ValueError(f"Invalid --model spec: {text}. Expected name::path::n_src")
    name, path, n_src = parts
    return ModelSpec(name=name, path=path, n_src=int(n_src))


def sisdr(est, ref, eps=1e-8):
    est = est - np.mean(est)
    ref = ref - np.mean(ref)
    s_target = (np.sum(est * ref) / (np.sum(ref ** 2) + eps)) * ref
    e_noise = est - s_target
    return float(10 * np.log10((np.sum(s_target ** 2) + eps) / (np.sum(e_noise ** 2) + eps)))


def channel_rms_db(x, eps=1e-12):
    rms = np.sqrt(np.mean(np.asarray(x) ** 2) + eps)
    return float(20 * np.log10(rms + eps))


def summarize(vals):
    if not vals:
        return {}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def evaluate_fixed_active_count(
    model,
    device,
    n_src,
    n_active,
    source_dir,
    sample_rate,
    segment_s,
    num_examples,
    seed,
    active_rms_db,
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
    pit = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    active_sisdr = []
    silent_rms = []
    pred_counts = []

    with torch.no_grad():
        for i in range(len(ds)):
            mix, src = ds[i]
            mix = mix.to(device)
            src = src.to(device)
            est = model(mix.unsqueeze(0))
            _, reordered = pit(est, src.unsqueeze(0), return_est=True)
            re_np = reordered.squeeze(0).cpu().numpy()
            src_np = src.cpu().numpy()

            for k in range(n_active):
                active_sisdr.append(sisdr(re_np[k], src_np[k]))
            for k in range(n_active, n_src):
                silent_rms.append(channel_rms_db(re_np[k]))

            pred_counts.append(int(sum(channel_rms_db(re_np[k]) > active_rms_db for k in range(n_src))))

    pred_arr = np.asarray(pred_counts, dtype=np.float64)
    out = {
        "n_active": int(n_active),
        "num_examples": int(len(ds)),
        "mean_si_sdr_db": float(np.mean(active_sisdr)) if active_sisdr else None,
        "median_si_sdr_db": float(np.median(active_sisdr)) if active_sisdr else None,
        "active_count_mae": float(np.mean(np.abs(pred_arr - n_active))) if pred_counts else None,
        "p_exact_active_count": float(np.mean(pred_arr == n_active)) if pred_counts else None,
        "mean_predicted_active_channels": float(np.mean(pred_arr)) if pred_counts else None,
        "active_sisdr_summary": summarize(active_sisdr),
        "silent_rms_summary_db": summarize(silent_rms),
    }
    return out


def build_summary(report, ordered_models):
    lines = []
    lines.append("# n_src=3 Baseline Comparison")
    lines.append("")
    lines.append("## Models")
    for name in ordered_models:
        spec = report["models"][name]["spec"]
        lines.append(f"- `{name}`: n_src={spec['n_src']} path=`{spec['path']}`")
    lines.append("")

    for name in ordered_models:
        m = report["models"][name]
        lines.append(f"## {name}")
        for k in sorted(m["per_active"].keys()):
            p = m["per_active"][k]
            lines.append(
                "- "
                f"{k}: mean_si_sdr={p['mean_si_sdr_db']:.3f} dB, "
                f"mean_pred_active={p['mean_predicted_active_channels']:.3f}, "
                f"active_count_mae={p['active_count_mae']:.3f}, "
                f"p_exact={p['p_exact_active_count']:.3f}, "
                f"silent_rms_mean={p['silent_rms_summary_db'].get('mean', float('nan')):.3f} dB"
            )
        s = m["streaming"]
        lines.append(
            "- "
            f"streaming ({s['window_ms']:.0f}ms/{s['hop_ms']:.0f}ms): "
            f"lat={s['avg_step_latency_ms']:.3f}ms, p95={s['p95_step_latency_ms']:.3f}ms, rtf={s['rtf']:.3f}"
        )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="name::path_or_hub_id::n_src")
    parser.add_argument("--source_dir", default="~/data/librimix/LibriSpeech/test-clean")
    parser.add_argument("--sample_rate", type=int, default=8000)
    parser.add_argument("--segment_s", type=float, default=3.0)
    parser.add_argument("--num_examples_per_n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--active_rms_db", type=float, default=-40.0)
    parser.add_argument("--window_ms", type=float, default=1000.0)
    parser.add_argument("--hop_ms", type=float, default=50.0)
    parser.add_argument("--stream_num_examples", type=int, default=30)
    parser.add_argument("--stream_warmup_steps", type=int, default=10)
    parser.add_argument("--stream_timed_steps", type=int, default=50)
    parser.add_argument("--out", default="compare_baseline_models_nsrc3.json")
    parser.add_argument("--summary_md", default=None)
    args = parser.parse_args()

    specs = [parse_model_spec(x) for x in args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    report = {
        "device": device,
        "settings": {
            "sample_rate": args.sample_rate,
            "segment_s": args.segment_s,
            "num_examples_per_n": args.num_examples_per_n,
            "active_rms_db": args.active_rms_db,
            "window_ms": args.window_ms,
            "hop_ms": args.hop_ms,
        },
        "models": {},
    }

    for m in specs:
        model = BaseModel.from_pretrained(m.path).eval().to(device)

        per_active = {}
        for n_active in (1, 2, 3):
            if n_active > m.n_src:
                continue
            per_active[f"{n_active}spk"] = evaluate_fixed_active_count(
                model=model,
                device=device,
                n_src=m.n_src,
                n_active=n_active,
                source_dir=args.source_dir,
                sample_rate=args.sample_rate,
                segment_s=args.segment_s,
                num_examples=args.num_examples_per_n,
                seed=args.seed + n_active,
                active_rms_db=args.active_rms_db,
            )

        stream = evaluate_streaming(
            model=model,
            sample_rate=args.sample_rate,
            source_dir=args.source_dir,
            n_src=m.n_src,
            n_active=min(3, m.n_src),
            segment_s=args.segment_s,
            num_examples=args.stream_num_examples,
            seed=args.seed + 1000,
            window_ms=args.window_ms,
            hop_ms=args.hop_ms,
            warmup_steps=args.stream_warmup_steps,
            timed_steps=args.stream_timed_steps,
            device=device,
        )

        report["models"][m.name] = {
            "spec": {"path": m.path, "n_src": m.n_src},
            "per_active": per_active,
            "streaming": stream,
        }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    ordered = [m.name for m in specs]
    summary = build_summary(report, ordered)
    if args.summary_md:
        with open(args.summary_md, "w") as f:
            f.write(summary + "\n")

    print(json.dumps(report, indent=2))
    print(f"\nSaved report to {args.out}")
    if args.summary_md:
        print(f"Saved summary to {args.summary_md}")


if __name__ == "__main__":
    main()
