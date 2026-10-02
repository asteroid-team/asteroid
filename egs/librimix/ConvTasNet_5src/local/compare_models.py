#!/usr/bin/env python
"""Compare separation checkpoints on standardized debug benchmarks.

Example:
python local/compare_models.py \
  --model "new12::exp/debug_2in5_e12/best_model.pth::5" \
  --model "legacy::exp/train_convtasnet_5src_online_5src/best_model.pth::5" \
  --model "hub2::Cosentino/ConvTasNet_LibriMix_sep_clean::2" \
  --out compare_report.json
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch

from asteroid.data import OnlineMixDataset, VariableLibriMix
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.models.base_models import BaseModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECIPE_DIR = os.path.dirname(SCRIPT_DIR)
if RECIPE_DIR not in sys.path:
    sys.path.insert(0, RECIPE_DIR)

from eval import reorder_active_first


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
    s_target = (np.sum(est * ref) / (np.sum(ref**2) + eps)) * ref
    e_noise = est - s_target
    return 10 * np.log10((np.sum(s_target**2) + eps) / (np.sum(e_noise**2) + eps))


def channel_rms_db(signal, eps=1e-12):
    rms = np.sqrt(np.mean(signal**2) + eps)
    return float(20 * np.log10(rms + eps))


def magnitude_spectrogram(signal_np, n_fft=256, hop_length=64):
    sig_t = torch.from_numpy(np.asarray(signal_np)).float()
    window = torch.hann_window(n_fft)
    stft = torch.stft(
        sig_t,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    return torch.abs(stft).numpy()


def log_spectral_distance_db(estimate, reference, eps=1e-8, n_fft=256, hop_length=64):
    est_mag = magnitude_spectrogram(estimate, n_fft=n_fft, hop_length=hop_length)
    ref_mag = magnitude_spectrogram(reference, n_fft=n_fft, hop_length=hop_length)
    est_log = 20.0 * np.log10(est_mag + eps)
    ref_log = 20.0 * np.log10(ref_mag + eps)
    return float(np.mean(np.sqrt((est_log - ref_log) ** 2)))


def spectral_convergence(estimate, reference, eps=1e-8, n_fft=256, hop_length=64):
    est_mag = magnitude_spectrogram(estimate, n_fft=n_fft, hop_length=hop_length)
    ref_mag = magnitude_spectrogram(reference, n_fft=n_fft, hop_length=hop_length)
    num = np.linalg.norm(ref_mag - est_mag, ord="fro")
    den = np.linalg.norm(ref_mag, ord="fro") + eps
    return float(num / den)


def summarize_silent_rms(silent_rms_db):
    if not silent_rms_db:
        return {}
    values = np.asarray(silent_rms_db, dtype=np.float64)
    return {
        "mean_silent_rms_db": float(np.mean(values)),
        "max_silent_rms_db": float(np.max(values)),
        "p50_silent_rms_db": float(np.percentile(values, 50)),
        "p95_silent_rms_db": float(np.percentile(values, 95)),
        "p99_silent_rms_db": float(np.percentile(values, 99)),
    }


def summarize_spectral_metrics(lsd_values_db, sc_values):
    if not lsd_values_db or not sc_values:
        return {}
    lsd = np.asarray(lsd_values_db, dtype=np.float64)
    sc = np.asarray(sc_values, dtype=np.float64)
    return {
        "mean_log_spectral_distance_db": float(np.mean(lsd)),
        "p95_log_spectral_distance_db": float(np.percentile(lsd, 95)),
        "mean_spectral_convergence": float(np.mean(sc)),
        "p95_spectral_convergence": float(np.percentile(sc, 95)),
    }


def benchmark_online_2spk(model, n_src, device, root, num_examples, seed):
    ds = OnlineMixDataset(
        source_dir=os.path.expanduser(root),
        n_src=5,
        sample_rate=8000,
        segment=3.0,
        min_speakers=2,
        max_speakers=2,
        num_examples=num_examples,
        seed=seed,
    )
    pit2 = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    active_scores = []
    active_lsd_db = []
    active_sc = []
    silent_rms_db = []
    predicted_active = []
    expected_active = 2
    with torch.no_grad():
        for i in range(len(ds)):
            mix, src5 = ds[i]
            est = model(mix.unsqueeze(0).to(device))
            if n_src == 5:
                src = src5.to(device)
                re = reorder_active_first(est, src.unsqueeze(0), 1e-5).squeeze(0).cpu().numpy()
                ref = src.cpu().numpy()
                for k in (0, 1):
                    active_scores.append(sisdr(re[k], ref[k]))
                    active_lsd_db.append(log_spectral_distance_db(re[k], ref[k]))
                    active_sc.append(spectral_convergence(re[k], ref[k]))
                for k in (2, 3, 4):
                    silent_rms_db.append(channel_rms_db(re[k]))
                predicted_active.append(
                    int(
                        sum(channel_rms_db(re[k]) > -40 for k in range(5))
                    )
                )
            elif n_src == 2:
                src2 = src5[:2].to(device)
                _, re = pit2(est, src2.unsqueeze(0), return_est=True)
                re = re.squeeze(0).cpu().numpy()
                ref = src2.cpu().numpy()
                for k in (0, 1):
                    active_scores.append(sisdr(re[k], ref[k]))
                    active_lsd_db.append(log_spectral_distance_db(re[k], ref[k]))
                    active_sc.append(spectral_convergence(re[k], ref[k]))
                predicted_active.append(
                    int(
                        sum(channel_rms_db(re[k]) > -40 for k in range(2))
                    )
                )
            else:
                raise ValueError(f"Unsupported n_src={n_src}. Only 2 or 5 are supported.")

    predicted_arr = np.asarray(predicted_active, dtype=np.float64)
    out = {
        "mean_active_sisdr_db": float(np.mean(active_scores)),
        "median_active_sisdr_db": float(np.median(active_scores)),
        "mean_predicted_active_channels": float(np.mean(predicted_arr)),
        "active_count_mae": float(np.mean(np.abs(predicted_arr - expected_active))),
        "p_active_2_correct": float(np.mean(predicted_arr == expected_active)),
    }
    out.update(summarize_silent_rms(silent_rms_db))
    out.update(summarize_spectral_metrics(active_lsd_db, active_sc))
    return out


def benchmark_variable_test(model, n_src, device, test_dir, per_n_count=300):
    ds = VariableLibriMix(
        csv_dirs=test_dir,
        task="sep_clean",
        sample_rate=8000,
        n_src=5,
        segment=None,
    )
    idx_by_n = {2: [], 3: []}
    for i in range(len(ds)):
        _, src = ds[i]
        n_active = int(((src.numpy() ** 2).sum(axis=1) > 1e-5).sum())
        if n_active in idx_by_n and len(idx_by_n[n_active]) < per_n_count:
            idx_by_n[n_active].append(i)
        if all(len(v) >= per_n_count for v in idx_by_n.values()):
            break

    pit2 = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    results = {}
    with torch.no_grad():
        for n_active in (2, 3):
            if n_src == 2 and n_active != 2:
                results[f"{n_active}spk"] = {"status": "skipped_for_2src_model"}
                continue
            idxs = idx_by_n[n_active]
            sisdr_vals = []
            silent_vals = []
            for idx in idxs:
                mix, src5 = ds[idx]
                est = model(mix.unsqueeze(0).to(device))
                if n_src == 5:
                    src = src5.to(device)
                    re = reorder_active_first(est, src.unsqueeze(0), 1e-5).squeeze(0).cpu().numpy()
                    ref = src.cpu().numpy()
                    for k in range(n_active):
                        sisdr_vals.append(sisdr(re[k], ref[k]))
                    for k in range(n_active, 5):
                        silent_vals.append(channel_rms_db(re[k]))
                else:
                    src2 = src5[:2].to(device)
                    _, re = pit2(est, src2.unsqueeze(0), return_est=True)
                    re = re.squeeze(0).cpu().numpy()
                    ref = src2.cpu().numpy()
                    for k in (0, 1):
                        sisdr_vals.append(sisdr(re[k], ref[k]))
            entry = {
                "num_examples": len(idxs),
                "mean_si_sdr_db": float(np.mean(sisdr_vals)),
                "median_si_sdr_db": float(np.median(sisdr_vals)),
            }
            entry.update(summarize_silent_rms(silent_vals))
            results[f"{n_active}spk"] = entry
    return results


def evaluate_online_gate(candidate_metrics, baseline_metrics, args):
    gate = {
        "status": "not_evaluated",
        "reasons": [],
    }
    if baseline_metrics is None:
        gate["status"] = "skipped_no_baseline"
        gate["reasons"].append("baseline metrics not available")
        return gate

    if "max_silent_rms_db" not in candidate_metrics:
        gate["status"] = "skipped_no_silent_metrics"
        gate["reasons"].append("silent RMS metrics missing")
        return gate
    if (
        "mean_log_spectral_distance_db" not in candidate_metrics
        or "p95_log_spectral_distance_db" not in candidate_metrics
        or "mean_spectral_convergence" not in candidate_metrics
        or "p95_spectral_convergence" not in candidate_metrics
    ):
        gate["status"] = "skipped_no_spectral_metrics"
        gate["reasons"].append("spectral metrics missing")
        return gate

    delta_active = candidate_metrics["mean_active_sisdr_db"] - baseline_metrics["mean_active_sisdr_db"]
    cond_active = delta_active >= args.gate_min_active_sisdr_delta_db
    cond_silent = candidate_metrics["max_silent_rms_db"] <= args.gate_max_silent_rms_db
    mean_active = candidate_metrics["mean_predicted_active_channels"]
    cond_count = args.gate_active_count_min <= mean_active <= args.gate_active_count_max
    cond_lsd_mean = (
        candidate_metrics["mean_log_spectral_distance_db"] <= args.gate_max_mean_lsd_db
    )
    cond_lsd_p95 = (
        candidate_metrics["p95_log_spectral_distance_db"] <= args.gate_max_p95_lsd_db
    )
    cond_sc_mean = (
        candidate_metrics["mean_spectral_convergence"] <= args.gate_max_mean_sc
    )
    cond_sc_p95 = (
        candidate_metrics["p95_spectral_convergence"] <= args.gate_max_p95_sc
    )
    all_ok = (
        cond_active
        and cond_silent
        and cond_count
        and cond_lsd_mean
        and cond_lsd_p95
        and cond_sc_mean
        and cond_sc_p95
    )

    gate.update(
        {
            "status": "pass" if all_ok else "fail",
            "delta_active_sisdr_db": float(delta_active),
            "conditions": {
                "active_sisdr_non_regression": bool(cond_active),
                "max_silent_rms": bool(cond_silent),
                "mean_predicted_active_range": bool(cond_count),
                "mean_log_spectral_distance_db": bool(cond_lsd_mean),
                "p95_log_spectral_distance_db": bool(cond_lsd_p95),
                "mean_spectral_convergence": bool(cond_sc_mean),
                "p95_spectral_convergence": bool(cond_sc_p95),
            },
            "thresholds": {
                "min_active_sisdr_delta_db": args.gate_min_active_sisdr_delta_db,
                "max_silent_rms_db": args.gate_max_silent_rms_db,
                "mean_predicted_active_min": args.gate_active_count_min,
                "mean_predicted_active_max": args.gate_active_count_max,
                "max_mean_lsd_db": args.gate_max_mean_lsd_db,
                "max_p95_lsd_db": args.gate_max_p95_lsd_db,
                "max_mean_sc": args.gate_max_mean_sc,
                "max_p95_sc": args.gate_max_p95_sc,
            },
        }
    )
    if not cond_active:
        gate["reasons"].append("active SI-SDR regressed beyond threshold")
    if not cond_silent:
        gate["reasons"].append("silent-channel leakage above threshold")
    if not cond_count:
        gate["reasons"].append("mean predicted active channel count out of range")
    if not cond_lsd_mean:
        gate["reasons"].append("mean log spectral distance above threshold")
    if not cond_lsd_p95:
        gate["reasons"].append("p95 log spectral distance above threshold")
    if not cond_sc_mean:
        gate["reasons"].append("mean spectral convergence above threshold")
    if not cond_sc_p95:
        gate["reasons"].append("p95 spectral convergence above threshold")
    return gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model spec: name::path_or_hub_id::n_src",
    )
    parser.add_argument("--online_source_dir", default="~/data/librimix/LibriSpeech/test-clean")
    parser.add_argument("--online_num_examples", type=int, default=200)
    parser.add_argument("--online_seed", type=int, default=123)
    parser.add_argument("--test_dir", default="data/wav8k/min/test")
    parser.add_argument("--test_per_n", type=int, default=300)
    parser.add_argument("--gate_baseline", type=str, default="opt3")
    parser.add_argument("--gate_min_active_sisdr_delta_db", type=float, default=-0.5)
    parser.add_argument("--gate_max_silent_rms_db", type=float, default=-22.0)
    parser.add_argument("--gate_active_count_min", type=float, default=1.8)
    parser.add_argument("--gate_active_count_max", type=float, default=2.6)
    parser.add_argument("--gate_max_mean_lsd_db", type=float, default=15.0)
    parser.add_argument("--gate_max_p95_lsd_db", type=float, default=20.0)
    parser.add_argument("--gate_max_mean_sc", type=float, default=0.82)
    parser.add_argument("--gate_max_p95_sc", type=float, default=0.90)
    parser.add_argument("--out", default="compare_models_report.json")
    args = parser.parse_args()

    specs = [parse_model_spec(m) for m in args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {"device": device, "models": {}}

    for spec in specs:
        model = BaseModel.from_pretrained(spec.path).eval().to(device)
        online = benchmark_online_2spk(
            model=model,
            n_src=spec.n_src,
            device=device,
            root=args.online_source_dir,
            num_examples=args.online_num_examples,
            seed=args.online_seed,
        )
        variable = benchmark_variable_test(
            model=model,
            n_src=spec.n_src,
            device=device,
            test_dir=args.test_dir,
            per_n_count=args.test_per_n,
        )
        report["models"][spec.name] = {
            "spec": {"path": spec.path, "n_src": spec.n_src},
            "online_2spk": online,
            "variable_test": variable,
        }

    baseline_metrics = None
    if args.gate_baseline in report["models"]:
        baseline_metrics = report["models"][args.gate_baseline]["online_2spk"]

    gate_summary = {}
    for name, model_report in report["models"].items():
        gate_summary[name] = evaluate_online_gate(
            candidate_metrics=model_report["online_2spk"],
            baseline_metrics=baseline_metrics,
            args=args,
        )

    report["gate_summary"] = {
        "baseline_name": args.gate_baseline,
        "online_2spk": gate_summary,
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved report to {args.out}")


if __name__ == "__main__":
    main()
