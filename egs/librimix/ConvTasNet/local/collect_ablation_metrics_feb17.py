#!/usr/bin/env python
"""Collect and summarize ConvTasNet Feb 17 ablation metrics."""

import argparse
import json
from pathlib import Path


def _read_json(path):
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def _get_full_metrics(exp_dir, eval_tag):
    full = _read_json(Path(exp_dir) / eval_tag / "final_metrics.json")
    if not full:
        return {}
    return {
        "full_si_sdr_db": full.get("si_sdr"),
        "full_si_sdr_imp_db": full.get("si_sdr_imp"),
        "full_stoi": full.get("stoi"),
        "full_sdr_db": full.get("sdr"),
    }


def _get_stream_metrics(compare_json_path, model_key):
    payload = _read_json(compare_json_path)
    if not payload:
        return {}
    m = payload.get("models", {}).get(model_key, {})
    streaming = m.get("streaming", {})
    per_active = m.get("per_active", {})

    out = {
        "stream_avg_latency_ms": streaming.get("avg_step_latency_ms"),
        "stream_p95_latency_ms": streaming.get("p95_step_latency_ms"),
        "stream_rtf": streaming.get("rtf"),
        "stream_mean_corr": streaming.get("mean_full_vs_stream_corr"),
        "stream_mean_max_abs_diff": streaming.get("mean_full_vs_stream_max_abs_diff"),
    }

    for k in ("1spk", "2spk", "3spk"):
        row = per_active.get(k, {})
        out[f"{k}_si_sdr_db"] = row.get("mean_si_sdr_db")
        out[f"{k}_active_count_mae"] = row.get("active_count_mae")
        out[f"{k}_mean_pred_active"] = row.get("mean_predicted_active_channels")
        out[f"{k}_p_exact_active_count"] = row.get("p_exact_active_count")
    return out


def _infer_cfg(conf_path):
    if not Path(conf_path).is_file():
        return {}
    try:
        import yaml

        conf = yaml.safe_load(Path(conf_path).read_text())
    except Exception:
        return {}

    m = conf.get("masknet", {})
    f = conf.get("filterbank", {})
    return {
        "n_blocks": m.get("n_blocks"),
        "n_repeats": m.get("n_repeats"),
        "bn_chan": m.get("bn_chan"),
        "skip_chan": m.get("skip_chan"),
        "hid_chan": m.get("hid_chan"),
        "n_filters": f.get("n_filters"),
    }


def collect(run_root, eval_tag, compare_json_name):
    run_root = Path(run_root)
    out = {"run_root": str(run_root), "models": {}}

    # Collect any model directory that contains a config, regardless of ID prefix.
    for exp_dir in sorted([p for p in run_root.iterdir() if p.is_dir() and (p / "conf.yml").is_file()]):
        if not exp_dir.is_dir():
            continue
        model_id = exp_dir.name
        best_model = exp_dir / "best_model.pth"
        model_size_mb = best_model.stat().st_size / (1024 * 1024) if best_model.exists() else None

        full = _get_full_metrics(exp_dir, eval_tag)
        compare_path = exp_dir / "reports" / compare_json_name
        stream = _get_stream_metrics(compare_path, model_id)
        cfg = _infer_cfg(exp_dir / "conf.yml")

        out["models"][model_id] = {
            "model_size_mb": model_size_mb,
            **cfg,
            **full,
            **stream,
        }

    return out


def write_summary_md(payload, out_md):
    lines = []
    lines.append("# Ablation Feb 17 Summary")
    lines.append("")
    lines.append(f"- run_root: `{payload.get('run_root')}`")
    lines.append("")
    lines.append("| Run ID | SI-SDR Full | SI-SDR Imp | 1spk | 2spk | 3spk | Lat(ms) | P95(ms) | RTF | Size(MB) | n_blocks | n_repeats | n_filters | bn | skip | hid |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for run_id, row in sorted(payload.get("models", {}).items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    run_id,
                    _fmt(row.get("full_si_sdr_db")),
                    _fmt(row.get("full_si_sdr_imp_db")),
                    _fmt(row.get("1spk_si_sdr_db")),
                    _fmt(row.get("2spk_si_sdr_db")),
                    _fmt(row.get("3spk_si_sdr_db")),
                    _fmt(row.get("stream_avg_latency_ms")),
                    _fmt(row.get("stream_p95_latency_ms")),
                    _fmt(row.get("stream_rtf")),
                    _fmt(row.get("model_size_mb")),
                    _fmt(row.get("n_blocks"), nd=0),
                    _fmt(row.get("n_repeats"), nd=0),
                    _fmt(row.get("n_filters"), nd=0),
                    _fmt(row.get("bn_chan"), nd=0),
                    _fmt(row.get("skip_chan"), nd=0),
                    _fmt(row.get("hid_chan"), nd=0),
                ]
            )
            + " |"
        )

    Path(out_md).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--eval_tag", default="eval_fullclip_feb17")
    parser.add_argument("--compare_json_name", default="compare_models_nsrc3.json")
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    payload = collect(args.run_root, args.eval_tag, args.compare_json_name)
    Path(args.out_json).write_text(json.dumps(payload, indent=2))
    write_summary_md(payload, args.out_md)

    print(json.dumps({"out_json": args.out_json, "out_md": args.out_md}, indent=2))


if __name__ == "__main__":
    main()
