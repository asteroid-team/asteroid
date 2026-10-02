# ConvTasNet 3-Source Ablation (Feb 17)

## Objective
Find the best `n_src=3` ConvTasNet tradeoff for:
- inference speed
- model size
- maintained separation quality

Primary goal: optimize speed and size while keeping quality within an acceptable range (<= 1.0 dB full-context SI-SDR drop vs best baseline).

## Scope and Constraints
- Recipe directory: `egs/librimix/ConvTasNet`
- Task: `sep_clean`
- Data mode: `online_mix` (`min_speakers=1`, `max_speakers=3`, weights `0.20,0.30,0.50`)
- Sample rate: `8 kHz`
- GPUs available: 4 (run one training per GPU in parallel)
- W&B project: `convtasnet-three-src-ablation`

## Validation Modes (required for each run)
1. Full context (full utterance):
- `eval.py` with full-context evaluation output under `eval_fullclip_feb17/`

2. Streaming context:
- `window=1000 ms`, `hop=50 ms`
- `validate.py` visuals + `compare_models_nsrc3.py` quantitative streaming metrics

## Two-Stage Ablation Protocol
### Stage 1 (screening)
- Train all candidates to 40 epochs.
- Run both validation modes.

### Stage 2 (promotion)
- Promote top 3 candidates with:
  - full-context SI-SDR drop <= 1.0 dB vs best stage-1 run
  - then rank by streaming average latency
  - then rank by model size
- Continue promoted candidates to 200 epochs.
- Re-run both validation modes.

## Candidate Set
Defined in:
- `local/ablation_manifest_feb17.json`

Default IDs:
- `A00_ref_1srf`
- `A01_width_m`
- `A02_width_s`
- `A03_depth_m`
- `A04_depth_s`
- `A05_repeat_cut`
- `A06_small_balanced`
- `A07_tiny_probe`

## How to Run
From `ConvTasNet/`:

```bash
nohup env GPU_SLOTS='0;1;2;3' MAX_PARALLEL=4 INFER_CUDA_DEVICES='3' WANDB_PROJECT='convtasnet-three-src-ablation' bash run_ablation_feb17_three_src.sh > logs/nohup_ablation_feb17_three_src.out 2>&1 < /dev/null &
```

Watch logs:

```bash
tail -f logs/nohup_ablation_feb17_three_src.out
```

## Outputs
Primary run root:
- `exp/ablation_feb17_three_src_<timestamp>/`

Per-model outputs:
- `<run_root>/<run_id>/best_model.pth`
- `<run_root>/<run_id>/eval_fullclip_feb17/final_metrics.json`
- `<run_root>/<run_id>/validation_rt1s_hop50ms/`
- `<run_root>/<run_id>/reports/compare_models_nsrc3.json`

Study-level outputs:
- `<run_root>/reports/ablation_feb17_summary.json`
- `<run_root>/reports/ablation_feb17_summary.md`

## Blank Metrics Table

| Run ID | Stage | Epochs | n_blocks | n_repeats | n_filters | bn_chan | skip_chan | hid_chan | RF (ms, est) | Params (M) | Model Size (MB) | Full SI-SDR (dB) | Full SI-SDR Δ vs Ref (dB) | 1spk SI-SDR (dB) | 2spk SI-SDR (dB) | 3spk SI-SDR (dB) | Streaming Avg Latency (ms) @1000/50 | Streaming P95 Latency (ms) | RTF @1000/50 | Mean Pred Active | Active Count MAE | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A00_ref_1srf | S1 | 40 | 8 | 2 | 512 | 128 | 128 | 512 | 1021 | 3.5 | 13.45 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | best train val_loss=15.436 |
| A01_width_m | S1 | 40 | 8 | 2 | 384 | 96 | 96 | 384 | 1021 | 2.0 | 7.67 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | best train val_loss=15.597 |
| A02_width_s | S1 | 40 | 8 | 2 | 256 | 64 | 64 | 256 | 1021 | 0.9 | 3.52 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | best train val_loss=15.190 |
| A03_depth_m | S1 | 40 | 7 | 2 | 384 | 96 | 96 | 384 | 509 | 1.8 | 6.79 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | best train val_loss=15.618 |
| A04_depth_s | S1 | 40 | 6 | 2 | 384 | 96 | 96 | 384 | 253 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | config defined, training not completed in latest run root |
| A05_repeat_cut | S1 | 40 | 8 | 1 | 384 | 96 | 96 | 384 | 509 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | config defined, training not completed in latest run root |
| A06_small_balanced | S1 | 40 | 7 | 1 | 256 | 64 | 64 | 256 | 253 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | config defined, training not completed in latest run root |
| A07_tiny_probe | S1 | 40 | 6 | 1 | 256 | 64 | 64 | 256 | 125 | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | config defined, training not completed in latest run root |
| P1_* | S2 | 200 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| P2_* | S2 | 200 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| P3_* | S2 | 200 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Notes
- `Params (M)` and `RF (ms, est)` can be computed post-run from config/checkpoints and appended.
- Use `reports/ablation_feb17_summary.md` as source of truth when filling the table.
- Current populated values come from `exp/ablation_feb17_three_src_20260217_202539` training logs/checkpoints.
- Full-context (`eval_fullclip_feb17`) and streaming compare JSONs were not produced yet for these rows, so those metrics remain `pending`.
- As of now, only `A00_ref_1srf`, `A01_width_m`, `A02_width_s`, and `A03_depth_m` have `best_model.pth` in `exp/ablation_feb17_three_src_20260217_202539`.

## Latest Snapshot (from available outputs)
- Data source: `exp/ablation_feb17_three_src_20260217_202539/logs/train_*_s1.log` + `best_model.pth` sizes.
- Only stage-1 training metrics are available right now (no full-context or streaming eval metrics were produced in this run root).

### Stage-1 Ranking (best `val_loss`, lower is better)
| Rank | Run ID | Best val_loss | Model Size (MB) | Epoch 39 Throughput (it/s) | Epoch 39 Wall Time |
|---|---|---:|---:|---:|---:|
| 1 | A02_width_s | 15.190 | 3.51 | 16.27 | 0:05:08 |
| 2 | A00_ref_1srf | 15.436 | 13.45 | 15.77 | 0:05:09 |
| 3 | A01_width_m | 15.597 | 7.67 | 15.92 | 0:05:16 |
| 4 | A03_depth_m | 15.618 | 6.79 | 17.57 | 0:04:37 |

Interpretation:
- `A02_width_s` is currently the best quality/size point from completed runs.
- `A03_depth_m` is currently the fastest training-time variant from completed runs.
- Because full-clip and streaming eval outputs are missing, promotion to stage-2 should wait until those evals are generated for apples-to-apples comparison.

## Next Round (Feb 18, Round 3)
- Manifest: `local/ablation_manifest_feb18_round3.json`
- Launcher: `run_ablation_feb18_round3_three_src.sh`
- Focus: around `A02` and `A03` anchors, with smaller/faster sweeps (`C00` to `C09`) to tighten the quality-speed-size frontier.

## Round-3 Status (Updated Feb 19)
- Primary successful stage-1 root: `exp/ablation_feb17_three_src_20260218_220513`
- Earlier attempt root: `exp/ablation_feb17_three_src_20260218_220323` (aborted early)

### What finished
- Stage-1 training completed for all `C00..C09` in `exp/ablation_feb17_three_src_20260218_220513`.
- `best_model.pth` exists for all `C00..C09`.

### What failed
- Full-context validation failed at first model (`C00`) in `exp/ablation_feb17_three_src_20260218_220513`.
- Failure log: `exp/ablation_feb17_three_src_20260218_220513/logs/fullctx_C00_anchor_a02.log`
- Error: `KeyError: 'mixture_path'` in `eval.py`, indicating the provided `--test_dir` points to metadata format incompatible with expected LibriMix CSV columns.
- Because this failed before streaming validation, no `final_metrics.json` or `compare_models_nsrc3.json` was produced for round-3 yet.

## W&B Guide (What To Look At)
- Project: `convtasnet-three-src-ablation`
- Useful group filters:
  - `ablation_feb18_three_src_round3_20260218_220513_s1` (main completed stage-1 run)
  - `ablation_feb18_three_src_round3_20260218_220323_s1` (partial/failed attempt)
- Run names follow: `C##_..._s1`
- Recommended panels to compare:
  - `val_loss` (primary quality proxy used here)
  - epoch runtime / throughput (`it/s`) for training speed
  - system/GPU memory usage to identify concurrency headroom
  - config values (`n_blocks`, `n_repeats`, `n_filters`, `bn_chan`, `skip_chan`, `hid_chan`) in run config

## Round-3 Variables (C00..C09)
These were the ablation knobs:
- `n_blocks` (5 to 8)
- `n_repeats` (1 or 2)
- width tuple: `n_filters`, `bn_chan`, `skip_chan`, `hid_chan`
- fixed for this sweep: `n_src=3`, `segment=1.0`, streaming validation target `1000ms/50ms`

| Run ID | n_blocks | n_repeats | n_filters | bn/skip/hid |
|---|---:|---:|---:|---|
| C00_anchor_a02 | 8 | 2 | 256 | 64/64/256 |
| C01_anchor_a03 | 7 | 2 | 384 | 96/96/384 |
| C02_fast_7x1_256 | 7 | 1 | 256 | 64/64/256 |
| C03_fast_6x1_256 | 6 | 1 | 256 | 64/64/256 |
| C04_fast_6x1_224 | 6 | 1 | 224 | 56/56/224 |
| C05_fast_6x1_192 | 6 | 1 | 192 | 48/48/192 |
| C06_mid_7x2_256 | 7 | 2 | 256 | 64/64/256 |
| C07_mid_8x1_256 | 8 | 1 | 256 | 64/64/256 |
| C08_mid_7x1_224 | 7 | 1 | 224 | 56/56/224 |
| C09_probe_5x1_192 | 5 | 1 | 192 | 48/48/192 |

## Round-3 Stage-1 Snapshot (Training-Only)
Source: `exp/ablation_feb17_three_src_20260218_220513/logs/train_*_s1.log` + `best_model.pth` sizes.

| Rank | Run ID | Best val_loss | Trainable Params | Model Size (MB) | Epoch 39 it/s | Epoch 39 wall |
|---|---|---:|---:|---:|---:|---:|
| 1 | C01_anchor_a03 | 14.697 | 1.8 M | 6.79 | 16.30 | 0:20:24 |
| 2 | C00_anchor_a02 | 15.068 | 900 K | 3.51 | 14.21 | 0:23:05 |
| 3 | C07_mid_8x1_256 | 15.486 | 487 K | 1.90 | 29.13 | 0:12:57 |
| 4 | C02_fast_7x1_256 | 15.671 | 436 K | 1.70 | 28.15 | 0:11:47 |
| 5 | C05_fast_6x1_192 | 15.713 | 220 K | 0.88 | 33.88 | 0:10:06 |
| 6 | C04_fast_6x1_224 | 15.794 | 297 K | 1.17 | 32.90 | 0:10:08 |
| 7 | C08_mid_7x1_224 | 15.836 | 336 K | 1.32 | 28.75 | 0:11:28 |
| 8 | C06_mid_7x2_256 | 15.966 | 797 K | 3.11 | 19.44 | 0:20:37 |
| 9 | C09_probe_5x1_192 | 15.974 | 191 K | 0.76 | 38.33 | 0:08:53 |
| 10 | C03_fast_6x1_256 | 16.163 | 384 K | 1.50 | 31.95 | 0:10:11 |

Interpretation:
- Best quality proxy (stage-1 `val_loss`): `C01_anchor_a03`, then `C00_anchor_a02`.
- Best speed/size extreme: `C09_probe_5x1_192` (fastest and smallest, with quality drop vs anchors).
- Balanced efficiency pick before eval: `C07_mid_8x1_256` (much smaller/faster than anchors with moderate quality drop).
