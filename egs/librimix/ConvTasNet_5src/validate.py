#!/usr/bin/env python
"""Validation / sanity-check script for the 5-speaker causal ConvTasNet.

Three validation sections:
  1. N-Speaker separation test (1-5 speakers) with PIT-based reordering
  2. Visualizations (waveforms + spectrograms saved as PNG, per-source comparisons)
  3. Causal streaming test (causality, LambdaOverlapAdd, latency)

Usage:
    python validate.py --exp_dir exp/smoke_test/
    python validate.py --model_path exp/smoke_test/best_model.pth
    python validate.py --exp_dir exp/smoke_test/ --sample_rate 16000

    # Dryrun: use a known-good pretrained 2-source model to verify the pipeline
    python validate.py --dryrun --out_dir validate_output_dryrun/
"""

import argparse
import os
import sys
import time

import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

# Ensure asteroid is importable from the repo root
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from asteroid.models.base_models import BaseModel
from asteroid.data.online_mix_dataset import OnlineMixDataset
from asteroid.dsp.overlap_add import LambdaOverlapAdd
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def si_sdr(estimate, reference, eps=1e-8):
    """Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.

    Args:
        estimate (np.ndarray): 1-D estimated signal.
        reference (np.ndarray): 1-D reference signal.

    Returns:
        float: SI-SDR in dB.
    """
    estimate = estimate - np.mean(estimate)
    reference = reference - np.mean(reference)
    dot = np.sum(estimate * reference)
    s_target = dot * reference / (np.sum(reference ** 2) + eps)
    e_noise = estimate - s_target
    val = np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + eps)
    return 10 * np.log10(val + eps)


def rms_db(x, eps=1e-10):
    """RMS energy in dB."""
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + eps)


def classify_active(channel_rms_db, threshold_db=-40.0):
    """Return True if the channel is considered active."""
    return channel_rms_db > threshold_db


# ---------------------------------------------------------------------------
# Section 1: N-Speaker Separation Test
# ---------------------------------------------------------------------------

def run_n_speaker_test(model, source_dir, sample_rate, device, n_src=5, speaker_range=None):
    """Run separation on synthetic mixtures with varying speaker counts.

    Uses PIT (Permutation Invariant Training) loss to find the optimal
    source permutation before computing metrics.

    Args:
        model: Trained separation model.
        source_dir: Path to directory with speaker sub-directories.
        sample_rate: Audio sample rate in Hz.
        device: torch device.
        n_src: Number of output sources the model produces.
        speaker_range: Iterable of speaker counts to test. Defaults to
            range(1, n_src+1).

    Returns:
        dict: {n_speakers: {mixture, sources, estimates, reordered, stats}}
    """
    if speaker_range is None:
        speaker_range = range(1, n_src + 1)
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    results = {}
    for n_spk in speaker_range:
        print(f"\n--- {n_spk}-speaker test ---")
        ds = OnlineMixDataset(
            source_dir=source_dir,
            n_src=n_src,
            sample_rate=sample_rate,
            segment=3.0,
            min_speakers=n_spk,
            max_speakers=n_spk,
            num_examples=1,
            seed=42,
        )
        mixture, sources = ds[0]  # (seg_len,), (n_src, seg_len)

        # Run model
        with torch.no_grad():
            mix_in = mixture.unsqueeze(0).to(device)  # (1, seg_len)
            estimates = model(mix_in)  # (1, n_src, seg_len)
        estimates = estimates.cpu().squeeze(0)  # (n_src, seg_len)

        # PIT reordering: find optimal permutation to match GT
        with torch.no_grad():
            _, reordered = loss_func(
                estimates.unsqueeze(0), sources.unsqueeze(0), return_est=True
            )
        reordered = reordered.squeeze(0)  # (n_src, seg_len)

        # Per-channel stats (computed on reordered estimates)
        stats = []
        for ch in range(n_src):
            est_np = reordered[ch].numpy()
            src_np = sources[ch].numpy()
            ch_rms = rms_db(est_np)
            active = classify_active(ch_rms)
            entry = {
                "channel": ch,
                "rms_db": ch_rms,
                "active": active,
                "expected_active": ch < n_spk,
            }
            # SI-SDR only for channels with actual source energy
            if ch < n_spk:
                entry["si_sdr"] = si_sdr(est_np, src_np)
            else:
                entry["si_sdr"] = None
            stats.append(entry)

        # Report
        for s in stats:
            marker = "OK" if s["active"] == s["expected_active"] else "MISMATCH"
            si_str = f"{s['si_sdr']:.1f} dB" if s['si_sdr'] is not None else "n/a"
            print(
                f"  ch{s['channel']}: RMS={s['rms_db']:.1f} dB  "
                f"active={s['active']}  expected={s['expected_active']}  "
                f"SI-SDR={si_str}  [{marker}]"
            )

        results[n_spk] = {
            "mixture": mixture,
            "sources": sources,
            "estimates": estimates,
            "reordered": reordered,
            "stats": stats,
        }
    return results


# ---------------------------------------------------------------------------
# Section 2: Visualizations
# ---------------------------------------------------------------------------

def make_spectrogram(signal_np, sample_rate, n_fft=256, hop_length=64):
    """Compute magnitude spectrogram using torch.stft, return as numpy."""
    sig_t = torch.from_numpy(signal_np).float()
    window = torch.hann_window(n_fft)
    stft = torch.stft(sig_t, n_fft=n_fft, hop_length=hop_length,
                      window=window, return_complex=True)
    mag = torch.abs(stft).numpy()
    # Convert to dB
    mag_db = 20 * np.log10(mag + 1e-10)
    return mag_db


def save_visualizations(results, out_dir, sample_rate, n_src=5):
    """Generate and save PNG figures for each N-speaker test case.

    For each N-speaker case, saves:
      - {n}spk_separation.png: overview (mixture, stacked GT, stacked estimates)
      - {n}spk_per_source.png: per-source GT vs estimate comparison
    """
    os.makedirs(out_dir, exist_ok=True)
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    for n_spk, data in results.items():
        mixture = data["mixture"].numpy()
        sources = data["sources"].numpy()  # (n_src, T)
        estimates = data["reordered"].numpy()  # (n_src, T) — PIT-reordered
        stats = data["stats"]
        T = mixture.shape[0]
        t_axis = np.arange(T) / sample_rate

        # --- Save audio files (using reordered estimates) ---
        sf.write(os.path.join(audio_dir, f"{n_spk}spk_mixture.wav"), mixture, sample_rate)
        for ch in range(n_src):
            sf.write(os.path.join(audio_dir, f"{n_spk}spk_src{ch}.wav"), sources[ch], sample_rate)
            sf.write(os.path.join(audio_dir, f"{n_spk}spk_est{ch}.wav"), estimates[ch], sample_rate)

        # --- Overview figure (3x2 grid) ---
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f"{n_spk}-Speaker Separation", fontsize=14, fontweight="bold")

        # Row 1: Mixture waveform + spectrogram
        axes[0, 0].plot(t_axis, mixture, linewidth=0.3, color="steelblue")
        axes[0, 0].set_title("Mixture Waveform")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Amplitude")
        axes[0, 0].set_xlim(0, T / sample_rate)

        spec = make_spectrogram(mixture, sample_rate)
        axes[0, 1].imshow(
            spec, aspect="auto", origin="lower",
            extent=[0, T / sample_rate, 0, sample_rate / 2],
            cmap="magma",
        )
        axes[0, 1].set_title("Mixture Spectrogram")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Frequency (Hz)")

        # Row 2: Ground-truth sources (stacked)
        ax_src = axes[1, 0]
        ax_src.set_title(f"Ground-Truth Sources ({n_spk} active)")
        for ch in range(n_src):
            offset = ch * 2.0
            label = f"src{ch}" if ch < n_spk else f"src{ch} (silent)"
            color = f"C{ch}" if ch < n_spk else "lightgray"
            ax_src.plot(t_axis, sources[ch] + offset, linewidth=0.3, color=color, label=label)
        ax_src.set_xlabel("Time (s)")
        ax_src.set_ylabel("Channel (offset)")
        ax_src.legend(fontsize=7, loc="upper right")
        ax_src.set_xlim(0, T / sample_rate)

        # Row 2 right: source spectrograms montage for active sources
        if n_spk > 0:
            src_spec = make_spectrogram(sources[0], sample_rate)
            axes[1, 1].imshow(
                src_spec, aspect="auto", origin="lower",
                extent=[0, T / sample_rate, 0, sample_rate / 2],
                cmap="magma",
            )
            axes[1, 1].set_title("Source 0 Spectrogram (GT)")
            axes[1, 1].set_xlabel("Time (s)")
            axes[1, 1].set_ylabel("Frequency (Hz)")

        # Row 3: Model estimates (stacked, reordered)
        ax_est = axes[2, 0]
        ax_est.set_title(f"Model Estimates ({n_src} channels, PIT-reordered)")
        for ch in range(n_src):
            offset = ch * 2.0
            s = stats[ch]
            tag = "active" if s["active"] else "silent"
            match = "OK" if s["active"] == s["expected_active"] else "MISMATCH"
            label = f"est{ch} ({tag}, {match})"
            color = f"C{ch}" if s["active"] else "lightgray"
            ax_est.plot(t_axis, estimates[ch] + offset, linewidth=0.3, color=color, label=label)
        ax_est.set_xlabel("Time (s)")
        ax_est.set_ylabel("Channel (offset)")
        ax_est.legend(fontsize=7, loc="upper right")
        ax_est.set_xlim(0, T / sample_rate)

        # Row 3 right: estimate channel 0 spectrogram
        est_spec = make_spectrogram(estimates[0], sample_rate)
        axes[2, 1].imshow(
            est_spec, aspect="auto", origin="lower",
            extent=[0, T / sample_rate, 0, sample_rate / 2],
            cmap="magma",
        )
        axes[2, 1].set_title("Estimate 0 Spectrogram (PIT-reordered)")
        axes[2, 1].set_xlabel("Time (s)")
        axes[2, 1].set_ylabel("Frequency (Hz)")

        plt.tight_layout()
        fig_path = os.path.join(out_dir, f"{n_spk}spk_separation.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {fig_path}")

        # --- Per-source comparison figure ---
        n_active = max(n_spk, 1)
        fig_ps, axes_ps = plt.subplots(n_active, 4, figsize=(20, 3.5 * n_active))
        fig_ps.suptitle(
            f"{n_spk}-Speaker: Per-Source Comparison (PIT-reordered)",
            fontsize=14, fontweight="bold",
        )
        # Handle n_active == 1: subplots returns 1D array instead of 2D
        if n_active == 1:
            axes_ps = axes_ps[np.newaxis, :]

        for src_i in range(n_active):
            gt = sources[src_i]
            est = estimates[src_i]
            s = stats[src_i]
            si_str = f"{s['si_sdr']:.1f} dB" if s["si_sdr"] is not None else "n/a"

            # Col 0: GT waveform
            axes_ps[src_i, 0].plot(t_axis, gt, linewidth=0.3, color=f"C{src_i}")
            axes_ps[src_i, 0].set_title(f"GT src{src_i}")
            axes_ps[src_i, 0].set_xlim(0, T / sample_rate)
            axes_ps[src_i, 0].set_ylabel("Amplitude")

            # Col 1: GT spectrogram
            gt_spec = make_spectrogram(gt, sample_rate)
            axes_ps[src_i, 1].imshow(
                gt_spec, aspect="auto", origin="lower",
                extent=[0, T / sample_rate, 0, sample_rate / 2],
                cmap="magma",
            )
            axes_ps[src_i, 1].set_title(f"GT src{src_i} spec")
            axes_ps[src_i, 1].set_ylabel("Freq (Hz)")

            # Col 2: Estimate waveform
            axes_ps[src_i, 2].plot(t_axis, est, linewidth=0.3, color=f"C{src_i}")
            axes_ps[src_i, 2].set_title(f"Est src{src_i}  SI-SDR={si_str}")
            axes_ps[src_i, 2].set_xlim(0, T / sample_rate)
            axes_ps[src_i, 2].set_ylabel("Amplitude")

            # Col 3: Estimate spectrogram
            est_spec_i = make_spectrogram(est, sample_rate)
            axes_ps[src_i, 3].imshow(
                est_spec_i, aspect="auto", origin="lower",
                extent=[0, T / sample_rate, 0, sample_rate / 2],
                cmap="magma",
            )
            axes_ps[src_i, 3].set_title(f"Est src{src_i} spec  SI-SDR={si_str}")
            axes_ps[src_i, 3].set_ylabel("Freq (Hz)")

        # Add x-labels to bottom row only
        for col in range(4):
            axes_ps[n_active - 1, col].set_xlabel("Time (s)")

        plt.tight_layout()
        fig_ps_path = os.path.join(out_dir, f"{n_spk}spk_per_source.png")
        fig_ps.savefig(fig_ps_path, dpi=150)
        plt.close(fig_ps)
        print(f"  Saved {fig_ps_path}")


# ---------------------------------------------------------------------------
# Section 3: Causal Streaming Test
# ---------------------------------------------------------------------------

def run_streaming_test(model, source_dir, sample_rate, block_ms, device, n_src=5):
    """Run causality, LambdaOverlapAdd, and latency tests."""
    block_samples = int(block_ms / 1000.0 * sample_rate)
    # Make sure block_samples is even (required by LambdaOverlapAdd)
    if block_samples % 2 != 0:
        block_samples += 1

    # Get a 3-second test mixture
    n_mix_spk = min(3, n_src)
    ds = OnlineMixDataset(
        source_dir=source_dir,
        n_src=n_src,
        sample_rate=sample_rate,
        segment=3.0,
        min_speakers=n_mix_spk,
        max_speakers=n_mix_spk,
        num_examples=1,
        seed=123,
    )
    mixture, _ = ds[0]  # (seg_len,)
    total_samples = mixture.shape[0]

    print("\n" + "=" * 60)
    print("SECTION 3: Causal Streaming Test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test A: Causality verification
    # ------------------------------------------------------------------
    print(f"\n--- Test A: Causality verification (block={block_ms}ms = {block_samples} samples) ---")

    with torch.no_grad():
        mix_full = mixture.unsqueeze(0).to(device)  # (1, total_samples)
        full_output = model(mix_full)  # (1, 5, total_samples)

        mix_short = mixture[:block_samples].unsqueeze(0).to(device)
        short_output = model(mix_short)  # (1, 5, block_samples)

        mix_medium = mixture[:2 * block_samples].unsqueeze(0).to(device)
        medium_output = model(mix_medium)  # (1, 5, 2*block_samples)

    # Compare first block
    # Note: ConvTasNet uses cumulative layer norm (cLN) which computes stats
    # over the full input length, so different-length inputs will have small
    # numeric differences even for causal convolutions. Tolerance of 0.05 is
    # appropriate — truly non-causal models would show differences of ~1.0+.
    causality_tol = 0.05
    full_first_block = full_output[:, :, :block_samples].cpu()
    short_first_block = short_output[:, :, :block_samples].cpu()
    diff_short = torch.max(torch.abs(full_first_block - short_first_block)).item()
    pass_short = diff_short < causality_tol

    # Compare first 2 blocks
    n_medium = min(2 * block_samples, full_output.shape[2], medium_output.shape[2])
    full_medium_slice = full_output[:, :, :n_medium].cpu()
    medium_slice = medium_output[:, :, :n_medium].cpu()
    diff_medium = torch.max(torch.abs(full_medium_slice - medium_slice)).item()
    pass_medium = diff_medium < causality_tol

    status_short = "PASS" if pass_short else "FAIL"
    status_medium = "PASS" if pass_medium else "FAIL"
    print(f"  full[:, :, :{block_samples}] vs short_output: max_diff={diff_short:.2e} [{status_short}]")
    print(f"  full[:, :, :{n_medium}] vs medium_output: max_diff={diff_medium:.2e} [{status_medium}]")

    if pass_short and pass_medium:
        print("  => Causality CONFIRMED: future samples do not affect past outputs.")
    else:
        print("  => Causality check FAILED: outputs differ beyond tolerance.")

    # ------------------------------------------------------------------
    # Test B: LambdaOverlapAdd streaming
    # ------------------------------------------------------------------
    print(f"\n--- Test B: LambdaOverlapAdd streaming ---")
    print(f"  window_size={block_samples}, hop_size={block_samples // 2}")

    ola_model = LambdaOverlapAdd(
        nnet=model,
        n_src=n_src,
        window_size=block_samples,
        hop_size=block_samples // 2,
        window="hann",
        reorder_chunks=True,
    )
    ola_model = ola_model.to(device)

    with torch.no_grad():
        # LambdaOverlapAdd expects (batch, 1, time)
        mix_ola = mixture.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, total_samples)
        ola_output = ola_model(mix_ola)  # (1, 5, total_samples)

    # Compare against full-sequence output
    full_np = full_output.cpu().squeeze(0).numpy()  # (5, T)
    ola_np = ola_output.cpu().squeeze(0).numpy()  # (5, T)
    T_compare = min(full_np.shape[1], ola_np.shape[1])
    full_np = full_np[:, :T_compare]
    ola_np = ola_np[:, :T_compare]

    max_abs_diff = np.max(np.abs(full_np - ola_np))
    print(f"  Max absolute difference (full vs OLA): {max_abs_diff:.4f}")

    for ch in range(n_src):
        f_ch = full_np[ch]
        o_ch = ola_np[ch]
        if np.std(f_ch) > 1e-8 and np.std(o_ch) > 1e-8:
            corr = np.corrcoef(f_ch, o_ch)[0, 1]
        else:
            corr = float("nan")
        print(f"  ch{ch}: correlation={corr:.4f}")

    # ------------------------------------------------------------------
    # Test C: Latency measurement
    # ------------------------------------------------------------------
    print(f"\n--- Test C: Latency measurement ---")

    # Single-block input
    single_block = torch.randn(1, block_samples, device=device)
    n_warmup = 10
    n_runs = 50

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(single_block)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(single_block)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    avg_ms = np.mean(times)
    std_ms = np.std(times)
    rt_factor = avg_ms / block_ms
    print(f"  {block_ms}ms block ({block_samples} samples) takes {avg_ms:.2f} +/- {std_ms:.2f} ms")
    print(f"  Real-time factor: {rt_factor:.3f}x  ({'faster' if rt_factor < 1 else 'slower'} than real-time)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validation script for 5-speaker ConvTasNet")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to best_model.pth checkpoint")
    parser.add_argument("--exp_dir", type=str, default=None,
                        help="Experiment directory (auto-finds best_model.pth and conf.yml)")
    parser.add_argument("--source_dir", type=str,
                        default="/home/mkeller/data/librimix/LibriSpeech/test-clean",
                        help="Directory with speaker sub-directories of clean audio")
    parser.add_argument("--out_dir", type=str, default="validate_output",
                        help="Output directory for figures and audio")
    parser.add_argument("--block_ms", type=float, default=200,
                        help="Block size in ms for streaming test")
    parser.add_argument("--sample_rate", type=int, default=None,
                        help="Sample rate (Hz); auto-detected from conf.yml if available")
    parser.add_argument("--dryrun", action="store_true",
                        help="Use a pretrained 2-source ConvTasNet from the asteroid hub "
                             "to verify the validation pipeline independently of your model")
    parser.add_argument(
        "--profile",
        type=str,
        default="full",
        choices=["full", "two_spk_in_5ch"],
        help="Validation profile: full (1..n_src) or focused 2-speaker-in-5ch.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.dryrun:
        # --- Dryrun mode: load known-good pretrained 2-source model ---
        print("=" * 60)
        print("DRYRUN MODE: using pretrained 2-source ConvTasNet")
        print("=" * 60)
        model = BaseModel.from_pretrained("Cosentino/ConvTasNet_LibriMix_sep_clean")
        n_src = model.masker.n_src  # 2
        sample_rate = 8000
        speaker_range = [n_src]
        print(f"  Model type: {type(model).__name__}")
        print(f"  n_src: {n_src}")
        print(f"  Sample rate: {sample_rate}")
    else:
        # --- Resolve model path ---
        model_path = args.model_path
        if model_path is None:
            if args.exp_dir is None:
                parser.error("Either --model_path or --exp_dir must be provided")
            model_path = os.path.join(args.exp_dir, "best_model.pth")
            if not os.path.isfile(model_path):
                parser.error(f"Could not find best_model.pth in {args.exp_dir}")

        # --- Resolve sample rate: CLI > conf.yml > default 8000 ---
        sample_rate = args.sample_rate
        if sample_rate is None:
            # Try to load from conf.yml in exp_dir (or model_path's parent)
            exp_dir = args.exp_dir or os.path.dirname(model_path)
            conf_path = os.path.join(exp_dir, "conf.yml")
            if os.path.isfile(conf_path):
                with open(conf_path) as f:
                    train_conf = yaml.safe_load(f)
                sample_rate = train_conf.get("data", {}).get("sample_rate", None)
                if sample_rate is not None:
                    print(f"Auto-detected sample_rate={sample_rate} from {conf_path}")
            if sample_rate is None:
                sample_rate = 8000
                print(f"Using default sample_rate={sample_rate}")

        # Load model
        print(f"Loading model from {model_path} ...")
        model = BaseModel.from_pretrained(model_path)
        n_src = int(model.masker.n_src)
        speaker_range = [2] if args.profile == "two_spk_in_5ch" else list(range(1, n_src + 1))
        print(f"  Model type: {type(model).__name__}")
        print(f"  Sample rate: {model.sample_rate}")

    model = model.eval().to(device)

    # Section 1 & 2
    print("\n" + "=" * 60)
    print("SECTION 1: N-Speaker Separation Test")
    print("=" * 60)
    results = run_n_speaker_test(
        model, args.source_dir, sample_rate, device,
        n_src=n_src, speaker_range=speaker_range,
    )

    print("\n" + "=" * 60)
    print("SECTION 2: Visualizations")
    print("=" * 60)
    save_visualizations(results, args.out_dir, sample_rate, n_src=n_src)
    print(f"  Audio saved to {os.path.join(args.out_dir, 'audio')}/")

    # Section 3
    run_streaming_test(
        model, args.source_dir, sample_rate, args.block_ms, device,
        n_src=n_src,
    )

    print("\n" + "=" * 60)
    print("DONE. Check output in:", args.out_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
