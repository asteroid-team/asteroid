"""End-to-end demo: load model, separate speakers, classify active/silent channels.

Usage:
    python demo.py --model_path exp/train_convtasnet_5src_xxx/best_model.pth \
                   --wav_path /path/to/mixture.wav \
                   --out_dir demo_output/
"""
import os
import argparse

import numpy as np
import soundfile as sf
import torch

from asteroid import ConvTasNet
from asteroid.dsp.normalization import normalize_estimates


DEFAULT_RMS_DB_THRESHOLD = -40.0


def rms_db(signal):
    """Compute RMS energy in dB."""
    rms = np.sqrt(np.mean(signal ** 2))
    if rms < 1e-10:
        return -100.0
    return 20 * np.log10(rms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--wav_path", type=str, required=True, help="Path to input mixture wav")
    parser.add_argument("--out_dir", type=str, default="demo_output", help="Output directory")
    parser.add_argument(
        "--threshold_db", type=float, default=DEFAULT_RMS_DB_THRESHOLD,
        help="RMS dB threshold for active/silent classification",
    )
    parser.add_argument("--use_gpu", type=int, default=0, help="Use GPU")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}")
    model = ConvTasNet.from_pretrained(args.model_path)
    model.eval()

    if args.use_gpu and torch.cuda.is_available():
        model.cuda()
    device = next(model.parameters()).device

    print(f"Loading mixture from {args.wav_path}")
    mixture, sr = sf.read(args.wav_path, dtype="float32")
    mix_tensor = torch.from_numpy(mixture).unsqueeze(0).to(device)

    with torch.no_grad():
        est_sources = model(mix_tensor)

    est_sources_np = est_sources.squeeze(0).cpu().numpy()
    est_sources_np = normalize_estimates(est_sources_np, mixture)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nResults (threshold: {args.threshold_db} dB):")
    print(f"{'Channel':<10} {'RMS (dB)':<12} {'Status':<10}")
    print("-" * 32)

    for i in range(est_sources_np.shape[0]):
        channel = est_sources_np[i]
        db = rms_db(channel)
        status = "ACTIVE" if db > args.threshold_db else "silent"
        print(f"  {i+1:<8} {db:<12.2f} {status:<10}")

        out_path = os.path.join(args.out_dir, f"speaker_{i+1}.wav")
        sf.write(out_path, channel, sr)

    # Also save the input mixture
    sf.write(os.path.join(args.out_dir, "mixture.wav"), mixture, sr)
    print(f"\nSaved {est_sources_np.shape[0]} speaker files + mixture to {args.out_dir}/")


if __name__ == "__main__":
    main()
