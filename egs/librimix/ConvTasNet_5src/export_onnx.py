"""Export trained ConvTasNet 5-source model to ONNX format.

Exports with dynamic batch and time axes for flexible inference.
Validates the exported model with onnx.checker.

Next steps for Hailo-8 deployment:
  1. Install HailoRT and Hailo Dataflow Compiler (DFC)
  2. Parse ONNX: hailo parser onnx convtasnet_5src.onnx
  3. Optimize: hailo optimize --hw-arch hailo8
  4. Compile: hailo compiler --hw-arch hailo8
"""
import os
import argparse

import torch
import onnx

from asteroid import ConvTasNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default="exp/tmp", help="Experiment directory")
    parser.add_argument("--out_name", default="convtasnet_5src.onnx", help="Output ONNX filename")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    model_path = os.path.join(args.exp_dir, "best_model.pth")
    print(f"Loading model from {model_path}")
    model = ConvTasNet.from_pretrained(model_path)
    model.eval()
    model.cpu()

    # Create dummy input: batch=1, time=24000 (3 seconds at 8kHz)
    dummy_input = torch.randn(1, 24000)

    out_path = os.path.join(args.exp_dir, args.out_name)
    print(f"Exporting to {out_path} (opset {args.opset})")

    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        opset_version=args.opset,
        input_names=["mixture"],
        output_names=["sources"],
        dynamic_axes={
            "mixture": {0: "batch", 1: "time"},
            "sources": {0: "batch", 2: "time"},
        },
    )

    # Validate
    onnx_model = onnx.load(out_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model saved and validated: {out_path}")
    print(f"  Input: {[inp.name for inp in onnx_model.graph.input]}")
    print(f"  Output: {[out.name for out in onnx_model.graph.output]}")


if __name__ == "__main__":
    main()
