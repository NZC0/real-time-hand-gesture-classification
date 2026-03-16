"""
Export a trained GestureMLP checkpoint to ONNX format.

Usage:
    python model/export_onnx.py \
        --checkpoint model/best_model.pt \
        --output     web/public/model/gesture_mlp.onnx \
        [--quantize]   # optional INT8 quantization

After export the script runs a quick numerical parity check between the
PyTorch model and the ONNX runtime to catch any conversion issues.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Make sure we can import from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.gesture_mlp import GestureMLP, NUM_FEATURES, NUM_CLASSES, load_model  # noqa: E402


def export_onnx(model: GestureMLP, output_path: Path) -> None:
    """Export model to ONNX with dynamic batch size."""
    model.eval()
    dummy_input = torch.zeros(1, NUM_FEATURES)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={
            "landmarks": {0: "batch_size"},
            "logits":    {0: "batch_size"},
        },
    )
    print(f"ONNX model saved to {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")


def verify_parity(model: GestureMLP, onnx_path: Path, n_tests: int = 20) -> None:
    """Verify that PyTorch and ONNX outputs match within tolerance."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — skipping parity check.")
        return

    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model.eval()

    max_diff = 0.0
    for _ in range(n_tests):
        x_np = np.random.randn(1, NUM_FEATURES).astype(np.float32)
        x_pt = torch.from_numpy(x_np)

        with torch.no_grad():
            pt_out = model(x_pt).numpy()

        ort_out = ort_session.run(["logits"], {"landmarks": x_np})[0]
        diff = float(np.abs(pt_out - ort_out).max())
        max_diff = max(max_diff, diff)

    print(f"Parity check ({n_tests} random inputs): max absolute diff = {max_diff:.2e}")
    if max_diff < 1e-4:
        print("  PASS — outputs match within tolerance.")
    else:
        print("  WARNING — large discrepancy detected! Check ONNX export.")


def quantize_int8(onnx_path: Path) -> Path:
    """Apply static INT8 quantization to reduce model size."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("onnxruntime-tools not available — skipping quantization.")
        return onnx_path

    q_path = onnx_path.with_suffix(".quant.onnx")
    quantize_dynamic(str(onnx_path), str(q_path), weight_type=QuantType.QUInt8)
    orig_kb = onnx_path.stat().st_size / 1024
    quant_kb = q_path.stat().st_size / 1024
    print(f"INT8 quantized model: {q_path}  ({quant_kb:.1f} KB, was {orig_kb:.1f} KB)")
    return q_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GestureMLP to ONNX")
    parser.add_argument(
        "--checkpoint",
        default="model/best_model.pt",
        help="Path to the .pt checkpoint",
    )
    parser.add_argument(
        "--output",
        default="web/public/model/gesture_mlp.onnx",
        help="Output path for the .onnx file",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Also produce an INT8-quantized version",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        sys.exit(f"Checkpoint not found: {checkpoint_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint from {checkpoint_path} …")
    model = load_model(str(checkpoint_path))

    print("Exporting to ONNX …")
    export_onnx(model, output_path)

    print("Running parity check …")
    verify_parity(model, output_path)

    if args.quantize:
        print("Quantizing …")
        q_path = quantize_int8(output_path)
        print("Running parity check on quantized model …")
        verify_parity(model, q_path)


if __name__ == "__main__":
    main()
