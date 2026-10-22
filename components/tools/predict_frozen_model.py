"""One-image ML inference CLI for teammate integration, not an app or evaluation."""
import argparse
import json
from pathlib import Path

from component_inspector.vision.ksdd2_ensemble import DEFAULT_BUNDLE, Ensemble


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    if args.image.stat().st_size > 10 * 1024**2:
        raise ValueError("Image exceeds 10 MiB limit")
    prediction = Ensemble(args.bundle).predict_bytes(args.image.read_bytes())
    # Attention is available through the Python adapter; keep terminal output small.
    prediction.pop("attention_png", None)
    print(json.dumps(prediction, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
