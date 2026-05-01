"""
test.py — Visual Inference
===========================
Loads the best trained weights and runs prediction on test images,
saving visualised bounding-box overlays to disk.

Usage:
    python test.py                          # default: ./dataset/bdd100k/images/test/
    python test.py /path/to/custom/images   # custom folder
"""

import os
import sys
from ultralytics import YOLO

WEIGHTS = os.path.join("bdd_tuning", "yolo26n_run", "weights", "best.pt")
DEFAULT_SOURCE = os.path.join("dataset", "bdd100k", "images", "test")


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE

    if not os.path.isfile(WEIGHTS):
        print(f"✗ Trained weights not found at {WEIGHTS}")
        print("  Run train.py first to generate best.pt.")
        sys.exit(1)

    if not os.path.exists(source):
        print(f"✗ Source path not found: {source}")
        sys.exit(1)

    print(f"\n→ Loading trained model from {WEIGHTS} …")
    model = YOLO(WEIGHTS)

    print(f"→ Running inference on: {source}")
    print(f"  conf threshold : 0.15 (high-recall mode)")
    print(f"  save           : True\n")

    results = model.predict(
        source=source,
        conf=0.15,
        save=True,
        device=0,
        project="bdd_tuning",
        name="yolo26n_predictions",
    )

    print(f"\n✓ Inference complete!")
    print(f"  Processed {len(results)} image(s).")
    print(f"  Visualised results saved to: ./bdd_tuning/yolo26n_predictions/\n")


if __name__ == "__main__":
    main()
