"""
validate.py — Evaluation
=========================
Loads the best trained weights and runs YOLO .val() to compute
mAP, Precision, Recall, and Inference Latency.

Usage:
    python validate.py
"""

import os
import sys
from ultralytics import YOLO

WEIGHTS = os.path.join("bdd_tuning", "yolo26n_run", "weights", "best.pt")
YAML_PATH = os.path.join("dataset", "bdd_vehicles.yaml")


def main():
    if not os.path.isfile(WEIGHTS):
        print(f"✗ Trained weights not found at {WEIGHTS}")
        print("  Run train.py first to generate best.pt.")
        sys.exit(1)

    print(f"\n→ Loading trained model from {WEIGHTS} …")
    model = YOLO(WEIGHTS)

    print(f"→ Running validation on {YAML_PATH} …")
    print(f"  workers=16, device=0\n")
    results = model.val(data=YAML_PATH, workers=16, device=0)

    # Extract key metrics
    print("\n" + "=" * 60)
    print("  Validation Results — YOLO26n × BDD100K")
    print("=" * 60)

    try:
        map50 = results.box.map50
        map50_95 = results.box.map
        precision = results.box.mp
        recall = results.box.mr
        print(f"  mAP@0.5       : {map50:.4f}")
        print(f"  mAP@0.5:0.95  : {map50_95:.4f}")
        print(f"  Precision      : {precision:.4f}")
        print(f"  Recall         : {recall:.4f}")
    except AttributeError as e:
        print(f"  ⚠  Could not extract box metrics: {e}")
        print(f"  Raw results: {results}")

    try:
        speed = results.speed
        preprocess = speed.get("preprocess", 0)
        inference = speed.get("inference", 0)
        postprocess = speed.get("postprocess", 0)
        print(f"\n  Latency (ms/image):")
        print(f"    Pre-process  : {preprocess:.1f} ms")
        print(f"    Inference    : {inference:.1f} ms")
        print(f"    Post-process : {postprocess:.1f} ms")
        print(f"    Total        : {preprocess + inference + postprocess:.1f} ms")
    except Exception as e:
        print(f"  ⚠  Could not extract speed metrics: {e}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
