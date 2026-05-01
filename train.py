"""
train.py — Fine-Tuning Loop
============================
Orchestrates the full training pipeline:
  1. Prepares the BDD100K dataset (download + convert).
  2. Loads the YOLO26n base model.
  3. Runs fine-tuning with optimised hyper-parameters.

Usage:
    python train.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download import prepare_dataset
from model import get_model


def main():
    print("\n" + "=" * 70)
    print("  PHASE 1 — Dataset Preparation")
    print("=" * 70)
    yaml_path = prepare_dataset()
    print(f"  Dataset YAML : {yaml_path}")

    print("\n" + "=" * 70)
    print("  PHASE 2 — Model Initialisation")
    print("=" * 70)
    model, device = get_model()

    print("\n" + "=" * 70)
    print("  PHASE 3 — Fine-Tuning YOLO26n on BDD100K")
    print("=" * 70)
    print(f"  epochs=100, imgsz=736, optimizer=MuSGD, patience=50")
    print(f"  hsv_v=0.1, mixup=0.05, batch=64 (16/GPU × 4), workers=16")
    print(f"  device={device}\n")

    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=736,
        optimizer="MuSGD",
        patience=50,
        hsv_v=0.1,
        mixup=0.05,
        batch=64,
        workers=16,
        nbs=64,
        project="bdd_tuning",
        name="yolo26n_run",
        device=device,
    )

    print("\n✓ Training complete!")
    print("  Best weights: ./bdd_tuning/yolo26n_run/weights/best.pt\n")
    return results


if __name__ == "__main__":
    main()
