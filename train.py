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

import glob
import os
import sys
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download import prepare_dataset
from model import get_model


def get_latest_run_weights(project_dir="checkpoints"):
    """Find the most recent last.pt across all runs in the project directory."""
    search_pattern = os.path.join(project_dir, "*", "weights", "last.pt")
    runs = glob.glob(search_pattern)
    if not runs:
        return None
    # Sort by modification time, newest first
    runs.sort(key=os.path.getmtime, reverse=True)
    return runs[0]


def main():
    print("\n" + "=" * 70)
    print("  PHASE 1 — Dataset Preparation")
    print("=" * 70)
    yaml_path = prepare_dataset()
    print(f"  Dataset YAML : {yaml_path}")

    last_pt = get_latest_run_weights("checkpoints")

    if last_pt:
        print("\n" + "=" * 70)
        print("  PHASE 2 & 3 — Resuming Training")
        print("=" * 70)
        print(f"  Found interrupted run checkpoint: {last_pt}")
        print("  Resuming training...")
        
        # We just need the device setup from get_model
        _, device = get_model()
        model = YOLO(last_pt)
        results = model.train(resume=True, device=device)
    else:
        print("\n" + "=" * 70)
        print("  PHASE 2 — Model Initialisation")
        print("=" * 70)
        model, device = get_model()

        print("\n" + "=" * 70)
        print("  PHASE 3 — Fine-Tuning YOLO26n on BDD100K")
        print("=" * 70)
        print(f"  epochs=100, imgsz=736, optimizer=MuSGD, patience=15")
        print(f"  hsv_v=0.1, mixup=0.05, batch=64 (16/GPU × 4), workers=16")
        print(f"  device={device}\n")

        results = model.train(
            data=yaml_path,
            epochs=100,
            imgsz=736,
            optimizer="MuSGD",
            patience=15,
            hsv_v=0.1,
            mixup=0.05,
            batch=64,
            workers=16,
            nbs=64,
            project="checkpoints",
            name="yolo26n_run",
            device=device,
            save_period=5,
        )

    print("\n✓ Training complete!")
    print("  Best weights: ./checkpoints/yolo26n_run/weights/best.pt\n")
    return results


if __name__ == "__main__":
    main()
