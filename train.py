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
import requests
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download import prepare_dataset
from model import get_model


def send_discord_stats(trainer):
    """
    Callback function to send training stats to a Discord webhook.
    Triggered on 'on_train_epoch_end'.
    """
    WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
    USER_ID = os.getenv("DISCORD_USER_ID", "<@YOUR_USER_ID>")
    
    if not WEBHOOK_URL or WEBHOOK_URL == "YOUR_WEBHOOK_URL":
        return
        
    try:
        current_epoch = trainer.epoch + 1
        total_epochs = trainer.epochs
        
        # Extract losses
        if hasattr(trainer, 'tloss') and trainer.tloss is not None:
            box_loss = float(trainer.tloss[0]) if len(trainer.tloss) > 0 else 0.0
            cls_loss = float(trainer.tloss[1]) if len(trainer.tloss) > 1 else 0.0
            dfl_loss = float(trainer.tloss[2]) if len(trainer.tloss) > 2 else 0.0
        else:
            box_loss = cls_loss = dfl_loss = 0.0

        # Extract metrics
        metrics = getattr(trainer, 'metrics', {})
        map50 = metrics.get('metrics/mAP50(B)', 0.0)
        map50_95 = metrics.get('metrics/mAP50-95(B)', 0.0)
        
        embed = {
            "title": f"Epoch {current_epoch}/{total_epochs} Completed",
            "color": 3447003,
            "fields": [
                {
                    "name": "📉 Losses",
                    "value": f"**Box:** {box_loss:.4f}\n**Class:** {cls_loss:.4f}\n**DFL:** {dfl_loss:.4f}",
                    "inline": True
                },
                {
                    "name": "📊 Metrics",
                    "value": f"**mAP50:** {map50:.4f}\n**mAP50-95:** {map50_95:.4f}",
                    "inline": True
                }
            ]
        }
        
        data = {
            "content": f"{USER_ID} - Epoch update!",
            "embeds": [embed]
        }
        
        requests.post(WEBHOOK_URL, json=data, timeout=10)
    except Exception as e:
        print(f"Failed to send Discord webhook: {e}")


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
        model.add_callback("on_train_epoch_end", send_discord_stats)
        results = model.train(resume=True, device=device)
    else:
        print("\n" + "=" * 70)
        print("  PHASE 2 — Model Initialisation")
        print("=" * 70)
        model, device = get_model()
        model.add_callback("on_train_epoch_end", send_discord_stats)

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
