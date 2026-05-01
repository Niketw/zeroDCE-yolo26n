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
import threading
import time
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from download import prepare_dataset
from model import get_model


training_done = threading.Event()

def get_latest_results_csv(project_dir="checkpoints"):
    search_pattern = os.path.join(project_dir, "*", "results.csv")
    csvs = glob.glob(search_pattern)
    if not csvs:
        return None
    csvs.sort(key=os.path.getmtime, reverse=True)
    return csvs[0]

def discord_watcher_thread(total_epochs=100):
    WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
    if not WEBHOOK_URL or WEBHOOK_URL == "YOUR_WEBHOOK_URL":
        return
        
    try:
        requests.post(WEBHOOK_URL, json={
            "content": f"🚀 **Training Started**",
            "embeds": [{
                "title": "Monitor Active",
                "description": f"Target: {total_epochs} epochs. Watching results.csv...",
                "color": 3066993
            }]
        }, timeout=10)
    except:
        pass
        
    last_processed_epoch = -1
    last_csv_path = None
    
    def check_and_send():
        nonlocal last_processed_epoch, last_csv_path
        try:
            csv_path = get_latest_results_csv()
            if not csv_path: return
            
            if last_csv_path != csv_path:
                last_csv_path = csv_path
                last_processed_epoch = -1
                
            with open(csv_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            if len(lines) <= 1: return
            
            header = [h.strip() for h in lines[0].split(',')]
            last_row = [v.strip() for v in lines[-1].split(',')]
            
            if len(header) != len(last_row): return
            
            row_dict = dict(zip(header, last_row))
            
            current_epoch_str = row_dict.get('epoch', '')
            if not current_epoch_str: return
            
            current_epoch = int(float(current_epoch_str))
            
            if current_epoch > last_processed_epoch:
                box_loss = row_dict.get('train/box_loss', '0.0')
                cls_loss = row_dict.get('train/cls_loss', '0.0')
                dfl_loss = row_dict.get('train/dfl_loss', '0.0')
                
                map50 = row_dict.get('metrics/mAP50(B)', '0.0')
                map50_95 = row_dict.get('metrics/mAP50-95(B)', '0.0')
                
                # Format to 4 decimal places if it's a number
                def fmt(val):
                    try: return f"{float(val):.4f}"
                    except: return val
                
                embed = {
                    "title": f"Epoch {current_epoch}/{total_epochs} Completed",
                    "color": 3447003,
                    "fields": [
                        {
                            "name": "📉 Losses",
                            "value": f"**Box:** {fmt(box_loss)}\n**Class:** {fmt(cls_loss)}\n**DFL:** {fmt(dfl_loss)}",
                            "inline": True
                        },
                        {
                            "name": "📊 Metrics",
                            "value": f"**mAP50:** {fmt(map50)}\n**mAP50-95:** {fmt(map50_95)}",
                            "inline": True
                        }
                    ]
                }
                
                requests.post(WEBHOOK_URL, json={
                    "content": "Epoch update!",
                    "embeds": [embed]
                }, timeout=10)
                
                last_processed_epoch = current_epoch
        except Exception:
            pass

    # Wait for the first epoch/files to initialize
    training_done.wait(10)
    
    while not training_done.is_set():
        check_and_send()
        training_done.wait(30)
        
    # Final check
    check_and_send()
    
    # Send completion
    try:
        requests.post(WEBHOOK_URL, json={
            "content": f"✅ **Training Process Completed!**"
        }, timeout=10)
    except:
        pass


def get_latest_run_weights(project_dir="checkpoints"):
    """Find the most recent checkpoint (.pt) across all runs in the project directory."""
    search_pattern = os.path.join(project_dir, "*", "weights", "*.pt")
    runs = glob.glob(search_pattern)
    
    # Filter out best.pt as we want the latest training state (last.pt or epochX.pt)
    valid_runs = [r for r in runs if not r.endswith("best.pt")]
    
    if not valid_runs:
        return None
        
    # Sort by modification time, newest first
    valid_runs.sort(key=os.path.getmtime, reverse=True)
    
    print("\n  [Debug] Found the following checkpoints (newest first):")
    for r in valid_runs[:5]:
        print(f"    - {r}")
        
    return valid_runs[0]


def main():
    print("\n" + "=" * 70)
    print("  PHASE 1 — Dataset Preparation")
    print("=" * 70)
    yaml_path = prepare_dataset()
    print(f"  Dataset YAML : {yaml_path}")

    last_pt = get_latest_run_weights("checkpoints")

    watcher = threading.Thread(target=discord_watcher_thread, args=(100,), daemon=True)
    watcher.start()

    if last_pt:
        print("\n" + "=" * 70)
        print("  PHASE 2 & 3 — Resuming Training")
        print("=" * 70)
        print(f"  Found interrupted run checkpoint: {last_pt}")
        
        # Patch args.yaml in case the run folder was moved (e.g. from runs/ to checkpoints/)
        # YOLO hardcodes the absolute save_dir in args.yaml. If it doesn't match the current 
        # physical location, resume=True silently fails and starts from epoch 1.
        run_dir = os.path.dirname(os.path.dirname(last_pt))  # parent of weights/
        args_yaml_path = os.path.join(run_dir, "args.yaml")
        
        if os.path.isfile(args_yaml_path):
            import yaml
            try:
                with open(args_yaml_path, "r") as f:
                    args_data = yaml.safe_load(f)
                
                # Update the paths to reflect the current real absolute path
                abs_run_dir = os.path.abspath(run_dir)
                abs_project_dir = os.path.dirname(abs_run_dir)
                
                changed = False
                if args_data.get("save_dir") != abs_run_dir:
                    args_data["save_dir"] = abs_run_dir
                    changed = True
                if args_data.get("project") != abs_project_dir:
                    args_data["project"] = abs_project_dir
                    changed = True
                    
                if changed:
                    with open(args_yaml_path, "w") as f:
                        yaml.dump(args_data, f)
                    print(f"  [Fixed] Patched args.yaml with new directory location.")
            except Exception as e:
                print(f"  ⚠ Could not patch args.yaml: {e}")

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

    training_done.set()
    watcher.join(timeout=5)

    print("\n✓ Training complete!")
    print("  Best weights: ./checkpoints/yolo26n_run/weights/best.pt\n")
    return results


if __name__ == "__main__":
    main()
