"""
model.py — Model Initialization
================================
Provides get_model() which loads YOLO26n and detects
the best available hardware accelerator (CUDA / MPS / CPU).

Usage:
    from model import get_model
    model, device = get_model()
"""

import torch
from ultralytics import YOLO


def get_model():
    """
    Initialize the YOLO26n base model and select the optimal device(s).

    Returns
    -------
    model : ultralytics.YOLO
        The loaded YOLO26n model.
    device : str or list[int]
        A list of GPU indices (e.g. [0,1,2,3]) for multi-GPU DDP,
        a single string "mps", or "cpu".
    """
    # ------------------------------------------------------------------
    # Device detection (multi-GPU aware)
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"\n✓ CUDA detected  →  {num_gpus} GPU(s) available")
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            vram = torch.cuda.get_device_properties(i).total_mem / (1024 ** 3)
            print(f"    GPU {i}: {name}  ({vram:.1f} GB VRAM)")

        if num_gpus > 1:
            device = list(range(num_gpus))  # e.g. [0, 1, 2, 3]
            print(f"  → Multi-GPU DDP mode: devices {device}")
        else:
            device = 0
            print(f"  → Single-GPU mode: device 0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("\n✓ Apple MPS detected  →  using Metal Performance Shaders")
    else:
        device = "cpu"
        print("\n⚠  No GPU found  →  falling back to CPU (training will be slow)")

    print(f"  Selected device: {device}")

    # ------------------------------------------------------------------
    # Model initialisation
    # ------------------------------------------------------------------
    print("\n→ Loading YOLO26n base model (yolo26n.pt) …")
    model = YOLO("yolo26n.pt")
    print("✓ Model loaded successfully.\n")

    return model, device


# ---------------------------------------------------------------------------
# CLI quick-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model, device = get_model()
    print(f"Model type : {type(model)}")
    print(f"Device     : {device}")
