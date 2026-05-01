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
    Initialize the YOLO26n base model and select the optimal device.

    Returns
    -------
    model : ultralytics.YOLO
        The loaded YOLO26n model.
    device : str
        One of "cuda", "mps", or "cpu".
    """
    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        print(f"\n✓ CUDA detected  →  {gpu_name}  ({vram_gb:.1f} GB VRAM)")
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
