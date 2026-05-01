"""
download.py — The Data Engine
=============================
Handles fetching, extracting, organizing, and converting the BDD100K dataset
from Kaggle into YOLO-format labels for 8 target vehicle/object classes.

Usage:
    python download.py          # standalone
    from download import prepare_dataset   # imported by train.py
"""

import os
import sys
import json
import shutil
import glob
import zipfile
import tarfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import yaml

# Number of parallel workers — tuned for Xeon Gold + 256 GB RAM
NUM_WORKERS = 16


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
YAML_PATH = os.path.join(DATASET_DIR, "bdd_vehicles.yaml")

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

# BDD100K original category → our YOLO class id
# "rider" is merged into "person" (class 0)
CLASS_MAP = {
    "person": 0,
    "rider": 0,        # merged into person
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 4,
    "train": 5,
    "truck": 6,
    "traffic light": 7,
}

# Canonical YOLO names dict (zero-indexed)
CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "bus",
    5: "train",
    6: "truck",
    7: "traffic light",
}


# ---------------------------------------------------------------------------
# Helper: locate files inside Kaggle cache
# ---------------------------------------------------------------------------

def _find_files(root: str, extensions: tuple) -> list:
    """Recursively find files matching given extensions under *root*."""
    matches = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.lower().endswith(extensions):
                matches.append(os.path.join(dirpath, fname))
    return matches


def _find_dirs(root: str, target_name: str) -> list:
    """Recursively find directories whose basename matches *target_name*."""
    matches = []
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d == target_name:
                matches.append(os.path.join(dirpath, d))
    return matches


# ---------------------------------------------------------------------------
# Step 1 – Download via kagglehub
# ---------------------------------------------------------------------------

def _download_dataset() -> str:
    """Download BDD100K from Kaggle directly into the project directory."""
    print("\n" + "=" * 70)
    print("  STEP 1 / 5 — Downloading BDD100K from Kaggle …")
    print("=" * 70)

    # Force kagglehub to cache inside the project directory
    cache_dir = os.path.join(PROJECT_ROOT, ".kaggle_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["KAGGLEHUB_CACHE"] = cache_dir
    print(f"  → Download directory: {cache_dir}")

    try:
        import kagglehub
        path = kagglehub.dataset_download("solesensei/solesensei_bdd100k")
        print(f"  ✓ Dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step 2 – Organise images & labels into local project tree
# ---------------------------------------------------------------------------

def _copy_single_item(args):
    """Copy a single file/directory — designed for use inside a thread pool."""
    s, d = args
    try:
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    except Exception as e:
        return f"  ⚠  Could not copy {s} → {d}: {e}"
    return None


def _copy_tree(src: str, dst: str, desc: str = ""):
    """
    Copy an entire directory tree from *src* into *dst*, skipping
    segmentation-related files/folders.  Uses a ThreadPool with
    NUM_WORKERS threads for I/O-bound parallel copying.
    """
    if not os.path.isdir(src):
        print(f"  ⚠  Source not found, skipping: {src}")
        return

    os.makedirs(dst, exist_ok=True)

    # Build work list (skip segmentation artefacts)
    work = []
    for item in os.listdir(src):
        lower = item.lower()
        if "seg" in lower or "drivable" in lower or "lane" in lower:
            continue
        work.append((os.path.join(src, item), os.path.join(dst, item)))

    print(f"  Copying {desc} ({len(work)} items) with {NUM_WORKERS} threads …")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(_copy_single_item, w): w for w in work}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"  Copying {desc}", unit="file"):
            err = fut.result()
            if err:
                print(err)


def _organise_files(cache_path: str):
    """
    Mirror the relevant images & JSON labels from the Kaggle cache
    into the local ./dataset/ directory.
    """
    print("\n" + "=" * 70)
    print("  STEP 2 / 5 — Organising images & labels …")
    print("=" * 70)

    # --- Images ---------------------------------------------------------- #
    # Kaggle layout may vary; search for directories called "train", "val",
    # "test" that sit somewhere under an "images" parent.
    image_base_candidates = _find_dirs(cache_path, "images")
    image_base = None
    for cand in image_base_candidates:
        # Prefer the one that directly contains train/val/test subdirs
        children = set(os.listdir(cand))
        if {"train", "val"}.issubset(children) or {"100k"}.issubset(children):
            image_base = cand
            break
    if image_base is None and image_base_candidates:
        image_base = image_base_candidates[0]

    if image_base is None:
        print("  ✗ Could not locate an 'images' directory in the cache.")
        sys.exit(1)

    # BDD100K on Kaggle sometimes nests under images/100k/{train,val,test}
    nested_100k = os.path.join(image_base, "100k")
    if os.path.isdir(nested_100k):
        image_base = nested_100k

    for split in ("train", "val", "test"):
        src = os.path.join(image_base, split)
        dst = os.path.join(DATASET_DIR, "bdd100k", "images", split)
        if os.path.isdir(dst) and len(os.listdir(dst)) > 0:
            print(f"  → {split} images already present ({len(os.listdir(dst))} files), skipping copy.")
            continue
        _copy_tree(src, dst, desc=f"{split} images")

    # --- Labels (JSON) --------------------------------------------------- #
    # Stage JSON files in a SEPARATE directory so they don't pollute
    # the YOLO labels path (bdd100k/labels/).  YOLO .txt output goes
    # to bdd100k/labels/ later in Step 4.
    json_staging = os.path.join(DATASET_DIR, "_json_staging")
    os.makedirs(os.path.join(json_staging, "train"), exist_ok=True)
    os.makedirs(os.path.join(json_staging, "val"), exist_ok=True)

    # Create the YOLO labels directories (must exist for Step 4)
    yolo_labels_base = os.path.join(DATASET_DIR, "bdd100k", "labels")
    os.makedirs(os.path.join(yolo_labels_base, "train"), exist_ok=True)
    os.makedirs(os.path.join(yolo_labels_base, "val"), exist_ok=True)

    # Find JSON label files in the Kaggle cache
    json_files = _find_files(cache_path, (".json",))
    label_jsons = [f for f in json_files if "label" in f.lower() and "seg" not in f.lower()]
    print(f"  Found {len(label_jsons)} label JSON file(s) in cache")

    for jf in label_jsons:
        fname = os.path.basename(jf).lower()
        if "train" in fname:
            dst = os.path.join(json_staging, "train", os.path.basename(jf))
        elif "val" in fname:
            dst = os.path.join(json_staging, "val", os.path.basename(jf))
        else:
            dst = os.path.join(json_staging, "train", os.path.basename(jf))

        if not os.path.exists(dst):
            try:
                shutil.copy2(jf, dst)
                print(f"  ✓ Copied label JSON → {dst}")
            except Exception as e:
                print(f"  ⚠  Could not copy {jf}: {e}")

    # Also copy label directory trees if they exist
    label_dirs = _find_dirs(cache_path, "labels")
    for ld in label_dirs:
        lower = ld.lower()
        if "seg" in lower or "drivable" in lower or "lane" in lower:
            continue
        for split in ("train", "val"):
            src_split = os.path.join(ld, split)
            dst_split = os.path.join(json_staging, split)
            if os.path.isdir(src_split):
                _copy_tree(src_split, dst_split, desc=f"{split} labels")


# ---------------------------------------------------------------------------
# Step 3 – Clean up archives
# ---------------------------------------------------------------------------

def _cleanup_archives():
    """Delete any .zip / .tar / .tar.gz files that were copied locally."""
    print("\n" + "=" * 70)
    print("  STEP 3 / 5 — Cleaning up local archives …")
    print("=" * 70)

    archive_exts = ("*.zip", "*.tar", "*.tar.gz", "*.tgz")
    removed = 0
    for ext in archive_exts:
        for archive in glob.glob(os.path.join(DATASET_DIR, "**", ext), recursive=True):
            try:
                # Extract first if needed
                if archive.endswith(".zip"):
                    extract_dir = os.path.splitext(archive)[0]
                    if not os.path.isdir(extract_dir):
                        print(f"  → Extracting {os.path.basename(archive)} …")
                        with zipfile.ZipFile(archive, "r") as zf:
                            zf.extractall(os.path.dirname(archive))
                elif archive.endswith((".tar", ".tar.gz", ".tgz")):
                    extract_dir = archive.replace(".tar.gz", "").replace(".tgz", "").replace(".tar", "")
                    if not os.path.isdir(extract_dir):
                        print(f"  → Extracting {os.path.basename(archive)} …")
                        with tarfile.open(archive, "r:*") as tf:
                            tf.extractall(os.path.dirname(archive))

                os.remove(archive)
                print(f"  ✓ Removed {os.path.basename(archive)}")
                removed += 1
            except Exception as e:
                print(f"  ⚠  Could not handle {archive}: {e}")

    if removed == 0:
        print("  → No archives found. Nothing to clean.")
    else:
        print(f"  ✓ Removed {removed} archive(s).")


# ---------------------------------------------------------------------------
# Step 4 – Convert JSON labels → YOLO .txt (with class filtering)
# ---------------------------------------------------------------------------

def _process_frame(args):
    """
    Convert a single BDD100K frame dict into a YOLO .txt file.
    Designed for use inside a ProcessPoolExecutor.

    Returns (kept_count, skipped_count).
    """
    frame, output_dir = args
    image_name = frame.get("name", "")
    txt_name = os.path.splitext(image_name)[0] + ".txt"
    txt_path = os.path.join(output_dir, txt_name)

    lines = []
    kept = 0
    skipped = 0
    labels = frame.get("labels", [])
    if labels is None:
        labels = []

    for label in labels:
        category = label.get("category", "")
        if category not in CLASS_MAP:
            skipped += 1
            continue

        box2d = label.get("box2d")
        if box2d is None:
            skipped += 1
            continue

        x1 = float(box2d["x1"])
        y1 = float(box2d["y1"])
        x2 = float(box2d["x2"])
        y2 = float(box2d["y2"])

        # Clamp to image bounds
        x1 = max(0.0, min(x1, IMAGE_WIDTH))
        y1 = max(0.0, min(y1, IMAGE_HEIGHT))
        x2 = max(0.0, min(x2, IMAGE_WIDTH))
        y2 = max(0.0, min(y2, IMAGE_HEIGHT))

        # Skip degenerate boxes
        if x2 <= x1 or y2 <= y1:
            skipped += 1
            continue

        # Normalised YOLO coordinates
        x_center = ((x1 + x2) / 2.0) / IMAGE_WIDTH
        y_center = ((y1 + y2) / 2.0) / IMAGE_HEIGHT
        w = (x2 - x1) / IMAGE_WIDTH
        h = (y2 - y1) / IMAGE_HEIGHT

        class_id = CLASS_MAP[category]
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        kept += 1

    # Write .txt even if empty (YOLO expects the file for negative samples)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return kept, skipped


def _convert_json_to_yolo(json_path: str, output_dir: str, split_name: str):
    """
    Parse a BDD100K JSON label file and write per-image YOLO .txt files
    using a ProcessPoolExecutor with NUM_WORKERS processes.

    Each line: class_id x_center y_center width height  (all normalised)
    Only classes present in CLASS_MAP are kept.
    """
    print(f"\n  Converting {split_name} labels → YOLO format …")
    print(f"    Source : {json_path}")
    print(f"    Output : {output_dir}")
    print(f"    Workers: {NUM_WORKERS}")

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ✗ Failed to load JSON: {e}")
        return

    # --- Diagnostic: show the structure of the first frame ---------------
    if isinstance(data, list) and len(data) > 0:
        sample = data[0]
        print(f"    JSON type  : list ({len(data)} frames)")
        print(f"    Frame keys : {list(sample.keys())}")
        sample_labels = sample.get("labels", None)
        if sample_labels and len(sample_labels) > 0:
            print(f"    Label keys : {list(sample_labels[0].keys())}")
            print(f"    Sample cat : {sample_labels[0].get('category', 'N/A')}")
        else:
            print(f"    ⚠  First frame 'labels' field: {sample_labels}")
            print(f"    ⚠  Frame dump: {json.dumps(sample, indent=2)[:500]}")
    elif isinstance(data, dict):
        print(f"    JSON type  : dict (top-level keys: {list(data.keys())})")
        # Some BDD100K versions wrap frames under a key
        for key in ("frames", "labels", "annotations", "images"):
            if key in data and isinstance(data[key], list):
                print(f"    → Unwrapping data['{key}'] ({len(data[key])} items)")
                data = data[key]
                break
    else:
        print(f"    ⚠  Unexpected JSON root type: {type(data)}")
        return

    os.makedirs(output_dir, exist_ok=True)

    total_images = len(data)
    total_boxes = 0
    skipped_boxes = 0

    # Build work items: (frame_dict, output_dir)
    work = [(frame, output_dir) for frame in data]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        for kept, skipped in tqdm(
            pool.map(_process_frame, work, chunksize=256),
            total=total_images,
            desc=f"    Filtering {split_name} JSON labels",
            unit="img",
        ):
            total_boxes += kept
            skipped_boxes += skipped

    print(f"  ✓ {split_name}: {total_boxes} boxes kept across {total_images} images "
          f"({skipped_boxes} boxes discarded).")

    if total_boxes == 0:
        print(f"  ⚠  WARNING: Zero boxes extracted! Check JSON structure above.")


def _convert_all_labels():
    """Read JSON from _json_staging, write YOLO .txt to bdd100k/labels/."""
    print("\n" + "=" * 70)
    print("  STEP 4 / 5 — Converting JSON labels to YOLO .txt …")
    print("=" * 70)

    json_staging = os.path.join(DATASET_DIR, "_json_staging")
    yolo_labels = os.path.join(DATASET_DIR, "bdd100k", "labels")

    for split in ("train", "val"):
        src_dir = os.path.join(json_staging, split)
        out_dir = os.path.join(yolo_labels, split)

        if not os.path.isdir(src_dir):
            print(f"  ⚠  JSON staging directory not found: {src_dir}")
            continue

        # Find JSON(s) in the staging directory
        jsons = [f for f in os.listdir(src_dir) if f.lower().endswith(".json")]
        print(f"  {split}: found {len(jsons)} JSON file(s) in {src_dir}")
        if not jsons:
            print(f"  ⚠  No JSON files found for {split} split")
            continue

        for jf in jsons:
            json_full = os.path.join(src_dir, jf)
            _convert_json_to_yolo(json_full, out_dir, split_name=f"{split}/{jf}")


# ---------------------------------------------------------------------------
# Step 5 – Generate bdd_vehicles.yaml
# ---------------------------------------------------------------------------

def _generate_yaml():
    """Programmatically write ./dataset/bdd_vehicles.yaml."""
    print("\n" + "=" * 70)
    print("  STEP 5 / 5 — Generating YOLO dataset YAML …")
    print("=" * 70)

    yaml_content = {
        "path": os.path.abspath(DATASET_DIR),
        "train": os.path.join("bdd100k", "images", "train"),
        "val": os.path.join("bdd100k", "images", "val"),
        "names": CLASS_NAMES,
    }

    os.makedirs(DATASET_DIR, exist_ok=True)
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    print(f"  ✓ YAML written to {YAML_PATH}")
    print(f"    Classes: {list(CLASS_NAMES.values())}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _labels_are_valid() -> bool:
    """Check that at least some YOLO .txt label files have content."""
    train_labels = os.path.join(DATASET_DIR, "bdd100k", "labels", "train")
    if not os.path.isdir(train_labels):
        return False
    txt_files = [f for f in os.listdir(train_labels) if f.endswith(".txt")]
    if not txt_files:
        return False
    # Spot-check first 20 files — at least 1 must be non-empty
    non_empty = 0
    for tf in txt_files[:20]:
        if os.path.getsize(os.path.join(train_labels, tf)) > 0:
            non_empty += 1
    return non_empty > 0


def prepare_dataset():
    """
    End-to-end dataset preparation.
    Exits early only if the YAML exists AND labels are valid.
    """
    if os.path.isfile(YAML_PATH) and _labels_are_valid():
        print(f"\n✓ Dataset YAML and labels already exist — skipping preparation.")
        return YAML_PATH

    if os.path.isfile(YAML_PATH):
        print(f"\n⚠  YAML exists but labels are missing/empty — re-running conversion …")
        os.remove(YAML_PATH)

    print("\n" + "#" * 70)
    print("#  BDD100K → YOLO Dataset Preparation Pipeline")
    print("#" * 70)

    cache_path = _download_dataset()       # Step 1
    _organise_files(cache_path)            # Step 2
    _cleanup_archives()                    # Step 3
    _convert_all_labels()                  # Step 4
    _generate_yaml()                       # Step 5

    print("\n" + "#" * 70)
    print("#  ✓ Dataset preparation complete!")
    print(f"#  YAML : {YAML_PATH}")
    print("#" * 70 + "\n")

    return YAML_PATH


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prepare_dataset()
