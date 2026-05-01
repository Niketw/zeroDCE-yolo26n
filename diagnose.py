"""
diagnose.py — Check the dataset layout and label state.
Run this on the server: python diagnose.py
"""
import os, json, glob

ROOT = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(ROOT, "dataset")

def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

# 1. JSON staging
section("1. JSON STAGING DIRECTORY")
staging = os.path.join(DS, "_json_staging")
if not os.path.isdir(staging):
    print(f"  NOT FOUND: {staging}")
else:
    for dirpath, dirs, files in os.walk(staging):
        rel = os.path.relpath(dirpath, staging)
        jsons = [f for f in files if f.endswith(".json")]
        others = [f for f in files if not f.endswith(".json")]
        print(f"  {rel}/ → {len(jsons)} JSON, {len(others)} other files")
        for j in jsons:
            fpath = os.path.join(dirpath, j)
            size_mb = os.path.getsize(fpath) / (1024*1024)
            print(f"    JSON: {j} ({size_mb:.1f} MB)")
            # Peek at structure
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    print(f"    Type: list, {len(data)} frames")
                    if data:
                        frame = data[0]
                        print(f"    Frame[0] keys: {list(frame.keys())}")
                        print(f"    Frame[0] name: {frame.get('name','N/A')}")
                        labels = frame.get("labels")
                        if labels and len(labels) > 0:
                            print(f"    Frame[0] labels[0]: {labels[0]}")
                        else:
                            print(f"    Frame[0] labels: {labels}")
                elif isinstance(data, dict):
                    print(f"    Type: dict, keys: {list(data.keys())}")
            except Exception as e:
                print(f"    ERROR reading: {e}")

# 2. Images
section("2. IMAGE DIRECTORIES")
for split in ("train", "val", "test"):
    img_dir = os.path.join(DS, "bdd100k", "images", split)
    if os.path.isdir(img_dir):
        files = os.listdir(img_dir)
        print(f"  {split}: {len(files)} files")
        if files:
            print(f"    Sample: {files[:3]}")
    else:
        print(f"  {split}: NOT FOUND")

# 3. YOLO labels
section("3. YOLO LABEL DIRECTORIES")
for split in ("train", "val"):
    lbl_dir = os.path.join(DS, "bdd100k", "labels", split)
    if not os.path.isdir(lbl_dir):
        print(f"  {split}: NOT FOUND")
        continue
    all_files = os.listdir(lbl_dir)
    txt_files = [f for f in all_files if f.endswith(".txt")]
    json_files = [f for f in all_files if f.endswith(".json")]
    other = len(all_files) - len(txt_files) - len(json_files)
    print(f"  {split}: {len(txt_files)} .txt, {len(json_files)} .json, {other} other")

    # Check how many .txt are non-empty
    non_empty = 0
    for tf in txt_files[:100]:
        if os.path.getsize(os.path.join(lbl_dir, tf)) > 0:
            non_empty += 1
    total_checked = min(100, len(txt_files))
    print(f"    Non-empty (of first {total_checked}): {non_empty}")

    # Show a sample non-empty .txt
    for tf in txt_files[:50]:
        fpath = os.path.join(lbl_dir, tf)
        if os.path.getsize(fpath) > 0:
            with open(fpath) as f:
                content = f.read().strip()
            print(f"    Sample label ({tf}):")
            for line in content.split("\n")[:3]:
                print(f"      {line}")
            break

# 4. Cross-check: do val image names match any JSON frame names?
section("4. VAL IMAGE vs JSON FRAME NAME CROSS-CHECK")
val_img_dir = os.path.join(DS, "bdd100k", "images", "val")
if os.path.isdir(val_img_dir):
    val_images = set(os.listdir(val_img_dir))
    print(f"  Val images: {len(val_images)}")
    if val_images:
        print(f"    Sample val image names: {list(val_images)[:5]}")

    # Check each JSON in staging
    for dirpath, _, files in os.walk(staging):
        for jf in files:
            if not jf.endswith(".json"):
                continue
            fpath = os.path.join(dirpath, jf)
            try:
                with open(fpath) as f:
                    data = json.load(f)
                if isinstance(data, list):
                    frame_names = {fr.get("name", "") for fr in data}
                    overlap = val_images & frame_names
                    print(f"    {jf}: {len(data)} frames, {len(overlap)} match val images")
            except:
                pass
else:
    print("  Val images dir not found")

# 5. Cache files
section("5. CACHE FILES")
for cache in glob.glob(os.path.join(DS, "bdd100k", "labels", "*.cache")):
    print(f"  {os.path.basename(cache)} ({os.path.getsize(cache)} bytes)")

# 6. YAML
section("6. YAML FILE")
yaml_path = os.path.join(DS, "bdd_vehicles.yaml")
if os.path.isfile(yaml_path):
    with open(yaml_path) as f:
        print(f"  {f.read()}")
else:
    print("  NOT FOUND")

print("\n" + "="*60)
print("  DIAGNOSIS COMPLETE")
print("="*60)
