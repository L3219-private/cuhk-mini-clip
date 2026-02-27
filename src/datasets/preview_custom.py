"""
preview_custom.py

Purpose:
    Sanity check for custom image-caption dataset.
    This script does not validate data quality, it only verifies that:
        - captions.jsonl (data/custom/images) can be opened and parsed line by line
        - each sample contains expected keys ("image", "caption")
        - image paths correct
        - first several (the value of limit of function preview_custom) images can be opened by PTI without error

Why needed:
    To detect most common errors before wiring datasets into training pipeline
"""

import json
from pathlib import Path
from PIL import Image

def find_img_path(images_dir: Path, img_rel: str) -> Path:
    p = Path(img_rel)
    # start with "data/..."
    if p.as_posix().startswith("data/"):
        return p
    return images_dir / p   # concatenation of paths into "data/..."

def preview_custom(images_dir: str, captions_file: str, limit: int = 3):
    images_dir = Path(images_dir)
    n = 0
    with open(captions_file, "r", encoding="utf-8") as f:    
        for line in f:
            if not line.strip():  
                continue
            sample = json.loads(line)
            img_rel = sample.get("image")   # path for image
            caption = sample.get("caption")   
            if img_rel is None or caption is None:
                print("[Warn] missing keys in line:", line[:120])  # extract
                continue

            img_path = find_img_path(images_dir, img_rel)
            try:
                with Image.open(img_path) as img:
                    print(f"[OK] {img_path}  size={img.size}  mode={img.mode}")
            except Exception as e:
                print(f"[ERR] cannot open {img_path}: {e}")

            print(" caption:", caption)
            n += 1
            if n >= limit:
                break

if __name__ == "__main__":
    preview_custom("data/custom/images", "data/custom/captions.jsonl", limit=3)
