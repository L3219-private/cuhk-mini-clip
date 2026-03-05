# scripts/clean_and_split.py
"""
Purpose:
  Lightweight check for a JSONL image-caption dataset
  create train/val splits.

What it checks:
  - Each line is valid JSON with keys: "image", "caption"
  - Path resolves correctly
  - Sample a few images to test if they can be opened properly or not
  - Basic stats: total, missing keys, empty captions, missing files, duplicates

"""

import argparse, json, random
from pathlib import Path
from PIL import Image

def resolve_img_path(images_dir: Path, img_rel: str) -> Path:
    """turn all the paths to be absolute or start with data/"""
    p = Path(img_rel.strip())
    posix = p.as_posix()  # '\' -> '/'
    if posix.startswith(("data/")) or p.is_absolute():
        return p
    return images_dir / p

def iter_jsonl(path: Path):
    """
    generator that reads JSONL line by line 
    that yields:
        tuple (line_no, obj):
            - line_no(int): the line number of the JSONL file
            - obj(dict): the parsed JSONL onject if line is proper
                         the error message otherwise
    """
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except Exception as e:
                yield line_no, {"__parse_error__": str(e), "__raw__": line}

def quick_open(img_path: Path) -> bool:
    """verify if the image can be opened properly"""
    try:
        with Image.open(img_path) as img:
            img.verify()  # quick check
        return True
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="data/custom/images")
    ap.add_argument("--captions",   default="data/custom/captions.jsonl")
    ap.add_argument("--sample_open", type=int, default=50, help="try open N images (0=disable)")
    ap.add_argument("--do_split", action="store_true", help="write train.jsonl/val.jsonl")
    ap.add_argument("--val_ratio", type=float, default=0.1)  # the ratio of validation data 
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    captions   = Path(args.captions)  # the path of .jsonl

    total = 0
    missing_keys = 0
    empty_captions = 0
    missing_files = 0
    parse_errors = 0

    seen_paths = set()
    seen_caps  = set()
    dup_paths = 0
    dup_caps  = 0

    samples = []

    for line_no, obj in iter_jsonl(captions):
        total += 1
        # line is improper in format
        if "__parse_error__" in obj:
            parse_errors += 1
            print(f"[PARSE_ERR] line {line_no}: {obj['__parse_error__']}")
            continue

        img_rel = obj.get("image")
        cap     = obj.get("caption")

        # missing keys
        if img_rel is None or cap is None:
            missing_keys += 1
            print(f"[MISS_KEYS] line {line_no}: {obj}")
            continue

        img_path = resolve_img_path(images_dir, img_rel)

        # incorrect path
        if not img_path.exists():
            missing_files += 1
            print(f"[NO_FILE] line {line_no}: {img_path}")
            continue

        # duplicates
        key_path = img_path.as_posix()
        if key_path in seen_paths:
            dup_paths += 1
            print(f"[DUPLICATE_PATH] line {line_no}: {obj}")
            continue
        else:
            seen_paths.add(key_path)

        norm_cap = cap.strip().lower()
        if norm_cap in seen_caps:
            dup_caps += 1
            print(f"[DUPLICATE_CAP] line {line_no}: {obj}")
            continue
        else:
            seen_caps.add(norm_cap)

        # empty caption
        if cap.strip() == "":
            empty_captions += 1
            print(f"[EMPTY_CAP] line {line_no}: {obj}")
            continue

        samples.append({"image": img_rel, "caption": cap})

    # optional quick open a few images
    if args.sample_open > 0:
        to_check = random.sample(samples, min(args.sample_open, len(samples)))
        bad_open = 0
        for s in to_check:
            img_path = resolve_img_path(images_dir, s["image"])  # return a list
            if not quick_open(img_path):
                bad_open += 1
                print(f"[OPEN_ERR] {img_path}")
        print(f"[CHECK] quick-open {len(to_check)} samples, failed={bad_open}")

    # summary of quality of dataset
    print("\n=== Summary ===")
    print(f"total: {total}")
    print(f"parse_errors: {parse_errors}")
    print(f"missing_keys: {missing_keys}")
    print(f"empty_captions: {empty_captions}")
    print(f"missing_files: {missing_files}")
    print(f"dup_paths: {dup_paths}")
    print(f"dup_captions: {dup_caps}")

    # split
    if args.do_split and total > 1:
        random.shuffle(samples)  # shuffle
        n_val = max(1, int(len(samples) * args.val_ratio))
        val_samples = samples[:n_val]
        train_samples = samples[n_val:]

        def write_jsonl(path: Path, rows):
            with path.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        write_jsonl(Path("val.jsonl"),   val_samples)
        write_jsonl(Path("train.jsonl"), train_samples)
        print(f"[SPLIT] wrote train.jsonl={len(train_samples)}, val.jsonl={len(val_samples)}")

if __name__ == "__main__":
    main()