# scripts/clean_and_split.py
"""
Purpose:
  Lightweight quality check for a JSONL image-caption dataset,
  then optionally split into train / val / test.

What it checks:
  - Each line is valid JSON with keys: "image", "caption"
  - Path resolves correctly
  - Sample a few images to test if they can be opened properly or not
  - Basic stats: total, missing keys, empty captions, missing files, duplicates

Splitting modes (--do_split):
  - Default (row-level): shuffle all rows and split (e.g. custom dataset).
  - --group_by image   : group rows by image file, shuffle images, then split.
    Use this for datasets with multiple captions per image (e.g. Flickr30k)
    so the same image never leaks across splits.
"""

import argparse, json, random
from collections import defaultdict
from pathlib import Path
from PIL import Image
import yaml

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

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def split_by_row(samples, val_ratio, test_ratio, seed):
    """Shuffle all rows and split into train / val / test. (for cuhk custom)"""
    random.seed(seed)
    random.shuffle(samples)
    n = len(samples)
    n_test = max(1, int(n * test_ratio)) if test_ratio > 0 else 0
    n_val = max(1, int(n * val_ratio)) if val_ratio > 0 else 0
    if n_val + n_test >= n:
        raise ValueError(
            f"val + test leaves no training data. "
            f"val={n_val}, test={n_test}, total={n}"
        )
    test_samples = samples[:n_test]
    val_samples = samples[n_test:n_test + n_val]
    train_samples = samples[n_test + n_val:]
    return train_samples, val_samples, test_samples

def split_by_image(samples, val_ratio, test_ratio, seed):
    """Group rows by image, shuffle image groups, then split.(for flickr30k)"""
    grouped = defaultdict(list)
    for s in samples:
        grouped[s["image"]].append(s)

    image_keys = sorted(grouped.keys())
    random.seed(seed)
    random.shuffle(image_keys)

    n = len(image_keys)
    n_test = max(1, int(n * test_ratio)) if test_ratio > 0 else 0
    n_val = max(1, int(n * val_ratio)) if val_ratio > 0 else 0
    if n_val + n_test >= n:
        raise ValueError(
            f"val + test leaves no training images. "
            f"val={n_val}, test={n_test}, total={n}"
        )

    test_keys = image_keys[:n_test]
    val_keys = image_keys[n_test:n_test + n_val]
    train_keys = image_keys[n_test + n_val:]

    flatten = lambda keys: [row for k in keys for row in grouped[k]]
    return flatten(train_keys), flatten(val_keys), flatten(test_keys)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None, help="YAML config file path")
    ap.add_argument("--images_dir", default="data/flickr30k/images")
    ap.add_argument("--captions",   default="data/flickr30k/all.jsonl")
    ap.add_argument("--out_dir",    default=None, help="Output directory (default: same as captions dir)")
    ap.add_argument("--sample_open", type=int, default=None, help="try open N images (0=disable)")
    ap.add_argument("--do_split", action="store_true", help="split into train.jsonl / val.jsonl / test.jsonl")
    ap.add_argument("--group_by", choices=["row", "image"], default="image",
                    help="Split: 'image' (default-flickr30k) or 'row' (cuhk custom)")
    ap.add_argument("--val_ratio", type=float, default=None)
    ap.add_argument("--test_ratio", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None, help="seed for split")
    args = ap.parse_args()

    cfg = {}
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"[ERR] config file doesn't exist: {cfg_path}")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

        paths = cfg.get("paths", {}) or {}
        data = cfg.get("data", {}) or {}
        exp = cfg.get("experiment", {}) or {}

        # prioritize CLI than config file
        if args.images_dir == "data/flickr30k/images":
            args.images_dir = paths.get("images", args.images_dir)
        if args.captions == "data/flickr30k/flickr30k_sample.jsonl":
            args.captions = paths.get("captions", args.captions)

        if args.val_ratio is None and isinstance(data.get("val_ratio", None), (int, float)):
            args.val_ratio = float(data["val_ratio"])
            if not (0.0 < args.val_ratio < 1.0):
                raise ValueError(f"[ERR] data.val_ratio should between 0.0 and 1.0, but receive {args.val_ratio}")
        if args.sample_open is None and isinstance(data.get("sample_open", None), int):
            args.sample_open = int(data.get("sample_open", 50))

        if args.seed is None and isinstance(exp.get("seed", None), int):
            args.seed = int(exp["seed"])

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

        # empty caption
        if cap.strip() == "":
            empty_captions += 1
            print(f"[EMPTY_CAP] line {line_no}: {obj}")
            continue
        
        # duplicates (only flag exact path+caption pairs when grouping by row)
        if args.group_by == "row":
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

        samples.append(obj)

    # optional quick open a few images
    if args.sample_open and args.sample_open > 0:
        to_check = random.sample(samples, min(args.sample_open, len(samples)))
        bad_open = 0
        for s in to_check:
            img_path = resolve_img_path(images_dir, s["image"])   # return a list
            if not quick_open(img_path):
                bad_open += 1
                print(f"[OPEN_ERR] {img_path}")
        print(f"[CHECK] quick-open {len(to_check)} samples, failed={bad_open}")

    # summary of quality of dataset
    print("\n=== Summary ===")
    print(f"total_sample: {total}")
    print(f"parse_errors: {parse_errors}")
    print(f"missing_keys: {missing_keys}")
    print(f"empty_captions: {empty_captions}")
    print(f"missing_files: {missing_files}")
    print(f"dup_paths: {dup_paths}")
    print(f"dup_captions: {dup_caps}")
    print(f"clean_samples: {len(samples)}")

    # split
    if args.do_split and len(samples) > 1:
        val_ratio = args.val_ratio if args.val_ratio else 0.1
        test_ratio = args.test_ratio if args.test_ratio else 0.0
        seed = args.seed if args.seed else 42

        if args.group_by == "image":
            train_samples, val_samples, test_samples = split_by_image(
                samples, val_ratio, test_ratio, seed
            )
        else:
            train_samples, val_samples, test_samples = split_by_row(
                samples, val_ratio, test_ratio, seed
            )

        out_dir = Path(args.out_dir) if args.out_dir else captions.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        write_jsonl(out_dir / "train.jsonl", train_samples)
        write_jsonl(out_dir / "val.jsonl",   val_samples)
        print(f"[SPLIT] train.jsonl={len(train_samples)}, val.jsonl={len(val_samples)}", end="")
        if test_samples:
            write_jsonl(out_dir / "test.jsonl", test_samples)
            print(f", test.jsonl={len(test_samples)}", end="")
        print(f"  (group_by={args.group_by})")

if __name__ == "__main__":
    main()