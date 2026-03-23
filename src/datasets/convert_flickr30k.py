"""Convert Flickr30k annotations to all.jsonl.

Expected CSV format:
    image_name| comment_number| comment
    1000092795.jpg| 0| Two young guys with shaggy hair ...
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

ALLOWED_EXTS = {".jpg", ".jpeg", ".png"}


def iter_pipe_rows(annotations_path: Path) -> Iterator[Tuple[str, int, str]]:
    """return the generator containing img_name, idx and caption"""
    with annotations_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="|", skipinitialspace=True)
        if not reader.fieldnames:
            raise ValueError(f"Missing header in annotation file: {annotations_path}")

        stripped_fields = {name.strip() for name in reader.fieldnames if name is not None}
        required_fields = {"image_name", "comment_number", "comment"}
        if not required_fields.issubset(stripped_fields):
            raise ValueError(
                f"Unexpected header in {annotations_path}: {reader.fieldnames}\n"
                "Expected: image_name| comment_number| comment"
            )

        for orig_row in reader:
            pcd_row = {
                (key.strip() if key is not None else key): \
                (value.strip() if isinstance(value, str) else value)
                for key, value in orig_row.items()
            }  # {field1: value1}, {field2: value2}

            img_name = pcd_row.get("image_name") or ""
            caption = pcd_row.get("comment") or ""
            caption_idx = pcd_row.get("comment_number") or ""
            if not img_name or not caption:
                continue

            try:
                idx = int(caption_idx) if caption_idx else 0
            except ValueError:
                idx = 0
            yield img_name, idx, caption


def write_jsonl(rows: List[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert(
    annotations_path: str,
    images_dir: str,
    out_dir: str,
) -> None:
    """Convert Flickr30k annotations to Jsonl and create split manifest."""
    annotations_path = Path(annotations_path)
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    available_images = {
        path.name: path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTS
    }
    if not available_images:
        raise ValueError(f"No image files found under: {images_dir}")

    grouped_rows: Dict[str, List[dict]] = defaultdict(list)
    missing = 0

    for img_name, caption_idx, caption in iter_pipe_rows(annotations_path):
        if img_name not in available_images:
            missing += 1
            continue

        grouped_rows[img_name].append(
            {
                "image": img_name,
                "caption": caption,
                "image_id": Path(img_name).stem,
                "caption_idx": caption_idx,
            }
        )

    if missing:
        print(f"[convert_flickr30k] Warning: {missing} image files not found")
    if not grouped_rows:
        raise ValueError("No valid image-caption pairs were found.")

    for rows in grouped_rows.values():
        rows.sort(key=lambda row: (row["caption_idx"], row["caption"]))

    all_rows: List[dict] = []
    for img_name in sorted(grouped_rows):
        all_rows.extend(grouped_rows[img_name])

    write_jsonl(all_rows, out_dir / "all.jsonl")

    manifest = {
        "annotations_path": str(annotations_path),
        "images_dir": str(images_dir),
        "total_images": len(grouped_rows),
        "total_pairs": len(all_rows),
    }
    with (out_dir / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("[convert_flickr30k] Done!")
    print(f"  Images      : {len(grouped_rows):,}")
    print(f"  Total pairs : {len(all_rows):,}")
    print(f"  Output      : {out_dir / 'all.jsonl'}")
    print(f"  Manifest    : {out_dir / 'split_manifest.json'}")

    
def main():
    parser = argparse.ArgumentParser(
        description="Convert Flickr30k CSV annotations to all.jsonl"
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to Flickr30k CSV",
    )
    parser.add_argument(
        "--images_dir",
        required=True,
        help="Directory containing Flickr30k JPEG images",
    )
    parser.add_argument(
        "--out_dir",
        default="data/flickr30k",
        help="Output directory for all.jsonl and split_manifest.json",
    )
    args = parser.parse_args()
    convert(
        annotations_path=args.annotations,
        images_dir=args.images_dir,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
