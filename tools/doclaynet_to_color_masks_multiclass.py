from pathlib import Path
from datasets import load_dataset
from PIL import Image, ImageDraw
import json, os, tqdm
from typing import List, Optional
import math

PROJECT = "doclaynet_multiclass"
ROOT = Path(f"data/{PROJECT}")
IMAGES = ROOT / "images"
LABELS = ROOT / "labels"
COLOR_LABELS_JSON = ROOT / "color_labels.json"

DATASET_ID = "docling-project/DocLayNet-v1.1"
SPLITS = ["train", "val"]

# DocLayNet classes (order matters; 0 reserved for background)
CLASSES = [
    "Caption",        # 1
    "Footnote",       # 2
    "Formula",        # 3
    "List-item",      # 4
    "Page-footer",    # 5
    "Page-header",    # 6
    "Picture",        # 7
    "Section-header", # 8
    "Table",          # 9
    "Text",           # 10
    "Title"           # 11
]
NAME_TO_CID = {name: i for i, name in enumerate(CLASSES)}  # 0..10

# Palette (0 = background, then 1..len(CLASSES))
PALETTE = [
    (0, 0, 0),        # 0 background
    (220, 20, 60),    # 1 Caption
    (0, 128, 0),      # 2 Footnote
    (30, 144, 255),   # 3 Formula
    (255, 140, 0),    # 4 List-item
    (138, 43, 226),   # 5 Page-footer
    (255, 215, 0),    # 6 Page-header
    (0, 206, 209),    # 7 Picture
    (199, 21, 133),   # 8 Section-header
    (70, 130, 180),   # 9 Table
    (154, 205, 50),   # 10 Text
    (255, 99, 71),    # 11 Title
]

def safe_save(img, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    img.save(tmp, format="PNG")
    os.replace(tmp, path)

def already_done(img_path: Path, mask_path: Path) -> bool:
    return img_path.exists() and img_path.stat().st_size > 0 and mask_path.exists() and mask_path.stat().st_size > 0

def extract_boxes_and_ids(row):
    bboxes = row.get("bboxes", None)
    if bboxes is None:
        anns = row.get("annotations", {}).get("annotations", None)
        if not anns:
            raise KeyError("No 'bboxes' or 'annotations' in row")
        bboxes = [a["bbox"] for a in anns]
        cids = [int(a["category_id"]) for a in anns]  # 0..10
        return bboxes, cids

    if "category_id" in row:
        return bboxes, [int(c) for c in row["category_id"]]  # 0..10

    names = row.get("category_names") or row.get("categories")
    if names is not None:
        cids = [NAME_TO_CID.get(str(n), -1) for n in names]
        bb2, cid2 = [], []
        for bb, cid in zip(bboxes, cids):
            if cid >= 0:
                bb2.append(bb)
                cid2.append(cid)
        return bb2, cid2

    anns = row.get("annotations", {}).get("annotations", None)
    if anns:
        bboxes = [a["bbox"] for a in anns]
        cids = [int(a["category_id"]) for a in anns]
        return bboxes, cids

    raise KeyError("Could not extract categories")

def draw_multiclass_rgb_mask(size, bboxes, cids):
    w, h = size
    m = Image.new("RGB", (w, h), PALETTE[0])  # background
    d = ImageDraw.Draw(m)
    # Note: dataset ids are 0..10; we map to 1..11 for colors
    for (x, y, bw, bh), cid in zip(bboxes, cids):
        if not (0 < cid < len(PALETTE)):
            print(f"category id out of bounds {cid}")
            exit(1)
        d.rectangle([x, y, x + bw, y + bh], fill=PALETTE[cid])
    return m

def calculate_class_weights(num_pixels: List[int]):
    total_pixels = 0
    for count in num_pixels:
        total_pixels += count
    weights = [1.0 / math.log(1.02 + count/total_pixels) for count in num_pixels]
    total_weights = 0
    for w in weights:
        total_weights += w
    return [w / total_weights for w in weights]

def write_color_labels(num_pixels: List[int]):
    labels = ["background"] + CLASSES
    payload = {
        "colors": [list(c) for c in PALETTE],
        "one_hot_encoding": None,
        "labels": labels,
        "weights": calculate_class_weights(num_pixels),
    }
    COLOR_LABELS_JSON.parent.mkdir(parents=True, exist_ok=True)
    COLOR_LABELS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(num_pixels)
    print(f"[info] color_labels.json written → {COLOR_LABELS_JSON}")

def count_pixels(bboxes, cids, num_pixels: List[int]):
    background_pixels = 1024*1024
    for (x, y, bw, bh), cid in zip(bboxes, cids):
        if not (0 < cid < len(PALETTE)):
            print(f"category id out of bounds {cid}")
            exit(1)
        box_pixels = bw * bh
        background_pixels -= box_pixels
        num_pixels[cid] += box_pixels
    num_pixels[0] += background_pixels

def process_split(split: str, num_pixels: List[int]):
    print(f"=== Processing {split} (multiclass, color PNG) ===")
    ds = load_dataset(DATASET_ID, split=split)
    (IMAGES / split).mkdir(parents=True, exist_ok=True)
    (LABELS / split).mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(tqdm.tqdm(ds, desc=split)):
        bboxes, cids = extract_boxes_and_ids(row)
        if split == "train":
            count_pixels(bboxes, cids, num_pixels)

        ip = IMAGES / split / f"{i:06d}.png"
        mp = LABELS / split / f"{i:06d}.png"
        if already_done(ip, mp):
            continue

        img = row["image"].convert("RGB")
        safe_save(img, ip)
        mask = draw_multiclass_rgb_mask(img.size, bboxes, cids)
        safe_save(mask, mp)

def write_split_csv(split: str):
    """Create ROOT/<split>.csv with rows: images/<split>/<file>,labels/<split>/<file>"""
    img_dir = IMAGES / split
    lbl_dir = LABELS / split
    assert img_dir.exists() and lbl_dir.exists(), f"Missing dirs for split '{split}'"

    rows = []
    for ip in sorted(img_dir.glob("*.png")):
        rp_img = ip.relative_to(ROOT).as_posix()
        mp = lbl_dir / ip.name
        if mp.exists() and mp.stat().st_size > 0 and ip.stat().st_size > 0:
            rp_mask = mp.relative_to(ROOT).as_posix()
            rows.append(f"{rp_img},{rp_mask}")

    csv_path = ROOT / f"{split}.csv"
    csv_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    print(f"[info] {split}.csv written → {csv_path} ({len(rows)} pairs)")

def main():
    num_pixels = [0] * len(PALETTE)
    for s in SPLITS:
        process_split(s, num_pixels)
        write_split_csv(s)
    write_color_labels(num_pixels)
    print(f"✅ Done. Now you can point dhSegment-torch to data/{PROJECT}/{{train,val}}.csv")

if __name__ == "__main__":
    main()

