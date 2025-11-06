from pathlib import Path
from datasets import load_dataset
from PIL import Image, ImageDraw
import json, os, tqdm

PROJECT = "doclaynet_mc"
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

    if "category_ids" in row:
        return bboxes, [int(c) for c in row["category_ids"]]  # 0..10

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
        idx = int(cid) + 1
        if not (0 <= idx < len(PALETTE)):
            continue
        d.rectangle([x, y, x + bw, y + bh], fill=PALETTE[idx])
    return m

def write_color_labels():
    labels = ["background"] + CLASSES
    payload = {
        "colors": [list(c) for c in PALETTE],
        "one_hot_encoding": None,
        "labels": labels
    }
    COLOR_LABELS_JSON.parent.mkdir(parents=True, exist_ok=True)
    COLOR_LABELS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[info] color_labels.json written → {COLOR_LABELS_JSON}")

def process_split(split: str):
    print(f"=== Processing {split} (multiclass, color PNG) ===")
    ds = load_dataset(DATASET_ID, split=split)
    (IMAGES / split).mkdir(parents=True, exist_ok=True)
    (LABELS / split).mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(tqdm.tqdm(ds, desc=split)):
        img = row["image"].convert("RGB")
        bboxes, cids = extract_boxes_and_ids(row)

        ip = IMAGES / split / f"{i:06d}.png"
        mp = LABELS / split / f"{i:06d}.png"
        if already_done(ip, mp):
            continue

        safe_save(img, ip)
        mask = draw_multiclass_rgb_mask(img.size, bboxes, cids)
        safe_save(mask, mp)

def main():
    write_color_labels()
    for s in SPLITS:
        process_split(s)
    print("✅ Done. Now run prepare_data.py with data/doclaynet_mc")

if __name__ == "__main__":
    main()
