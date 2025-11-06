from pathlib import Path
from datasets import load_dataset
from PIL import Image, ImageDraw
import json, os, tqdm

# Project layout for prepare_data.py
PROJECT = "doclaynet_binary"
ROOT = Path(f"data/{PROJECT}")
IMAGES = ROOT / "images"
LABELS = ROOT / "labels"
COLOR_LABELS_JSON = ROOT / "color_labels.json"

DATASET_ID = "docling-project/DocLayNet-v1.1"
SPLITS = ["train", "val"]

# Colors for prepare_data.py (RGB)
BG = (0, 0, 0)          # background
FG = (255, 255, 255)    # content

def safe_save(img: Image.Image, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    img.save(tmp, format="PNG")
    os.replace(tmp, path)

def already_done(img_path: Path, mask_path: Path) -> bool:
    return img_path.exists() and img_path.stat().st_size > 0 and mask_path.exists() and mask_path.stat().st_size > 0

def draw_binary_rgb_mask(size, bboxes):
    w, h = size
    m = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(m)
    for (x, y, bw, bh) in bboxes:
        d.rectangle([x, y, x + bw, y + bh], fill=FG)
    return m

def write_color_labels():
    COLOR_LABELS_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "colors": [list(BG), list(FG)],
        "one_hot_encoding": None,
        "labels": ["background", "content"]
    }
    COLOR_LABELS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[info] color_labels.json written → {COLOR_LABELS_JSON}")

def process_split(split: str):
    print(f"=== Processing {split} (binary, color PNG) ===")
    ds = load_dataset(DATASET_ID, split=split)
    (IMAGES / split).mkdir(parents=True, exist_ok=True)
    (LABELS / split).mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(tqdm.tqdm(ds, desc=split)):
        img: Image.Image = row["image"].convert("RGB")
        # v1.1 exposes bboxes
        bboxes = row.get("bboxes")
        if bboxes is None:
            # fallback for legacy layout if ever needed
            anns = row.get("annotations", {}).get("annotations", [])
            bboxes = [a["bbox"] for a in anns]

        ip = IMAGES / split / f"{i:06d}.png"
        mp = LABELS / split / f"{i:06d}.png"

        if already_done(ip, mp):
            continue

        # Save image (we keep original page RGB)
        safe_save(img, ip)

        # Draw binary RGB mask (white = content)
        mask = draw_binary_rgb_mask(img.size, bboxes)
        safe_save(mask, mp)

def main():
    write_color_labels()
    for s in SPLITS:
        process_split(s)
    print("✅ Done. Now run prepare_data.py with a config pointing to data/doclaynet_binary")

if __name__ == "__main__":
    main()
