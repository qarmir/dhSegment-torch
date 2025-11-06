from pathlib import Path
from datasets import load_dataset
from PIL import Image, ImageDraw
import tqdm, json, os

ROOT = Path("datasets/doclaynet_multiclass")
SPLITS = ["train", "val"]
DATASET_ID = "docling-project/DocLayNet-v1.1"

# DocLayNet class list (order is important):
DOCLAYNET_CLASSES = [
    "Caption",        # id 0
    "Footnote",       # id 1
    "Formula",        # id 2
    "List-item",      # id 3
    "Page-footer",    # id 4
    "Page-header",    # id 5
    "Picture",        # id 6
    "Section-header", # id 7
    "Table",          # id 8
    "Text",           # id 9
    "Title"           # id 10
]

CID_TO_MASKIDX = {cid: (cid + 1) for cid in range(len(DOCLAYNET_CLASSES))}
MASKIDX_TO_NAME = {0: "Background"}
for cid, midx in CID_TO_MASKIDX.items():
    MASKIDX_TO_NAME[midx] = DOCLAYNET_CLASSES[cid]


def safe_save(img: Image.Image, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    img.save(tmp, format="PNG")
    os.replace(tmp, path)


def draw_multiclass_mask(size, bboxes, category_ids):
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for (x, y, bw, bh), cid in zip(bboxes, category_ids):
        midx = CID_TO_MASKIDX.get(int(cid), 0)
        if midx <= 0:
            continue
        draw.rectangle([x, y, x + bw, y + bh], fill=int(midx))

    return mask


def already_done(img_path: Path, mask_path: Path):
    return (img_path.exists() and img_path.stat().st_size > 0 and mask_path.exists() and mask_path.stat().st_size > 0)


def rebuild_index(root: Path, split: str):
    imgs = sorted((root / split / "images").glob("*.png"))
    masks = root / split / "masks"
    index = []
    for ip in imgs:
        mp = masks / ip.name
        if mp.exists() and mp.stat().st_size > 0:
            index.append({"image": str(ip), "mask": str(mp)})
    (root / f"{split}_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[{split}] index rebuilt: {len(index)} entries")


def write_class_map(root: Path):
    """Save class_map.json once (mask index -> class name)."""
    class_map_path = root / "class_map.json"
    class_map = {
        "background_index": 0,
        "mask_index_to_name": MASKIDX_TO_NAME,
        "doclaynet_source_classes": DOCLAYNET_CLASSES,
        "cid_to_mask_index": CID_TO_MASKIDX
    }
    class_map_path.write_text(json.dumps(class_map, indent=2), encoding="utf-8")
    print(f"class_map saved to {class_map_path}")


def process_split(split: str):
    print(f"=== Processing {split} split (multiclass) ===")
    ds = load_dataset(DATASET_ID, split=split)
    (ROOT / split / "images").mkdir(parents=True, exist_ok=True)
    (ROOT / split / "masks").mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(tqdm.tqdm(ds, desc=split)):
        img: Image.Image = row["image"].convert("RGB")
        bboxes = row["bboxes"]
        cids   = row["category_id"]
        ip = ROOT / split / "images" / f"{i:06d}.png"
        mp = ROOT / split / "masks" / f"{i:06d}.png"
        if already_done(ip, mp):
            continue
        safe_save(img, ip)
        mask = draw_multiclass_mask(img.size, bboxes, cids)
        safe_save(mask, mp)

    rebuild_index(ROOT, split)


def main():
    try:
        for s in SPLITS:
            process_split(s)
        write_class_map(ROOT)
        print("✅ Conversion finished (multiclass).")
    except KeyboardInterrupt:
        print("⏹ Interrupted by user, partial files kept.")
        for s in SPLITS:
            rebuild_index(ROOT, s)
        write_class_map(ROOT)


if __name__ == "__main__":
    main()
