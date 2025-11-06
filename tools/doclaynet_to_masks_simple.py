from pathlib import Path
from datasets import load_dataset
from PIL import Image, ImageDraw
import tqdm, json, os

ROOT = Path("datasets/doclaynet")
SPLITS = ["train", "val"]
DATASET_ID = "docling-project/DocLayNet-v1.1"


def safe_save(img: Image.Image, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    img.save(tmp, format="PNG")
    os.replace(tmp, path)


def draw_mask(size, bboxes):
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for x, y, bw, bh in bboxes:
        draw.rectangle([x, y, x + bw, y + bh], fill=255)
    return mask


def already_done(img_path: Path, mask_path: Path):
    return img_path.exists() and mask_path.exists() and img_path.stat().st_size > 0 and mask_path.stat().st_size > 0


def rebuild_index(root: Path, split: str):
    imgs = sorted((root / split / "images").glob("*.png"))
    masks = root / split / "masks"
    index = []
    for ip in imgs:
        mp = masks / ip.name
        if mp.exists():
            index.append({"image": str(ip), "mask": str(mp)})
    (root / f"{split}_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[{split}] index rebuilt: {len(index)} entries")


def process_split(split: str):
    print(f"=== Processing {split} split ===")
    ds = load_dataset(DATASET_ID, split=split)
    (ROOT / split / "images").mkdir(parents=True, exist_ok=True)
    (ROOT / split / "masks").mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(tqdm.tqdm(ds, desc=split)):
        img: Image.Image = row["image"].convert("RGB")
        bboxes = row["bboxes"]
        ip = ROOT / split / "images" / f"{i:06d}.png"
        mp = ROOT / split / "masks" / f"{i:06d}.png"
        if already_done(ip, mp):
            continue
        safe_save(img, ip)
        mask = draw_mask(img.size, bboxes)
        safe_save(mask, mp)

    rebuild_index(ROOT, split)


def main():
    try:
        for s in SPLITS:
            process_split(s)
        print("✅ Conversion finished.")
    except KeyboardInterrupt:
        print("⏹ Interrupted by user, partial files kept.")
        for s in SPLITS:
            rebuild_index(ROOT, s)


if __name__ == "__main__":
    main()
