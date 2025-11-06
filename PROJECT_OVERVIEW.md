# 📰 Project Overview — Newspaper Page Segmentation

## 1. Problem Statement

The goal is to develop an automated system for **visual segmentation of newspaper and magazine pages**, capable of detecting distinct **visual blocks** such as:
- text columns,  
- images and captions,  
- tables, etc.

The system should not perform OCR or semantic understanding, but rather **identify and separate visual layout components**.  
Expected outputs:
- a **JSON** file listing rectangular blocks (`x, y, width, height`) in reading order (top-down, left-right) with corresponding categories;

---

## 2. Overall Approach

The project follows a **deep-learning-based layout segmentation** approach, using **dhSegment-torch**, a PyTorch implementation of EPFL DHLab’s `dhSegment`.

### Main Strategy

1. Use **DocLayNet** as the **pretraining dataset** — it provides 11 standard document layout categories (1: Caption, 2: Footnote, 3: Formula, 4: List-item, 5: Page-footer, 6: Page-header, 7: Picture, 8: Section-header, 9: Table, 10: Text, 11: Title).
2. **Fine-tune** the model on custom data (historical newspaper pages) to adapt it to specific visual characteristics.
3. Optionally, introduce **new classes** (e.g. "Advertisement") at the fine-tuning stage by extending the color map and mask generator.

---

## 3. Tools & Technologies

| Component | Purpose |
|------------|----------|
| **dhSegment-torch** | Main segmentation framework (PyTorch). |
| **DocLayNet (v1.1)** | Benchmark dataset for document layout segmentation. |
| **CVAT / Label Studio / VIA (VGG Image Annotator)** | Manual annotation or correction tools. |
| **Python 3.9 + PyTorch + CUDA** | Training environment. |
| **Conda environment** | Isolated dependency management. |

### Data Format

The expected dataset structure for dhSegment-torch:
```
data/<project_name>/
├── images/
│   ├── train
│   │   ├── <page_0001>.png
│   │   └── ...
│   └── val
│       ├── <page_0001>.png
│       └── ...
├── labels/
│   ├── train
│   │   ├── <page_0001>.png # RGB mask
│   │   └── ...
│   └── val
│       ├── <page_0001>.png # RGB mask
│       └── ...
├── color_labels.json
└── data.csv / train.csv / val.csv
```

Each mask is a **color PNG**, where each unique RGB triplet corresponds to a semantic class from `color_labels.json`.

---

## 4. Key Repositories

| Name | Description | URL |
|------|--------------|-----|
| **dhSegment-torch** | Main PyTorch training and inference framework. | [github.com/DHLAB-EPFL/dhSegment-torch](https://github.com/DHLAB-EPFL/dhSegment-torch) |
| **DocLayNet Dataset** | Standard document segmentation dataset used for pretraining. | [huggingface.co/datasets/docling-project/DocLayNet-v1.1](https://huggingface.co/datasets/docling-project/DocLayNet-v1.1) |
| **dhSegment (TensorFlow)** | Original implementation from EPFL DHLab. | [github.com/dhlab-epfl/dhSegment](https://github.com/dhlab-epfl/dhSegment) |

---

## 5. Steps Already Completed

### ✅ Environment Setup
- Created conda environment `dhs` with Python 3.9.
- Installed dependencies and fixed CUDA/PyTorch version issues.
- Verified successful installation of `dhSegment-torch` via `setup.py`.

### ✅ Data Conversion
Two dataset converters were implemented:

| Script | Purpose |
|--------|----------|
| `tools/doclaynet_to_color_masks_binary.py` | Converts DocLayNet into **binary RGB masks** (background/content). |
| `tools/doclaynet_to_color_masks_multiclass.py` | Converts DocLayNet into **multi-class RGB masks** (12 categories). |

Both produce:
- RGB masks (`labels/{train,val}/*.png`);
- `color_labels.json` (mapping of colors → class names);
- Directory structure compatible with dhSegment-torch.

### ✅ CSV Generation
Using:
```bash
python scripts/prepare_data.py configs/prepare_doclaynet_binary_train.jsonnet
```
Config uses:
```jsonnet
{
  data_path: "data/doclaynet_binary",
  images_dir: "data/doclaynet_binary/images/train",
  labels_dir: "data/doclaynet_binary/labels/train",
  color_labels_file_path: "data/doclaynet_binary/color_labels.json",
  csv_path: "data/doclaynet_binary/train.csv",
  type: "image",
  relative_path: true,
  num_processes: 8,
}
```

```bash
python scripts/prepare_data.py configs/prepare_doclaynet_binary_val.jsonnet
```
Config uses:
```jsonnet
{
  data_path: "data/doclaynet_binary",
  images_dir: "data/doclaynet_binary/images/val",
  labels_dir: "data/doclaynet_binary/labels/val",
  color_labels_file_path: "data/doclaynet_binary/color_labels.json",
  csv_path: "data/doclaynet_binary/val.csv",
  type: "image",
  relative_path: true,
  num_processes: 8,
}
```

This produces:
- `train.csv`
- `val.csv`

### ✅ Model Training Setup
- Analyzed `train.jsonnet`, `train_base.libsonnet`, `predict_probas.jsonnet`.
- Confirmed correct config format for `train.py` and `predict_probas.py`.
- Prepared clean configs for binary and multiclass training:
  - `configs/train_doclaynet_binary.jsonnet`
  - `configs/predict_probas_binary.jsonnet`

### ✅ Model Behavior
- dhSegment-torch dynamically resizes inputs (not restricted to 1024×1024).
- Supports pausing/resuming training via `model_best.pth`.
- Supports multi-GPU training via:
  ```bash
  torchrun --nproc_per_node=2 scripts/train.py configs/train_doclaynet_binary.jsonnet
  ```

---

## 6. Key Implementation Notes

- **`annotation_reader` behavior:**
  - In standard `prepare_data.py`, `type` should be `"image"`.
  - The special type `"via2_project"` is only for importing **VGG Image Annotator (VIA 2.x)** projects directly — it reads vector annotations and generates masks automatically.

- **Color masks** are mandatory:
  - Background = `[0,0,0]`.
  - Foreground / classes = defined in `color_labels.json`.

- **Fine-tuning on custom data:**
  - You can extend `color_labels.json` with a new RGB color for additional classes (e.g. Advertisement).
  - Re-run the converter and `prepare_data.py` for the new dataset.
  - Set `model.num_classes` accordingly in `train.jsonnet`.

- **Inference:**
  - Use `scripts/predict_probas.py` for probability maps.
  - Use `scripts/predict_annotations.py` for postprocessed segmentation outputs.

---

## 7. Planned Next Steps

1. **Fine-tune** on binary DocLayNet data.
2. **Evaluate** accuracy / IoU on validation set.
3. **Augment** dataset with real newspaper scans.
4. **Add new “Advertisement” class** and retrain (fine-tuning stage).
5. **Integrate** into GUI pipeline (e.g., CVAT for correction).

---

## 8. Summary of Core Scripts

| Category | Script | Purpose |
|-----------|--------|----------|
| Data Conversion | `tools/doclaynet_to_color_masks_binary.py` | Convert DocLayNet to RGB masks (binary). |
| Data Conversion | `tools/doclaynet_to_color_masks_multiclass.py` | Convert DocLayNet to RGB masks (12 classes). |
| Dataset Prep | `scripts/prepare_data.py` | Generate `data.csv`, split train/val/test. |
| Training | `scripts/train.py` | Train segmentation model using `.jsonnet` config. |
| Inference | `scripts/predict_probas.py` | Generate per-pixel probability maps. |
| Inference | `scripts/predict_annotations.py` | Generate final annotation masks. |

---

## 9. Additional References

- [DocLayNet: A Large Human-Annotated Dataset for Document Layout Analysis](https://arxiv.org/abs/2206.01062)
- [EPFL DHLab Publications](https://dh.unil.ch/)
- [CVAT Annotation Tool](https://github.com/opencv/cvat)
- [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/)

---

## 10. Repository Structure (Current State)

```
dhSegment-torch/
├── tools/
│   ├── doclaynet_to_color_masks_binary.py
│   └── doclaynet_to_color_masks_multiclass.py
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── predict_probas.py
│   └── predict_annotations.py
├── configs/
│   ├── prepare_doclaynet_binary.jsonnet
│   ├── train_doclaynet_binary.jsonnet
│   ├── predict_probas_binary.jsonnet
│   ├── train_base.libsonnet
│   └── segmentation.libsonnet
└── data/
    └── doclaynet_binary/
        ├── images/
        ├── labels/
        ├── color_labels.json
        ├── data.csv
        ├── train.csv
        └── val.csv
```

---

## 11. Quick Commands Recap

```bash
# Generate RGB masks from DocLayNet
python tools/doclaynet_to_color_masks_binary.py

# Prepare dataset CSV
python scripts/prepare_data.py configs/prepare_doclaynet_binary.jsonnet

# Train model
torchrun --nproc_per_node=2 scripts/train.py configs/train_doclaynet_binary.jsonnet

# Predict segmentation
python scripts/predict_probas.py configs/predict_probas_binary.jsonnet
```
