# PBS_metrics Usage Documentation

## Overview

PBS_metrics calculates PBS (Product Background Similarity) scores between two sets of images. The core pipeline:

```
Image Folder → Base64 Encoding → Feature Extraction → Similarity Calculation
```

---

## Directory Structure

```
PBS_metrics/
├── main_inference.py              # Main feature extraction script
├── run_inference.sh               # Shell script to run inference
├── convert_images_to_base64.py    # Image to base64 conversion script
├── compute_pbs.py                 # PBS calculation script
├── moco/                          # MoCo model components
├── model/                         # ViT model definitions
└── util/                          # Utility functions
```

---

## Dependencies

```bash
pip install torch torchvision numpy Pillow timm pandas scikit-learn
```

---

## Quick Start

Assume you have two folders `images_a/` and `images_b/`, and want to compute PBS similarity for images with the same filename.

---

### Step 1: Convert Images to Base64 Format

```bash
python convert_images_to_base64.py -i ./images_a -o images_a -f
python convert_images_to_base64.py -i ./images_b -o images_b -f
```

**Parameters**:

| Parameter | Description                                         |
| --------- | --------------------------------------------------- |
| `-i`    | Input image folder path                             |
| `-o`    | Output text file name                               |
| `-f`    | Use filename as ID (for matching same-named images) |

**Input**: Image folder (supports jpg, jpeg, png, bmp, webp, tiff)

**Output**: Text file

Each line format:

```
hashid base64_string
```

Example:

```
img001 /9j/4AAQSkZJRgABAQAAAQABAAD...
img002 /9j/4AAQSkZJRgABAQAAAQABAAD...
```

---

### Step 2: Extract Features

```bash
python main_inference.py \
    --data_path . \
    --nb_knn 20 \
    --pretrained_weights /path/to/moco_checkpoint.pth \
    --arch vit_base_with_compress \
    --patch_size 16 \
    --moco_dim 64 \
    --batch_size_per_gpu 2048 \
    --checkpoint_key "student" \
    --dump_features ./output \
    --part_index images_a

python main_inference.py \
    --data_path . \
    --nb_knn 20 \
    --pretrained_weights /path/to/moco_checkpoint.pth \
    --arch vit_base_with_compress \
    --patch_size 16 \
    --moco_dim 64 \
    --batch_size_per_gpu 2048 \
    --checkpoint_key "student" \
    --dump_features ./output \
    --part_index images_b
```

**Parameters**:

| Parameter                | Description                                      |
| ------------------------ | ------------------------------------------------ |
| `--data_path`          | Directory containing input txt files             |
| `--nb_knn`             | Number of k-NN neighbors                         |
| `--pretrained_weights` | Pretrained model path                            |
| `--arch`               | Model architecture                               |
| `--patch_size`         | Patch resolution for ViT                         |
| `--moco_dim`           | Feature dimension                                |
| `--batch_size_per_gpu` | Batch size per GPU                               |
| `--checkpoint_key`     | Checkpoint key                                   |
| `--dump_features`      | Output directory for features                    |
| `--part_index`         | Input txt filename (e.g., `images_a.txt`)        |

**Input**: Input path = `data_path + '/' + part_index` (e.g., `./images_a`)

**Output**: 3 files (named with `{part_index}` as prefix)

| File                                      | Format                            | Description                                 |
| ----------------------------------------- | --------------------------------- | ------------------------------------------- |
| `{part_index}_train_feat.pth`           | PyTorch Tensor, shape `(N, 64)` | Feature vector matrix                       |
| `{part_index}_train_label.npy`          | NumPy Array, shape `(N,)`       | Image ID list                               |
| `{part_index}.txt`                      | Text file                         | Each line:`hashid feat1 feat2 ... feat64` |

Example: if `part_index=images_a`, output files are:
- `images_a_train_feat.pth`
- `images_a_train_label.npy`
- `images_a.txt`

---

### Step 3: Calculate PBS Similarity

```bash
python compute_pbs.py \
    -fa ./output/images_a_train_feat.pth \
    -la ./output/images_a_train_label.npy \
    -fb ./output/images_b_train_feat.pth \
    -lb ./output/images_b_train_label.npy \
    -o pbs_results.csv
```

**Parameters**:

| Parameter | Description                      |
| --------- | -------------------------------- |
| `-fa`   | Feature file for folder A (.pth) |
| `-la`   | Label file for folder A (.npy)   |
| `-fb`   | Feature file for folder B (.pth) |
| `-lb`   | Label file for folder B (.npy)   |
| `-o`    | Output CSV file name             |

**Input**: Two sets of `.pth` feature files and `.npy` label files

**Output**: CSV file

```
filename,pbs
img001,0.846912
img002,0.562468
img003,0.923451
```

PBS range: -1 to 1. Higher values indicate greater similarity.
