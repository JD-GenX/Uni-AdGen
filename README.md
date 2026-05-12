# Design Your Ad: Personalized Advertising Image and Text Generation with Unified Autoregressive Models

[CVPR 2026] Official PyTorch Code for "UniAdGen: Personalized Advertising Image and Text Generation with Unified Autoregressive Models"

## 🔧 Quick Start

### 1. Installation

> CUDA >= 12.4 is required for this project.

```bash
# Clone the repository
git clone https://github.com/JD-GenX/Uni-AdGen.git
cd Uni-AdGen

# Create conda environment
conda create -n UniAdGen python=3.8
conda activate UniAdGen

# Install dependencies
pip install -r requirements.txt

# Install flash-attn separately (requires CUDA >= 12.4)
pip install flash-attn --no-build-isolation
```

### 2. Prepare Models

Download the following model weights and place them in the `ckpt/` folder:

| Model         | Download Link                                                               | Target Path             |
| ------------- | --------------------------------------------------------------------------- | ----------------------- |
| DINOv2-small  | [HuggingFace](https://huggingface.co/facebook/dinov2-small)                    | `ckpt/dinov2-small/`  |
| Janus-Pro-7B  | [HuggingFace](https://huggingface.co/deepseek-ai/Janus-Pro-7B)                 | `ckpt/Janus-Pro-7B/`  |
| SDXL-Base-1.0 | [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | `ckpt/SDXL_Base_1.0/` |

We also release our trained [checkpoints](https://3.cn/11f4I-YYG), the password is vid5mz.

After downloading, your `ckpt/` directory should look like this:

```
ckpt/
├── dinov2-small
├── Janus-Pro-7B
└── SDXL_Base_1.0

```

### 3. PAd1M Dataset

#### Step1: Dataset Download

**Test Set**

| Download Link                                     | Password |
| ------------------------------------------------- | -------- |
| [JSON](https://3.cn/11-f4JjZ3)                       | 4g1mid   |
| [Product RGB Images](https://3.cn/11cwA9-lQ)         | gi847p   |
| [Product Transparent Images](https://3.cn/11cw-AcfS) | sr3a81   |

> **Note**: The Product RGB Images folder contains both target product images and history images.

**Training Set** (Coming Soon)

> **Note**: Due to the large dataset size, we are currently uploading the complete version. A [demo training set](https://3.cn/-11fbRwgs) is available for preview (password: t6391y).

#### Step2: Update Image Paths

After downloading, update the image paths in JSON files to match your local directory structure:

```bash
python dataset/tools/update_json_paths.py \
    --json_file /path/to/test.json \
    --rgb_dir /path/to/rgb_images \
    --trans_dir /path/to/trans_images \
    --history_rgb_dir /path/to/rgb_images \
    --output /path/to/test_updated.json
```

> **Note**: Since history images and target product images are in the same folder, `--history_rgb_dir` should be set to the same path as `--rgb_dir`.

For detailed dataset structure and custom dataset building, please refer to [dataset/README.md](dataset/README.md).

### 4. Inference

**Step 1: Setup Config**

Edit the config file to specify the test dataset and checkpoint path:

```python
grit_json = "/path/to/test_dataset.json"  # path to test dataset

# Inference settings
test_data = dict(task_type='p2i', data_name='hico', batch_size=4)  # batch_size: number of products per batch
max_test_len = 250  # maximum number of test batches
```

**Step 2: Run Inference**

```bash
cd Personalize
python inference.py --cfg project/uniadgen/cfg/uni/config.py --opt test=True resume=/path/to/checkpoint
```

**Step 3: Output Files**

After inference, the output directory structure is as follows:

```
output_dir/
└── test/
    └── {data_name}_{task_type}_{val_num}/
        ├── {global_step}/                    # Main output directory
        │   ├── gt_image/                     # Ground truth product images
        │   ├── pr_image/                     # Generated product marketing images
        │   ├── image_ids/                    # Generated images (named by image_id)
        │   ├── gt_image_ids/                 # Ground truth images (named by image_id)
        │   ├── trans_image/                  # Transparent product images (background removed)
        │   └── fusion_image/                 # Fusion images (generated image + transparent product)
        └── {global_step}_batch/              # Batch output directory
            ├── {batch_idx}_layout.json       # JSON files with prediction results
            ├── {batch_idx}.png               # Combined visualization (GT + Predicted + Condition)
            └── {batch_idx}/                  # Individual images per sample
                └── {row}_{col}.png           # row: 0=GT, 1=Predicted, 2=Condition
```

**JSON Output Fields:**

| Field            | Description                                                  |
| ---------------- | ------------------------------------------------------------ |
| `pr_grounding` | Predicted title wrapped in `<prompt>...</prompt>` tags     |
| `pr_title`     | Extracted predicted title (same as `pr_grounding` content) |
| `gt_title`     | Ground truth product title                                   |

**Image Naming Convention:**

- `{idx}_{i}.png`: `idx` is the batch index, `i` is the sample index within the batch
- `{row}_{col}.png` in batch subdirectories:
  - `row=0`: Ground truth image
  - `row=1`: Predicted image
  - `row=2`: Condition image (transparent product on background)
  - `col`: Sample index within the batch

### 5. PBS Metric

PBS (Product Background Similarity) measures the background similarity between images.

**Step 1: Model Prepare**

Download our released [checkpoint](https://3.cn/116-WV6aG).

**Step 2: Calculate PBS**

```bash
cd PBS_metrics

# Step 1: Convert images to base64
python convert_images_to_base64.py -i ./images_a -o images_a -f
python convert_images_to_base64.py -i ./images_b -o images_b -f

# Step 2: Extract features
python main_inference.py --data_path . --pretrained_weights /path/to/model.pth \
    --arch vit_base_with_compress --patch_size 16 --moco_dim 64 \
    --dump_features ./output --part_index images_a

python main_inference.py --data_path . --pretrained_weights /path/to/model.pth \
    --arch vit_base_with_compress --patch_size 16 --moco_dim 64 \
    --dump_features ./output --part_index images_b

# Step 3: Calculate PBS
python compute_pbs.py \
    -fa ./output/images_a_train_feat.pth -la ./output/images_a_train_label.npy \
    -fb ./output/images_b_train_feat.pth -lb ./output/images_b_train_label.npy \
    -o pbs_results.csv
```

For more details, please refer to [PBS_metrics/README.md](PBS_metrics/README.md).

### 6. Text Metrics

BLEU and ROUGE metrics evaluate generated text quality.

**Dependencies**

```bash
pip install jieba nltk sacrebleu rouge
git clone https://github.com/luckycucu/deep-learning-metrics.git
```

**Usage**

```bash
# Extract pr_title and prompt from layout JSON files
python summary_script.py /path/to/{global_step}_batch -o summary_results.json

# Calculate BLEU and ROUGE scores
python metrics_script.py summary_results.json
```

Output includes average BLEU and ROUGE scores (0-100 scale).

## Acknowledgements

This project builds upon the following excellent works:

- [Janus](https://github.com/deepseek-ai/Janus)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [PlanGen](https://github.com/360CVGroup/PlanGen)

## Copyright & Licensing

© JD.COM. All rights reserved. The datasets and code provided in this repository are licensed exclusively for academic research purposes. Commercial use, reproduction, or distribution requires express written permission from JD.COM. Unauthorized commercial use constitutes a violation of these terms and is strictly prohibited.
