# PAd1M Dataset

We construct a large-scale and diverse **P**ersonalized **Ad**vertising image-text (**PAd1M**) dataset fromJD.com. The dataset of 1,145,371 users, with 18,923,555 clicked product images and texts, averaging more than sixteen multimodal historical behaviors per user.

## Dataset Download

> **Note**: Due to the large size of the complete dataset, it is still being uploaded. Currently, only the test set (JSON, RGB Images, Transparent Images) is available for download.

**Test Set**

| Download Link                                     | Password |
| ------------------------------------------------- | -------- |
| [JSON](https://3.cn/11-f4JjZ3)                       | 4g1mid   |
| [Product RGB Images](https://3.cn/11cwA9-lQ)         | gi847p   |
| [Product Transparent Images](https://3.cn/11cw-AcfS) | sr3a81   |

> **Note**: The Product RGB Images folder contains both target product images and history images.

**Complete Dataset** (Coming Soon)

## Data Format

The dataset is provided in JSON format, where each entry contains the following fields:

| Field                 | Type   | Description                                                                                                                        |
| --------------------- | ------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `f_path`            | string | Path to the product RGB image                                                                                                      |
| `f_path_trans`      | string | Path to the product transparent image (background removed)                                                                         |
| `ref_exps`          | list   | Bounding box coordinates `[[x1, y1, x2, y2]]` (top-left and bottom-right points), not used in personalization code               |
| `width`             | int    | Image width                                                                                                                        |
| `height`            | int    | Image height                                                                                                                       |
| `caption`           | string | Background description of the product RGB image                                                                                    |
| `cid3_english_name` | string | Product category (set to "object", not used in code)                                                                               |
| `sku_title_cn`      | string | Product text description                                                                                                           |
| `gt_title_cn`       | string | Ground truth product titles (multiple titles separated by `<title_split>`)                                                       |
| `word_list`         | string | Product selling points (comma-separated)                                                                                           |
| `f_path_white`      | string | Path to white background image (set to "", not used in code; transparent image is converted to white background during processing) |
| `f_ocr_path`        | string | Path to text glyph image from product image (set to "", not used in code)                                                          |
| `max_ratio`         | float  | Scaling ratio of the product in the transparent image                                                                              |
| `history_images`    | list   | Paths to historically clicked product images by the user                                                                           |
| `history_titles`    | list   | Titles of historically clicked products by the user                                                                                |
| `text_similarity`   | list   | Text similarity scores between historical product titles and the target product description (`sku_title_cn`)                     |

## Example

```json
{
    "f_path": "/path/to/rgb_image.png",
    "f_path_trans": "/path/to/trans_image.png",
    "ref_exps": [[0, 0, 0, 0]],
    "width": 512,
    "height": 512,
    "caption": "",
    "cid3_english_name": "object",
    "sku_title_cn": "Product description text",
    "gt_title_cn": "Ground truth title",
    "word_list": "point1, point2, point3",
    "f_path_white": "",
    "f_ocr_path": "",
    "max_ratio": 1.0,
    "history_images": [
        "/path/to/history1.png",
        "/path/to/history2.png"
    ],
    "history_titles": [
        "History product title 1",
        "History product title 2"
    ],
    "text_similarity": [0.52, 0.55]
}
```

## Update Image Paths

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

### Arguments

| Argument              | Description                                                             |
| --------------------- | ----------------------------------------------------------------------- |
| `--json_file`       | Path to input JSON file                                                 |
| `--rgb_dir`         | Directory containing RGB images                                         |
| `--trans_dir`       | Directory containing transparent images                                 |
| `--history_rgb_dir` | Directory containing history RGB images (same as `--rgb_dir`)         |
| `--output`          | Path to output JSON file (optional, defaults to overwriting input file) |

## Custom Dataset

To build your own dataset, follow these steps:

### Step 1: Prepare JSON File

Organize your data in the same JSON format as described in the Data Format section. Prepare the `history_images` and `history_titles` fields, but you can leave `text_similarity` as an empty list (it will be computed in Step 3).

### Step 2: Download Checkpoint

Download the text embedding model from [HuggingFace](https://huggingface.co/shibing624/text2vec-base-chinese)

### Step 3: Compute Text Similarity

Run the following command to compute `text_similarity` scores:

```bash
python dataset/tools/text_similarity.py /path/to/input.json --output /path/to/output.json --checkpoint /path/to/text2vec-base-chinese
```

### Arguments

| Argument         | Description                                                |
| ---------------- | ---------------------------------------------------------- |
| `input_file`   | Path to input JSON file                                    |
| `--output`     | Path to output JSON file                                   |
| `--checkpoint` | Path to text embedding model (e.g., text2vec-base-chinese) |
