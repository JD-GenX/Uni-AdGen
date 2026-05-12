"""
Script to update image paths in dataset JSON files for open-source release.

Usage:
    python update_json_paths.py --json_file /path/to/train.json --rgb_dir /path/to/rgb_images --trans_dir /path/to/trans_images --output /path/to/output.json

For personalized dataset, also provide --history_rgb_dir:
    python update_json_paths.py --json_file /path/to/personalized/train.json --rgb_dir /path/to/rgb_images --trans_dir /path/to/trans_images --history_rgb_dir /path/to/history_images --output /path/to/output.json
"""

import json
import os
import argparse
from pathlib import Path


def get_filename_from_path(path):
    """Extract filename from a full path."""
    return os.path.basename(path)


def update_json_paths(json_file, rgb_dir, trans_dir, history_rgb_dir=None, output_file=None):
    """
    Update image paths in JSON file.

    Args:
        json_file: Path to input JSON file
        rgb_dir: Directory containing RGB images
        trans_dir: Directory containing transparent images
        history_rgb_dir: Directory containing history RGB images (for personalized dataset)
        output_file: Path to output JSON file (default: overwrite input file)
    """
    # Load JSON data
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Processing {len(data)} entries...")

    for i, entry in enumerate(data):
        # Update f_path (RGB image)
        if entry.get('f_path'):
            rgb_filename = get_filename_from_path(entry['f_path'])
            entry['f_path'] = os.path.join(rgb_dir, rgb_filename)

        # Update f_path_trans (transparent image)
        if entry.get('f_path_trans'):
            trans_filename = get_filename_from_path(entry['f_path_trans'])
            entry['f_path_trans'] = os.path.join(trans_dir, trans_filename)

        # Update history_images (for personalized dataset)
        if entry.get('history_images') and history_rgb_dir:
            updated_history = []
            for hist_path in entry['history_images']:
                hist_filename = get_filename_from_path(hist_path)
                updated_history.append(os.path.join(history_rgb_dir, hist_filename))
            entry['history_images'] = updated_history

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(data)} entries...")

    # Write output
    if output_file is None:
        output_file = json_file

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Done! Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Update image paths in dataset JSON files')
    parser.add_argument('--json_file', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--rgb_dir', type=str, required=True, help='Directory containing RGB images')
    parser.add_argument('--trans_dir', type=str, required=True, help='Directory containing transparent images')
    parser.add_argument('--history_rgb_dir', type=str, default=None, help='Directory containing history RGB images (for personalized dataset)')
    parser.add_argument('--output', type=str, default=None, help='Path to output JSON file (default: overwrite input file)')

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.json_file):
        raise FileNotFoundError(f"JSON file not found: {args.json_file}")
    if not os.path.isdir(args.rgb_dir):
        raise NotADirectoryError(f"RGB directory not found: {args.rgb_dir}")
    if not os.path.isdir(args.trans_dir):
        raise NotADirectoryError(f"Transparent image directory not found: {args.trans_dir}")
    if args.history_rgb_dir and not os.path.isdir(args.history_rgb_dir):
        raise NotADirectoryError(f"History RGB directory not found: {args.history_rgb_dir}")

    update_json_paths(
        json_file=args.json_file,
        rgb_dir=args.rgb_dir,
        trans_dir=args.trans_dir,
        history_rgb_dir=args.history_rgb_dir,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
