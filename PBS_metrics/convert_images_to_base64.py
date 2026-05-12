#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert Images in Folder to Base64 Text File

Converts all images in a specified folder to a base64-encoded text file,
for use with main_inference.py.

Usage:
    python convert_images_to_base64.py --input_dir ./images --output_file images.txt
"""

import os
import argparse
import base64
from io import BytesIO
from PIL import Image


def get_image_files(input_dir, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')):
    """
    Get paths of all image files in a folder.

    Args:
        input_dir: Path to input folder
        extensions: Supported image extensions

    Returns:
        List of image file paths
    """
    image_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(extensions):
                image_files.append(os.path.join(root, f))

    # Sort to ensure consistent ordering
    image_files.sort()
    return image_files


def image_to_base64(image_path):
    """
    Convert an image file to a base64-encoded string.

    Args:
        image_path: Path to image file

    Returns:
        Base64-encoded string
    """
    image = Image.open(image_path).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    base64_str = base64.urlsafe_b64encode(buffer.getvalue()).decode()
    return base64_str


def convert_folder_to_base64(input_dir, output_file, use_filename_as_id=False):
    """
    Convert all images in a folder to a base64 text file.

    Args:
        input_dir: Path to input folder
        output_file: Path to output text file
        use_filename_as_id: Whether to use filename as ID (default uses numeric index)
    """
    # Get all image files
    image_files = get_image_files(input_dir)

    if len(image_files) == 0:
        print(f"Error: No image files found in {input_dir}!")
        return

    print(f"Found {len(image_files)} images")

    # Convert and write to file
    success_count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, img_path in enumerate(image_files):
            try:
                # Generate ID
                if use_filename_as_id:
                    # Use filename (without extension) as ID
                    hashid = os.path.splitext(os.path.basename(img_path))[0]
                else:
                    # Use numeric index as ID
                    hashid = str(idx)

                # Convert to base64
                base64_str = image_to_base64(img_path)

                # Write to file
                f.write(f"{hashid} {base64_str}\n")
                success_count += 1

                # Show progress
                if (idx + 1) % 100 == 0:
                    print(f"Processed: {idx + 1}/{len(image_files)}")

            except Exception as e:
                print(f"Failed to process [{img_path}]: {e}")

    print(f"\nConversion completed!")
    print(f"  - Success: {success_count}")
    print(f"  - Failed: {len(image_files) - success_count}")
    print(f"  - Output file: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert image folder to base64 text file')
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help='Path to input image folder')
    parser.add_argument('--output_file', '-o', type=str, default='images.txt',
                        help='Path to output text file (default: images.txt)')
    parser.add_argument('--use_filename_as_id', '-f', action='store_true',
                        help='Use filename as ID (default uses numeric index)')

    args = parser.parse_args()

    # Check if input directory exists
    if not os.path.isdir(args.input_dir):
        print(f"Error: Directory {args.input_dir} does not exist!")
        return

    # Execute conversion
    convert_folder_to_base64(
        input_dir=args.input_dir,
        output_file=args.output_file,
        use_filename_as_id=args.use_filename_as_id
    )


if __name__ == '__main__':
    main()
