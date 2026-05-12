#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract pr_title and prompt fields from all JSON files in a directory
"""
import json
import re
import argparse
from pathlib import Path


def extract_prompt_content(prompt_text):
    """Remove <prompt></prompt> tags"""
    match = re.match(r'<prompt>(.*)</prompt>', prompt_text, re.DOTALL)
    if match:
        return match.group(1)
    return prompt_text


def main():
    parser = argparse.ArgumentParser(description='Extract pr_title and prompt fields from JSON files')
    parser.add_argument('input_dir', type=str, help='Input directory containing JSON files')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file path (default: summary_results.json in input directory)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Directory does not exist - {input_dir}")
        return

    output_file = Path(args.output) if args.output else input_dir / "summary_results.json"

    all_results = []
    json_files = sorted(input_dir.glob("*_layout.json"))

    print(f"Found {len(json_files)} JSON files")

    for json_file in json_files:
        print(f"Processing: {json_file.name}")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        pr_titles = data.get('pr_title', [])
        prompts = data.get('prompt', [])

        if len(pr_titles) != len(prompts):
            print(f"Warning: {json_file.name} - pr_title and prompt count mismatch: {len(pr_titles)} vs {len(prompts)}")
            continue

        for i, (title, prompt) in enumerate(zip(pr_titles, prompts)):
            clean_prompt = extract_prompt_content(prompt)
            all_results.append({
                'source_file': json_file.name,
                'index': i,
                'pr_title': title,
                'prompt': clean_prompt
            })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nDone! Processed {len(json_files)} JSON files, extracted {len(all_results)} records")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
