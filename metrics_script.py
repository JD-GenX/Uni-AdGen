#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate BLEU and ROUGE scores between pr_title and prompt pairs in summary_results.json
"""
import json
import sys
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings("ignore")

# Add path to reference files
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "deep-learning-metrics-master" / "src"))

from bleu_zh import bleu_zh
from rouge_zh import rouge_l_zh


def main():
    parser = argparse.ArgumentParser(description='Calculate BLEU and ROUGE scores between pr_title and prompt')
    parser.add_argument('input_file', type=str, help='Path to summary_results.json')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output file path (default: metrics_results.json in same directory as input)')
    parser.add_argument('--skip-empty', action='store_true', help='Skip entries with empty pr_title')
    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists():
        print(f"Error: File does not exist - {input_file}")
        return

    output_file = Path(args.output) if args.output else input_file.parent / "metrics_results.json"

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} records")

    all_results = []
    bleu_scores = []
    rouge_scores = []
    skipped_count = 0

    for i, item in enumerate(data):
        pr_title = item['pr_title']
        prompt = item['prompt']

        # Check whether to skip empty pr_title
        if args.skip_empty and (not pr_title or pr_title.strip() == ''):
            skipped_count += 1
            continue

        # Calculate BLEU: pr_title as candidate, prompt as reference
        try:
            bleu_score = bleu_zh(
                references=[[prompt]],
                candidates=[pr_title],
                use_jieba=True,
                model_name='nltk'
            )
        except Exception as e:
            print(f"BLEU calculation error (index {i}): {e}")
            bleu_score = 0.0

        # Calculate ROUGE-L
        try:
            rouge_score = rouge_l_zh(
                references=[[prompt]],
                candidates=[pr_title],
                use_jieba=True
            )
        except Exception as e:
            print(f"ROUGE calculation error (index {i}): {e}")
            rouge_score = 0.0

        bleu_scores.append(bleu_score)
        rouge_scores.append(rouge_score)

        result = {
            'source_file': item.get('source_file', ''),
            'index': item.get('index', i),
            'pr_title': pr_title,
            'prompt': prompt,
            'bleu': bleu_score,
            'rouge_l': rouge_score
        }
        all_results.append(result)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(data)} records")

    # Calculate average scores
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0

    output_data = {
        'results': all_results,
        'summary': {
            'total_count': len(data),
            'processed_count': len(bleu_scores),
            'skipped_count': skipped_count,
            'average_bleu': round(avg_bleu, 6),
            'average_rouge_l': round(avg_rouge, 6)
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("Calculation complete!")
    print(f"Total records: {len(data)}")
    print(f"Processed: {len(bleu_scores)}")
    print(f"Skipped: {skipped_count}")
    print(f"Average BLEU score: {avg_bleu:.6f}")
    print(f"Average ROUGE-L score: {avg_rouge:.6f}")
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == '__main__':
    main()
