#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PBS (Product Background Similarity) Calculator

Computes cosine similarity between image features from two folders.

Usage:
    python compute_pbs.py --features_a images_a_train_feat.pth --labels_a images_a_train_label.npy \
                          --features_b images_b_train_feat.pth --labels_b images_b_train_label.npy \
                          --output pbs_results.csv
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def load_features_and_labels(feat_path, label_path):
    """
    Load features and labels from files.

    Args:
        feat_path: Path to .pth feature file
        label_path: Path to .npy label file

    Returns:
        Dictionary mapping label -> feature vector
    """
    features = torch.load(feat_path, map_location='cpu')
    labels = np.load(label_path, allow_pickle=True)

    if torch.is_tensor(features):
        features = features.numpy()

    label_to_feature = {}
    for label, feat in zip(labels, features):
        label_to_feature[str(label)] = feat

    return label_to_feature


def main():
    parser = argparse.ArgumentParser(description='Compute PBS (cosine similarity) scores')
    parser.add_argument('--features_a', '-fa', type=str, required=True,
                        help='Path to features .pth file from folder A')
    parser.add_argument('--labels_a', '-la', type=str, required=True,
                        help='Path to labels .npy file from folder A')
    parser.add_argument('--features_b', '-fb', type=str, required=True,
                        help='Path to features .pth file from folder B')
    parser.add_argument('--labels_b', '-lb', type=str, required=True,
                        help='Path to labels .npy file from folder B')
    parser.add_argument('--output', '-o', type=str, default='pbs_results.csv',
                        help='Path to output CSV file (default: pbs_results.csv)')

    args = parser.parse_args()

    # Load features
    print("Loading features from folder A...")
    features_a = load_features_and_labels(args.features_a, args.labels_a)
    print(f"  Loaded {len(features_a)} features")

    print("Loading features from folder B...")
    features_b = load_features_and_labels(args.features_b, args.labels_b)
    print(f"  Loaded {len(features_b)} features")

    # Find common labels
    common_labels = set(features_a.keys()) & set(features_b.keys())

    if len(common_labels) == 0:
        print("Error: No matching labels found!")
        return

    print(f"\nFound {len(common_labels)} matching image pairs")

    # Compute PBS (cosine similarity)
    results = []
    for label in sorted(common_labels):
        pbs = cosine_similarity([features_a[label]], [features_b[label]])[0, 0]
        results.append({'filename': label, 'pbs': pbs})

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False, float_format='%.6f')

    # Print summary
    print(f"\nPBS Summary:")
    print(f"  Mean:   {df['pbs'].mean():.4f}")
    print(f"  Std:    {df['pbs'].std():.4f}")
    print(f"  Min:    {df['pbs'].min():.4f}")
    print(f"  Max:    {df['pbs'].max():.4f}")
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
