"""
K-Fold Cross-Validation for Cattle ReID
=========================================
Stolen from: https://github.com/Phoenix4582/CowIDentifier (config_kfold_fused.yaml)
Adapted for cattle ReID evaluation.

Instead of a single train/gallery/query split, this runs K folds and averages
the results for more reliable metrics.

Usage:
  python kfold_eval.py --data_dir data/processed --k_folds 5
"""

import os
import numpy as np
import torch
from glob import glob
from pathlib import Path
from collections import defaultdict
import json


class KFoldEvaluator:
    """
    K-Fold cross-validation evaluator for cattle ReID.

    Based on CowIDentifier's KFoldMultiCamCowsDataModule approach.
    Splits cow IDs into K folds, then for each fold:
      - K-1 folds for training
      - 1 fold for testing
    Reports averaged metrics across all folds.

    Reference: CowIDentifier/config_kfold_fused.yaml
      data:
        num_folds: 10
        split_seed: 12345
    """
    def __init__(self, k_folds=5, split_seed=12345):
        self.k_folds = k_folds
        self.split_seed = split_seed
        self.results = []

    def split_cows(self, cow_ids):
        """
        Split cow IDs into K folds using deterministic random seed.
        Returns list of (train_ids, test_ids) tuples.

        Based on CowIDentifier's approach:
          - Fold k: test = fold_k, train = all other folds
        """
        rng = np.random.RandomState(self.split_seed)
        shuffled = cow_ids.copy()
        rng.shuffle(shuffled)

        # Split into K roughly equal folds
        folds = np.array_split(shuffled, self.k_folds)

        splits = []
        for k in range(self.k_folds):
            test_ids = set(folds[k].tolist())
            train_ids = set()
            for j in range(self.k_folds):
                if j != k:
                    train_ids.update(folds[j].tolist())
            splits.append((train_ids, test_ids))

        return splits

    def evaluate_fold(self, fold_num, train_ids, test_ids, embeddings_dict,
                      gallery_dict=None, threshold=0.6):
        """
        Evaluate one fold.

        Args:
            fold_num: which fold (0-indexed)
            train_ids: set of cow IDs for training
            test_ids: set of cow IDs for testing
            embeddings_dict: {cow_id: [embedding1, embedding2, ...]}
            gallery_dict: optional separate gallery (if None, use train_ids)
            threshold: L2 distance threshold for matching

        Returns:
            dict with fold metrics
        """
        if gallery_dict is None:
            gallery_dict = embeddings_dict

        # Build gallery from train cows
        gallery = {}
        for cid in train_ids:
            if cid in gallery_dict and len(gallery_dict[cid]) > 0:
                embs = np.array(gallery_dict[cid])
                gallery[cid] = {
                    'mean': np.mean(embs, axis=0),
                    'embeddings': embs,
                }

        if len(gallery) == 0:
            return {'fold': fold_num, 'accuracy': 0, 'mAP': 0}

        # Test on test cows
        correct = 0
        total = 0
        all_distances = []
        all_labels = []

        for cid in test_ids:
            if cid not in embeddings_dict or len(embeddings_dict[cid]) == 0:
                continue

            for emb in embeddings_dict[cid]:
                emb = np.array(emb)

                # Find best match in gallery
                best_dist = float('inf')
                best_match = None
                for gid, gdata in gallery.items():
                    dist = np.sqrt(np.mean((emb - gdata['mean']) ** 2))
                    if dist < best_dist:
                        best_dist = dist
                        best_match = gid

                all_distances.append(best_dist)
                all_labels.append(cid)

                if best_match == cid and best_dist < threshold:
                    correct += 1
                total += 1

        accuracy = (correct / max(total, 1)) * 100

        # Compute mAP (simplified)
        all_distances = np.array(all_distances)
        all_labels = np.array(all_labels)
        mAP = self._compute_mAP(all_distances, all_labels, threshold)

        result = {
            'fold': fold_num,
            'accuracy': accuracy,
            'mAP': mAP,
            'test_cows': len(test_ids),
            'gallery_cows': len(gallery),
            'total_queries': total,
            'threshold': threshold,
        }

        return result

    def _compute_mAP(self, distances, labels, threshold):
        """Simplified Mean Average Precision computation."""
        unique_labels = np.unique(labels)
        aps = []

        for label in unique_labels:
            # Binary relevance: 1 if same cow, 0 if different
            relevance = (labels == label).astype(float)

            # Sort by distance (ascending = most similar first)
            sorted_idx = np.argsort(distances)
            sorted_relevance = relevance[sorted_idx]

            # Compute precision at each rank
            tp_cumsum = np.cumsum(sorted_relevance)
            precision_at_k = tp_cumsum / np.arange(1, len(sorted_relevance) + 1)

            # AP = average precision for this query
            ap = np.sum(precision_at_k * sorted_relevance) / max(np.sum(relevance), 1)
            aps.append(ap)

        return np.mean(aps) * 100 if aps else 0

    def run_kfold(self, cow_ids, embeddings_dict, threshold=0.6, verbose=True):
        """
        Run full K-Fold evaluation.

        Args:
            cow_ids: array of cow IDs with embeddings
            embeddings_dict: {cow_id: [embedding1, embedding2, ...]}
            threshold: L2 distance threshold
            verbose: print progress

        Returns:
            dict with averaged metrics and per-fold results
        """
        splits = self.split_cows(cow_ids)
        self.results = []

        for k, (train_ids, test_ids) in enumerate(splits):
            if verbose:
                print(f"\n--- Fold {k+1}/{self.k_folds} ---")
                print(f"  Train: {len(train_ids)} cows, Test: {len(test_ids)} cows")

            result = self.evaluate_fold(
                fold_num=k,
                train_ids=train_ids,
                test_ids=test_ids,
                embeddings_dict=embeddings_dict,
                threshold=threshold,
            )
            self.results.append(result)

            if verbose:
                print(f"  Accuracy: {result['accuracy']:.1f}%")
                print(f"  mAP: {result['mAP']:.1f}%")

        # Compute averages
        avg_accuracy = np.mean([r['accuracy'] for r in self.results])
        avg_mAP = np.mean([r['mAP'] for r in self.results])
        std_accuracy = np.std([r['accuracy'] for r in self.results])
        std_mAP = np.std([r['mAP'] for r in self.results])

        summary = {
            'k_folds': self.k_folds,
            'threshold': threshold,
            'avg_accuracy': avg_accuracy,
            'std_accuracy': std_accuracy,
            'avg_mAP': avg_mAP,
            'std_mAP': std_mAP,
            'fold_results': self.results,
        }

        if verbose:
            print(f"\n{'='*50}")
            print(f"K-Fold Results ({self.k_folds} folds)")
            print(f"{'='*50}")
            print(f"Accuracy: {avg_accuracy:.1f}% +/- {std_accuracy:.1f}%")
            print(f"mAP:      {avg_mAP:.1f}% +/- {std_mAP:.1f}%")
            print(f"{'='*50}")

        return summary

    def find_best_threshold(self, cow_ids, embeddings_dict,
                            thresholds=None, verbose=True):
        """
        Test multiple thresholds and find the best one.
        Based on CowIDentifier's approach of tuning the pseudo_thr parameter.
        """
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        best_acc = 0
        best_thr = 0.6
        results = []

        for thr in thresholds:
            summary = self.run_kfold(cow_ids, embeddings_dict,
                                     threshold=thr, verbose=False)
            acc = summary['avg_accuracy']
            results.append({'threshold': thr, 'accuracy': acc})

            if acc > best_acc:
                best_acc = acc
                best_thr = thr

            if verbose:
                print(f"  Threshold {thr:.1f} -> Accuracy {acc:.1f}%")

        if verbose:
            print(f"\nBest threshold: {best_thr:.1f} (accuracy: {best_acc:.1f}%)")

        return best_thr, best_acc, results


def save_results(summary, output_path):
    """Save K-Fold results to JSON."""
    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2, default=convert)
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='K-Fold Evaluation for Cattle ReID')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Root data directory with train/gallery/query')
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Number of folds')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='L2 distance threshold')
    parser.add_argument('--find_threshold', action='store_true',
                        help='Search for best threshold')
    parser.add_argument('--output', type=str, default='kfold_results.json',
                        help='Output JSON file')
    args = parser.parse_args()

    print("K-Fold evaluation requires trained embeddings.")
    print("Run the training pipeline first, then use KFoldEvaluator:")
    print()
    print("  from kfold_eval import KFoldEvaluator")
    print("  evaluator = KFoldEvaluator(k_folds=5)")
    print("  summary = evaluator.run_kfold(cow_ids, embeddings_dict)")
