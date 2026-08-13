"""
KNN-Based Matching for Cattle ReID
====================================
Stolen from: https://github.com/Phoenix4582/CowIDentifier (utils_misc.py references)
Uses K-Nearest Neighbors instead of simple L2 mean-distance matching.

Why KNN is better than mean-matching:
  - Mean embedding assumes one "average" look per cow
  - Real cows look different from different angles (front vs side)
  - KNN checks against ALL gallery images, not just the average
  - More robust to unusual camera angles in CCTV footage

Usage:
  from knn_matcher import KNNMatcher
  matcher = KNNMatcher(k=5)
  matcher.register('Cow_001', embeddings_list)
  result = matcher.match(query_embedding)
"""

import numpy as np
from collections import defaultdict


class KNNMatcher:
    """
    K-Nearest Neighbor matcher for cattle re-identification.

    Based on CowIDentifier's KNNClusterPerformance and KNNAccuracy functions.
    Instead of comparing against mean embedding, compares against all
    gallery embeddings and uses majority voting among K nearest neighbors.

    Reference: CowIDentifier uses KNN for evaluation:
      knn_accuracy = KNNAccuracy(train_embd, train_lbls, embd, lbls)

    Reference: CowIDentifier/Supervised/lightning_supervised_model.py
      from utilities.utils_misc import KNNAccuracy, KNNMetrics
    """
    def __init__(self, k=5, threshold=0.6):
        """
        Args:
            k: number of nearest neighbors to consider
            threshold: maximum L2 distance to consider a match
        """
        self.k = k
        self.threshold = threshold
        self.gallery = {}  # {cow_id: {'embeddings': [array], 'mean': array, 'n': int}}

    def register(self, cow_id, embeddings):
        """
        Register a cow with multiple embeddings.

        Args:
            cow_id: string identifier (e.g., 'Cow_001')
            embeddings: list of numpy arrays, each 512-dim (or any dim)
        """
        embs = np.array(embeddings)
        self.gallery[cow_id] = {
            'embeddings': embs,
            'mean': np.mean(embs, axis=0),
            'n': len(embs),
        }

    def remove(self, cow_id):
        """Remove a cow from the gallery."""
        if cow_id in self.gallery:
            del self.gallery[cow_id]

    def names(self):
        """Return list of all registered cow IDs."""
        return list(self.gallery.keys())

    def match(self, query_embedding):
        """
        Match a query embedding against the gallery using KNN.

        Args:
            query_embedding: numpy array, shape (embed_dim,)

        Returns:
            dict with 'id', 'confidence', 'distance', 'knn_distances', 'knn_ids'
        """
        if not self.gallery:
            return {
                'id': 'Unknown',
                'confidence': 0.0,
                'distance': float('inf'),
                'knn_distances': [],
                'knn_ids': [],
            }

        # Compute distances to ALL gallery embeddings
        all_distances = []
        all_ids = []

        for cow_id, data in self.gallery.items():
            for emb in data['embeddings']:
                dist = np.sqrt(np.mean((query_embedding - emb) ** 2))
                all_distances.append(dist)
                all_ids.append(cow_id)

        all_distances = np.array(all_distances)
        all_ids = np.array(all_ids)

        # Find K nearest neighbors
        k = min(self.k, len(all_distances))
        sorted_idx = np.argsort(all_distances)[:k]
        knn_distances = all_distances[sorted_idx]
        knn_ids = all_ids[sorted_idx]

        # Majority voting among K nearest neighbors
        vote_counts = defaultdict(float)
        for i, (dist, cid) in enumerate(zip(knn_distances, knn_ids)):
            # Weight votes by inverse distance (closer = stronger vote)
            weight = 1.0 / (dist + 1e-6)
            vote_counts[cid] += weight

        # Get the cow with most votes
        best_id = max(vote_counts, key=vote_counts.get)
        best_distance = float(np.mean(knn_distances[knn_ids == best_id]))

        # Confidence based on vote margin and distance
        total_weight = sum(vote_counts.values())
        best_weight = vote_counts[best_id]
        vote_confidence = best_weight / total_weight

        # Distance-based confidence
        if best_distance < self.threshold:
            dist_confidence = 1.0 - (best_distance / self.threshold)
        else:
            dist_confidence = 0.0

        # Combined confidence
        confidence = 0.6 * vote_confidence + 0.4 * dist_confidence

        return {
            'id': best_id if best_distance < self.threshold else 'Unknown',
            'confidence': float(confidence),
            'distance': float(best_distance),
            'knn_distances': knn_distances.tolist(),
            'knn_ids': knn_ids.tolist(),
            'vote_counts': dict(vote_counts),
        }

    def match_batch(self, query_embeddings):
        """Match multiple query embeddings at once."""
        return [self.match(q) for q in query_embeddings]

    def save(self, path):
        """Save gallery to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.gallery, f)
        print(f"Gallery saved to {path} ({len(self.gallery)} cows)")

    def load(self, path):
        """Load gallery from disk."""
        import pickle
        with open(path, 'rb') as f:
            self.gallery = pickle.load(f)
        print(f"Gallery loaded from {path} ({len(self.gallery)} cows)")


class KNNMetrics:
    """
    Evaluation metrics using KNN matching.
    Based on CowIDentifier's utils_misc functions.

    Reference: CowIDentifier/Supervised/lightning_supervised_model.py
      from utilities.utils_misc import KNNAccuracy, KNNMetrics, additional_metrics
    """
    @staticmethod
    def knn_accuracy(gallery_embs, gallery_labels, query_embs, query_labels, k=5):
        """
        Compute KNN classification accuracy.

        Args:
            gallery_embs: numpy array (N_gallery, embed_dim)
            gallery_labels: numpy array (N_gallery,)
            query_embs: numpy array (N_query, embed_dim)
            query_labels: numpy array (N_query,)
            k: number of neighbors

        Returns:
            accuracy percentage
        """
        correct = 0
        total = len(query_labels)

        for i in range(total):
            query = query_embs[i]
            true_label = query_labels[i]

            # Compute distances to all gallery items
            dists = np.sqrt(np.mean((gallery_embs - query) ** 2, axis=1))

            # K nearest neighbors
            k_actual = min(k, len(dists))
            nearest_idx = np.argsort(dists)[:k_actual]
            nearest_labels = gallery_labels[nearest_idx]

            # Majority vote
            unique_labels, counts = np.unique(nearest_labels, return_counts=True)
            predicted = unique_labels[np.argmax(counts)]

            if predicted == true_label:
                correct += 1

        return (correct / total) * 100

    @staticmethod
    def precision_recall_f1(gallery_embs, gallery_labels, query_embs, query_labels, k=5):
        """
        Compute precision, recall, and F1 score using KNN.
        """
        tp = fp = fn = 0

        for i in range(len(query_labels)):
            query = query_embs[i]
            true_label = query_labels[i]

            dists = np.sqrt(np.mean((gallery_embs - query) ** 2, axis=1))
            k_actual = min(k, len(dists))
            nearest_idx = np.argsort(dists)[:k_actual]
            nearest_labels = gallery_labels[nearest_idx]

            unique_labels, counts = np.unique(nearest_labels, return_counts=True)
            predicted = unique_labels[np.argmax(counts)]

            if predicted == true_label:
                tp += 1
            else:
                fn += 1
                fp += 1

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)

        return precision * 100, recall * 100, f1 * 100
