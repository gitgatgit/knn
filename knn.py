import numpy as np
from collections import Counter


class KNN():
    """
    K-nearest neighbour classifier/matcher.

    Supports euclidean, manhattan, and cosine distance metrics.
    Provides both classification (predict) and nearest-neighbor lookup (find_neighbors).
    """

    def __init__(self, num_neighbours=3, distance='euclidean'):
        self.num_neighbours = num_neighbours
        self.distance = distance

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def euclidean(self, v1, v2):
        return np.sqrt(np.sum((v1 - v2) ** 2))

    def manhattan(self, v1, v2):
        return np.sum(np.abs(v1 - v2))

    def cosine(self, v1, v2):
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return 1.0
        return 1.0 - np.dot(v1, v2) / norm

    def _compute_distance(self, v1, v2):
        if self.distance == 'euclidean':
            return self.euclidean(v1, v2)
        elif self.distance == 'manhattan':
            return self.manhattan(v1, v2)
        elif self.distance == 'cosine':
            return self.cosine(v1, v2)
        raise ValueError(f"Unknown distance metric: {self.distance}")

    def _sorted_distances(self, x):
        distances = [(i, self._compute_distance(x, x_train))
                     for i, x_train in enumerate(self.X_train)]
        distances.sort(key=lambda t: t[1])
        return distances

    def neighbours(self, x):
        """Return class labels of the k nearest neighbors."""
        sorted_dist = self._sorted_distances(x)
        return [self.y_train[i] for i, _ in sorted_dist[:self.num_neighbours]]

    def find_neighbors(self, x):
        """Return top-K (index, distance) pairs sorted by ascending distance."""
        return self._sorted_distances(x)[:self.num_neighbours]

    def predict(self, X, weighted=False):
        """
        Predict class labels for each sample in X.

        weighted: if True, closer neighbors vote with weight 1/distance.
        """
        predictions = []
        for x in X:
            top_k = self._sorted_distances(x)[:self.num_neighbours]
            if weighted:
                votes = {}
                for idx, dist in top_k:
                    label = self.y_train[idx]
                    weight = 1.0 / (dist + 1e-10)
                    votes[label] = votes.get(label, 0) + weight
                top = max(votes, key=votes.get)
            else:
                labels = [self.y_train[idx] for idx, _ in top_k]
                top = Counter(labels).most_common(1)[0][0]
            predictions.append(top)
        return np.array(predictions)


def accuracy(x, y):
    x, y = np.array(x), np.array(y)
    return np.mean(x == y)
