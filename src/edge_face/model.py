"""
KNN classifier wrapper.
Encapsulates training, prediction, and the confidence heuristic
so the inference loop stays clean.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class FaceKNN:
    def __init__(self, k: int, weights: str):
        self.model = KNeighborsClassifier(
            n_neighbors=k, weights=weights, algorithm="auto"
        )

    def train(self, X: np.ndarray, y: list[str]):
        self.model.fit(X, y)
        print(f"[INFO] Trained KNN (k={self.model.n_neighbors}, "
              f"weights='{self.model.weights}') on {len(X)} samples")

    def predict(self, X: np.ndarray) -> str:
        """Return predicted label for a single face vector."""
        return self.model.predict(X)[0]

    def confidence(self, X: np.ndarray) -> float:
        """Heuristic confidence score in [0, 100].

        Computed as 100 minus the mean distance to the k nearest neighbors.
        This is *not* a calibrated probability — it's a relative score useful
        for thresholding (high confidence = close to known faces, low = far away).
        Negative values are clipped to 0.
        """
        distances, _ = self.model.kneighbors(X)
        mean_dist = distances[0].mean()

        score = 100.0 * np.exp(-mean_dist / 4500.0)
        return float(score)

    def score(self, X: np.ndarray) -> float:
        return self.confidence(X)