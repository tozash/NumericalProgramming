# cp1/src/clustering_scratch.py

import numpy as np

class KMeansScratch:
    """
    K-Means implementation from scratch with support for different norms.
    """
    def __init__(self, k=2, max_iter=100, norm='L2', weights=None):
        self.k = k
        self.max_iter = max_iter
        self.norm = norm
        self.weights = np.array(weights) if weights is not None else None
        self.centroids = None
        self.labels = None
        
    def _distance(self, p1, p2):
        """Compute distance between p1 and p2 based on selected norm."""
        diff = p1 - p2
        
        if self.norm == 'L2':
            return np.sqrt(np.sum(diff**2))
            
        elif self.norm == 'L1':
            return np.sum(np.abs(diff))
            
        elif self.norm == 'Linf':
            return np.max(np.abs(diff))
            
        elif self.norm == 'WeightedL2':
            if self.weights is None:
                raise ValueError("Weights must be provided for WeightedL2 norm")
            return np.sqrt(np.sum(self.weights * (diff**2)))
            
        else:
            raise ValueError(f"Unknown norm: {self.norm}")

    def fit(self, X):
        n_samples, n_features = X.shape
        
        # 1. Initialize centroids
        k = min(self.k, n_samples)
        if k == 0: return np.array([]) # Handle empty
        
        indices = np.random.choice(n_samples, k, replace=False)
        self.centroids = X[indices].copy()
        
        for iteration in range(self.max_iter):
            # 2. Assign points to nearest centroid
            new_labels = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                distances = [self._distance(X[i], c) for c in self.centroids]
                new_labels[i] = np.argmin(distances)
            
            # Check convergence
            if np.array_equal(self.labels, new_labels):
                break
            self.labels = new_labels
            
            # 3. Recompute centroids
            for j in range(k):
                points_in_cluster = X[self.labels == j]
                if len(points_in_cluster) > 0:
                    # Mean is the minimizer for L2.
                    self.centroids[j] = np.mean(points_in_cluster, axis=0)
                    
        return self.labels

