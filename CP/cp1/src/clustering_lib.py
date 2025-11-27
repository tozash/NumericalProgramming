# cp1/src/clustering_lib.py

from sklearn.cluster import KMeans
import numpy as np

def cluster_sklearn(X, k=2):
    """
    Wrapper for scikit-learn KMeans (uses Euclidean/L2 distance).
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels, kmeans.cluster_centers_

