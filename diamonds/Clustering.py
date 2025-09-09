"""
Clustering.py
Wraps k-means, agglomerative clustering and mean-shift from sklearn.
Provides fit(X, model_name, **kwargs) and predict(X_new)
"""

from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift
import numpy as np

class Clustering:
    def __init__(self):
        self.model = None
        self.name = None

    def fit(self, X, model_name="kmeans", **kwargs):
        model_name = model_name.lower()
        self.name = model_name
        if model_name == "kmeans":
            n_clusters = kwargs.get("n_clusters", 3)
            self.model = KMeans(n_clusters=n_clusters, random_state=42)
            self.model.fit(X)
        elif model_name == "agglomerative":
            n_clusters = kwargs.get("n_clusters", 3)
            self.model = AgglomerativeClustering(n_clusters=n_clusters)
            self.model.fit(X)
        elif model_name == "meanshift":
            self.model = MeanShift()
            self.model.fit(X)
        else:
            raise ValueError("Unknown clustering model")
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("No model trained")
        # Note: AgglomerativeClustering has no predict method in sklearn; use labels_ if same data
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        else:
            # fallback: if data used for fit is same as X, return labels_
            try:
                return self.model.labels_
            except AttributeError:
                raise RuntimeError("Model cannot predict new data (Agglomerative in sklearn)")
