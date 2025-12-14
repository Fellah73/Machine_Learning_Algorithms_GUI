import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs


class ClusteringMetrics:
    def __init__(self):
        self.optimal_k = None
        self.silhouette_score_value = None

    def analyze_clusters(self, data):
        try:
            if data is None or len(data) == 0:
                # Generate sample data if no data provided
                data, _ = make_blobs(n_samples=300, centers=4, n_features=2,
                                   random_state=42, cluster_std=0.60)

            # Calculate elbow method metrics
            k_range = range(1, 11)
            wcss_values = []
            silhouette_scores = []

            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(data)
                wcss_values.append(kmeans.inertia_)

                if k > 1:
                    labels = kmeans.labels_
                    sil_score = silhouette_score(data, labels)
                    silhouette_scores.append(sil_score)
                else:
                    silhouette_scores.append(0)

            # Find optimal K using elbow method
            optimal_k = self.find_optimal_k(wcss_values)
            self.optimal_k = optimal_k
            self.silhouette_score_value = silhouette_scores[optimal_k-1]

            return {
                'error': False,
                'optimal_k': optimal_k,
                'silhouette_score': self.silhouette_score_value,
                'k_range': list(k_range),
                'wcss_values': wcss_values,
                'silhouette_scores': silhouette_scores
            }

        except Exception as e:
            return {
                'error': True,
                'message': str(e)
            }

    def find_optimal_k(self, wcss_values):
        diffs = np.diff(wcss_values, 2)
        optimal_k = np.argmax(diffs) + 2
        return min(optimal_k, 8)

    def get_optimal_k(self):
        return self.optimal_k

    def get_silhouette_score(self):
        return self.silhouette_score_value