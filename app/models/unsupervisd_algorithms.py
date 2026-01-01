from scipy.spatial.distance import cdist
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage


class UnsupervisedAlgorithms:
    def __init__(self):
        self.models = {}
        self.results = {}

    def apply_partitioning_algorithm(self, algorithm_name, data, params, selected_columns):
        """K-Means, K-Medoids"""
        try:
            n_clusters = params.get('n_clusters', 3)
            max_iter = params.get('max_iter', 300)
            distance_metric = params.get('distance_metric', 'euclidean')

            if algorithm_name == "K-Means":
                kmeans = KMeans(n_clusters=n_clusters,
                                max_iter=max_iter,
                                random_state=42,
                                n_init=10)
                labels = kmeans.fit_predict(data)
                return {
                    'data': data,
                    'selected_columns': selected_columns,
                    'labels': labels,
                    'centers': kmeans.cluster_centers_,
                    'algorithm': algorithm_name,
                    'n_clusters': n_clusters,
                    'inertia': kmeans.inertia_
                }

            elif algorithm_name == "K-Medoids":
                result = self.kmedoids_clustering(
                    data, n_clusters, distance_metric, max_iter)

                # Vérifier s'il y a une erreur
                if 'error' in result:
                    print(f"K-Medoids error: {result['error']}")
                    return None
                return {
                    'data': data,
                    'selected_columns': selected_columns,
                    'labels': result['labels'],
                    'centers': result['medoids'],  # Medoids comme centres
                    'algorithm': algorithm_name,
                    'n_clusters': n_clusters,
                    'inertia': result['inertia'],
                    'medoids': result['medoids'],
                    'medoid_indices': result['medoid_indices'],
                    'iterations_run': result['iterations_run']
                }

        except Exception as e:
            print(f"Error in {algorithm_name}: {str(e)}")
            print(f"Full error details: {e}")
            return None

    def kmedoids_clustering(self, data, n_clusters, distance_metric, max_iter):
        try:
            np.random.seed(42)

            n_samples = data.shape[0]

            metric_mapping = {
                'euclidean': 'euclidean',
                'manhattan': 'cityblock'
            }

            metric = metric_mapping.get(distance_metric, 'euclidean')

            medoid_indices = np.random.choice(
                n_samples, n_clusters, replace=False)
            medoids = data[medoid_indices]

            for iteration in range(max_iter):
                distances = cdist(data, medoids, metric=metric)
                labels = np.argmin(distances, axis=1)

                new_medoid_indices = []
                cost_improved = False

                for cluster_id in range(n_clusters):
                    cluster_points = data[labels == cluster_id]
                    cluster_indices = np.where(labels == cluster_id)[0]

                    if len(cluster_points) == 0:
                        new_medoid_indices.append(medoid_indices[cluster_id])
                        continue

                    current_medoid_idx = medoid_indices[cluster_id]
                    current_cost = np.sum(
                        cdist([data[current_medoid_idx]], cluster_points, metric=metric))

                    best_medoid_idx = current_medoid_idx
                    best_cost = current_cost

                    for point_idx in cluster_indices:
                        cost = np.sum(
                            cdist([data[point_idx]], cluster_points, metric=metric))
                        if cost < best_cost:
                            best_cost = cost
                            best_medoid_idx = point_idx
                            cost_improved = True

                    new_medoid_indices.append(best_medoid_idx)

                medoid_indices = np.array(new_medoid_indices)
                new_medoids = data[medoid_indices]

                if not cost_improved or np.array_equal(medoids, new_medoids):
                    break

                medoids = new_medoids

            distances = cdist(data, medoids, metric=metric)
            final_labels = np.argmin(distances, axis=1)

            total_inertia = 0
            for i, label in enumerate(final_labels):
                total_inertia += distances[i, label]

            return {
                'labels': final_labels,
                'medoids': medoids,
                'medoid_indices': medoid_indices,
                'n_clusters': n_clusters,
                'algorithm': 'K-Medoids',
                'distance_metric': distance_metric,
                'max_iter': max_iter,
                'n_points': len(data),
                'inertia': total_inertia,
                'iterations_run': iteration + 1
            }

        except Exception as e:
            return {'error': str(e)}

    def apply_hierarchical_algorithm(self, algorithm_name, data, params):
        """AGNES, DIANA"""
        try:
            n_clusters = params.get('n_clusters', 3)
            distance_metric = params.get('distance_metric', 'euclidean')
            linkage_method = params.get('linkage', 'single')

            # Calculate distances and linkage
            if distance_metric == 'manhattan':
                distances = pdist(data, metric='cityblock')
            else:
                distances = pdist(data, metric='euclidean')

            linkage_matrix = linkage(distances, method=linkage_method)

            return {
                'n_clusters': n_clusters,
                'linkage_matrix': linkage_matrix,
                'algorithm': algorithm_name,
                'distance_metric': distance_metric,
                'linkage_method': linkage_method
            }
        except Exception as e:
            print(f"Error in {algorithm_name}: {str(e)}")
            return None

    def apply_density_algorithm(self, algorithm_name, data, params):
        """DBSCAN"""
        try:
            eps = params.get('eps', 0.5)
            min_samples = params.get('min_samples', 5)

            if algorithm_name == "DBSCAN":
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(data)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)

                return {
                    'data': data,
                    'labels': labels,
                    'algorithm': algorithm_name,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'eps': eps,
                    'min_samples': min_samples
                }
        except Exception as e:
            print(f"Error in {algorithm_name}: {str(e)}")
            return None
