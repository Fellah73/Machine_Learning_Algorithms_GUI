import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage, fcluster
from scipy.spatial.distance import pdist, cdist
from sklearn.preprocessing import StandardScaler


class UnsupervisedComparisonFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#f0f0f0")
        self.controller = controller
        self.comparison_algorithms = ['K-Means',
                                      'K-Medoids', 'AGNES', 'DIANA', 'DBSCAN']
        self.comparison_results = {}
        self.setup_ui()
        self.auto_run_comparison()

    def setup_ui(self):
        # Title
        title_label = tk.Label(
            self,
            text="Algorithm Comparison Interface",
            bg="#f0f0f0",
            fg="#24367E",
            font=("Arial", 13, "bold")
        )
        title_label.pack(pady=2)

        # Main container frame
        main_frame = tk.Frame(self, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=2)

        # Grid configuration
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # Left section - same type comparison
        left_section = tk.Frame(main_frame, bd=2)
        left_section.grid(row=0, column=0, sticky="nsew", padx=(0, 3))

        # Left metrics table frame
        left_metrics_frame = tk.Frame(
            left_section, bg="white", relief="sunken", bd=2)
        left_metrics_frame.pack(fill=tk.X, padx=10, pady=5)

        left_metrics_title = tk.Label(left_metrics_frame, text="Silhouette Score Comparison",
                                      bg="white", fg="#24367E", font=("Arial", 10, "bold"))
        left_metrics_title.pack(pady=2)

        # Left treeview (table) for metrics
        self.left_treeview = ttk.Treeview(left_metrics_frame, columns=(
            'My Algorithm', 'Algorithm2'), height=1, show='headings')
        self.left_treeview.pack(fill=tk.X, padx=10, pady=5)

        self.left_treeview.heading('My Algorithm', text='My Algorithm')
        self.left_treeview.heading('Algorithm2', text='Algorithm 2')
        self.left_treeview.column('My Algorithm', width=120)
        self.left_treeview.column('Algorithm2', width=120)

        # Left plots frame
        left_plots_frame = tk.Frame(
            left_section, bg="white", relief="sunken", bd=2)
        left_plots_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        left_plots_title = tk.Label(left_plots_frame, text="Visual Comparison",
                                    bg="white", fg="#24367E", font=("Arial", 10, "bold"))
        left_plots_title.pack(pady=3)

        # Left plot area
        left_single_plot_frame = tk.Frame(
            left_plots_frame, bg="white", relief="groove", bd=1)
        left_single_plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        self.left_single_plot_canvas = tk.Frame(
            left_single_plot_frame, bg="white")
        self.left_single_plot_canvas.pack(fill=tk.BOTH, expand=True)

        # Right section - all algorithms comparison
        right_section = tk.Frame(main_frame, bd=2)
        right_section.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        right_title_label = tk.Label(right_section, text="All Algorithms Comparison",
                                     bg="#f8f9fa", fg="#24367E", font=("Arial", 11, "bold"))
        right_title_label.pack(pady=3)

        # Right metrics table frame
        right_metrics_frame = tk.Frame(
            right_section, bg="white", relief="sunken", bd=2)
        right_metrics_frame.pack(fill=tk.X, padx=10, pady=3)

        right_metrics_title = tk.Label(right_metrics_frame, text="Silhouette Score Comparison",
                                       bg="white", fg="#24367E", font=("Arial", 10, "bold"))
        right_metrics_title.pack(pady=2)

        # Right treeview (table) for metrics
        self.right_treeview = ttk.Treeview(right_metrics_frame, columns=(
            'MyGlobalAlgorithm', 'algorithm2', 'algorithm3', 'algorithm4'), height=1, show='headings')
        self.right_treeview.pack(fill=tk.X, padx=10, pady=5)

        self.right_treeview.heading('MyGlobalAlgorithm', text='My Algorithm')
        self.right_treeview.heading('algorithm2', text='Algorithm 2')
        self.right_treeview.heading('algorithm3', text='Algorithm 3')
        self.right_treeview.heading('algorithm4', text='Algorithm 4')
        self.right_treeview.column('MyGlobalAlgorithm', width=80)
        self.right_treeview.column('algorithm2', width=80)
        self.right_treeview.column('algorithm3', width=80)
        self.right_treeview.column('algorithm4', width=80)

        # Right plots frame
        right_plots_frame = tk.Frame(
            right_section, bg="white", relief="sunken", bd=2)
        right_plots_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        right_plots_title = tk.Label(right_plots_frame, text="All Algorithms Visualization",
                                     bg="white", fg="#24367E", font=("Arial", 10, "bold"))
        right_plots_title.pack(pady=2)

        # Right plots grid
        right_plots_grid = tk.Frame(right_plots_frame, bg="white")
        right_plots_grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        right_plots_grid.grid_columnconfigure(0, weight=1)
        right_plots_grid.grid_columnconfigure(1, weight=1)
        right_plots_grid.grid_rowconfigure(0, weight=1)
        right_plots_grid.grid_rowconfigure(1, weight=1)

        # Create plot frames for right section
        self.right_plot_frames = []
        self.right_plot_labels = []

        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        for i, (row, col) in enumerate(positions):
            plot_frame = tk.Frame(
                right_plots_grid, bg="white", relief="groove", bd=1)
            plot_frame.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)

            plot_label = tk.Label(plot_frame, text=f"Algorithm {i+1}",
                                  bg="white", fg="#24367E", font=("Arial", 9, "bold"))
            plot_label.pack()

            plot_canvas = tk.Frame(plot_frame, bg="white", height=80)
            plot_canvas.pack(fill=tk.BOTH, expand=True)

            self.right_plot_frames.append(plot_canvas)
            self.right_plot_labels.append(plot_label)

    def prepare_data_for_clustering(self):
     dataset = self.controller.dataset_loader.data
     if dataset is None:
         return None, "No dataset loaded"

     last_column = dataset.columns[-1]
     features = dataset.drop(last_column, axis=1)
     numeric_features = features.select_dtypes(include=[np.number])

     if len(numeric_features.columns) == 0:
        return None, "No numeric features found"

     # AJOUT DE LA NORMALISATION
     data = numeric_features.values
     scaler = StandardScaler()
     normalized_data = scaler.fit_transform(data)
     return normalized_data, None
    
    
    def get_parameter_values(self):
        optimal_k = getattr(self.controller, 'optimal_k', None) or 3

        selected_algorithm = self.controller.get_selected_algorithm()

        user_params = getattr(self.controller, 'algorithm_parameters', {})

        return {
            'n_clusters': user_params.get('n_clusters', optimal_k),
            'distance_metric': user_params.get('distance_metric', 'euclidean'),
            'linkage': user_params.get('linkage_method', 'single'),
            'max_iter': user_params.get('max_iter', 300),
            'eps': user_params.get('eps', 1.3),
            'min_samples': user_params.get('min_samples', 2)
        }

    # Algorithm implementations
    def kmeans_clustering(self, data, n_clusters, distance_metric, max_iter):
        try:
            kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter,
                            random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)

            return {
                'labels': labels,
                'centers': kmeans.cluster_centers_,
                'n_clusters': n_clusters,
                'algorithm': 'K-Means',
                'distance_metric': distance_metric,
                'max_iter': max_iter,
                'n_points': len(data),
                'inertia': kmeans.inertia_
            }
        except Exception as e:
            return {'error': str(e)}

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

    def agnes_clustering(self, data, n_clusters, linkage_method, distance_metric):
        try:
            distance_mapping = {
                'euclidean': 'euclidean',
                'manhattan': 'cityblock',
            }

            scipy_distance = distance_mapping.get(distance_metric, 'euclidean')
            distances = pdist(data, metric=scipy_distance)
            linkage_matrix = scipy_linkage(distances, method=linkage_method)
            labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

            return {
                'labels': labels - 1,
                'linkage_matrix': linkage_matrix,
                'n_clusters': n_clusters,
                'algorithm': 'AGNES',
                'linkage_method': linkage_method,
                'distance_metric': distance_metric,
                'n_points': len(data)
            }
        except Exception as e:
            return {'error': str(e)}

    def diana_clustering(self, data, n_clusters, distance_metric):
        try:
            distance_mapping = {
                'euclidean': 'euclidean',
                'manhattan': 'cityblock',
            }

            scipy_distance = distance_mapping.get(distance_metric, 'euclidean')
            distances = pdist(data, metric=scipy_distance)
            linkage_matrix = scipy_linkage(distances, method='complete')
            labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

            return {
                'labels': labels - 1,
                'linkage_matrix': linkage_matrix,
                'n_clusters': n_clusters,
                'algorithm': 'DIANA',
                'distance_metric': distance_metric,
                'n_points': len(data)
            }
        except Exception as e:
            return {'error': str(e)}

    def dbscan_clustering(self, data, eps, min_samples):
        try:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            n_core = len(dbscan.core_sample_indices_)
            n_border = len(data) - n_core - n_noise

            return {
                'labels': labels,
                'core_sample_indices': dbscan.core_sample_indices_,
                'n_clusters': n_clusters,
                'algorithm': 'DBSCAN',
                'eps': eps,
                'min_samples': min_samples,
                'n_points': len(data),
                'n_core_points': n_core,
                'n_border_points': n_border,
                'n_noise_points': n_noise
            }
        except Exception as e:
            return {'error': str(e)}

    # Comparison logic
    def calculate_silhouette_scores(self):
        data, error = self.prepare_data_for_clustering()
        if data is None or error:
            return None

        self.comparison_results = {}
        params = self.get_parameter_values()

        k_clusters = params['n_clusters']
        distance_metric = params['distance_metric']
        linkage_method = params['linkage']
        n_iterations = params['max_iter']
        eps = params['eps']
        min_samples = params['min_samples']

        for algo in self.comparison_algorithms:
            try:
                if algo == 'K-Means':
                    result = self.kmeans_clustering(
                        data, k_clusters, distance_metric, n_iterations)
                elif algo == 'K-Medoids':
                    result = self.kmedoids_clustering(
                        data, k_clusters, distance_metric, n_iterations)
                elif algo == 'AGNES':
                    result = self.agnes_clustering(
                        data, k_clusters, linkage_method, distance_metric)
                elif algo == 'DIANA':
                    result = self.diana_clustering(
                        data, k_clusters, distance_metric)
                elif algo == 'DBSCAN':
                    result = self.dbscan_clustering(data, eps, min_samples)

                if 'error' not in result:
                    if algo == 'DBSCAN' and result['n_clusters'] < 2:
                        self.comparison_results[algo] = {
                            'silhouette': None, 'result': result}
                    else:
                        sil_score = silhouette_score(data, result['labels'])
                        self.comparison_results[algo] = {
                            'silhouette': sil_score, 'result': result}
                else:
                    self.comparison_results[algo] = {
                        'silhouette': None, 'result': None}

            except Exception as e:
                self.comparison_results[algo] = {
                    'silhouette': None, 'result': None}

        return self.comparison_results

    def get_same_type_algorithms(self):
        selected_algorithm = self.controller.get_selected_algorithm()
        algorithm_type = self.controller.get_algorithm_type()

        if algorithm_type == "Partitioning":
            return ['K-Means', 'K-Medoids']
        elif algorithm_type == "Hierarchical":
            return ['AGNES', 'DIANA']
        else:
            return ['DBSCAN']

    def update_comparison_tables(self):
        selected_algorithm = self.controller.get_selected_algorithm()

        if not selected_algorithm:
            return

        scores = self.calculate_silhouette_scores()
        if not scores:
            return

        same_type_algos = self.get_same_type_algorithms()
        self.left_treeview.delete(*self.left_treeview.get_children())

        if selected_algorithm != 'DBSCAN' and len(same_type_algos) > 1:
            other_algo = [a for a in same_type_algos if a !=
                          selected_algorithm][0]

            self.left_treeview.heading('#1', text=selected_algorithm)
            self.left_treeview.heading('#2', text=other_algo)

            my_score = scores.get(selected_algorithm, {}
                                  ).get('silhouette', None)
            other_score = scores.get(other_algo, {}).get('silhouette', None)

            my_score_str = f"{my_score:.3f}" if (
                my_score is not None and isinstance(my_score, (int, float))) else 'N/A'
            other_score_str = f"{other_score:.3f}" if (
                other_score is not None and isinstance(other_score, (int, float))) else 'N/A'

            item = self.left_treeview.insert(
                '', 'end', values=(my_score_str, other_score_str))
            self.left_treeview.item(item, tags=('selected',))

        self.right_treeview.delete(*self.right_treeview.get_children())

        other_algos = [
            a for a in self.comparison_algorithms if a != selected_algorithm][:3]

        self.right_treeview.heading('#1', text=selected_algorithm)

        for i, algo in enumerate(other_algos):
            if i < 3:
                self.right_treeview.heading(f'#{i+2}', text=algo)

        for i in range(len(other_algos) + 1, 4):
            if i < 4:
                self.right_treeview.heading(f'#{i+1}', text='')

        values = []
        my_score = scores.get(selected_algorithm, {}).get('silhouette', None)
        values.append(f"{my_score:.3f}" if (my_score is not None and isinstance(
            my_score, (int, float)) and -1 <= my_score <= 1) else 'N/A')

        for algo in other_algos:
            score = scores.get(algo, {}).get('silhouette', None)
            values.append(f"{score:.3f}" if (score is not None and isinstance(
                score, (int, float)) and -1 <= score <= 1) else 'N/A')

        while len(values) < 4:
            values.append('')
        values = values[:4]

        item = self.right_treeview.insert('', 'end', values=values)
        self.right_treeview.item(item, tags=('selected',))

    # Plotting methods
    def plot_algorithm_comparison(self, algorithm_name, canvas_frame, size=(2, 1)):
        for widget in canvas_frame.winfo_children():
            widget.destroy()

        if algorithm_name not in self.comparison_results or not self.comparison_results[algorithm_name]['result']:
            placeholder = tk.Label(canvas_frame, text=f"{algorithm_name}\nNo Data",
                                   bg="white", fg="#666", font=("Arial", 8))
            placeholder.pack(expand=True)
            return

        result = self.comparison_results[algorithm_name]['result']
        data, _ = self.prepare_data_for_clustering()

        if data is None:
            return

        fig, ax = plt.subplots(figsize=size)

        if algorithm_name in ['AGNES', 'DIANA']:
            try:
                dend = dendrogram(result['linkage_matrix'], ax=ax,
                                  truncate_mode='lastp', p=10,
                                  show_leaf_counts=False)
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_title(algorithm_name, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            except Exception as e:
                ax.text(0.5, 0.5, f"{algorithm_name}\nDendrogram Error",
                        ha='center', va='center', transform=ax.transAxes)
        else:
            if data.shape[1] >= 2:
                x_data = data[:, 0]
                y_data = data[:, 1]
            else:
                x_data = data[:, 0]
                y_data = np.zeros(len(data))

            if algorithm_name == 'DBSCAN':
                labels = result['labels']
                unique_labels = set(labels)
                colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

                for k, col in zip(unique_labels, colors):
                    if k == -1:
                        col = [0, 0, 0, 1]
                    class_member_mask = (labels == k)
                    xy = np.column_stack(
                        [x_data[class_member_mask], y_data[class_member_mask]])
                    if len(xy) > 0:
                        ax.scatter(xy[:, 0], xy[:, 1], c=[
                                   col], s=20, alpha=0.7)
            else:
                scatter = ax.scatter(
                    x_data, y_data, c=result['labels'], cmap='viridis', s=20, alpha=0.7)

                if 'centers' in result:
                    centers = result['centers']
                    if centers.shape[1] >= 2:
                        ax.scatter(centers[:, 0], centers[:, 1],
                                   c='red', marker='x', s=100, linewidths=2)
                    else:
                        ax.scatter(centers[:, 0], np.zeros(
                            len(centers)), c='red', marker='x', s=100, linewidths=2)
                elif 'medoids' in result:
                    medoids = result['medoids']
                    if medoids.shape[1] >= 2:
                        ax.scatter(medoids[:, 0], medoids[:, 1],
                                   c='red', marker='s', s=100, linewidths=2)
                    else:
                        ax.scatter(medoids[:, 0], np.zeros(
                            len(medoids)), c='red', marker='s', s=100, linewidths=2)

            ax.set_title(algorithm_name, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_partitioning_inertia_comparison(self):
        for widget in self.left_single_plot_canvas.winfo_children():
            widget.destroy()

        kmeans_inertia = None
        kmedoids_inertia = None
        selected_algorithm = self.controller.get_selected_algorithm()

        if 'K-Means' in self.comparison_results and self.comparison_results['K-Means']['result']:
            kmeans_result = self.comparison_results['K-Means']['result']
            if 'inertia' in kmeans_result:
                kmeans_inertia = kmeans_result['inertia']

        if 'K-Medoids' in self.comparison_results and self.comparison_results['K-Medoids']['result']:
            kmedoids_result = self.comparison_results['K-Medoids']['result']
            if 'inertia' in kmedoids_result:
                kmedoids_inertia = kmedoids_result['inertia']

        fig, ax = plt.subplots(figsize=(4, 2))

        algorithms = []
        inertias = []
        colors = []

        if kmeans_inertia is not None:
            algorithms.append('K-Means')
            inertias.append(kmeans_inertia)
            colors.append('#1f77b4' if selected_algorithm ==
                          'K-Means' else '#aec7e8')

        if kmedoids_inertia is not None:
            algorithms.append('K-Medoids')
            inertias.append(kmedoids_inertia)
            colors.append('#ff7f0e' if selected_algorithm ==
                          'K-Medoids' else '#ffbb78')

        if len(algorithms) > 0:
            bars = ax.bar(algorithms, inertias, color=colors,
                          alpha=0.8, edgecolor='black')

            for bar, inertia in zip(bars, inertias):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{inertia:.2f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

            if selected_algorithm in algorithms:
                selected_idx = algorithms.index(selected_algorithm)
                bars[selected_idx].set_edgecolor('red')
                bars[selected_idx].set_linewidth(3)

            ax.set_title('Inertia Comparison', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.left_single_plot_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_hierarchical_dendrograms_comparison(self):
        for widget in self.left_single_plot_canvas.winfo_children():
            widget.destroy()

        selected_algorithm = self.controller.get_selected_algorithm()
        agnes_result = self.comparison_results.get('AGNES', {}).get('result')
        diana_result = self.comparison_results.get('DIANA', {}).get('result')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))

        if agnes_result and 'linkage_matrix' in agnes_result:
            try:
                dend1 = dendrogram(agnes_result['linkage_matrix'], ax=ax1,
                                   truncate_mode='lastp', p=15,
                                   show_leaf_counts=False)
                ax1.set_title('AGNES', fontsize=10, fontweight='bold')

                if selected_algorithm == 'AGNES':
                    ax1.set_facecolor('#ffe6e6')
                    for spine in ax1.spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(2)
            except Exception as e:
                ax1.text(0.5, 0.5, 'AGNES\nDendrogram Error',
                         ha='center', va='center', transform=ax1.transAxes, fontsize=10)
        else:
            ax1.text(0.5, 0.5, 'AGNES\nNo Data',
                     ha='center', va='center', transform=ax1.transAxes, fontsize=10)

        if diana_result and 'linkage_matrix' in diana_result:
            try:
                dend2 = dendrogram(diana_result['linkage_matrix'], ax=ax2,
                                   truncate_mode='lastp', p=15,
                                   show_leaf_counts=False)
                ax2.set_title('DIANA', fontsize=10, fontweight='bold')

                if selected_algorithm == 'DIANA':
                    ax2.set_facecolor('#ffe6e6')
                    for spine in ax2.spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(2)
            except Exception as e:
                ax2.text(0.5, 0.5, 'DIANA\nDendrogram Error',
                         ha='center', va='center', transform=ax2.transAxes, fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'DIANA\nNo Data',
                     ha='center', va='center', transform=ax2.transAxes, fontsize=10)

        fig.suptitle('Hierarchical Algorithms Comparison',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.left_single_plot_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def plot_dbscan_single(self):
        for widget in self.left_single_plot_canvas.winfo_children():
            widget.destroy()

        dbscan_result = self.comparison_results.get('DBSCAN', {}).get('result')

        if not dbscan_result:
            placeholder = tk.Label(self.left_single_plot_canvas, text="DBSCAN\nNo Data Available",
                                   bg="white", fg="#666", font=("Arial", 12))
            placeholder.pack(expand=True)
            return

        data, _ = self.prepare_data_for_clustering()
        if data is None:
            return

        fig, ax = plt.subplots(figsize=(4, 3))

        if data.shape[1] >= 2:
            x_data = data[:, 0]
            y_data = data[:, 1]
        else:
            x_data = data[:, 0]
            y_data = np.zeros(len(data))

        labels = dbscan_result['labels']
        core_samples_mask = np.zeros_like(labels, dtype=bool)
        core_samples_mask[dbscan_result['core_sample_indices']] = True

        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy_core = np.column_stack([x_data[class_member_mask & core_samples_mask],
                                       y_data[class_member_mask & core_samples_mask]])
            if len(xy_core) > 0:
                if k == -1:
                    ax.scatter(xy_core[:, 0], xy_core[:, 1], s=50, c=[col], marker='x',
                               alpha=0.8, label='Noise' if k == -1 else f'Core {k}')
                else:
                    ax.scatter(xy_core[:, 0], xy_core[:, 1], s=80, c=[col], marker='o',
                               edgecolors='black', linewidths=1, alpha=0.8)

            xy_border = np.column_stack([x_data[class_member_mask & ~core_samples_mask],
                                         y_data[class_member_mask & ~core_samples_mask]])
            if len(xy_border) > 0 and k != -1:
                ax.scatter(xy_border[:, 0], xy_border[:, 1], s=40, c=[col], marker='o',
                           alpha=0.6, edgecolors='black', linewidths=0.5)

        ax.grid(True, alpha=0.3)
        ax.set_title(f'DBSCAN Clustering (eps={dbscan_result["eps"]}, min_samples={dbscan_result["min_samples"]})',
                     fontsize=8, fontweight='bold')

        n_clusters = dbscan_result['n_clusters']
        n_noise = dbscan_result['n_noise_points']
        ax.text(0.02, 0.98, f'Clusters: {n_clusters}\nNoise Points: {n_noise}',
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.left_single_plot_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def update_left_single_plot(self):
        selected_algorithm = self.controller.get_selected_algorithm()
        algorithm_type = self.controller.get_algorithm_type()

        if not selected_algorithm:
            return

        if algorithm_type == "Partitioning":
            self.plot_partitioning_inertia_comparison()
        elif algorithm_type == "Hierarchical":
            self.plot_hierarchical_dendrograms_comparison()
        elif algorithm_type == "Density-based":
            self.plot_dbscan_single()
        else:
            for widget in self.left_single_plot_canvas.winfo_children():
                widget.destroy()
            placeholder = tk.Label(self.left_single_plot_canvas, text="No Algorithm Selected",
                                   bg="white", fg="#666", font=("Arial", 12))
            placeholder.pack(expand=True)

    def update_right_plots(self):
        selected_algorithm = self.controller.get_selected_algorithm()
        if not selected_algorithm:
            return

        other_algos = [
            a for a in self.comparison_algorithms if a != selected_algorithm]

        for i, (algo, frame, label) in enumerate(zip(other_algos, self.right_plot_frames, self.right_plot_labels)):
            if i < len(self.right_plot_frames):
                label.config(text=algo)
                self.plot_algorithm_comparison(algo, frame, (2.5, 1.5))

    def run_comparison_analysis(self):
        selected_algorithm = self.controller.get_selected_algorithm()
        if not selected_algorithm:
            return

        try:
            self.calculate_silhouette_scores()
            self.update_comparison_tables()
            self.update_left_single_plot()
            self.update_right_plots()
        except Exception as e:
            print(f"Error in comparison analysis: {e}")

    def auto_run_comparison(self):
        self.after(100, self.run_comparison_analysis)

    def on_previous_step(self):
        self.event_generate("<<PreviousStep>>")
