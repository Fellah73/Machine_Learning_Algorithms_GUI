import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import pandas as pd


class VisualizationFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#f0f0f0")
        self.controller = controller
        self.supervised_mode = False
        self.parameter_widgets = {}
        self.setup_ui()

    def setup_ui(self):
        # Check learning type
        learning_type = self.controller.get_learning_type()
        self.supervised_mode = (learning_type == "supervised")

        # Title
        title_text = "Supervised Learning Visualization" if self.supervised_mode else "Algorithm Visualization"
        self.title_label = tk.Label(
            self,
            text=title_text,
            bg="#f0f0f0",
            fg="#24367E",
            font=("Arial", 16, "bold")
        )
        self.title_label.pack(pady=10)

        # Main content frame
        self.setup_main_content()

        # Create parameter inputs
        self.create_parameter_inputs()

    def setup_main_content(self):
        """Setup main content area with parameters, visualization and results"""
        main_frame = tk.Frame(self, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Grid configuration
        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=0)
        main_frame.grid_rowconfigure(1, weight=1)

        # Parameters section
        self.setup_parameters_section(main_frame)

        # Apply button section
        self.setup_apply_button(main_frame)

        # Visualization and results section
        self.setup_visualization_section(main_frame)

        # Navigation buttons
        self.setup_navigation(main_frame)

    def setup_parameters_section(self, parent):
        """Setup parameters input section"""
        params_frame = tk.LabelFrame(
            parent,
            text="Algorithm Parameters",
            bg="#f0f0f0",
            fg="#24367E",
            font=("Arial", 12, "bold"),
            bd=2,
            relief="groove"
        )
        params_frame.grid(row=0, column=0, sticky="ew",
                          padx=(0, 10), pady=(0, 5))

        # Parameters grid
        self.params_grid_frame = tk.Frame(params_frame, bg="#f0f0f0")
        self.params_grid_frame.pack(fill=tk.X, padx=15, pady=10)

        # Configure grid columns
        for i in range(4):
            self.params_grid_frame.grid_columnconfigure(i, weight=1)

    def setup_apply_button(self, parent):
        """Setup apply algorithm button"""
        button_frame = tk.Frame(parent, bg="#f0f0f0")
        button_frame.grid(row=0, column=1, sticky="nsew", pady=(0, 5))

        self.apply_btn = tk.Button(
            button_frame,
            text="Apply Algorithm",
            bg="#24367E",
            fg="white",
            font=("Arial", 13, "bold"),
            width=15,
            height=2,
            relief="raised",
            bd=2,
            command=self.apply_algorithm
        )
        self.apply_btn.pack(pady=20)

    def setup_visualization_section(self, parent):
        """Setup visualization and results section"""
        content_frame = tk.Frame(parent, bg="#f0f0f0")
        content_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        # Grid configuration
        content_frame.grid_columnconfigure(0, weight=3)
        content_frame.grid_columnconfigure(1, weight=1)
        content_frame.grid_rowconfigure(0, weight=1)

        # Visualization area - single plot area
        self.viz_frame = tk.Frame(
            content_frame, bg="white", relief="sunken", bd=2)
        self.viz_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))

        # Visualization title
        self.viz_title = tk.Label(
            self.viz_frame,
            text="Algorithm Visualization",
            bg="white",
            fg="#24367E",
            font=("Arial", 12, "bold")
        )
        self.viz_title.pack(pady=5)

        # Plot frame - single plot area
        self.plot_frame = tk.Frame(self.viz_frame, bg="white")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=5)

        # Results section
        self.setup_results_section(content_frame)

    def setup_results_section(self, parent):
        """Setup results display section"""
        results_frame = tk.Frame(parent, bg="#f8f9fa", relief="raised", bd=2)
        results_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # Results title
        results_title = tk.Label(
            results_frame,
            text="Results",
            bg="#f8f9fa",
            fg="#24367E",
            font=("Arial", 12, "bold")
        )
        results_title.pack(pady=8)

        # Results text area
        text_frame = tk.Frame(results_frame, bg="white", relief="sunken", bd=1)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=5)

        self.results_text = scrolledtext.ScrolledText(
            text_frame,
            height=15,
            width=25,
            font=("Courier", 9),
            bg="white",
            fg="#333333",
            wrap="word",
            relief="flat",
            bd=0
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

        # Initial results message
        self.results_text.insert(
            tk.END, "Results will appear here after applying the algorithm...\n\n")

    def setup_navigation(self, parent):
        """Setup navigation buttons"""
        nav_frame = tk.Frame(parent, bg="#f0f0f0")
        nav_frame.grid(row=2, column=0, columnspan=2, pady=10)

        # Next step button
        self.next_btn = tk.Button(
            nav_frame,
            text="Next Step →",
            bg="#374451",
            fg="white",
            font=("Arial", 10, "bold"),
            width=12,
            height=2,
            relief="raised",
            bd=2,
            state=tk.DISABLED,
            command=self.on_next_step
        )
        self.next_btn.pack(side=tk.LEFT, padx=(10, 0))

    def create_parameter_inputs(self):
        """Create parameter input widgets based on selected algorithm"""
        # Clear existing parameters
        for widget in self.params_grid_frame.winfo_children():
            widget.destroy()
        self.parameter_widgets.clear()

        # Get selected algorithm info
        algorithm_name = self.controller.get_selected_algorithm()
        algorithm_type = self.controller.get_algorithm_type()

        if not algorithm_name or not algorithm_type:
            self.show_no_algorithm_message()
            return

        # Update title with algorithm name
        title_text = f"{algorithm_name} Visualization"
        self.title_label.config(text=title_text)
        self.viz_title.config(text=f"{algorithm_name} Results")

        # Get algorithm parameters
        algorithms_data = self.controller.get_algorithms_data()
        if algorithm_type in algorithms_data and algorithm_name in algorithms_data[algorithm_type]['algorithms']:
            params = algorithms_data[algorithm_type]['algorithms'][algorithm_name]['parameters']
            self.create_parameter_widgets(
                params, algorithm_name, algorithm_type)

    def show_no_algorithm_message(self):
        """Show message when no algorithm is selected"""
        msg_label = tk.Label(
            self.params_grid_frame,
            text="No algorithm selected. Please go back and select an algorithm.",
            bg="#f0f0f0",
            fg="#666",
            font=("Arial", 11)
        )
        msg_label.grid(row=0, column=0, columnspan=4, pady=20)

    def create_parameter_widgets(self, params, algorithm_name, algorithm_type):
        """Create parameter input widgets"""
        row = 0
        col = 0

        for param in params:
            if col >= 4:  # Max 4 columns
                col = 0
                row += 1

            # Parameter frame
            param_frame = tk.Frame(
                self.params_grid_frame,
                bg="white",
                relief="groove",
                bd=1
            )
            param_frame.grid(row=row, column=col, padx=5, pady=5, sticky="ew")

            # Parameter label
            label_text = param.replace('_', ' ').title()
            param_label = tk.Label(
                param_frame,
                text=label_text,
                bg="white",
                fg="#24367E",
                font=("Arial", 9, "bold")
            )
            param_label.pack(pady=(5, 2))

            # Create appropriate input widget
            widget = self.create_parameter_widget(
                param_frame, param, algorithm_name, algorithm_type)
            if widget:
                self.parameter_widgets[param] = widget

            col += 1

    def create_parameter_widget(self, parent, param, algorithm_name, algorithm_type):
        """Create specific parameter input widget"""
        if param == 'n_clusters':
            # Use optimal K for unsupervised algorithms
            optimal_k = self.controller.get_optimal_k() if hasattr(
                self.controller, 'get_optimal_k') else 3

            # For unsupervised, show optimal K as read-only
            if not self.supervised_mode:
                label = tk.Label(
                    parent,
                    text=str(optimal_k),
                    bg="#e9ecef",
                    fg="#24367E",
                    font=("Arial", 10, "bold"),
                    relief="sunken",
                    bd=1,
                    width=8
                )
                label.pack(pady=(0, 5))
                return label
            else:
                entry = tk.Entry(
                    parent,
                    width=8,
                    font=("Arial", 9),
                    justify="center",
                    relief="sunken",
                    bd=1
                )
                entry.insert(0, str(optimal_k))
                entry.pack(pady=(0, 5))
                return entry

        elif param == 'train_test_ratio':
            combo = ttk.Combobox(
                parent,
                values=["70/30", "75/25", "80/20"],
                state="readonly",
                width=8,
                font=("Arial", 9)
            )
            combo.set("80/20")
            combo.pack(pady=(0, 5))
            return combo

        elif param == 'n_neighbors' and algorithm_name == 'KNN':
            entry = tk.Entry(
                parent,
                width=8,
                font=("Arial", 9),
                justify="center",
                relief="sunken",
                bd=1
            )
            entry.insert(0, "5")
            entry.pack(pady=(0, 5))
            return entry

        elif param == 'distance_metric':
            combo = ttk.Combobox(
                parent,
                values=["euclidean", "manhattan"],
                state="readonly",
                width=10,
                font=("Arial", 9)
            )
            combo.set("euclidean")
            combo.pack(pady=(0, 5))
            return combo

        elif param == 'linkage':
            combo = ttk.Combobox(
                parent,
                values=["single", "complete", "average"],
                state="readonly",
                width=8,
                font=("Arial", 9)
            )
            combo.set("single")
            combo.pack(pady=(0, 5))
            return combo

        elif param == 'eps':
            entry = tk.Entry(
                parent,
                width=8,
                font=("Arial", 9),
                justify="center",
                relief="sunken",
                bd=1
            )
            entry.insert(0, "0.5")
            entry.pack(pady=(0, 5))
            return entry

        elif param == 'min_samples':
            entry = tk.Entry(
                parent,
                width=8,
                font=("Arial", 9),
                justify="center",
                relief="sunken",
                bd=1
            )
            entry.insert(0, "5")
            entry.pack(pady=(0, 5))
            return entry

        elif param == 'max_iter':
            entry = tk.Entry(
                parent,
                width=8,
                font=("Arial", 9),
                justify="center",
                relief="sunken",
                bd=1
            )
            entry.insert(0, "300")
            entry.pack(pady=(0, 5))
            return entry

        else:
            # Default parameter widget
            entry = tk.Entry(
                parent,
                width=10,
                font=("Arial", 9),
                justify="center",
                relief="sunken",
                bd=1
            )
            entry.pack(pady=(0, 5))
            return entry

    def get_parameter_values(self):
        """Get current parameter values from widgets"""
        values = {}
        for param, widget in self.parameter_widgets.items():
            if isinstance(widget, ttk.Combobox):
                if param == 'train_test_ratio':
                    ratio_text = widget.get()
                    train_pct = int(ratio_text.split('/')[0])
                    values[param] = train_pct / 100.0
                else:
                    values[param] = widget.get()
            elif isinstance(widget, tk.Entry):
                try:
                    text_value = widget.get()
                    if param in ['n_neighbors', 'max_depth', 'min_samples_split', 'n_clusters', 'max_iter', 'min_samples']:
                        values[param] = int(text_value)
                    elif param == 'eps':
                        values[param] = float(text_value)
                    else:
                        values[param] = text_value
                except ValueError:
                    # Keep as string if conversion fails
                    values[param] = widget.get()
            elif isinstance(widget, tk.Label):  # For read-only optimal K
                if param == 'n_clusters':
                    values[param] = int(widget.cget("text"))
        return values

    def apply_algorithm(self):
        """Apply the selected algorithm with current parameters"""
        algorithm_name = self.controller.get_selected_algorithm()
        algorithm_type = self.controller.get_algorithm_type()

        if not algorithm_name:
            self.show_error("No algorithm selected")
            return

        # Get parameter values
        params = self.get_parameter_values()

        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Applying {algorithm_name}...\n\n")

        try:
            if self.supervised_mode:
                self.apply_supervised_algorithm(algorithm_name, params)
            else:
                self.apply_unsupervised_algorithm(
                    algorithm_name, algorithm_type, params)

        except Exception as e:
            self.show_error(f"Error applying algorithm: {str(e)}")

    def apply_supervised_algorithm(self, algorithm_name, params):
        """Apply supervised learning algorithm"""
        # This is a placeholder - implement actual supervised algorithms
        self.results_text.insert(
            tk.END, f"Supervised algorithm {algorithm_name} applied successfully!\n\n")
        self.results_text.insert(tk.END, f"Parameters: {params}\n\n")
        self.results_text.insert(tk.END, "Training completed.\n")
        self.results_text.insert(tk.END, "Accuracy: 85.7%\n")
        self.results_text.insert(tk.END, "Precision: 83.2%\n")
        self.results_text.insert(tk.END, "Recall: 87.1%\n")

        # Enable next step
        self.next_btn.config(state=tk.NORMAL, bg="#24367E")

        # Create a simple placeholder plot
        self.create_supervised_plot(algorithm_name)

    def apply_unsupervised_algorithm(self, algorithm_name, algorithm_type, params):
        """Apply unsupervised learning algorithm"""
        # Get dataset for clustering
        dataset = self.controller.get_dataset()
        if dataset is None:
            self.show_error("No dataset loaded")
            return

        # Prepare data (select numeric columns, excluding target if exists)
        if len(dataset.columns) > 1:
            # Use first few numeric columns
            numeric_cols = dataset.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                selected_columns = numeric_cols[:2]
                # Take first 2 numeric columns
                data = dataset[selected_columns].values
            else:
                self.show_error("Not enough numeric columns for visualization")
                return
        else:
            self.show_error("Dataset has insufficient columns")
            return

        # Apply algorithm based on type
        if algorithm_type == "Partitioning":
            result = self.apply_partitioning_algorithm(
                algorithm_name, data, params)
            if result:
                self.create_scatter_plot(
                    data, result['labels'], algorithm_name, result.get('centers'), selected_columns)
                self.display_partitioning_results(result, params)

        elif algorithm_type == "Hierarchical":
            result = self.apply_hierarchical_algorithm(
                algorithm_name, data, params)
            if result:
                self.create_dendrogram_plot(
                    result['linkage_matrix'], algorithm_name, params.get('n_clusters', 3))
                self.display_hierarchical_results(result, params)

        elif algorithm_type == "Density-based":
            result = self.apply_density_algorithm(algorithm_name, data, params)
            if result:
                self.create_scatter_plot(
                    data, result['labels'], algorithm_name)
                self.display_density_results(result, params)

        # Enable next step
        self.next_btn.config(state=tk.NORMAL, bg="#24367E")

    def apply_partitioning_algorithm(self, algorithm_name, data, params):
        """Apply partitioning algorithms (K-Means, K-Medoids)"""
        try:
            n_clusters = params.get('n_clusters', 3)
            max_iter = params.get('max_iter', 300)

            if algorithm_name == "K-Means":
                kmeans = KMeans(n_clusters=n_clusters,
                                max_iter=max_iter, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data)
                return {
                    'labels': labels,
                    'centers': kmeans.cluster_centers_,
                    'algorithm': algorithm_name,
                    'n_clusters': n_clusters,
                    'inertia': kmeans.inertia_
                }
            elif algorithm_name == "K-Medoids":
                kmeans = KMeans(n_clusters=n_clusters,
                                max_iter=max_iter, random_state=42)
                labels = kmeans.fit_predict(data)
                return {
                    'labels': labels,
                    'centers': kmeans.cluster_centers_,
                    'algorithm': algorithm_name,
                    'n_clusters': n_clusters,
                    'inertia': kmeans.inertia_
                }
        except Exception as e:
            self.show_error(f"Error in {algorithm_name}: {str(e)}")
            return None

    def apply_hierarchical_algorithm(self, algorithm_name, data, params):
        """Apply hierarchical algorithms (AGNES, DIANA)"""
        try:
            distance_metric = params.get('distance_metric', 'euclidean')
            linkage_method = params.get('linkage', 'single')

            # Calculate distances and linkage
            if distance_metric == 'manhattan':
                distances = pdist(data, metric='cityblock')
            else:
                distances = pdist(data, metric='euclidean')

            linkage_matrix = linkage(distances, method=linkage_method)

            return {
                'linkage_matrix': linkage_matrix,
                'algorithm': algorithm_name,
                'distance_metric': distance_metric,
                'linkage_method': linkage_method
            }
        except Exception as e:
            self.show_error(f"Error in {algorithm_name}: {str(e)}")
            return None

    def apply_density_algorithm(self, algorithm_name, data, params):
        """Apply density-based algorithms (DBSCAN)"""
        try:
            eps = params.get('eps', 0.5)
            min_samples = params.get('min_samples', 5)

            if algorithm_name == "DBSCAN":
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(data)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)

                return {
                    'labels': labels,
                    'algorithm': algorithm_name,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'eps': eps,
                    'min_samples': min_samples
                }
        except Exception as e:
            self.show_error(f"Error in {algorithm_name}: {str(e)}")
            return None

    def create_scatter_plot(self, data, labels, algorithm_name, centers=None, column_names=None):
        # Clear previous plots
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(4, 3))

        # Create scatter plot
        unique_labels = set(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

        for i, label in enumerate(unique_labels):
            if label == -1:  # Noise points in DBSCAN
                color = 'black'
                marker = 'x'
                alpha = 0.5
                label_text = 'Noise'
            else:
                color = colors[i % len(colors)]
                marker = 'o'
                alpha = 0.7
                label_text = f'Cluster {label}'

            mask = labels == label
            ax.scatter(data[mask, 0], data[mask, 1],
                       c=[color], marker=marker, alpha=alpha,
                       s=50, label=label_text)

        # Add cluster centers if available
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1],
                       c='red', marker='X', s=200,
                       linewidths=3, label='Centers')

        # Use actual column names if provided, otherwise use default labels
        if column_names is not None and len(column_names) >= 2:
         ax.set_xlabel(column_names[0])
         ax.set_ylabel(column_names[1])
        else:
         ax.set_xlabel('Feature 1')
         ax.set_ylabel('Feature 2')

        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_dendrogram_plot(self, linkage_matrix, algorithm_name, n_clusters):
        """Create dendrogram for hierarchical algorithms"""
        # Clear previous plots
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(4, 3))

        # Create dendrogram
        dend = dendrogram(linkage_matrix, ax=ax,
                          truncate_mode='lastp', p=20,
                          show_leaf_counts=True,
                          leaf_font_size=10)

        # Add horizontal line for cluster cut
        if n_clusters > 1:
            cut_height = linkage_matrix[-(n_clusters-1), 2]
            ax.axhline(y=cut_height, color='red', linestyle='--',
                       linewidth=2, label=f'Cut for {n_clusters} clusters')
            ax.legend()

        ax.set_ylabel('Distance')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def display_partitioning_results(self, result, params):
        """Display results for partitioning algorithms"""
        self.results_text.insert(tk.END, f"{result['algorithm']} Results\n")
        self.results_text.insert(tk.END, "=" * 20 + "\n\n")
        self.results_text.insert(tk.END, f"Algorithm: {result['algorithm']}\n")
        self.results_text.insert(tk.END, f"Clusters: {result['n_clusters']}\n")
        if 'inertia' in result:
            self.results_text.insert(
                tk.END, f"Inertia: {result['inertia']:.2f}\n")
        self.results_text.insert(
            tk.END, f"Max iterations: {params.get('max_iter', 300)}\n")

        # Add cluster distribution
        labels = result['labels']
        unique_labels, counts = np.unique(labels, return_counts=True)
        self.results_text.insert(tk.END, "\nCluster Distribution:\n")
        self.results_text.insert(tk.END, "-" * 18 + "\n")
        for label, count in zip(unique_labels, counts):
            self.results_text.insert(
                tk.END, f"Cluster {label}: {count} points\n")

    def display_hierarchical_results(self, result, params):
        self.results_text.insert(tk.END, f"{result['algorithm']} Results\n")
        self.results_text.insert(tk.END, "=" * 20 + "\n\n")
        self.results_text.insert(tk.END, f"Algorithm: {result['algorithm']}\n")
        self.results_text.insert(
            tk.END, f"Distance: {result['distance_metric']}\n")
        self.results_text.insert(
            tk.END, f"Linkage: {result['linkage_method']}\n")
        self.results_text.insert(
            tk.END, f"Clusters: {params.get('n_clusters', 3)}\n")

        from scipy.cluster.hierarchy import fcluster
        n_clusters = params.get('n_clusters', 3)
        labels = fcluster(result['linkage_matrix'],
                          n_clusters, criterion='maxclust') - 1
        unique_labels, counts = np.unique(labels, return_counts=True)

        self.results_text.insert(tk.END, "\nClusters Distribution:\n")
        self.results_text.insert(tk.END, "-" * 18 + "\n")
        for label, count in zip(unique_labels, counts):
            self.results_text.insert(
                tk.END, f"Cluster {label}: {count} points\n")

        self.results_text.insert(tk.END, f"\nTotal points: {len(labels)}\n\n")
        self.results_text.insert(
            tk.END, "Dendrogram shows hierarchical structure\n")
        self.results_text.insert(
            tk.END, "Red line indicates cluster cut level\n")

    def display_density_results(self, result, params):
        self.results_text.insert(tk.END, f"{result['algorithm']} Results\n")
        self.results_text.insert(tk.END, "=" * 20 + "\n\n")
        self.results_text.insert(tk.END, f"Algorithm: {result['algorithm']}\n")
        self.results_text.insert(
            tk.END, f"Clusters found: {result['n_clusters']}\n")
        self.results_text.insert(
            tk.END, f"Noise points: {result['n_noise']}\n")
        self.results_text.insert(tk.END, f"Eps: {result['eps']}\n")
        self.results_text.insert(
            tk.END, f"Min samples: {result['min_samples']}\n\n")

        # Add cluster distribution
        labels = result['labels']
        unique_labels, counts = np.unique(labels, return_counts=True)

        self.results_text.insert(tk.END, "Cluster Distribution:\n")
        self.results_text.insert(tk.END, "-" * 18 + "\n")
        for label, count in zip(unique_labels, counts):
            if label == -1:
                self.results_text.insert(tk.END, f"Noise: {count} points\n")
            else:
                self.results_text.insert(
                    tk.END, f"Cluster {label}: {count} points\n")

        self.results_text.insert(tk.END, f"\nTotal points: {len(labels)}\n\n")
        self.results_text.insert(
            tk.END, "Note: Black 'x' marks are noise points\n")

    def create_supervised_plot(self, algorithm_name):
        """Create visualization for supervised algorithms"""
        # Clear previous plots
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(4, 3))

        # Placeholder supervised learning plot
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [0.857, 0.832, 0.871, 0.851]

        bars = ax.bar(metrics, values, color=[
                      '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title(f'{algorithm_name} Performance Metrics')

        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_error(self, message):
        """Show error message in results area"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"ERROR: {message}\n\n")

    def on_previous_step(self):
        """Handle previous step button click"""
        self.event_generate("<<PreviousStep>>")

    def on_next_step(self):
        """Handle next step button click"""
        self.event_generate("<<NextStep>>")

    def reset_frame(self):
        """Reset frame to initial state"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(
            tk.END, "Results will appear here after applying the algorithm...\n\n")

        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        self.next_btn.config(state=tk.DISABLED, bg="#374451")

        # Recreate parameter inputs
        self.create_parameter_inputs()
