# controllers/app_controller.py
from tkinter import filedialog, messagebox
from typing import Optional, Dict, Any
from app.models.dataSet_loader import DataSetLoader
from app.models.app_state import AppState
from app.models.data_preprocessing import DataPreprocessor
from app.config.constants import step_mapping, algorithms
from app.models.clustering_metrics import ClusteringMetrics
from app.models.supervised_algorithms import SupervisedAlgorithms
from app.models.unsupervisd_algorithms import UnsupervisedAlgorithms
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score


class AppController:

    def __init__(self):
        self.dataset_loader = DataSetLoader()
        self.app_state = AppState()
        self.view = None  # Will be set by the view
        self.preprocessor = DataPreprocessor()
        self.preprocessing_results = {}
        self.learning_type = None
        self.clustering_metrics = ClusteringMetrics()
        self.supervised_algorithms = SupervisedAlgorithms()
        self.unsupervised_algorithms = UnsupervisedAlgorithms()
        self.selected_algorithm = None
        self.selected_algorithm_family = None
        self.algorithm_parameters = None
        self.test_sizes = [0.1, 0.2, 0.3]  # 10%, 20%, 30%

    def set_view(self, view):
        self.view = view

    # Navigation Methods

    def can_proceed_to_next_step(self) -> bool:
        return self.app_state.has_dataset()

    def increment_step(self):
        self.app_state.increment_step()
        if self.view:
            self.view.show_current_step()

    def decrement_step(self):
        self.app_state.decrement_step()
        if self.view:
            self.view.show_current_step()

    def set_current_step(self, step: int):
        self.app_state.current_step = step

        return (
            self.app_state.preprocessing_steps.get('missing_values', False) and
            self.app_state.preprocessing_steps.get('outliers', False) and
            self.app_state.preprocessing_steps.get('normalization', False)
        )

    def get_current_step(self) -> int:
        return self.app_state.current_step

      # Dataset Controller Methods

    def get_step_mapping_for_current_learning_type(self):
        learning_type = self.learning_type or "unsupervised"  # Default to unsupervised
        return step_mapping.get(learning_type, step_mapping["unsupervised"])

    # Dataset Upload Controller Methods

    def handle_file_upload(self) -> Optional[dict]:
        # Open file selection dialog
        file_path = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv")]
        )

        # If no file selected
        if not file_path:
            return None

        # Load file via model
        result = self.dataset_loader.load_csv(file_path)

        # If success, update global state
        if result['success']:
            self.app_state.set_dataset(self.dataset_loader.get_data())

            # Show success message
            messagebox.showinfo(
                "Success",
                f"CSV file loaded successfully!\nPlease click on the next step button."
            )
        else:
            # Show error message
            messagebox.showerror("Error", result['message'])

        return result

    # Preprocessing Controller Methods

    def analyze_missing_values(self) -> Dict[str, Any]:
        if not self.dataset_loader.has_data():
            return {
                'error': True,
                'message': 'No dataset loaded. Please upload a CSV file first.'
            }

        data = self.dataset_loader.get_data()
        result = self.preprocessor.analyze_missing_values(data)

        if not result['error']:
            # Update the dataset with cleaned data
            self.dataset_loader.data = result['cleaned_data']
            self.preprocessing_results['missing_values'] = result
            self.app_state.set_preprocessing_step_completed(
                'missing_values', True)

        return result

    def analyze_outliers(self) -> Dict[str, Any]:
        if not self.dataset_loader.has_data():
            return {
                'error': True,
                'message': 'No dataset loaded. Please complete missing values analysis first.'
            }

        data = self.dataset_loader.get_data()
        result = self.preprocessor.analyze_outliers(data)

        if not result['error']:
            self.preprocessing_results['outliers'] = result
            self.app_state.set_preprocessing_step_completed('outliers', True)

        return result

    def normalize_data(self) -> Dict[str, Any]:
        if not self.dataset_loader.has_data():
            return {
                'error': True,
                'message': 'No dataset loaded. Please complete previous preprocessing steps first.'
            }

        data = self.dataset_loader.get_data()
        result = self.preprocessor.normalize_data(data)

        if not result['error']:
            # Update the dataset with normalized data
            self.dataset_loader.data = result['normalized_data']
            self.preprocessing_results['normalization'] = result
            self.app_state.set_preprocessing_step_completed(
                'normalization', True)

        return result

    def is_preprocessing_complete(self) -> bool:
        return (
            self.app_state.preprocessing_steps.get('missing_values', False) and
            self.app_state.preprocessing_steps.get('outliers', False) and
            self.app_state.preprocessing_steps.get('normalization', False)
        )

    # Learning Type Controller Methods

    def set_learning_type(self, learning_type: str):
        self.learning_type = learning_type
        self.app_state.selected_learning_type = learning_type

        # Notify view to update navigation
        if self.view:
            self.view.update_navigation_for_learning_type(learning_type)

    def get_learning_type(self) -> str:
        return self.learning_type

    def is_learning_type_selected(self) -> bool:
        return self.learning_type is not None

    # Clustering Metrics Controller Methods

    def analyze_clusters(self):
        if not self.dataset_loader.has_data():
            return {
                'error': True,
                'message': 'No dataset loaded. Please upload a CSV file first.'
            }

        data = self.dataset_loader.get_data()

        # Select only numeric columns for clustering
        last_column = data.columns[-1]
        features_data = data.drop(last_column, axis=1)
        numeric_data = features_data.select_dtypes(include=['number'])

        if len(numeric_data.columns) == 0:
            # Use sample data if no numeric features
            result = self.clustering_metrics.analyze_clusters(None)
        else:
            result = self.clustering_metrics.analyze_clusters(
                numeric_data.values)

        # Store optimal K and silhouette score in app state
        if not result.get('error'):
            self.app_state.optimal_k = result['optimal_k']
            self.app_state.silhouette_score = result['silhouette_score']

        return result

    # Algorithms Controller Methods

    def get_algorithms_data(self):
        learning_type = self.get_learning_type()

        if learning_type == "unsupervised":
            return algorithms['unsupervised']

        else:
            return algorithms['supervised']

    def set_selected_algorithm(self, algorithm_name, algorithm_family):
        # update state
        self.app_state.set_selected_algorithm(algorithm_name)
        self.app_state.set_algorithm_type(algorithm_family)

        # update in controller
        self.selected_algorithm = algorithm_name
        self.selected_algorithm_family = algorithm_family

    # Visulaization Controller Methods

    def get_selected_algorithm(self):
        return self.app_state.get_selected_algorithm()

    def get_algorithm_type(self):
        return self.app_state.get_algorithm_type()

    #  Suppervised Algos Function
    def apply_supervised_algorithm(self, algorithm_name, params):
        # Get dataset for supervised learning
        dataset = self.dataset_loader.data

        if dataset is None:
            self.show_error("No dataset loaded")
            return

        # Prepare data for supervised learning
        if len(dataset.columns) < 2:
            self.show_error(
                "Dataset needs at least 2 columns (features + target)")
            return

        # Get features and target
        # Assume last column is target
        X = dataset.iloc[:, :-1]
        y = dataset.iloc[:, -1]

        # Select numeric features only
        numeric_features = X.select_dtypes(include=[np.number])
        if len(numeric_features.columns) == 0:
            self.show_error("No numeric features found")
            return

        X_numeric = numeric_features.values

        # Encode target if it's categorical
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Apply algorithm based on name
        if algorithm_name == "KNN":
            result = self.supervised_algorithms.apply_knn_algorithm(
                X_numeric, y_encoded, params)

        elif algorithm_name == "Naive Bayes":
            result = self.supervised_algorithms.apply_naive_bayes_algorithm(
                X_numeric, y_encoded, params)

        elif algorithm_name == "C4.5":
            result = self.supervised_algorithms.apply_c45_algorithm(
                X_numeric, y_encoded, params, numeric_features.columns)

        return result, numeric_features, le

    #  Unsupervised Algos Function
    def apply_unsupervised_algorithm(self, algorithm_name, algorithm_type, params):

        dataset = self.dataset_loader.data
        if dataset is None:
            self.show_error("No dataset loaded")
            print("No dataset loaded")

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
            result = self.unsupervised_algorithms.apply_partitioning_algorithm(
                algorithm_name, data, params, selected_columns)

        elif algorithm_type == "Hierarchical":
            result = self.unsupervised_algorithms.apply_hierarchical_algorithm(
                algorithm_name, data, params)

        elif algorithm_type == "Density-based":
            result = self.unsupervised_algorithms.apply_density_algorithm(
                algorithm_name, data, params)

        return result

    def get_optimal_k(self):
        return self.clustering_metrics.get_optimal_k()

    def is_algorithm_selected(self):
        return self.app_state.get_selected_algorithm() is not None

    def get_selected_algorithm(self):
        return self.app_state.get_selected_algorithm()

    def get_optimal_k(self):
        return self.app_state.get_optimal_k()

    def set_algorithm_parameters(self, parameters):
        self.algorithm_parameters = parameters

    # Supervised Comparison Function

    def get_parameter_values(self):
        """Get algorithm parameters"""
        user_params = self.algorithm_parameters or {}

        return {
            'n_neighbors': user_params.get('n_neighbors', 5),
            'max_depth': user_params.get('max_depth', None),
            'criterion': user_params.get('criterion', 'gini'),
            'var_smoothing': user_params.get('var_smoothing', 1e-9)
        }

    def prepare_data_for_classification(self, test_size=0.3):
        """Prepare data for supervised classification with specified test size"""
        dataset = self.dataset_loader.data
        if dataset is None:
            return None, None, None, None, "No dataset loaded"

        # Separate features and target
        target_column = dataset.columns[-1]
        features = dataset.drop(target_column, axis=1)
        target = dataset[target_column]

        numeric_features = features.select_dtypes(include=[np.number])
        if len(numeric_features.columns) == 0:
            return None, None, None, None, "No numeric features found"

        label_encoder = LabelEncoder()
        if not np.issubdtype(target.dtype, np.number):
            target_encoded = label_encoder.fit_transform(target)
        else:
            target_encoded = target.values

        X_train, X_test, y_train, y_test = train_test_split(
            numeric_features.values, target_encoded, test_size=test_size, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, None

    def calculate_performance_metrics(self):
        self.comparison_results = {}
        params = self.get_parameter_values()

        n_neighbors = params['n_neighbors']
        max_depth = params['max_depth']
        criterion = params['criterion']
        var_smoothing = params['var_smoothing']

        # Test each algorithm with each test size
        for test_size in self.test_sizes:
            X_train, X_test, y_train, y_test, error = self.prepare_data_for_classification(
                test_size)
            if error:
                print(
                    f"Data preparation error for test_size {test_size}: {error}")
                continue

            test_size_key = f"{int(test_size * 100)}%"
            self.comparison_results[test_size_key] = {}

            # Apply each algorithm
            algorithms = {
                'KNN': lambda: self.supervised_algorithms.knn_classification_metrics(X_train, X_test, y_train, y_test, n_neighbors),
                'C4.5': lambda: self.supervised_algorithms.c45_classification_metrics(X_train, X_test, y_train, y_test, max_depth, criterion),
                'Naive Bayes': lambda: self.supervised_algorithms.naive_bayes_classification(X_train, X_test, y_train, y_test, var_smoothing)
            }

            for algo_name, algo_func in algorithms.items():
                try:
                    result = algo_func()
                    if 'error' not in result:
                        self.comparison_results[test_size_key][algo_name] = result
                    else:
                        self.comparison_results[test_size_key][algo_name] = None
                        print(
                            f"Error in {algo_name} with test_size {test_size_key}: {result['error']}")
                except Exception as e:
                    self.comparison_results[test_size_key][algo_name] = None
                    print(
                        f"Exception in {algo_name} with test_size {test_size_key}: {str(e)}")

        return self.comparison_results

    # Unsupervised Comparison Function

    def prepare_data_for_clustering(self):
        dataset = self.dataset_loader.data
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
        optimal_k = self.get_optimal_k()

        user_params = self.algorithm_parameters or {}

        return {
            'n_clusters': user_params.get('n_clusters', optimal_k),
            'distance_metric': user_params.get('distance_metric', 'euclidean'),
            'linkage': user_params.get('linkage_method', 'single'),
            'max_iter': user_params.get('max_iter', 300),
            'eps': user_params.get('eps', 1.3),
            'min_samples': user_params.get('min_samples', 2)
        }

       # Comparison logic

    def calculate_silhouette_scores(self):
        data, error = self.prepare_data_for_clustering()
        if data is None or error:
            return None

        comparison_algorithms = ['K-Means',
                                 'K-Medoids', 'AGNES', 'DIANA', 'DBSCAN']
        comparison_results = {}
        params = self.get_parameter_values()

        k_clusters = params['n_clusters']
        distance_metric = params['distance_metric']
        linkage_method = params['linkage']
        n_iterations = params['max_iter']
        eps = params['eps']
        min_samples = params['min_samples']

        for algo in comparison_algorithms:
            try:
                if algo == 'K-Means':
                    result = self.unsupervised_algorithms.kmeans_clustering(
                        data, k_clusters, distance_metric, n_iterations)
                elif algo == 'K-Medoids':
                    result = self.unsupervised_algorithms.kmedoids_clustering(
                        data, k_clusters, distance_metric, n_iterations)
                elif algo == 'AGNES':
                    result = self.unsupervised_algorithms.agnes_clustering(
                        data, k_clusters, linkage_method, distance_metric)
                elif algo == 'DIANA':
                    result = self.unsupervised_algorithms.diana_clustering(
                        data, k_clusters, distance_metric)
                elif algo == 'DBSCAN':
                    result = self.unsupervised_algorithms.dbscan_clustering(
                        data, eps, min_samples)

                if 'error' not in result:
                    if algo == 'DBSCAN' and result['n_clusters'] < 2:
                        comparison_results[algo] = {
                            'silhouette': None, 'result': result}
                    else:
                        sil_score = silhouette_score(data, result['labels'])
                        comparison_results[algo] = {
                            'silhouette': sil_score, 'result': result}
                else:
                    comparison_results[algo] = {
                        'silhouette': None, 'result': None}

            except Exception as e:
                comparison_results[algo] = {
                    'silhouette': None, 'result': None}

        return comparison_results
