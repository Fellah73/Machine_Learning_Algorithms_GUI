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
