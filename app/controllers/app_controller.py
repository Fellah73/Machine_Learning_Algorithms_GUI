# controllers/app_controller.py
from tkinter import filedialog, messagebox
from typing import Optional, Dict, Any
from app.models.dataSet_loader import DataSetLoader
from app.models.app_state import AppState
from app.models.data_preprocessing import DataPreprocessor
from app.config.constants import step_mapping, algorithms
from app.models.clustering_metrics import ClusteringMetrics
from app.models.supervised_algorithms import SupervisedAlgorithms


class AppController:
    """Main controller for managing application interactions"""

    def __init__(self):
        self.dataset_loader = DataSetLoader()
        self.app_state = AppState()
        self.view = None  # Will be set by the view
        self.preprocessor = DataPreprocessor()
        self.preprocessing_results = {}
        self.learning_type = None
        self.clustering_metrics = ClusteringMetrics()
        self.supervised_algorithms = SupervisedAlgorithms()
        self.selected_algorithm = None
        self.selected_algorithm_family = None
        self.algorithm_parameters = None
        
    def set_view(self, view):
        self.view = view

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

    def get_current_step(self) -> int:
        return self.app_state.current_step

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

    def get_preprocessing_results(self) -> Dict[str, Any]:
        return self.preprocessing_results

    def is_preprocessing_complete(self) -> bool:
        return (
            self.app_state.preprocessing_steps.get('missing_values', False) and
            self.app_state.preprocessing_steps.get('outliers', False) and
            self.app_state.preprocessing_steps.get('normalization', False)
        )

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

    def set_current_step(self, step: int):
        self.app_state.current_step = step

        return (
            self.app_state.preprocessing_steps.get('missing_values', False) and
            self.app_state.preprocessing_steps.get('outliers', False) and
            self.app_state.preprocessing_steps.get('normalization', False)
        )

    def get_step_mapping_for_current_learning_type(self):
        learning_type = self.learning_type or "unsupervised"  # Default to unsupervised
        return step_mapping.get(learning_type, step_mapping["unsupervised"])

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

    def get_optimal_k(self):
        return self.clustering_metrics.get_optimal_k()

    def get_silhouette_score(self):
        return self.clustering_metrics.get_silhouette_score()

    def get_algorithms_data(self):
        learning_type = self.get_learning_type()

        if learning_type == "unsupervised":
            return algorithms['unsupervised']

        else:
            return algorithms['supervised']

    def set_selected_algorithm(self, algorithm_name, algorithm_family):
        self.selected_algorithm = algorithm_name
        self.selected_algorithm_family = algorithm_family

    def get_selected_algorithm(self):
        return self.selected_algorithm

    def get_selected_algorithm_family(self):
        return self.selected_algorithm_family

    def is_algorithm_selected(self):
        return self.app_state.get_selected_algorithm() is not None

    def train_supervised_algorithm(self, algorithm_name, parameters=None):
        if not self.dataset_loader.has_data():
            return {
                'error': True,
                'message': 'No dataset loaded. Please upload a CSV file first.'
            }

        try:
            data = self.dataset_loader.get_data()

            # Prepare features and target
            X = data.iloc[:, :-1]  # All columns except last
            y = data.iloc[:, -1]   # Last column as target

            # Convert to numeric if needed
            X = X.select_dtypes(include=['number'])

            if len(X.columns) == 0:
                return {
                    'error': True,
                    'message': 'No numeric features found for training.'
                }

            # Train algorithm based on selection
            if algorithm_name == "KNN":
                params = parameters or {}
                return self.supervised_algorithms.train_knn(
                    X.values, y.values,
                    n_neighbors=params.get('n_neighbors', 5),
                    weights=params.get('weights', 'uniform'),
                    algorithm=params.get('algorithm', 'auto')
                )
            elif algorithm_name == "Naive Bayes":
                params = parameters or {}
                return self.supervised_algorithms.train_naive_bayes(
                    X.values, y.values,
                    var_smoothing=params.get('var_smoothing', 1e-9)
                )
            elif algorithm_name == "C4.5":
                params = parameters or {}
                return self.supervised_algorithms.train_c45(
                    X.values, y.values,
                    criterion=params.get('criterion', 'entropy'),
                    max_depth=params.get('max_depth', None),
                    min_samples_split=params.get('min_samples_split', 2)
                )
            else:
                return {
                    'error': True,
                    'message': f'Algorithm {algorithm_name} not implemented.'
                }

        except Exception as e:
            return {
                'error': True,
                'message': str(e)
            }

    def cleanup(self):
        try:
            import matplotlib.pyplot as plt
            plt.close('all')

            if hasattr(self, 'dataset_loader'):
                self.dataset_loader.data = None

            if hasattr(self, 'supervised_algorithms'):
                self.supervised_algorithms.clear_results()

            self.preprocessing_results.clear()
            print("Cleanup completed successfully")

        except Exception as e:
            print(f"Error during cleanup: {e}")

    def get_selected_algorithm(self):
        return self.app_state.get_selected_algorithm()

    def get_algorithm_type(self):
        return self.app_state.get_algorithm_type()

    def set_selected_algorithm(self, algorithm_name, algorithm_type):
        self.app_state.set_selected_algorithm(algorithm_name)
        self.app_state.set_algorithm_type(algorithm_type)

    def get_optimal_k(self):
        return self.app_state.get_optimal_k()

    def get_dataset(self):
        return self.dataset_loader.get_data()

    def is_algorithm_applied(self):
        if hasattr(self, 'algorithm_results') and self.algorithm_results:
            return True
        return False

    def set_algorithm_results(self, results):
        self.algorithm_results = results
        
    def set_algorithm_parameters(self, parameters):
     self.algorithm_parameters = parameters

    def get_algorithm_parameters(self):
     return getattr(self, 'algorithm_parameters', {})
