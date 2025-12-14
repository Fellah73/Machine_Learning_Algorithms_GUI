# models/app_state.py
from typing import Optional
import pandas as pd
from app.models.data_preprocessing import DataPreprocessor

class AppState:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppState, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.dataset_data: Optional[pd.DataFrame] = None
        self.current_step: int = 0
        self.optimal_k: Optional[int] = None
        self.silhouette_score: Optional[float] = None
        self.selected_algorithm: Optional[str] = None
        self.current_algorithm_type: Optional[str] = None
        self.clustering_results: Optional[dict] = None
        self.selected_learning_type: Optional[str] = None
        self.optimal_k: Optional[int] = None
        self.silhouette_score: Optional[float] = None
        

        # Preprocessing flags
        self.preprocessing_steps = {
            'missing_values': False,
            'outliers': False,
            'normalization': False
        }

        self._initialized = True

    def set_dataset(self, data: pd.DataFrame):
        self.dataset_data = data

    def get_dataset(self) -> Optional[pd.DataFrame]:
        return self.dataset_data

    def has_dataset(self) -> bool:
        return self.dataset_data is not None

    def increment_step(self):
        if self.current_step <= 5:  
            self.current_step += 1

    def decrement_step(self):
        if self.current_step > 0:
            self.current_step -= 1

    def mark_preprocessing_step_complete(self, step: str):
        if step in self.preprocessing_steps:
            self.preprocessing_steps[step] = True

    def is_preprocessing_complete(self) -> bool:
        return all(self.preprocessing_steps.values())

    def set_preprocessing_step_completed(self, step_name: str, completed: bool):
        if step_name in self.preprocessing_steps:
            self.preprocessing_steps[step_name] = completed

    def get_preprocessing_step_status(self, step_name: str) -> bool:
        return self.preprocessing_steps.get(step_name, False)

    def reset_preprocessing(self):
        self.preprocessing_steps = {
            'missing_values': False,
            'outliers': False,
            'normalization': False
        }
        
    def analyze_missing_values(self):
        if not self.dataset_loader.has_data():
            return {
                'error': True,
                'message': 'No dataset loaded. Please upload a CSV file first.'
            }
        
        data = self.dataset_loader.get_data()
        # Utiliser le preprocessor de app_state
        result = self.app_state.preprocessor.analyze_missing_values(data)
        return result
    
    def analyze_outliers(self):
        if not self.dataset_loader.has_data():
            return {
                'error': True,
                'message': 'No dataset loaded. Please upload a CSV file first.'
            }
        
        data = self.dataset_loader.get_data()
        result = self.app_state.preprocessor.analyze_outliers(data)
        return result
    
    def normalize_data(self):
        if not self.dataset_loader.has_data():
            return {
                'error': True,
                'message': 'No dataset loaded. Please upload a CSV file first.'
            }
        
        data = self.dataset_loader.get_data()
        result = self.app_state.preprocessor.normalize_data(data)
        return result

    def reset(self):
        # ...existing code...
        self.selected_learning_type = None
        self.optimal_k = None
        self.silhouette_score = None
        
    def get_learning_type(self):
     learning_type = self.selected_learning_type  # Retourne None si pas défini
     return learning_type

    def set_learning_type(self, learning_type):
     self.selected_learning_type = learning_type
     
    def get_selected_algorithm(self):
     return self.selected_algorithm

    def set_selected_algorithm(self, algorithm):
     self.selected_algorithm = algorithm

    def get_algorithm_type(self):
     return self.current_algorithm_type

    def set_algorithm_type(self, algorithm_type):
     self.current_algorithm_type = algorithm_type

    def get_optimal_k(self):
     return self.optimal_k

    def set_optimal_k(self, optimal_k):
     self.optimal_k = optimal_k
