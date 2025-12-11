import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        self.encoders: Dict[str, LabelEncoder] = {}
        self.preprocessing_steps: List[str] = []
    
    def analyze_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze and fill missing values in the dataset"""
        try:
            if data is None:
                return {
                    'error': True,
                    'message': 'No dataset provided. Please upload a CSV file first.',
                    'data': None
                }
            
            # Store original data info
            original_rows = data.shape[0]
            original_cols = data.shape[1]
            
            # Count missing values ('?' characters)
            missing_counts = {}
            for col in data.columns:
                missing_count = (data[col] == '?').sum()
                missing_counts[col] = missing_count
            
            # DataFrame for missing values
            missing_df = pd.DataFrame(list(missing_counts.items()),
                                    columns=['Column', 'Missing_Values'])
            missing_df['Percentage'] = (missing_df['Missing_Values'] / len(data)) * 100
            missing_df = missing_df.sort_values('Missing_Values', ascending=False)
            
            columns_with_missing = missing_df[missing_df['Missing_Values'] > 0]
            total_missing = missing_df['Missing_Values'].sum()
            
            numeric_cols = data.select_dtypes(include=['number']).columns
            categorical_cols = data.select_dtypes(include=['object']).columns
            
            numeric_missing = sum((data[col] == '?').sum() for col in numeric_cols)
            categorical_missing = sum((data[col] == '?').sum() for col in categorical_cols)
            
            cleaned_data = data.copy()
            
            # Drop columns with more than 50% missing values
            high_missing_cols = columns_with_missing[
                columns_with_missing['Percentage'] > 50
            ]['Column'].tolist()
            
            if high_missing_cols:
                cleaned_data = cleaned_data.drop(columns=high_missing_cols)
            
            # Fill numerical columns with median
            numeric_cols_cleaned = cleaned_data.select_dtypes(include=['number']).columns
            numeric_filled = []
            for col in numeric_cols_cleaned:
                if (cleaned_data[col] == '?').sum() > 0:
                    temp_col = cleaned_data[col].replace('?', pd.NA)
                    temp_col = pd.to_numeric(temp_col, errors='coerce')
                    median_value = temp_col.median()
                    cleaned_data[col] = cleaned_data[col].replace('?', median_value)
                    numeric_filled.append(col)
            
            # Fill categorical columns with mode
            categorical_cols_cleaned = cleaned_data.select_dtypes(include=['object']).columns
            categorical_filled = []
            for col in categorical_cols_cleaned:
                if (cleaned_data[col] == '?').sum() > 0:
                    temp_col = cleaned_data[col][cleaned_data[col] != '?']
                    if len(temp_col) > 0:
                        mode_value = temp_col.mode()
                        if len(mode_value) > 0:
                            cleaned_data[col] = cleaned_data[col].replace('?', mode_value.iloc[0])
                        else:
                            cleaned_data[col] = cleaned_data[col].replace('?', 'Unknown')
                    else:
                        cleaned_data[col] = cleaned_data[col].replace('?', 'Unknown')
                    categorical_filled.append(col)
            
            self.preprocessing_steps.append("Missing values analysis and filling")
            
            return {
                'error': False,
                'original_shape': (original_rows, original_cols),
                'new_shape': cleaned_data.shape,
                'total_missing': int(total_missing),
                'numeric_missing': int(numeric_missing),
                'categorical_missing': int(categorical_missing),
                'columns_with_missing': columns_with_missing.set_index('Column')['Missing_Values'].to_dict(),
                'missing_percentages': columns_with_missing.set_index('Column')['Percentage'].to_dict(),
                'dropped_columns': high_missing_cols,
                'numeric_columns_filled': numeric_filled,
                'categorical_columns_filled': categorical_filled,
                'cleaned_data': cleaned_data,
                'data_cleaned': True
            }
            
        except Exception as e:
            return {
                'error': True,
                'message': f'Error during missing values analysis: {str(e)}',
                'data': None
            }
    
    def analyze_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        try:
            if data is None:
                return {
                    'error': True,
                    'message': 'No dataset provided.',
                    'data': None
                }
            
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            
            if not numeric_cols:
                return {
                    'error': True,
                    'message': 'No numeric columns found in the dataset.',
                    'data': None
                }
            
            outliers_info = {}
            total_outliers = 0
            
            for col in numeric_cols:
                # Drop NaN values
                col_data = pd.to_numeric(data[col], errors='coerce').dropna()
                
                if len(col_data) > 0:
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    outlier_count = len(outliers)
                    
                    if outlier_count > 0:
                        outliers_info[col] = {
                            'count': outlier_count,
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound,
                            'percentage': (outlier_count / len(col_data)) * 100
                        }
                        total_outliers += outlier_count
            
            self.preprocessing_steps.append("Outliers detection")
            
            return {
                'error': False,
                'numeric_columns': len(numeric_cols),
                'total_outliers': total_outliers,
                'outliers_info': outliers_info,
                'columns_with_outliers': len(outliers_info),
                'data_processed': True
            }
            
        except Exception as e:
            return {
                'error': True,
                'message': f'Error during outliers analysis: {str(e)}',
                'data': None
            }
    
    def normalize_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Normalize data using StandardScaler and LabelEncoder"""
        try:
            if data is None:
                return {
                    'error': True,
                    'message': 'No dataset provided.',
                    'data': None
                }
            
            original_shape = data.shape
            
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
            
            df_encoded = data.copy()
            last_column = data.columns[-1]
            
            # Encode categorical columns
            encoded_categorical = []
            for col in categorical_cols:
                if col in df_encoded.columns:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.encoders[col] = le
                    encoded_categorical.append(col)
            
            # Separate features and target
            X = df_encoded.drop(last_column, axis=1)
            y = df_encoded[last_column]
            
            # Get numeric features for scaling
            numeric_features = [col for col in numeric_cols if col in X.columns]
            
            X_scaled = X.copy()
            scaled_features = []
            
            # Apply StandardScaler
            if numeric_features:
                self.scaler = StandardScaler()
                X_scaled[numeric_features] = self.scaler.fit_transform(X[numeric_features])
                scaled_features = numeric_features
            
            # Combine scaled features with target
            normalized_data = X_scaled.copy()
            normalized_data[last_column] = y
            
            self.preprocessing_steps.append("Data normalization and encoding")
            
            return {
                'error': False,
                'original_shape': original_shape,
                'new_shape': normalized_data.shape,
                'categorical_columns': len(categorical_cols),
                'numeric_columns': len(numeric_cols),
                'encoded_categorical': encoded_categorical,
                'scaled_features': scaled_features,
                'normalized_data': normalized_data,
                'has_target': True,
                'data_normalized': True
            }
            
        except Exception as e:
            return {
                'error': True,
                'message': f'Error during data normalization: {str(e)}',
                'data': None
            }
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get summary of all preprocessing steps performed"""
        return {
            'steps_completed': self.preprocessing_steps.copy(),
            'total_steps': len(self.preprocessing_steps),
            'encoders_used': list(self.encoders.keys()),
            'scaler_fitted': self.scaler is not None
        }
    
    def reset(self):
        """Reset all preprocessing state"""
        self.scaler = None
        self.encoders = {}
        self.preprocessing_steps = []