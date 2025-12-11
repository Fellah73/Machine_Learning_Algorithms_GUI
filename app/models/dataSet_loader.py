# models/dataSet_loader.py
import pandas as pd
from typing import Optional, Dict, Any

class DataSetLoader:
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.file_path: Optional[str] = None
        self.metadata: Dict[str, Any] = {}
    
    def load_csv(self, file_path: str) -> Dict[str, Any]:
        try:
            self.data = pd.read_csv(file_path)
            self.file_path = file_path
            
            # Extraire les métadonnées
            self.metadata = {
                'success': True,
                'file_name': file_path.split('/')[-1],
                'rows': len(self.data),
                'columns': len(self.data.columns),
                'column_names': list(self.data.columns),
                'message': 'File loaded successfully'
            }
            
            return self.metadata
            
        except pd.errors.EmptyDataError:
            return {
                'success': False,
                'message': 'The selected file is empty!',
                'error_type': 'EmptyDataError'
            }
            
        except pd.errors.ParserError:
            return {
                'success': False,
                'message': 'Error parsing the CSV file!',
                'error_type': 'ParserError'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'An error occurred: {str(e)}',
                'error_type': 'UnknownError'
            }
    
    def get_data(self) -> Optional[pd.DataFrame]:
        return self.data
    
    def has_data(self) -> bool:
        return self.data is not None
    
    def clear_data(self):
        self.data = None
        self.file_path = None
        self.metadata = {}