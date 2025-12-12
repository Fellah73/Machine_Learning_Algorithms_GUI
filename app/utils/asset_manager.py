import os
from pathlib import Path
from PIL import Image, ImageTk
import tkinter as tk

class AssetManager:
    def __init__(self):
        # Chemin vers le dossier assets dans app/
        self.project_root = Path(__file__).parent.parent  # app/
        self.assets_path = self.project_root / "assets"
        self.images_path = self.assets_path / "images" / "algorithms"
        
        # Cache pour les images chargées
        self._image_cache = {}
        
        
    def get_algorithm_image(self, learning_type, family_type, size=(180, 140)):
        """Récupérer l'image d'une famille d'algorithmes"""
        cache_key = f"{learning_type}_{family_type}_{size}"
        
        if cache_key in self._image_cache:
            return self._image_cache[cache_key]
        
        # Mapping des noms de fichiers
        image_mapping = {
            "unsupervised": {
                "Partitioning": "Partitioning.png",
                "Hierarchical": "Hierarchical.png", 
                "Density-based": "Density-based.png"
            },
            "supervised": {
                "Lazy Learning": "Lazy Learning.png",
                "Probabilistic": "Probabilistic.png",
                "Tree-based": "Tree-based.png"
            }
        }
        
        try:
            filename = image_mapping.get(learning_type, {}).get(family_type)
            if not filename:
                print(f"No mapping found for {learning_type} -> {family_type}")
                return self._get_placeholder_image(family_type, size)
            
            image_path = self.images_path / learning_type / filename
            
            if image_path.exists():
                # Charger et redimensionner l'image
                image = Image.open(image_path)
                image = image.resize(size, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                
                # Mettre en cache
                self._image_cache[cache_key] = photo
                return photo
            else:
                return self._get_placeholder_image(family_type, size)
                
        except Exception as e:
            print(f"Error loading image for {family_type}: {e}")
            return self._get_placeholder_image(family_type, size)
    
    def _get_placeholder_image(self, family_type, size):
        """Créer une image placeholder si l'image n'existe pas"""
        try:
            # Créer une image simple avec couleur de fond
            colors = {
                "Partitioning": "#4CAF50",
                "Hierarchical": "#2196F3", 
                "Density-based": "#FF9800",
                "Lazy Learning": "#9C27B0",
                "Probabilistic": "#F44336",
                "Tree-based": "#607D8B"
            }
            
            color = colors.get(family_type, "#666666")
            image = Image.new('RGB', size, color=color)
            
            # Convertir en PhotoImage
            photo = ImageTk.PhotoImage(image)
            return photo
        except Exception as e:
            print(f"Error creating placeholder: {e}")
            return None

# Instance globale
asset_manager = AssetManager()