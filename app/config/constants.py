# Flow de base (tous les types)
base_menuButtons = ["Upload Dataset", "Preprocessing",
                    "Learning Type", "Algorithms", "Visualization", "Comparison"]

# Flow spécifique pour unsupervised (avec clustering metrics)
unsupervised_menuButtons = ["Upload Dataset", "Preprocessing", "Learning Type",
                            "Clustering Metrics", "Algorithms", "Visualization", "Comparison"]

# Flow pour supervised (sans clustering metrics)
supervised_menuButtons = ["Upload Dataset", "Preprocessing",
                          "Learning Type", "Algorithms", "Visualization", "Comparison"]

# Configuration par défaut
menuButtons = base_menuButtons

# Fonction pour obtenir les boutons selon le type d'apprentissage


def get_menu_buttons_for_learning_type(learning_type):
    """Return appropriate menu buttons based on learning type"""
    if learning_type == "unsupervised":
        return unsupervised_menuButtons
    elif learning_type in ["classification", "regression"]:
        return supervised_menuButtons
    else:
        return base_menuButtons


# Mapping des étapes selon le type d'apprentissage
step_mapping = {
    "unsupervised": {
        0: "upload",
        1: "preprocessing",
        2: "learning_type",
        3: "clustering_metrics",
        4: "algorithms",
        5: "visualization",
        6: "comparison"
    },
    "supervised": {
        0: "upload",
        1: "preprocessing",
        2: "learning_type",
        3: "algorithms",  # Skip clustering_metrics
        4: "visualization",
        5: "comparison"
    }
}

algorithms = {
    'unsupervised': {
        "Partitioning": {
            "description": "Divise les données en k partitions non-chevauchantes où chaque point appartient à exactement un cluster.",
            "image": "https://files.edgestore.dev/643tuked7tdgupmf/publicFiles/_public/5e568f84-ada0-488c-b7a9-8de92bed1d5f.png",
            "algorithms": {
                "K-Means": {
                    "parameters": ["n_clusters", "distance_metric", 'max_iter'],
                },
                "K-Medoids": {
                    "parameters": ["n_clusters", "distance_metric", "max_iter"],
                }
            }
        },

        "Hierarchical": {
            "description": "Crée une hiérarchie de clusters en formant un arbre de clusters (dendrogramme).",
            "image": "https://files.edgestore.dev/643tuked7tdgupmf/publicFiles/_public/321e89c8-098c-4cf8-93ec-2de7ce5ddb02.png",
            "algorithms": {
                "AGNES": {
                    "parameters": ["n_clusters", "linkage", 'distance_metric'],
                },
                "DIANA": {
                    "parameters": ["n_clusters", 'distance_metric'],
                }
            }
        },

        "Density-based": {
            "description": "Identifie les clusters comme des zones denses séparées par des zones de faible densité.",
            "image": "https://files.edgestore.dev/643tuked7tdgupmf/publicFiles/_public/f5038606-6641-4c31-8a12-69b2ac3d9873.png",
            "algorithms": {
                "DBSCAN": {
                    "parameters": ["eps", "min_samples"],
                }
            }
        }
    },
    'supervised': {
        "Lazy Learning": {
            "description": "Algorithms that defer computation until a query is made. No explicit training phase.",
            "image": "https://via.placeholder.com/200x150/4CAF50/white?text=Lazy+Learning",
            "algorithms": {
                "KNN": {
                    "parameters": ["n_neighbors", "weights", "algorithm"]
                }
            }
        },
        "Probabilistic": {
            "description": "Algorithms based on probability theory and statistical inference.",
            "image": "https://via.placeholder.com/200x150/FF9800/white?text=Probabilistic",
            "algorithms": {
                "Naive Bayes": {
                    "parameters": ["var_smoothing"]
                }
            }
        },
        "Tree-based": {
            "description": "Algorithms that create decision trees to make predictions based on feature splits.",
            "image": "https://via.placeholder.com/200x150/9C27B0/white?text=Tree+Based",
            "algorithms": {
                "C4.5": {
                    "parameters": ["criterion", "max_depth", "min_samples_split"]
                }
            }
        }
    }
}
