base_menuButtons = ["Upload Dataset", "Preprocessing",
                    "Learning Type", "Algorithms", "Visualization", "Comparison"]

unsupervised_menuButtons = ["Upload Dataset", "Preprocessing", "Learning Type",
                            "Clustering Metrics", "Algorithms", "Visualization", "Comparison"]

supervised_menuButtons = ["Upload Dataset", "Preprocessing",
                          "Learning Type", "Algorithms", "Visualization", "Comparison"]

menuButtons = base_menuButtons



def get_menu_buttons_for_learning_type(learning_type):
    if learning_type == "unsupervised":
        return unsupervised_menuButtons
    elif learning_type in ["classification", "regression"]:
        return supervised_menuButtons
    else:
        return base_menuButtons

step_mapping = {
    "unsupervised": {
        0: "upload",
        1: "preprocessing",
        2: "learning_type",
        3: "clustering_metrics",
        4: "algorithms",
        5: "visualization",
        6: "unsup_comparison"
    },
    "supervised": {
        0: "upload",
        1: "preprocessing",
        2: "learning_type",
        3: "algorithms",  
        4: "visualization",
        5: "sup_comparison"
    }
}

algorithms = {
    'unsupervised': {
        "Partitioning": {
            "description": "Divise les données en k partitions non-chevauchantes où chaque point appartient à exactement un cluster.",
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
            "algorithms": {
                "KNN": {
                    "parameters": ["n_neighbors", "training perc"]
                }
            }
        },
        "Probabilistic": {
            "description": "Algorithms based on probability theory and statistical inference.",
            "algorithms": {
                "Naive Bayes": {
                    "parameters": ["training perc"]
                }
            }
        },
        "Tree-based": {
            "description": "Algorithms that create decision trees to make predictions based on feature splits.",
            "algorithms": {
                "C4.5": {
                    "parameters": ["training perc"]
                }
            }
        }
    }
}
