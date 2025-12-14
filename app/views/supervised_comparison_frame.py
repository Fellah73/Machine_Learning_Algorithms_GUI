import tkinter as tk
from tkinter import ttk
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


class SupervisedComparisonFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#f0f0f0")
        self.controller = controller
        self.comparison_algorithms = ['KNN', 'C4.5', 'Naive Bayes']
        self.test_sizes = [0.1, 0.2, 0.3]  # 10%, 20%, 30%
        self.comparison_results = {}
        self.setup_ui()
        self.auto_run_comparison()

    def setup_ui(self):
        title_label = tk.Label(
            self,
            text="Supervised algorithms comparison",
            bg="#f0f0f0",
            fg="#24367E",
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=20)

        main_frame = tk.Frame(self, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=50, pady=20)

        table_frame = tk.Frame(main_frame, bd=2)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        table_title = tk.Label(
            table_frame,
            text="Performance Metrics Comparison (Multiple Test Sizes)",
            fg="#24367E",
            font=("Arial", 16, "bold")
        )
        table_title.pack(pady=15)

        # Create treeview for comparison table
        self.comparison_treeview = ttk.Treeview(
            table_frame,
            columns=('Test_Size', 'Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score'),
            height=12,
            show='headings'
        )
        self.comparison_treeview.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Configure column headings
        self.comparison_treeview.heading('Test_Size', text='Test Size')
        self.comparison_treeview.heading('Algorithm', text='Algorithm')
        self.comparison_treeview.heading('Accuracy', text='Accuracy')
        self.comparison_treeview.heading('Precision', text='Precision')
        self.comparison_treeview.heading('Recall', text='Recall')
        self.comparison_treeview.heading('F1-Score', text='F1-Score')

        self.comparison_treeview.column('Test_Size', width=100)
        self.comparison_treeview.column('Algorithm', width=120)
        self.comparison_treeview.column('Accuracy', width=120)
        self.comparison_treeview.column('Precision', width=150)
        self.comparison_treeview.column('Recall', width=130)
        self.comparison_treeview.column('F1-Score', width=150)

        self.style = ttk.Style()
        self.style.configure("Treeview.Heading", font=("Arial", 12, "bold"))
        self.style.configure("Treeview", font=("Arial", 11), rowheight=30)
        self.style.configure("Selected.Treeview", background="#e6f3ff", foreground="#0066cc")
        self.style.configure("Normal.Treeview", background="white", foreground="black")

        nav_frame = tk.Frame(self, bg="#f0f0f0")
        nav_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=10)

        self.prev_btn = tk.Button(
            nav_frame,
            text="Previous",
            command=self.on_previous_step,
            bg="#6c757d",
            fg="white",
            font=("Arial", 12, "bold"),
            padx=25,
            pady=8,
            cursor="hand2"
        )
        self.prev_btn.pack(side=tk.LEFT)

    def prepare_data_for_classification(self, test_size=0.3):
        """Prepare data for supervised classification with specified test size"""
        dataset = self.controller.dataset_loader.data
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

    def get_parameter_values(self):
        """Get algorithm parameters"""
        user_params = getattr(self.controller, 'algorithm_parameters', {})

        return {
            'n_neighbors': user_params.get('n_neighbors', 5),
            'max_depth': user_params.get('max_depth', None),
            'criterion': user_params.get('criterion', 'gini'),
            'var_smoothing': user_params.get('var_smoothing', 1e-9)
        }

    def knn_classification(self, X_train, X_test, y_train, y_test, n_neighbors):
        """Apply KNN classification"""
        try:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'algorithm': 'KNN',
                'n_neighbors': n_neighbors
            }
        except Exception as e:
            return {'error': str(e)}

    def c45_classification(self, X_train, X_test, y_train, y_test, max_depth, criterion):
        """Apply C4.5 (Decision Tree) classification"""
        try:
            # C4.5 is approximated using DecisionTreeClassifier with entropy
            dt = DecisionTreeClassifier(
                criterion='entropy',  # C4.5 uses information gain (entropy)
                max_depth=max_depth,
                random_state=42
            )
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'algorithm': 'C4.5',
                'max_depth': max_depth
            }
        except Exception as e:
            return {'error': str(e)}

    def naive_bayes_classification(self, X_train, X_test, y_train, y_test, var_smoothing):
        """Apply Naive Bayes classification"""
        try:
            nb = GaussianNB(var_smoothing=var_smoothing)
            nb.fit(X_train, y_train)
            y_pred = nb.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'algorithm': 'Naive Bayes',
                'var_smoothing': var_smoothing
            }
        except Exception as e:
            return {'error': str(e)}

    def calculate_performance_metrics(self):
        """Calculate performance metrics for all algorithms and test sizes"""
        self.comparison_results = {}
        params = self.get_parameter_values()

        n_neighbors = params['n_neighbors']
        max_depth = params['max_depth']
        criterion = params['criterion']
        var_smoothing = params['var_smoothing']

        # Test each algorithm with each test size
        for test_size in self.test_sizes:
            X_train, X_test, y_train, y_test, error = self.prepare_data_for_classification(test_size)
            if error:
                print(f"Data preparation error for test_size {test_size}: {error}")
                continue

            test_size_key = f"{int(test_size * 100)}%"
            self.comparison_results[test_size_key] = {}

            # Apply each algorithm
            algorithms = {
                'KNN': lambda: self.knn_classification(X_train, X_test, y_train, y_test, n_neighbors),
                'C4.5': lambda: self.c45_classification(X_train, X_test, y_train, y_test, max_depth, criterion),
                'Naive Bayes': lambda: self.naive_bayes_classification(X_train, X_test, y_train, y_test, var_smoothing)
            }

            for algo_name, algo_func in algorithms.items():
                try:
                    result = algo_func()
                    if 'error' not in result:
                        self.comparison_results[test_size_key][algo_name] = result
                    else:
                        self.comparison_results[test_size_key][algo_name] = None
                        print(f"Error in {algo_name} with test_size {test_size_key}: {result['error']}")
                except Exception as e:
                    self.comparison_results[test_size_key][algo_name] = None
                    print(f"Exception in {algo_name} with test_size {test_size_key}: {str(e)}")

        return self.comparison_results

    def update_comparison_table(self):
     """Update the comparison table with performance metrics for all test sizes"""
     selected_algorithm = self.controller.get_selected_algorithm()

     for item in self.comparison_treeview.get_children():
        self.comparison_treeview.delete(item)

     results = self.calculate_performance_metrics()
     if not results:
        self.comparison_treeview.insert('', 'end', values=('N/A', 'No data available', '', '', '', ''))
        return

     ordered_algorithms = []
     if selected_algorithm and selected_algorithm in self.comparison_algorithms:
        ordered_algorithms.append(selected_algorithm)
        for algo in self.comparison_algorithms:
            if algo != selected_algorithm:
                ordered_algorithms.append(algo)
     else:
        ordered_algorithms = self.comparison_algorithms.copy()

     for test_size_key in ['10%', '20%', '30%']: 
        if test_size_key in results:
            for algo_name in ordered_algorithms:  
                if algo_name in results[test_size_key] and results[test_size_key][algo_name]:
                    result = results[test_size_key][algo_name]
                    values = (
                        test_size_key if (test_size_key and selected_algorithm == algo_name) else '',
                        algo_name,
                        f"{result['accuracy']:.3f}",
                        f"{result['precision']:.3f}",
                        f"{result['recall']:.3f}",
                        f"{result['f1_score']:.3f}"
                    )
                else:
                    values = (test_size_key, algo_name, 'N/A', 'N/A', 'N/A', 'N/A')

                # Insert item and apply style based on selection
                item = self.comparison_treeview.insert('', 'end', values=values)

                # Highlight selected algorithm row
                if selected_algorithm and algo_name == selected_algorithm:
                    self.comparison_treeview.item(item, tags=('selected',))
                else:
                    self.comparison_treeview.item(item, tags=('normal',))

     # Configure tag styles for highlighting
     self.comparison_treeview.tag_configure('selected', background='#e6f3ff', foreground='#0066cc', font=("Arial", 11, "bold"))
     self.comparison_treeview.tag_configure('normal', background='white', foreground='black', font=("Arial", 11))
    
    def run_comparison_analysis(self):
        """Run the complete comparison analysis"""
        selected_algorithm = self.controller.get_selected_algorithm()
        if not selected_algorithm:
            print("No algorithm selected for comparison")
            return

        try:
            self.update_comparison_table()
        except Exception as e:
            print(f"Error in supervised comparison analysis: {e}")

    def auto_run_comparison(self):
        """Automatically run comparison after frame initialization"""
        self.after(100, self.run_comparison_analysis)

    def on_previous_step(self):
        """Handle previous step button click"""
        self.event_generate("<<PreviousStep>>")