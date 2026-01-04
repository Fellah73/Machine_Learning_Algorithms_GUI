import tkinter as tk
from tkinter import ttk

class SupervisedComparisonFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#f0f0f0")
        self.controller = controller
        self.comparison_algorithms = ['KNN', 'C4.5', 'Naive Bayes']
        self.comparison_results = {}
        self.setup_ui()
        self.after(100, self.run_comparison_analysis)

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

    def run_comparison_analysis(self):
        selected_algorithm = self.controller.get_selected_algorithm()
        if not selected_algorithm:
            print("No algorithm selected for comparison")
            return

        try:
            self.update_comparison_table()
        except Exception as e:
            print(f"Error in supervised comparison analysis: {e}")

    def update_comparison_table(self):
     selected_algorithm = self.controller.get_selected_algorithm()

     for item in self.comparison_treeview.get_children():
        self.comparison_treeview.delete(item)

     results = self.controller.calculate_performance_metrics()
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
    

    