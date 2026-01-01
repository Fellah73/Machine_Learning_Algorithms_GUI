import tkinter as tk
from tkinter import scrolledtext
from app.utils.asset_manager import asset_manager

class AlgorithmsFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#f0f0f0")
        self.controller = controller
        self.current_algorithm_type = None
        self.selected_algorithm = None
        self.algorithm_buttons = {}
        self.setup_ui()
    
    def setup_ui(self): 
        title_label = tk.Label(
            self,
            text="Algorithm selection",
            bg="#f0f0f0",
            fg="#24367E",
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=10)

        # Get learning type and setup appropriate interface
        learning_type = self.controller.get_learning_type()
         
        
        if learning_type == "unsupervised":
            self.setup_unsupervised_algorithms()
        elif learning_type == "supervised":
            self.setup_supervised_algorithms()
        else : 
            print(f"Unknown learning type '{learning_type}'")   

    def setup_unsupervised_algorithms(self):
        # Algorithm type buttons frame
        type_buttons_frame = tk.Frame(self, bg="#f0f0f0")
        type_buttons_frame.pack(pady=10, fill=tk.X, padx=30)

        # Grid configuration for type buttons
        type_buttons_frame.grid_columnconfigure(0, weight=1)
        type_buttons_frame.grid_columnconfigure(1, weight=1)
        type_buttons_frame.grid_columnconfigure(2, weight=1)

        # Partitioning button
        self.partitioning_btn = tk.Button(
            type_buttons_frame,
            text="Partitioning",
            bg="#24367E",
            fg="white",
            font=("Arial", 14, "bold"),
            width=15,
            height=2,
            relief="raised",
            bd=2,
            command=lambda: self.show_algorithm_family("Partitioning")
        )
        self.partitioning_btn.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Hierarchical button
        self.hierarchical_btn = tk.Button(
            type_buttons_frame,
            text="Hierarchical",
            bg="#374451",
            fg="white",
            font=("Arial", 14, "bold"),
            width=15,
            height=2,
            relief="raised",
            bd=2,
            command=lambda: self.show_algorithm_family("Hierarchical")
        )
        self.hierarchical_btn.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # Density-based button
        self.density_btn = tk.Button(
            type_buttons_frame,
            text="Density-based",
            bg="#374451",
            fg="white",
            font=("Arial", 14, "bold"),
            width=15,
            height=2,
            relief="raised",
            bd=2,
            command=lambda: self.show_algorithm_family("Density-based")
        )
        self.density_btn.grid(row=0, column=2, padx=10, pady=10, sticky="ew")

        self.setup_content_area()
        self.show_algorithm_family("Partitioning")  # Default selection

    def setup_supervised_algorithms(self):
        """Setup supervised learning algorithms interface"""
        # Algorithm type buttons frame
        type_buttons_frame = tk.Frame(self, bg="#f0f0f0")
        type_buttons_frame.pack(pady=10, fill=tk.X, padx=30)

        # Grid configuration for type buttons
        type_buttons_frame.grid_columnconfigure(0, weight=1)
        type_buttons_frame.grid_columnconfigure(1, weight=1)
        type_buttons_frame.grid_columnconfigure(2, weight=1)

        # Lazy Learning button
        self.lazy_btn = tk.Button(
            type_buttons_frame,
            text="Lazy Learning",
            bg="#24367E",
            fg="white",
            font=("Arial", 14, "bold"),
            width=15,
            height=2,
            relief="raised",
            bd=2,
            command=lambda: self.show_algorithm_family("Lazy Learning")
        )
        self.lazy_btn.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Probabilistic button
        self.probabilistic_btn = tk.Button(
            type_buttons_frame,
            text="Probabilistic",
            bg="#374451",
            fg="white",
            font=("Arial", 14, "bold"),
            width=15,
            height=2,
            relief="raised",
            bd=2,
            command=lambda: self.show_algorithm_family("Probabilistic")
        )
        self.probabilistic_btn.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # Tree-based button
        self.tree_btn = tk.Button(
            type_buttons_frame,
            text="Tree-based",
            bg="#374451",
            fg="white",
            font=("Arial", 14, "bold"),
            width=15,
            height=2,
            relief="raised",
            bd=2,
            command=lambda: self.show_algorithm_family("Tree-based")
        )
        self.tree_btn.grid(row=0, column=2, padx=10, pady=10, sticky="ew")

        self.setup_content_area()
        self.show_algorithm_family("Lazy Learning")  # Default selection

    def setup_content_area(self):
        # Main content frame
        self.main_content_frame = tk.Frame(self, bg="#f0f0f0")
        self.main_content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Grid configuration
        self.main_content_frame.grid_columnconfigure(0, weight=2)
        self.main_content_frame.grid_columnconfigure(1, weight=1)
        self.main_content_frame.grid_rowconfigure(0, weight=1)

        # Left part: Description and algorithms
        self.description_frame = tk.Frame(
            self.main_content_frame, bg="white", relief="sunken", bd=2
        )
        self.description_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Description text area
        self.desc_text = scrolledtext.ScrolledText(
            self.description_frame,
            height=8,
            width=50,
            font=("Arial", 10),
            bg="#f9f9f9",
            wrap="word",
            relief="flat",
            bd=1
        )
        self.desc_text.pack(pady=5, padx=10, fill=tk.X)

        # Algorithms list frame
        self.algorithms_list_frame = tk.Frame(self.description_frame, bg="white")
        self.algorithms_list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Right part: Image and navigation
        self.right_frame = tk.Frame(
            self.main_content_frame, bd=2
        )
        self.right_frame.grid(row=0, column=1, sticky="nsew")

        # Image display frame
        self.image_display_frame = tk.Frame(
            self.right_frame, relief="sunken", bd=1, width=250, height=210
        )
        self.image_display_frame.pack(pady=0, padx=0)
        self.image_display_frame.pack_propagate(True)

        # Image label
        self.image_label = tk.Label(
            self.image_display_frame,
            fg="#666",
            font=("Arial", 10),
            width=240,
            height=200,
        )
        self.image_label.pack(expand=True, fill=tk.BOTH)

        # Navigation buttons
        self.setup_navigation_buttons()

    def setup_navigation_buttons(self):
        nav_frame = tk.Frame(self.right_frame, bg="#f8f9fa")
        nav_frame.pack(pady=20)

        # Next step button
        self.next_button = tk.Button(
            nav_frame,
            text="Next Step →",
            bg="#374451",
            fg="white",
            font=("Arial", 11, "bold"),
            width=15,
            height=2,
            relief="flat",
            cursor="hand2",
            state=tk.DISABLED,
            command=self.on_next_step
        )
        self.next_button.pack(pady=5)

    def show_algorithm_family(self, family_type):
     self.current_algorithm_type = family_type
     self.selected_algorithm = None

     # Reset type buttons based on learning type
     learning_type = self.controller.get_learning_type()
     
     if learning_type == "unsupervised":
        # Reset unsupervised buttons
        if hasattr(self, 'partitioning_btn'):
            self.partitioning_btn.config(bg="#374451")
        if hasattr(self, 'hierarchical_btn'):
            self.hierarchical_btn.config(bg="#374451")
        if hasattr(self, 'density_btn'):
            self.density_btn.config(bg="#374451")

        # Highlight selected button
        if family_type == "Partitioning" and hasattr(self, 'partitioning_btn'):
            self.partitioning_btn.config(bg="#24367E")
        elif family_type == "Hierarchical" and hasattr(self, 'hierarchical_btn'):
            self.hierarchical_btn.config(bg="#24367E")
        elif family_type == "Density-based" and hasattr(self, 'density_btn'):
            self.density_btn.config(bg="#24367E")
            
     else:
        # Reset supervised buttons
        if hasattr(self, 'lazy_btn'):
            self.lazy_btn.config(bg="#374451")
        if hasattr(self, 'probabilistic_btn'):
            self.probabilistic_btn.config(bg="#374451")
        if hasattr(self, 'tree_btn'):
            self.tree_btn.config(bg="#374451")

        # Highlight selected button
        if family_type == "Lazy Learning" and hasattr(self, 'lazy_btn'):
            self.lazy_btn.config(bg="#24367E")
        elif family_type == "Probabilistic" and hasattr(self, 'probabilistic_btn'):
            self.probabilistic_btn.config(bg="#24367E")
        elif family_type == "Tree-based" and hasattr(self, 'tree_btn'):
            self.tree_btn.config(bg="#24367E")

     # Display family info
     self.display_family_info(family_type)

     # Create algorithm buttons
     self.create_algorithm_buttons(family_type)

     # Disable next button
     self.next_button.config(state=tk.DISABLED, bg="#374451")
    
    def display_family_info(self, family_type):
        # Get algorithm families data
        algorithms_data = self.controller.get_algorithms_data()
        
        if family_type not in algorithms_data:
            return

        family_info = algorithms_data[family_type]

        # Update description
        self.desc_text.config(state=tk.NORMAL)
        self.desc_text.delete(1.0, tk.END)

        description = f"{family_type} algorithms\n"
        description += f"{family_info['description']}\n\n"
        description += "Available algorithms:\n\n"

        for algo_name in family_info['algorithms'].keys():
            description += f"• {algo_name}\n"

        self.desc_text.insert(tk.END, description)
        self.desc_text.config(state=tk.DISABLED)

        # Load and display image
        self.load_family_image(family_info.get('image', ''))

    def load_family_image(self, image_url=""):
     try:
        learning_type = self.controller.get_learning_type()
        family_type = self.current_algorithm_type
         
        photo = asset_manager.get_algorithm_image(learning_type, family_type)
        
        if photo:
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep reference
            self.image_label.grid(row=0, column=0, sticky="nsew")

     except Exception as e:
        print(f"Error in load_family_image for {self.current_algorithm_type}: {e}")

    def create_algorithm_buttons(self, family_type):
        # Clear existing buttons
        for widget in self.algorithms_list_frame.winfo_children():
            widget.destroy()
            
        self.algorithm_buttons.clear()

        algorithms_data = self.controller.get_algorithms_data()
        if family_type not in algorithms_data:
            return

        algorithms = algorithms_data[family_type]['algorithms']

        # Grid frame for algorithms
        grid_frame = tk.Frame(self.algorithms_list_frame, bg="white")
        grid_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Grid configuration
        grid_frame.grid_columnconfigure(0, weight=1)
        grid_frame.grid_columnconfigure(1, weight=1)

        algorithm_names = list(algorithms.keys())

        # Create buttons for each algorithm
        for i, algo_name in enumerate(algorithm_names):
            row = i // 2
            col = i % 2

            # Algorithm frame
            algo_frame = tk.Frame(grid_frame, bg="#f9f9f9", relief="solid", bd=1)
            algo_frame.grid(row=row, column=col, sticky="nsew", padx=3, pady=3)

            # Grid configuration for algorithm frame
            algo_frame.grid_rowconfigure(0, weight=0)
            algo_frame.grid_rowconfigure(1, weight=0)
            algo_frame.grid_rowconfigure(2, weight=1)
            algo_frame.grid_columnconfigure(0, weight=1)

            name_label = tk.Label(
                algo_frame,
                text=f"{algo_name}",
                bg="#f9f9f9",
                fg="#24367E",
                font=("Arial", 11, "bold")
            )
            name_label.grid(row=0, column=0, pady=(8, 2), sticky="ew")

            # Parameters info
            params = algorithms[algo_name].get('parameters', [])
            params_text = ", ".join(params) if params else "No parameters"
            params_label = tk.Label(
                algo_frame,
                text=params_text,
                bg="#f9f9f9",
                fg="black",
                font=("Arial", 13),
                wraplength=120
            )
            params_label.grid(row=1, column=0, pady=1, sticky="ew")

            # Select button
            select_btn = tk.Button(
                algo_frame,
                text="Select",
                bg="#7b9fc2",
                fg="white",
                font=("Arial", 9, "bold"),
                width=15,
                height=1,
                relief="raised",
                bd=2,
                cursor="hand2",
                command=lambda name=algo_name: self.select_algorithm(name)
            )
            select_btn.grid(row=2, column=0, pady=(0, 8), padx=10, sticky="ew")

            grid_frame.grid_rowconfigure(row, weight=1)

            # Store button reference
            self.algorithm_buttons[algo_name] = {
                'frame': algo_frame,
                'name_label': name_label,
                'button': select_btn
            }

    def select_algorithm(self, algo_name):
        self.selected_algorithm = algo_name

        # Update controller
        self.controller.set_selected_algorithm(algo_name, self.current_algorithm_type)

        # Update button styles
        self.update_algorithm_selection_styles(algo_name)

        # Enable next step button
        self.next_button.config(state=tk.NORMAL, bg="#24367E")

    def update_algorithm_selection_styles(self, selected_name):
        """Update visual styles for selected algorithm"""
        for algo_name, components in self.algorithm_buttons.items():
            if algo_name == selected_name:
                # Selected style
                components['frame'].config(bg="#d4edda", relief="solid", bd=2)
                components['name_label'].config(bg="#d4edda")
                components['button'].config(
                    text=f"{selected_name} (Selected)",
                    bg="#28a745"
                )
            else:
                # Default style
                components['frame'].config(bg="#f9f9f9", relief="solid", bd=1)
                components['name_label'].config(bg="#f9f9f9")
                components['button'].config(text="Select", bg="#7b9fc2")

    def on_next_step(self):
        """Handle next step button click"""
        if self.selected_algorithm:
            self.event_generate("<<NextStep>>")
