import tkinter as tk
from tkinter import ttk


class LearningTypeFrame(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        title_label = tk.Label(
            self,
            text="Learning Type Selection",
            bg="#f0f0f0",
            fg="#24367E",
            padx=5,
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=(15, 5))

        desc_label = tk.Label(
            self,
            text="Choose the type of machine learning approach:",
            bg="#f0f0f0",
            fg="#333333",
            font=("Arial", 13, "normal")
        )
        desc_label.pack(pady=(0, 15))

        # Main container frame
        main_frame = tk.Frame(self, bg="#f0f0f0")
        main_frame.pack(pady=10, padx=30, fill=tk.BOTH, expand=True)

        # Learning type selection frame (plus compact)
        selection_frame = tk.Frame(
            main_frame, bg="#ffffff", relief="solid", bd=1)
        selection_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Variable to store selected learning type
        self.learning_type_var = tk.StringVar()

        # Unsupervised Learning option
        unsupervised_frame = tk.Frame(selection_frame, bg="#ffffff")
        unsupervised_frame.pack(fill=tk.X, pady=10, padx=20)

        self.unsupervised_radio = tk.Radiobutton(
            unsupervised_frame,
            text="Unsupervised Learning (Clustering)",
            variable=self.learning_type_var,
            value="unsupervised",
            font=("Arial", 13, "bold"),
            bg="#ffffff",
            fg="#24367E",
            selectcolor="#E8F4FD",
            command=self.on_selection_change
        )
        self.unsupervised_radio.pack(anchor=tk.W)

        unsupervised_desc = tk.Label(
            unsupervised_frame,
            text="• Algorithms: K-Means, K-Medoids, DBSCAN, Diana, Agnes\n• Metrics: Silhouette Score, Inertia",
            font=("Arial", 12, 'normal'),
            bg="#ffffff",
            fg="#666666",
            padx=5,
            justify=tk.LEFT
        )
        unsupervised_desc.pack(anchor=tk.W, padx=15, pady=(2, 8))

        # Separator
        separator1 = tk.Frame(selection_frame, height=1, bg="#e0e0e0")
        separator1.pack(fill=tk.X, padx=20, pady=5)

        # Supervised Learning - Classification option
        classification_frame = tk.Frame(selection_frame, bg="#ffffff")
        classification_frame.pack(fill=tk.X, pady=10, padx=20)

        self.classification_radio = tk.Radiobutton(
            classification_frame,
            text="Supervised Learning (Classification)",
            variable=self.learning_type_var,
            value="supervised",
            font=("Arial", 13, "bold"),
            bg="#ffffff",
            fg="#24367E",
            selectcolor="#E8F4FD",
            command=self.on_selection_change
        )
        self.classification_radio.pack(anchor=tk.W)

        classification_desc = tk.Label(
            classification_frame,
            text="• Algorithms: KNN, Naive Bayes, C4.5\n• Metrics: Accuracy, Precision, Recall, F1-Score",
            font=("Arial", 12, 'normal'),
            bg="#ffffff",
            fg="#666666",
            padx=5,
            justify=tk.LEFT
        )
        classification_desc.pack(anchor=tk.W, padx=15, pady=(2, 8))

        # Separator
        separator2 = tk.Frame(selection_frame, height=1, bg="#e0e0e0")
        separator2.pack(fill=tk.X, padx=20, pady=5)

        # Status frame (compact)
        status_frame = tk.Frame(main_frame, bg="#f0f0f0")
        status_frame.pack(pady=10, fill=tk.X)

        self.status_label = tk.Label(
            status_frame,
            text="Please select a learning type to continue",
            bg="#f0f0f0",
            fg="#666666",
            font=("Arial", 10, "italic")
        )
        self.status_label.pack()

        # Buttons frame (plus visible)
        buttons_frame = tk.Frame(main_frame, bg="#f0f0f0")
        buttons_frame.pack(pady=15, fill=tk.X)

        # Next step button (plus visible)
        self.next_button = tk.Button(
            buttons_frame,
            text="Next Step",
            font=("Arial", 13, "bold"),
            bg="#374451",
            fg="white",
            width=15,
            height=2,
            relief="flat",
            cursor="hand2",
            state=tk.DISABLED,
            command=self.on_next_step
        )
        self.next_button.pack(side=tk.RIGHT, padx=10)

     # Selected learning type handler

    # Selected learning type handler

    def on_selection_change(self):
        selected_type = self.learning_type_var.get()

        if selected_type:
            # Enable next button avec couleur
            self.next_button.config(state=tk.NORMAL,
                                    bg="#24367E",
                                    padx=5,
                                    pady=5)

            # Update status avec couleur verte
            type_names = {
                "unsupervised": "Unsupervised Learning (Clustering)",
                "supervised": "Supervised Learning (Classification)",
            }

            self.status_label.config(
                text=f"Selected: {type_names[selected_type]}",
                fg="#257538",
                font=("Arial", 12, "bold"),
                padx=5
            )

            # update controller state
            self.controller.set_learning_type(selected_type)

    def on_next_step(self):
        selected_type = self.learning_type_var.get()
        if selected_type:
            self.event_generate("<<NextStep>>")
