# views/preprocessing_frame.py
import tkinter as tk
from tkinter import ttk, scrolledtext


class PreprocessingFrame(ttk.Frame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # Title label
        titleLabel = tk.Label(
            self,
            text="Data preprocessing",
            bg="#f0f0f0",
            fg="#24367E",
            font=("Arial", 18, "bold"),
            padx=20
        )
        titleLabel.pack(pady=15)

        # Steps buttons frame
        stepsFrame = tk.Frame(self, bg="#f0f0f0")
        stepsFrame.pack(pady=10, fill=tk.X, padx=30)

        # Grid configuration for steps buttons
        stepsFrame.grid_columnconfigure(0, weight=1)
        stepsFrame.grid_columnconfigure(1, weight=1)
        stepsFrame.grid_columnconfigure(2, weight=1)
        stepsFrame.grid_columnconfigure(3, weight=1)
        stepsFrame.grid_columnconfigure(4, weight=1)

        # 1 - Missing values button
        self.missing_values_button = tk.Button(
            stepsFrame,
            text="1. Analyze missing values",
            bg="#24367E",
            fg="white",
            font=("Arial", 12, "bold"),
            width=16,
            height=2,
            relief="raised",
            bd=2,
            padx=5,
            command=self.analyze_missing_values
        )
        self.missing_values_button.grid(
            row=0, column=0, padx=5, pady=10, sticky="ew")

        # 2 - Outliers button
        self.outliers_button = tk.Button(
            stepsFrame,
            text="2. Detect outliers",
            bg="#374451",
            fg="white",
            font=("Arial", 12, "bold"),
            width=16,
            height=2,
            relief="raised",
            bd=2,
            padx=5,
            command=self.analyze_outliers
        )
        self.outliers_button.grid(
            row=0, column=1, padx=5, pady=10, sticky="ew")
        self.outliers_button.config(state=tk.DISABLED)

        # 3 - Normalization button
        self.normalization_button = tk.Button(
            stepsFrame,
            text="3. Normalize data",
            bg="#374451",
            fg="white",
            font=("Arial", 12, "bold"),
            width=16,
            height=2,
            relief="raised",
            bd=2,
            padx=5,
            command=self.normalize_data
        )
        self.normalization_button.grid(
            row=0, column=2, padx=5, pady=10, sticky="ew")
        self.normalization_button.config(state=tk.DISABLED)

        self.nextStepPreprocessing = tk.Button(
            stepsFrame,
            text="Next Step",
            bg="#4F545A",
            fg="white",
            font=("Arial", 12, "bold"),
            width=16,
            height=2,
            relief="raised",
            bd=2,
            padx=5,
            command=self.on_next_step_click
        )
        self.nextStepPreprocessing.grid(row=0, column=3, columnspan=3, pady=10)
        self.nextStepPreprocessing.config(state=tk.DISABLED)

        # Separator
        separator = tk.Frame(self, height=2, bg="#7b9fc2")
        separator.pack(fill=tk.X, padx=30, pady=15)

        resultsFrame = tk.Frame(self, bg="#f0f0f0")
        resultsFrame.pack(pady=5, fill=tk.BOTH, expand=True, padx=30)

        self.resultsText = scrolledtext.ScrolledText(
            resultsFrame,
            height=12,
            font=("Courier", 12),
            bg="white",
            fg="#333333",
            wrap="word",
            relief="sunken",
            bd=2
        )
        self.resultsText.pack(fill=tk.BOTH, expand=True)

        # Initialize text area
        self._initialize_results_text()
        
        
    def _initialize_results_text(self):
        """Initialize the results text area with default message"""
        self.resultsText.insert(tk.END, "Preprocessing panel:\n")
        self.resultsText.insert(tk.END, "=" * 25 + "\n\n")
        self.resultsText.insert(
            tk.END,
            "Select The analysis & fill missing values step to begin.\n\n"
        )

    def on_next_step_click(self):
        """Handler for next step button"""
        # Emit event to notify main view
        self.event_generate("<<NextStep>>")

    def update_results(self, text: str):
        """Update the results text area"""
        self.resultsText.delete(1.0, tk.END)
        self.resultsText.insert(tk.END, text)

    def enable_outliers_button(self):
        """Enable outliers detection button"""
        self.missing_values_button.config(state=tk.DISABLED, bg="#374451")
        self.outliers_button.config(state=tk.NORMAL, bg="#24367E")
        self.normalization_button.config(state=tk.DISABLED, bg="#374451")

    def enable_normalization_button(self):
        """Enable normalization button"""
        self.missing_values_button.config(state=tk.DISABLED, bg="#374451")
        self.outliers_button.config(state=tk.DISABLED, bg="#374451")
        self.normalization_button.config(state=tk.NORMAL, bg="#24367E")

    def enable_next_step_button(self):
        """Enable next step button"""
        self.missing_values_button.config(state=tk.DISABLED, bg="#374451")
        self.outliers_button.config(state=tk.DISABLED, bg="#374451")
        self.normalization_button.config(state=tk.DISABLED, bg="#374451")
        self.nextStepPreprocessing.config(state=tk.NORMAL, bg="#24367E")

    def setup_buttons(self):
        self.missing_values_button.config(command=self.analyze_missing_values)
        self.outliers_button.config(command=self.analyze_outliers)
        self.normalization_button.config(command=self.normalize_data)
        self.nextStepPreprocessing.config(command=self.on_next_step)

    def analyze_missing_values(self):
        """Handle missing values analysis"""
        self.resultsText.delete(1.0, tk.END)
        self.resultsText.insert(tk.END, "Analyzing missing values...\n")

        result = self.controller.analyze_missing_values()

        if result['error']:
            self.resultsText.insert(tk.END, f"ERROR: {result['message']}\n\n")
            return

        # Display results
        self.display_missing_values_results(result)

        # Update button states
        self.missing_values_button.config(state=tk.DISABLED, bg="#374451")
        self.outliers_button.config(state=tk.NORMAL, bg="#24367E")

    def analyze_outliers(self):
        """Handle outliers analysis"""
        self.resultsText.delete(1.0, tk.END)
        self.resultsText.insert(tk.END, "Detecting outliers...\n")

        result = self.controller.analyze_outliers()

        if result['error']:
            self.resultsText.insert(tk.END, f"ERROR: {result['message']}\n\n")
            return

        # Display results
        self.display_outliers_results(result)

        # Update button states
        self.outliers_button.config(state=tk.DISABLED, bg="#374451")
        self.normalization_button.config(state=tk.NORMAL, bg="#24367E")

    def normalize_data(self):
        """Handle data normalization"""
        self.resultsText.delete(1.0, tk.END)
        self.resultsText.insert(tk.END, "Normalizing data...\n")

        result = self.controller.normalize_data()

        if result['error']:
            self.resultsText.insert(tk.END, f"ERROR: {result['message']}\n\n")
            return

        # Display results
        self.display_normalization_results(result)

        # Update button states
        self.normalization_button.config(state=tk.DISABLED, bg="#374451")
        self.nextStepPreprocessing.config(state=tk.NORMAL, bg="#24367E")

    def display_missing_values_results(self, result):
        """Display missing values analysis results"""
        self.resultsText.insert(tk.END, "Missing values analysis:\n")
        self.resultsText.insert(tk.END, "=" * 25 + "\n\n")

        self.resultsText.insert(tk.END, f"Dataset Info:\n")
        self.resultsText.insert(
            tk.END, f"- Original shape: {result['original_shape'][0]} rows, {result['original_shape'][1]} columns\n")
        self.resultsText.insert(
            tk.END, f"- New shape: {result['new_shape'][0]} rows, {result['new_shape'][1]} columns\n")
        self.resultsText.insert(
            tk.END, f"- Total missing values found: {result['total_missing']}\n")
        self.resultsText.insert(
            tk.END, f"- Numeric missing values: {result['numeric_missing']}\n")
        self.resultsText.insert(
            tk.END, f"- Categorical missing values: {result['categorical_missing']}\n\n")

        # Display actions taken
        if result['dropped_columns']:
            self.resultsText.insert(
                tk.END, f"Dropped columns (>50% missing): {len(result['dropped_columns'])} columns\n")

        if result['numeric_columns_filled']:
            self.resultsText.insert(
                tk.END, f"Numeric columns filled with median: {len(result['numeric_columns_filled'])} columns\n")

        if result['categorical_columns_filled']:
            self.resultsText.insert(
                tk.END, f"Categorical columns filled with mode: {len(result['categorical_columns_filled'])} columns\n")

        self.resultsText.insert(
            tk.END, "Dataset is now ready for outlier detection.\n")

    def display_outliers_results(self, result):
        """Display outliers detection results"""
        self.resultsText.insert(tk.END, "Outliers detection analysis:\n")
        self.resultsText.insert(tk.END, "=" * 30 + "\n\n")

        self.resultsText.insert(tk.END, f"Outliers Detection Info:\n")
        self.resultsText.insert(
            tk.END, f"- Numeric columns analyzed: {result['numeric_columns']}\n")
        self.resultsText.insert(
            tk.END, f"- Columns with outliers: {result['columns_with_outliers']}\n")
        self.resultsText.insert(
            tk.END, f"- Total outliers found: {result['total_outliers']}\n\n")

        if result['outliers_info']:
            self.resultsText.insert(tk.END, "Outliers by Column:\n")
            for column, info in result['outliers_info'].items():
                self.resultsText.insert(
                    tk.END, f"{column}: {info['count']} outliers ({info['percentage']:.1f}%)\n")
        else:
            self.resultsText.insert(
                tk.END, "No outliers found in any numeric columns.\n")

        self.resultsText.insert(
            tk.END, "Dataset is now ready for normalization.\n")

    def display_normalization_results(self, result):
        """Display normalization results"""
        self.resultsText.insert(tk.END, "Data normalization:\n")
        self.resultsText.insert(tk.END, "=" * 25 + "\n\n")

        self.resultsText.insert(tk.END, f"Dataset Info:\n")
        self.resultsText.insert(
            tk.END, f"- Categorical columns: {result['categorical_columns']}\n")
        self.resultsText.insert(
            tk.END, f"- Numeric columns: {result['numeric_columns']}\n")

        if result['encoded_categorical']:
            self.resultsText.insert(
                tk.END, f"Encoded categorical columns: {len(result['encoded_categorical'])}\n")

        if result['scaled_features']:
            self.resultsText.insert(
                tk.END, f"Scaled numeric features: {len(result['scaled_features'])}\n")

        self.resultsText.insert(
            tk.END, "Dataset is now ready for clustering analysis.\n")

    def on_next_step(self):
        """Handle next step button click"""
        if self.controller.is_preprocessing_complete():
            self.event_generate("<<NextStep>>")  # Ceci ira vers Learning Type maintenant
        else:
            from tkinter import messagebox
            messagebox.showwarning("Warning", "Please complete all preprocessing steps before proceeding.")
