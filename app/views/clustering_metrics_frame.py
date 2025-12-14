import tkinter as tk
from tkinter import scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class ClusteringMetricsFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="#f0f0f0")
        self.controller = controller
        self.setup_ui()

    def setup_ui(self):
        # Title label
        title_label = tk.Label(
            self,
            text="Clustering analysis",
            bg="#f0f0f0",
            fg="#24367E",
            font=("Arial", 18, "bold")
        )
        title_label.pack(pady=10)

        # Main cluster frame
        main_cluster_frame = tk.Frame(self, bg="#f0f0f0")
        main_cluster_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Grid configuration
        main_cluster_frame.grid_columnconfigure(0, weight=4)
        main_cluster_frame.grid_columnconfigure(1, weight=2)
        main_cluster_frame.grid_rowconfigure(0, weight=1)

        # Left part: graph area
        self.graph_frame = tk.Frame(main_cluster_frame, bg="white", relief="sunken", bd=2)
        self.graph_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Right part: metrics area
        metrics_frame = tk.Frame(main_cluster_frame, bg="#f8f9fa", relief="raised", bd=2)
        metrics_frame.grid(row=0, column=1, sticky="nsew")

        # Metrics title
        metrics_title = tk.Label(
            metrics_frame,
            text="Clustering metrics",
            bg="#f8f9fa",
            fg="#24367E",
            font=("Arial", 14, "bold")
        )
        metrics_title.pack(pady=10)

        # Metrics labels
        self.optimal_k_label = tk.Label(
            metrics_frame,
            text="Optimal K: Calculating...",
            bg="#f8f9fa",
            fg="#333",
            font=("Arial", 13)
        )
        self.optimal_k_label.pack(pady=10)

        self.silhouette_label = tk.Label(
            metrics_frame,
            text="Silhouette Score: Calculating...",
            bg="#f8f9fa",
            fg="#333",
            font=("Arial", 13)
        )
        self.silhouette_label.pack(pady=5)

        # Analyze clusters button
        self.analyze_btn = tk.Button(
            metrics_frame,
            text="Analyze Clusters",
            bg="#7b9fc2",
            fg="white",
            font=("Arial", 12, "bold"),
            width=15,
            height=2,
            relief="raised",
            bd=2,
            command=self.analyze_clusters
        )
        self.analyze_btn.pack(pady=20)

        # Next step button
        self.next_step_button = tk.Button(
            metrics_frame,
            text="Next Step",
            bg="#374451",
            fg="white",
            font=("Arial", 12, "bold"),
            width=15,
            height=2,
            relief="raised",
            bd=2,
            state=tk.DISABLED,
            command=self.on_next_step
        )
        self.next_step_button.pack(pady=10)

        # Back button
        self.back_button = tk.Button(
            metrics_frame,
            text="Back",
            bg="#6c757d",
            fg="white",
            font=("Arial", 12, "bold"),
            width=15,
            height=2,
            relief="raised",
            bd=2,
            command=self.on_back_step
        )
        self.back_button.pack(pady=5)

    def analyze_clusters(self):
        """Analyze clusters and display results"""
        try:
            # Get clustering analysis from controller
            result = self.controller.analyze_clusters()
            
            if result.get('error'):
                self.optimal_k_label.config(text="Error in analysis")
                self.silhouette_label.config(text="Please try again")
                return
            
            # Update labels with results
            self.optimal_k_label.config(text=f"Optimal K: {result['optimal_k']}")
            self.silhouette_label.config(text=f"Silhouette Score: {result['silhouette_score']:.3f}")
            
            # Plot elbow curve
            self.plot_elbow_curve(result)
            
            # Enable next step button
            self.next_step_button.config(state=tk.NORMAL, bg="#24367E")
            
        except Exception as e:
            print(f"Error in cluster analysis: {e}")
            self.optimal_k_label.config(text="Error in analysis")
            self.silhouette_label.config(text="Please try again")

    def plot_elbow_curve(self, result):
        """Plot elbow curve"""
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(6, 4))
        
        k_range = result['k_range']
        wcss_values = result['wcss_values']
        optimal_k = result['optimal_k']
        
        ax.plot(k_range, wcss_values, 'bo-', linewidth=2, markersize=6)
        ax.scatter(optimal_k, wcss_values[optimal_k-1], color='red', s=100, zorder=5)
        ax.annotate(f'Optimal K = {optimal_k}',
                   xy=(optimal_k, wcss_values[optimal_k-1]),
                   xytext=(optimal_k+1, wcss_values[optimal_k-1]+200),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=10, color='red')

        ax.set_title('Elbow Method for Optimal K', fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Clusters (K)')
        ax.set_ylabel('WCSS')
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def on_next_step(self):
        """Handle next step button click"""
        self.event_generate("<<NextStep>>")

    def on_back_step(self):
        """Handle back step button click"""
        self.event_generate("<<PreviousStep>>")