# views/main_view.py
import tkinter as tk
from tkinter import ttk, messagebox

from app.views.navbar_menu import NavbarMenu
from app.views.uploadDataSet_frame import UploadDataSetFrame
from app.views.preprocessing_frame import PreprocessingFrame
from app.views.learningType_frame import LearningTypeFrame
from app.controllers.app_controller import AppController
from app.config.constants import step_mapping
from app.views.clustering_metrics_frame import ClusteringMetricsFrame
from app.views.algorithms_frame import AlgorithmsFrame
from app.views.visualization_frame import VisualizationFrame
from app.views.unsupervised_comparaison_frame import UnsupervisedComparisonFrame
from app.views.supervised_comparison_frame import SupervisedComparisonFrame


class MainView(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Machine Learning GUI")
        self.geometry("1000x600")
        self.resizable(False, False)
        self.configure(bg="#f0f0f0")

        # Initialize controller
        self.controller = AppController()
        self.controller.set_view(self)

        # -------- NAVBAR CONTAINER --------
        self.navbar_container = ttk.Frame(self)
        self.navbar_container.pack(side=tk.TOP, fill=tk.X, pady=10)

        # -------- NAVBAR --------
        self.navbar = NavbarMenu(self.navbar_container)
        self.navbar.pack(anchor='center')
        self.navbar.set_controller(self.controller)

        # -------- MAIN AREA --------
        self.container = ttk.Frame(self)
        self.container.pack(fill=tk.BOTH, expand=True)

        # Dictionary to store frames
        self.frames = {}

        # Create all frames
        self._create_frames()

        # Show initial frame
        self.show_current_step()

    def _create_frames(self):
        """Create all application frames"""
        # Upload frame
        self.frames['upload'] = UploadDataSetFrame(
            self.container, self.controller)
        self.frames['upload'].bind("<<NextStep>>", self.on_next_step)

        # Preprocessing frame
        self.frames['preprocessing'] = PreprocessingFrame(
            self.container, self.controller)
        self.frames['preprocessing'].bind(
            "<<NextStep>>", self.on_preprocessing_next)

        # Learning Type frame - NOUVEAU
        self.frames['learning_type'] = LearningTypeFrame(
            self.container, self.controller)
        self.frames['learning_type'].bind(
            "<<NextStep>>", self.on_learning_type_next)
        self.frames['learning_type'].bind(
            "<<PreviousStep>>", self.on_learning_type_back)

        # Clustering Metrics frame - NOUVEAU
        self.frames['clustering_metrics'] = ClusteringMetricsFrame(
            self.container, self.controller)
        self.frames['clustering_metrics'].bind(
            "<<NextStep>>", self.on_clustering_metrics_next)
        self.frames['clustering_metrics'].bind(
            "<<PreviousStep>>", self.on_clustering_metrics_back)

        # Place all frames in the same location
        for frame in self.frames.values():
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)

    def show_current_step(self):
        """Show frame corresponding to current step with dynamic mapping"""
        current_step = self.controller.get_current_step()
        learning_type = self.controller.get_learning_type()

        # Update navbar
        self.navbar.update_active_step(current_step)

        # Get dynamic step mapping based on learning type
        current_step_mapping = step_mapping.get(
            learning_type, step_mapping["unsupervised"])

        # Get frame name for current step
        frame_name = current_step_mapping.get(current_step)

        # Créer le frame algorithms seulement quand nécessaire
        if frame_name == 'algorithms' and 'algorithms' not in self.frames:
            self.frames['algorithms'] = AlgorithmsFrame(
                self.container, self.controller)
            self.frames['algorithms'].bind(
                "<<NextStep>>", self.on_algorithms_next)
            self.frames['algorithms'].place(
                relx=0, rely=0, relwidth=1, relheight=1)

        # Add visualization frame creation
        if frame_name == 'visualization' and 'visualization' not in self.frames:
            self.frames['visualization'] = VisualizationFrame(
                self.container, self.controller)
            self.frames['visualization'].bind(
                "<<NextStep>>", self.on_visualization_next)
            self.frames['visualization'].place(
                relx=0, rely=0, relwidth=1, relheight=1)

        # Comparison frame
        if frame_name == 'unsup_comparison' and 'unsup_comparison' not in self.frames:
            self.frames['unsup_comparison'] = UnsupervisedComparisonFrame(
                self.container, self.controller)
            self.frames['unsup_comparison'].bind(
                "<<PreviousStep>>", self.on_previous_step)
            self.frames['unsup_comparison'].place(
                relx=0, rely=0, relwidth=1, relheight=1)
            
        # Comparison frame
        if frame_name == 'sup_comparison' and 'sup_comparison' not in self.frames:
            self.frames['sup_comparison'] = SupervisedComparisonFrame(
                self.container, self.controller)
            self.frames['sup_comparison'].bind(
                "<<PreviousStep>>", self.on_previous_step)
            self.frames['sup_comparison'].place(
                relx=0, rely=0, relwidth=1, relheight=1)

        if frame_name and frame_name in self.frames:
            self.frames[frame_name].lift()

    def on_next_step(self, event=None):
        """Handler for moving to next step (général)"""
        if self.controller.can_proceed_to_next_step():
            self.controller.increment_step()
            self.show_current_step()
        else:
            # Show warning if dataset not loaded
            messagebox.showwarning(
                "Warning",
                "Please complete the current step before proceeding."
            )

    def on_preprocessing_next(self, event=None):
        """Handler spécifique pour le passage du preprocessing vers learning type"""
        if self.controller.is_preprocessing_complete():
            self.controller.increment_step()  # Passe à l'étape 2 (learning_type)
            self.show_current_step()
        else:
            messagebox.showwarning(
                "Warning",
                "Please complete all preprocessing steps before proceeding."
            )

    def on_learning_type_back(self, event=None):
        """Handler pour retourner au preprocessing depuis learning type"""
        self.controller.decrement_step()  # Retour à l'étape 1 (preprocessing)
        self.show_current_step()

    def on_previous_step(self, event=None):
        """Handler for moving to previous step (général)"""
        self.controller.decrement_step()
        self.show_current_step()

    def update_navigation_for_learning_type(self, learning_type):
        """Update navigation when learning type is selected"""
        self.navbar.update_for_learning_type(learning_type)

    def on_learning_type_next(self, event=None):
        """Handler pour le passage du learning type vers la suite"""
        if self.controller.is_learning_type_selected():
            learning_type = self.controller.get_learning_type()

            # Update navigation based on learning type
            self.update_navigation_for_learning_type(learning_type)

            if learning_type == "unsupervised":
                # Go to step  (Clustering Metrics)
                self.controller.set_current_step(3)

            else:
                # For supervised, go to step 3 but it maps to algorithms
                self.controller.set_current_step(3)

            self.show_current_step()
        else:
            messagebox.showwarning(
                "Warning", "Please select a learning type first.")

    def on_clustering_metrics_next(self, event=None):
        """Handler for next step from clustering metrics"""
        self.controller.increment_step()
        self.show_current_step()

    def on_clustering_metrics_back(self, event=None):
        """Handler for back step from clustering metrics"""
        self.controller.decrement_step()
        self.show_current_step()

    def on_algorithms_next(self, event=None):
        """Handler for next step from algorithms"""
        if self.controller.is_algorithm_selected():
            self.controller.increment_step()
            self.show_current_step()
        else:
            print("no algorithm selected",
                  self.controller.is_algorithm_selected())
            messagebox.showwarning(
                "Warning", "Please select an algorithm first.")

    def on_visualization_next(self, event=None):
     """Handler for next step from visualization"""    
     try:
        self.controller.increment_step()
        self.show_current_step()
     except Exception as e:
        print(f"Error in visualization next: {e}")  # Debug
  
    def on_comparison_next(self, event=None):
        messagebox.showinfo("Info", "Comparison completed successfully!")
