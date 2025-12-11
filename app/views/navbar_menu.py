# views/navbar_menu.py
import tkinter as tk
from tkinter import ttk
from app.config.constants import menuButtons, get_menu_buttons_for_learning_type

class NavbarMenu(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.controller = None
        self.current_step = 0

        self.config(padding=0, relief='flat', height=50)

        # Create buttons based on constants
        self.buttons = {}
        self.create_buttons(menuButtons)  # Use base buttons initially
    
    def create_buttons(self, button_list):
        """Create buttons from button list"""
        # Clear existing buttons
        for widget in self.winfo_children():
            widget.destroy()
        self.buttons = {}
        
        # Create new buttons
        for button_name in button_list:
            button = tk.Button(
                self, 
                text=button_name, 
                bg="#d9e4f5", 
                height=1, 
                fg="#24367E", 
                font=("Arial", 13), 
                bd=0, 
                relief='ridge', 
                highlightthickness=5, 
                highlightbackground="#7b9fc2"
            )
            button.pack(side=tk.LEFT, padx=10, pady=15)
            self.buttons[button_name] = button

    def update_for_learning_type(self, learning_type):
        """Update navbar based on learning type"""
        new_buttons = get_menu_buttons_for_learning_type(learning_type)
        self.create_buttons(new_buttons)
        
        # Restore current step highlighting
        self.update_active_step(self.current_step)

    def update_active_step(self, step_index: int):
        """Update the active step visual indicator"""
        self.current_step = step_index
        
        for i, btn in enumerate(self.buttons.values()):
            if i == step_index:
                # Active step
                btn.config(bg="#0F1737", fg="white")
            elif i < step_index:
                # Completed steps
                btn.config(bg="#24367E", fg="white")
            else:
                # Future steps
                btn.config(bg="#d9e4f5", fg="#24367E")   

    def set_controller(self, controller):
        """Set controller reference"""
        self.controller = controller
                  
    def get_current_step(self) -> int:
        """Get current step index"""
        return self.current_step