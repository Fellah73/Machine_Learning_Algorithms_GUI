# views/main_frame.py
import tkinter as tk
from tkinter import ttk


class UploadDataSetFrame(ttk.Frame):
    """Main upload frame for dataset selection"""

    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(style='Main.TFrame')

        # Upload dataSet frame
        uploadDataSetFrame = tk.Frame(self, bg="#f0f0f0", height=400)
        uploadDataSetFrame.pack(fill=tk.BOTH, expand=True)

        # Center frame
        centerFrame = tk.Frame(uploadDataSetFrame, bg="#f0f0f0")
        centerFrame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # Label
        self.uploadLabel = tk.Label(
            centerFrame,
            text="upload your dataset here",
            bg="#f0f0f0",
            fg="#24367E",
            font=("Arial", 18),
            padx=10,
        )
        self.uploadLabel.pack(pady=20)

        # Buttons frame
        buttonFrame = tk.Frame(centerFrame, bg="#f0f0f0")
        buttonFrame.pack(pady=10)

        # Upload button
        self.uploadDataSetButton = tk.Button(
            buttonFrame,
            text="Choose File",
            bg="#7b9fc2",
            fg="white",
            font=("Arial", 16),
            bd=0,
            relief='ridge',
            highlightthickness=5,
            highlightbackground="#7b9fc2",
            command=self.on_upload_click
        )
        self.uploadDataSetButton.pack(side=tk.LEFT, padx=10)

        # NextStep button
        self.nextStepUpload = tk.Button(
            buttonFrame,
            text="Next Step",
            fg="white",
            font=("Arial", 16),
            bd=0,
            relief='ridge',
            highlightthickness=5,
            highlightbackground="#7b9fc2",
            command=self.on_next_step_click
        )
        self.nextStepUpload.pack(side=tk.LEFT, padx=10)

        # NextStep button disabled initially
        self.nextStepUpload.config(state=tk.DISABLED, bg="#a0a0a0")

    def on_upload_click(self):
        result = self.controller.handle_file_upload()

        if result and result['success']:
            # Update label with file information
            self.uploadLabel.config(
                text=f"File uploaded successfully\n"
                f"File: {result['file_name']}\n"
                f"Rows: {result['rows']}\n"
                f"Columns: {result['columns']}"
            )

            # Change button text
            self.uploadDataSetButton.config(text="Change File")

            # Enable Next Step button
            self.nextStepUpload.config(
                state=tk.NORMAL,
                bg="#2a4d6f",
                cursor="hand2"
            )
        elif result and not result['success']:
            # Reset label on error
            self.uploadLabel.config(
                text="Upload failed - " + result.get('error_type', 'Error')
            )

    def on_next_step_click(self):
        """Handler for next step button click"""
        if self.controller.can_proceed_to_next_step():
            # Emit event to notify main view
            self.event_generate("<<NextStep>>")
