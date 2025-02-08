import os
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk

class ImageAnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Annotation Tool")
        self.root.state("zoomed")  # Maximize the window

        # Let rows/columns grow as needed:
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)

        self.image_list = []
        self.current_image_index = 0
        self.annotations = {}
        self.image_folder = ""
        self.segment_folder = ""
        self.label_folder = ""
        self.annotation_file = ""
        
        self.init_ui()

    def init_ui(self):
        # --- Top row: Image folder path entry + Browse button ---
        self.path_entry = tk.Entry(self.root, width=50)
        self.path_entry.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        self.load_button = tk.Button(self.root, text="Browse", command=self.load_images)
        self.load_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        # --- Next row: annotation rules label ---
        self.rules_text = (
            "0 - if all masks are incorrect or reflections\n"
            "1 - if all masks are new objects\n"
            "2 - if all masks are of existing object appearance change\n"
            "3 - if all masks are new object due to viewpoint change\n"
            "If annotation is mix of correct and incorrect masks select above options with sequence from label_file.txt\n"
            "5 - if the mixture of correct & incorrect masks count is too much (more than 4)"
        )
        self.rules_label = tk.Label(self.root, text=self.rules_text, justify=tk.LEFT, anchor="w")
        self.rules_label.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="w")
        
        # --- A frame that will hold the two images (top and bottom) ---
        self.image_frame = tk.Frame(self.root)
        self.image_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=5, sticky="nsew")

        # Top image (concat)
        self.image_label_top = tk.Label(self.image_frame)
        self.image_label_top.pack(side=tk.TOP, expand=True)

        # Bottom image (segment)
        self.image_label_bottom = tk.Label(self.image_frame)
        self.image_label_bottom.pack(side=tk.TOP, expand=True)
        
        # --- Label File Content (text box) ---
        self.label_text = tk.Text(self.root, width=40, height=10)
        self.label_text.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")
        
        # --- Frame for the file listbox + scrollbar ---
        self.list_frame = tk.Frame(self.root)
        self.list_frame.grid(row=2, column=2, rowspan=2, padx=10, pady=10, sticky="nsew")
        
        self.file_listbox = tk.Listbox(self.list_frame, width=40, height=20)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Vertical scrollbar for the file list
        self.list_scrollbar = tk.Scrollbar(self.list_frame, orient="vertical")
        self.list_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tie the scrollbar and the listbox together
        self.file_listbox.config(yscrollcommand=self.list_scrollbar.set)
        self.list_scrollbar.config(command=self.file_listbox.yview)

        # --- Annotation Entry (single line) ---
        self.annotation_entry = tk.Entry(self.root, width=20)
        self.annotation_entry.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        # --- Horizontal scrollbar for the annotation_entry (if you need it) ---
        self.annotation_scrollbar = tk.Scrollbar(self.root, orient="horizontal", command=self.annotation_entry.xview)
        self.annotation_entry.config(xscrollcommand=self.annotation_scrollbar.set)
        self.annotation_scrollbar.grid(row=4, column=0, columnspan=2, sticky="ew")

        # Key bindings
        self.root.bind("<Up>", self.previous_image)
        self.root.bind("<Down>", self.next_image)
        self.file_listbox.bind('<<ListboxSelect>>', self.on_listbox_select)

        # On close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_images(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        
        # The folder you pick is your 'concat' folder
        self.image_folder = folder
        
        # The sibling folder is called 'segment_folder' 
        self.segment_folder = os.path.join(os.path.dirname(self.image_folder), "segment_folder")
        if not os.path.exists(self.segment_folder):
            messagebox.showerror("Error", f"Segment folder not found at: {self.segment_folder}")
            return
        
        # For label files
        self.label_folder = os.path.join(os.path.dirname(self.image_folder), "labels")
        if not os.path.exists(self.label_folder):
            messagebox.showerror("Error", "Label folder not found!")
            return
        
        # For annotation file
        self.annotation_file = os.path.join(os.path.dirname(self.image_folder), "human_annotation.txt")
        
        # Gather list of images in the chosen concat folder
        all_files = [f for f in os.listdir(self.image_folder) if f.endswith('.png')]
        # Sort them numeric if possible
        self.image_list = sorted(
            all_files,
            key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x
        )
        if not self.image_list:
            messagebox.showerror("Error", "No .png images found in the selected folder!")
            return
        
        self.load_annotation_file()
        self.update_ui()

    def load_annotation_file(self):
        """
        Load existing annotation file or initialize an empty dictionary.
        """
        if os.path.exists(self.annotation_file):
            with open(self.annotation_file, 'r') as f:
                lines = f.readlines()
            self.annotations = {}
            for line in lines:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    filename, ann = parts
                else:
                    filename, ann = parts[0], ""
                self.annotations[filename] = ann
        else:
            self.annotations = {image_name: "" for image_name in self.image_list}
    
    def save_annotations(self):
        with open(self.annotation_file, 'w') as f:
            for image_name, annotation in self.annotations.items():
                f.write(f"{image_name} {annotation}\n")

    def update_ui(self):
        if not self.image_list:
            return

        # Figure out which image is selected
        image_name = self.image_list[self.current_image_index]
        
        # 1) Load top image from concat folder
        top_path = os.path.join(self.image_folder, image_name)
        top_cv = cv2.imread(top_path)
        if top_cv is None:
            messagebox.showerror("Error", f"Could not read: {top_path}")
            return
        top_cv = cv2.cvtColor(top_cv, cv2.COLOR_BGR2RGB)
        
        # 2) Load bottom image from segment folder
        bottom_path = os.path.join(self.segment_folder, image_name)
        bottom_cv = cv2.imread(bottom_path)
        if bottom_cv is None:
            # Possibly no error, or show a warning. We'll just place a blank if missing.
            bottom_cv = 255 * (1 - top_cv)  # dummy or something
        else:
            bottom_cv = cv2.cvtColor(bottom_cv, cv2.COLOR_BGR2RGB)

        # Optional: scale them if they don't fit.  (For example, scale to a max width.)
        def scale_for_display_if_needed(cv_img, max_width=1280):
            h, w = cv_img.shape[:2]
            if w > max_width:
                scale = max_width / float(w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            return cv_img
        
        top_cv = scale_for_display_if_needed(top_cv, max_width=1440)     # or whatever max you want
        bottom_cv = scale_for_display_if_needed(bottom_cv, max_width=960)

        # Convert to ImageTk
        self.top_image = ImageTk.PhotoImage(Image.fromarray(top_cv))
        self.bottom_image = ImageTk.PhotoImage(Image.fromarray(bottom_cv))

        # Update labels
        self.image_label_top.config(image=self.top_image)
        self.image_label_top.image = self.top_image  # keep a reference

        self.image_label_bottom.config(image=self.bottom_image)
        self.image_label_bottom.image = self.bottom_image  # keep a reference

        # Load label file content for the current image if present
        label_path = os.path.join(self.label_folder, image_name.replace('.png', '.txt'))
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label_content = f.read()
            self.label_text.delete(1.0, tk.END)
            self.label_text.insert(tk.END, label_content)
        else:
            self.label_text.delete(1.0, tk.END)
            self.label_text.insert(tk.END, "Label file not found.")
        
        # Update the file listbox
        self.file_listbox.delete(0, tk.END)
        for idx, filename in enumerate(self.image_list):
            self.file_listbox.insert(tk.END, filename)
        self.file_listbox.select_set(self.current_image_index)
        self.file_listbox.see(self.current_image_index)  # ensure visible

        # Update the annotation entry
        self.annotation_entry.delete(0, tk.END)
        self.annotation_entry.insert(0, self.annotations.get(image_name, ""))

    def previous_image(self, event=None):
        self.save_current_annotation()
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_ui()

    def next_image(self, event=None):
        self.save_current_annotation()
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.update_ui()

    def save_current_annotation(self):
        if not self.image_list:
            return
        current_annotation = self.annotation_entry.get().strip()
        current_name = self.image_list[self.current_image_index]
        self.annotations[current_name] = current_annotation
        self.save_annotations()

    def on_listbox_select(self, event):
        selection = event.widget.curselection()
        if selection:
            self.save_current_annotation()
            self.current_image_index = selection[0]
            self.update_ui()

    def on_closing(self):
        self.save_current_annotation()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    tool = ImageAnnotationTool(root)
    root.mainloop()
