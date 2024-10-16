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
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)

        self.image_list = []
        self.current_image_index = 0
        self.annotations = []
        self.image_folder = ""
        self.label_folder = ""
        self.annotation_file = ""
        
        self.init_ui()

    def init_ui(self):
        # Image Path Entry
        self.path_entry = tk.Entry(self.root, width=50)
        self.path_entry.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.load_button = tk.Button(self.root, text="Browse", command=self.load_images)
        self.load_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        # Annotation Rules
        self.rules_text = """0 - if all masks are incorrect or reflections
1 - if all masks are new objects
2 - if all masks are of existing object appearance change
3 - if all masks are new object due to viewpoint change
If annotation is mix of correct and incorrect masks select above options with sequence from label_file.txt
5 - if the mixture of correct & incorrect masks count is too much (more than 4)"""
        self.rules_label = tk.Label(self.root, text=self.rules_text, justify=tk.LEFT, anchor="w")
        self.rules_label.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="w")
        
        # Image Display
        self.image_label = tk.Label(self.root)
        self.image_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        # Label File Content
        self.label_text = tk.Text(self.root, width=40, height=10)
        self.label_text.grid(row=1, column=2, padx=10, pady=10, sticky="nsew")
        
        # File List
        self.file_listbox = tk.Listbox(self.root, width=40, height=20)
        self.file_listbox.grid(row=2, column=2, padx=10, pady=10, sticky="nsew")
        self.file_listbox.bind('<<ListboxSelect>>', self.on_listbox_select)
        
        # Annotation Entry Box (Single line)
        self.annotation_entry = tk.Entry(self.root, width=20)
        self.annotation_entry.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        # Scrollbar for Annotation Entry Box
        self.annotation_scrollbar = tk.Scrollbar(self.root, orient="horizontal", command=self.annotation_entry.xview)
        self.annotation_entry.config(xscrollcommand=self.annotation_scrollbar.set)
        self.annotation_scrollbar.grid(row=4, column=0, columnspan=2, sticky="ew")

        self.root.bind("<Up>", self.previous_image)
        self.root.bind("<Down>", self.next_image)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_images(self):
        self.image_folder = filedialog.askdirectory()
        if not self.image_folder:
            return
        
        # Set label folder as "labels" directory in the parent directory of the image folder
        self.label_folder = os.path.join(os.path.dirname(self.image_folder), "labels")
        if not os.path.exists(self.label_folder):
            messagebox.showerror("Error", "Label folder not found!")
            return
        
        self.annotation_file =  os.path.join(os.path.dirname(self.image_folder), "human_annotation.txt")
        self.image_list = sorted([f for f in os.listdir(self.image_folder) if f.endswith('.png')], key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)
        if not self.image_list:
            messagebox.showerror("Error", "No images found in the selected folder!")
            return
        
        self.load_annotation_file()
        self.update_ui()

    def load_annotation_file(self):
        if os.path.exists(self.annotation_file):
            with open(self.annotation_file, 'r') as f:
                self.annotations = {line.split(' ', 1)[0]: line.split(' ', 1)[1] if ' ' in line else '' for line in f.readlines()}
        else:
            self.annotations = {image_name: "" for image_name in self.image_list}
    
    def save_annotations(self):
        with open(self.annotation_file, 'w') as f:
            for image_name, annotation in self.annotations.items():
                f.write(f"{image_name} {annotation}\n")

    def update_ui(self):
        if not self.image_list:
            return

        image_path = os.path.join(self.image_folder, self.image_list[self.current_image_index])
        label_path = os.path.join(self.label_folder, self.image_list[self.current_image_index].replace('.png', '.txt'))
        
        # Load and display image
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            max_size = 1600
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            image = Image.fromarray(image)
            image = ImageTk.PhotoImage(image)
            self.image_label.config(image=image)
            self.image_label.image = image
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
            return
        
        # Load label content
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label_content = f.read()
                self.label_text.delete(1.0, tk.END)
                self.label_text.insert(tk.END, label_content)
        else:
            self.label_text.delete(1.0, tk.END)
            self.label_text.insert(tk.END, "Label file not found.")
        
        # Update file list
        self.file_listbox.delete(0, tk.END)
        for idx, filename in enumerate(self.image_list):
            self.file_listbox.insert(tk.END, filename)
            if idx == self.current_image_index:
                self.file_listbox.select_set(idx)
        
        # Update annotation entry
        self.annotation_entry.delete(0, tk.END)
        self.annotation_entry.insert(0, self.annotations.get(self.image_list[self.current_image_index], ""))

    def previous_image(self, event):
        self.save_current_annotation()
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_ui()

    def next_image(self, event):
        self.save_current_annotation()
        if self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.update_ui()

    def save_current_annotation(self):
        current_annotation = self.annotation_entry.get().strip()
        self.annotations[self.image_list[self.current_image_index]] = current_annotation
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