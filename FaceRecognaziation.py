import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox, filedialog
import cv2
import os
import numpy as np
from PIL import Image, ImageTk, ImageOps, ImageDraw, ImageFilter
import datetime
import csv
import threading
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from concurrent.futures import ThreadPoolExecutor, as_completed
import pathlib
import json
from typing import List, Union

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Face ID")
        self.root.geometry("1400x800")
        self.root.minsize(1200, 700)

        # Variables
        self.camera_index = 0
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.known_faces = {}
        self.attendance_log = []
        self.current_frame = None
        self.camera_thread = None
        self.is_running = False
        self.capture_active = False
        self.sample_count = 0
        self.max_samples = 30
        self.last_face_detection_time = 0
        self.face_detection_interval = 0.5  # seconds

        # Load icons
        self.load_icons()

        # Create GUI
        self.create_widgets()

        # Load existing data
        self.load_recognizer_data()
        self.load_known_faces()

        # Start with camera off
        self.camera_state = False

    def load_icons(self):
        """Load icons for buttons"""
        try:
            # Create simple icons using PIL if actual icons aren't available
            self.camera_icon = self.create_icon("📷", "#4F46E5")
            self.stop_icon = self.create_icon("⏹", "#EF4444")
            self.capture_icon = self.create_icon("🎯", "#10B981")
            self.train_icon = self.create_icon("🧠", "#8B5CF6")
            self.database_icon = self.create_icon("📊", "#3B82F6")
            self.export_icon = self.create_icon("💾", "#10B981")
            self.settings_icon = self.create_icon("⚙", "#6B7280")
        except:
            # Fallback to text if icons fail
            self.camera_icon = None
            self.stop_icon = None
            self.capture_icon = None
            self.train_icon = None
            self.database_icon = None
            self.export_icon = None
            self.settings_icon = None

    def create_icon(self, text, color):
        """Create simple text-based icons"""
        img = Image.new('RGBA', (24, 24), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((6, 2), text, fill=color)
        return ImageTk.PhotoImage(img)

    def create_widgets(self):
        """Create the application interface"""
        # Configure grid layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Create sidebar frame
        self.sidebar_frame = ctk.CTkFrame(self.root, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1)

        # App logo and title
        self.logo_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.logo_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        self.logo_label = ctk.CTkLabel(self.logo_frame, text="Smart Face ID",
                                     font=ctk.CTkFont(size=22, weight="bold", family="Helvetica"))
        self.logo_label.pack(side="left", padx=10)


        self.logo_label = ctk.CTkLabel(self.logo_frame, text="created by : Diyar Penjweny",
                                     font=ctk.CTkFont(size=12, weight="bold", family="Helvetica"))
        self.logo_label.pack(side="left", padx=10)

        # Navigation buttons
        self.nav_buttons_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        self.nav_buttons_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Camera controls section
        cam_control_frame = ctk.CTkFrame(self.nav_buttons_frame, fg_color="transparent")
        cam_control_frame.pack(fill="x", pady=(0, 15))

        ctk.CTkLabel(cam_control_frame, text="CAMERA CONTROLS",
                   font=ctk.CTkFont(weight="bold", size=12)).pack(anchor="w", padx=10, pady=(0, 5))

        self.camera_combobox = ctk.CTkComboBox(cam_control_frame,
                                             values=self.get_available_cameras())
        self.camera_combobox.pack(fill="x", padx=10, pady=5)
        self.camera_combobox.set("Camera 0")

        self.start_button = ctk.CTkButton(cam_control_frame,
                                        text="Start Camera",
                                        image=self.camera_icon,
                                        compound="left",
                                        command=self.start_camera)
        self.start_button.pack(fill="x", padx=10, pady=5)

        self.stop_button = ctk.CTkButton(cam_control_frame,
                                       text="Stop Camera",
                                       image=self.stop_icon,
                                       compound="left",
                                       command=self.stop_camera,
                                       state="disabled")
        self.stop_button.pack(fill="x", padx=10, pady=5)

        # Face recognition section
        face_recog_frame = ctk.CTkFrame(self.nav_buttons_frame, fg_color="transparent")
        face_recog_frame.pack(fill="x", pady=(15, 15))

        ctk.CTkLabel(face_recog_frame, text="FACE RECOGNITION",
                   font=ctk.CTkFont(weight="bold", size=12)).pack(anchor="w", padx=10, pady=(0, 5))

        self.name_entry = ctk.CTkEntry(face_recog_frame,
                                    placeholder_text="Enter person's name")
        self.name_entry.pack(fill="x", padx=10, pady=5)

        self.capture_button = ctk.CTkButton(face_recog_frame,
                                          text="Capture Faces",
                                          image=self.capture_icon,
                                          compound="left",
                                          command=self.start_capture_faces)
        self.capture_button.pack(fill="x", padx=10, pady=5)

        self.train_button = ctk.CTkButton(face_recog_frame,
                                        text="Train Model",
                                        image=self.train_icon,
                                        compound="left",
                                        command=self.train_model)
        self.train_button.pack(fill="x", padx=10, pady=5)

        # Database section
        db_frame = ctk.CTkFrame(self.nav_buttons_frame, fg_color="transparent")
        db_frame.pack(fill="x", pady=(15, 15))

        ctk.CTkLabel(db_frame, text="DATABASE",
                   font=ctk.CTkFont(weight="bold", size=12)).pack(anchor="w", padx=10, pady=(0, 5))

        self.view_db_button = ctk.CTkButton(db_frame,
                                          text="View Database",
                                          image=self.database_icon,
                                          compound="left",
                                          command=self.view_database)
        self.view_db_button.pack(fill="x", padx=10, pady=5)

        self.export_button = ctk.CTkButton(db_frame,
                                         text="Export Attendance",
                                         image=self.export_icon,
                                         compound="left",
                                         command=self.export_attendance)
        self.export_button.pack(fill="x", padx=10, pady=5)

        # Settings section at bottom
        settings_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        settings_frame.grid(row=8, column=0, padx=10, pady=(0, 20), sticky="sew")

        ctk.CTkLabel(settings_frame, text="SETTINGS",
                   font=ctk.CTkFont(weight="bold", size=12)).pack(anchor="w", padx=10, pady=(0, 5))

        # Appearance mode switcher
        self.appearance_mode_menu = ctk.CTkOptionMenu(
            settings_frame,
            values=["Dark", "Light", "System"],
            command=self.change_appearance_mode)
        self.appearance_mode_menu.pack(fill="x", padx=10, pady=5)
        self.appearance_mode_menu.set("Dark")

        # Create main display area
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=10)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Camera display area
        self.camera_display = ctk.CTkFrame(self.main_frame, corner_radius=15)
        self.camera_display.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.camera_display.grid_rowconfigure(0, weight=1)
        self.camera_display.grid_columnconfigure(0, weight=1)

        self.camera_label = ctk.CTkLabel(self.camera_display, text="Camera Feed\n\nClick Start Camera to begin",
                                       font=ctk.CTkFont(size=16),
                                       fg_color="#F3F4F6" if ctk.get_appearance_mode() == "Light" else "#1F2937")
        self.camera_label.grid(row=0, column=0, sticky="nsew")

        # Stats frame below camera
        self.stats_frame = ctk.CTkFrame(self.main_frame, height=100, corner_radius=10)
        self.stats_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))

        # Stats cards
        self.registered_card = self.create_stat_card("Registered Faces", "0", "#4F46E5")
        self.registered_card.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="ew")

        self.today_card = self.create_stat_card("Today's Attendance", "0", "#10B981")
        self.today_card.grid(row=0, column=1, padx=5, pady=10, sticky="ew")

        self.samples_card = self.create_stat_card("Training Samples", "0", "#8B5CF6")
        self.samples_card.grid(row=0, column=2, padx=(5, 10), pady=10, sticky="ew")

        # Configure column weights
        self.stats_frame.grid_columnconfigure(0, weight=1)
        self.stats_frame.grid_columnconfigure(1, weight=1)
        self.stats_frame.grid_columnconfigure(2, weight=1)

        # Progress bar for face capture
        self.progress_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.progress_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        self.progress_label = ctk.CTkLabel(self.progress_frame, text="", anchor="w")
        self.progress_label.pack(fill="x", padx=5)

        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, orientation="horizontal")
        self.progress_bar.pack(fill="x", padx=5, pady=(0, 5))
        self.progress_bar.set(0)
        self.progress_frame.grid_remove()

        # Status bar
        self.status_bar = ctk.CTkFrame(self.main_frame, height=30, corner_radius=0)
        self.status_bar.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))

        self.status_label = ctk.CTkLabel(self.status_bar, text="Ready",
                                       font=ctk.CTkFont(size=12),
                                       anchor="w")
        self.status_label.pack(side="left", padx=10, fill="x", expand=True)

        self.update_stats()

    def create_stat_card(self, title, value, color):
        """Create a stat card"""
        frame = ctk.CTkFrame(self.stats_frame, fg_color="transparent")

        # Title label
        ctk.CTkLabel(frame, text=title,
                     font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=10, pady=(10, 0))

        # Value label
        ctk.CTkLabel(frame, text=value,
                     font=ctk.CTkFont(size=24, weight="bold"),
                     text_color=color).pack(anchor="w", padx=10, pady=(0, 10))

        # Decorative line
        line = ctk.CTkFrame(frame, height=3, fg_color=color, corner_radius=2)
        line.pack(fill="x", padx=10, pady=(0, 5), anchor="w")

        return frame

    def update_stats(self):
        """Update the statistics cards"""
        registered = len(self.known_faces)
        today = len([log for log in self.attendance_log
                     if log[1].startswith(datetime.datetime.now().strftime("%Y-%m-%d"))])

        samples = 0
        if os.path.exists("faces"):
            samples = len(os.listdir("faces"))

        # Update card values
        for widget in self.registered_card.winfo_children():
            if isinstance(widget, ctk.CTkLabel) and widget.cget("font").cget("size") == 24:
                widget.configure(text=str(registered))

        for widget in self.today_card.winfo_children():
            if isinstance(widget, ctk.CTkLabel) and widget.cget("font").cget("size") == 24:
                widget.configure(text=str(today))

        for widget in self.samples_card.winfo_children():
            if isinstance(widget, ctk.CTkLabel) and widget.cget("font").cget("size") == 24:
                widget.configure(text=str(samples))

        # Update every 5 seconds
        self.root.after(5000, self.update_stats)

    def get_available_cameras(self):
        """Check available cameras with timeout to prevent freezing"""
        cameras = []

        def check_camera(index):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                return ret
            return False

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(check_camera, i): i for i in range(4)}
            for future in as_completed(futures):
                if future.result():
                    cameras.append(f"Camera {futures[future]}")

        return cameras if cameras else ["Camera 0"]

    def start_camera(self):
        """Start video capture with optimized settings"""
        if not self.is_running:
            self.camera_index = int(self.camera_combobox.get().split()[-1])
            self.is_running = True
            self.camera_state = True
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.status_label.configure(text="Camera running - detecting faces...")

            # Update UI
            self.camera_label.configure(text="Initializing camera...")

            # Start camera in a separate thread
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.camera_thread.start()

    def _camera_loop(self):
        """Main camera processing loop with performance optimizations"""
        # Initialize camera with optimized settings
        self.video_capture = cv2.VideoCapture(self.camera_index)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.video_capture.set(cv2.CAP_PROP_FPS, 30)
        self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size

        # Skip frames at startup to allow camera to stabilize
        for _ in range(5):
            self.video_capture.read()

        last_frame_time = time.time()

        while self.is_running:
            # Skip frames if processing is taking too long
            if time.time() - last_frame_time < 0.033:  # ~30fps
                continue

            ret, frame = self.video_capture.read()
            if not ret:
                continue

            # Process frame (resize for display only)
            display_frame = cv2.resize(frame, (800, 450))
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            # Only do face detection at specified intervals
            current_time = time.time()
            if current_time - self.last_face_detection_time > self.face_detection_interval:
                processed_frame = self.process_frame(display_frame)
                self.last_face_detection_time = current_time
            else:
                processed_frame = display_frame

            self.current_frame = processed_frame

            # Update display in main thread
            self.root.after(0, self.update_display, processed_frame)

            # Handle face capture if active
            if self.capture_active:
                self.root.after(0, self.capture_face_samples)

            last_frame_time = time.time()

        self.video_capture.release()

    def process_frame(self, frame):
        """Process frame for face detection and recognition"""
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Equalize histogram to improve detection in low light
        gray = cv2.equalizeHist(gray)

        # Face detection with optimized parameters
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,  # Slightly more aggressive scaling
            minNeighbors=6,  # Require more neighbors for better accuracy
            minSize=(80, 80),  # Larger minimum size for faster processing
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw rectangles and recognize faces
        for (x, y, w, h) in faces:
            # Draw rounded rectangle
            self.draw_rounded_rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 15)

            # Face recognition (only if we have a trained model)
            if hasattr(self, 'recognizer') and self.recognizer:
                id_, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])

                if confidence < 100:  # Confidence threshold
                    name = self.known_faces.get(id_, "Unknown")
                    confidence_pct = round(100 - confidence)

                    # Draw name with confidence
                    text = f"{name} ({confidence_pct}%)"
                    text_size = cv2.getTextSize(text, self.font, 0.7, 2)[0]

                    # Background for text
                    cv2.rectangle(frame, (x, y - text_size[1] - 10),
                                  (x + text_size[0], y), (0, 0, 0), -1)

                    cv2.putText(frame, text, (x, y - 5),
                                self.font, 0.7, (255, 255, 255), 2)

                    # Log attendance
                    if name != "Unknown":
                        self.log_attendance(name)
                else:
                    cv2.putText(frame, "Unknown", (x, y - 5),
                                self.font, 0.7, (255, 255, 255), 2)

        return frame

    def draw_rounded_rectangle(self, img, pt1, pt2, color, thickness, corner_radius):
        """Draw a rounded rectangle on the image"""
        x1, y1 = pt1
        x2, y2 = pt2

        # Draw straight lines
        cv2.line(img, (x1 + corner_radius, y1), (x2 - corner_radius, y1), color, thickness)
        cv2.line(img, (x1 + corner_radius, y2), (x2 - corner_radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + corner_radius), (x1, y2 - corner_radius), color, thickness)
        cv2.line(img, (x2, y1 + corner_radius), (x2, y2 - corner_radius), color, thickness)

        # Draw arcs
        cv2.ellipse(img, (x1 + corner_radius, y1 + corner_radius),
                    (corner_radius, corner_radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - corner_radius, y1 + corner_radius),
                    (corner_radius, corner_radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + corner_radius, y2 - corner_radius),
                    (corner_radius, corner_radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - corner_radius, y2 - corner_radius),
                    (corner_radius, corner_radius), 0, 0, 90, color, thickness)

    def update_display(self, frame):
        """Update the camera display in the GUI"""
        img = Image.fromarray(frame)

        # Apply creative filter (vignette)
        img = self.apply_vignette(img, level=3)

        # Convert to PhotoImage
        imgtk = ImageTk.PhotoImage(image=img)

        # Update label
        self.camera_label.configure(image=imgtk, text="")
        self.camera_label.image = imgtk  # Keep reference

    def apply_vignette(self, img, level=2):
        """Apply vignette effect to image"""
        width, height = img.size
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)

        # Draw white ellipse
        draw.ellipse((-width // level, -height // level,
                      width + width // level, height + height // level),
                     fill=255)

        # Apply blur
        mask = mask.filter(ImageFilter.GaussianBlur(radius=width // 6))

        # Composite with original image
        vignette = Image.new('RGB', (width, height), (0, 0, 0))
        img = Image.composite(img, vignette, mask)

        return img

    def stop_camera(self):
        """Stop video capture"""
        if self.is_running:
            self.is_running = False
            self.camera_state = False
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.status_label.configure(text="Camera stopped")

            if self.camera_thread and self.camera_thread.is_alive():
                self.camera_thread.join(timeout=1)

            # Clear camera display
            self.camera_label.configure(image=None,
                                      text="Camera Feed\n\nClick Start Camera to begin")

    def start_capture_faces(self):
        """Start face capture process"""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name")
            return

        if not self.is_running:
            messagebox.showerror("Error", "Camera is not running")
            return

        # Generate ID (use existing if name exists)
        face_id = None
        for id_, existing_name in self.known_faces.items():
            if existing_name.lower() == name.lower():
                face_id = id_
                break

        if face_id is None:
            face_id = max(self.known_faces.keys()) + 1 if self.known_faces else 1

        self.face_id = face_id
        self.face_name = name
        self.capture_active = True
        self.sample_count = 0

        # Create directory if not exists
        if not os.path.exists("faces"):
            os.makedirs("faces")

        # Show progress bar
        self.progress_frame.grid()
        self.progress_bar.set(0)
        self.progress_label.configure(text=f"Capturing faces for {name}...")
        self.status_label.configure(text=f"Status: Capturing faces for {name}...")

    def capture_face_samples(self):
        """Capture face samples in the camera thread"""
        if not self.capture_active or self.sample_count >= self.max_samples:
            return

        if self.current_frame is None:
            return

        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 6)

        for (x, y, w, h) in faces:
            if self.sample_count >= self.max_samples:
                break

            self.sample_count += 1

            # Save the face image
            timestamp = int(time.time())
            cv2.imwrite(f"faces/User.{self.face_id}.{timestamp}.jpg", gray[y:y + h, x:x + w])

            # Update progress
            progress = self.sample_count / self.max_samples
            self.progress_bar.set(progress)
            self.progress_label.configure(text=f"Captured {self.sample_count}/{self.max_samples} samples")

            # Flash effect on face detection
            self.root.after(0, self.flash_detection, (x, y, w, h))

            # Small delay between captures
            time.sleep(0.1)

        if self.sample_count >= self.max_samples:
            self.finish_face_capture()

    def flash_detection(self, rect):
        """Show flash effect when face is detected"""
        x, y, w, h = rect
        flash_img = Image.new('RGB', (self.camera_label.winfo_width(), self.camera_label.winfo_height()),
                              (255, 255, 255))
        flash_img = ImageTk.PhotoImage(flash_img)

        self.camera_label.configure(image=flash_img)
        self.camera_label.image = flash_img
        self.root.after(100, lambda: self.camera_label.configure(image=self.camera_label.image))

    def finish_face_capture(self):
        """Finish the face capture process"""
        self.capture_active = False
        self.known_faces[self.face_id] = self.face_name
        self.save_known_faces()

        self.progress_frame.grid_remove()
        self.status_label.configure(text=f"Status: Captured {self.sample_count} samples for {self.face_name}")

        # Show completion message
        messagebox.showinfo("Success", f"Captured {self.sample_count} face samples for {self.face_name}")

        # Update stats
        self.update_stats()

    def train_model(self):
        """Train the face recognition model with progress feedback"""
        if not os.path.exists("faces"):
            messagebox.showerror("Error", "No face samples found")
            return

        # Show progress
        self.status_label.configure(text="Status: Training model...")

        # Create progress window
        self.train_window = ctk.CTkToplevel(self.root)
        self.train_window.title("Training Model")
        self.train_window.geometry("400x200")
        self.train_window.transient(self.root)
        self.train_window.grab_set()

        ctk.CTkLabel(self.train_window, text="Training face recognition model...",
                   font=ctk.CTkFont(size=14, weight="bold")).pack(pady=20)

        progress_bar = ctk.CTkProgressBar(self.train_window)
        progress_bar.pack(fill="x", padx=20, pady=10)
        progress_bar.set(0)

        status_label = ctk.CTkLabel(self.train_window, text="Preparing data...")
        status_label.pack()

        # Train in a separate thread
        train_thread = threading.Thread(target=self._train_model_thread,
                                      args=(progress_bar, status_label),
                                      daemon=True)
        train_thread.start()

    def _train_model_thread(self, progress_bar, status_label):
        """Thread for training the model with progress updates"""
        faces = []
        ids = []

        image_paths = [os.path.join("faces", f) for f in os.listdir("faces")]
        total_images = len(image_paths)

        for i, image_path in enumerate(image_paths):
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            face_id = int(os.path.split(image_path)[-1].split(".")[1])
            faces.append(img)
            ids.append(face_id)

            # Update progress
            progress = (i + 1) / total_images
            self.root.after(0, progress_bar.set, progress)
            self.root.after(0, status_label.configure,
                          text=f"Processing image {i + 1}/{total_images}")
            time.sleep(0.05)  # Small delay for UI update

        if not ids:
            self.root.after(0, messagebox.showerror, "Error", "No training data available")
            self.root.after(0, self.train_window.destroy)
            return

        # Train the model
        self.root.after(0, status_label.configure, text="Training model...")
        self.recognizer.train(faces, np.array(ids))
        self.recognizer.save("trainer.yml")

        # Update UI
        self.root.after(0, status_label.configure, text="Training complete!")
        self.root.after(0, progress_bar.set, 1)
        self.root.after(0, lambda: self.train_window.destroy())
        self.root.after(0, self.status_label.configure,
                      text=f"Status: Model trained with {len(ids)} samples")
        self.root.after(0, messagebox.showinfo, "Success",
                      f"Model trained with {len(ids)} samples")

    def load_recognizer_data(self):
        """Load trained recognizer data"""
        if os.path.exists("trainer.yml"):
            self.recognizer.read("trainer.yml")

    def load_known_faces(self):
        """Load known faces from file"""
        if os.path.exists("known_faces.csv"):
            with open("known_faces.csv", "r") as f:
                reader = csv.reader(f)
                self.known_faces = {int(row[0]): row[1] for row in reader}

    def save_known_faces(self):
        """Save known faces to file"""
        with open("known_faces.csv", "w", newline="") as f:
            writer = csv.writer(f)
            for id_, name in self.known_faces.items():
                writer.writerow([id_, name])

    def log_attendance(self, name):
        """Log attendance with timestamp"""
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        # Check if already logged today
        today = now.strftime("%Y-%m-%d")
        for entry in self.attendance_log:
            if entry[0] == name and entry[1].startswith(today):
                return

        self.attendance_log.append((name, timestamp))
        self.status_label.configure(text=f"Status: Logged attendance for {name}")
        self.update_stats()

    def view_database(self):
        """Show known faces database"""
        if not self.known_faces:
            messagebox.showinfo("Database", "No faces in database")
            return

        db_window = ctk.CTkToplevel(self.root)
        db_window.title("Face Database")
        db_window.geometry("900x700")
        db_window.transient(self.root)

        # Create scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(db_window)
        scroll_frame.pack(expand=True, fill="both", padx=10, pady=10)

        # Create a grid layout for faces
        row, col = 0, 0
        for id_, name in sorted(self.known_faces.items()):
            # Create card for each person
            card = ctk.CTkFrame(scroll_frame, width=200, height=250, corner_radius=15)
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

            # Person's name
            ctk.CTkLabel(card, text=name,
                       font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(15, 5))

            # ID
            ctk.CTkLabel(card, text=f"ID: {id_}").pack()

            # Sample images
            sample_images = [f for f in os.listdir("faces") if f.startswith(f"User.{id_}.")]
            if sample_images:
                try:
                    img_path = os.path.join("faces", sample_images[0])
                    img = Image.open(img_path)
                    img = img.resize((150, 150), Image.LANCZOS)

                    # Apply circular mask
                    mask = Image.new('L', (150, 150), 0)
                    draw = ImageDraw.Draw(mask)
                    draw.ellipse((0, 0, 150, 150), fill=255)
                    img = ImageOps.fit(img, mask.size, centering=(0.5, 0.5))
                    img.putalpha(mask)

                    img_tk = ImageTk.PhotoImage(img)

                    img_label = ctk.CTkLabel(card, image=img_tk, text="")
                    img_label.image = img_tk  # Keep reference
                    img_label.pack(pady=10)
                except Exception as e:
                    print(f"Error loading image: {e}")

            # Samples count
            ctk.CTkLabel(card, text=f"{len(sample_images)} samples").pack(pady=(0, 10))

            # Update grid position
            col += 1
            if col > 3:
                col = 0
                row += 1

        # Configure grid weights
        for i in range(4):
            scroll_frame.grid_columnconfigure(i, weight=1)

    def export_attendance(self):
        """Export attendance log to CSV"""
        if not self.attendance_log:
            messagebox.showerror("Error", "No attendance data to export")
            return

        # Create custom dialog
        export_dialog = ctk.CTkToplevel(self.root)
        export_dialog.title("Export Attendance")
        export_dialog.geometry("500x300")
        export_dialog.transient(self.root)
        export_dialog.grab_set()

        ctk.CTkLabel(export_dialog, text="Export Attendance Log",
                   font=ctk.CTkFont(size=16, weight="bold")).pack(pady=20)


        # Date range selection
        date_frame = ctk.CTkFrame(export_dialog, fg_color="transparent")
        date_frame.pack(pady=10)

        ctk.CTkLabel(date_frame, text="From:").grid(row=0, column=0, padx=5)
        self.from_date = ctk.CTkEntry(date_frame, placeholder_text="YYYY-MM-DD")
        self.from_date.grid(row=0, column=1, padx=5)

        ctk.CTkLabel(date_frame, text="To:").grid(row=0, column=2, padx=5)
        self.to_date = ctk.CTkEntry(date_frame, placeholder_text="YYYY-MM-DD")
        self.to_date.grid(row=0, column=3, padx=5)

        # Format selection
        format_frame = ctk.CTkFrame(export_dialog, fg_color="transparent")
        format_frame.pack(pady=10)

        ctk.CTkLabel(format_frame, text="Format:").pack(side="left", padx=5)
        self.format_var = ctk.StringVar(value="csv")
        ctk.CTkRadioButton(format_frame, text="CSV", variable=self.format_var, value="csv").pack(side="left", padx=5)
        ctk.CTkRadioButton(format_frame, text="Excel", variable=self.format_var, value="xlsx").pack(side="left", padx=5)

        # Export button
        ctk.CTkButton(export_dialog, text="Export",
                      command=lambda: self._perform_export(export_dialog),
                      fg_color=self.secondary_color).pack(pady=20)

    def _perform_export(self, dialog):
        """Perform the actual export"""
        from_date = self.from_date.get()
        to_date = self.to_date.get()
        file_format = self.format_var.get()

        try:
            # Filter by date range if specified
            filtered_log = self.attendance_log
            if from_date:
                filtered_log = [log for log in filtered_log if log[1] >= from_date]
            if to_date:
                filtered_log = [log for log in filtered_log if log[1] <= to_date + " 23:59:59"]

            if not filtered_log:
                messagebox.showerror("Error", "No data in selected date range")
                return

            # Get save filename
            filetypes = [("CSV files", "*.csv")] if file_format == "csv" else [("Excel files", "*.xlsx")]

            filename = filedialog.asksaveasfilename(
                defaultextension=f".{file_format}",
                filetypes=filetypes,
                title="Save attendance log")

            if filename:
                if file_format == "csv":
                    with open(filename, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Name", "Timestamp"])
                        for entry in sorted(filtered_log, key=lambda x: x[1]):
                            writer.writerow(entry)
                else:
                    # For Excel export, we would use openpyxl or pandas
                    # This is just a placeholder as we'd need additional dependencies
                    messagebox.showerror("Error", "Excel export requires additional libraries")
                    return

                messagebox.showinfo("Success", f"Attendance log saved to {filename}")
                dialog.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")

    def change_appearance_mode(self, new_appearance_mode):
        """Change appearance mode (light/dark/system)"""
        ctk.set_appearance_mode(new_appearance_mode)

        # Update specific colors that don't automatically change
        if new_appearance_mode == "Light":
            self.camera_label.configure(fg_color="#F3F4F6")
            self.status_label.configure(text_color="#6B7280")
        else:
            self.camera_label.configure(fg_color="#1F2937")
            self.status_label.configure(text_color="#9CA3AF")

    def on_closing(self):
        """Cleanup when closing the application"""
        self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    # Create theme file if it doesn't exist
    theme_path = "assets/theme.json"
    if not os.path.exists(os.path.dirname(theme_path)):
        os.makedirs(os.path.dirname(theme_path))

    if not os.path.exists(theme_path):
        theme = {
            "NeuralFace": {
                "fg_color": "#1E1E2D",
                "text_color": "#FFFFFF",
                "button_color": "#4F46E5",
                "button_hover_color": "#4338CA",
                "frame_color": "#252535"
            }
        }
        import json

        with open(theme_path, "w") as f:
            json.dump(theme, f)

    root = ctk.CTk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()