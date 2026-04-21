import os
import sys
import threading
import tkinter as tk
from tkinter import ttk
import tkinter.filedialog
import tkinter.messagebox
import tkinter.font as tkfont

import cv2
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import halftone as ht
from halftone import ordered_dither, create_normal_map
from raft_flow import compute_raft_flow, warp_image_with_flow
from video_utils import read_video, write_video

NORMAL_KEYFRAME_INTERVAL = 5

WINDOW_BG = "#f4f4f4"
PANEL_BG = "#f7f7f7"
TEXT_FG = "#111111"
SUBTLE_FG = "#666666"
BORDER = "#cfcfcf"
ENTRY_BG = "#ffffff"
BTN_BG = "#ffffff"
BTN_FG = "#111111"
PROG_BG = "#dddddd"
PROG_FILL = "#111111"

PREVIEW_BIG_W = 920
PREVIEW_BIG_H = 620

GRID_COLS = 2
GRID_ROWS = 2
GRID_CELL_W = 430
GRID_CELL_H = 255


class HalftoneApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Surface Stable Halftoning")
        self.window.configure(bg=WINDOW_BG)
        self.window.resizable(True, True)
        self.window.minsize(1320, 860)

        if sys.platform == "darwin":
            try:
                self.window.tk.call("tk", "scaling", 1.35)
            except Exception:
                pass

        self._configure_fonts()

        self.processing = False
        self.preview_loading = False
        self._preview_generation_id = 0

        self._video_path = None
        self._video_total_frames = 0
        self._video_fps = 30.0

        self._grid_cell_photos = [None] * 4
        self._big_preview_photo = None

        self._build_ui()

    # ------------------------
    # UI setup
    # ------------------------
    def _configure_fonts(self):
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=12)
        text_font = tkfont.nametofont("TkTextFont")
        text_font.configure(size=12)
        fixed_font = tkfont.nametofont("TkFixedFont")
        fixed_font.configure(size=12)

        self.font_title = ("TkDefaultFont", 16, "bold")
        self.font_section = ("TkDefaultFont", 12, "bold")
        self.font_label = ("TkDefaultFont", 11)
        self.font_small = ("TkDefaultFont", 10)
        self.font_button = ("TkDefaultFont", 12, "bold")
        self.font_status = ("TkDefaultFont", 10)

    def _build_ui(self):
        self.window.columnconfigure(0, weight=0)
        self.window.columnconfigure(1, weight=1)
        self.window.rowconfigure(0, weight=1)

        left = tk.Frame(self.window, bg=PANEL_BG, padx=12, pady=12, bd=1, relief="solid")
        left.grid(row=0, column=0, sticky="nsw")
        right = tk.Frame(self.window, bg=WINDOW_BG, padx=12, pady=12)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)

        tk.Label(
            left,
            text="Surface Stable Halftoning",
            bg=PANEL_BG,
            fg=TEXT_FG,
            font=self.font_title
        ).pack(anchor="w", pady=(0, 8))

        # Input video
        tk.Label(left, text="Input Video", bg=PANEL_BG, fg=TEXT_FG, font=self.font_section).pack(anchor="w")
        file_row = tk.Frame(left, bg=PANEL_BG)
        file_row.pack(fill=tk.X, pady=(4, 8))

        self.file_entry = tk.Entry(
            file_row,
            width=32,
            font=self.font_label,
            bg=ENTRY_BG,
            fg=TEXT_FG,
            insertbackground=TEXT_FG,
            relief="solid",
            bd=1
        )
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Button(
            file_row,
            text="Browse",
            command=self._browse_in,
            font=self.font_label,
            bg=BTN_BG,
            fg=BTN_FG,
            activebackground="#ececec",
            activeforeground=BTN_FG,
            relief="solid",
            bd=1,
            cursor="hand2",
            padx=12
        ).pack(side=tk.LEFT, padx=(8, 0))

        # Output folder
        tk.Label(left, text="Output Folder", bg=PANEL_BG, fg=TEXT_FG, font=self.font_section).pack(anchor="w")
        out_row = tk.Frame(left, bg=PANEL_BG)
        out_row.pack(fill=tk.X, pady=(4, 10))

        self.out_entry = tk.Entry(
            out_row,
            width=32,
            font=self.font_label,
            bg=ENTRY_BG,
            fg=TEXT_FG,
            insertbackground=TEXT_FG,
            relief="solid",
            bd=1
        )
        self.out_entry.insert(0, "outputs/videos")
        self.out_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Button(
            out_row,
            text="Browse",
            command=self._browse_out,
            font=self.font_label,
            bg=BTN_BG,
            fg=BTN_FG,
            activebackground="#ececec",
            activeforeground=BTN_FG,
            relief="solid",
            bd=1,
            cursor="hand2",
            padx=12
        ).pack(side=tk.LEFT, padx=(8, 0))

        self._divider(left)

        tk.Label(left, text="Halftone Parameters", bg=PANEL_BG, fg=TEXT_FG, font=self.font_section).pack(anchor="w", pady=(2, 4))
        self.cell_size = self._slider(left, "Cell Size", 4, 32, 8, 1)
        self.downsample = self._slider(left, "Downsample", 1, 32, 4, 1)
        self.moge_max_size = self._slider(left, "MoGe Max Size", 64, 512, 256, 32)
        self.n_buckets = self._slider(left, "Normal Buckets", 64, 2048, 1024, 64)
        self.max_stretch = self._slider(left, "Max Stretch", 1.0, 6.0, 3.0, 0.1)

        self._divider(left)

        tk.Label(left, text="Stabilization Parameters", bg=PANEL_BG, fg=TEXT_FG, font=self.font_section).pack(anchor="w", pady=(2, 4))
        self.alpha = self._slider(left, "Blend Alpha", 0.0, 1.0, 0.6, 0.05)
        self.normal_blend_alpha = self._slider(left, "Normal Blend Alpha", 0.0, 1.0, 0.5, 0.05)
        self.scene_threshold = self._slider(left, "Scene Threshold", 5, 100, 30, 1)

        self._divider(left)

        self.color_mode = tk.BooleanVar(value=False)
        tk.Checkbutton(
            left,
            text="RGB Color Mode",
            variable=self.color_mode,
            bg=PANEL_BG,
            fg=TEXT_FG,
            activebackground=PANEL_BG,
            activeforeground=TEXT_FG,
            selectcolor=ENTRY_BG,
            font=self.font_label
        ).pack(anchor="w", pady=(4, 6))

        self._divider(left)

        self.load_preview_btn = tk.Button(
            left,
            text="Load Previews",
            command=self._load_previews_clicked,
            font=self.font_button,
            bg=BTN_BG,
            fg=BTN_FG,
            activebackground="#ececec",
            activeforeground=BTN_FG,
            relief="solid",
            bd=1,
            cursor="hand2",
            width=22,
            pady=6
        )
        self.load_preview_btn.pack(anchor="w", pady=(6, 8))

        self.run_btn = tk.Button(
            left,
            text="Process Video",
            command=self._start_processing,
            font=self.font_button,
            bg=BTN_BG,
            fg=BTN_FG,
            activebackground="#ececec",
            activeforeground=BTN_FG,
            relief="solid",
            bd=1,
            cursor="hand2",
            width=22,
            pady=6
        )
        self.run_btn.pack(anchor="w", pady=(0, 8))

        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(
            left,
            textvariable=self.status_var,
            bg=PANEL_BG,
            fg=TEXT_FG,
            font=self.font_status,
            wraplength=320,
            justify="left",
            anchor="w"
        ).pack(anchor="w", fill=tk.X, pady=(0, 8))

        prog_bg = tk.Frame(left, bg=PROG_BG, height=8, width=320)
        prog_bg.pack(anchor="w", fill=tk.X)
        prog_bg.pack_propagate(False)
        self._prog_bg = prog_bg
        self.prog_fill = tk.Frame(prog_bg, bg=PROG_FILL, height=8, width=0)
        self.prog_fill.place(x=0, y=0, relheight=1)

        # Right side
        tk.Label(right, text="Processed Preview", bg=WINDOW_BG, fg=TEXT_FG, font=self.font_title).grid(row=0, column=0, sticky="w")
        self.preview_mode_var = tk.StringVar(
            value="Click “Load Previews” to generate a 2x2 processed preview grid."
        )
        tk.Label(
            right,
            textvariable=self.preview_mode_var,
            bg=WINDOW_BG,
            fg=SUBTLE_FG,
            font=self.font_small
        ).grid(row=1, column=0, sticky="w", pady=(2, 6))

        self.loading_frame = tk.Frame(right, bg=WINDOW_BG)
        self.loading_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.loading_frame.columnconfigure(0, weight=1)

        self.loading_label = tk.Label(
            self.loading_frame,
            text="",
            bg=WINDOW_BG,
            fg=SUBTLE_FG,
            font=self.font_small
        )
        self.loading_label.grid(row=0, column=0, sticky="w")

        self.preview_progress = ttk.Progressbar(
            self.loading_frame,
            mode="indeterminate",
            length=220
        )
        self.preview_progress.grid(row=0, column=1, sticky="e")

        self.preview_area = tk.Frame(right, bg=WINDOW_BG)
        self.preview_area.grid(row=3, column=0, sticky="nsew")
        self.preview_area.rowconfigure(0, weight=1)
        self.preview_area.columnconfigure(0, weight=1)

        # Grid preview frame
        self.grid_frame = tk.Frame(self.preview_area, bg=WINDOW_BG)
        self.grid_frame.grid(row=0, column=0, sticky="nsew")
        for r in range(GRID_ROWS):
            self.grid_frame.rowconfigure(r, weight=1)
        for c in range(GRID_COLS):
            self.grid_frame.columnconfigure(c, weight=1)

        self.grid_cells = []
        for i in range(4):
            cell = tk.Frame(self.grid_frame, bg=WINDOW_BG, padx=6, pady=6)
            cell.grid(row=i // 2, column=i % 2, sticky="nsew")

            title = tk.Label(
                cell,
                text=f"Frame {i + 1}",
                bg=WINDOW_BG,
                fg=TEXT_FG,
                font=self.font_section
            )
            title.pack(anchor="w", pady=(0, 4))

            img_label = tk.Label(
                cell,
                bg="#efefef",
                bd=1,
                relief="solid"
            )
            img_label.pack()

            placeholder = ImageTk.PhotoImage(Image.new("L", (GRID_CELL_W, GRID_CELL_H), 238))
            img_label.configure(image=placeholder, text="")
            self.grid_cells.append({
                "title": title,
                "label": img_label,
                "photo": placeholder,
            })

        # Big processing preview
        self.big_preview_frame = tk.Frame(self.preview_area, bg=WINDOW_BG)
        self.big_preview_frame.grid(row=0, column=0, sticky="nsew")
        self.big_preview_frame.grid_remove()

        self.big_preview_title_var = tk.StringVar(value="Processing Preview")
        tk.Label(
            self.big_preview_frame,
            textvariable=self.big_preview_title_var,
            bg=WINDOW_BG,
            fg=TEXT_FG,
            font=self.font_section
        ).pack(anchor="w", pady=(0, 4))

        self.big_preview_label = tk.Label(
            self.big_preview_frame,
            bg="#efefef",
            bd=1,
            relief="solid"
        )
        self.big_preview_label.pack(anchor="nw")

        placeholder_big = ImageTk.PhotoImage(Image.new("L", (PREVIEW_BIG_W, PREVIEW_BIG_H), 238))
        self._big_preview_photo = placeholder_big
        self.big_preview_label.configure(image=placeholder_big, text="")

    def _divider(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, pady=8)

    def _slider(self, parent, label, from_, to, default, resolution):
        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill=tk.X, pady=2)

        tk.Label(
            row,
            text=label,
            bg=PANEL_BG,
            fg=TEXT_FG,
            font=self.font_label,
            width=18,
            anchor="w"
        ).pack(side=tk.LEFT)

        var = tk.DoubleVar(value=default)

        scale = tk.Scale(
            row,
            from_=from_,
            to=to,
            orient=tk.HORIZONTAL,
            variable=var,
            resolution=resolution,
            length=180,
            showvalue=0,
            bg=PANEL_BG,
            fg=TEXT_FG,
            activebackground="#666666",
            highlightthickness=0,
            relief="flat",
            troughcolor="#d8d8d8"
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        value_label = tk.Label(
            row,
            textvariable=var,
            bg=PANEL_BG,
            fg=TEXT_FG,
            font=self.font_label,
            width=6,
            anchor="w"
        )
        value_label.pack(side=tk.LEFT, padx=(8, 0))

        return var

    # ------------------------
    # Helpers
    # ------------------------
    def _show_grid_mode(self):
        self.big_preview_frame.grid_remove()
        self.grid_frame.grid()
        self.preview_mode_var.set("2x2 processed preview grid. Click “Load Previews” after changing settings to refresh it.")

    def _show_processing_mode(self):
        self.grid_frame.grid_remove()
        self.big_preview_frame.grid()
        self.preview_mode_var.set("Processing preview updates every ~0.5 seconds while the full output video is rendered.")

    def _set_progress(self, fraction):
        self._prog_bg.update_idletasks()
        total_w = self._prog_bg.winfo_width()
        self.prog_fill.place(x=0, y=0, relheight=1, width=int(total_w * max(0.0, min(1.0, fraction))))

    def _start_loading_indicator(self, text):
        self.loading_label.configure(text=text)
        try:
            self.preview_progress.start(10)
        except Exception:
            pass

    def _stop_loading_indicator(self, text=""):
        try:
            self.preview_progress.stop()
        except Exception:
            pass
        self.loading_label.configure(text=text)

    def _set_grid_placeholder(self, idx, title_text, body_text="Loading..."):
        cell = self.grid_cells[idx]
        cell["title"].configure(text=title_text)

        img = Image.new("RGB", (GRID_CELL_W, GRID_CELL_H), (245, 245, 245))
        photo = ImageTk.PhotoImage(img)
        cell["photo"] = photo
        cell["label"].configure(
            image=photo,
            text=body_text,
            compound="center",
            fg=SUBTLE_FG,
            font=self.font_label
        )

    def _set_grid_image(self, idx, title_text, frame_img):
        cell = self.grid_cells[idx]
        cell["title"].configure(text=title_text)

        photo = self._frame_to_photo(frame_img, GRID_CELL_W, GRID_CELL_H)
        cell["photo"] = photo
        cell["label"].configure(image=photo, text="")

    def _show_big_preview(self, frame_img, caption):
        self._show_processing_mode()
        self.big_preview_title_var.set(caption)
        photo = self._frame_to_photo(frame_img, PREVIEW_BIG_W, PREVIEW_BIG_H)
        self._big_preview_photo = photo
        self.big_preview_label.configure(image=photo, text="")

    def _frame_to_photo(self, frame, target_w, target_h):
        if frame.ndim == 2:
            img = Image.fromarray(frame)
            canvas = Image.new("L", (target_w, target_h), 255)
        else:
            img = Image.fromarray(frame)
            canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))

        img.thumbnail((target_w, target_h), Image.NEAREST)
        paste_x = (target_w - img.width) // 2
        paste_y = (target_h - img.height) // 2
        canvas.paste(img, (paste_x, paste_y))
        return ImageTk.PhotoImage(canvas)

    def _get_video_info(self, path):
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        if fps <= 0:
            fps = 30.0
        return total, fps

    def _read_frame_at(self, cap, idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            return None
        return frame

    def _compute_preview_indices(self, total_frames):
        if total_frames <= 0:
            return [0, 0, 0, 0]
        return [min(total_frames - 1, int(round(i * total_frames / 4.0))) for i in range(4)]

    def _seconds_label(self, frame_idx, fps):
        return max(1, int(round(frame_idx / fps)) + 1)

    def _apply_halftone_settings(self):
        cell_size = int(self.cell_size.get())
        downsample = int(self.downsample.get())
        moge_max_size = int(self.moge_max_size.get())
        n_buckets = int(self.n_buckets.get())
        max_stretch = float(self.max_stretch.get())

        ht.CELL_SIZE = cell_size
        ht.DOWNSAMPLE = downsample
        ht.MOGE_MAX_SIZE = moge_max_size
        ht.N_BUCKETS = n_buckets
        ht.MAX_STRETCH = max_stretch

        if hasattr(ht, "NORMAL_BLEND_ALPHA"):
            ht.NORMAL_BLEND_ALPHA = float(self.normal_blend_alpha.get())
        if hasattr(ht, "SCENE_CHANGE_THRESHOLD"):
            ht.SCENE_CHANGE_THRESHOLD = float(self.scene_threshold.get())

        cy, cx = cell_size // 2, cell_size // 2
        ky, kx = np.ogrid[0:cell_size, 0:cell_size]
        dx, dy = np.broadcast_arrays((kx - cx).astype(np.float32), (ky - cy).astype(np.float32))
        ht.OFFSETS = np.stack([dx.ravel(), dy.ravel()], axis=1)
        ht.RADIUS_SQ = (cell_size * 0.48) ** 2

    def _ordered_dither_compat(self, gray, frame, normal_map, valid_mask, prev_normal=None, prev_gray=None, color=False):
        try:
            return ordered_dither(
                gray=gray,
                frame=frame,
                normal_map=normal_map,
                valid_mask=valid_mask,
                prev_normal=prev_normal,
                prev_gray=prev_gray,
                color=color
            )
        except TypeError:
            try:
                return ordered_dither(
                    gray=gray,
                    frame=frame,
                    normal_map=normal_map,
                    valid_mask=valid_mask,
                    color=color
                )
            except TypeError:
                return ordered_dither(
                    gray=gray,
                    frame=frame,
                    normal_map=normal_map,
                    valid_mask=valid_mask
                )

    def _render_processed_frame(self, curr_frame_bgr, prev_frame_bgr=None, curr_normal=None, curr_mask=None, prev_normal=None):
        self._apply_halftone_settings()

        alpha = float(self.alpha.get())
        color = bool(self.color_mode.get())

        curr_gray = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2GRAY)

        if curr_normal is None or curr_mask is None:
            curr_normal, curr_mask = create_normal_map(curr_frame_bgr)

        prev_gray = None
        stabilized_gray = curr_gray

        if prev_frame_bgr is not None:
            prev_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
            flow = compute_raft_flow(prev_frame_bgr, curr_frame_bgr)
            warped_prev_gray = warp_image_with_flow(prev_gray, flow, border_value=255)
            stabilized_gray = cv2.addWeighted(
                curr_gray,
                alpha,
                warped_prev_gray,
                1.0 - alpha,
                0
            )

        return self._ordered_dither_compat(
            gray=stabilized_gray,
            frame=curr_frame_bgr,
            normal_map=curr_normal,
            valid_mask=curr_mask,
            prev_normal=prev_normal,
            prev_gray=prev_gray,
            color=color
        )

    # ------------------------
    # Browsing / preview grid
    # ------------------------
    def _browse_in(self):
        path = tk.filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")]
        )
        if not path:
            return
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, path)

        self._video_path = path
        self._video_total_frames, self._video_fps = self._get_video_info(path)
        self.status_var.set(
            f"Loaded video. Frames: {self._video_total_frames}, FPS: {self._video_fps:.2f}. Click “Load Previews” to generate the 2x2 grid."
        )
        self._show_grid_mode()

    def _browse_out(self):
        path = tk.filedialog.askdirectory()
        if path:
            self.out_entry.delete(0, tk.END)
            self.out_entry.insert(0, path)

    def _load_previews_clicked(self):
        if self.processing or self.preview_loading:
            return

        path = self.file_entry.get().strip()
        if not path or not os.path.isfile(path):
            tk.messagebox.showerror("Error", "Please select a valid input video file first.")
            return

        self._video_path = path
        self._video_total_frames, self._video_fps = self._get_video_info(path)

        self._show_grid_mode()
        self._preview_generation_id += 1
        generation_id = self._preview_generation_id
        sample_indices = self._compute_preview_indices(self._video_total_frames)

        for i, idx in enumerate(sample_indices):
            sec = self._seconds_label(idx, self._video_fps)
            self._set_grid_placeholder(i, f"≈ {sec}s", "Loading...")

        self.preview_loading = True
        self.load_preview_btn.config(state=tk.DISABLED, text="Loading...")
        self._start_loading_indicator("Generating processed preview grid...")
        self.status_var.set("Generating preview grid...")

        threading.Thread(
            target=self._preview_worker,
            args=(generation_id, path, sample_indices, self._video_fps),
            daemon=True
        ).start()

    def _preview_worker(self, generation_id, path, sample_indices, fps):
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                self.window.after(0, lambda: self.status_var.set("Could not open video for preview."))
                self.window.after(0, self._finish_preview_loading)
                return

            for preview_idx, frame_idx in enumerate(sample_indices):
                if generation_id != self._preview_generation_id:
                    cap.release()
                    return

                curr_frame = self._read_frame_at(cap, frame_idx)
                if curr_frame is None:
                    continue

                prev_frame = None
                prev_normal = None

                if frame_idx > 0:
                    prev_frame = self._read_frame_at(cap, frame_idx - 1)
                    if prev_frame is not None:
                        try:
                            prev_normal, _ = create_normal_map(prev_frame)
                        except Exception:
                            prev_normal = None

                curr_normal, curr_mask = create_normal_map(curr_frame)
                processed = self._render_processed_frame(
                    curr_frame_bgr=curr_frame,
                    prev_frame_bgr=prev_frame,
                    curr_normal=curr_normal,
                    curr_mask=curr_mask,
                    prev_normal=prev_normal
                )

                sec = self._seconds_label(frame_idx, fps)
                self.window.after(
                    0,
                    lambda idx=preview_idx, title=f"≈ {sec}s", img=processed, gid=generation_id: self._maybe_update_grid_cell(gid, idx, title, img)
                )

            cap.release()
            self.window.after(0, lambda gid=generation_id: self._finish_preview_generation(gid))

        except Exception as e:
            self.window.after(0, lambda: self.status_var.set(f"Preview error: {e}"))
            self.window.after(0, self._finish_preview_loading)

    def _maybe_update_grid_cell(self, generation_id, idx, title, img):
        if generation_id != self._preview_generation_id:
            return
        self._set_grid_image(idx, title, img)

    def _finish_preview_loading(self):
        self.preview_loading = False
        self.load_preview_btn.config(state=tk.NORMAL, text="Load Previews")
        self._stop_loading_indicator("")

    def _finish_preview_generation(self, generation_id):
        if generation_id != self._preview_generation_id:
            self._finish_preview_loading()
            return
        self._stop_loading_indicator("Preview grid ready.")
        self.status_var.set("Preview grid ready.")
        self._finish_preview_loading()

    # ------------------------
    # Full processing
    # ------------------------
    def _start_processing(self):
        if self.processing:
            return

        input_path = self.file_entry.get().strip()
        if not input_path or not os.path.isfile(input_path):
            tk.messagebox.showerror("Error", "Please select a valid input video file.")
            return

        out_dir = self.out_entry.get().strip()
        os.makedirs(out_dir, exist_ok=True)

        self.processing = True
        self.run_btn.config(state=tk.DISABLED, text="Processing...")
        self.load_preview_btn.config(state=tk.DISABLED)
        self._set_progress(0)
        self._show_processing_mode()
        self._start_loading_indicator("Processing video... large preview will update every ~0.5 seconds.")
        self.status_var.set("Reading video...")

        threading.Thread(target=self._process, args=(input_path, out_dir), daemon=True).start()

    def _process(self, input_path, out_dir):
        try:
            self._apply_halftone_settings()

            alpha = float(self.alpha.get())
            color = bool(self.color_mode.get())

            frames = read_video(input_path)
            total = len(frames)
            if total == 0:
                self.window.after(0, lambda: self.status_var.set("Error: no frames read."))
                return

            fps = self._video_fps if self._video_fps > 0 else 30.0
            preview_step = max(1, int(round(fps * 0.5)))

            stem, ext = os.path.splitext(os.path.basename(input_path))
            alpha_tag = str(alpha).replace(".", "")
            mode_tag = "rgb" if color else "gray"
            out_path = os.path.join(out_dir, f"{stem}_halftone_{mode_tag}_a{alpha_tag}{ext}")

            output_frames = []
            prev_frame_bgr = None
            prev_gray = None
            prev_normal = None

            cached_normal = None
            cached_mask = None

            for i, frame in enumerate(frames):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if i % NORMAL_KEYFRAME_INTERVAL == 0:
                    self.window.after(0, lambda f=i + 1, t=total: self.status_var.set(f"Normal map — frame {f}/{t}"))
                    cached_normal, cached_mask = create_normal_map(frame)

                stabilized_gray = gray

                if prev_frame_bgr is not None:
                    flow = compute_raft_flow(prev_frame_bgr, frame)
                    warped_prev_gray = warp_image_with_flow(prev_gray, flow, border_value=255)
                    stabilized_gray = cv2.addWeighted(
                        gray,
                        alpha,
                        warped_prev_gray,
                        1.0 - alpha,
                        0
                    )

                self.window.after(0, lambda f=i + 1, t=total: self.status_var.set(f"Halftoning frame {f}/{t}..."))

                curr_halftone = self._ordered_dither_compat(
                    gray=stabilized_gray,
                    frame=frame,
                    normal_map=cached_normal,
                    valid_mask=cached_mask,
                    prev_normal=prev_normal,
                    prev_gray=prev_gray,
                    color=color
                )

                output_frames.append(curr_halftone)

                if (i % preview_step == 0) or (i == total - 1):
                    sec_label = self._seconds_label(i, fps)
                    caption = f"Processing Preview — ≈ {sec_label}s"
                    self.window.after(
                        0,
                        lambda img=curr_halftone.copy(), cap=caption: self._show_big_preview(img, cap)
                    )

                prev_frame_bgr = frame
                prev_gray = gray
                prev_normal = cached_normal

                self.window.after(0, lambda frac=(i + 1) / total: self._set_progress(frac))

            self.window.after(0, lambda: self.status_var.set("Writing video..."))
            write_video(output_frames, out_path, is_color=color)

            self.window.after(0, lambda: self.status_var.set(f"Done! Saved to: {out_path}"))
            self.window.after(0, lambda: self._set_progress(1.0))
            self.window.after(0, lambda p=out_path: tk.messagebox.showinfo("Complete", f"Video saved to:\n{p}"))

        except Exception as e:
            self.window.after(0, lambda msg=str(e): self.status_var.set(f"Error: {msg}"))
            self.window.after(0, lambda msg=str(e): tk.messagebox.showerror("Error", msg))
        finally:
            self.processing = False
            self.window.after(0, lambda: self.run_btn.config(state=tk.NORMAL, text="Process Video"))
            self.window.after(0, lambda: self.load_preview_btn.config(state=tk.NORMAL, text="Load Previews"))
            self.window.after(0, lambda: self._stop_loading_indicator(""))

def main():
    root = tk.Tk()
    app = HalftoneApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()