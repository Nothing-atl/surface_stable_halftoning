import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import threading
import os
import cv2
import numpy as np
from PIL import Image, ImageTk

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from halftone import ordered_dither, create_normal_map
from raft_flow import compute_raft_flow, warp_image_with_flow
from video_utils import read_video, write_video

NORMAL_KEYFRAME_INTERVAL = 5
PREVIEW_W = 760
PREVIEW_H = 480


class HalftoneApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Surface Stable Halftoning")
        self.window.configure(bg="white")
        self.window.resizable(False, False)

        self._first_frame_bgr = None
        self._cached_normal = None
        self._cached_mask = None
        self._preview_job = None
        self.processing = False

        self._build_ui()

    def _build_ui(self):
        # Left panel
        left = tk.Frame(self.window, bg="white", padx=8, pady=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(left, text="Surface Stable Halftoning", bg="white", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(0, 4))

        tk.Label(left, text="Input Video", bg="white", font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        file_row = tk.Frame(left, bg="white")
        file_row.pack(fill=tk.X, pady=(1, 3))
        self.file_entry = tk.Entry(file_row, width=26, font=("TkDefaultFont", 9))
        self.file_entry.pack(side=tk.LEFT)
        tk.Button(file_row, text="Browse", font=("TkDefaultFont", 9), command=self._browse_in).pack(side=tk.LEFT, padx=3)

        tk.Label(left, text="Output Folder", bg="white", font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        out_row = tk.Frame(left, bg="white")
        out_row.pack(fill=tk.X, pady=(1, 6))
        self.out_entry = tk.Entry(out_row, width=26, font=("TkDefaultFont", 9))
        self.out_entry.insert(0, "outputs/videos")
        self.out_entry.pack(side=tk.LEFT)
        tk.Button(out_row, text="Browse", font=("TkDefaultFont", 9), command=self._browse_out).pack(side=tk.LEFT, padx=3)

        tk.Frame(left, bg="#cccccc", height=1).pack(fill=tk.X, pady=2)
        tk.Label(left, text="Halftone Parameters", bg="white", font=("TkDefaultFont", 9, "bold")).pack(anchor="w", pady=(3, 1))

        self.cell_size = self._slider(left, "Cell Size", 4, 32, 8, 1, live=True)
        self.downsample = self._slider(left, "Downsample", 1, 32, 8, 1, live=True)
        self.moge_max_size = self._slider(left, "MoGe Max Size", 64, 512, 256, 32, live=False)
        self.n_buckets = self._slider(left, "Normal Buckets", 64, 2048, 1024, 64, live=False)
        self.max_stretch = self._slider(left, "Max Stretch", 1.0, 6.0, 3.0, 0.1, live=True)

        tk.Frame(left, bg="#cccccc", height=1).pack(fill=tk.X, pady=2)
        tk.Label(left, text="Stabilization Parameters", bg="white", font=("TkDefaultFont", 9, "bold")).pack(anchor="w", pady=(3, 1))

        self.alpha = self._slider(left, "Blend Alpha", 0.0, 1.0, 0.6, 0.05, live=False)
        self.normal_blend_alpha = self._slider(left, "Normal Blend Alpha", 0.0, 1.0, 0.5, 0.05, live=False)
        self.scene_threshold = self._slider(left, "Scene Threshold", 5, 100, 30, 1, live=False)

        tk.Frame(left, bg="#cccccc", height=1).pack(fill=tk.X, pady=2)

        # Color mode toggle
        self.color_mode = tk.BooleanVar(value=False)
        tk.Checkbutton(left, text="RGB Color Mode", variable=self.color_mode,
                       bg="white", font=("TkDefaultFont", 9),
                       command=self._schedule_preview).pack(anchor="w", pady=(4, 2))
 
        tk.Frame(left, bg="#cccccc", height=1).pack(fill=tk.X, pady=4)

        self.run_btn = tk.Button(left, text="Process Video",
                                 command=self._start_processing,
                                 font=("TkDefaultFont", 10, "bold"), width=22,
                                 bg="white", relief="solid", bd=1, cursor="hand2")
        self.run_btn.pack(pady=3)

        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(left, textvariable=self.status_var, bg="white",
                 font=("TkDefaultFont", 8), wraplength=240,
                 justify="left", anchor="w").pack(anchor="w", pady=(3, 1))

        prog_bg = tk.Frame(left, bg="#dddddd", height=5, width=240)
        prog_bg.pack(anchor="w")
        prog_bg.pack_propagate(False)
        self.prog_fill = tk.Frame(prog_bg, bg="black", height=5, width=0)
        self.prog_fill.place(x=0, y=0, relheight=1)
        self._prog_bg = prog_bg
 
        # Right panel: preview
        right = tk.Frame(self.window, bg="white", padx=8, pady=8)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(right, text="First Frame Preview", bg="white", font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        tk.Label(right, text="Live: Cell Size / Downsample / Max Stretch / Color Mode", bg="white", fg="#666666", font=("TkDefaultFont", 8)).pack(anchor="w", pady=(0, 3))

        self.preview_label = tk.Label(right, bg="#eeeeee", relief="sunken", bd=1)
        self.preview_label.pack()

        # Set placeholder image to fix the label to PREVIEW_W x PREVIEW_H pixels
        placeholder = ImageTk.PhotoImage(Image.new("L", (PREVIEW_W, PREVIEW_H), 238))
        self._photo = placeholder
        self.preview_label.config(image=placeholder, text="")

    def _slider(self, parent, label, from_, to, default, resolution, live=False):
        row = tk.Frame(parent, bg="white")
        row.pack(fill=tk.X, pady=0)
        tk.Label(row, text=label, bg="white", font=("TkDefaultFont", 9), width=17, anchor="w").pack(side=tk.LEFT)
        var = tk.DoubleVar(value=default)
        tk.Scale(row, from_=from_, to=to, orient=tk.HORIZONTAL,
                 variable=var, resolution=resolution, length=130,
                 showvalue=0, bg="white", highlightthickness=0, relief="flat",
                 troughcolor="#cccccc").pack(side=tk.LEFT)
        tk.Label(row, textvariable=var, bg="white", font=("TkDefaultFont", 9), width=5, anchor="w").pack(side=tk.LEFT)
        if live:
            var.trace_add("write", lambda *_: self._schedule_preview())
        return var

    def _browse_in(self):
        path = tk.filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")])
        if not path:
            return
        self.file_entry.delete(0, tk.END)
        self.file_entry.insert(0, path)
        self._load_first_frame(path)

    def _browse_out(self):
        path = tk.filedialog.askdirectory()
        if path:
            self.out_entry.delete(0, tk.END)
            self.out_entry.insert(0, path)

    def _load_first_frame(self, path):
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            self.status_var.set("Could not read first frame.")
            return
        self._first_frame_bgr = frame
        self._cached_normal = None
        self._cached_mask = None
        self.status_var.set("Computing normal map...")
        self.window.update_idletasks()
        import halftone as ht
        ht.MOGE_MAX_SIZE = int(self.moge_max_size.get())
        self._cached_normal, self._cached_mask = create_normal_map(frame)
        self.status_var.set("Preview ready.")
        self._render_preview()

    def _schedule_preview(self):
        if self._preview_job is not None:
            self.window.after_cancel(self._preview_job)
        self._preview_job = self.window.after(200, self._render_preview)

    def _render_preview(self):
        self._preview_job = None
        if self._first_frame_bgr is None or self._cached_normal is None:
            return
        import halftone as ht
        cell_size = int(self.cell_size.get())
        downsample = int(self.downsample.get())
        max_stretch = float(self.max_stretch.get())
        n_buckets = int(self.n_buckets.get())
        color = self.color_mode.get()
        ht.CELL_SIZE = cell_size
        ht.DOWNSAMPLE = downsample
        ht.MAX_STRETCH = max_stretch
        ht.N_BUCKETS = n_buckets
        cy, cx = cell_size // 2, cell_size // 2
        ky, kx = np.ogrid[0:cell_size, 0:cell_size]
        dx, dy = np.broadcast_arrays((kx - cx).astype(np.float32), (ky - cy).astype(np.float32))
        ht.OFFSETS = np.stack([dx.ravel(), dy.ravel()], axis=1)
        ht.RADIUS_SQ = (cell_size * 0.48) ** 2
        gray = cv2.cvtColor(self._first_frame_bgr, cv2.COLOR_BGR2GRAY)
        dither = ordered_dither(gray, self._first_frame_bgr, self._cached_normal, self._cached_mask, color=color)
        self._show_preview(dither)

    def _show_preview(self, halftone):
        if halftone.ndim == 2:
            img = Image.fromarray(halftone)
            canvas = Image.new("L", (PREVIEW_W, PREVIEW_H), 255)
        else:
            img = Image.fromarray(halftone)
            canvas = Image.new("RGB", (PREVIEW_W, PREVIEW_H), (255, 255, 255))
        img.thumbnail((PREVIEW_W, PREVIEW_H), Image.NEAREST)
        canvas.paste(img, ((PREVIEW_W - img.width) // 2, (PREVIEW_H - img.height) // 2))
        photo = ImageTk.PhotoImage(canvas)
        self._photo = photo
        self.preview_label.config(image=photo, text="")

    def _set_progress(self, fraction):
        self._prog_bg.update_idletasks()
        total_w = self._prog_bg.winfo_width()
        self.prog_fill.place(x=0, y=0, relheight=1, width=int(total_w * fraction))

    def _start_processing(self):
        if self.processing:
            return
        path = self.file_entry.get().strip()
        if not path or not os.path.isfile(path):
            tk.messagebox.showerror("Error", "Please select a valid input video file.")
            return
        out_dir = self.out_entry.get().strip()
        os.makedirs(out_dir, exist_ok=True)
        self.processing = True
        self.run_btn.config(state=tk.DISABLED, text="Processing...")
        self._set_progress(0)
        threading.Thread(target=self._process, args=(path, out_dir), daemon=True).start()

    def _process(self, input_path, out_dir):
        try:
            import halftone as ht
            cell_size = int(self.cell_size.get())
            downsample = int(self.downsample.get())
            moge_max_size = int(self.moge_max_size.get())
            n_buckets = int(self.n_buckets.get())
            max_stretch = float(self.max_stretch.get())
            alpha = float(self.alpha.get())
            normal_blend_alpha = float(self.normal_blend_alpha.get())
            scene_threshold = float(self.scene_threshold.get())
            color = self.color_mode.get()

            ht.CELL_SIZE = cell_size
            ht.DOWNSAMPLE = downsample
            ht.MOGE_MAX_SIZE = moge_max_size
            ht.N_BUCKETS = n_buckets
            ht.MAX_STRETCH = max_stretch
            ht.NORMAL_BLEND_ALPHA = normal_blend_alpha
            ht.SCENE_CHANGE_THRESHOLD = scene_threshold
            cy, cx = cell_size // 2, cell_size // 2
            ky, kx = np.ogrid[0:cell_size, 0:cell_size]
            dx, dy = np.broadcast_arrays((kx - cx).astype(np.float32), (ky - cy).astype(np.float32))
            ht.OFFSETS = np.stack([dx.ravel(), dy.ravel()], axis=1)
            ht.RADIUS_SQ = (cell_size * 0.48) ** 2

            self.status_var.set("Reading video...")
            frames = read_video(input_path)
            total = len(frames)
            if total == 0:
                self.status_var.set("Error: no frames read.")
                return

            stem, ext = os.path.splitext(os.path.basename(input_path))
            alpha_tag = str(alpha).replace(".", "")
            mode_tag = "rgb" if color else "gray"
            out_path = os.path.join(out_dir, f"{stem}_halftone_{mode_tag}_a{alpha_tag}{ext}")

            output_frames = []
            cached_normal = None
            cached_mask = None
            prev_halftone = None
            prev_frame_bgr = None
            prev_normal = None
            prev_gray = None

            for i, frame in enumerate(frames):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if i % NORMAL_KEYFRAME_INTERVAL == 0:
                    self.status_var.set(f"Normal map — frame {i+1}/{total}")
                    cached_normal, cached_mask = ht.create_normal_map(frame)
                self.status_var.set(f"Halftoning frame {i+1}/{total}...")
                curr_halftone = ht.ordered_dither(
                    gray=gray, frame=frame,
                    normal_map=cached_normal, valid_mask=cached_mask,
                    prev_normal=prev_normal, prev_gray=prev_gray,
                    color=color
                )
                if i == 0:
                    self.window.after(0, lambda h=curr_halftone: self._show_preview(h))
                if prev_halftone is None or prev_frame_bgr is None:
                    stabilized = curr_halftone
                else:
                    flow = compute_raft_flow(prev_frame_bgr, frame)
                    h_out, w_out = curr_halftone.shape[:2]
                    h_in, w_in = gray.shape
                    flow_r = cv2.resize(flow, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
                    flow_r[..., 0] *= w_out / w_in
                    flow_r[..., 1] *= h_out / h_in
                    warped = warp_image_with_flow(prev_halftone, flow_r, border_value=255)
                    blended = cv2.addWeighted(curr_halftone, alpha, warped, 1.0 - alpha, 0)
                    if color:
                        stabilized = (blended > 127).astype(np.uint8) * 255
                    else:
                        _, stabilized = cv2.threshold(blended, 127, 255, cv2.THRESH_BINARY)
                output_frames.append(stabilized)
                prev_halftone = stabilized
                prev_frame_bgr = frame
                prev_normal = cached_normal
                prev_gray = gray
                self.window.after(0, lambda f=i+1: self._set_progress(f / total))

            self.status_var.set("Writing video...")
            write_video(output_frames, out_path, is_color=color)
            self.status_var.set(f"Done! Saved to: {out_path}")
            self.window.after(0, lambda: self._set_progress(1.0))
            tk.messagebox.showinfo("Complete", f"Video saved to:\n{out_path}")

        except Exception as e:
            self.status_var.set(f"Error: {e}")
            tk.messagebox.showerror("Error", str(e))
        finally:
            self.processing = False
            self.window.after(0, lambda: self.run_btn.config(state=tk.NORMAL, text="Process Video"))

def main():
    root = tk.Tk()
    app = HalftoneApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()