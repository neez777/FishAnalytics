import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", message="cannot import name '_C' from 'sam2'")
warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")

# Use bfloat16 for the entire script
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# MODEL CHECKPOINTS
sam2_checkpoint = "C:\\Users\\neez\\Documents\\FishAnalytics\\SAM2\\segment-anything-2\\checkpoints\\sam2_hiera_large.pt"

# CORRESPONDING CHECKPOINT YAML FILE
model_cfg = "sam2_hiera_l.yaml"

# CREATE PREDICTOR OBJECT
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

# MASK FUNCTION
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])  # DeepSkyBlue
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# FUNCTION FOR SHOWING POINTS
def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# FOLDER THAT CONTAINS JPG FRAMES
video_dir = "C:\\Users\\neez\\Documents\\FishAnalytics\\SAM2\\segment-anything-2\\video-frames"

# SCAN ALL JPG FRAME NAMES IN FOLDER
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", "JPEG"]
]
frame_names.sort(key=lambda p: int(''.join(filter(str.isdigit, os.path.splitext(p)[0]))))

class SAM2GUI:
    def __init__(self, master):
        self.master = master
        master.title("SAM2 Interactive GUI")

        self.frame_idx = 0
        self.inference_state = None
        self.video_segments = {}

        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.canvas.mpl_connect('button_press_event', self.on_click)

        button_frame = ttk.Frame(master)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.prev_button = ttk.Button(button_frame, text="Previous", command=self.prev_frame)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = ttk.Button(button_frame, text="Next", command=self.next_frame)
        self.next_button.pack(side=tk.RIGHT)

        self.reset_button = ttk.Button(button_frame, text="Reset", command=self.reset_state)
        self.reset_button.pack(side=tk.BOTTOM)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(master, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=10)

        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(master, textvariable=self.status_var)
        self.status_label.pack()

        self.reset_state()

    def reset_state(self):
        self.inference_state = predictor.init_state(video_path=video_dir)
        predictor.reset_state(self.inference_state)
        self.video_segments = {}
        self.frame_idx = 0
        self.update_image()
        self.status_var.set("Ready. Click on an object to start segmentation.")

    def update_image(self):
        self.ax.clear()
        img = Image.open(os.path.join(video_dir, frame_names[self.frame_idx]))
        self.ax.imshow(img)
        self.ax.set_title(f"Frame {self.frame_idx}")

        if self.frame_idx in self.video_segments:
            for obj_id, mask in self.video_segments[self.frame_idx].items():
                show_mask(mask, self.ax)

        self.canvas.draw()
        self.status_var.set(f"Viewing frame {self.frame_idx + 1} of {len(frame_names)}")

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata
        points = np.array([[x, y]], dtype=np.float32)
        labels = np.array([1], np.int32)

        try:
            _, out_obj_ids, out_mask_logits = predictor.add_new_points(
                inference_state=self.inference_state,
                frame_idx=self.frame_idx,
                obj_id=1,
                points=points,
                labels=labels,
            )

            mask = (out_mask_logits[0] > 0.0).cpu().numpy()
            if self.frame_idx not in self.video_segments:
                self.video_segments[self.frame_idx] = {}
            self.video_segments[self.frame_idx][out_obj_ids[0]] = mask

            self.update_image()
            self.status_var.set("Segmentation added. Use 'Next' to propagate.")
        except RuntimeError as e:
            messagebox.showerror("Error", str(e))

    def prev_frame(self):
        if self.frame_idx > 0:
            self.frame_idx -= 1
            self.update_image()

def next_frame(self):
    if self.frame_idx < len(frame_names) - 1:
        self.frame_idx += 1

        if self.frame_idx not in self.video_segments and self.video_segments:
            try:
                # Find the last frame with a mask before the current frame
                last_mask_frame = max(idx for idx in self.video_segments.keys() if idx < self.frame_idx)

                self.status_var.set("Propagating mask...")
                self.master.update_idletasks()

                # Propagate masks from the last frame with a mask to the current frame
                for prop_frame in range(last_mask_frame + 1, self.frame_idx + 1):
                    # Run mask propagation on the current frame
                    for _, out_obj_ids, out_mask_logits in predictor.propagate_in_video(self.inference_state):
                        if out_obj_ids:
                            mask = (out_mask_logits[0] > 0.0).cpu().numpy()

                            # Store the propagated mask in the video_segments dictionary
                            if prop_frame not in self.video_segments:
                                self.video_segments[prop_frame] = {}
                            self.video_segments[prop_frame][out_obj_ids[0]] = mask
                        break

                    # Update progress bar during propagation
                    progress = (prop_frame - last_mask_frame) / (self.frame_idx - last_mask_frame) * 100
                    self.progress_var.set(progress)
                    self.master.update_idletasks()

                self.status_var.set("Mask propagation complete.")
            except RuntimeError as e:
                messagebox.showwarning("Warning", f"Could not propagate mask: {str(e)}")

        # Ensure the propagated mask is shown on the current frame
        self.update_image()
    self.progress_var.set(0)


root = tk.Tk()
gui = SAM2GUI(root)
root.mainloop()
