# 🎥 Surface-Stable Halftoning

This project focuses on generating **temporally stable halftone videos**.  
Instead of applying halftoning independently to each frame (which causes flickering), we combine **geometry awareness** and **motion-based stabilization** to produce visually coherent stylized videos.

---

##  Overview

Halftoning converts images into patterns of dots based on intensity. While effective for static images, applying it to video frame-by-frame leads to unstable, noisy results.

This project improves stability by combining:

-  **Surface normals (MoGe-2)** → geometry-aware dot shaping  
-  **Optical flow (RAFT)** → motion tracking across frames  
-  **Warping + blending** → temporal consistency  

---

##  Pipeline

1. **Frame Extraction**
   - Read video and convert frames to grayscale  

2. **Surface Normal Estimation**
   - Use MoGe-2 to estimate per-pixel normals  

3. **Ordered Dithering**
   - Apply Bayer matrix thresholding  

4. **Normal-Based Dot Shaping**
   - Generate ellipse-shaped dots aligned with surface orientation  

5. **Optical Flow (RAFT)**
   - Compute dense motion between frames  

6. **Warping + Stabilization**
   - Warp previous frame using flow  
   - Blend with current frame before dithering  

---

##  Features

- ✔ Surface-aware halftone rendering  
- ✔ Motion-aware temporal stabilization  
- ✔ Adjustable parameters (cell size, alpha, etc.)  
- ✔ GUI for interactive experimentation  
- ✔ Live preview during processing  

---

## Running the Project

### GUI (Recommended)
```bash
python src/gui.py
```
If you are on Mac (MPS issue with MoGe):
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python src/gui.py
```

### Testing / Batch Mode
For development or batch processing without the GUI:
```bash
python src/main.py
```

##  Performance

This pipeline is computationally expensive due to:

- Deep learning models (MoGe + RAFT)  
- Per-frame processing  

Typical runtime:

- CPU: very slow (hours)  
- GPU (Colab): much faster (minutes)  

---

##  Key Technologies

- PyTorch  
- OpenCV  
- MoGe-2 (surface normal estimation)  
- RAFT (optical flow)  
- Tkinter (GUI)  

---

##  Project Structure

```bash
src/
├── main.py # Main pipeline (batch processing)
├── gui.py # GUI application
├── halftone.py # Halftone + MoGe logic
├── raft_flow.py # Optical flow + warping
├── video_utils.py # Video I/O utilities

```

---

##  Results

Compared to standard halftoning:

- ✔ Reduced flickering  
- ✔ Improved temporal consistency  
- ✔ Geometry-aware dot patterns  

---

##  Limitations

- High computational cost  
- Optical flow errors can cause artifacts  
- Dots are not fully persistent (blended, not tracked)  

---

##  Future Work

- Persistent dot tracking in surface space  
- Real-time optimization  
- Lightweight optical flow models  
- Fractal or noise-based halftoning  

---

##  Authors
- Aleena Treesa Leejoy
- Brendon Kim 
- Prateek S. Arora

---

##  Notes

- MoGe may require CPU fallback on Mac due to MPS limitations  
- Best performance achieved using any CUDA-capable GPU

---

##  Acknowledgements

- MoGe-2 (surface estimation)  
- RAFT (optical flow)  
- Classic ordered dithering techniques  
