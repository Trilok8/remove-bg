# remove-bg

A FastAPI server for background removal using [rembg](https://github.com/danielgatis/rembg) and [ONNXRuntime](https://onnxruntime.ai/) with GPU (CUDA) or CPU fallback.

### Example 1
![Input 1](input1.JPG)
![Output 1](outputs/input1.png)

### Example 2
![Input 2](input2.JPG)
![Output 2](outputs/input2.png)

### Example 3
![Input 3](input3.JPG)
![Output 3](outputs/input3.png)

---

## Features
- Upload image via POST `/cutout`
- Returns cutout as transparent PNG or alpha mask
- Saves processed files automatically to `outputs/`
- Health check endpoint (`/health`)
- Provider info endpoint (`/providers`) to confirm GPU usage
- Supports CUDAExecutionProvider (GPU) with CPU fallback

---

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/remove-bg.git
cd remove-bg

# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

# GPU version (requires CUDA + compatible drivers)
pip install -r requirements-gpu.txt

# OR CPU-only version (simpler, slower)
pip install -r requirements.txt