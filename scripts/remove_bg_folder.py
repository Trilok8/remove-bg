import os, glob, time
from PIL import Image
from rembg import remove, new_session

# ===== GPU providers (CUDA only) =====
PROVIDERS = [
    "CUDAExecutionProvider",  # GPU
]

# Allow FP16 on CUDA for extra speed (safe for segmentation)
os.environ["ORT_CUDA_ALLOW_FP16_PRECISION"] = "1"

# I/O
INPUT_DIR  = "inputs/1X"
OUTPUT_DIR = "outputs/5X"
MODEL      = "isnet-general-use"   # best quality people model in rembg

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----- Build a single high-perf session (CUDA → CPU) -----
try:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = new_session(MODEL, providers=PROVIDERS, sess_options=so)
except Exception:
    session = new_session(MODEL)

# (Optional) show which EPs are actually active
try:
    import onnxruntime as ort
    print("Available:", ort.get_available_providers())
    print("Using   :", getattr(session, "_providers", "rembg default"))
except Exception:
    pass

def cutout_one(path: str) -> None:
    t0 = time.time()
    img = Image.open(path).convert("RGB")   # keep original resolution
    out = remove(img, session=session)      # RGBA, same size
    base = os.path.splitext(os.path.basename(path))[0]
    dst  = os.path.join(OUTPUT_DIR, f"{base}_cutout.png")
    out.save(dst)
    print(f"{base}: {img.size[0]}x{img.size[1]} -> {os.path.getsize(dst)//1024} KB in {time.time()-t0:.2f}s")

# ----- Batch all images -----
files = []
for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
    files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))

if not files:
    print(f"No images found in {INPUT_DIR}")
else:
    t_all = time.time()
    for f in files:
        cutout_one(f)
    print(f"Total {len(files)} images in {time.time()-t_all:.2f}s")