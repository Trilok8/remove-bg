import io
import os
import time
from typing import Literal, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse
from PIL import Image
from rembg import remove, new_session

# ===== (Optional) CUDA speed-ups =====
# Keep your CUDA FP16 flag for extra speed on supported GPUs
os.environ["ORT_CUDA_ALLOW_FP16_PRECISION"] = "1"

# ===== Providers (GPU preferred) =====
PROVIDERS = ["CUDAExecutionProvider"]  # GPU-only preference; we will fallback to CPU in try/except

# ===== Initialize a single high-perf session on startup =====
def build_session():
    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = new_session("isnet-general-use", providers=PROVIDERS, sess_options=so)
        using = getattr(session, "_providers", ["rembg default"])
        return session, {"available": ort.get_available_providers(), "using": using}
    except Exception:
        # Fallback to rembg defaults (usually CPU)
        session = new_session("isnet-general-use")
        try:
            import onnxruntime as ort
            using = getattr(session, "_providers", ["rembg default"])
            return session, {"available": ort.get_available_providers(), "using": using}
        except Exception:
            # onnxruntime may not be importable in some environments; return minimal info
            return session, {"available": [], "using": ["rembg default"]}

app = FastAPI(title="Rembg Cutout Server", version="1.0.0", docs_url="/docs")

SESSION, PROVIDER_INFO = build_session()


def _read_image_from_upload(file: UploadFile) -> Image.Image:
    # Load bytes → PIL (keep original resolution)
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload.")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=415, detail="Unsupported or corrupted image file.")
    return img


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"


@app.get("/providers")
def providers():
    # Helpful for confirming GPU usage from outside
    return JSONResponse(PROVIDER_INFO)


@app.post(
    "/cutout",
    summary="Foreground cutout",
    description="Upload an image and get back a PNG. Default returns RGBA cutout. Use return=mask for alpha mask."
)
def cutout(
    file: UploadFile = File(..., description="Image file (png/jpg/jpeg/webp)"),
    ret: Literal["rgba", "mask"] = Query("rgba", alias="return", description="rgba (default) or mask"),
    download_name: Optional[str] = Query(None, description="Optional download filename (without extension)")
):
    t0 = time.time()
    img = _read_image_from_upload(file)

    # Compute cutout / mask
    try:
        out = remove(img, session=SESSION)  # RGBA result, same size as input
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation error: {e}")

    if ret == "mask":
        # Convert RGBA alpha channel to single-channel mask (white=fg, black=bg)
        if out.mode != "RGBA":
            out = out.convert("RGBA")
        alpha = out.split()[-1]
        payload = _to_png_bytes(alpha)
    else:
        payload = _to_png_bytes(out)


    base_name = os.path.splitext(file.filename or "cutout")[0]
    safe_name = base_name.replace(" ", "_").replace("/", "_")
    out_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    dst_path = os.path.join(out_dir, safe_name + ".png")

    with open(dst_path, "wb") as f:
        f.write(payload)

    print(f"[SAVED] {dst_path}")

    elapsed = time.time() - t0
    headers = {"X-Processing-Time": f"{elapsed:.2f}s"}

    filename = (download_name or os.path.splitext(file.filename or "cutout")[0]) + ".png"
    headers["Content-Disposition"] = f'inline; filename="{filename}"'

    return StreamingResponse(io.BytesIO(payload), media_type="image/png", headers=headers)