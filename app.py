import os
import gc
import gradio as gr
from gradio import Server
from fastapi.responses import HTMLResponse
import numpy as np
import spaces
import torch
import random
import base64
import json
from io import BytesIO
from PIL import Image
from transformers import pipeline as hf_pipeline

MAX_SEED = np.iinfo(np.int32).max
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("Using device:", device)

from diffusers import FlowMatchEulerDiscreteScheduler
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from qwenimage.transformer_qwenimage import QwenImageTransformer2DModel
from qwenimage.qwen_fa3_processor import QwenDoubleStreamAttnProcessorFA3

dtype = torch.bfloat16

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "FireRedTeam/FireRed-Image-Edit-1.1",
    transformer=QwenImageTransformer2DModel.from_pretrained(
        "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19",
        torch_dtype=dtype,
        device_map="cuda",
    ),
    torch_dtype=dtype,
).to(device)

try:
    pipe.transformer.set_attn_processor(QwenDoubleStreamAttnProcessorFA3())
    print("Flash Attention 3 Processor set successfully.")
except Exception as e:
    print(f"Warning: Could not set FA3 processor: {e}")

# ── NCII safety guard ────────────────────────────────────────────────────────
NCII_MODEL_ID = "hfmlsoc/ncii-light-guard-v01"
NCII_UNSAFE_LABEL = "ncii"
NCII_THRESHOLD = 0.5
NCII_BLOCK_MESSAGE = "You entered prompt is NCII (non-consensual intimate imagery) and your request will not be processed. Try with Safe Prompts."

print("Loading NCII safety guard model...")
try:
    ncii_guard = hf_pipeline("text-classification", model=NCII_MODEL_ID, device=-1)
    print("NCII guard loaded successfully.")
except Exception as e:
    ncii_guard = None
    print(f"Warning: Could not load NCII guard model ({NCII_MODEL_ID}): {e}")

def check_ncii_safety(prompt_text):
    if ncii_guard is None or not prompt_text or not prompt_text.strip():
        return False, "unknown", 0.0
    try:
        out = ncii_guard(prompt_text.strip())
        if isinstance(out, list) and len(out) > 0:
            top = out[0]
            label = str(top.get("label", "")).lower()
            score = float(top.get("score", 0.0))
            is_unsafe = (label == NCII_UNSAFE_LABEL) and (score >= NCII_THRESHOLD)
            return is_unsafe, label, score
    except Exception as e:
        print(f"NCII guard inference error: {e}")
    return False, "unknown", 0.0

# ── Examples Config ───────────────────────────────────────────────────────────
EXAMPLES_CONFIG = [
    {"images": ["examples/1.jpg"], "prompt": "cinematic polaroid with soft grain subtle vignette gentle lighting white frame handwritten photographed 'Fire-Edit' preserving realistic texture and details."},
    {"images": ["examples/2.jpg"], "prompt": "Transform the image into a dotted cartoon style."},
    {"images": ["examples/3.jpeg"], "prompt": "Convert it to black and white."},
    {"images": ["examples/4.jpg", "examples/5.jpg"], "prompt": "Replace her glasses with the new glasses from image 1."},
    {"images": ["examples/8.jpg", "examples/9.png"], "prompt": "Replace the current clothing with the clothing from the reference image 2. Keep the person's face, hairstyle, body pose, background, lighting, and camera angle unchanged. Ensure the new outfit fits naturally with realistic fabric texture, proper shadows, folds, and accurate proportions. Match the lighting, color tone, and overall style for a seamless and high-quality result."},
    {"images": ["examples/10.jpg", "examples/11.png"], "prompt": "Replace the current clothing with the clothing from the reference image 2. Keep the person's face, hairstyle, body pose, background, lighting, and camera angle unchanged. Ensure the new outfit fits naturally with realistic fabric texture, proper shadows, folds, and accurate proportions. Match the lighting, color tone, and overall style for a seamless and high-quality result."},
]

def make_thumb_b64(path, max_dim=220):
    if not os.path.exists(path):
        return ""
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_dim, max_dim), LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=65)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
    except Exception as e:
        print(f"Thumbnail error for {path}: {e}")
        return ""

def encode_full_image(path):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            data = f.read()
        ext = path.rsplit(".", 1)[-1].lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")
        return f"data:{mime};base64,{base64.b64encode(data).decode()}"
    except Exception as e:
        print(f"Encode error for {path}: {e}")
        return ""

def build_client_config():
    """Static config consumed by the frontend: example cards."""
    examples = []
    for i, ex in enumerate(EXAMPLES_CONFIG):
        examples.append({
            "idx": i,
            "thumbs": [make_thumb_b64(p) for p in ex["images"]],
            "n_images": len(ex["images"]),
            "prompt": ex["prompt"],
        })
    return {
        "examples": examples,
    }

print("Building client config (example thumbnails)…")
CLIENT_CONFIG = build_client_config()
print(f"Built config with {len(EXAMPLES_CONFIG)} examples.")

def b64_to_pil_list(b64_json_str):
    if not b64_json_str or b64_json_str.strip() in ("", "[]"):
        return []
    try:
        b64_list = json.loads(b64_json_str)
    except Exception:
        return []
    pil_images = []
    for b64_str in b64_list:
        if not b64_str or not isinstance(b64_str, str):
            continue
        try:
            if b64_str.startswith("data:image"):
                _, data = b64_str.split(",", 1)
            else:
                data = b64_str
            image_data = base64.b64decode(data)
            pil_images.append(Image.open(BytesIO(image_data)).convert("RGB"))
        except Exception as e:
            print(f"Error decoding image: {e}")
    return pil_images

def pil_to_b64_png(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

def update_dimensions_on_upload(image):
    if image is None:
        return 1024, 1024
    w, h = image.size
    if w > h:
        nw = 1024
        nh = int(nw * h / w)
    else:
        nh = 1024
        nw = int(nh * w / h)
    return (nw // 8) * 8, (nh // 8) * 8

# ── Gradio Server (Server mode): FastAPI + Gradio queue/API engine ────────────
app = Server(title="FireRed-Image-Edit-1.0-Fast")

@app.mcp.tool(name="edit_image")
@app.api(name="edit_image")
@spaces.GPU(size="xlarge")
def infer(
    images_b64_json: str,
    prompt: str,
    seed: int,
    randomize_seed: bool,
    guidance_scale: float,
    steps: int,
) -> dict:
    """Edit one or more images with FireRed-Image-Edit-1.1.

    Returns {"image": <base64 PNG data URL>, "seed": <seed used>, "status": "success"} 
    or {"status": "blocked", "message": <warning>} if NCII triggers.
    """
    gc.collect()
    torch.cuda.empty_cache()

    pil_images = b64_to_pil_list(images_b64_json)
    if not pil_images:
        raise gr.Error("Please upload at least one image to edit.")
    if not prompt or prompt.strip() == "":
        raise gr.Error("Please enter an edit prompt.")

    # ── NCII safety check ──
    is_unsafe, _, _ = check_ncii_safety(prompt)
    if is_unsafe:
        gc.collect()
        torch.cuda.empty_cache()
        # Returning a blocked status instead of raising an error
        # so the frontend can gracefully catch it and display a warning toast.
        return {"image": "", "seed": seed, "status": "blocked", "message": NCII_BLOCK_MESSAGE}

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=device).manual_seed(seed)
    negative_prompt = (
        "worst quality, low quality, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, fewer digits, cropped, jpeg artifacts, signature, watermark, username, blurry"
    )
    width, height = update_dimensions_on_upload(pil_images[0])

    try:
        result_image = pipe(
            image=pil_images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            generator=generator,
            true_cfg_scale=guidance_scale,
        ).images[0]
        return {"image": pil_to_b64_png(result_image), "seed": seed, "status": "success"}
    except Exception as e:
        raise e
    finally:
        gc.collect()
        torch.cuda.empty_cache()


@app.api(name="load_example", queue=False)
def load_example(idx: float) -> dict:
    """Return base64-encoded example images + prompt for a given example index."""
    try:
        i = int(idx)
    except (ValueError, TypeError):
        i = -1
    if i < 0 or i >= len(EXAMPLES_CONFIG):
        return {"images": [], "prompt": "", "names": [], "status": "error"}
    ex = EXAMPLES_CONFIG[i]
    b64_list, names = [], []
    for path in ex["images"]:
        b64 = encode_full_image(path)
        if b64:
            b64_list.append(b64)
            names.append(os.path.basename(path))
    return {"images": b64_list, "prompt": ex["prompt"], "names": names, "status": "ok"}


@app.get("/api/config")
def client_config():
    """Plain FastAPI route: example card data for the frontend."""
    return CLIENT_CONFIG


@app.get("/", response_class=HTMLResponse)
async def homepage():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    app.launch(show_error=True, mcp_server=True)