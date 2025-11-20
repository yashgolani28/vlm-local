import os
import io
import time
import json
import base64
import tempfile
import sys
import copy
from datetime import datetime
import re
from collections import Counter

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

from retrieval_db import RetrievalDB  
from build_retrieval_db_from_captions import build_retrieval_db_from_jsonl
from anpr import run_anpr, run_anpr_multi, INDIA_STATE_CODES
from auto_caption_traffic_folder import main as auto_caption_main
from train_lora_llava import run_llava_lora
from train_lora_qwenvl import run_qwenvl_lora

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# Qwen2.5-VL + LoRA imports
import torch

try:
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        BitsAndBytesConfig,
    )
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info

    _HAS_QWEN = True
    _QWEN_IMPORT_ERROR = None
except Exception as e:
    _HAS_QWEN = False
    _QWEN_IMPORT_ERROR = e

    AutoProcessor = None
    Qwen2_5_VLForConditionalGeneration = None
    BitsAndBytesConfig = None
    PeftModel = None
    process_vision_info = None

# =========================
# Global config
# =========================

# Ollama / LLaVA config
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "llava:13b")

# Default RTSP for your Axis M1125
DEFAULT_RTSP = (
    "rtsp://root:2024@192.168.1.241/axis-media/media.amp?streamprofile=stream1"
)

# Dataset paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
IMAGES_DIR = os.path.join(DATA_DIR, "traffic_images")
DATASET_PATH = os.path.join(DATA_DIR, "traffic_vqa.jsonl")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Training / auto-caption log file
TRAIN_LOG_PATH = os.path.join(DATA_DIR, "train_logs.txt")


def append_train_log(header: str, text: str) -> None:
    """
    Append a chunk of log text into TRAIN_LOG_PATH with a small header + timestamp.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(TRAIN_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n\n===== {header} @ {ts} =====\n")
            f.write(text)
            f.write("\n")
    except Exception as e:
        # Silent fail so Streamlit doesn't crash
        print(f"[TRAIN_LOG] Failed to write log: {e}")


def run_with_logging(header: str, fn, *args, **kwargs):
    """
    Run a callable while capturing stdout/stderr, then append them to TRAIN_LOG_PATH.

    Usage:
        run_with_logging("TRAIN_LLAVA", run_llava_lora, train_path=..., out_dir=..., ...)
    """
    import io as _io
    from contextlib import redirect_stdout, redirect_stderr

    buf = _io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            result = fn(*args, **kwargs)
    except Exception as e:
        log_text = buf.getvalue()
        log_text += f"\n\n[ERROR] {repr(e)}\n"
        append_train_log(header, log_text)
        raise
    else:
        log_text = buf.getvalue()
        append_train_log(header, log_text)
        return result

# Backend selection + Qwen LoRA defaults
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_BACKEND = os.environ.get("VLM_BACKEND", "qwenvl")  # default to Qwen on the new GPU
QWEN_MODEL_ID = os.environ.get("VLM_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")

QWEN_LORA_DIR = os.environ.get(
    "VLM_LORA_DIR",
    os.path.join(PROJECT_ROOT, "outputs", "qwenvl_lora"),
)

# ---- VLM generation budgets (tokens) ----
BIG_MAX_NEW_TOKENS = int(os.environ.get("VLM_BIG_MAX_NEW_TOKENS", "512"))
LIVE_MAX_NEW_TOKENS = int(os.environ.get("VLM_LIVE_MAX_NEW_TOKENS", "192"))
DATASET_MAX_NEW_TOKENS = int(os.environ.get("VLM_DATASET_MAX_NEW_TOKENS", "384"))
FAST_MAX_NEW_TOKENS = int(os.environ.get("VLM_MAX_NEW_TOKENS", "256"))

# OCR config
USE_VLM_OCR = os.environ.get("USE_VLM_OCR", "").lower() in ("1", "true", "yes", "on")
PLATE_OCR_BACKEND = (os.environ.get("PLATE_OCR_BACKEND") or DEFAULT_BACKEND or "").lower()

_qwen_model = None
_qwen_processor = None

# Retrieval DB (semantic + plate search over stored items)
_retrieval_db = None

def reset_retrieval_db():
    """Reset the cached RetrievalDB instance so it reloads on next access."""
    global _retrieval_db
    _retrieval_db = None

def get_retrieval_db():
    """
    Lazy-load the RetrievalDB so importing vlm_dashboard.py stays cheap.
    """
    global _retrieval_db
    if _retrieval_db is None:
        _retrieval_db = RetrievalDB()
    return _retrieval_db


# YOLO-based vehicle detector (COCO classes)
_VEH_DET_MODEL = None
# COCO IDs for vehicles: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_CLASS_IDS = {2, 3, 5, 7}


def get_vehicle_detector():
    """Lazy-load YOLOv8n for vehicle detection."""
    global _VEH_DET_MODEL
    if _VEH_DET_MODEL is not None:
        return _VEH_DET_MODEL
    if YOLO is None:
        return None

    weights_path = os.path.join(BASE_DIR, "yolov8n.pt")
    if not os.path.exists(weights_path):
        # No vehicle detector weights found
        return None

    try:
        _VEH_DET_MODEL = YOLO(weights_path)
    except Exception:
        _VEH_DET_MODEL = None
    return _VEH_DET_MODEL


# =========================
# Helper functions
# =========================

def frame_to_base64_jpeg(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    jpg_bytes = buf.tobytes()
    return base64.b64encode(jpg_bytes).decode("utf-8")


def pil_to_base64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------- LLaVA (Ollama) helpers ----------

def call_llava_single_image(b64_image: str, prompt: str) -> str:
    """
    Call LLaVA via Ollama with a single base64 JPEG image.
    """
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64_image],
            }
        ],
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def call_llava_multi_images(b64_images, prompt: str) -> str:
    """
    Call LLaVA via Ollama with multiple base64 JPEG images (e.g. video frames).
    """
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": b64_images,
            }
        ],
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=900)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


# ---------- Qwen2.5-VL + LoRA helpers ----------

@st.cache_resource(show_spinner="Loading Qwen2.5-VL model (one-time, cached)...")
def _load_qwenvl_lora_cached(model_id: str, lora_dir: str | None):
    """
    Load Qwen2.5-VL + optional LoRA once per Streamlit server process.
    Subsequent calls reuse the cached model + processor.
    """
    # 4-bit quantization for 7B on 16 GB GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load base Qwen2.5-VL
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = base

    # Try to attach LoRA only if the directory actually exists
    if lora_dir and os.path.isdir(lora_dir):
        try:
            model = PeftModel.from_pretrained(base, lora_dir)
        except Exception as e:
            # Log / print but don't crash – just use base model
            print(f"[QWEN] Failed to load LoRA from {lora_dir}: {e}")
            model = base
    else:
        print(f"[QWEN] No valid LoRA dir at {lora_dir!r}, using base model only.")

    model.eval()

    # use_fast=True gets rid of the warning and uses the fast tokenizer / processor
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    return model, processor

def get_qwenvl_lora():
    """
    Thin wrapper that uses the cached loader above.
    The cache key is (QWEN_MODEL_ID, QWEN_LORA_DIR).
    """
    return _load_qwenvl_lora_cached(QWEN_MODEL_ID, QWEN_LORA_DIR)

def _resize_image_for_vlm(img: Image.Image, max_long_side: int = 480) -> Image.Image:
    """
    Downscale input image for VLM only (ANPR uses full-res elsewhere).
    Keeps aspect ratio; caps the longer side to max_long_side.
    """
    w, h = img.size
    long_side = max(w, h)
    if long_side <= max_long_side:
        return img
    scale = max_long_side / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BILINEAR)


def _estimate_edge_density(img: Image.Image) -> float:
    """
    Cheap 'complexity' metric: edge density on a small grayscale thumbnail.
    Returns ~0.0 (very flat scene) up to ~0.2+ (very busy).
    """
    # Small thumbnail to keep this cheap
    arr = np.array(img.resize((320, 320)))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    # edges is 0..255; mean/255 ≈ fraction of edge pixels
    return float(edges.mean() / 255.0)


def _auto_select_max_new_tokens(pil_images) -> int:
    """
    Pick max_new_tokens based on average visual complexity of the images.
    Fewer tokens for very simple scenes; more for busy scenes.
    """
    if not pil_images:
        # reasonable default if we somehow have no images
        return 256

    densities = [_estimate_edge_density(im) for im in pil_images]
    score = float(np.mean(densities))

    # very flat rural road / empty scene
    if score < 0.015:
        return 256
    # normal scenes with a few vehicles / pedestrians
    elif score < 0.035:
        return 320
    # denser traffic / more clutter
    elif score < 0.07:
        return 448
    # very busy scene – allow long answers
    else:
        return 640


def _qwenvl_generate(messages, max_new_tokens: int | None = None) -> str:
    """
    Shared generation helper for Qwen2.5-VL.
    - Downscales images only for the VLM call (ANPR uses full-res elsewhere).
    - If max_new_tokens is None, automatically picks a value based on
      scene complexity (edge density heuristic).
    `messages` is a Qwen-style chat list with image(s) + text.
    """
    model, processor = get_qwenvl_lora()

    # Work on a deep copy so we don't mutate caller's messages in-place
    msgs = copy.deepcopy(messages)

    pil_images_for_complexity = []
    for m in msgs:
        for c in m.get("content", []):
            if isinstance(c, dict) and c.get("type") == "image":
                img = c.get("image")
                if isinstance(img, Image.Image):
                    resized = _resize_image_for_vlm(img)
                    c["image"] = resized
                    pil_images_for_complexity.append(resized)

    if max_new_tokens is None:
        max_new_tokens = _auto_select_max_new_tokens(pil_images_for_complexity)

    text = processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(msgs)
    proc_inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )

    device = model.device
    for k, v in proc_inputs.items():
        if isinstance(v, torch.Tensor):
            proc_inputs[k] = v.to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **proc_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  
            temperature=1.0,   
        )

    # Strip the prompt tokens
    gen_ids = outputs[0, proc_inputs["input_ids"].shape[1]:]
    out = processor.tokenizer.decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return out.strip()


def call_qwenvl_single_image(
    img: Image.Image,
    prompt: str,
    max_new_tokens: int | None = None,
) -> str:
    """
    Run Qwen2.5-VL + LoRA on a single PIL image.
    If max_new_tokens is None, a value is chosen automatically
    based on scene complexity.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return _qwenvl_generate(messages, max_new_tokens=max_new_tokens)


def call_qwenvl_multi_frames(
    frames,
    prompt: str,
    max_new_tokens: int | None = None,
) -> str:
    """
    Run Qwen2.5-VL + LoRA on multiple video frames (list of BGR numpy arrays).
    Treats them as multiple images in a single user turn.
    If max_new_tokens is None, it's chosen automatically from overall complexity.
    """
    pil_images = [
        Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames
    ]
    content = [{"type": "image", "image": img} for img in pil_images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    return _qwenvl_generate(messages, max_new_tokens=max_new_tokens)

# ---------- Plate OCR via VLM (Qwen / LLaVA) ----------

def _clean_vlm_plate_text(raw_text: str) -> str:
    """
    Post-process raw VLM output into a clean plate string:
    - take first line
    - drop prefixes like 'Plate:'
    - keep only A–Z, 0–9 and spaces
    - collapse multiple spaces
    """
    if not raw_text:
        return ""

    import re

    first_line = str(raw_text).strip().splitlines()[0]

    # Drop leading 'plate:' or similar
    first_line = re.sub(r"^[Pp]late[:\-]\s*", "", first_line)

    filtered_chars = []
    for ch in first_line:
        if ch.isalnum():
            filtered_chars.append(ch.upper())
        elif ch in {" ", "-"}:
            filtered_chars.append(" ")

    plate = "".join(filtered_chars).strip()
    # Collapse multiple spaces
    plate = re.sub(r"\s+", " ", plate)

    # Very short strings are likely garbage
    if len(plate.replace(" ", "")) < 4:
        return ""

    return plate


def vlm_ocr_plate_from_path(crop_path, backend=None):
    """
    Run Qwen or LLaVA OCR on a single cropped plate image.

    Returns (plate_text, pseudo_confidence).
    If OCR fails, returns ("", 0.0).
    """
    if not crop_path:
        return "", 0.0

    try:
        img = Image.open(crop_path).convert("RGB")
    except Exception as e:
        print(f"[VLM_OCR] Failed to open crop {crop_path}: {e}")
        return "", 0.0

    backend = (backend or PLATE_OCR_BACKEND or DEFAULT_BACKEND or "").lower()
    use_qwen = "qwen" in backend

    prompt = (
        "You are an OCR engine specialised in reading vehicle registration plates.\n"
        "The image is a tightly cropped number plate from a traffic scene.\n"
        "Return ONLY the exact registration number as a single line, for example:\n"
        "  MH12AB1234\n"
        "Do not add any extra words, punctuation, labels, or explanation."
    )

    try:
        if use_qwen:
            raw = call_qwenvl_single_image(img, prompt, max_new_tokens=32)
        else:
            # LLaVA via Ollama
            b64 = pil_to_base64_jpeg(img)
            raw = call_llava_single_image(b64, prompt)
    except Exception as e:
        print(f"[VLM_OCR] Backend call failed for {crop_path}: {e}")
        return "", 0.0

    plate = _clean_vlm_plate_text(raw)
    if not plate:
        return "", 0.0

    pseudo_conf = 0.9
    return plate, pseudo_conf

# ---------- Video / RTSP helpers ----------

def sample_video_frames_from_file(file_bytes: bytes, num_frames: int = 6):
    """
    Save uploaded video bytes to a temp file, then sample frames with OpenCV.
    Returns a list of frames (np.ndarray, BGR) and indices.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        cap.release()
        os.remove(tmp_path)
        raise RuntimeError("Could not open uploaded video file.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        os.remove(tmp_path)
        raise RuntimeError("Video appears to have no frames.")

    step = max(1, frame_count // num_frames)

    frames = []
    indices = []
    for i in range(num_frames):
        idx = min(i * step, frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frames.append(frame)
        indices.append(idx)

    cap.release()
    os.remove(tmp_path)

    if not frames:
        raise RuntimeError("Failed to extract frames from video.")

    return frames, indices


def capture_rtsp_frame(rtsp_url: str):
    """
    Capture a single frame from an RTSP stream.
    """
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open RTSP stream: {rtsp_url}")

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Failed to read frame from RTSP stream.")
    return frame


# ---------- Dataset helpers ----------

def save_dataset_example(img: Image.Image, instruction: str, answer: str, tag: str = None):
    """
    Save an example into traffic_vqa.jsonl and store the image in traffic_images/.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    img_name = f"traffic_{ts}.jpg"
    img_path = os.path.join(IMAGES_DIR, img_name)

    img.save(img_path, format="JPEG")

    rel_img_path = os.path.relpath(img_path, DATA_DIR).replace("\\", "/")

    record = {
        "image": rel_img_path,
        "conversations": [
            {"from": "human", "value": instruction},
            {"from": "assistant", "value": answer},
        ],
    }
    if tag:
        record["tag"] = tag

    with open(DATASET_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return rel_img_path


def load_dataset_preview(max_rows: int = 20):
    if not os.path.exists(DATASET_PATH):
        return [], 0
    rows = []
    total = 0
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            if len(rows) < max_rows:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows, total


# ---------- Vehicle + ANPR helpers ----------

def detect_vehicles(frame: np.ndarray):
    """
    Run YOLO on the frame and return a list of vehicle detections with xyxy boxes.
    """
    model = get_vehicle_detector()
    if model is None:
        return []

    try:
        res = model.predict(
            source=frame,
            imgsz=640,
            conf=0.25,
            iou=0.5,
            verbose=False,
        )[0]
    except Exception:
        return []

    if not getattr(res, "boxes", None) or len(res.boxes) == 0:
        return []

    boxes = res.boxes.xyxy.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    confs = res.boxes.conf.cpu().numpy()

    vehicles = []
    for (x1, y1, x2, y2), c, s in zip(boxes, clss, confs):
        if c in VEHICLE_CLASS_IDS:
            vehicles.append(
                {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "cls": int(c),
                    "conf": float(s),
                }
            )
    return vehicles


def run_anpr_on_frame(frame: np.ndarray, save_root: str = None, min_conf: float = 0.40):
    """
    Detect vehicles in a frame and run ANPR on each vehicle ROI.
    If no vehicles are detected (or the vehicle model is missing),
    fall back to full-frame ANPR via run_anpr_multi.

    Returns list of dicts:
      {
        "vehicle_bbox": [x1, y1, x2, y2] or None,
        "vehicle_conf": float,
        "plate_text": str,
        "plate_conf": float,
        "plate_bbox": [x1, y1, x2, y2] or None,
        "crop_path": str or None,
      }
    """
    if save_root is None:
        save_root = os.path.join(DATA_DIR, "snapshots")
    os.makedirs(save_root, exist_ok=True)

    # Save the frame once
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    frame_path = os.path.join(save_root, f"frame_{ts}.jpg")
    cv2.imwrite(frame_path, frame)

    # Try vehicle detection (using yolov8n.pt). If this fails or finds nothing,
    # we'll fall back to full-frame ANPR.
    vehicles = detect_vehicles(frame)
    results = []

    # ---------- Fallback path: no vehicles ----------
    if not vehicles:
        try:
            outs = run_anpr_multi(
                frame_path,
                roi=None,
                save_dir=save_root,
                det_conf=0.25,
                ocr_min_conf=min_conf,
            )
        except Exception:
            outs = []

        for out in outs:
            plate_text = (out.get("plate_text") or "").strip()
            plate_conf = float(out.get("plate_conf") or 0.0)
            crop_path = out.get("crop_path")

            if USE_VLM_OCR and crop_path:
                vlm_text, vlm_conf = vlm_ocr_plate_from_path(crop_path)
                if vlm_text:
                    # Prefer the higher-confidence source
                    if vlm_conf >= plate_conf:
                        plate_text = vlm_text
                        plate_conf = float(vlm_conf)

            # Final filtering after possible VLM override
            if not plate_text or plate_conf < min_conf:
                continue

            results.append(
                {
                    "vehicle_bbox": None,                   
                    "vehicle_conf": 1.0,                   
                    "plate_text": plate_text,
                    "plate_conf": plate_conf,
                    "plate_bbox": out.get("plate_bbox"),        
                    "crop_path": crop_path,
                }
            )

        return results

    # ---------- Normal path: we have vehicle detections ----------
    for v in vehicles:
        roi = v["bbox"]
        try:
            anpr_out = run_anpr(frame_path, roi=roi, save_dir=save_root)
        except Exception:
            anpr_out = {
                "plate_text": "",
                "plate_conf": 0.0,
                "plate_bbox": None,
                "crop_path": None,
            }

        plate = (anpr_out.get("plate_text") or "").strip()
        conf = float(anpr_out.get("plate_conf") or 0.0)
        if not plate or conf < min_conf:
            continue

        results.append(
            {
                "vehicle_bbox": roi,
                "vehicle_conf": v["conf"],
                "plate_text": plate,
                "plate_conf": conf,
                "plate_bbox": anpr_out.get("plate_bbox"),
                "crop_path": anpr_out.get("crop_path"),
            }
        )

    return results

# ---- Tiny timing wrappers for VLM / ANPR / VLM_OCR ----

# Global timing store (seconds)
VLM_TIMINGS = {
    "vlm": 0.0,      # call_qwenvl_single_image
    "anpr": 0.0,     # run_anpr_on_frame
    "vlm_ocr": 0.0,  # vlm_ocr_plate_from_path (sum over all calls)
}

def reset_timing_counters():
    """Reset timing counters before each end-to-end run."""
    for k in VLM_TIMINGS:
        VLM_TIMINGS[k] = 0.0

def _wrap_with_timing(func, key):
    """Return a wrapped function that accumulates wall time in VLM_TIMINGS[key]."""
    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        out = func(*args, **kwargs)
        VLM_TIMINGS[key] += time.perf_counter() - start
        return out
    return wrapped

# Keep original refs in case you ever need them
_call_qwenvl_single_image = call_qwenvl_single_image
_run_anpr_on_frame = run_anpr_on_frame
_vlm_ocr_plate_from_path = vlm_ocr_plate_from_path

# Replace globals with timed versions everywhere in this file
call_qwenvl_single_image = _wrap_with_timing(_call_qwenvl_single_image, "vlm")
run_anpr_on_frame = _wrap_with_timing(_run_anpr_on_frame, "anpr")
vlm_ocr_plate_from_path = _wrap_with_timing(_vlm_ocr_plate_from_path, "vlm_ocr")


def _edit_distance(a: str, b: str) -> int:
    """
    Simple Levenshtein edit distance between two plate strings.
    Used to group near-identical plates across frames.
    """
    a = (a or "").upper().replace(" ", "")
    b = (b or "").upper().replace(" ", "")
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    cur = [0] * (lb + 1)
    for i in range(1, la + 1):
        cur[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cb = b[j - 1]
            cost = 0 if ca == cb else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, cur = cur, prev
    return prev[lb]


def _refine_plate_clusters(anpr_results, max_dist: int = 2):
    """
    Cluster plate strings that are within `max_dist` edits of each other
    and pick a stable representative per cluster.
    """
    raw = []
    for r in anpr_results:
        p = re.sub(r"[^A-Z0-9]", "", (r.get("plate") or r.get("plate_text") or "").upper())
        if len(p) < 6:
            continue
        raw.append(p)
    if not raw:
        return []

    clusters = []
    for p in raw:
        placed = False
        for cl in clusters:
            if _edit_distance(p, cl["rep"]) <= max_dist:
                cl["items"].append(p)
                placed = True
                break
        if not placed:
            clusters.append({"rep": p, "items": [p]})

    refined = []
    for cl in clusters:
        counts = Counter(cl["items"])
        best_p = None
        best_score = None
        for cand, cnt in counts.items():
            dsum = sum(_edit_distance(cand, other) for other in cl["items"])
            score = (-cnt, dsum, len(cand))
            if best_score is None or score < best_score:
                best_score = score
                best_p = cand
        refined.append(best_p)

    # Remove near-duplicates between clusters too
    unique = []
    for p in refined:
        if all(_edit_distance(p, q) > 1 for q in unique):
            unique.append(p)
    return unique


def _infer_region_from_plates(plates):
    """
    Use Indian state code from the first two letters (if present in INDIA_STATE_CODES)
    to guess region. Returns dict or None.
    """
    codes = []
    for p in plates:
        if len(p) < 2:
            continue
        code = p[:2].upper()
        if code in INDIA_STATE_CODES:
            codes.append(code)
    if not codes:
        return None
    code, count = Counter(codes).most_common(1)[0]
    conf = count / len(plates)
    return {
        "state_code": code,
        "country": "India",
        "confidence": conf,
    }


def run_anpr_on_video_frames(frames, frame_indices, min_conf: float = 0.40):
    """
    Run multi-plate ANPR on each sampled video frame.

    Returns a flat list of dicts:
      - frame_idx: index in the original video
      - plate: decoded plate text
      - conf: confidence
      - crop_path: saved crop path (if any)
    """
    here = BASE_DIR
    project_root = os.path.dirname(here)
    snapshot_dir = os.path.join(project_root, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    results = []

    for frame, vid_idx in zip(frames, frame_indices):
        frame_name = f"video_{vid_idx:06d}.jpg"
        frame_path = os.path.join(snapshot_dir, frame_name)
        cv2.imwrite(frame_path, frame)

        try:
            outs = run_anpr_multi(
                frame_path,
                roi=None,
                save_dir=snapshot_dir,
                det_conf=0.25,
                ocr_min_conf=min_conf,
            )
        except Exception:
            continue

        for out in outs:
            plate = (out.get("plate_text") or "").strip()
            conf = float(out.get("plate_conf") or 0.0)
            crop_path = out.get("crop_path")

            # Optional: VLM OCR on the crop for video frames too
            if USE_VLM_OCR and crop_path:
                vlm_text, vlm_conf = vlm_ocr_plate_from_path(crop_path)
                if vlm_text:
                    if vlm_conf >= conf:
                        plate = vlm_text
                        conf = float(vlm_conf)

            if not plate or conf < min_conf:
                continue

            results.append(
                {
                    "frame_idx": vid_idx,
                    "plate": plate,
                    "conf": conf,
                    "crop_path": crop_path,
                }
            )

    return results


def render_anpr_results(results, container):
    """
    Pretty-print ANPR results table inside a given Streamlit container.
    """
    if not results:
        container.info("No vehicles / plates detected by ANPR for this frame.")
        return

    rows = []
    for i, r in enumerate(results, start=1):
        rows.append(
            {
                "idx": i,
                "vehicle_bbox": r["vehicle_bbox"],
                "veh_conf": f"{r['vehicle_conf']:.2f}",
                "plate": r["plate_text"] or "",
                "plate_conf": f"{r['plate_conf']:.2f}",
            }
        )

    container.markdown("#### ANPR results (per detected vehicle)")
    container.table(rows)


def update_live_anpr_history(anpr_results, summary_container, min_conf: float = 0.40, max_history: int = 200):
    """
    Update session-level history of live ANPR hits and render a summary:
    - Unique plates across the session (approx, using clustering)
    - Region guess from Indian state codes, if available.
    """
    if "live_anpr_history" not in st.session_state:
        st.session_state["live_anpr_history"] = []

    history = st.session_state["live_anpr_history"]

    # Append new hits from this frame
    for r in anpr_results or []:
        plate = (r.get("plate_text") or "").strip()
        conf = float(r.get("plate_conf") or 0.0)
        if not plate or conf < min_conf:
            continue
        history.append({"plate": plate, "conf": conf})

    # Trim history to avoid unbounded growth
    if len(history) > max_history:
        history[:] = history[-max_history:]

    if not history:
        summary_container.info("Live ANPR summary: no plates decoded yet in this session.")
        return

    refined = _refine_plate_clusters(history)
    if not refined:
        summary_container.info(
            "Live ANPR summary: plates detected but none stable enough yet "
            "(too short or inconsistent across frames)."
        )
        return

    lines = []
    lines.append("#### Live ANPR summary (current session)")
    lines.append(f"- Unique plates seen (approx): {', '.join(refined)}")

    region = _infer_region_from_plates(refined)
    if region:
        lines.append(
            f"- Likely region: {region['country']} (state code {region['state_code']}), "
            f"confidence ≈ {region['confidence']:.2f}"
        )

    summary_container.markdown("\n".join(lines))

# ===== VLM history helpers (live + video) =====

def _shorten_for_history(text: str, max_chars: int = 600) -> str:
    """
    Compress a VLM answer to a short, single-line summary suitable for feeding back
    as history in later prompts. Keeps at most `max_chars` characters.
    """
    if not text:
        return ""
    # Collapse whitespace and strip
    import re as _re
    text = _re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    # Cut on word boundary if possible
    cut = text[:max_chars]
    last_space = cut.rfind(" ")
    if last_space > 40:  # avoid trimming too aggressively at the start
        cut = cut[:last_space]
    return cut + " ..."


def build_live_vlm_prompt(base_prompt: str, max_history_items: int = 3) -> str:
    """
    Build a history-aware prompt for the live camera.

    - On the FIRST frame, send the full BIG_TRAFFIC_PROMPT once,
      plus any extra instructions from `base_prompt`.
    - On later frames, don't repeat the big prompt. Instead, show a
      short history of previous summaries and treat `base_prompt`
      as a small add-on / refinement.
    """
    history = st.session_state.get("live_vlm_history", [])

    base_prompt = (base_prompt or "").strip()

    if not history:
        # First ever call for this session
        prefix = (
            "You are observing frames from a fixed CCTV traffic camera. "
            "This is the FIRST frame for this session.\n\n"
            "First, give a clear, structured description of the current scene "
            "using the detailed guidelines below.\n"
        )
        extra = ""
        if base_prompt:
            extra = "\n\nAdditional instructions from the operator:\n" + base_prompt

        return prefix + "\n" + default_prompt + extra

    # We already have some history – focus on changes / deltas
    recent = history[-max_history_items:]
    history_lines = [
        f"[update {len(history) - len(recent) + i + 1}] {h}"
        for i, h in enumerate(recent)
    ]
    history_text = "\n".join(history_lines)

    prefix = (
        "You are observing a sequence of frames from the SAME fixed CCTV traffic camera.\n"
        "Below is a brief history of your OWN previous summaries for earlier frames, "
        "from oldest to newest:\n"
        f"{history_text}\n\n"
        "Now look at the NEW frame and:\n"
        "- Describe the current state succinctly.\n"
        "- Explicitly mention what CHANGED since the previous updates "
        "(new vehicles/pedestrians entering/leaving, traffic density changes, "
        "any risky behaviour starting or stopping).\n"
        "- You can assume the reader already knows the basic layout of the scene "
        "from your first detailed summary, so you don't need to repeat everything.\n"
    )

    extra = ""
    if base_prompt:
        extra = "\n\nAdditional instructions for this update:\n" + base_prompt

    return prefix + extra

def append_live_vlm_history(answer: str, max_items: int = 10) -> None:
    """
    Append the latest VLM answer into st.session_state['live_vlm_history'].
    """
    short = _shorten_for_history(answer)
    history = st.session_state.get("live_vlm_history", [])
    history.append(short)
    st.session_state["live_vlm_history"] = history[-max_items:]


def build_video_vlm_prompt(
    clip_prompt: str,
    user_prompt: str,
    video_key: str,
    max_history_items: int = 3,
) -> str:
    """
    Build a history-aware prompt for a given uploaded video.

    - For the FIRST call for a given video_key, send BIG_TRAFFIC_PROMPT once
      + clip-level instructions + the current user prompt.
    - For later calls, don't repeat BIG_TRAFFIC_PROMPT. Use a short history of
      previous answers and treat user_prompt as an incremental / refinement query.
    """
    hist_dict = st.session_state.get("video_vlm_history", {})
    history = hist_dict.get(video_key, [])

    user_prompt = (user_prompt or "").strip()
    clip_prompt = (clip_prompt or "").strip()

    if not history:
        # First question about this clip in this session
        base = default_prompt
        if clip_prompt:
            base += "\n\nClip-level instructions:\n" + clip_prompt
        if user_prompt:
            base += "\n\nUser prompt:\n" + user_prompt
        return base

    # There is history for this video: focus on consistency and refinement
    recent = history[-max_history_items:]
    history_lines = [f"[prev {i+1}] {h}" for i, h in enumerate(recent)]
    history_text = "\n".join(history_lines)

    prefix = (
        "You are analysing the SAME CCTV traffic video clip as in earlier questions.\n"
        "Here are your previous summaries / answers for this clip (oldest to newest):\n"
        f"{history_text}\n\n"
        "Use these as context and stay consistent when answering the NEW user prompt below. "
        "Extend or refine your earlier answers instead of contradicting them, "
        "unless you clearly correct a previous mistake.\n"
    )

    base = ""
    if clip_prompt:
        base += "Clip-level instructions:\n" + clip_prompt + "\n\n"

    base += prefix
    if user_prompt:
        base += "\nUser prompt:\n" + user_prompt

    return base



def append_video_vlm_history(video_key: str, answer: str, max_items: int = 5) -> None:
    """
    Append the latest answer for a video into st.session_state['video_vlm_history'][video_key].
    """
    short = _shorten_for_history(answer)
    hist_dict = st.session_state.get("video_vlm_history", {})
    history = hist_dict.get(video_key, [])
    history.append(short)
    hist_dict[video_key] = history[-max_items:]
    st.session_state["video_vlm_history"] = hist_dict

# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="Traffic VLM Dashboard",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Traffic VLM Dashboard – LLaVA / Qwen2.5-VL")
st.caption("Image / Video / Live Cam testing + Dataset building for traffic scenarios.")


# Sidebar
st.sidebar.header("Backend & Connection")

backend_choice = st.sidebar.radio(
    "Vision backend",
    ["Ollama LLaVA (llava:13b)", "Qwen2.5-VL + LoRA"],
    index=0 if DEFAULT_BACKEND == "ollama_llava" else 1,
)

use_qwen = backend_choice.startswith("Qwen2.5")
VLM_BACKEND = "qwenvl" if use_qwen else "ollama_llava"

if not use_qwen:
    st.sidebar.markdown(f"**Model:** `{MODEL_NAME}`")
    ollama_host = st.sidebar.text_input("Ollama URL", value=OLLAMA_URL)
    if ollama_host != OLLAMA_URL:
        OLLAMA_URL = ollama_host

    st.sidebar.info(
        "This mode runs Qwen2.5-VL-7B (optionally with LoRA) directly in Python using 4-bit "
        "quantization. Your 16 GB GPU should comfortably handle this."
    )
else:
    st.sidebar.markdown(f"**Qwen base model:** `{QWEN_MODEL_ID}`")
    QWEN_LORA_DIR = st.sidebar.text_input(
        "LoRA directory",
        value=QWEN_LORA_DIR,
        help="Folder where `train_lora_qwenvl.py` saved adapters "
             "(e.g. ./outputs/qwenvl_idd_lora_50).",
    )
    st.sidebar.info(
        "Make sure Ollama is running and `llava:13b` is pulled.\n\n"
        "In PowerShell: `ollama pull llava:13b` then (optionally) "
        "`ollama run llava:13b` once to sanity-check."
    )


tab_playground, tab_live, tab_search, tab_dataset = st.tabs(
    [
        "🧪 Playground (Image / Video)",
        "📡 Live Camera",
        "🔍 Search / Retrieval",
        "📚 Dataset & Training",
    ]
)
# =========================
# TAB 1 – Playground
# =========================

with tab_playground:
    st.subheader("🧪 Playground – Test VLM on uploads")

    mode = st.radio(
        "Select input type:",
        ["Image", "Video"],
        horizontal=True,
    )

    default_prompt = (
        "You are a traffic-analysis assistant. Use the following structure in your answer:\n"
        "1. GLOBAL SCENE OVERVIEW (location type, camera angle, surroundings)\n"
        "2. ROAD GEOMETRY & LANES (lane count per direction, divider, crossings, markings)\n"
        "3. TRAFFIC COMPOSITION & COUNTS (per vehicle type + density)\n"
        "4. PEDESTRIANS & OTHER ROAD USERS (counts + positions + behaviour)\n"
        "5. TIME / LIGHTING / WEATHER (time of day, visibility, wet/dry roads)\n"
        "6. INFRASTRUCTURE & SIGNAGE (signals, boards, barriers, speed breakers, etc.)\n"
        "7. BEHAVIOUR & SAFETY ANALYSIS (lane discipline, conflicts, risks)\n"
        "8. LICENSE PLATES (only mention that [license plate visible] or not, do NOT write numbers)\n"
        "9. 3–5 BULLET SUMMARY of key facts and risks.\n\n"
        "Write a detailed answer, but stay factual and avoid repetition."
    )

    prompt = st.text_area(
        "Prompt (instruction to VLM)",
        value=default_prompt,
        height=180,
    )

    # --------------------------------
    # IMAGE BRANCH
    # --------------------------------
    if mode == "Image":
        uploaded_img = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
        )

        col1, col2 = st.columns([1, 1])

        if uploaded_img is not None:
            img = Image.open(uploaded_img).convert("RGB")
            with col1:
                st.image(img, caption="Uploaded Image", width='stretch')

        if st.button("Run VLM on Image", type="primary"):
            with st.spinner("Calling VLM backend + ANPR..."):
                try:
                    if use_qwen:
                        get_qwenvl_lora()

                    reset_timing_counters()

                    # VLM
                    if use_qwen:

                        answer = call_qwenvl_single_image(
                            img,
                            prompt,
                            max_new_tokens=BIG_MAX_NEW_TOKENS,
                        )
                    else:
                        b64 = pil_to_base64_jpeg(img)
                        answer = call_llava_single_image(b64, prompt)

                    # ANPR on all detected vehicles
                    frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    anpr_results = run_anpr_on_frame(frame_bgr)

                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                else:
                    with col2:
                        st.markdown("### VLM Output")
                        st.write(answer)

                        # timing line
                        st.caption(
                            f"VLM: {VLM_TIMINGS['vlm']:.2f} s, "
                            f"ANPR: {VLM_TIMINGS['anpr']:.2f} s, "
                            f"VLM_OCR: {VLM_TIMINGS['vlm_ocr']:.2f} s"
                        )

                        anpr_container = st.container()
                        render_anpr_results(anpr_results, anpr_container)

                        # Cache last analyzed image + outputs for retrieval DB indexing
                        try:
                            st.session_state["pg_last_img_bytes"] = uploaded_img.getvalue()
                        except Exception:
                            buf = io.BytesIO()
                            img.save(buf, format="JPEG")
                            st.session_state["pg_last_img_bytes"] = buf.getvalue()

                        st.session_state["pg_last_answer"] = answer
                        st.session_state["pg_last_plates"] = [
                            (r.get("plate_text") or "").strip()
                            for r in (anpr_results or [])
                            if r.get("plate_text")
                        ]
        else:
            st.info("Upload an image to begin.")

        # ---- Index last analyzed image into retrieval DB ----
        st.markdown("---")
        st.markdown("#### ➕ Index last analyzed image into retrieval DB")

        if "pg_last_img_bytes" not in st.session_state:
            st.info("Run VLM on an image first; then you can index it into the retrieval DB.")
        else:
            if st.button("Add last analyzed image to retrieval DB", key="add_image_to_retrieval_db"):
                try:
                    img_bytes = st.session_state["pg_last_img_bytes"]
                    img_for_db = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                    caption = (st.session_state.get("pg_last_answer") or "").strip()
                    if not caption:
                        caption = "Traffic scene"

                    plates = st.session_state.get("pg_last_plates", [])

                    db = get_retrieval_db()
                    item_id = db.add_image_from_pil(
                        img_for_db,
                        caption=caption,
                        plates=plates,
                        source="playground_image",
                        extra={"from": "playground_image"},
                    )
                    st.success(f"Image added to retrieval DB as `{item_id}`.")
                except Exception as e:
                    st.error(f"Failed to add image to retrieval DB: {e}")

    # --------------------------------
    # VIDEO BRANCH
    # --------------------------------
    else:  # Video
        uploaded_vid = st.file_uploader(
            "Upload a video",
            type=["mp4", "avi", "mov", "mkv"],
        )

        if uploaded_vid is not None:
            vid_key = f"{uploaded_vid.name}_{uploaded_vid.size}"
            if st.button("Reset VLM history for this video"):
                if "video_vlm_history" in st.session_state:
                    st.session_state["video_vlm_history"].pop(vid_key, None)
                st.success("Cleared VLM history for this video in this session.")

        num_frames = st.slider(
            "Number of frames to sample from video",
            min_value=3,
            max_value=12,
            value=6,
            step=1,
        )

        col1, col2 = st.columns([1, 1])

        if uploaded_vid is not None:
            with col1:
                st.video(uploaded_vid)

        if st.button("Run VLM on Video", type="primary"):
            if uploaded_vid is None:
                st.error("Upload a video first.")
            else:
                with st.spinner("Sampling frames and calling VLM backend..."):
                    try:
                        file_bytes = uploaded_vid.getvalue()
                        frames, indices = sample_video_frames_from_file(
                            file_bytes, num_frames=num_frames
                        )

                        video_prompt = ("Describe what happens across these frames.")
                        video_key = f"{uploaded_vid.name}_{uploaded_vid.size}"
                        full_prompt = build_video_vlm_prompt(
                            video_prompt,
                            prompt,
                            video_key=video_key,
                        )

                        if use_qwen:
                            answer = call_qwenvl_multi_frames(
                                frames,
                                full_prompt,
                                max_new_tokens=FAST_MAX_NEW_TOKENS,
                            )

                        else:
                            b64_list = [frame_to_base64_jpeg(f) for f in frames]
                            answer = call_llava_multi_images(b64_list, full_prompt)
                        append_video_vlm_history(video_key, answer)

                    except Exception as e:
                        st.error(f"VLM video analysis failed: {e}")
                    else:
                        # Then, independently, run ANPR across the same sampled frames
                        video_anpr_results = []
                        anpr_error = None
                        try:
                            video_anpr_results = run_anpr_on_video_frames(
                                frames, indices, min_conf=0.40
                            )
                        except Exception as anpr_exc:
                            anpr_error = f"ANPR on video frames failed: {anpr_exc}"

                        with col2:
                            st.markdown(f"### Sampled Frames ({len(frames)} total)")
                            for frame, idx in zip(frames, indices):
                                st.image(
                                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                    caption=f"Frame index {idx}",
                                    width='stretch',
                                )

                            st.markdown("### VLM Video Summary")
                            st.write(answer)

                            if anpr_error:
                                st.warning(anpr_error)
                            else:
                                if video_anpr_results:
                                    st.markdown("### ANPR across sampled frames")

                                    rows = [
                                        {
                                            "frame_idx": r["frame_idx"],
                                            "plate": r["plate"],
                                            "conf": f"{r['conf']:.2f}",
                                        }
                                        for r in video_anpr_results
                                    ]
                                    st.table(rows)

                                    refined_plates = _refine_plate_clusters(video_anpr_results)
                                    if refined_plates:
                                        st.markdown("#### Refined unique plates across video")
                                        st.write(", ".join(refined_plates))

                                        region = _infer_region_from_plates(refined_plates)
                                        if region:
                                            st.markdown(
                                                f"Likely region from plates: "
                                                f"{region['country']} (state code {region['state_code']}), "
                                                f"confidence ≈ {region['confidence']:.2f}"
                                            )
                                else:
                                    st.info(
                                        "No plates decoded on the sampled frames "
                                        "(or all below confidence threshold)."
                                    )

                        # Cache last analyzed video + outputs for retrieval DB indexing
                        st.session_state["pg_last_video_bytes"] = file_bytes
                        st.session_state["pg_last_video_name"] = getattr(
                            uploaded_vid, "name", "uploaded_video.mp4"
                        )
                        st.session_state["pg_last_video_answer"] = answer
                        st.session_state["pg_last_video_plates"] = [
                            (r.get("plate") or "").strip()
                            for r in (video_anpr_results or [])
                            if r.get("plate")
                        ]

        else:
            st.info("Upload a video to test VLM on multiple frames.")

        # ---- Index last analyzed video into retrieval DB ----
        st.markdown("---")
        st.markdown("#### ➕ Index last analyzed video into retrieval DB")

        if "pg_last_video_bytes" not in st.session_state:
            st.info("Run VLM on a video first; then you can index it into the retrieval DB.")
        else:
            if st.button("Add last analyzed video to retrieval DB", key="add_video_to_retrieval_db"):
                try:
                    db = get_retrieval_db()
                    item_id = db.add_video_bytes(
                        st.session_state["pg_last_video_bytes"],
                        orig_name=st.session_state.get(
                            "pg_last_video_name", "uploaded_video.mp4"
                        ),
                        caption=(
                            st.session_state.get("pg_last_video_answer", "").strip()
                            or "Traffic video"
                        ),
                        plates=st.session_state.get("pg_last_video_plates", []),
                        source="playground_video",
                        extra={"from": "playground_video"},
                    )
                    st.success(f"Video added to retrieval DB as `{item_id}`.")
                except Exception as e:
                    st.error(f"Failed to add video to retrieval DB: {e}")

# =========================
# TAB 2 – Live Camera
# =========================

with tab_live:
    st.subheader("📡 Live Camera – Axis / RTSP + VLM")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        rtsp_url = st.text_input(
            "RTSP URL",
            value=DEFAULT_RTSP,
            help="Axis example: rtsp://root:2024@192.168.1.241/axis-media/media.amp?streamprofile=stream1",
        )

        live_prompt_default = (
            "You are a scene-analysis assistant observing frames from a fixed CCTV camera. "
            "Provide a detailed description of the current scene, including layout, "  
            "vehicle and pedestrian activity, time of day, weather conditions, and any notable events."
        )

        live_prompt = st.text_area(
            "Prompt for live frame analysis",
            value=live_prompt_default,
            height=200,
        )

        interval = st.number_input(
            "Interval between analyses (seconds)",
            min_value=1.0,
            max_value=60.0,
            value=10.0,
            step=1.0,
        )

        max_updates = st.number_input(
            "Number of updates in continuous mode",
            min_value=1,
            max_value=1000,
            value=20,
            step=1,
            help="How many times to capture + analyze before stopping.",
        )

        single_btn = st.button("Capture & Analyze single frame", type="secondary")
        continuous_btn = st.button("Start continuous analysis", type="primary")

    with col_r:
        frame_placeholder = st.empty()
        result_placeholder = st.empty()
        anpr_placeholder = st.empty()
        summary_placeholder = st.empty()
        status_placeholder = st.empty()

    # --- Single shot ---
    if single_btn:
        with st.spinner("Capturing frame and calling VLM + ANPR..."):
            try:
                frame = capture_rtsp_frame(rtsp_url)
                if use_qwen:
                    get_qwenvl_lora()

                reset_timing_counters()

                # VLM (history-aware)
                if use_qwen:
                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    full_live_prompt = build_live_vlm_prompt(live_prompt)
                    answer = call_qwenvl_single_image(
                        img_pil,
                        full_live_prompt,
                        max_new_tokens=LIVE_MAX_NEW_TOKENS,
                    )


                else:
                    b64 = frame_to_base64_jpeg(frame)
                    full_live_prompt = build_live_vlm_prompt(live_prompt)
                    answer = call_llava_single_image(b64, full_live_prompt)

                # ANPR on all detected vehicles in this frame
                anpr_results = run_anpr_on_frame(frame)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    frame_rgb, caption="Live Frame from RTSP", width='stretch'
                )
                result_placeholder.markdown("### VLM Live Analysis (single shot)")
                result_placeholder.write(answer)
                append_live_vlm_history(answer)
                render_anpr_results(anpr_results, anpr_placeholder)
                update_live_anpr_history(anpr_results, summary_placeholder)
                status_placeholder.info(
                    f"Last update: single capture  |  "
                    f"VLM: {VLM_TIMINGS['vlm']:.2f} s, "
                    f"ANPR: {VLM_TIMINGS['anpr']:.2f} s, "
                    f"VLM_OCR: {VLM_TIMINGS['vlm_ocr']:.2f} s"
                )

            except Exception as e:
                st.error(f"Live capture or analysis failed: {e}")

    # --- Continuous mode ---
    if continuous_btn:
        status_placeholder.info(
            f"Starting continuous analysis: every {interval:.1f}s for {max_updates} updates. "
            "This will run inside the app – to stop early, interrupt the app or reload the page."
        )
        if use_qwen:
            get_qwenvl_lora()
        for i in range(int(max_updates)):
            try:
                frame = capture_rtsp_frame(rtsp_url)
                reset_timing_counters()
                # VLM (history-aware)
                if use_qwen:
                    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    full_live_prompt = build_live_vlm_prompt(live_prompt)
                    answer = call_qwenvl_single_image(
                        img_pil,
                        full_live_prompt,
                        max_new_tokens=LIVE_MAX_NEW_TOKENS,
                    )

                else:
                    b64 = frame_to_base64_jpeg(frame)
                    full_live_prompt = build_live_vlm_prompt(live_prompt)
                    answer = call_llava_single_image(b64, full_live_prompt)

                # ANPR on all detected vehicles
                anpr_results = run_anpr_on_frame(frame)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    frame_rgb,
                    caption=f"Live Frame #{i+1} from RTSP",
                    width='stretch',
                )
                result_placeholder.markdown(
                    f"### VLM Live Analysis (update {i+1}/{int(max_updates)})"
                )
                result_placeholder.write(answer)
                append_live_vlm_history(answer)
                render_anpr_results(anpr_results, anpr_placeholder)
                update_live_anpr_history(anpr_results, summary_placeholder)

                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                status_placeholder.success(
                    f"Last update #{i+1} at {ts} – next in {interval:.1f}s | "
                    f"VLM: {VLM_TIMINGS['vlm']:.2f} s, "
                    f"ANPR: {VLM_TIMINGS['anpr']:.2f} s, "
                    f"VLM_OCR: {VLM_TIMINGS['vlm_ocr']:.2f} s"
                )

            except Exception as e:
                status_placeholder.error(f"Update #{i+1} failed: {e}")

            time.sleep(float(interval))

        if st.button("Reset live VLM history for this session"):
            st.session_state["live_vlm_history"] = []
            st.info("Live VLM history cleared.")

# =========================
# TAB 3 – Search / Retrieval
# =========================

def _highlight_query_in_text(text: str, query: str) -> str:
    """
    Simple case-insensitive highlighter for search query inside a caption.
    Wraps matches in ** ** so they appear bold in the Streamlit markdown.
    """
    if not text or not query:
        return text

    import re as _re
    pattern = _re.compile(_re.escape(query), _re.IGNORECASE)

    def repl(match):
        return f"**{match.group(0)}**"

    return pattern.sub(repl, text)

with tab_search:
    st.subheader("🔍 Search stored images / videos")

    # -- Current DB stats --
    db = get_retrieval_db()
    stats = db.stats()
    db_rel = os.path.relpath(stats["db_path"], PROJECT_ROOT)

    st.markdown(
        f"**Indexed items:** {stats['total_items']} "
        f"(images: {stats['num_images']}, videos: {stats['num_videos']})  \n"
        f"DB file: `{db_rel}`"
    )

    # -- Debugging / export: inspect embedding matrix directly --
    with st.expander("🧬 Debug: Embeddings info / export", expanded=False):
        if db._X is None or not getattr(db, "_id_list", None):
            st.info("No embeddings loaded yet. Build the retrieval DB first.")
        else:
            st.write(f"Embedding matrix shape: {db._X.shape}")
            st.write(f"First 3 item IDs: {db._id_list[:3]}")

            # Show sample texts used for embeddings
            try:
                sample_texts = getattr(db, "_texts", [])[:3]
                if sample_texts:
                    st.write("Sample texts used for embeddings:")
                    for t in sample_texts:
                        # Truncate long captions so UI doesn't explode
                        snippet = t[:400] + ("..." if len(t) > 400 else "")
                        st.code(snippet, language="text")
            except Exception:
                pass

            # Prepare downloads: .npy for embeddings, .json for ids
            import io as _io

            emb_buf = _io.BytesIO()
            np.save(emb_buf, db._X.astype("float32"))
            emb_buf.seek(0)

            ids_json_bytes = json.dumps(
                db._id_list, indent=2, ensure_ascii=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download embeddings (.npy)",
                data=emb_buf,
                file_name="retrieval_embeddings.npy",
                mime="application/octet-stream",
            )

            st.download_button(
                "⬇️ Download ID list (ids.json)",
                data=ids_json_bytes,
                file_name="retrieval_ids.json",
                mime="application/json",
            )

    snapshots_dir = os.path.join(DATA_DIR, "snapshots_retrieval")
    os.makedirs(snapshots_dir, exist_ok=True)

    # -- Admin tools: build / refresh DB from captions.jsonl --
    with st.expander("⚙️ Admin: Build / refresh retrieval DB from captions JSONL", expanded=False):
        default_captions = os.path.join(DATA_DIR, "traffic_captions.jsonl")
        captions_path_input = st.text_input(
            "Captions JSONL path",
            value=default_captions,
            help="JSONL produced by auto_caption_traffic_folder.py (relative or absolute path).",
        )

        limit_rows = st.number_input(
            "Limit number of rows (0 = use all rows in JSONL)",
            min_value=0,
            max_value=1_000_000,
            value=0,
            step=1,
        )

        skip_anpr_flag = st.checkbox(
            "Skip ANPR while building (only use captions for search)",
            value=False,
            help="Enable this if ANPR keeps failing or you only care about semantic description search.",
        )

    # Optional helper: auto-caption a folder into the JSONL above
    images_dir_default = os.path.join(DATA_DIR, "traffic_images")
    images_dir_input = st.text_input(
        "Images folder to auto-caption (optional helper)",
        value=images_dir_default,
        help=(
            "All images under this folder will be captioned and appended to the "
            "JSONL path above using auto_caption_traffic_folder.py."
        ),
    )

    max_caption_images = st.number_input(
        "Max images to caption in this run (0 = all found)",
        min_value=0,
        max_value=1_000_000,
        value=0,
        step=1,
    )

    if st.button("🧾 Auto-caption folder using LLaVA (Ollama)"):
        imgs_dir = images_dir_input.strip()
        out_jsonl = captions_path_input.strip()

        if not imgs_dir:
            st.error("Please specify an images folder to caption.")
        elif not out_jsonl:
            st.error("Please specify a captions JSONL path (above).")
        else:
            try:
                # Build argv for the CLI-style script
                old_argv = sys.argv
                argv = [
                    "auto_caption_traffic_folder",
                    "--images_dir", imgs_dir,
                    "--out", out_jsonl,
                    "--ollama_url", OLLAMA_URL,
                    "--model_name", MODEL_NAME,
                ]
                if max_caption_images and int(max_caption_images) > 0:
                    argv += ["--max_images", str(int(max_caption_images))]

                sys.argv = argv
                with st.spinner(
                    f"Auto-captioning images under {imgs_dir} "
                    f"into {out_jsonl} using {MODEL_NAME} via Ollama..."
                ):
                    auto_caption_main()

                st.success(
                    f"Auto-captioning completed. Captions written to: {out_jsonl}"
                )
            except Exception as e:
                st.error(f"Auto-captioning failed: {e}")
            finally:
                sys.argv = old_argv

    # Clear retrieval DB so you can rebuild from a clean slate
    col_clear, col_build = st.columns(2)

    with col_clear:
        clear_db_clicked = st.button("🧹 Clear retrieval DB (delete all items)")

    with col_build:
        build_db_clicked = st.button("🚧 Build / Rebuild retrieval DB from captions JSONL")

    if clear_db_clicked:
        try:
            with st.spinner("Clearing retrieval DB (JSON file and in-memory index)..."):
                db_clear = get_retrieval_db()
                db_clear.items = {}
                db_clear._rebuild_matrix()
                db_clear._save()
                reset_retrieval_db()
                db_after = get_retrieval_db()
                stats_after = db_after.stats()

            st.success(
                "Retrieval DB cleared. "
                f"Current items: {stats_after['total_items']} "
                f"(images: {stats_after['num_images']}, videos: {stats_after['num_videos']})."
            )
        except Exception as e:
            st.error(f"Clearing retrieval DB failed: {e}")

    if build_db_clicked:
        captions_path = captions_path_input.strip()
        if not captions_path:
            st.error("Please specify a captions JSONL path.")
        else:
            try:
                with st.spinner(
                    "Building retrieval DB from captions... "
                    "This might take a while for large JSONL files."
                ):
                    stats_build = build_retrieval_db_from_jsonl(
                        captions_jsonl=captions_path,
                        limit=int(limit_rows),
                        snapshots_dir=snapshots_dir,
                        skip_anpr=skip_anpr_flag,  # ANPR will run when this is False
                    )

                    reset_retrieval_db()
                    db = get_retrieval_db()
                    stats = db.stats()

                st.success(
                    f"Retrieval DB built from JSONL.\n\n"
                    f"- Rows considered: {stats_build['total_rows']}\n"
                    f"- Added: {stats_build['added']}\n"
                    f"- Skipped: {stats_build['skipped']}\n"
                    f"- Failed: {stats_build['failed']}\n\n"
                    f"Current DB items: {stats['total_items']} "
                    f"(images: {stats['num_images']}, videos: {stats['num_videos']})."
                )
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Building retrieval DB failed: {e}")

    # -- Normal search UI below --
    mode = st.radio(
        "Search mode",
        ["Semantic description", "Number plate"],
        horizontal=True,
    )

    placeholder = {
        "Semantic description": "e.g. man wearing a blue shirt near a blue car",
        "Number plate": "e.g. MH12AB1234 or KA09",
    }[mode]

    query = st.text_input("Search query", placeholder=placeholder)

    col_a, col_b = st.columns(2)

    with col_a:
        media_types = st.multiselect(
            "Restrict media types",
            options=["image", "video"],
            default=["image", "video"],
            help="Filter results to only images, only videos, or both.",
        )

    with col_b:
        top_k = st.slider("Max results", min_value=1, max_value=20, value=6)
        min_sim_pct = 0
        if mode == "Semantic description":
            min_sim_pct = st.slider(
                "Minimum similarity (%)",
                min_value=0,
                max_value=100,
                value=10,
                step=5,
            )

    if st.button("Search", type="primary"):
        if not query.strip():
            st.warning("Please enter a search query first.")
        else:
            results = []
            try:
                with st.spinner("Searching retrieval DB..."):
                    if mode == "Semantic description":
                        results = db.search_semantic(
                            query.strip(),
                            top_k=top_k,
                            min_score=(min_sim_pct or 0) / 100.0,
                            media_types=media_types or None,
                        )
                    else:
                        results = db.search_plate(
                            query.strip(),
                            top_k=top_k,
                            media_types=media_types or None,
                        )
            except Exception as e:
                st.error(f"Search failed: {e}")
                results = []

            if not results:
                st.info(
                    "No matching items found. "
                    "Make sure you've built retrieval_db.json using the admin panel above "
                    "or the offline script."
                )
            else:
                # Show the query once above all results
                norm_query = query.strip()
                st.markdown(f"**Search query:** `{norm_query}`")

                for idx, item in enumerate(results, start=1):
                    score = item.get("score")
                    media_type = item.get("type", "?")
                    header = f"Result #{idx} – {media_type}"
                    if score is not None:
                        header += f" (score: {score:.1f}%)"
                    st.markdown(f"#### {header}")

                    path = item.get("file_path")
                    caption = item.get("caption", "")
                    plates = item.get("plates") or []
                    created_at = item.get("created_at", "")

                    meta_col, preview_col = st.columns([2, 1])

                    with meta_col:
                        if path:
                            st.caption(os.path.basename(path))

                        if caption:
                            # Highlight query inside the retrieved caption (semantic mode especially)
                            highlighted = _highlight_query_in_text(caption, norm_query)
                            st.markdown(highlighted)

                        if plates:
                            st.markdown("**Plates:** " + ", ".join(plates))
                        if created_at:
                            st.caption(f"Indexed at: {created_at}")

                    with preview_col:
                        if path and os.path.exists(path):
                            if media_type == "image":
                                st.image(path, width='stretch')
                            elif media_type == "video":
                                st.video(path)
                        else:
                            st.warning("Media file missing on disk.")

                st.caption(
                    "Tip: first caption your image folder using auto_caption_traffic_folder.py, "
                    "then use the admin panel above to build the retrieval DB from that JSONL."
                )


# =========================
# TAB 4 – Dataset & Training
# =========================

with tab_dataset:
    st.subheader("📚 Dataset & Training – Build traffic visual-instruction data")

    # ---- Manage session state for the dataset answer (auto-labelling) ----
    # This must happen BEFORE we create the text_area with key="ds_answer_input"
    if "ds_answer_input" not in st.session_state:
        st.session_state["ds_answer_input"] = ""

    if "ds_answer_pending" not in st.session_state:
        st.session_state["ds_answer_pending"] = None

    # If we have a pending draft from auto-labelling, apply it now,
    # before the widget is instantiated in this rerun.
    if st.session_state["ds_answer_pending"]:
        st.session_state["ds_answer_input"] = st.session_state["ds_answer_pending"]
        st.session_state["ds_answer_pending"] = None

    st.markdown(
        "This section helps you build a **LLaVA/Qwen-style visual instruction dataset** "
        "for traffic scenarios. It stores images and JSONL records locally so you can "
        "later fine-tune models."
    )

    st.markdown("#### ➕ Add a new dataset example")

    col_a, col_b = st.columns([1, 1])

    with col_a:
        ds_img_file = st.file_uploader(
            "Upload a traffic frame for the dataset",
            type=["jpg", "jpeg", "png"],
            key="ds_img_uploader",
        )

    with col_b:
        tag = st.text_input(
            "Optional tag (e.g., 'day_junction', 'night_highway', 'rainy', etc.)",
            value="",
        )

        ds_instruction = st.text_area(
            "Instruction (what you would ask the model to do for this frame)",
            value=(
                "Describe this traffic scene, count vehicles and pedestrians, and read any clearly visible license plates. "
                "Mention lane structure and surroundings."
            ),
            height=120,
        )

        # The actual text area bound to session_state["ds_answer_input"]
        ds_answer = st.text_area(
            "Ideal answer (what a perfect model should say)",
            key="ds_answer_input",
            height=150,
            help=(
                "Write the ground-truth answer you want the fine-tuned model to produce. "
                "You can also auto-generate a first draft with the selected VLM and then edit it."
            ),
        )

        auto_label_btn = st.button("Auto-fill answer using selected VLM", type="secondary")
        add_example_btn = st.button("Save example to dataset", type="primary")

    if auto_label_btn:
        if ds_img_file is None:
            st.error("Please upload an image first for auto-labelling.")
        elif not ds_instruction.strip():
            st.error("Instruction cannot be empty when auto-labelling.")
        else:
            try:
                img = Image.open(ds_img_file).convert("RGB")
                auto_prompt = ds_instruction.strip()
                with st.spinner("Calling selected VLM backend to draft answer..."):
                    if use_qwen:
                        draft = call_qwenvl_single_image(
                            img,
                            auto_prompt,
                            max_new_tokens=DATASET_MAX_NEW_TOKENS,
                        )

                    else:
                        b64 = pil_to_base64_jpeg(img)
                        draft = call_llava_single_image(b64, auto_prompt)

                # IMPORTANT: do NOT write directly to ds_answer_input here.
                # Instead, stash it in ds_answer_pending and apply it on the next rerun
                # *before* the text_area is created.
                st.session_state["ds_answer_pending"] = draft

                st.success(
                    "Draft answer generated. It has been loaded into the text box above. "
                    "Review/edit it, then click 'Save example to dataset'."
                )
            except Exception as e:
                st.error(f"Auto-labelling failed: {e}")

    if add_example_btn:
        if ds_img_file is None:
            st.error("Please upload an image first.")
        elif not ds_instruction.strip() or not ds_answer.strip():
            st.error("Instruction and ideal answer cannot be empty.")
        else:
            try:
                img = Image.open(ds_img_file).convert("RGB")
                rel_path = save_dataset_example(
                    img, ds_instruction.strip(), ds_answer.strip(), tag.strip() or None
                )
                st.success(f"Saved example. Image stored as: {rel_path}")
            except Exception as e:
                st.error(f"Failed to save dataset example: {e}")

    st.markdown("---")
    st.markdown("#### 📂 Dataset preview")

    rows, total = load_dataset_preview(max_rows=20)
    st.write(f"**Dataset file:** `{os.path.relpath(DATASET_PATH, BASE_DIR)}`")
    st.write(f"**Total examples:** {total}")

    if rows:
        for i, row in enumerate(rows):
            st.markdown(f"**Example #{i+1}** — image: `{row.get('image', '?')}`")
            conv = row.get("conversations", [])
            tag_val = row.get("tag", None)
            if tag_val:
                st.write(f"Tag: `{tag_val}`")
            if len(conv) >= 2:
                st.write("**Instruction:**")
                st.code(conv[0].get("value", ""), language="text")
                st.write("**Ideal answer:**")
                st.code(conv[1].get("value", ""), language="text")
            st.markdown("---")
    else:
        st.info("No examples yet. Add one above to start building the dataset.")

    st.markdown("#### 🧪 Quick manual evaluation helper")

    st.markdown(
        "Use this to manually compare the **selected backend** vs your **ideal answer** "
        "for any image, without training yet."
    )

    eval_img_file = st.file_uploader(
        "Upload an image for evaluation",
        type=["jpg", "jpeg", "png"],
        key="eval_img_uploader",
    )
    eval_instruction = st.text_area(
        "Instruction for evaluation",
        value="Describe this traffic scene, count vehicles and pedestrians, and read any clearly visible license plates.",
        height=100,
    )
    eval_ideal = st.text_area(
        "Optional: paste an ideal answer (for comparison only)",
        value="",
        height=100,
    )

    if st.button("Run selected VLM on evaluation image"):
        if eval_img_file is None:
            st.error("Upload an image for evaluation.")
        else:
            try:
                img = Image.open(eval_img_file).convert("RGB")
                with st.spinner("Calling VLM backend..."):
                    if use_qwen:
                        answer = call_qwenvl_single_image(
                            img,
                            eval_instruction.strip(),
                            max_new_tokens=DATASET_MAX_NEW_TOKENS,
                        )


                    else:
                        b64 = pil_to_base64_jpeg(img)
                        answer = call_llava_single_image(b64, eval_instruction.strip())

                st.markdown("##### Uploaded Image")
                st.image(img, width='stretch')

                st.markdown("##### VLM Answer")
                st.write(answer)

                if eval_ideal.strip():
                    st.markdown("##### Your Ideal Answer (for comparison)")
                    st.write(eval_ideal.strip())

            except Exception as e:
                st.error(f"Evaluation failed: {e}")

    # =========================
    # Auto-caption + training helper
    # =========================

    st.markdown("---")
    st.markdown("#### 🚀 Auto-caption + training helper (LLaVA captions → LoRA)")

    # NOTE: by default, on your setup:
    # - train_images_dir = D:\vlm\vlm-local\data\traffic_images
    # - train_jsonl_path = D:\vlm\vlm-local\data\traffic_captions.jsonl
    train_images_dir = st.text_input(
        "Images folder for training (used by auto-caption step)",
        value=IMAGES_DIR,
        help="All images under this folder will be captioned by LLaVA and used as training data.",
    )

    default_train_jsonl = os.path.join(DATA_DIR, "traffic_captions.jsonl")
    train_jsonl_path = st.text_input(
        "Captions JSONL for training",
        value=default_train_jsonl,
        help="JSONL with `image`, `prompt`, `answer` fields (from auto_caption_traffic_folder.py).",
    )

    col_cap, col_train = st.columns(2)

    # --- Step 1: auto-caption with LLaVA via Ollama ---
    with col_cap:
        if st.button("🧾 Auto-caption dataset with LLaVA (Ollama)", key="btn_ds_autocap"):
            imgs_dir = train_images_dir.strip()
            out_jsonl = train_jsonl_path.strip()

            if not imgs_dir:
                st.error("Please specify an images folder to caption.")
            elif not out_jsonl:
                st.error("Please specify a captions JSONL path.")
            else:
                try:
                    def _run_autocap_ds():
                        old_argv = sys.argv
                        try:
                            argv = [
                                "auto_caption_traffic_folder",
                                "--images_dir",
                                imgs_dir,
                                "--out",
                                out_jsonl,
                                "--ollama_url",
                                OLLAMA_URL,
                                "--model_name",
                                MODEL_NAME,
                            ]
                            sys.argv = argv
                            auto_caption_main()
                        finally:
                            sys.argv = old_argv

                    with st.spinner(
                        f"Auto-captioning images under {imgs_dir} into {out_jsonl} "
                        f"using {MODEL_NAME} via Ollama..."
                    ):
                        run_with_logging(
                            f"AUTO_CAPTION_DATASET images_dir={imgs_dir} out={out_jsonl}",
                            _run_autocap_ds,
                        )

                    st.success(f"Auto-captioning complete. Captions written to: {out_jsonl}")
                except Exception as e:
                    st.error(f"Auto-captioning failed: {e}")

    # --- Step 2: training helpers for LLaVA / Qwen ---
    with col_train:
        train_target = st.selectbox(
            "Generate training command for",
            ["LLaVA-HF (llava-1.5-7b)", "Qwen2.5-VL-7B"],
            key="train_target_select",
        )

        default_out_dir = (
            os.path.join(PROJECT_ROOT, "outputs", "llava_traffic_lora")
            if train_target.startswith("LLaVA")
            else os.path.join(PROJECT_ROOT, "outputs", "qwenvl_traffic_lora")
        )

        train_out_dir = st.text_input(
            "LoRA output directory",
            value=default_out_dir,
            key="train_out_dir",
        )

        show_cmd_btn = st.button(
            "📋 Show training command (manual run)", key="btn_show_train_cmd"
        )

        if show_cmd_btn:
            jsonl_rel = os.path.relpath(train_jsonl_path, PROJECT_ROOT)
            out_rel = os.path.relpath(train_out_dir, PROJECT_ROOT)

            if train_target.startswith("LLaVA"):
                cmd = f"""# from the project root (where src/ lives)
python src\\train_lora_llava.py `
  --train "{jsonl_rel}" `
  --out "{out_rel}" `
  --model_id "llava-hf/llava-1.5-7b-hf" `
  --epochs 1 `
  --bs 1 `
  --ga 16 `
  --lr 2e-4 `
  --max_length 512 `
  --fourbit
"""
            else:
                cmd = f"""# from the project root (where src/ lives)
python src\\train_lora_qwenvl.py `
  --train "{jsonl_rel}" `
  --out "{out_rel}" `
  --model_id "{QWEN_MODEL_ID}" `
  --epochs 1 `
  --bs 1 `
  --ga 8 `
  --lr 2e-4 `
  --max_length 1024
"""

            st.markdown(
                "Copy–paste this into **PowerShell** from the project root to start training."
            )
            st.code(cmd, language="powershell")

        st.markdown("##### 🔌 One-click training on current LLaVA captions")

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            btn_train_llava = st.button(
                "🚀 Train LLaVA LoRA now", key="btn_train_llava_now"
            )
        with col_t2:
            btn_train_qwen = st.button(
                "🚀 Train Qwen2.5-VL LoRA now", key="btn_train_qwenvl_now"
            )

        if btn_train_llava or btn_train_qwen:
            captions_path = train_jsonl_path.strip()
            if not captions_path or not os.path.exists(captions_path):
                st.error(
                    "Captions JSONL not found. "
                    "Run the auto-caption step above first so LLaVA can generate captions."
                )
            else:
                os.makedirs(train_out_dir, exist_ok=True)
                try:
                    # LLaVA LoRA on LLaVA captions
                    if btn_train_llava:
                        with st.spinner(
                            "Training LLaVA-HF LoRA on LLaVA auto-captions "
                            "(this may take a while)..."
                        ):
                            run_with_logging(
                                f"TRAIN_LLAVA train={captions_path} out={train_out_dir}",
                                run_llava_lora,
                                train_path=captions_path,
                                out_dir=train_out_dir,
                                model_id="llava-hf/llava-1.5-7b-hf",
                                epochs=1,
                                bs=1,
                                ga=16,
                                lr=2e-4,
                                max_length=512,
                                fourbit=True,
                            )
                        st.success(
                            f"LLaVA LoRA training complete. "
                            f"Adapters saved to: `{train_out_dir}`"
                        )

                    # Qwen LoRA on the same LLaVA captions
                    if btn_train_qwen:
                        with st.spinner(
                            "Training Qwen2.5-VL LoRA on LLaVA auto-captions "
                            "(this may take a while)..."
                        ):
                            run_with_logging(
                                f"TRAIN_QWEN train={captions_path} out={train_out_dir}",
                                run_qwenvl_lora,
                                train_path=captions_path,
                                out_dir=train_out_dir,
                                model_id=QWEN_MODEL_ID,
                                epochs=1,
                                bs=1,
                                ga=8,
                                lr=2e-4,
                                max_length=1024,
                            )
                        st.success(
                            f"Qwen2.5-VL LoRA training complete. "
                            f"Adapters saved to: `{train_out_dir}`.\n\n"
                            "To start USING it in the dashboard, set the "
                            "Qwen LoRA directory in the sidebar to this path."
                        )

                except Exception as e:
                    st.error(f"LoRA training failed: {e}")

        # --- Step 3: ONE BUTTON to do everything ---
        st.markdown("---")
        st.markdown("##### 🧨 Full pipeline: auto-caption + train both models")

        btn_autocap_and_train = st.button(
            "🔥 Auto-caption folder + train LLaVA & Qwen",
            key="btn_autocap_and_train_all",
        )

        if btn_autocap_and_train:
            imgs_dir = train_images_dir.strip()
            captions_path = train_jsonl_path.strip()

            if not imgs_dir:
                st.error("Please specify an images folder in the left column.")
            elif not captions_path:
                st.error("Please specify a captions JSONL path above.")
            else:
                # Fixed default output dirs under your project:
                llava_out_dir = os.path.join(PROJECT_ROOT, "outputs", "llava_traffic_lora")
                qwenvl_out_dir = os.path.join(PROJECT_ROOT, "outputs", "qwenvl_traffic_lora")

                os.makedirs(os.path.dirname(llava_out_dir), exist_ok=True)
                os.makedirs(os.path.dirname(qwenvl_out_dir), exist_ok=True)

                try:
                    # STEP 1: Auto-caption with LLaVA via Ollama
                    def _run_autocap_full():
                        old_argv = sys.argv
                        try:
                            argv = [
                                "auto_caption_traffic_folder",
                                "--images_dir",
                                imgs_dir,
                                "--out",
                                captions_path,
                                "--ollama_url",
                                OLLAMA_URL,
                                "--model_name",
                                MODEL_NAME,
                            ]
                            sys.argv = argv
                            auto_caption_main()
                        finally:
                            sys.argv = old_argv

                    with st.spinner(
                        f"Auto-captioning images under {imgs_dir} into {captions_path} "
                        f"using {MODEL_NAME} via Ollama..."
                    ):
                        run_with_logging(
                            f"AUTO_CAPTION_FULL images_dir={imgs_dir} out={captions_path}",
                            _run_autocap_full,
                        )

                    # STEP 2: Train LLaVA LoRA on those captions
                    with st.spinner(
                        "Training LLaVA-HF LoRA on the generated captions "
                        "(this may take a while)..."
                    ):
                        run_with_logging(
                            f"TRAIN_LLAVA_FULL train={captions_path} out={llava_out_dir}",
                            run_llava_lora,
                            train_path=captions_path,
                            out_dir=llava_out_dir,
                            model_id="llava-hf/llava-1.5-7b-hf",
                            epochs=1,
                            bs=1,
                            ga=16,
                            lr=2e-4,
                            max_length=512,
                            fourbit=True,
                        )

                    # STEP 3: Train Qwen LoRA on the same captions
                    with st.spinner(
                        "Training Qwen2.5-VL LoRA on the generated captions "
                        "(this may take a while)..."
                    ):
                        run_with_logging(
                            f"TRAIN_QWEN_FULL train={captions_path} out={qwenvl_out_dir}",
                            run_qwenvl_lora,
                            train_path=captions_path,
                            out_dir=qwenvl_out_dir,
                            model_id=QWEN_MODEL_ID,
                            epochs=1,
                            bs=1,
                            ga=8,
                            lr=2e-4,
                            max_length=1024,
                        )

                    st.success(
                        "Full pipeline complete.\n\n"
                        f"- Captions JSONL: `{captions_path}`\n"
                        f"- LLaVA LoRA saved at: `{llava_out_dir}`\n"
                        f"- Qwen2.5-VL LoRA saved at: `{qwenvl_out_dir}`\n\n"
                        "You can inspect all logs in the 'Training / auto-caption logs' panel below. "
                        "To start USING the new Qwen LoRA in the dashboard, set the sidebar "
                        "LoRA directory to the Qwen output path above."
                    )
                except Exception as e:
                    st.error(f"Full pipeline failed: {e}")

    # =========================
    # Training / auto-caption logs viewer
    # =========================

    st.markdown("---")
    st.markdown("#### 📜 Training / auto-caption logs")

    col_log_left, col_log_right = st.columns(2)
    with col_log_left:
        refresh_log = st.button("🔄 Refresh log view", key="btn_refresh_train_log")
    with col_log_right:
        clear_log = st.button("🧹 Clear logs", key="btn_clear_train_log")

    log_exists = os.path.exists(TRAIN_LOG_PATH)

    if clear_log and log_exists:
        try:
            os.remove(TRAIN_LOG_PATH)
            st.success("Training logs cleared.")
            log_exists = False
        except Exception as e:
            st.error(f"Failed to clear training logs: {e}")

    if not log_exists:
        st.info("No training logs yet. Run auto-caption or LoRA training to populate this.")
    else:
        try:
            with open(TRAIN_LOG_PATH, "r", encoding="utf-8") as f:
                log_text = f.read()
        except Exception as e:
            st.error(f"Failed to read training log file: {e}")
        else:
            st.text_area(
                "Recent training / auto-caption logs",
                value=log_text,
                height=320,
            )
