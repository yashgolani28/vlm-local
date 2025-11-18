import os
import io
import time
import json
import base64
import tempfile
from datetime import datetime
import re
from collections import Counter

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

from anpr import run_anpr, run_anpr_multi, INDIA_STATE_CODES

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# =========================
# Global config
# =========================

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "llava:7b"

# Default RTSP for your Axis M1125
DEFAULT_RTSP = (
    "rtsp://root:2024@192.168.1.241/axis-media/media.amp?videocodec=h264"
)

# Dataset paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
IMAGES_DIR = os.path.join(DATA_DIR, "traffic_images")
DATASET_PATH = os.path.join(DATA_DIR, "traffic_vqa.jsonl")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

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


def sample_video_frames_from_file(file_bytes: bytes, num_frames: int = 6):
    """
    Save uploaded video bytes to a temp file, then sample frames with OpenCV.
    Returns a list of frames (np.ndarray, BGR).
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
    indices = [min(i * step, frame_count - 1) for i in range(num_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frames.append(frame)

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
            plate = (out.get("plate_text") or "").strip()
            conf = float(out.get("plate_conf") or 0.0)
            if not plate or conf < min_conf:
                continue

            results.append(
                {
                    "vehicle_bbox": None,
                    "vehicle_conf": 1.0,  # dummy value for display
                    "plate_text": plate,
                    "plate_conf": conf,
                    "plate_bbox": out.get("plate_bbox"),
                    "crop_path": out.get("crop_path"),
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
        p = re.sub(r"[^A-Z0-9]", "", (r.get("plate") or "").upper())
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
            # more frequent + closer to others is better; shorter is slightly preferred
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
    # Store raw frames and crops under repo_root/snapshots (same layout as CLI helper)
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

    # Cluster plate strings across all frames
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

# =========================
# Streamlit UI
# =========================

st.set_page_config(
    page_title="Traffic VLM Dashboard (LLaVA)",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Traffic VLM Dashboard – LLaVA + Ollama")
st.caption("Image / Video / Live Cam testing + Dataset building for traffic scenarios.")


# Sidebar
st.sidebar.header("Model & Connection")
st.sidebar.markdown(f"**Model:** `{MODEL_NAME}`")
ollama_host = st.sidebar.text_input("Ollama URL", value=OLLAMA_URL)
if ollama_host != OLLAMA_URL:
    # Update global for this run
    OLLAMA_URL = ollama_host

st.sidebar.info(
    "Make sure Ollama is running and `llava:7b` is pulled.\n\n"
    "In PowerShell: `ollama pull llava:7b` then `ollama run llava:7b` (once)."
)


tab_playground, tab_live, tab_dataset = st.tabs(
    ["🧪 Playground (Image / Video)", "📡 Live Camera", "📚 Dataset & Training"]
)

# =========================
# TAB 1 – Playground
# =========================

with tab_playground:
    st.subheader("🧪 Playground – Test LLaVA on uploads")

    mode = st.radio(
        "Select input type:",
        ["Image", "Video"],
        horizontal=True,
    )

    default_prompt = (
            "You are a traffic analysis assistant. Look at this traffic image or a small set of frames and reply in clear numbered points:\n"
            "1. Describe the place (road type, junction or straight road, surroundings, lighting, weather).\n"
            "2. Count visible vehicles by type (car, bike, truck, bus, auto, other) and mention whether they are mostly moving or stopped.\n"
            "3. Count visible pedestrians and briefly say where they are relative to the road (on footpath, crossing, standing near lane, etc.).\n"
            "4. Do NOT read or guess any license plate numbers yourself. A separate ANPR system will decode plates; just mention whether plates are visible and roughly how many.\n"
            "5. State whether it looks like day, evening, night, or dawn/dusk and give a short justification.\n"
            "6. Comment on any safety / risk factors you notice (e.g., jaywalking, vehicles too close, wrong-side driving, obscured lanes).\n"
            "7. If you can guess the approximate region (city/country), say it and how confident you are; otherwise say you cannot tell.\n"
            "Make the answer structured and reasonably detailed, but avoid unnecessary repetition."
        )

    prompt = st.text_area(
        "Prompt (instruction to LLaVA)",
        value=default_prompt,
        height=180,
    )

    if mode == "Image":
        uploaded_img = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png"],
        )

        col1, col2 = st.columns([1, 1])

        if uploaded_img is not None:
            img = Image.open(uploaded_img).convert("RGB")
            with col1:
                st.image(img, caption="Uploaded Image", use_column_width=True)

            if st.button("Run LLaVA on Image", type="primary"):
                with st.spinner("Calling LLaVA + ANPR..."):
                    try:
                        # LLaVA
                        b64 = pil_to_base64_jpeg(img)
                        answer = call_llava_single_image(b64, prompt)

                        # ANPR on all detected vehicles
                        frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                        anpr_results = run_anpr_on_frame(frame_bgr)
                    except Exception as e:
                        st.error(f"Analysis failed: {e}")
                    else:
                        with col2:
                            st.markdown("### LLaVA Output")
                            st.write(answer)

                            anpr_container = st.container()
                            render_anpr_results(anpr_results, anpr_container)
        else:
            st.info("Upload an image to begin.")

    else:  # Video
        uploaded_vid = st.file_uploader(
            "Upload a video",
            type=["mp4", "avi", "mov", "mkv"],
        )

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

        if st.button("Run LLaVA on Video", type="primary"):
                        with st.spinner("Sampling frames and calling LLaVA..."):
                            # First run LLaVA on a small set of sampled frames
                            try:
                                file_bytes = uploaded_vid.read()
                                frames, indices = sample_video_frames_from_file(
                                    file_bytes, num_frames=num_frames
                                )
                                b64_list = [frame_to_base64_jpeg(f) for f in frames]

                                video_prompt = (
                                    "You are a traffic analysis assistant. You are given several "
                                    "frames from a single CCTV traffic video, in chronological order. "
                                    "Look at ALL frames together and reply in clear numbered points:\n"
                                    "1. Describe the overall place, road layout (lane count, presence of divider/median), surroundings, lighting and weather across the clip.\n"
                                    "2. Summarise traffic across the whole clip: approximate number of vehicles by type (car, bike, truck, bus, auto, other), whether they are mostly moving or stopped, and the general flow direction.\n"
                                    "3. Describe any pedestrians: how many, where they appear with respect to the road, and whether they cross or walk along the road.\n"
                                    "4. Mention any interesting or important events (vehicles turning, stopping suddenly, close interactions, vehicles entering/exiting side roads, occlusions, etc.).\n"
                                    "5. State whether it looks like day, evening, night, or dawn/dusk and give a short justification.\n"
                                    "6. Comment on visible safety / risk factors (e.g., jaywalking, vehicles too close, wrong-side driving, blocked view, poor lighting).\n"
                                    "7. Do NOT read or guess any license plate numbers yourself. A separate ANPR system will decode plates; just mention if plates are visible and on roughly how many vehicles.\n"
                                    "Make the answer structured and detailed enough to be useful to a traffic engineer."
                                )
                                # Combine user prompt with this video-specific one
                                full_prompt = video_prompt + "\nUser prompt:\n" + prompt

                                answer = call_llava_multi_images(b64_list, full_prompt)
                            except Exception as e:
                                st.error(f"LLaVA video analysis failed: {e}")
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
                                    st.markdown("### Sampled Frames (first up to 4)")
                                    # Show up to 4 frames
                                    show_n = min(4, len(frames))
                                    for i in range(show_n):
                                        st.image(
                                            cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB),
                                            caption=f"Frame index {indices[i]}",
                                            use_column_width=True,
                                        )

                                    st.markdown("### LLaVA Video Summary")
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

        else:
            st.info("Upload a video to test LLaVA on multiple frames.")


# =========================
# TAB 2 – Live Camera
# =========================

with tab_live:
    st.subheader("📡 Live Camera – Axis / RTSP + LLaVA")

    col_l, col_r = st.columns([1, 1])

    with col_l:
        rtsp_url = st.text_input(
            "RTSP URL",
            value=DEFAULT_RTSP,
            help="Axis example: rtsp://user:pass@ip/axis-media/media.amp?videocodec=h264",
        )

        live_prompt_default = (
            "You are a traffic analysis assistant. Look at this single frame from a fixed CCTV traffic camera "
            "and answer concisely in numbered points:\n"
            "1. Describe the place (e.g., highway, junction, lane count, surroundings).\n"
            "2. Count visible vehicles (by type: car, bike, truck, bus, auto, other).\n"
            "3. Count visible pedestrians and where they are.\n"
            "4. Do NOT read or guess any license plate numbers yourself. A separate ANPR system will provide plate text; "
            "simply mention that plates are present or not.\n"
            "5. Is it day, evening, night, or dawn/dusk? Justify briefly.\n"
            "6. If you can guess approximate region (city/country), say it and your confidence; else say you cannot tell."
        )

        live_prompt = st.text_area(
            "Prompt for live frame analysis",
            value=live_prompt_default,
            height=200,
        )

        # Controls for continuous mode
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
        with st.spinner("Capturing frame and calling LLaVA + ANPR..."):
            try:
                frame = capture_rtsp_frame(rtsp_url)

                # LLaVA
                b64 = frame_to_base64_jpeg(frame)
                answer = call_llava_single_image(b64, live_prompt)

                # ANPR on all detected vehicles in this frame
                anpr_results = run_anpr_on_frame(frame)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    frame_rgb, caption="Live Frame from RTSP", use_column_width=True
                )
                result_placeholder.markdown("### LLaVA Live Analysis (single shot)")
                result_placeholder.write(answer)
                render_anpr_results(anpr_results, anpr_placeholder)
                update_live_anpr_history(anpr_results, summary_placeholder)
                status_placeholder.info("Last update: single capture")

            except Exception as e:
                st.error(f"Live capture or analysis failed: {e}")

    # --- Continuous mode ---
    if continuous_btn:
        status_placeholder.info(
            f"Starting continuous analysis: every {interval:.1f}s for {max_updates} updates. "
            "This will run inside the app – to stop early, interrupt the app or reload the page."
        )
        for i in range(int(max_updates)):
            try:
                frame = capture_rtsp_frame(rtsp_url)

                # LLaVA
                b64 = frame_to_base64_jpeg(frame)
                answer = call_llava_single_image(b64, live_prompt)

                # ANPR on all detected vehicles
                anpr_results = run_anpr_on_frame(frame)

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    frame_rgb,
                    caption=f"Live Frame #{i+1} from RTSP",
                    use_column_width=True,
                )
                result_placeholder.markdown(
                    f"### LLaVA Live Analysis (update {i+1}/{int(max_updates)})"
                )
                result_placeholder.write(answer)
                render_anpr_results(anpr_results, anpr_placeholder)
                update_live_anpr_history(anpr_results, summary_placeholder)

                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                status_placeholder.success(
                    f"Last update #{i+1} at {ts} – next in {interval:.1f}s"
                )
            except Exception as e:
                status_placeholder.error(f"Update #{i+1} failed: {e}")

            time.sleep(float(interval))

# =========================
# TAB 3 – Dataset & Training
# =========================

with tab_dataset:
    st.subheader("📚 Dataset & Training – Build traffic visual-instruction data")

    st.markdown(
        "This section helps you build a **LLaVA-style visual instruction dataset** "
        "for traffic scenarios. It stores images and JSONL records locally so you can "
        "later fine-tune LLaVA (e.g., in the official repo or on a cloud GPU)."
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

        # Keep a buffer for the answer in session_state so we can auto-fill + edit
        if "ds_answer_input" not in st.session_state:
            st.session_state["ds_answer_input"] = ""

        ds_answer = st.text_area(
            "Ideal answer (what a perfect model should say)",
            key="ds_answer_input",
            height=150,
            help=(
                "Write the ground-truth answer you want the fine-tuned model to produce. "
                "You can also auto-generate a first draft with base LLaVA and then edit it."
            ),
        )

        auto_label_btn = st.button("Auto-fill answer using base LLaVA", type="secondary")
        add_example_btn = st.button("Save example to dataset", type="primary")

    if auto_label_btn:
        if ds_img_file is None:
            st.error("Please upload an image first for auto-labelling.")
        elif not ds_instruction.strip():
            st.error("Instruction cannot be empty when auto-labelling.")
        else:
            try:
                img = Image.open(ds_img_file).convert("RGB")
                b64 = pil_to_base64_jpeg(img)
                auto_prompt = ds_instruction.strip()
                with st.spinner("Calling base LLaVA via Ollama to draft answer..."):
                    draft = call_llava_single_image(b64, auto_prompt)
                st.session_state["ds_answer_input"] = draft
                st.success("Draft answer generated. Review/edit it above, then click 'Save example to dataset'.")
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
        "Use this to manually compare **base LLaVA** vs your **ideal answer** "
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

    if st.button("Run base LLaVA on evaluation image"):
        if eval_img_file is None:
            st.error("Upload an image for evaluation.")
        else:
            try:
                img = Image.open(eval_img_file).convert("RGB")
                b64 = pil_to_base64_jpeg(img)
                with st.spinner("Calling LLaVA..."):
                    answer = call_llava_single_image(b64, eval_instruction.strip())

                st.markdown("##### Uploaded Image")
                st.image(img, use_column_width=True)

                st.markdown("##### LLaVA Answer")
                st.write(answer)

                if eval_ideal.strip():
                    st.markdown("##### Your Ideal Answer (for comparison)")
                    st.write(eval_ideal.strip())

            except Exception as e:
                st.error(f"Evaluation failed: {e}")

    st.markdown("---")
    st.markdown("#### 🛠 Training command template (for later)")

    st.markdown(
        "Below is an **example training command** you might run on a **cloud GPU** or a powerful Linux box "
        "using the official LLaVA repo. Adjust paths and hyperparameters as needed."
    )

    example_cmd = f"""# Example: LLaVA-LoRA training on traffic dataset (conceptual)
# Assuming:
#   - LLaVA repo cloned at ~/llava
#   - Base model: liuhaotian/llava-v1.5-7b
#   - Dataset JSONL: ./data/traffic_vqa.jsonl
#   - Images folder: ./data/traffic_images

cd ~/llava

torchrun --nproc_per_node=8 --master_port=29501 \
    train.py \\
    --model-path liuhaotian/llava-v1.5-7b \\
    --data-path ./data/traffic_vqa.jsonl \\
    --image-folder ./data/traffic_images \\
    --vision-tower openai/clip-vit-large-patch14-336 \\
    --mm-projector-type mlp2x_gelu \\
    --mm-vision-resolution 336 \\
    --output-dir ./checkpoints/llava-traffic-lora \\
    --num-epochs 3 \\
    --per-device-train-batch-size 2 \\
    --per-device-eval-batch-size 2 \\
    --learning-rate 2e-4 \\
    --lora-r 64 --lora-alpha 128 --lora-target-modules q_proj,v_proj,k_proj,o_proj \\
    --gradient-accumulation-steps 8 \\
    --save-steps 1000 --eval-steps 1000

# After training, you'd export the LoRA, merge it into the base,
# convert to GGUF, and create a new Ollama model (e.g., 'traffic-llava')."""

    st.code(example_cmd, language="bash")
