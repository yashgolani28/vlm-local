import cv2
import base64
import requests
import argparse
import os
import math
import tempfile
import re
from collections import Counter
from anpr import run_anpr_multi, INDIA_STATE_CODES

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "llava:7b"

def _edit_distance(a: str, b: str) -> int:
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
            score = ( -cnt, dsum, len(cand) )
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

def extract_sample_frames(video_path: str, num_frames: int = 6):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    # Choose evenly spaced indices across the video
    step = max(1, frame_count // num_frames)
    indices = [min(i * step, frame_count - 1) for i in range(num_frames)]

    print(f"[info] total frames: {frame_count}, sampling indices: {indices}")

    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"[warn] failed to read frame at index {idx}")
            continue
        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError("No frames could be extracted.")

    return frames, indices


def frames_to_base64(frames):
    frames_b64 = []
    for i, frame in enumerate(frames):
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            print(f"[warn] failed to encode frame #{i} as JPEG")
            continue
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        frames_b64.append(b64)
    if not frames_b64:
        raise RuntimeError("No frames could be encoded to JPEG/base64.")
    return frames_b64


def call_llava_with_frames(frames_b64, prompt: str):
    """
    Call Ollama /api/chat with multiple frames for LLaVA.
    The images MUST be attached to the message, not top-level.
    """
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": frames_b64,
            }
        ],
    }

    print("[info] sending request to LLaVA…")
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()

    data = resp.json()
    # /api/chat returns: { "message": { "role": "assistant", "content": "..." }, ... }
    return data["message"]["content"]

def run_anpr_on_frames(frames, frame_indices, min_conf: float = 0.40):
    """
    Run ANPR on each sampled frame using the custom model.

    Now uses run_anpr_multi(), so each frame can contribute multiple plates.
    Returns a flat list of dicts:
      - frame_idx: index in the original video
      - plate: decoded plate text
      - conf: confidence
      - crop_path: saved crop path (if any)
    """
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    snapshot_dir = os.path.join(project_root, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    results = []

    for frame, vid_idx in zip(frames, frame_indices):
        frame_name = f"frame_{vid_idx:06d}.jpg"
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
        except Exception as e:
            print(f"[anpr] error on frame {vid_idx}: {e}")
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

def summarize_plates(anpr_results):
    """
    Print per-frame detections, then refined unique plates and region guess.
    """
    if not anpr_results:
        return

    print("=== ANPR results on sampled frames ===\n")
    for r in anpr_results:
        print(
            f"Frame {r['frame_idx']:6d} -> plate: {r['plate']}  "
            f"(conf {r['conf']:.2f})  crop: {r['crop_path']}"
        )
    print("\n====================================\n")

    refined = _refine_plate_clusters(anpr_results)
    if not refined:
        return

    print("=== Refined unique plates across sampled frames ===\n")
    for p in refined:
        print(p)
    print("\n==========================================\n")

    region = _infer_region_from_plates(refined)
    if region:
        print(
            f"Likely region from plates: {region['country']} "
            f"(state code {region['state_code']}), "
            f"confidence ≈ {region['confidence']:.2f}\n"
        )

def main():
    parser = argparse.ArgumentParser(description="Test LLaVA + ANPR on a video via Ollama.")
    parser.add_argument("video", help="Path to the video file (e.g., C:\\videos\\test.mp4)")
    parser.add_argument(
        "--frames",
        type=int,
        default=6,
        help="Number of frames to sample from the video (default: 6)",
    )
    parser.add_argument(
        "--no-anpr",
        action="store_true",
        help="Skip ANPR and only run LLaVA (debug option).",
    )
    args = parser.parse_args()

    video_path = os.path.abspath(args.video)
    if not os.path.isfile(video_path):
        raise SystemExit(f"Video not found: {video_path}")

    print(f"[info] using video: {video_path}")

    frames, indices = extract_sample_frames(video_path, num_frames=args.frames)

    # Encode frames for LLaVA
    frames_b64 = frames_to_base64(frames)

    # Your traffic-analysis prompt
    prompt = (
        "You are a traffic analysis assistant. Look at this input and in numbered points:\n"
        "1. Describe the place (e.g., highway, junction, surroundings).\n"
        "2. Count visible vehicles (by type: car, bike, truck, bus, auto, other).\n"
        "3. Count visible pedestrians.\n"
        "4. Read any clearly visible license plates (if none, say 'no readable plates').\n"
        "5. Is it day, evening, night, or dawn/dusk?\n"
        "6. If you can guess the approximate region (city/country), say it and your confidence; "
        "otherwise say you cannot tell.\n"
        "Keep the answer compact but structured."
    )

    answer = call_llava_with_frames(frames_b64, prompt)

    print("\n=== LLaVA video summary ===\n")
    print(answer)
    print("\n===========================\n")

    if not args.no_anpr:
        anpr_results = run_anpr_on_frames(frames, indices, min_conf=0.40)

        if anpr_results:
            summarize_plates(anpr_results)
        else:
            print(
                "[ANPR] No plates decoded on the sampled frames "
                "(or all below confidence threshold).\n"
            )

if __name__ == "__main__":
    main()