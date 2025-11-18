import cv2
import base64
import requests
import argparse
import time
import os
from datetime import datetime

# Ollama / LLaVA config
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "llava:7b"

# Default RTSP for your Axis M1125
DEFAULT_RTSP = (
    "rtsp://root:2024@192.168.1.241/axis-media/media.amp?videocodec=h264"
)

# Prompt for traffic / road analysis
PROMPT_TEMPLATE = (
    "You are a traffic analysis assistant. Look at this single frame from a "
    "fixed CCTV traffic camera and answer concisely in numbered points:\n"
    "1. Describe the place (e.g., highway, junction, lane count, surroundings).\n"
    "2. Count visible vehicles, grouped by type (car, bike, truck, bus, auto, other). "
    "Say 0 if none.\n"
    "3. Count visible pedestrians and briefly describe where they are.\n"
    "4. Read aloud any clearly visible license plates. If none are clearly visible, "
    "say 'no readable plates'. If partially visible or too small, say 'unreadable'.\n"
    "5. Is it day, evening, night, or dawn/dusk? Answer with one option and a justification.\n"
    "6. If you can guess an approximate city or country, state it and how sure you are. "
    "Otherwise say you cannot tell.\n"
    "Keep the answer short but structured."
)


def frame_to_base64_jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    jpg_bytes = buf.tobytes()
    return base64.b64encode(jpg_bytes).decode("utf-8")


def call_llava(b64_image: str, prompt: str) -> str:
    """
    Call Ollama LLaVA (llava:7b) with a single base64 JPEG frame.
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

    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    # Ollama /api/chat: { "message": { "role": "assistant", "content": "..." }, ... }
    return data["message"]["content"]


def connect_capture(rtsp_url: str):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open RTSP stream: {rtsp_url}")
    return cap


def main():
    parser = argparse.ArgumentParser(description="Test LLaVA on a live RTSP camera.")
    parser.add_argument(
        "--rtsp",
        type=str,
        default=DEFAULT_RTSP,
        help="RTSP URL of the camera "
             f"(default: {DEFAULT_RTSP})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Seconds between LLaVA analyses (default: 10.0)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show a live preview window (press 'q' to quit).",
    )
    args = parser.parse_args()

    rtsp_url = args.rtsp
    interval = max(1.0, args.interval)

    print(f"[info] using RTSP: {rtsp_url}")
    print(f"[info] analysis interval: {interval} seconds")
    print("[info] connecting to camera...")

    cap = None
    last_analysis_time = 0.0

    try:
        cap = connect_capture(rtsp_url)
        print("[info] camera stream opened successfully.")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[warn] failed to read frame, attempting to reconnect...")
                cap.release()
                time.sleep(2.0)
                try:
                    cap = connect_capture(rtsp_url)
                    print("[info] reconnected to camera.")
                    continue
                except Exception as e:
                    print(f"[error] reconnection failed: {e}")
                    time.sleep(5.0)
                    continue

            # Show preview if requested
            if args.show:
                cv2.imshow("Axis M1125 Live (press 'q' to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[info] 'q' pressed, exiting...")
                    break

            now = time.time()
            if now - last_analysis_time >= interval:
                last_analysis_time = now

                try:
                    b64 = frame_to_base64_jpeg(frame)
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print("\n" + "=" * 60)
                    print(f"[info] {ts} - sending frame to LLaVA...")
                    answer = call_llava(b64, PROMPT_TEMPLATE)
                    print("\n=== LLaVA traffic analysis ===\n")
                    print(answer)
                    print("\n" + "=" * 60 + "\n")
                except Exception as e:
                    print(f"[error] LLaVA analysis failed: {e}")

    except KeyboardInterrupt:
        print("\n[info] keyboard interrupt, exiting...")
    except Exception as e:
        print(f"[fatal] {e}")
    finally:
        if cap is not None:
            cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
