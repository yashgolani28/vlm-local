import os
import io
import json
import time
import base64
import random
import argparse
from pathlib import Path
from typing import List, Dict

import requests
from PIL import Image

# Defaults match your dashboard
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL_NAME = "llava:7b"

# A good, traffic-focused prompt
DEFAULT_PROMPT = """You are a traffic analysis assistant. Look at this traffic scene and reply in clear numbered points:
1. Describe the place (road type, junction or straight road, surroundings, lighting, and weather).
2. Roughly count visible vehicles by type (car, bike, truck, bus, auto, other) and mention whether they are mostly moving or stopped.
3. Roughly count visible pedestrians and say where they are relative to the road (on footpath, crossing, standing near lane, etc.).
4. Mention lane structure if visible (number of lanes, presence of divider/median, turning lanes, etc.).
5. State whether it looks like day, evening, night, or dawn/dusk and give a short justification.
6. Comment on any safety / risk factors (e.g., jaywalking, vehicles too close, wrong-side driving, blocked view, poor lighting).
7. Do NOT read or guess exact license plate numbers. If a plate is visible, just write the literal string "[license plate visible]" instead of the number.

Keep the answer concise but reasonably detailed and well-structured.
"""


def pil_to_base64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_llava_single_image(
    b64_image: str,
    prompt: str,
    ollama_url: str,
    model_name: str,
    timeout: int = 600,
) -> str:
    """
    Call LLaVA via Ollama with a single base64 JPEG image.
    """
    payload = {
        "model": model_name,
        "stream": False,
        "options": {
            # context window (prompt + answer); 4096 is safe for llava:13b
            "num_ctx": 4096,
            # max new tokens to generate – bump this to get longer answers
            "num_predict": 900,
            # sampling – keep slightly conservative for consistency
            "temperature": 0.2,
            "top_p": 0.9,
        },
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64_image],
            }
        ],
    }

    resp = requests.post(ollama_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def list_images(root: Path, exts=(".jpg", ".jpeg", ".png", ".bmp")) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.suffix.lower() in exts:
            files.append(p)
    return files


def load_existing(out_path: Path) -> Dict[str, Dict]:
    """
    If out_jsonl already exists, load it so we can resume / skip.
    Keyed by absolute image path.
    """
    existing = {}
    if not out_path.exists():
        return existing
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            img_path = os.path.abspath(obj.get("image", ""))
            if img_path:
                existing[img_path] = obj
    return existing


def main():
    ap = argparse.ArgumentParser(
        description="Auto-caption a folder of traffic images using Ollama LLaVA and write a LLaVA-style jsonl."
    )
    ap.add_argument(
        "--images_dir",
        required=True,
        help="Root folder containing traffic images (will be scanned recursively).",
    )
    ap.add_argument(
        "--out",
        default="data/public_traffic_captions.jsonl",
        help="Output jsonl file (image/prompt/answer per line).",
    )
    ap.add_argument(
        "--max_images",
        type=int,
        default=0,
        help="Optional cap on number of images (0 = all).",
    )
    ap.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle image order before processing.",
    )
    ap.add_argument(
        "--ollama_url",
        default=DEFAULT_OLLAMA_URL,
        help="Ollama chat endpoint (default http://127.0.0.1:11434/api/chat).",
    )
    ap.add_argument(
        "--model_name",
        default=DEFAULT_MODEL_NAME,
        help="Ollama model name (default llava:7b).",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep in seconds between requests (e.g., 0.2).",
    )
    ap.add_argument(
        "--prompt_file",
        help="Optional .txt file containing the caption prompt. If not set, uses built-in DEFAULT_PROMPT.",
    )
    args = ap.parse_args()

    images_dir = Path(args.images_dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    os.makedirs(out_path.parent, exist_ok=True)

    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    else:
        prompt = DEFAULT_PROMPT

    print(f"Scanning for images under: {images_dir}")
    imgs = list_images(images_dir)
    if not imgs:
        raise SystemExit("No images found.")

    if args.shuffle:
        random.shuffle(imgs)

    if args.max_images and args.max_images > 0:
        imgs = imgs[: args.max_images]

    print(f"Total images to process: {len(imgs)}")

    existing = load_existing(out_path)
    print(f"Existing caption rows in {out_path.name}: {len(existing)}")

    # Open output in append mode so we can resume
    out_f = out_path.open("a", encoding="utf-8")

    processed = 0
    skipped = 0
    failures = 0

    for idx, img_path in enumerate(imgs, start=1):
        img_abs = str(img_path.resolve())
        if img_abs in existing:
            skipped += 1
            if idx % 50 == 0:
                print(f"[{idx}/{len(imgs)}] skipped already-captioned file")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[{idx}/{len(imgs)}] Failed to open image {img_path}: {e}")
            failures += 1
            continue

        b64 = pil_to_base64_jpeg(img)

        try:
            answer = call_llava_single_image(
                b64_image=b64,
                prompt=prompt,
                ollama_url=args.ollama_url,
                model_name=args.model_name,
            )
        except Exception as e:
            print(f"[{idx}/{len(imgs)}] LLaVA call failed for {img_path}: {e}")
            failures += 1
            continue

        row = {
            "image": img_abs,
            "prompt": prompt,
            "answer": answer,
        }
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()

        processed += 1
        if args.sleep > 0:
            time.sleep(args.sleep)

        if idx % 20 == 0:
            print(
                f"[{idx}/{len(imgs)}] processed={processed}, skipped={skipped}, failures={failures}"
            )

    out_f.close()
    print("Done.")
    print(f"Processed: {processed}, skipped: {skipped}, failures: {failures}")
    print(f"Output written to: {out_path}")


if __name__ == "__main__":
    main()
