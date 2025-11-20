import os
import io
import json
import argparse
import random
import base64
from typing import Any, Dict, List

import requests
from PIL import Image

# Match your dashboard defaults:
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL_NAME", "llava:13b")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def image_to_b64_jpeg(path: str, max_long_side: int = 480) -> str:
    """
    Load an image, downscale longest side to max_long_side, encode as JPEG -> base64.
    Mirrors the dashboard's 'resize for VLM' idea so eval ~= live behavior.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    long_side = max(w, h)
    if long_side > max_long_side:
        scale = max_long_side / float(long_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64


def call_ollama_llava_single(img_path: str, prompt: str) -> str:
    """
    Call Ollama LLaVA 13B with a single image + text prompt.
    Matches the shape used in vlm_dashboard.call_llava_single_image.
    """
    b64 = image_to_b64_jpeg(img_path)

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": [b64],
            }
        ],
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=str,
        required=True,
        help="Jsonl with image/prompt/answer rows (e.g. data/traffic_captions_val.jsonl)",
    )
    ap.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="How many random samples to print for inspection",
    )
    ap.add_argument(
        "--max_eval",
        type=int,
        default=64,
        help="Max rows to score for the substring metric (for speed)",
    )
    args = ap.parse_args()

    rows = read_jsonl(args.data)
    if not rows:
        raise SystemExit(f"No rows found in {args.data}")

    print(f"[eval-ollama] Loaded {len(rows)} rows from {args.data}")
    print(f"[eval-ollama] Using OLLAMA_URL={OLLAMA_URL}, model={OLLAMA_MODEL}")

    # Shuffle once
    random.shuffle(rows)

    # ---- Pretty examples ----
    sample_rows = rows[: args.num_samples]
    print("\n=== SAMPLE GENERATIONS (Ollama LLaVA 13B) ===")
    for r in sample_rows:
        img_path = r["image"]
        prompt = r["prompt"]
        gold = r.get("answer", "")

        try:
            pred = call_ollama_llava_single(img_path, prompt)
        except Exception as e:
            print("===")
            print(f"Image:  {img_path}")
            print(f"Prompt: {prompt}")
            print(f"ERROR calling Ollama: {e}")
            print("===")
            continue

        print("===")
        print(f"Image:  {img_path}")
        print(f"Prompt: {prompt}")
        print(f"Pred:   {pred}")
        print(f"Gold:   {gold}")
        print("===")

    # ---- Crude substring metric on a small batch ----
    eval_rows = rows[: args.max_eval]
    hit = 0
    total = 0

    print("\n[eval-ollama] Running substring sanity metric on "
          f"{len(eval_rows)} rows...")

    for r in eval_rows:
        gold = (r.get("answer") or "").strip()
        if not gold:
            continue

        img_path = r["image"]
        prompt = r["prompt"]

        try:
            pred = call_ollama_llava_single(img_path, prompt)
        except Exception as e:
            print(f"[warn] Skipping row (error calling Ollama): {e}")
            continue

        total += 1
        if gold.lower() in pred.lower():
            hit += 1

    if total > 0:
        print(
            f"\n[eval-ollama] exact-substring hits: {hit}/{total} "
            f"({100.0 * hit / total:.1f}%)"
        )
    else:
        print("\n[eval-ollama] No non-empty gold answers to score.")


if __name__ == "__main__":
    main()
