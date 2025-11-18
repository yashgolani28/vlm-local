# -*- coding: utf-8 -*-
"""
Evaluation harness for Qwen2.5-VL LoRA on traffic captions.

- Loads base Qwen2.5-VL-3B-Instruct in 4-bit.
- Applies LoRA adapter from --lora_dir.
- Runs on a jsonl file with fields: image, prompt, answer.
- Prints a few sample generations and a crude substring hit rate.
"""

import os
import json
import argparse
import random
from typing import List, Dict, Any

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_messages(image_path: str, prompt: str):
    # Single user turn with one image + text, same style as training.
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "resized_height": 560,
                    "resized_width": 560,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def generate_one(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 160,
) -> str:
    messages = build_messages(image_path, prompt)
    # Chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Vision preprocessing (critical for Qwen2.5-VL)
    image_inputs, video_inputs = process_vision_info([messages])
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    raw = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

    marker = "\nassistant\n"
    if marker in raw:
        answer = raw.split(marker)[-1].strip()
    else:
        answer = raw.strip()

    return answer

def main():
    ap = argparse.ArgumentParser("Eval Qwen2.5-VL LoRA on traffic jsonl")
    ap.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Base Qwen2.5-VL model id",
    )
    ap.add_argument(
        "--lora_dir",
        type=str,
        required=True,
        help="Path to LoRA adapter directory (e.g. outputs/qwenvl_idd_lora_50)",
    )
    ap.add_argument(
        "--data",
        type=str,
        required=True,
        help="Jsonl file with image/prompt/answer rows (e.g. data/idd_val_10.jsonl)",
    )
    ap.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="How many random samples to print",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=160,
        help="Max new tokens to generate",
    )
    args = ap.parse_args()

    rows = read_jsonl(args.data)
    if not rows:
        raise SystemExit(f"No rows found in {args.data}")

    print(f"[eval] Loaded {len(rows)} rows from {args.data}")

    # 4-bit quantization to fit on 3050 Ti
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print(f"[eval] Loading base model: {args.model_id}")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb,
        device_map="auto",
    )
    print(f"[eval] Attaching LoRA from: {args.lora_dir}")
    base = PeftModel.from_pretrained(base, args.lora_dir)
    base.eval()

    processor = AutoProcessor.from_pretrained(args.model_id)

    # Shuffle and subset for pretty printing
    random.shuffle(rows)
    sample_rows = rows[: args.num_samples]

    print("\n=== SAMPLE GENERATIONS (LoRA) ===")
    for r in sample_rows:
        pred = generate_one(
            base,
            processor,
            r["image"],
            r["prompt"],
            max_new_tokens=args.max_new_tokens,
        )
        print(f"Image:  {r['image']}")
        print(f"Prompt: {r['prompt']}")
        print(f"Pred:   {pred}")
        print(f"Gold:   {r.get('answer','')}")
        print("---")

    # crude exact match sanity metric: does the full gold caption appear as substring in pred?
    total = 0
    hit = 0
    for r in rows:
        gold = (r.get("answer") or "").strip()
        if not gold:
            continue
        pred = generate_one(
            base,
            processor,
            r["image"],
            r["prompt"],
            max_new_tokens=args.max_new_tokens,
        ).lower()
        total += 1
        if gold.lower() in pred:
            hit += 1
    if total > 0:
        print(f"\n[sanity] exact-substring hits: {hit}/{total} "
              f"({100.0 * hit / total:.1f}%)")
    else:
        print("\n[sanity] No non-empty gold answers to score.")


if __name__ == "__main__":
    main()
