import os, json, argparse, random
from typing import List, Dict, Any

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def generate_one(
    model: AutoModelForCausalLM,
    processor: AutoProcessor,
    image_path: str,
    prompt: str,
    max_new_tokens: int = 160,
) -> str:
    device = model.device
    img = Image.open(image_path).convert("RGB")

    # Chat-style input with image + text, same as training template
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    pixel_values = processor(images=img, return_tensors="pt")["pixel_values"]

    inputs = {k: v.to(device) for k, v in inputs.items()}
    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )

    # Strip off the prompt part
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = out[0][prompt_len:]
    text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="jsonl file with image/prompt/answer")
    ap.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--lora_dir", default="", help="Path to trained LoRA directory")
    ap.add_argument("--num_samples", type=int, default=5)
    args = ap.parse_args()

    rows = read_jsonl(args.data)
    if not rows:
        raise SystemExit(f"No rows found in {args.data}")

    random.shuffle(rows)
    rows = rows[: args.num_samples]

    print("Loading model + processor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else None

    processor = AutoProcessor.from_pretrained(args.model_id)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if args.lora_dir and os.path.isdir(args.lora_dir):
        base = PeftModel.from_pretrained(base, args.lora_dir)

    base.eval()

    for obj in rows:
        img_path = obj["image"]
        prompt = obj["prompt"]
        gold = obj.get("answer", "")

        pred = generate_one(base, processor, img_path, prompt, max_new_tokens=160)

        print("===")
        print(f"Image:  {img_path}")
        print(f"Prompt: {prompt}")
        print(f"Pred:   {pred}")
        print(f"Gold:   {gold}")
        print("===")


if __name__ == "__main__":
    main()
