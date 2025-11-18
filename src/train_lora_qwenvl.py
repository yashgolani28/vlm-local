import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info


# -----------------------------
# Dataset: just image / prompt / answer
# -----------------------------


class QwenVLDataset(Dataset):
    """
    Each line in the jsonl is:
      {
        "image": "C:\\abs\\path\\to\\image.jpg",
        "prompt": "instruction text",
        "answer": "target caption"
      }
    """

    def __init__(self, path: str):
        self.rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if not obj.get("image") or not obj.get("prompt") or not obj.get("answer"):
                    continue
                self.rows.append(obj)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Return raw fields; collator will do all heavy lifting
        return self.rows[idx]


# -----------------------------
# Collator: build messages, run processor, build training tensors
# -----------------------------


@dataclass
class QwenCollator:
    processor: AutoProcessor
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        features: list of { "image", "prompt", "answer" }
        """
        tokenizer = self.processor.tokenizer

        # 1) Build Qwen-style messages for the whole batch
        messages_batch: List[List[Dict[str, Any]]] = []
        answers: List[str] = []

        for f in features:
            img_path = f["image"]
            prompt = f["prompt"]
            answer = f["answer"]

            # Single-user-turn message with image + text
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": img_path,
                            "resized_height": 560,
                            "resized_width": 560,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            messages_batch.append(messages)
            answers.append(answer)

        # 2) Apply chat template to all messages
        texts: List[str] = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in messages_batch
        ]

        # 3) Vision preprocessing for the batch (this is the critical part)
        image_inputs, video_inputs = process_vision_info(messages_batch)
        proc_out = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        instr_input_ids: torch.Tensor = proc_out["input_ids"]         # (B, L_instr)
        instr_attention: torch.Tensor = proc_out["attention_mask"]    # (B, L_instr)
        pixel_values: torch.Tensor = proc_out["pixel_values"]         # vision batch
        image_grid_thw: torch.Tensor = proc_out["image_grid_thw"]     # grid metadata

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id

        new_input_ids: List[torch.Tensor] = []
        new_attention: List[torch.Tensor] = []
        new_labels: List[torch.Tensor] = []

        # 4) For each sample, append answer tokens and build labels
        for i, answer in enumerate(answers):
            instr_ids = instr_input_ids[i]
            instr_attn = instr_attention[i]

            # Answer tokens
            resp = tokenizer(answer, add_special_tokens=False)
            resp_ids = torch.tensor(resp["input_ids"], dtype=torch.long)
            resp_attn = torch.ones_like(resp_ids, dtype=torch.long)

            # Sequence: [INSTRUCTION][ANSWER][PAD]
            ids = torch.cat(
                [instr_ids, resp_ids, torch.tensor([pad_id], dtype=torch.long)], dim=0
            )
            attn = torch.cat(
                [instr_attn, resp_attn, torch.ones(1, dtype=torch.long)], dim=0
            )

            # Loss only on answer + final pad; ignore instruction
            labels_instr = torch.full_like(instr_ids, -100)
            labels = torch.cat(
                [labels_instr, resp_ids, torch.tensor([pad_id], dtype=torch.long)], dim=0
            )

            # Truncate to max_length if needed
            if ids.size(0) > self.max_length:
                ids = ids[: self.max_length]
                attn = attn[: self.max_length]
                labels = labels[: self.max_length]

            new_input_ids.append(ids)
            new_attention.append(attn)
            new_labels.append(labels)

        # 5) Pad to a batch tensor manually (avoids tokenizer.pad API differences)
        batch_size = len(new_input_ids)
        max_len = max(t.size(0) for t in new_input_ids)

        input_ids_batch = torch.full(
            (batch_size, max_len), pad_id, dtype=torch.long
        )
        attention_batch = torch.zeros(
            (batch_size, max_len), dtype=torch.long
        )
        labels_batch = torch.full(
            (batch_size, max_len), -100, dtype=torch.long
        )

        for i, (ids, attn, lab) in enumerate(
            zip(new_input_ids, new_attention, new_labels)
        ):
            L = ids.size(0)
            input_ids_batch[i, :L] = ids
            attention_batch[i, :L] = attn
            labels_batch[i, :L] = lab

        # 6) Final batch dict
        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_batch,
            "labels": labels_batch,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


# -----------------------------
# Main training entrypoint
# -----------------------------


def main():
    ap = argparse.ArgumentParser(
        description="LoRA fine-tuning for Qwen2.5-VL on traffic captions (jsonl)."
    )
    ap.add_argument("--train", required=True, help="Train jsonl (image/prompt/answer per line)")
    ap.add_argument("--val", help="Val jsonl (optional)")
    ap.add_argument("--out", required=True, help="Output dir for LoRA")
    ap.add_argument(
        "--model_id",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Base Qwen2.5-VL model",
    )
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--ga", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument(
        "--max_length", type=int, default=1024, help="Max token length for text sequence"
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"[train_lora_qwenvl] Loading processor + model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)

    # 4-bit quantization for your 3050 Ti
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)

    # LoRA config
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(base, lora_cfg)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(
        f"trainable params: {n_trainable:,} || all params: {n_all:,} || "
        f"trainable%: {100.0 * n_trainable / n_all:.4f}"
    )

    # Datasets
    train_ds = QwenVLDataset(args.train)
    val_ds = QwenVLDataset(args.val) if args.val else None

    print(f"[train_lora_qwenvl] Train samples: {len(train_ds)}")
    if val_ds:
        print(f"[train_lora_qwenvl] Val samples:   {len(val_ds)}")

    collator = QwenCollator(processor=processor, max_length=args.max_length)

    # TrainingArguments (use new eval_strategy name)
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=args.ga,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=5,
        eval_strategy=("epoch" if val_ds is not None else "no"),
        save_strategy=("epoch" if val_ds is not None else "no"),
        remove_unused_columns=False,
        fp16=True,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=processor,
    )

    trainer.train()
    trainer.save_model(args.out)
    print("Saved LoRA to:", args.out)


if __name__ == "__main__":
    main()
