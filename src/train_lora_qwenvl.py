import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

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
        return self.rows[idx]


@dataclass
class QwenCollator:
    processor: AutoProcessor
    max_length: int = 1024

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        features: list of { "image", "prompt", "answer" }
        """
        tokenizer = self.processor.tokenizer

        # 1) Messages + answers
        messages_batch: List[List[Dict[str, Any]]] = []
        answers: List[str] = []

        for f in features:
            img_path = f["image"]
            prompt = f["prompt"]
            answer = f["answer"]

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

        # 2) Chat template
        texts: List[str] = [
            self.processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in messages_batch
        ]

        # 3) Vision preprocessing
        image_inputs, video_inputs = process_vision_info(messages_batch)
        proc_out = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        instr_input_ids: torch.Tensor = proc_out["input_ids"]
        instr_attention: torch.Tensor = proc_out["attention_mask"]
        pixel_values: torch.Tensor = proc_out["pixel_values"]
        image_grid_thw: torch.Tensor = proc_out["image_grid_thw"]

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id

        new_input_ids: List[torch.Tensor] = []
        new_attention: List[torch.Tensor] = []
        new_labels: List[torch.Tensor] = []

        # 4) Append answer tokens and build labels
        for i, answer in enumerate(answers):
            instr_ids = instr_input_ids[i]
            instr_attn = instr_attention[i]

            resp = tokenizer(answer, add_special_tokens=False)
            resp_ids = torch.tensor(resp["input_ids"], dtype=torch.long)
            resp_attn = torch.ones_like(resp_ids, dtype=torch.long)

            ids = torch.cat(
                [instr_ids, resp_ids, torch.tensor([pad_id], dtype=torch.long)],
                dim=0,
            )
            attn = torch.cat(
                [instr_attn, resp_attn, torch.ones(1, dtype=torch.long)],
                dim=0,
            )

            labels_instr = torch.full_like(instr_ids, -100)
            labels = torch.cat(
                [labels_instr, resp_ids, torch.tensor([pad_id], dtype=torch.long)],
                dim=0,
            )

            if ids.size(0) > self.max_length:
                ids = ids[: self.max_length]
                attn = attn[: self.max_length]
                labels = labels[: self.max_length]

            new_input_ids.append(ids)
            new_attention.append(attn)
            new_labels.append(labels)

        # 5) Manual padding
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

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_batch,
            "labels": labels_batch,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


def _run_from_args(args: argparse.Namespace) -> str:
    os.makedirs(args.out, exist_ok=True)

    print(f"[train_lora_qwenvl] Loading processor + model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=True)

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
    if hasattr(base, "config"):
        base.config.use_cache = False

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

    train_ds = QwenVLDataset(args.train)
    val_ds = QwenVLDataset(args.val) if args.val else None

    print(f"[train_lora_qwenvl] Train samples: {len(train_ds)}")
    if val_ds:
        print(f"[train_lora_qwenvl] Val samples:   {len(val_ds)}")

    collator = QwenCollator(processor=processor, max_length=args.max_length)

    # Avoid passing evaluation_strategy/save_strategy so this works with a
    # wider range of Transformers versions. We skip scheduled evaluation.
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=args.ga,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=5,
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
        tokenizer=processor.tokenizer,
    )

    trainer.train()
    trainer.save_model(args.out)
    print("Saved Qwen LoRA to:", args.out)
    return args.out


def run_qwenvl_lora(
    train_path: str,
    out_dir: str,
    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    val_path: Optional[str] = None,
    epochs: int = 1,
    bs: int = 1,
    ga: int = 8,
    lr: float = 2e-4,
    max_length: int = 1024,
) -> str:
    """
    Programmatic entry point used by Streamlit dashboard.
    """
    args = argparse.Namespace(
        train=train_path,
        val=val_path,
        out=out_dir,
        model_id=model_id,
        epochs=epochs,
        bs=bs,
        ga=ga,
        lr=lr,
        max_length=max_length,
    )
    return _run_from_args(args)


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
    _run_from_args(args)


if __name__ == "__main__":
    main()
