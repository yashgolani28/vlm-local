import os, json, argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class VLDataset(Dataset):
    def __init__(self, path: str, processor: AutoProcessor, max_length: int = 1024):
        self.rows = [json.loads(l) for l in open(path, "r", encoding="utf-8") if l.strip()]
        self.processor, self.max_length = processor, max_length
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["image"]).convert("RGB")
        msgs = [
          {"role":"user","content":[{"type":"image","image":img},{"type":"text","text":r["prompt"]}]},
          {"role":"assistant","content":[{"type":"text","text":r["answer"]}]},
        ]
        full = self.processor.apply_chat_template(msgs, add_generation_prompt=False,
                                                  tokenize=True, return_tensors="pt")
        input_ids = full["input_ids"][0]; attn = full["attention_mask"][0]
        # mask user
        user = self.processor.apply_chat_template(
            [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":r["prompt"]}]}],
            add_generation_prompt=True, tokenize=True, return_tensors="pt")
        plen = user["input_ids"].shape[-1]
        labels = input_ids.clone(); labels[:plen] = -100
        # clip
        ml = self.max_length
        input_ids = input_ids[:ml]; attn = attn[:ml]; labels = labels[:ml]
        pix = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return {"input_ids":input_ids, "attention_mask":attn, "labels":labels, "pixel_values":pix}

@dataclass
class Collator:
    processor: AutoProcessor
    def __call__(self, feats: List[Dict[str,Any]])->Dict[str,Any]:
        import torch
        batch = {k:[f[k] for f in feats] for k in ("input_ids","attention_mask","labels")}
        out = self.processor.tokenizer.pad(batch, padding=True, return_tensors="pt")
        out["pixel_values"] = torch.stack([f["pixel_values"] for f in feats], dim=0)
        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--ga", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_length", type=int, default=512, help="Max token length for chat template.")
    ap.add_argument("--fourbit", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    proc = AutoProcessor.from_pretrained(args.model_id)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if args.fourbit:
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )
        base = LlavaForConditionalGeneration.from_pretrained(
            args.model_id,
            device_map="auto",
            quantization_config=qconf,
        )
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    else:
        base = LlavaForConditionalGeneration.from_pretrained(
            args.model_id,
            device_map="auto",
            torch_dtype=dtype,
        )
        base.gradient_checkpointing_enable()

    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"])
    model = get_peft_model(base, lcfg)

    tr = VLDataset(args.train, proc, max_length=args.max_length)
    ev = VLDataset(args.val, proc, max_length=args.max_length) if args.val else None

    coll = Collator(proc)
    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.ga,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tr,
        eval_dataset=ev,
        data_collator=coll,
        processing_class=proc,
    )

    trainer.train(); trainer.save_model(args.out)
    print("Saved LoRA to:", args.out)

if __name__ == "__main__":
    main()
