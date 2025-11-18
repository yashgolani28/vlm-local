## 0) Install basics (Windows 11 + WSL2 recommended)
- Update NVIDIA driver (Studio) on Windows.
- Enable **WSL2** and install Ubuntu 22.04 (optional but recommended).
- Inside Windows **or** WSL:
  ```bash
  # Miniconda (recommended)
  # (Download miniconda for your OS; ensure 'conda' is on PATH)
  conda create -n vlm-local python=3.10 -y
  conda activate vlm-local

  # CUDA-enabled PyTorch (adjust cudax if needed)
  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
  # Core libs
  pip install -r requirements.txt
  # Optional: to serve the API
  pip install "uvicorn[standard]" fastapi
  ```

## 1) Model (fully local)
We use **Qwen2.5‑VL‑3B‑Instruct** with **4‑bit (bitsandbytes)** to fit low VRAM.
You can run offline after the first download by setting `HF_HUB_OFFLINE=1` and pointing to a local cache.

### Quick test
```bash
python src/infer_qwenvl.py --image assets/sample.jpg --question "Summarize the visible components and any safety warnings."
```

### Video (frame-based)
```bash
python src/infer_qwenvl.py --video assets/sample.mp4 --fps 0.5 --question "Is the belt seated correctly between 10s and 20s?"
```

## 2) Run a local API
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
# POST /analyze with JSON:
# { "question":"...", "image_path":"...", "video_path":"...", "fps":1.0 }
```

## 3) Fine‑tuning with LoRA (QLoRA style)
- Put your JSONL in `data/train.jsonl` and `data/val.jsonl` with samples like:
```json
{"image":"path/to/img.png","prompt":"Locate the M8 holes.","answer":"Two holes at 60 mm centers"}
```
- Then:
```bash
python src/train_lora_qwenvl.py --train data/train.jsonl --val data/val.jsonl --out models/qwen2_5vl3b_lora
```
This uses **grad checkpointing + 4‑bit** and **batch_size=1** so it can run on 3050 Ti; it will be slow but local and stable.

## 4) Evaluate
```bash
python eval/eval_harness.py --data data/val.jsonl --model_id Qwen/Qwen2.5-VL-3B-Instruct --lora_dir models/qwen2_5vl3b_lora
```

## 5) Notes
- For **OCR‑tight tasks**, enable Tesseract (install system binary) and flip `--use_tesseract` to true; the pipeline will combine VLM text with OCR hints.
- If VRAM still OOMs, try closing apps, lowering `--max_new_tokens`, or running with `--cpu_offload`.
- For doc/CAD, consider pre‑cropping ROIs (less pixels → less memory and faster).

Happy building!
