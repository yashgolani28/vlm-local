
````markdown
# vlm-local – Traffic VLM + ANPR + Retrieval

Local visual–language pipeline for **traffic CCTV analysis**, built around:

- **Qwen2.5-VL 7B Instruct** (Hugging Face) + LoRA for traffic-tuned captions & OCR  
- **LLaVA 13B** behind **Ollama** for generic scene understanding & auto-captions  
- Custom **ANPR** (YOLO-based detector + recogniser)  
- A lightweight **retrieval DB** (semantic + plate search)  
- A **Streamlit dashboard** for:
  - image / video playground
  - live RTSP camera
  - semantic + plate search
  - dataset building & one-click LoRA training
  - simple eval of trained models

Designed to run fully offline on a single GPU box (tested on **RTX 4060 Ti 16 GB**).

---

## 1. Quick setup (copy–paste)

From a fresh machine / repo clone:
````
```bash
# clone
git clone https://github.com/yashgolani28/vlm-local.git
cd vlm-local
```
````
# create venv
python -m venv .venv

# activate venv
#   Windows PowerShell:
.venv\Scripts\Activate
#   (Linux / macOS):
# source .venv/bin/activate

# upgrade pip
pip install --upgrade pip

# 1) Install PyTorch with CUDA (adjust for your CUDA version from pytorch.org)
# Example for CUDA 12.4:
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2) Install rest of dependencies
pip install -r requirements.txt
````

### 1.1 Install & test Ollama + LLaVA 13B

```bash
# install Ollama from https://ollama.com (GUI / package)
# then pull the model:
ollama pull llava:13b

# quick sanity test:
ollama run llava:13b
```

Defaults (you can override via env vars):

* `OLLAMA_URL=http://127.0.0.1:11434/api/chat`
* `OLLAMA_MODEL_NAME=llava:13b`

### 1.2 Qwen2.5-VL 7B Instruct (Hugging Face)

The pipeline expects:

* Model ID: **`Qwen/Qwen2.5-VL-7B-Instruct`**
* You may need a **Hugging Face access token** in `HF_TOKEN` if the model is gated.

```bash
# example (PowerShell / bash)
set HF_TOKEN=your_hf_token_here
# or export HF_TOKEN=...
```

Optional env vars the dashboard understands:

```bash
# which VLM to use by default in the UI
set VLM_BACKEND=qwenvl           # or ollama_llava

# Qwen model + LoRA dir (once trained)
set VLM_MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct
set VLM_LORA_DIR=outputs\qwenvl_lora

# whether to use VLM for plate OCR on ANPR crops
set USE_VLM_OCR=1
set PLATE_OCR_BACKEND=qwenvl     # or ollama_llava
```

### 1.3 ANPR weights

Place your weights under `weights/`:

* `vehicle_det_clf_04.pt`          – your vehicle/plate detector
* `plate_reco_best_v2.pt`          – your plate character recogniser
* `yolov8n.pt` / `yolov8s.pt`      – YOLO base model(s) used by `anpr.py`

`anpr.py` and `vlm_dashboard.py` resolve them from `weights/` relative to repo root.

---

## 2. Project layout

```text
vlm-local/
├─ configs/
│  └─ config.yaml                   # basic config (paths, timings, etc.)
├─ data/
│  ├─ retrieval_images/             # images indexed into retrieval DB
│  ├─ retrieval_videos/             # videos indexed into retrieval DB
│  ├─ snapshots/                    # general snapshots (dashboard)
│  ├─ snapshots_retrieval/          # snapshots specifically for retrieval
│  ├─ traffic_images/               # raw traffic frames / dataset images
│  └─ traffic_captions.jsonl        # auto-captioned dataset (LLaVA / Qwen)
├─ eval/
│  ├─ eval_llava_harness.py         # HF LLaVA (7B) eval + optional LoRA
│  └─ eval_qwenvl_lora.py           # Qwen2.5-VL 7B LoRA eval
├─ outputs/
│  ├─ llava_traffic_lora/           # LLaVA LoRA adapters (HF)
│  └─ qwenvl_lora/                  # Qwen LoRA adapters
├─ prompts/
│  └─ traffic_caption.txt           # prompt template for traffic captions
├─ snapshots/                       # extra snapshots (UI)
├─ weights/
│  ├─ plate_reco_best_v2.pt
│  ├─ vehicle_det_clf_04.pt
│  └─ yolov8s.pt / yolov8n.pt
└─ src/
   ├─ anpr.py                       # YOLO-based ANPR (multi-plate)
   ├─ auto_caption_traffic_folder.py
   ├─ build_retrieval_db_from_captions.py
   ├─ convert_traffic_vqa_to_llava.py
   ├─ llava_live_cam_test.py
   ├─ llava_video_test.py
   ├─ retrieval_db.py               # embeddings + semantic/plate search
   ├─ train_lora_llava.py
   ├─ train_lora_qwenvl.py
   └─ vlm_dashboard.py              # Streamlit app (main entry point)
```

---

## 3. Running the dashboard

From repo root (with venv active and Ollama running):

```bash
streamlit run src/vlm_dashboard.py
```

In the **Streamlit sidebar**, you can:

* choose backend: **Qwen2.5-VL 7B + LoRA** or **Ollama LLaVA 13B**
* select ANPR options (YOLO weight, confidence, NMS),
* configure live RTSP URL, analysis interval, retrieval options, etc.

Main tabs:

1. **Playground (Image / Video)**

   * Upload one image or a short video.
   * Run VLM (Qwen or LLaVA via Ollama).
   * Run ANPR on same frames, show plates + state guesses.
   * See a timing breakdown (`VLM`, `ANPR`, `VLM_OCR`).

2. **Live Camera**

   * Point to RTSP / HTTP stream (e.g. Axis M1125).
   * Periodic analysis (e.g. every 10 s) with VLM + ANPR.
   * Keeps a short history and snapshot gallery.

3. **Search / Retrieval**

   * Semantic search over captions (sentence-transformers).
   * Plate search by text / prefix (e.g. `DL9C`, `MH12`).
   * Rebuild retrieval DB from JSONL of captions.

4. **Dataset & Training**

   * Auto-caption images to create `traffic_captions.jsonl`.
   * Build visual instruction data for Qwen/LLaVA.
   * One-click trigger of `train_lora_llava.py` and `train_lora_qwenvl.py`.
   * Quick manual evaluation: upload image, type ideal answer, compare with model.

---

## 4. Auto-caption & Retrieval DB

### 4.1 Auto-caption a folder using LLaVA 13B (Ollama)

```bash
# caption images under data/traffic_images using LLaVA 13B via Ollama
python src/auto_caption_traffic_folder.py ^
  --images_dir data/traffic_images ^
  --out data/traffic_captions.jsonl ^
  --ollama_url http://127.0.0.1:11434/api/chat ^
  --model_name llava:13b
```

This writes JSONL with rows:

```json
{"image": "D:/.../frame_001.jpg", "prompt": "...", "answer": "..."}
```

### 4.2 Build the retrieval DB

```bash
python src/build_retrieval_db_from_captions.py ^
  --captions_jsonl data/traffic_captions.jsonl ^
  --snapshots_dir data/snapshots_retrieval
```

This:

* populates a simple JSON-based DB in `data/`
* precomputes sentence-transformer embeddings for captions
* also stores plate info so you can search by registration number.

The **Search / Retrieval** tab in the dashboard uses this DB.

---

## 5. Training LoRA adapters

### 5.1 LLaVA-HF 7B LoRA (optional HF path)

If you have GPU VRAM for **HF LLaVA 7B**, you can fine-tune it on your captions JSONL:

```bash
python src/train_lora_llava.py ^
  --train "data/traffic_captions.jsonl" ^
  --out "outputs/llava_traffic_lora" ^
  --model_id "llava-hf/llava-1.5-7b-hf" ^
  --epochs 1 ^
  --bs 1 ^
  --ga 16 ^
  --lr 2e-4 ^
  --max_length 512 ^
  --fourbit
```

> Note: this is **HF LLaVA 7B**, separate from the Ollama `llava:13b` that the dashboard calls.

### 5.2 Qwen2.5-VL 7B Instruct LoRA (main path)

This is the main traffic-tuned model used by the dashboard when `VLM_BACKEND=qwenvl`:

```bash
python src/train_lora_qwenvl.py ^
  --train "data/traffic_captions.jsonl" ^
  --out "outputs/qwenvl_lora" ^
  --model_id "Qwen/Qwen2.5-VL-7B-Instruct" ^
  --epochs 1 ^
  --bs 1 ^
  --ga 8 ^
  --lr 2e-4 ^
  --max_length 1024
```

Once finished:

* set `VLM_LORA_DIR=outputs\qwenvl_lora` (or use the dashboard’s field for LoRA dir),
* restart the dashboard and choose **Qwen2.5-VL + LoRA** in sidebar.

---

## 6. Evaluating trained models (offline)

All eval scripts expect JSONL with fields: **`image`**, **`prompt`**, **`answer`**.

### 6.1 Qwen2.5-VL 7B + LoRA

```bash
python eval/eval_qwenvl_lora.py ^
  --model_id "Qwen/Qwen2.5-VL-7B-Instruct" ^
  --lora_dir "outputs/qwenvl_lora" ^
  --data "data/traffic_captions.jsonl" ^
  --num_samples 8 ^
  --max_new_tokens 256
```

This will:

* load Qwen2.5-VL 7B in 4-bit via `BitsAndBytesConfig`
* attach your LoRA adapter
* print a handful of sample generations
* compute a crude sanity metric: how often the ground-truth `answer` appears as a substring of the prediction.

You can also run it **without** `--lora_dir` (after making `lora_dir` optional in the script) to compare **base vs LoRA**.

### 6.2 LLaVA-HF 7B + LoRA (if used)

```bash
python eval/eval_llava_harness.py ^
  --data "data/traffic_captions.jsonl" ^
  --model_id "llava-hf/llava-1.5-7b-hf" ^
  --lora_dir "outputs/llava_traffic_lora" ^
  --num_samples 8
```

This will:

* load HF LLaVA 7B + optional LoRA
* print Pred vs Gold examples for visual inspection.

For your **actual Ollama `llava:13b`**, use:

* `auto_caption_traffic_folder.py` for batch captions, and/or
* the dashboard’s **manual eval helper** to compare model answers vs your ideal captions interactively.

---

## 7. requirements.txt (for reference)

The repo uses the following Python dependencies:

```text
numpy
opencv-python
Pillow
requests
streamlit
ultralytics
sentence-transformers
torch        # install from PyTorch's CUDA wheel index as shown above
transformers
peft
accelerate
bitsandbytes
tqdm
```

> On Windows, `bitsandbytes` may have limited support. If installation fails, you can:
>
> * train / run QLoRA on a Linux box, or
> * disable 4-bit paths in training/eval scripts and run full-precision instead.

---

## 8. Notes

* Large artifacts (weights, LoRA outputs, raw media, snapshots) are ignored in git via `.gitignore` so the GitHub repo stays light.
* The code is structured to be **local-first**: nothing calls any external API except Hugging Face model downloads and Ollama.
* For big datasets, consider splitting `traffic_captions.jsonl` into train / val and pointing the eval scripts at the val split.

```
