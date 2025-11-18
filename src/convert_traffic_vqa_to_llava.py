import os
import json
import argparse
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RAW_DATASET = os.path.join(DATA_DIR, "traffic_vqa.jsonl")


def _load_raw(path: str):
    rows = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw dataset not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            rows.append(obj)
    return rows


def _convert(rows):
    """
    Convert dashboard-style records into:
      {"image": <abs_path>, "prompt": <instr>, "answer": <ideal_answer>}
    """
    out = []
    for r in rows:
        img_rel = r.get("image")
        if not img_rel:
            continue

        # image is stored relative to data/ (e.g. "traffic_images/..")
        img_path = os.path.join(DATA_DIR, img_rel.replace("/", os.sep))
        conv = r.get("conversations", [])
        if len(conv) < 2:
            continue

        instr = (conv[0].get("value") or "").strip()
        ans = (conv[1].get("value") or "").strip()
        if not instr or not ans:
            continue

        out.append(
            {
                "image": os.path.abspath(img_path),
                "prompt": instr,
                "answer": ans,
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Convert traffic_vqa.jsonl -> train.jsonl / val.jsonl for LLaVA LoRA."
    )
    ap.add_argument("--val_frac", type=float, default=0.1, help="Fraction for validation set.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--train_out",
        default=os.path.join(DATA_DIR, "train.jsonl"),
        help="Output path for train jsonl.",
    )
    ap.add_argument(
        "--val_out",
        default=os.path.join(DATA_DIR, "val.jsonl"),
        help="Output path for val jsonl.",
    )
    args = ap.parse_args()

    rows = _load_raw(RAW_DATASET)
    conv_rows = _convert(rows)
    if not conv_rows:
        raise SystemExit("No valid examples found in traffic_vqa.jsonl.")

    random.seed(args.seed)
    random.shuffle(conv_rows)

    n_total = len(conv_rows)
    n_val = int(max(1, round(n_total * args.val_frac))) if n_total > 1 else 0
    val_rows = conv_rows[:n_val] if n_val > 0 else []
    train_rows = conv_rows[n_val:] if n_val > 0 else conv_rows

    def _write(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    _write(args.train_out, train_rows)
    if val_rows:
        _write(args.val_out, val_rows)

    print(f"Total usable examples: {n_total}")
    print(f"Train examples: {len(train_rows)} -> {args.train_out}")
    print(f"Val examples:   {len(val_rows)} -> {args.val_out}")


if __name__ == "__main__":
    main()
