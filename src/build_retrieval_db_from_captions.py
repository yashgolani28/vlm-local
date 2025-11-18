import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

from PIL import Image

# Allow both "python -m src.build_retrieval_db_from_captions"
# and "python src\\build_retrieval_db_from_captions.py"
try:
    from .anpr import run_anpr_multi           # when part of src package
except ImportError:
    from anpr import run_anpr_multi            # when run as a plain script

try:
    from .retrieval_db import RetrievalDB      # when part of src package
except ImportError:
    from retrieval_db import RetrievalDB       # when run as a plain script


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
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


def build_retrieval_db_from_jsonl(
    captions_jsonl: str,
    limit: int = 0,
    snapshots_dir: str = "snapshots_retrieval",
    skip_anpr: bool = False,
) -> Dict[str, int]:
    """
    Build (or extend) the retrieval DB from an auto-captioned JSONL file.

    This is the programmatic entry point used by both:
      - the CLI (python -m src.build_retrieval_db_from_captions ...)
      - the Streamlit dashboard (admin button on Search tab).

    Returns a dict with summary stats:
      {
        "total_rows": int,   # rows considered after 'limit'
        "added": int,
        "skipped": int,
        "failed": int,
      }
    """
    captions_path = Path(captions_jsonl).expanduser().resolve()
    if not captions_path.exists():
        raise FileNotFoundError(f"captions_jsonl not found: {captions_path}")

    rows = _load_jsonl(captions_path)
    if limit and limit > 0:
        rows = rows[:limit]

    print(f"Loaded {len(rows)} caption rows from {captions_path.name}")

    db = RetrievalDB()
    print(f"Using retrieval DB at: {db.path}")

    added = 0
    skipped = 0
    failed = 0

    os.makedirs(snapshots_dir, exist_ok=True)

    for idx, row in enumerate(rows, start=1):
        img_path_str = row.get("image") or row.get("image_path")
        if not img_path_str:
            skipped += 1
            continue

        img_path = Path(img_path_str).expanduser()
        # If captions stored relative paths, make them relative to jsonl dir
        if not img_path.is_absolute():
            img_path = (captions_path.parent / img_path).resolve()
        if not img_path.exists():
            print(f"[{idx}] Image missing on disk, skipping: {img_path}")
            skipped += 1
            continue

        caption = row.get("answer") or row.get("caption") or ""
        if not caption.strip():
            print(f"[{idx}] Empty caption, skipping: {img_path}")
            skipped += 1
            continue

        plates: List[str] = []
        if not skip_anpr:
            try:
                anpr_results = run_anpr_multi(
                    str(img_path),
                    save_dir=snapshots_dir,
                    ocr_min_conf=0.50,
                )
                plates = [
                    p["plate_text"]
                    for p in anpr_results
                    if p.get("plate_text")
                ]
            except Exception as e:
                print(f"[{idx}] ANPR failed for {img_path}: {e}")
                failed += 1

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[{idx}] Failed to open image {img_path}: {e}")
            failed += 1
            continue

        try:
            item_id = db.add_image_from_pil(
                img,
                caption=caption,
                plates=plates,
                source="captions_jsonl",
                extra={
                    "captions_jsonl": str(captions_path),
                    "row_index": idx - 1,
                },
            )
            added += 1
            print(
                f"[{idx}/{len(rows)}] Indexed {img_path.name} as {item_id} "
                f"(plates: {', '.join(plates) if plates else 'none'})"
            )
        except Exception as e:
            print(f"[{idx}] Failed to add to DB: {e}")
            failed += 1

    print("Done.")
    print(f"Added: {added}, skipped: {skipped}, failed: {failed}")
    print(f"Retrieval DB path: {db.path}")

    return {
        "total_rows": len(rows),
        "added": added,
        "skipped": skipped,
        "failed": failed,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Build retrieval_db.json from an auto-captioned jsonl file."
    )
    ap.add_argument(
        "--captions_jsonl",
        required=True,
        help="Path to jsonl file produced by auto_caption_traffic_folder.py",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit on number of rows to index (0 = all).",
    )
    ap.add_argument(
        "--snapshots_dir",
        default="snapshots_retrieval",
        help="Where ANPR crops will be saved (for debugging).",
    )
    ap.add_argument(
        "--skip_anpr",
        action="store_true",
        help="Skip ANPR plate detection when building DB.",
    )
    args = ap.parse_args()

    build_retrieval_db_from_jsonl(
        captions_jsonl=args.captions_jsonl,
        limit=args.limit,
        snapshots_dir=args.snapshots_dir,
        skip_anpr=args.skip_anpr,
    )


if __name__ == "__main__":
    main()
