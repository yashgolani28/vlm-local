import os
import sys
import json
from typing import List, Dict, Tuple

import cv2
from ultralytics import YOLO

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

# plate class id in the detection model
PLATE_CLASS_ID = 0

# character mapping for plate_reco_best_v2.pt
CLASS_MAPPING = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7",
    8: "8", 9: "9", 10: "A", 11: "B", 12: "C", 13: "D", 14: "E",
    15: "F", 16: "G", 17: "H", 18: "I", 19: "J", 20: "K", 21: "L",
    22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S",
    29: "T", 30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z",
}

# minimal list of India state / UT codes used by llava_video_test.py
INDIA_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY",
    "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}

__all__ = ["INDIA_STATE_CODES", "run_anpr", "run_anpr_multi"]

# lazy-loaded models (so importing anpr.py is cheap)
_DET_MODEL = None
_CHAR_MODEL = None


def _get_default_weight_paths():
    """
    Resolve default detection & char model paths relative to repo root.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    weights_dir = os.path.join(root, "weights")

    # you already wired this and tested it
    det_path = os.path.join(weights_dir, "vehicle_det_clf_04.pt")
    char_path = os.path.join(weights_dir, "plate_reco_best_v2.pt")
    return det_path, char_path


def _get_models(det_path: str | None = None, char_path: str | None = None):
    global _DET_MODEL, _CHAR_MODEL

    if _DET_MODEL is None or _CHAR_MODEL is None:
        if det_path is None or char_path is None:
            d_default, c_default = _get_default_weight_paths()
            det_path = det_path or d_default
            char_path = char_path or c_default

        _DET_MODEL = YOLO(det_path)
        _CHAR_MODEL = YOLO(char_path)

    return _DET_MODEL, _CHAR_MODEL


# ------------------------------------------------------------------
# Geometry / NMS helpers
# ------------------------------------------------------------------


def compute_iou(box1, box2) -> float:
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, x4 - x3) * max(0, y4 - y3)

    return inter_area / (area1 + area2 - inter_area + 1e-6)


def filter_overlapping_boxes(boxes, iou_threshold: float = 0.2):
    """
    Simple NMS on an iterable of ultralytics Box objects.
    """
    boxes = sorted(boxes, key=lambda b: float(b.conf[0]), reverse=True)
    filtered = []

    while boxes:
        best = boxes.pop(0)
        filtered.append(best)
        boxes = [
            b for b in boxes
            if compute_iou(best.xyxy[0], b.xyxy[0]) < iou_threshold
        ]

    return filtered


# ------------------------------------------------------------------
# Character recognition on a single plate crop
# ------------------------------------------------------------------


def _recognize_characters(
    cropped_plate,
    char_model,
    save_path: str,
    conf_char: float = 0.25,
) -> Tuple[str, float]:
    """
    Run the char-detector YOLO model on a cropped plate.
    Returns:
      - plate string ("" if none)
      - aggregated confidence (mean char conf)
    Also saves an annotated crop to `save_path`.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    res = char_model.predict(
        cropped_plate,
        conf=conf_char,
        verbose=False,
    )[0]

    if not len(res.boxes):
        cv2.imwrite(save_path, cropped_plate)
        return "", 0.0

    boxes = filter_overlapping_boxes(res.boxes, iou_threshold=0.2)
    boxes = sorted(boxes, key=lambda b: b.xyxy[0][0])  # left→right

    chars: list[str] = []
    confs: list[float] = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_idx = int(box.cls[0])
        ch = CLASS_MAPPING.get(cls_idx, "")
        if not ch:
            continue

        cv2.putText(
            cropped_plate,
            ch,
            (x1, max(0, y2 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        chars.append(ch)
        confs.append(float(box.conf[0]))

    plate_str = "".join(chars)
    plate_conf = float(sum(confs) / len(confs)) if confs else 0.0

    cv2.imwrite(save_path, cropped_plate)
    return plate_str, plate_conf


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def run_anpr_multi(
    image_path: str,
    roi=None,
    save_dir: str = "snapshots",
    det_conf: float = 0.25,
    char_conf: float = 0.25,
    ocr_min_conf: float = 0.0,
) -> List[Dict]:
    """
    Run multi-plate ANPR on a single image.

    Args:
        image_path: path to full frame image.
        roi: optional [x1, y1, x2, y2] to restrict detection (e.g. vehicle bbox).
        save_dir: where to save per-plate crops.
        det_conf: YOLO plate-detector confidence threshold.
        char_conf: char-detector confidence threshold.
        ocr_min_conf: minimum aggregated plate_conf to keep a plate.

    Returns a list of dicts:
        {
          "plate_text": "JK05K5336",
          "plate_conf": 0.93,
          "plate_bbox": [x1, y1, x2, y2],  # in full-frame coords
          "crop_path": "snapshots/sample_plate_0.jpg",
        }
    """
    det_model, char_model = _get_models()

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    H, W = img.shape[:2]
    stem = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(save_dir, exist_ok=True)

    # Optional ROI crop
    search_img = img
    off_x, off_y = 0, 0
    if roi is not None and isinstance(roi, (list, tuple)) and len(roi) == 4:
        rx1, ry1, rx2, ry2 = map(int, roi)
        rx1 = max(0, min(W - 1, rx1))
        rx2 = max(0, min(W, rx2))
        ry1 = max(0, min(H - 1, ry1))
        ry2 = max(0, min(H, ry2))
        if rx2 > rx1 and ry2 > ry1:
            search_img = img[ry1:ry2, rx1:rx2]
            off_x, off_y = rx1, ry1

    res = det_model.predict(search_img, conf=det_conf, verbose=False)[0]
    boxes = [b for b in res.boxes if int(b.cls[0]) == PLATE_CLASS_ID]
    boxes = filter_overlapping_boxes(boxes, iou_threshold=0.3)

    outputs: List[Dict] = []
    plate_idx = 0
    min_conf_thr = float(ocr_min_conf or 0.0)

    for box in boxes:
        sx1, sy1, sx2, sy2 = map(int, box.xyxy[0])

        gx1 = max(0, min(W - 1, sx1 + off_x))
        gx2 = max(0, min(W, sx2 + off_x))
        gy1 = max(0, min(H - 1, sy1 + off_y))
        gy2 = max(0, min(H, sy2 + off_y))
        if gx2 <= gx1 or gy2 <= gy1:
            continue

        crop = img[gy1:gy2, gx1:gx2]
        crop_name = f"{stem}_plate_{plate_idx}.jpg"
        crop_path = os.path.join(save_dir, crop_name)

        plate_str, plate_conf = _recognize_characters(
            crop,
            char_model,
            crop_path,
            conf_char=char_conf,
        )
        if not plate_str or plate_conf < min_conf_thr:
            continue

        outputs.append(
            {
                "plate_text": plate_str,
                "plate_conf": float(plate_conf),
                "plate_bbox": [int(gx1), int(gy1), int(gx2), int(gy2)],
                "crop_path": crop_path,
            }
        )
        plate_idx += 1

    return outputs


def run_anpr(
    image_path: str,
    roi=None,
    save_dir: str = "snapshots",
    det_conf: float = 0.25,
    char_conf: float = 0.25,
    ocr_min_conf: float = 0.0,
) -> Dict:
    """
    Single-plate convenience wrapper:
    returns the best plate (highest confidence) or an empty result.
    """
    plates = run_anpr_multi(
        image_path=image_path,
        roi=roi,
        save_dir=save_dir,
        det_conf=det_conf,
        char_conf=char_conf,
        ocr_min_conf=ocr_min_conf,
    )
    if not plates:
        return {
            "plate_text": "",
            "plate_conf": 0.0,
            "plate_bbox": None,
            "crop_path": None,
        }

    best = max(plates, key=lambda p: p["plate_conf"])
    return best


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def _cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="YOLO-based ANPR image test (multi-plate)."
    )
    parser.add_argument(
        "image",
        help="Path to the input image (e.g. assets/sample.jpg)",
    )
    parser.add_argument(
        "--save-dir",
        default="snapshots",
        help="Directory to save plate crops (default: snapshots)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Return only the best plate instead of all plates.",
    )
    parser.add_argument(
        "--det-conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold for plate detector.",
    )
    parser.add_argument(
        "--char-conf",
        type=float,
        default=0.25,
        help="Character detector confidence threshold.",
    )
    parser.add_argument(
        "--ocr-min-conf",
        type=float,
        default=0.0,
        help="Minimum aggregated plate confidence to keep a plate.",
    )

    args = parser.parse_args()

    image_path = os.path.abspath(args.image)

    if args.single:
        out = run_anpr(
            image_path=image_path,
            roi=None,
            save_dir=args.save_dir,
            det_conf=args.det_conf,
            char_conf=args.char_conf,
            ocr_min_conf=args.ocr_min_conf,
        )
        print(json.dumps(out, indent=2))
    else:
        plates = run_anpr_multi(
            image_path=image_path,
            roi=None,
            save_dir=args.save_dir,
            det_conf=args.det_conf,
            char_conf=args.char_conf,
            ocr_min_conf=args.ocr_min_conf,
        )
        print(json.dumps({"plates": plates}, indent=2))


if __name__ == "__main__":
    _cli()
