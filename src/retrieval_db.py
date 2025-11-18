import os
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Try to import sentence-transformers for dense embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # graceful fallback; we'll raise when actually used
    SentenceTransformer = None  # type: ignore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

RETR_DB_PATH = os.path.join(DATA_DIR, "retrieval_db.json")
RETR_IMG_DIR = os.path.join(DATA_DIR, "retrieval_images")
RETR_VIDEO_DIR = os.path.join(DATA_DIR, "retrieval_videos")

os.makedirs(RETR_IMG_DIR, exist_ok=True)
os.makedirs(RETR_VIDEO_DIR, exist_ok=True)


def _normalize_plate(p: str) -> str:
    """
    Normalise license plate strings for comparison.
    Uppercase + remove spaces and common separators.
    """
    if not p:
        return ""
    p = str(p).upper()
    for ch in (" ", "-", "_", "."):
        p = p.replace(ch, "")
    return p


# --- tiny domain-specific synonym expansion for traffic captions --- #
_SYNONYMS = {
    "car": [
        "car",
        "cars",
        "vehicle",
        "vehicles",
        "four wheeler",
        "four-wheeler",
    ],
    "bike": [
        "bike",
        "bikes",
        "motorcycle",
        "motorcycles",
        "two wheeler",
        "two-wheeler",
        "scooter",
        "scooters",
    ],
    "truck": [
        "truck",
        "trucks",
        "lorry",
        "lorries",
        "goods vehicle",
        "goods vehicles",
    ],
    "bus": [
        "bus",
        "buses",
        "coach",
        "coaches",
    ],
    "auto": [
        "auto",
        "auto rickshaw",
        "autorickshaw",
        "three wheeler",
        "three-wheeler",
    ],
    "pedestrian": [
        "pedestrian",
        "pedestrians",
        "people",
        "person",
        "crowd",
        "crowded",
    ],
    "junction": [
        "junction",
        "intersection",
        "crossroads",
        "cross road",
    ],
}


def _expand_query(query: str) -> str:
    """
    Expand short user queries with simple traffic synonyms so that
    the embedding sees more related tokens for small vocab queries.
    """
    q = (query or "").strip()
    if not q:
        return ""

    q_l = q.lower()
    extra_tokens: List[str] = []

    for key, syns in _SYNONYMS.items():
        if key in q_l:
            extra_tokens.extend(syns)

    if not extra_tokens:
        return q

    # Example: "white car" -> "white car car cars vehicle vehicles ..."
    return q + " " + " ".join(extra_tokens)


# ----------------------------------------------------------------------
# Embedding backend (SentenceTransformer)
# ----------------------------------------------------------------------
_EMBED_MODEL: Optional[Any] = None

def _get_embedder() -> Any:
    """
    Lazy-load a sentence-transformer model for dense embeddings.
    """
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL

    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is required for RetrievalDB embeddings. "
            "Install with: pip install sentence-transformers"
        )

    model_name = os.getenv(
        "RETRIEVAL_EMBED_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    _EMBED_MODEL = SentenceTransformer(model_name)
    return _EMBED_MODEL


def _embed_texts(texts: List[str]) -> np.ndarray:
    """
    Encode a list of texts into L2-normalized dense vectors.
    Shape: (N, D)
    """
    texts = [t if str(t).strip() else "[EMPTY]" for t in texts]
    model = _get_embedder()
    emb = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so dot product == cosine similarity
    )
    # use float32 to save RAM
    return emb.astype("float32")


class RetrievalDB:
    """
    Lightweight retrieval DB over (image / video) items.

    Embeddings are computed using a sentence-transformer model over
    captions + plate texts. No external service / internet required;
    everything is local to this machine.
    """

    def __init__(self, path: str = RETR_DB_PATH):
        self.path = path
        self.items: Dict[str, Dict[str, Any]] = {}

        # In-memory retrieval structures
        self._id_list: List[str] = []
        self._texts: List[str] = []
        # Dense embedding matrix (N, D), L2-normalized
        self._X: Optional[np.ndarray] = None

        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "items" in data:
                    self.items = data.get("items", {})
                else:
                    # backwards-compatible if it was just a dict
                    self.items = data
            except Exception:
                self.items = {}
        else:
            self.items = {}
        self._rebuild_matrix()

    def _save(self):
        tmp_path = self.path + ".tmp"
        data = {"items": self.items}
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.path)

    # ------------------------------------------------------------------
    # Text utilities
    # ------------------------------------------------------------------
    def _item_text(self, item: Dict[str, Any]) -> str:
        """
        Build the text used for retrieval for a single item:
        caption + normalised plates.
        """
        caption = item.get("caption") or ""
        plates = item.get("plates") or []
        if plates:
            caption = f"{caption} Plates: {' '.join(plates)}"
        return caption.strip()

    def _rebuild_matrix(self):
        """
        Build a dense embedding matrix over all items using a
        sentence-transformer model.

        Called automatically whenever items are added or DB is loaded.
        """
        docs: List[str] = []
        ids: List[str] = []

        for item_id, item in self.items.items():
            text = self._item_text(item)
            if not text:
                continue
            ids.append(item_id)
            docs.append(text)

        if not docs:
            self._id_list = []
            self._texts = []
            self._X = None
            return

        X = _embed_texts(docs)  # (N, D)

        self._id_list = ids
        self._texts = docs
        self._X = X

    # ------------------------------------------------------------------
    # Ingestion helpers
    # ------------------------------------------------------------------
    def add_image_from_pil(
        self,
        img,
        caption: str,
        plates: Optional[List[str]] = None,
        source: str = "playground_image",
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a PIL image into the retrieval image folder and index it.

        Returns the new item_id.
        """
        from PIL import Image  # lazy import

        if not isinstance(img, Image.Image):
            raise TypeError("img must be a PIL.Image.Image")

        plates = plates or []
        norm_plates = [p for p in {_normalize_plate(p) for p in plates} if p]
        extra = extra or {}

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        item_id = f"img_{ts}"
        file_name = f"{item_id}.jpg"
        abs_path = os.path.join(RETR_IMG_DIR, file_name)

        img.save(abs_path, format="JPEG")

        self.items[item_id] = {
            "id": item_id,
            "type": "image",
            "file_path": abs_path,
            "source": source,
            "caption": caption,
            "plates": norm_plates,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "extra": extra,
        }
        self._rebuild_matrix()
        self._save()
        return item_id

    def add_video_bytes(
        self,
        data: bytes,
        orig_name: str,
        caption: str,
        plates: Optional[List[str]] = None,
        source: str = "playground_video",
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Persist an uploaded video (raw bytes) and index it as a single item.

        We don't segment the clip – the caption should describe the overall content.
        """
        plates = plates or []
        norm_plates = [p for p in {_normalize_plate(p) for p in plates} if p]
        extra = extra or {}

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ext = os.path.splitext(orig_name)[1].lower() or ".mp4"
        item_id = f"vid_{ts}"
        file_name = f"{item_id}{ext}"
        abs_path = os.path.join(RETR_VIDEO_DIR, file_name)

        with open(abs_path, "wb") as f:
            f.write(data)

        self.items[item_id] = {
            "id": item_id,
            "type": "video",
            "file_path": abs_path,
            "original_name": orig_name,
            "source": source,
            "caption": caption,
            "plates": norm_plates,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "extra": extra,
        }
        self._rebuild_matrix()
        self._save()
        return item_id

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        num_images = sum(1 for it in self.items.values() if it.get("type") == "image")
        num_videos = sum(1 for it in self.items.values() if it.get("type") == "video")
        return {
            "total_items": len(self.items),
            "num_images": num_images,
            "num_videos": num_videos,
            "db_path": self.path,
        }

    def export_embeddings(self, out_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Export the current embedding matrix and id list to disk for inspection.

        Writes:
          - <out_dir>/retrieval_embeddings.npy  (float32, shape (N, D))
          - <out_dir>/retrieval_ids.json       (list of item_ids in the same order)

        Returns a dict with the file paths.
        """
        if self._X is None or not self._id_list:
            raise RuntimeError("No embeddings available to export (DB is empty?).")

        # Default to the same folder that holds retrieval_db.json
        out_dir = out_dir or os.path.dirname(self.path)
        os.makedirs(out_dir, exist_ok=True)

        emb_path = os.path.join(out_dir, "retrieval_embeddings.npy")
        ids_path = os.path.join(out_dir, "retrieval_ids.json")

        # Save as float32 to keep size reasonable
        np.save(emb_path, self._X.astype("float32"))
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(self._id_list, f, indent=2, ensure_ascii=False)

        return {"embeddings": emb_path, "ids": ids_path}

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _iter_filtered_items(self, media_types: Optional[List[str]]):
        if not media_types:
            yield from self.items.values()
        else:
            allowed = set(media_types)
            for it in self.items.values():
                if it.get("type") in allowed:
                    yield it

    def search_semantic(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.25,
        media_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over captions using sentence-transformer embeddings.

        Returns a list of item dicts augmented with 'score' (0–100).

        Behaviour:
        - Expand the query with simple traffic synonyms.
        - Encode query and items into dense embeddings.
        - First, apply the user-provided min_score threshold.
        - If that yields no results, fall back to "best match" mode
          (ignore threshold and just return top_k highest sims > 0).
        """
        query = (query or "").strip()
        if not query:
            return []
        if self._X is None or not len(self._id_list):
            return []

        # synonym-expanded query for better overlap
        expanded_query = _expand_query(query)
        q_emb = _embed_texts([expanded_query])[0]  # (D,)

        # self._X is (N, D), q_emb is (D,) and both are L2-normalized
        sims = np.dot(self._X, q_emb)  # cosine similarities in [-1, 1]

        scored: List[Tuple[float, Dict[str, Any]]] = []

        # 1) normal path: honor min_score
        for idx, item_id in enumerate(self._id_list):
            item = self.items.get(item_id)
            if item is None:
                continue
            if media_types and item.get("type") not in media_types:
                continue

            score = float(sims[idx])
            if score < float(min_score):
                continue

            scored.append((score, item))

        # 2) if nothing passes the threshold, fall back to "best we have"
        if not scored:
            for idx, item_id in enumerate(self._id_list):
                item = self.items.get(item_id)
                if item is None:
                    continue
                if media_types and item.get("type") not in media_types:
                    continue

                score = float(sims[idx])
                if score <= 0.0:
                    continue
                scored.append((score, item))

        if not scored:
            return []

        scored.sort(key=lambda t: t[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for score, item in scored[:top_k]:
            copy = dict(item)
            # convert cosine similarity [-1, 1] to a 0–100% style score (clip at 0)
            copy["score"] = round(max(score, 0.0) * 100.0, 2)
            out.append(copy)
        return out

    def search_plate(
        self,
        plate_query: str,
        top_k: int = 10,
        media_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Substring-based licence plate search, case-insensitive, using the
        pre-extracted ANPR plate texts stored in each item.
        """
        pq = _normalize_plate(plate_query)
        if not pq:
            return []

        results: List[Tuple[float, Dict[str, Any]]] = []

        for item in self._iter_filtered_items(media_types):
            plates = [p for p in item.get("plates", []) if p]
            best_score = 0.0
            for p in plates:
                if pq in p:
                    s = len(pq) / max(len(p), 1)
                    if s > best_score:
                        best_score = s
            if best_score > 0:
                results.append((best_score, item))

        if not results:
            return []

        results.sort(key=lambda t: t[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for score, item in results[:top_k]:
            copy = dict(item)
            copy["score"] = round(float(score) * 100.0, 2)
            out.append(copy)
        return out
