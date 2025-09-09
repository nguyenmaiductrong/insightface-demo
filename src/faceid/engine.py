from typing import Optional, List, Dict, Tuple
from pathlib import Path
import json
import numpy as np
import faiss

from .utils import (
    load_cfg, get_app, detect_faces, face_to_embedding,
    l2_normalize, cosine_similarity
)

class FaceEngine:
    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or load_cfg("configs/default.yaml")
        self.threshold = float(self.cfg.get("threshold"))
        self.paths = self.cfg.get("paths")
        self.app = get_app(self.cfg)

        self._library_loaded = False
        self._library_index = None
        self._library_meta: Optional[List[dict]] = None

        self._facebank_loaded = False
        self._facebank_index = None
        self._facebank_labels: Optional[List[str]] = None

    def embed_first_face(self, img_bgr) -> Optional[np.ndarray]:
        faces = detect_faces(self.app, img_bgr)
        if not faces:
            return None
        emb = face_to_embedding(faces[0])
        if emb is None:
            return None
        return emb.astype(np.float32).reshape(-1)

    def verify(self, imgA_bgr: np.ndarray, imgB_bgr: np.ndarray) -> Tuple[Optional[float], bool]:
        e1 = self.embed_first_face(imgA_bgr)
        e2 = self.embed_first_face(imgB_bgr)
        if e1 is None or e2 is None:
            return None, False
        e1 = l2_normalize(e1.reshape(1, -1))[0]
        e2 = l2_normalize(e2.reshape(1, -1))[0]
        score = float(cosine_similarity(e1, e2))
        return score, bool(score >= self.threshold)

    def ensure_faiss(self):
        if self._library_loaded and self._library_index is not None and self._library_meta is not None:
            return

        index_path = Path(self.cfg.get("library_index", "outputs/library.index"))
        meta_path  = Path(self.cfg.get("library_meta",  "outputs/library_meta.json"))
        if not index_path.exists() or not meta_path.exists():
            self._library_loaded = True
            self._library_index = None
            self._library_meta = []
            return
        self._library_index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self._library_meta = json.load(f)
        self._library_loaded = True

    def search_by_embedding(self, emb: np.ndarray, k: int = 10) -> List[dict]:
        self.ensure_faiss()
        if self._library_index is None or self._library_meta is None:
            return []
        q = emb.astype(np.float32).reshape(1, -1)
        q = l2_normalize(q)
        scores, idxs = self._library_index.search(q, int(max(1, k)))
        hits: List[dict] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self._library_meta):
                continue
            meta = self._library_meta[idx] or {}
            hits.append({"sim": float(score), "meta": meta})
        return hits

    def search(self, img_bgr, k: int = 10) -> List[dict]:
        emb = self.embed_first_face(img_bgr)
        if emb is None:
            return []
        return self.search_by_embedding(emb, k=k)

    def ensure_facebank(self):
        if self._facebank_loaded and self._facebank_index is not None and self._facebank_labels is not None:
            return

        index_path  = Path(self.cfg.get("facebank_index",  "outputs/facebank.index"))
        labels_path = Path(self.cfg.get("facebank_labels", "outputs/labels.json"))

        if not index_path.exists() or not labels_path.exists():
            self._facebank_loaded = True
            self._facebank_index = None
            self._facebank_labels = []
            return

        self._facebank_index = faiss.read_index(str(index_path))
        with open(labels_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "labels" in data:
            self._facebank_labels = [str(x) for x in data["labels"]]
        elif isinstance(data, list):
            self._facebank_labels = [str(x) for x in data]
        else:
            self._facebank_labels = []
        self._facebank_loaded = True

    def recognize_faces(
    self,
    img_bgr: np.ndarray,
    topk: int = 1,
    unknown_label: str = "Unknown",
    ) -> Tuple[List[Tuple[int,int,int,int]], List[str], List[float]]:
        self.ensure_facebank()
        faces = detect_faces(self.app, img_bgr)
        if not faces:
            return [], [], []

        H, W = img_bgr.shape[:2]

        boxes: List[Tuple[int,int,int,int]] = []
        embs: List[Optional[np.ndarray]] = []
        for f in faces:
            bb = getattr(f, "bbox", None)
            if bb is None:
                boxes.append((0, 0, 0, 0))
            else:
                x1, y1, x2, y2 = bb
                if x2 < x1: x1, x2 = x2, x1
                if y2 < y1: y1, y2 = y2, y1
                x1 = max(0, min(int(round(x1)), W - 1))
                y1 = max(0, min(int(round(y1)), H - 1))
                x2 = max(0, min(int(round(x2)), W - 1))
                y2 = max(0, min(int(round(y2)), H - 1))
                boxes.append((x1, y1, x2, y2))

            e = face_to_embedding(f)
            if e is None:
                embs.append(None)
            else:
                embs.append(e.astype(np.float32).reshape(-1))

        labels: List[str] = []
        scores: List[float] = []

        if self._facebank_index is None or not self._facebank_labels:
            for e in embs:
                labels.append(unknown_label)
                scores.append(float("nan"))
            return boxes, labels, scores

        dim = next((e.shape[0] for e in embs if e is not None), 512)
        Q = np.zeros((len(embs), dim), dtype=np.float32)
        valid = np.zeros(len(embs), dtype=bool)
        for i, e in enumerate(embs):
            if e is not None and e.size == dim:
                Q[i] = e
                valid[i] = True
        Q = l2_normalize(Q)

        k = max(1, int(topk))
        s_all, i_all = self._facebank_index.search(Q, k)

        for i in range(len(embs)):
            if not valid[i]:
                labels.append(unknown_label)
                scores.append(float("nan"))
                continue
            best_sim = float(s_all[i, 0])
            best_idx = int(i_all[i, 0])
            if 0 <= best_idx < len(self._facebank_labels) and best_sim >= self.threshold:
                labels.append(self._facebank_labels[best_idx])
                scores.append(best_sim)
            else:
                labels.append(unknown_label)
                scores.append(best_sim if best_idx >= 0 else float("nan"))
        return boxes, labels, scores

_ENGINE: Optional[FaceEngine] = None
def get_engine(cfg: Optional[dict] = None) -> FaceEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = FaceEngine(cfg=cfg)
    return _ENGINE
