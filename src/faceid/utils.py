from typing import List, Tuple, Optional
from pathlib import Path
import yaml
import numpy as np
import cv2
from insightface.app import FaceAnalysis
from PIL import Image, ImageOps

def load_cfg(path_or_dict):
    p = Path(path_or_dict)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_app(cfg: dict):
    det_size = tuple(cfg.get("det_size"))
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    ctx_id = -1 if str(cfg.get("device", "cpu")).lower() else 0
    app.prepare(ctx_id=ctx_id, det_size=det_size)
    return app

def detect_faces(app, img_bgr) -> list:
    if img_bgr is None or not hasattr(img_bgr, "shape"):
        return []
    faces = app.get(img_bgr)
    return faces or []

def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    if x.ndim == 1:
        denom = np.sqrt(np.maximum((x * x).sum(), eps))
        return (x / denom).astype(np.float32)
    denom = np.sqrt(np.maximum((x * x).sum(axis=axis, keepdims=True), eps))
    return (x / denom).astype(np.float32)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float((a * b).sum())

def face_to_embedding(face) -> Optional[np.ndarray]:
    if hasattr(face, "normed_embedding") and face.normed_embedding is not None:
        return np.asarray(face.normed_embedding, dtype=np.float32)
    if hasattr(face, "embedding") and face.embedding is not None:
        emb = np.asarray(face.embedding, dtype=np.float32)
        return l2_normalize(emb.reshape(1, -1))[0]
    return None

def read_image(file) -> Optional[Image.Image]:
    if not file:
        return None
    try:
        img = Image.open(file)
        return ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        return None

def draw_boxes(img_bgr, boxes, labels=None, scores=None, in_place=False):
    labels = labels or []
    scores = scores or []
    out = img_bgr if in_place else img_bgr.copy()
    for i, b in enumerate(boxes or []):
        x1, y1, x2, y2 = [int(v) for v in b]
        lab = labels[i] if i < len(labels) else ""
        sc  = scores[i] if i < len(scores) else None
        color = (0, 255, 0) if lab and lab != "Unknown" else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if lab:
            txt = f"{lab}" + (f" {sc:.2f}" if (isinstance(sc,(int,float)) and np.isfinite(sc)) else "")
            cv2.putText(out, txt, (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out
