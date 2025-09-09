from .utils import (
    load_cfg, get_app, detect_faces, face_to_embedding,
    cosine_similarity, read_image, draw_boxes
)
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd

# ========== Data models ==========
class FaceEngine:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.threshold: float = float(cfg.get("threshold"))
        self.paths = cfg.get("paths", {})
        self.app = get_app(cfg)

    def embed_first_face(self, img_bgr) -> Optional[np.ndarray]:
        faces = detect_faces(self.app, img_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
        return face_to_embedding(face)

    def verify(self, img1_bgr, img2_bgr) -> Tuple[Optional[float], bool]:
        e1 = self.embed_first_face(img1_bgr)
        e2 = self.embed_first_face(img2_bgr)
        if e1 is None or e2 is None:
            return None, False
        score = cosine_similarity(e1, e2) 
        return score, (score >= self.threshold)

if __name__ == "__main__":
    cfg = load_cfg(Path(__file__).parents[2] / "configs" / "default.yaml")
    engine = FaceEngine(cfg)

    ROOT = Path(__file__).parents[2] / "data" / "facebank"
    PEOPLE = ["nguyen_mai_duc_trong", "quoc_anh", "kieu_linh_trang", "le_duc_hieu"]
    N_PER = 5
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def first_n_images(person_dir: Path, n: int):
        files = [p for p in person_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
        files.sort(key=lambda p: p.name)
        return files[:n]

    paths: List[Path] = []
    for name in PEOPLE:
        pdir = ROOT / name
        if pdir.exists():
            paths.extend(first_n_images(pdir, N_PER))

    labels = [f"{p.parent.name}/{p.name}" for p in paths]
    N = len(paths)

    embs: List[Optional[np.ndarray]] = []
    for p in paths:
        img = read_image(p)
        e = engine.embed_first_face(img)
        embs.append(e)

    S = np.full((N, N), np.nan, dtype=float)
    for i in range(N):
        for j in range(N):
            ei, ej = embs[i], embs[j]
            if ei is not None and ej is not None:
                S[i, j] = cosine_similarity(ei, ej)  

    df = pd.DataFrame(S, index=labels, columns=labels)
    print(df.round(3))

    out_dir = Path(__file__).parents[2] / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "similarity_matrix_20x20.csv", float_format="%.6f")
    print(f"Saved -> {out_dir / 'similarity_matrix_20x20.csv'}")
