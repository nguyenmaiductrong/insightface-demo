from pathlib import Path
import argparse
import json
import numpy as np
import faiss

from .utils import (
    load_cfg,
    get_app, detect_faces, face_to_embedding,
    l2_normalize, read_image,
)

try:
    from tqdm.auto import tqdm
except Exception: 
    tqdm = None

EXTS = {".jpg", ".jpeg", ".png"}

def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in EXTS
    )

def build_library_embeddings(
    cfg: dict,
    lib_dir: Path,
    min_det_score: float = 0.5,
    min_face_size: int = 48,
    show_progress: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    app = get_app(cfg)
    paths = list_images(lib_dir)

    iterator = paths
    if show_progress and tqdm is not None:
        iterator = tqdm(paths, desc="Building library embeddings", unit="img")

    embeddings: list[np.ndarray] = []
    metadatas: list[dict] = []
    nid = 0

    for img_path in iterator:
        pil_img = read_image(img_path)
        if pil_img is None:
            continue
        img = np.array(pil_img)[..., ::-1]

        faces = detect_faces(app, img)
        for face_index, face in enumerate(faces):
            score = float(getattr(face, "det_score", 0.0))
            if score < min_det_score:
                continue

            bbox = getattr(face, "bbox", None)
            x1 = y1 = x2 = y2 = None
            if bbox is not None:
                x1, y1, x2, y2 = [float(v) for v in np.asarray(bbox).reshape(-1)[:4]]
                w, h = x2 - x1, y2 - y1
                if min(w, h) < min_face_size:
                    continue

            e = face_to_embedding(face)
            if e is None:
                continue
            e = e.astype("float32")
            e = l2_normalize(e[None, :]).squeeze(0) 
            embeddings.append(e)

            metadatas.append({
                "id": nid,
                "image_path": str(img_path),
                "face_index": face_index,
                "bbox": [x1, y1, x2, y2] if bbox is not None else None,
                "det_score": score,
            })
            nid += 1

    if not embeddings:
        return np.empty((0, 512), dtype=np.float32), []

    E = np.vstack(embeddings).astype("float32")
    return E, metadatas

if __name__ == "__main__":
    cfg_path = "configs/default.yaml"
    cfg = load_cfg(cfg_path)

    lib_cfg   = cfg.get("library", {})     
    paths_cfg = cfg.get("paths", {})

    library_dir = Path(paths_cfg.get("library_dir", "data/library")).expanduser()
    index_path  = Path(paths_cfg.get("library_index", "outputs/library.index")).expanduser()
    meta_path   = Path(paths_cfg.get("library_meta",  "outputs/library_meta.json")).expanduser()

    min_det_score = float(lib_cfg.get("min_det_score", 0.5))
    min_face_size = int(lib_cfg.get("min_face_size", 48))
    show_progress = bool(lib_cfg.get("show_progress", True))

    E, metas = build_library_embeddings(
        cfg=cfg,
        lib_dir=library_dir,
        min_det_score=min_det_score,
        min_face_size=min_face_size,
        show_progress=show_progress,
    )

    dim = E.shape[1] if E.size else 512
    index = faiss.IndexFlatIP(dim) 
    if E.size:
        index.add(E)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    meta_path.write_text(json.dumps(metas, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Built library index with {len(metas)} faces")
    print(f"Index -> {index_path}")
    print(f"Meta  -> {meta_path}")
