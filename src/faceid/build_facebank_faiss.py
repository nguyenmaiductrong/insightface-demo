from pathlib import Path
import json
import numpy as np
import faiss

from .utils import (
    load_cfg, get_app, detect_faces, face_to_embedding,
    l2_normalize, read_image,
)

EXTS = {".jpg", ".jpeg", ".png"}

def list_person_dirs(facebank_dir: str | Path) -> list[Path]:
    root = Path(facebank_dir)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())

def list_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*") 
        if p.is_file() and p.suffix.lower() in EXTS
    )

def build_facebank_embeddings(cfg: dict, facebank_dir: Path) -> tuple[np.ndarray, list[str]]:
    app = get_app(cfg)
    reps: list[np.ndarray] = []
    labels: list[str] = []

    for person_dir in list_person_dirs(facebank_dir):
        embs = []
        for img_path in list_images(person_dir):
            pil_img = read_image(img_path)
            if pil_img is None:
                continue
            img = np.array(pil_img)[..., ::-1]
            faces = detect_faces(app, img)
            if not faces:
                continue
            f = faces[0] 
            e = face_to_embedding(f)
            if e is None:
                continue
            e = l2_normalize(e.astype("float32")[None, :]).squeeze(0)
            embs.append(e)

        if not embs:
            continue

        rep = np.mean(np.vstack(embs).astype("float32"), axis=0)
        rep = l2_normalize(rep[None, :]).squeeze(0)
        reps.append(rep)
        labels.append(person_dir.name)

    if not reps:
        return np.empty((0, 512), dtype=np.float32), []

    E = np.vstack(reps).astype("float32")
    return E, labels

if __name__ == "__main__":
    cfg = load_cfg("configs/default.yaml")
    paths = cfg.get("paths", {})

    facebank_dir = Path(paths.get("facebank_dir", "data/facebank")).expanduser()
    index_path   = Path(paths.get("index_path",   "outputs/facebank.index")).expanduser()
    labels_path  = Path(paths.get("labels_path",  "outputs/labels.json")).expanduser()

    E, labels = build_facebank_embeddings(cfg, facebank_dir)

    dim = E.shape[1] if E.size else 512
    index = faiss.IndexFlatIP(dim)  
    if E.size:
        index.add(E)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    labels_path.write_text(
        json.dumps({"labels": labels}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Built facebank with {len(labels)} people")
    print(f"Index  -> {index_path}")
    print(f"Labels -> {labels_path}")
