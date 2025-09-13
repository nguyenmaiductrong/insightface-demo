from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
from PySide6.QtGui import QImage, QPixmap

EXTS = {".jpg", ".jpeg", ".png"}

def load_image_as_bgr(image_path: str | Path) -> np.ndarray | None:
    try:
        pil_img = Image.open(image_path)
        pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
        rgb_array = np.array(pil_img)
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        return bgr_array
    except Exception:
        return None

def bgr_to_qpix(bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)

def pil_to_rgb_np(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))

def rgb_to_qpix(rgb: np.ndarray) -> QPixmap:
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)

def list_library_images(lib_dir: Path) -> list[Path]:
    if not lib_dir.exists():
        return []
    files = [p for p in lib_dir.rglob("*") if p.suffix.lower() in EXTS and p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files