from typing import Optional, List, Dict
import streamlit as st
from PIL import Image, ImageOps
from pathlib import Path
from faceid.engine import get_engine
from faceid.utils import read_image, draw_boxes
import math
import cv2
import numpy as np
import tempfile
import os, sys, subprocess

st.set_page_config(page_title="InsightFace Demo", layout="wide")
st.markdown("""
<style>
.block-container{ max-width: 1600px; padding: 1rem 1rem 2rem; }
.app-title{ text-align:center; font-weight:800; font-size:48px; line-height:1.1; margin:0; padding:.25rem 0 1rem 0; }
.uploadLabel{ text-align:center; font-weight:600; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)
st.markdown('<h1 class="app-title">InsightFace Demo</h1>', unsafe_allow_html=True)

EXTS = {".jpg", ".jpeg", ".png"}

def make_thumb_no_upscale(img: Image.Image, max_w=260, max_h=260) -> Image.Image:
    th = img.copy()
    th.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    return th

def upload_or_preview(title: str, key: str, thumb_w: int = 260, thumb_h: int = 260) -> Optional[Image.Image]:
    show_key = f"show_{key}"
    img_key  = f"img_{key}"
    st.session_state.setdefault(show_key, True)
    st.session_state.setdefault(img_key, None)

    with st.container():
        st.markdown(f'<div class="uploadLabel">{title}</div>', unsafe_allow_html=True)
        if st.session_state[show_key]:
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                f = st.file_uploader("Chọn ảnh", type=["jpg","jpeg","png"], key=key, label_visibility="collapsed")
                st.markdown('<div class="empty-slot"></div>', unsafe_allow_html=True)
            if f is not None:
                img = read_image(f)
                if img is not None:
                    st.session_state[img_key] = img
                    st.session_state[show_key] = False
                    st.rerun()
            return None
        else:
            img: Image.Image = st.session_state[img_key]
            if img is None:
                st.session_state[show_key] = True
                st.rerun()
            thumb = make_thumb_no_upscale(img, max_w=thumb_w, max_h=thumb_h)
            c1, c2, c3 = st.columns([1, 2, 1])
            with c2:
                st.image(thumb, width=thumb.width)
            _, cbtn, _ = st.columns([1, 2, 1])
            with cbtn:
                if st.button("Đổi ảnh", key=f"reset_{key}", use_container_width=True):
                    st.session_state[show_key] = True
                    st.session_state[img_key]  = None
                    st.session_state.pop(key, None)
                    st.session_state.pop("last_result", None)
                    st.rerun()
            return img

def show_verify_result(score: Optional[float], matched: bool, thr: float):
    if score is None or (isinstance(score, float) and math.isnan(score)):
        st.warning("Không tìm thấy khuôn mặt ở một trong hai ảnh.")
        return
    color = "green" if matched else "red"
    verdict = "Khả năng cao là cùng một người" if matched else "Khả năng cao là hai người khác nhau"
    st.markdown(
        f"""
        <div style="font-size:18px; line-height:1.6">
          Độ tương đồng cosine: <span style="color:{color}; font-weight:700">{score:.3f}</span><br/>
          Kết luận: <span style="color:{color}; font-weight:700">{verdict}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def set_mode(new_mode: str):
    st.session_state.mode = new_mode

@st.cache_data(show_spinner=False)
def list_library_images(lib_dir: Path) -> List[Path]:
    if not lib_dir.exists():
        return []
    files = [p for p in lib_dir.rglob("*") if p.suffix.lower() in EXTS and p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files

def show_image_grid(paths: List[Path], thumbs: int = 260, cols: int = 5):
    if not paths:
        return
    rows = (len(paths) + cols - 1) // cols
    idx = 0
    for _ in range(rows):
        cs = st.columns(cols, gap="small")
        for c in cs:
            if idx >= len(paths):
                break
            p = paths[idx]
            try:
                img = Image.open(p)
                img = ImageOps.exif_transpose(img).convert("RGB")
                th = make_thumb_no_upscale(img, max_w=thumbs, max_h=thumbs)
                with c:
                    st.image(th, caption=p.name, width=th.width)
            except Exception:
                with c:
                    st.warning(f"Không đọc được ảnh: {p.name}")
            idx += 1

def _save_uploaded_to_temp(uploaded_file) -> str:
    suf = Path(uploaded_file.name).suffix if uploaded_file and Path(uploaded_file.name).suffix else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

@st.cache_resource(show_spinner=False)
def get_rt_engine_cached(_ver: int = 3):
    eng = get_engine()
    try:
        eng.ensure_facebank()
    except Exception:
        pass
    try:
        _ = eng.recognize_faces(
            np.zeros((160, 160, 3), dtype=np.uint8),
            topk=1,
            unknown_label="Unknown"
        )
    except Exception:
        pass
    return eng


def _infer_faces(engine, frame, topk=1, unknown_label="Unknown"):
    try:
        return engine.recognize_faces(frame, topk=topk, unknown_label=unknown_label)
    except Exception:
        return [], [], []


def process_video(input_path: str, engine) -> str:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Không mở được video đầu vào")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1 or math.isnan(fps):
        fps = 25.0

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Không đọc được frame đầu tiên")

    height, width = first_frame.shape[:2]
    w = width + (width % 2)
    h = height + (height % 2)

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".avi").name
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Không thể tạo video writer")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))

        boxes, labels, scores = _infer_faces(engine, frame, topk=1, unknown_label="Unknown")
        annotated_frame = draw_boxes(frame, boxes, labels, scores)

        writer.write(annotated_frame)

    cap.release()
    writer.release()

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Video output không được tạo")

    return output_path


#========================================================================
tab1, tab2, tab3 = st.tabs(["So sánh Khuôn mặt", "Tìm kiếm Ảnh trong kho", "Nhận diện qua Video"])

with tab1:
    colL, colR = st.columns(2, gap="large")
    with colL:
        imgA = upload_or_preview("Ảnh thứ nhất", key="verifyA", thumb_w=260, thumb_h=260)
    with colR:
        imgB = upload_or_preview("Ảnh thứ hai", key="verifyB", thumb_w=260, thumb_h=260)

    _, mid, _ = st.columns([2, 1, 2])
    with mid:
        ready = (imgA is not None and imgB is not None)
        clicked = st.button("So sánh", use_container_width=True, disabled=not ready)
        if not ready:
            st.info("Hãy tải đủ **2 ảnh** trước khi so sánh.")
        elif clicked:
            st.session_state["last_result"] = None
            bgrA = cv2.cvtColor(np.array(imgA), cv2.COLOR_RGB2BGR)
            bgrB = cv2.cvtColor(np.array(imgB), cv2.COLOR_RGB2BGR)
            engine = get_engine()
            score, matched = engine.verify(bgrA, bgrB)
            show_verify_result(score, matched, thr=engine.threshold)

with tab2:
    left_spacer, col_library_btn, col_search_btn, col_library_faiss_btn = st.columns([7, 1, 1, 1])
    with col_library_btn:
        st.button("Thư viện ảnh", use_container_width=True, on_click=set_mode, args=("library",))
    with col_search_btn:
        st.button("Tìm kiếm ảnh", use_container_width=True, on_click=set_mode, args=("search",))
    with col_library_faiss_btn:
        if st.button("Khởi tạo faiss", use_container_width=True):
            try:
                with st.status("Đang khởi tạo FAISS...", expanded=True) as status:
                    env = os.environ.copy()
                    env["PYTHONPATH"] = f"{Path('.').resolve()}:{env.get('PYTHONPATH','')}"
                    cmd = [sys.executable, "-m", "src.faceid.build_library_faiss"]
                    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
                st.session_state["mode"] = "search"
                st.success("FAISS đã sẵn sàng.")
            except Exception as e:
                st.exception(e)
                st.error("Khởi tạo FAISS thất bại.")

    mode = st.session_state.get("mode", "library")
    if mode == "library":
        lib_dir = Path("data/library").resolve()
        paths = list_library_images(lib_dir)
        st.caption(f"Thư viện: **{lib_dir}** ({len(paths)} ảnh)")
        if not paths:
            if not lib_dir.exists():
                st.info(f"Thư mục **{lib_dir}** chưa tồn tại. Hãy tạo và thêm ảnh vào đó.")
            else:
                st.info("Thư mục thư viện chưa có ảnh. Hãy thêm ảnh vào `data/library`.")
        else:
            show_image_grid(paths, thumbs=260, cols=5)

    elif mode == "search":
        colL, colR = st.columns([1, 3], gap="large")
        with colL:
            imgQ = upload_or_preview("Ảnh truy vấn (người cần tìm)", key="searchQ", thumb_w=260, thumb_h=260)
            ready = (imgQ is not None)
            clicked = st.button("Tìm kiếm", use_container_width=True, disabled=not ready)
        with colR:
            if clicked:
                bgrQ = cv2.cvtColor(np.array(imgQ), cv2.COLOR_RGB2BGR)
                engine = get_engine()
                hits = engine.search(bgrQ, k=20)
                if not hits:
                    st.warning("Không tìm thấy kết quả phù hợp")
                else:
                    for i in range(0, len(hits), 2):
                        c1, c2 = st.columns([1, 1], gap="large")
                        for col, h in zip((c1, c2), hits[i:i+2]):
                            meta = h.get("meta") or {}
                            p = meta.get("image_path")
                            if not p or not Path(p).exists():
                                continue
                            im = Image.open(p)
                            im = ImageOps.exif_transpose(im).convert("RGB")
                            th = make_thumb_no_upscale(im, max_w=360, max_h=360)
                            with col:
                                st.image(th, caption=f'{Path(p).name} | cosine = {h["sim"]:.3f}', width=th.width)

with tab3:
    video_file = st.file_uploader(
        "Drag and drop file here",
        type=["mp4", "mov", "avi"],
    )
    
    if video_file is not None:
        file_size = len(video_file.getbuffer()) / (1024 * 1024) 
        st.info(f"**{video_file.name}** ({file_size:.1f} MB)")
        
        if st.button("Nhận diện", use_container_width=True, type="primary"):
            with st.spinner("Đang xử lý video..."):
                try:
                    input_path = _save_uploaded_to_temp(video_file)
                    
                    engine = get_rt_engine_cached()
                    output_path = process_video(input_path, engine)
                    
                    with open(output_path, "rb") as f:
                        video_data = f.read()
                    
                    original_name = Path(video_file.name).stem
                    output_filename = f"{original_name}_detected.avi"
                    
                    st.success("Xử lý hoàn tất!")
                    
                    st.download_button(
                        label="Tải video đã nhận diện",
                        data=video_data,
                        file_name=output_filename,
                        mime="video/x-msvideo",
                        use_container_width=True
                    )
                    st.info("Video đã được xử lý với bounding boxes và labels cho tất cả khuôn mặt được phát hiện")
                    try:
                        os.remove(input_path)
                        os.remove(output_path)
                    except:
                        pass
                        
                except Exception as e:
                    st.error(f"Lỗi xử lý: {str(e)}")
    else:
        st.info("Chọn video để bắt đầu nhận diện khuôn mặt")
