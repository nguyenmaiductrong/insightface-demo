import os
import sys
from typing import Optional
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QLineEdit, QListWidget, QListWidgetItem, QGroupBox,
    QFileDialog, QMessageBox, QListView
)

from ..core import get_rt_engine_cached
from ..utils.image_utils import load_image_as_bgr, bgr_to_qpix, list_library_images


class LibraryTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = get_rt_engine_cached()

        self.lib_dir = QLineEdit(str(Path("data/library").resolve()))
        btn_browse = QPushButton("Chọn thư mục")
        btn_refresh = QPushButton("Tải lại")

        btn_browse.clicked.connect(self.choose_dir)
        btn_refresh.clicked.connect(self.refresh_grid)

        self.grid = QListWidget()
        self.grid.setViewMode(QListView.IconMode)
        self.grid.setIconSize(QSize(180, 180))
        self.grid.setResizeMode(QListView.Adjust)
        self.grid.setSpacing(8)
        self.grid.setUniformItemSizes(True)

        top = QHBoxLayout()
        top.addWidget(self.lib_dir, 3)
        top.addWidget(btn_browse, 1)
        top.addWidget(btn_refresh, 1)

        self.q_preview = QLabel("Ảnh truy vấn")
        self.q_preview.setAlignment(Qt.AlignCenter)
        self.q_preview.setMinimumSize(220, 180)
        self.q_preview.setStyleSheet("border:1px dashed #aaa;")

        btn_pick_q = QPushButton("Chọn ảnh truy vấn")
        btn_search = QPushButton("Tìm kiếm")
        btn_pick_q.clicked.connect(self.pick_query)
        btn_search.clicked.connect(self.run_search)

        self.q_bgr: Optional[np.ndarray] = None
        self.result_list = QListWidget()
        self.result_list.setViewMode(QListView.IconMode)
        self.result_list.setResizeMode(QListView.Adjust)
        self.result_list.setIconSize(QSize(220, 220))
        self.result_list.setSpacing(8)
        self.result_list.setUniformItemSizes(True)

        search_box = QGroupBox("Tìm kiếm ảnh trong kho")
        sb = QHBoxLayout(search_box)
        left = QVBoxLayout()
        left.addWidget(self.q_preview)
        left.addWidget(btn_pick_q)
        left.addWidget(btn_search)
        sb.addLayout(left, 1)
        sb.addWidget(self.result_list, 3)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.grid, 1)
        lay.addWidget(search_box, 2)

        self.refresh_grid()

    def choose_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Chọn thư mục thư viện", self.lib_dir.text())
        if path:
            self.lib_dir.setText(path)
            self.refresh_grid()

    def refresh_grid(self):
        self.grid.clear()
        paths = list_library_images(Path(self.lib_dir.text()))
        for p in paths:
            try:
                pil = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
                pil.thumbnail((180, 180), Image.Resampling.LANCZOS)
                qimg = QImage(np.array(pil).data, pil.width, pil.height, pil.width * 3, QImage.Format.Format_RGB888)
                pm = QPixmap.fromImage(qimg)
                item = QListWidgetItem(p.name)
                item.setToolTip(str(p))
                item.setIcon(QIcon(pm))
                self.grid.addItem(item)
            except Exception:
                pass

    def pick_query(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh truy vấn", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        self.q_bgr = load_image_as_bgr(path)
        if self.q_bgr is not None:
            pm = bgr_to_qpix(self.q_bgr).scaled(300, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.q_preview.setPixmap(pm)

    def run_search(self):
        self.result_list.clear()
        if self.q_bgr is None:
            QMessageBox.information(self, "Thiếu ảnh", "Hãy chọn ảnh truy vấn trước.")
            return
        try:
            hits = self.engine.search(self.q_bgr, k=20)
        except Exception as e:
            QMessageBox.warning(self, "Lỗi tìm kiếm", str(e))
            return
        if not hits:
            QMessageBox.information(self, "Kết quả", "Không tìm thấy kết quả phù hợp.")
            return
        threshold = self.engine.threshold
        matched_count = 0
        
        for h in hits:
            meta = h.get("meta") or {}
            p = meta.get("image_path")
            if not p or not Path(p).exists():
                continue
                
            sim_score = h["sim"]
            if sim_score < threshold:
                continue
                
            try:
                pil = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
                pil.thumbnail((220, 220), Image.Resampling.LANCZOS)
                qimg = QImage(np.array(pil).data, pil.width, pil.height, pil.width * 3, QImage.Format.Format_RGB888)
                pm = QPixmap.fromImage(qimg)
                
                item_text = f'{Path(p).name} | cosine={sim_score:.3f}'
                
                item = QListWidgetItem(item_text)
                item.setToolTip(f"{p}\nĐộ tương đồng: {sim_score:.3f}\nNgưỡng: {threshold:.3f}")
                item.setIcon(QIcon(pm))
                
                item.setForeground(Qt.GlobalColor.darkGreen)
                
                self.result_list.addItem(item)
                matched_count += 1
            except Exception:
                continue
        
        if matched_count == 0:
            QMessageBox.information(self, "Kết quả", f"Không tìm thấy ảnh nào có độ tương đồng >= {threshold:.3f}")
        else:
            print(f"Tìm thấy {matched_count} ảnh khớp với ngưỡng {threshold:.3f}")

