import math
from typing import Optional
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox
)

from ..core import get_rt_engine_cached
from ..utils.image_utils import load_image_as_bgr, bgr_to_qpix


class VerifyTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = get_rt_engine_cached()

        self.imgA_bgr: np.ndarray | None = None
        self.imgB_bgr: np.ndarray | None = None

        self.labelA = QLabel("Ảnh thứ nhất")
        self.labelA.setAlignment(Qt.AlignCenter)
        self.labelA.setMinimumSize(320, 240)
        self.labelA.setStyleSheet("border:1px dashed #aaa;")

        self.labelB = QLabel("Ảnh thứ hai")
        self.labelB.setAlignment(Qt.AlignCenter)
        self.labelB.setMinimumSize(320, 240)
        self.labelB.setStyleSheet("border:1px dashed #aaa;")

        btnA = QPushButton("Chọn ảnh thứ nhất")
        btnB = QPushButton("Chọn ảnh thứ hai")
        btnCmp = QPushButton("So sánh")
        btnCmp.setDefault(True)

        btnA.clicked.connect(self.pick_A)
        btnB.clicked.connect(self.pick_B)
        btnCmp.clicked.connect(self.compare)

        self.result = QLabel("")
        self.result.setAlignment(Qt.AlignCenter)
        self.result.setStyleSheet("font-size:16px; font-weight:600;")

        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        colA = QVBoxLayout()
        colB = QVBoxLayout()
        colA.addWidget(self.labelA)
        colA.addWidget(btnA)
        colB.addWidget(self.labelB)
        colB.addWidget(btnB)
        row.addLayout(colA, 1)
        row.addLayout(colB, 1)

        lay.addLayout(row)
        lay.addWidget(btnCmp)
        lay.addWidget(self.result)

    def pick_A(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh thứ nhất", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.imgA_bgr = load_image_as_bgr(path)
            if self.imgA_bgr is not None:
                self.labelA.setPixmap(bgr_to_qpix(self.imgA_bgr).scaled(
                    420, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))

    def pick_B(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh thứ hai", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.imgB_bgr = load_image_as_bgr(path)
            if self.imgB_bgr is not None:
                self.labelB.setPixmap(bgr_to_qpix(self.imgB_bgr).scaled(
                    420, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))

    def compare(self):
        if self.imgA_bgr is None or self.imgB_bgr is None:
            QMessageBox.information(self, "Thiếu ảnh", "Bạn cần chọn đủ 2 ảnh.")
            return
        score, matched = self.engine.verify(self.imgA_bgr, self.imgB_bgr)
        if score is None or (isinstance(score, float) and math.isnan(score)):
            self.result.setText("Không tìm thấy khuôn mặt ở một trong hai ảnh.")
            self.result.setStyleSheet("color:#c00; font-size:16px; font-weight:600;")
            return
        color = "#16a34a" if matched else "#dc2626"
        verdict = "Khả năng cao là cùng một người" if matched else "Khả năng cao là hai người khác nhau"
        self.result.setText(f"Độ tương đồng cosine: {score:.3f} \n{verdict}")
        self.result.setStyleSheet(f"color:{color}; font-size:16px; font-weight:700;")
