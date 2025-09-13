import time
import numpy as np
import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QMessageBox
)

from ..core import get_rt_engine_cached
from ..utils.image_utils import bgr_to_qpix
from ..utils import draw_boxes


class RealtimeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.engine = get_rt_engine_cached()

        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        self.target_width_value = 640
        self.every_n_value = 3
        self.topk_value = 1

        self.video_label = QLabel("Webcam sẽ hiển thị ở đây")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background:#111; color:#bbb; border:1px solid #333;")

        controls = QHBoxLayout()
        controls.addWidget(self.btn_start)
        controls.addWidget(self.btn_stop)

        lay = QVBoxLayout(self)
        lay.addLayout(controls)
        lay.addWidget(self.video_label, 1)

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self._i = 0
        self._last = ([], [], [])
        self._t_last = 0.0
        self._min_interval = 1.0 / 15.0 

        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)

    def start(self):
        if self.cap is not None:
            return
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Webcam", "Không mở được camera.")
            self.cap = None
            return
        self.timer.start(0) 
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop(self):
        if self.cap is not None:
            self.timer.stop()
            self.cap.release()
            self.cap = None
        self.video_label.setText("Đã dừng webcam")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._i = 0
        self._last = ([], [], [])
        self._t_last = 0.0

    def _on_timer(self):
        ok, frame = self.cap.read()
        if not ok:
            return

        h0, w0 = frame.shape[:2]
        tgt_w = self.target_width_value
        scale = 1.0
        if w0 > tgt_w:
            scale = tgt_w / w0
            small = cv2.resize(frame, (int(w0 * scale), int(h0 * scale)))
        else:
            small = frame

        now = time.time()
        if (self._i % self.every_n_value == 0) and (now - self._t_last >= self._min_interval):
            try:
                boxes, labels, scores = self.engine.recognize_faces(
                    small, topk=self.topk_value, unknown_label="Unknown"
                )
                if boxes:
                    boxes = (np.array(boxes) / scale).astype(int).tolist()
                self._last = (boxes, labels, scores)
            except Exception:
                pass
            self._t_last = now
        self._i += 1

        boxes, labels, scores = self._last
        annotated = draw_boxes(frame, boxes, labels, scores)
        self.video_label.setPixmap(bgr_to_qpix(annotated).scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def closeEvent(self, e):
        self.stop()
        super().closeEvent(e)
