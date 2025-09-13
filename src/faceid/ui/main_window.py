from PySide6.QtWidgets import QMainWindow, QTabWidget
from .verify_tab import VerifyTab
from .library_tab import LibraryTab
from .realtime_tab import RealtimeTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("InsightFace Demo")
        self.resize(1200, 800)
        
        tabs = QTabWidget()
        tabs.addTab(VerifyTab(), "So sánh Khuôn mặt")
        tabs.addTab(LibraryTab(), "Tìm kiếm Ảnh trong Kho")
        tabs.addTab(RealtimeTab(), "Nhận diện Realtime (Webcam)")
        
        self.setCentralWidget(tabs)