"""Main entry point for the application."""

import os
import sys
import cv2
from PySide6.QtWidgets import QApplication
from .ui import MainWindow

# Performance optimizations
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
cv2.setNumThreads(1)


def main():
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
