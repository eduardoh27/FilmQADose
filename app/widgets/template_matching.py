from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
import os
import cv2
import numpy as np
import tifffile
from datetime import datetime
from calibration.image_processing import template_matching  
from PIL import Image as PILImage
from io import BytesIO
from PySide6.QtWidgets import (
    QDialog, 
    QWidget, 
    QVBoxLayout, 
    QHBoxLayout, 
    QLineEdit,
    QPushButton,
    QLabel,
    QDialogButtonBox,
    QFileDialog
)


class TemplateDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.default_dir = os.path.join(os.path.dirname(__file__), "..", "media")
        self.setWindowTitle("Configure Template Matching")
        self.setMinimumWidth(500)
        self.template_path = ''
        self.film_path = ''

        layout = QVBoxLayout(self)

        # Template
        hl = QHBoxLayout()
        self.template_edit = QLineEdit()
        self.template_edit.setPlaceholderText('TPS template (.dcm)')
        btn = QPushButton('Browse Template')
        btn.clicked.connect(self.browse_template)
        hl.addWidget(self.template_edit)
        hl.addWidget(btn)
        layout.addLayout(hl)

        # Film
        hl = QHBoxLayout()
        self.film_edit = QLineEdit()
        self.film_edit.setPlaceholderText('Scanned Film (.tif)')
        btn = QPushButton('Browse Film')
        btn.clicked.connect(self.browse_film)
        hl.addWidget(self.film_edit)
        hl.addWidget(btn)
        layout.addLayout(hl)

        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def browse_template(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select TPS template", 
            self.default_dir,
            filter="DICOM Files (*.dcm)"
        )
        if path:
            self.template_path = path
            self.template_edit.setText(os.path.basename(path))

    def browse_film(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select scanned film", 
            self.default_dir,
            filter="TIFF Files (*.tif *.tiff)"
        )
        if path:
            self.film_path = path
            self.film_edit.setText(os.path.basename(path))

