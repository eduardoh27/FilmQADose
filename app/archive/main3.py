import sys
import os
import numpy as np
import pydicom
import matplotlib
os.environ["QT_API"] = "PySide6"
# Use QtAgg backend for embedding in Qt
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QFileDialog,
    QTabWidget,
    QSizePolicy
)

# Import modified isodose logic returning a Figure
from dose_analysis.isodoses3 import IsodoseWidget   # en lugar de plot_isodose_map

class ScaledLabel(QLabel):
    """
    QLabel that scales pixmap preserving aspect ratio and stores metadata.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.metadata = {}
        self._pixmap = None

    def setPixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap
        self._update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update()

    def _update(self):
        if self._pixmap and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            super().setPixmap(scaled)
        else:
            super().setPixmap(QPixmap())

class FilmQADoseMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FilmQADose")

        menubar = self.menuBar()
        # File
        file_menu = menubar.addMenu("File")
        file_menu.addAction(QAction("Open...", self, triggered=self.open_file))
        # Calibration placeholder
        menubar.addMenu("Calibration")
        # Tools
        tools_menu = menubar.addMenu("Tools")
        tools_menu.addAction(QAction("Isodoses", self, triggered=self.open_isodoses))

        # Central layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left panel info
        self.info_label = QLabel("Welcome to FilmQADose")
        left = QVBoxLayout()
        left.addWidget(self.info_label)
        layout.addLayout(left, 1)

        # Tabs on right
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(lambda i: self.tabs.removeTab(i))
        self.tabs.currentChanged.connect(self.refresh_info)
        layout.addWidget(self.tabs, 3)

        self._add_welcome_tab()
        self.resize(800, 600)

    def _add_welcome_tab(self):
        lbl = ScaledLabel()
        img = QPixmap(os.path.join(os.path.dirname(__file__), 'media', 'Uniandes.png'))
        if not img.isNull():
            lbl.setPixmap(img)
        else:
            lbl.setText("Uniandes.png not found.")
        lbl.metadata = {'type': 'welcome'}
        self.tabs.addTab(lbl, "Welcome")

    def open_file(self):
        dlg = QFileDialog(self, "Open File")
        dlg.setNameFilter("Images (*.png *.jpg *.jpeg *.tiff *.tif);;DICOM (*.dcm);;All Files (*)")
        if dlg.exec():
            path = dlg.selectedFiles()[0]
            self._add_image_tab(path)

    def _add_image_tab(self, path):
        lbl = ScaledLabel()
        pix = QPixmap(path)
        if not pix.isNull() and not path.lower().endswith('.dcm'):
            lbl.setPixmap(pix)
            lbl.metadata = {'type': 'image', 'file': path, 'dims': f"{pix.width()}x{pix.height()}"}
        else:
            pixmap, info = self._load_dicom(path)
            if not pixmap.isNull():
                lbl.setPixmap(pixmap)
                lbl.metadata = {'type':'dicom', 'file':path, **info}
            else:
                lbl.setText(f"Cannot load: {os.path.basename(path)}")
                lbl.metadata = {'type':'unknown','file':path}
        self.tabs.addTab(lbl, os.path.basename(path))
        self.tabs.setCurrentIndex(self.tabs.count()-1)

    def open_isodoses(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Dose Map", filter="Dose Maps (*.dcm *.npy)")
        if path:
            self._add_isodose_tab(path)


    def _add_isodose_tab(self, path):
        widget = IsodoseWidget(path)
        # Para asegurarnos de que canvas reciba eventos:
        widget.setFocusPolicy(Qt.ClickFocus)
        widget.setFocus()

        tab_name = f"Isodose: {os.path.basename(path)}"
        self.tabs.addTab(widget, tab_name)
        self.tabs.setCurrentIndex(self.tabs.count()-1)


    def _load_dicom(self, path):
        from pydicom.filereader import InvalidDicomError
        try:
            ds = pydicom.dcmread(path)
        except (InvalidDicomError, FileNotFoundError):
            return QPixmap(), {}
        info = {k: str(ds.get(k,'N/A')) for k in [
            'PatientName','PatientID','Modality','Rows','Columns','DoseGridScaling'
        ]}
        arr = ds.pixel_array
        norm = (arr.astype(float) - arr.min())/(arr.max()-arr.min())*255 if arr.max()!=arr.min() else np.zeros_like(arr)
        img = QImage(norm.astype(np.uint8).data, norm.shape[1], norm.shape[0], norm.shape[1], QImage.Format_Grayscale8)
        return QPixmap.fromImage(img), info

    def refresh_info(self, idx):
        w = self.tabs.widget(idx)
        md = getattr(w, 'metadata', {})
        t = md.get('type','')
        if t=='welcome': self.info_label.setText("Welcome to FilmQADose")
        elif t=='image': self.info_label.setText(f"Image: {os.path.basename(md['file'])}\nDims: {md['dims']}")
        elif t=='dicom': self.info_label.setText("DICOM: " + os.path.basename(md['file']))
        else: self.info_label.setText("")

if __name__=='__main__':
    app = QApplication(sys.argv)
    win = FilmQADoseMainWindow()
    win.show()
    sys.exit(app.exec())
