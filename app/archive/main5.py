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
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QLineEdit,
    QDialogButtonBox,
    QTabWidget,
    QSizePolicy
)

# Import widgets for Tools
from dose_analysis.isodoses3 import IsodoseWidget
from dose_analysis.dose_profile2 import DoseProfileWidget

class GammaDialog(QDialog):
    """
    Dialog to select reference and evaluation dose maps,
    and specify Dose Diff. and DTA parameters.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # Make dialog wider
        self.setMinimumWidth(400)
        self.setWindowTitle("Configure Gamma Analysis")
        self.ref_path = ''
        self.eval_path = ''
        self.dose_diff = 1
        self.dta = 1

        layout = QVBoxLayout(self)

        # Reference selector
        ref_layout = QHBoxLayout()
        self.ref_edit = QLineEdit()
        self.ref_edit.setPlaceholderText('Reference map (.dcm or .npy)')
        btn_ref = QPushButton('Browse Reference')
        btn_ref.clicked.connect(self.browse_reference)
        ref_layout.addWidget(self.ref_edit)
        ref_layout.addWidget(btn_ref)
        layout.addLayout(ref_layout)

        # Evaluation selector
        eval_layout = QHBoxLayout()
        self.eval_edit = QLineEdit()
        self.eval_edit.setPlaceholderText('Evaluation map (.dcm or .npy)')
        btn_eval = QPushButton('Browse Evaluation')
        btn_eval.clicked.connect(self.browse_evaluation)
        eval_layout.addWidget(self.eval_edit)
        eval_layout.addWidget(btn_eval)
        layout.addLayout(eval_layout)

        # Dose Difference parameter
        dose_diff_layout = QHBoxLayout()
        lbl_diff = QLabel('Dose Diff.:')
        self.spin_diff = QSpinBox()
        self.spin_diff.setMinimum(1)
        self.spin_diff.setValue(1)
        # Suffix and narrower width
        self.spin_diff.setSuffix('%')
        self.spin_diff.setFixedWidth(90)
        dose_diff_layout.addWidget(lbl_diff)
        dose_diff_layout.addWidget(self.spin_diff)
        layout.addLayout(dose_diff_layout)

        # DTA parameter
        dta_layout = QHBoxLayout()
        lbl_dta = QLabel('DTA:')
        self.spin_dta = QSpinBox()
        self.spin_dta.setMinimum(1)
        self.spin_dta.setValue(1)
        # Suffix and narrower width
        self.spin_dta.setSuffix('mm')
        self.spin_dta.setFixedWidth(90)
        dta_layout.addWidget(lbl_dta)
        dta_layout.addWidget(self.spin_dta)
        layout.addLayout(dta_layout)

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def browse_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Dose Map", filter="Dose Maps (*.dcm *.npy)"
        )
        if path:
            self.ref_path = path
            self.ref_edit.setText(os.path.basename(path))

    def browse_evaluation(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Evaluation Dose Map", filter="Dose Maps (*.dcm *.npy)"
        )
        if path:
            self.eval_path = path
            self.eval_edit.setText(os.path.basename(path))


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
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled)
        else:
            super().setPixmap(QPixmap())


class FilmQADoseMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FilmQADose")

        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        file_menu.addAction(QAction("Open...", self, triggered=self.open_file))

        # Calibration placeholder
        menubar.addMenu("Calibration")

        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        tools_menu.addAction(QAction("Isodoses", self, triggered=self.open_isodoses))
        tools_menu.addAction(QAction("Profiles", self, triggered=self.open_profiles))
        tools_menu.addAction(QAction("Gamma", self, triggered=self.open_gamma))

        # Central layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left info panel
        self.info_label = QLabel("Welcome to FilmQADose")
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.info_label)
        layout.addLayout(left_layout, 1)

        # Right tabs
        self.tabs = QTabWidget()
        self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(lambda i: self.tabs.removeTab(i))
        self.tabs.currentChanged.connect(self.refresh_info)
        layout.addWidget(self.tabs, 3)

        self._add_welcome_tab()
        self.resize(800, 600)

    def _add_welcome_tab(self):
        lbl = ScaledLabel()
        img_path = os.path.join(os.path.dirname(__file__), 'media', 'Uniandes.png')
        pix = QPixmap(img_path)
        if not pix.isNull():
            lbl.setPixmap(pix)
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
                metadata = {'type': 'dicom', 'file': path}
                metadata.update(info)
                lbl.metadata = metadata
            else:
                lbl.setText(f"Cannot load: {os.path.basename(path)}")
                lbl.metadata = {'type': 'unknown', 'file': path}
        self.tabs.addTab(lbl, os.path.basename(path))
        self.tabs.setCurrentIndex(self.tabs.count()-1)

    def open_isodoses(self):
        dlg = QFileDialog(self, "Select Dose Map for Isodoses", filter="Dose Maps (*.dcm *.npy)")
        if dlg.exec():
            path = dlg.selectedFiles()[0]
            self._add_isodose_tab(path)

    def _add_isodose_tab(self, path):
        widget = IsodoseWidget(path)
        widget.setFocusPolicy(Qt.ClickFocus)
        widget.setFocus()
        self.tabs.addTab(widget, f"Isodose: {os.path.basename(path)}")
        self.tabs.setCurrentIndex(self.tabs.count()-1)

    def open_profiles(self):
        dlg = QFileDialog(self, "Select Dose Map for Profiles", filter="Dose Maps (*.dcm *.npy)")
        if dlg.exec():
            path = dlg.selectedFiles()[0]
            self._add_profiles_tab(path)

    def _add_profiles_tab(self, path):
        widget = DoseProfileWidget(path)
        widget.setFocusPolicy(Qt.ClickFocus)
        widget.setFocus()
        self.tabs.addTab(widget, f"Profiles: {os.path.basename(path)}")
        self.tabs.setCurrentIndex(self.tabs.count()-1)

    def open_gamma(self):
        dialog = GammaDialog(self)
        if dialog.exec() == QDialog.Accepted:
            ref, eval_ = dialog.ref_path, dialog.eval_path
            if ref and eval_:
                self._add_gamma_tab(ref, eval_)

    def _add_gamma_tab(self, ref_path, eval_path):
        widget = QWidget()
        box = QVBoxLayout(widget)
        lbl = QLabel(
            f"Reference map: {os.path.basename(ref_path)}\n"
            f"Evaluation map: {os.path.basename(eval_path)}"
        )
        lbl.setAlignment(Qt.AlignCenter)
        box.addWidget(lbl)
        widget.metadata = {'type': 'gamma', 'reference': ref_path, 'evaluation': eval_path}
        tab_label = f"Gamma: {os.path.basename(ref_path)} vs {os.path.basename(eval_path)}"
        self.tabs.addTab(widget, tab_label)
        self.tabs.setCurrentIndex(self.tabs.count()-1)

    def _load_dicom(self, path):
        from pydicom.filereader import InvalidDicomError
        try:
            ds = pydicom.dcmread(path)
        except (InvalidDicomError, FileNotFoundError):
            return QPixmap(), {}
        info = {k: str(ds.get(k, 'N/A')) for k in [
            'PatientName', 'PatientID', 'Modality', 'Rows', 'Columns', 'DoseGridScaling'
        ]}
        arr = ds.pixel_array
        norm = ((arr.astype(float) - arr.min()) / (arr.max() - arr.min()) * 255
                if arr.max()!=arr.min() else np.zeros_like(arr))
        img = QImage(norm.astype(np.uint8).data,
                     norm.shape[1], norm.shape[0], norm.shape[1],
                     QImage.Format_Grayscale8)
        return QPixmap.fromImage(img), info

    def refresh_info(self, idx):
        w = self.tabs.widget(idx)
        md = getattr(w, 'metadata', {})
        t = md.get('type', '')
        if t == 'welcome':
            self.info_label.setText("Welcome to FilmQADose")
        elif t == 'image':
            self.info_label.setText(
                f"Image: {os.path.basename(md['file'])}\nDims: {md['dims']}"
            )
        elif t == 'dicom':
            self.info_label.setText(
                "DICOM: " + os.path.basename(md['file'])
            )
        elif t == 'gamma':
            self.info_label.setText(
                f"Gamma: {os.path.basename(md['reference'])} vs {os.path.basename(md['evaluation'])}"
            )
        else:
            self.info_label.setText("")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = FilmQADoseMainWindow()
    win.show()
    sys.exit(app.exec())
