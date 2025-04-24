import sys
import os
import numpy as np
import pydicom
from PIL import Image

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
    QSizePolicy,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem
)

import pyqtgraph as pg
from pyqtgraph import RectROI

class CalibrationWidget(QWidget):
    def __init__(self, tiff_path, parent=None):
        super().__init__(parent)
        self.tiff_path = tiff_path
        self.rois = []
        self.roi_data = []

        layout = QVBoxLayout(self)
        self.pg_view = pg.GraphicsLayoutWidget()
        layout.addWidget(self.pg_view)

        self.vb = self.pg_view.addViewBox(lockAspect=True)
        self.vb.invertY()

        arr = np.rot90(np.asarray(Image.open(tiff_path)))
        self.image_item = pg.ImageItem(arr)
        self.vb.addItem(self.image_item)

        self.vb.scene().sigMouseClicked.connect(self.on_click)

    def on_click(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.scenePos()
            vb_pos = self.vb.mapSceneToView(pos)
            bounds = self.image_item.boundingRect()
            w, h = 100, 100
            x0 = min(max(vb_pos.x(), bounds.left()), bounds.right() - w)
            y0 = min(max(vb_pos.y(), bounds.top()), bounds.bottom() - h)
            self.add_roi((x0, y0), (w, h))

    def add_roi(self, pos, size):
        roi = RectROI(pos, size, pen=pg.mkPen('r', width=2), maxBounds=self.image_item.boundingRect())
        roi.addScaleHandle([1,1], [0,0])
        roi.addScaleHandle([0,0], [1,1])
        self.vb.addItem(roi)
        roi.sigRegionChangeFinished.connect(self.update_roi_data)
        self.rois.append(roi)
        self.update_roi_data()

    def update_roi_data(self):
        self.roi_data = []
        for roi in self.rois:
            p, s = roi.pos(), roi.size()
            self.roi_data.append((int(p.x()), int(p.y()), int(s.x()), int(s.y())))
        # Emit a custom signal or simply rely on the main window polling


class ScaledLabel(QLabel):
    """
    A QLabel subclass that scales its pixmap to fit its current size,
    preserving aspect ratio. It does NOT force the window to grow,
    thanks to size policy settings.

    We also add a `metadata` dict to store info about the displayed file:
      - type: "dicom" or "image" or "welcome" or "unknown"
      - various DICOM/image info
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        # Prevent forcing a large minimum size on the parent
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        # Here we store info about the image/file
        self.metadata = {}
        self._original_pixmap = None

    def setPixmap(self, pixmap: QPixmap):
        """Store the original pixmap and update the displayed pixmap."""
        self._original_pixmap = pixmap
        self.updateScaledPixmap()

    def resizeEvent(self, event):
        """Whenever the label is resized, scale the pixmap."""
        super().resizeEvent(event)
        self.updateScaledPixmap()

    def updateScaledPixmap(self):
        """
        Scale the original pixmap to fit the label's current size,
        preserving aspect ratio.
        """
        if self._original_pixmap and not self._original_pixmap.isNull():
            scaled = self._original_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled)
        else:
            super().setPixmap(QPixmap())  # Clear if there's no valid pixmap
class FilmQADoseMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FilmQADose")

        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        open_action = QAction("Open...", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        calibration_menu = menubar.addMenu("Calibration")
        new_calib_action = QAction("New Calibration", self)
        new_calib_action.triggered.connect(self.new_calibration)
        calibration_menu.addAction(new_calib_action)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Panel
        self.left_panel = QVBoxLayout()
        self.info_label = QLabel("Welcome to FilmQADose")
        self.left_panel.addWidget(self.info_label)

        # ROI table: hidden until calibration tab selected
        self.roi_table = QTableWidget()
        self.roi_table.setColumnCount(5)
        self.roi_table.setHorizontalHeaderLabels([
            "x_upperleft", "y_upperleft", "width", "height", "Dose (Gy)"
        ])
        self.roi_table.hide()
        self.left_panel.addWidget(self.roi_table)

        main_layout.addLayout(self.left_panel, 1)

        # Right: Tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        self.tab_widget.currentChanged.connect(self.update_left_panel)
        main_layout.addWidget(self.tab_widget, 3)

        self.add_welcome_tab()
        self.resize(800, 600)

    def new_calibration(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Calibration TIFF", "", "TIFF Files (*.tiff *.tif)"
        )
        if file_path:
            self.add_calibration_tab(file_path)

    def add_calibration_tab(self, tiff_path):
        calib = CalibrationWidget(tiff_path)
        # mark metadata for update_left_panel
        calib.metadata = {"type": "calibration", "file_path": tiff_path}
        name = os.path.basename(tiff_path)
        self.tab_widget.addTab(calib, f"Calib: {name}")
        self.tab_widget.setCurrentWidget(calib)

    def add_welcome_tab(self):
        """Create a default welcome tab showing Uniandes.png."""
        welcome_label = ScaledLabel()

        image_path = os.path.join(os.path.dirname(__file__), 'media', "Uniandes.png")
        #image_path = os.path.join(os.path.dirname(__file__), 'media', "logo full without bg.svg")
        print(image_path)
        welcome_pixmap = QPixmap(image_path)

        if not welcome_pixmap.isNull():
            welcome_label.setPixmap(welcome_pixmap)
            print("Loaded Uniandes.png")
        else:
            print("Could not load Uniandes.png")
            welcome_label.setText("Uniandes.png not found or could not be loaded.")

        # Store a basic metadata for the welcome tab
        welcome_label.metadata = {
            "type": "welcome"
        }

        self.tab_widget.addTab(welcome_label, "Welcome")

    def open_file(self):
        """Prompt for a file and display it in a new tab."""
        file_dialog = QFileDialog(self, "Open File")
        file_dialog.setNameFilter(
            "Images (*.png *.jpg *.jpeg *.tiff *.tif);;DICOM Files (*.dcm);;All Files (*)"
        )

        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            # For simplicity, handle one file at a time
            file_path = file_paths[0]
            self.add_file_tab(file_path)

    def add_file_tab(self, file_path):
        """
        Creates a new tab for the given file (image or DICOM).
        1) Try as a normal image.
        2) If extension is .dcm or if normal loading fails, try as DICOM.
        """
        label = ScaledLabel()
        pixmap = QPixmap(file_path)

        # Check if it's obviously a non-DICOM image
        if not pixmap.isNull() and not file_path.lower().endswith(".dcm"):
            # Valid non-DICOM image
            label.setPixmap(pixmap)
            label.metadata = {
                "type": "image",
                "file_path": file_path,
                "dimensions": f"{pixmap.width()} x {pixmap.height()}"
            }
        else:
            # Attempt DICOM load
            dcm_pixmap, dcm_info = self.load_dicom_as_pixmap(file_path)
            if not dcm_pixmap.isNull():
                label.setPixmap(dcm_pixmap)
                # Store all relevant DICOM info in metadata
                label.metadata = {
                    "type": "dicom",
                    "file_path": file_path,
                    # dimension from the QPixmap
                    "dimensions": f"{dcm_pixmap.width()} x {dcm_pixmap.height()}",
                    **dcm_info  # Merge in the extra DICOM fields
                }
            else:
                # Fallback if nothing worked
                label.setText(f"Could not load file: {os.path.basename(file_path)}")
                label.setAlignment(Qt.AlignCenter)
                label.metadata = {
                    "type": "unknown",
                    "file_path": file_path
                }

        filename = os.path.basename(file_path)
        self.tab_widget.addTab(label, filename)
        # Optionally switch to that new tab automatically
        self.tab_widget.setCurrentIndex(self.tab_widget.count() - 1)

    def load_dicom_as_pixmap(self, file_path):
        """
        Use pydicom to read DICOM pixel data and convert to a QPixmap.
        Returns: (QPixmap, dict_of_dicom_fields)
        """
        from pydicom.filereader import InvalidDicomError
        try:
            ds = pydicom.dcmread(file_path)
        except (InvalidDicomError, FileNotFoundError) as e:
            print(f"Error reading DICOM: {e}")
            return QPixmap(), {}

        # Extract DICOM metadata of interest
        # For tags that may not exist, we use .get(...) or fallback.
        dcm_info = {
            "PatientName": str(ds.get("PatientName", "N/A")),
            "PatientID": str(ds.get("PatientID", "N/A")),
            "Modality": str(ds.get("Modality", "N/A")),
            "ContentDate": str(ds.get("ContentDate", "N/A")),
            "ContentTime": str(ds.get("ContentTime", "N/A")),
            "Allergies": str(ds.get("Allergies", "N/A")),
            "SamplesPerPixel": str(ds.get("SamplesPerPixel", "N/A")),
            "PhotometricInterpretation": str(ds.get("PhotometricInterpretation", "N/A")),
            "Rows": str(ds.get("Rows", "N/A")),
            "Columns": str(ds.get("Columns", "N/A")),
            "PixelSpacing": str(ds.get("PixelSpacingg", "N/A")), # change for PixelSpacing if want to use it
            "BitsAllocated": str(ds.get("BitsAllocated", "N/A")),
            "BitsStored": str(ds.get("BitsStored", "N/A")),
            "HighBit": str(ds.get("HighBit", "N/A")),
            "PixelRepresentation": str(ds.get("PixelRepresentation", "N/A")),
            "DoseUnits": str(ds.get("DoseUnits", "N/A")),
            "NormalizationPoint": str(ds.get("NormalizationPoint", "N/A")),
            "DoseGridScaling": str(ds.get("DoseGridScaling", "N/A")),
            "IsocenterPosition": str(ds.get("IsocenterPosition", "N/A")),
        }

        try:
            arr = ds.pixel_array  # NumPy array
        except AttributeError:
            # No pixel data
            return QPixmap(), dcm_info

        # Convert pixel data to 8-bit QPixmap
        if len(arr.shape) == 2:
            # Grayscale
            arr_min, arr_max = float(arr.min()), float(arr.max())
            if arr_min == arr_max:
                # avoid divide-by-zero if uniform
                arr_normalized = np.zeros_like(arr, dtype=np.uint8)
            else:
                arr_normalized = (
                    (arr - arr_min) / (arr_max - arr_min) * 255.0
                ).astype(np.uint8)

            height, width = arr_normalized.shape
            qimage = QImage(
                arr_normalized.data,
                width,
                height,
                width,  # bytesPerLine
                QImage.Format_Grayscale8
            )
            return QPixmap.fromImage(qimage), dcm_info

        elif len(arr.shape) == 3 and arr.shape[2] == 3:
            # RGB
            height, width, channels = arr.shape
            arr_min, arr_max = float(arr.min()), float(arr.max())
            if arr_min != arr_max:
                arr_8bit = (
                    (arr - arr_min) / (arr_max - arr_min) * 255.0
                ).astype(np.uint8)
            else:
                arr_8bit = arr.astype(np.uint8)

            arr_8bit = np.ascontiguousarray(arr_8bit, dtype=np.uint8)
            qimage = QImage(
                arr_8bit.data,
                width,
                height,
                channels * width,
                QImage.Format_RGB888
            )
            return QPixmap.fromImage(qimage), dcm_info

        # Unhandled format
        return QPixmap(), dcm_info

    def update_left_panel(self, index):
        widget = self.tab_widget.widget(index)
        if not widget or not hasattr(widget, 'metadata'):
            self.info_label.setText("No data available.")
            self.roi_table.hide()
            return

        md = widget.metadata
        t = md.get('type', 'unknown')

        # Hide table by default
        self.roi_table.hide()

        if t == 'welcome':
            self.info_label.setText("Welcome to FilmQADose")

        elif t == 'calibration':
            # Show ROI table
            self.info_label.setText(f"Calibration: {os.path.basename(md.get('file_path',''))}")
            data = widget.roi_data
            self.roi_table.setRowCount(len(data))
            for row, (x, y, w, h) in enumerate(data):
                self.roi_table.setItem(row, 0, QTableWidgetItem(str(x)))
                self.roi_table.setItem(row, 1, QTableWidgetItem(str(y)))
                self.roi_table.setItem(row, 2, QTableWidgetItem(str(w)))
                self.roi_table.setItem(row, 3, QTableWidgetItem(str(h)))
                dose_item = QTableWidgetItem("")
                dose_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEditable | Qt.ItemIsEnabled)
                self.roi_table.setItem(row, 4, dose_item)
            self.roi_table.show()

        else:
            # Unknown or unhandled file
            file_path = md.get("file_path", "N/A")
            self.info_label.setText(f"Could not load file: {file_path}")

    def close_tab(self, index):
        """Close the tab at the given index."""
        self.tab_widget.removeTab(index)


def main():
    app = QApplication(sys.argv)
    window = FilmQADoseMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
