import sys
import os
import cv2
import numpy as np
import pyqtgraph as pg
from pathlib import Path
from io import BytesIO
from pyqtgraph import RectROI
from pydicom import dcmread
from pydicom.filereader import InvalidDicomError
from PIL import Image
from datetime import datetime
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QPixmap, QImage, QIcon, QFont
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
    QDialog,
)
from widgets.isodoses import IsodoseWidget   
from widgets.dose_profile import DoseProfilesWidget
from widgets.gamma import GammaWidget, GammaDialog
from widgets.template_matching import TemplateDialog
from widgets.new_calibration import CalibrationDialog, CalibrationWidget  
from calibration.image_processing import template_matching

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
        icon_path = os.path.join(os.path.dirname(__file__), "media", "logo square.svg")
        self.setWindowIcon(QIcon(icon_path))

        self.default_dir = os.path.join(os.path.dirname(__file__), "media")
        self.output_folder = os.path.join(self.default_dir, "outputs")


        # --- Create Menu Bar ---
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        # "Open..." Action
        open_action = QAction("Open...", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        # --- Calibration menu ---
        calibration_menu = menubar.addMenu("Calibration")
        # TODO:
        calibration_menu.addAction(QAction("New Calibration", self, triggered=self.new_calibration))
        # TODO LATER: Load calibration


        # --- Tools menu ---
        tools_menu = menubar.addMenu("Tools")
        tools_menu.addAction(QAction("Isodoses", self, triggered=self.open_isodoses))
        tools_menu.addAction(QAction("Profiles", self, triggered=self.open_profiles))
        tools_menu.addAction(QAction("Gamma", self, triggered=self.open_gamma))
        tools_menu.addAction(QAction("Crop with TM", self, triggered=self.open_crop_with_tm))



        # --- View menu ---
        view_menu = menubar.addMenu("View")
        fullscreen_action = QAction("Toggle Fullscreen", self)
        fullscreen_action.setShortcut("F11")
        fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(fullscreen_action)
        self.addAction(fullscreen_action)  # Enables global shortcut even if menu not focused

        # --- Main Layout Setup ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Panel
        self.left_panel = QVBoxLayout()
        self.info_label = QLabel("") # Default text
        self.left_panel.addWidget(self.info_label)

        # Right Panel as a Tab Widget
        self.tab_widget = QTabWidget()
        # Make tabs closable
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        # Detect whenever the current tab changes
        self.tab_widget.currentChanged.connect(self.update_left_panel)

        main_layout.addLayout(self.left_panel, 1)
        main_layout.addWidget(self.tab_widget, 4)

        # Add default "Welcome" tab
        self.add_welcome_tab()

        # Set an initial window size
        self.resize(800, 600)

    def open_crop_with_tm(self):
        dialog = TemplateDialog(self)
        if dialog.exec() == QDialog.Accepted and dialog.template_path and dialog.film_path:
            tps_path = dialog.template_path
            film_path = dialog.film_path

            # Crear output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_folder, f"crop_tm_{timestamp}.tif")

            self.add_tab_crop_with_tm(tps_path, film_path, output_path, timestamp)
    
    def add_tab_crop_with_tm(self, tps_path, film_path, output_path, timestamp):
        """Create a default welcome tab showing the logo"""
        tm_label = ScaledLabel()

        template_matching(tps_path, film_path, output_path)

        tm_pixmap = QPixmap(output_path)

        if not tm_pixmap.isNull():
            tm_label.setPixmap(tm_pixmap)
        else:
            tm_label.setText(f"{output} not found or could not be loaded.")

        tm_label.metadata = {
            "type": "crop_tm",
            "file_path": output_path,
            "tps": os.path.basename(tps_path),
            "film": os.path.basename(film_path)
        }

        tab_name = f"Crop TM ({timestamp})"
        self.tab_widget.addTab(tm_label, tab_name)
        self.tab_widget.setCurrentWidget(tm_label)


    def add_welcome_tab(self):
        """Create a default welcome tab showing the logo"""
        welcome_label = ScaledLabel()

        #image_path = os.path.join(os.path.dirname(__file__), 'media', "Uniandes.png")
        logo_name = "logo full without bg.svg"
        image_path = os.path.join(os.path.dirname(__file__), 'media', logo_name)
        welcome_pixmap = QPixmap(image_path)

        if not welcome_pixmap.isNull():
            welcome_label.setPixmap(welcome_pixmap)
            #print("Loaded logo")
        else:
            print("Could not load logo")
            welcome_label.setText(f"{logo_name} not found or could not be loaded.")

        # Store a basic metadata for the welcome tab
        welcome_label.metadata = {
            "type": "welcome"
        }

        self.tab_widget.addTab(welcome_label, "Welcome")

    def open_gamma(self):
        dialog = GammaDialog(self)
        if dialog.exec() == QDialog.Accepted and dialog.ref_path and dialog.eval_path:
            ref = dialog.ref_path
            eval_ = dialog.eval_path
            dd = dialog.spin_diff.value()
            dta = dialog.spin_dta.value()
            self.add_gamma_tab(ref, eval_, dd, dta)

    def add_gamma_tab(self, ref_path, eval_path, dose_diff, dta):
        # Show waiting message in the tab
        placeholder = QWidget()
        box = QVBoxLayout(placeholder)
        wait_lbl = QLabel("Calculating gamma analysis, please wait...")
        wait_lbl.setAlignment(Qt.AlignCenter)
        box.addWidget(wait_lbl)
        tab_label = f"Gamma: {os.path.basename(ref_path)} vs {os.path.basename(eval_path)}"
        index = self.tab_widget.addTab(placeholder, tab_label)
        self.tab_widget.setCurrentIndex(index)

        # Ejecutar cálculo tras breve retardo para renderizar mensaje
        from PySide6.QtCore import QTimer
        QTimer.singleShot(100, lambda: self.replace_gamma_tab(
            index, ref_path, eval_path, dose_diff, dta
        ))

    def replace_gamma_tab(self, index, ref_path, eval_path, dose_diff, dta):
        # Crear y mostrar el widget definitivo
        gamma_widget = GammaWidget(ref_path, eval_path, dose_diff, dta)
        gamma_widget.setFocusPolicy(Qt.ClickFocus)
        gamma_widget.setFocus()

        tab_label = f"Gamma: {os.path.basename(ref_path)} vs {os.path.basename(eval_path)}"
        # Reemplazar la pestaña de espera
        self.tab_widget.removeTab(index)
        self.tab_widget.insertTab(index, gamma_widget, tab_label)
        self.tab_widget.setCurrentIndex(index)


    def open_isodoses(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Dose Map",
            self.default_dir, 
            filter="Dose Maps (*.dcm *.npy)"
        )
        if path:
            self.add_isodose_tab(path)

    def add_isodose_tab(self, path):
        isodose_widget = IsodoseWidget(path)
        # Para asegurarnos de que canvas reciba eventos:
        isodose_widget.setFocusPolicy(Qt.ClickFocus)
        isodose_widget.setFocus()

        # Añadir metadata para que update_left_panel pueda mostrar algo
        isodose_widget.metadata = {
            "type": "isodose",
            "file_path": path,
            "filename": os.path.basename(path)
        }

        tab_name = f"Isodose: {os.path.basename(path)}"
        self.tab_widget.addTab(isodose_widget, tab_name)
        self.tab_widget.setCurrentWidget(isodose_widget)


    def open_profiles(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Dose Map for Profiles",
            self.default_dir, 
            filter="Dose Maps (*.dcm *.npy)"
        )
        if path:
            self.add_profiles_tab(path)

    def add_profiles_tab(self, path):
        # Instantiate the profiles widget
        profiles_widget = DoseProfilesWidget(path)
        profiles_widget.setFocusPolicy(Qt.ClickFocus)
        profiles_widget.setFocus()

        # Añadir metadata para que update_left_panel pueda mostrar algo
        profiles_widget.metadata = {
            "type": "profiles",
            "file_path": path,
            "filename": os.path.basename(path)
        }

        tab_name = f"Profiles: {os.path.basename(path)}"
        self.tab_widget.addTab(profiles_widget, tab_name)
        self.tab_widget.setCurrentWidget(profiles_widget)

    def new_calibration(self):
        dialog = CalibrationDialog(self)
        if dialog.exec() == QDialog.Accepted:
            tif_path = dialog.calibration_path
            calib_type = dialog.calib_type
            self.add_new_calibration_tab(tif_path, calib_type)

    def add_new_calibration_tab(self, tif_path, calib_type):
        """
        Crea una nueva pestaña con el CalibrationWidget
        para la única película .tif y el tipo de calibración.
        """
        calib_widget = CalibrationWidget(tif_path, calib_type)
        calib_widget.metadata = {
            "type": "calibration",
            "file": os.path.basename(tif_path),
            "calibration_type": calib_type
        }
        tab_name = f"Calibration ({calib_type})"
        self.tab_widget.addTab(calib_widget, tab_name)
        self.tab_widget.setCurrentWidget(calib_widget)

    def add_welcome_tab(self):
        """Create a default welcome tab showing the logo"""
        welcome_label = ScaledLabel()

        #image_path = os.path.join(os.path.dirname(__file__), 'media', "Uniandes.png")
        logo_name = "logo full without bg.svg"
        image_path = os.path.join(os.path.dirname(__file__), 'media', logo_name)
        welcome_pixmap = QPixmap(image_path)

        if not welcome_pixmap.isNull():
            welcome_label.setPixmap(welcome_pixmap)
            #print("Loaded logo")
        else:
            print("Could not load logo")
            welcome_label.setText(f"{logo_name} not found or could not be loaded.")

        # Store a basic metadata for the welcome tab
        welcome_label.metadata = {
            "type": "welcome"
        }

        self.tab_widget.addTab(welcome_label, "Welcome")

    def open_file(self):
        """Prompt for a file and display it in a new tab."""
        file_dialog = QFileDialog(self, "Open File", self.default_dir)
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
        try:
            ds = dcmread(file_path)
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
        """
        Called whenever the current tab changes.
        Retrieve the widget’s metadata and display
        the relevant info on the left panel.
        """
        widget = self.tab_widget.widget(index)
        if not widget or not hasattr(widget, "metadata"):
            self.info_label.setText("No data available.")
            return

        md = widget.metadata
        tab_type = md.get("type", "unknown")

        if tab_type == "welcome":
            self.info_label.setText("Welcome to FilmQADose!")
        elif tab_type == "image":
            file_path = md.get("file_path", "N/A")
            dims = md.get("dimensions", "N/A")
            text = (
                f"<b>Image File:</b> {os.path.basename(file_path)}<br>"
                f"<b>Dimensions:</b> {dims}"
            )
            self.info_label.setText(text)
        elif tab_type == "dicom":
            # We'll display all relevant fields in HTML format
            text = (
                f"<b>DICOM File:</b> {os.path.basename(md.get('file_path', 'N/A'))}<br>"
                f"<b>Dimensions:</b> {md.get('dimensions','N/A')}<br>"
                f"<b>Patient Name:</b> {md.get('PatientName','N/A')}<br>"
                f"<b>Patient ID:</b> {md.get('PatientID','N/A')}<br>"
                f"<b>Modality:</b> {md.get('Modality','N/A')}<br>"
                f"<b>Content Date:</b> {md.get('ContentDate','N/A')}<br>"
                f"<b>Content Time:</b> {md.get('ContentTime','N/A')}<br>"
                f"<b>Allergies:</b> {md.get('Allergies','N/A')}<br>"
                f"<b>Samples/Pixel:</b> {md.get('SamplesPerPixel','N/A')}<br>"
                f"<b>Photometric Interpretation:</b> {md.get('PhotometricInterpretation','N/A')}<br>"
                f"<b>Rows:</b> {md.get('Rows','N/A')}<br>"
                f"<b>Columns:</b> {md.get('Columns','N/A')}<br>"
                f"<b>Pixel Spacing:</b> {md.get('PixelSpacing','N/A')}<br>"
                f"<b>Bits Allocated:</b> {md.get('BitsAllocated','N/A')}<br>"
                f"<b>Bits Stored:</b> {md.get('BitsStored','N/A')}<br>"
                f"<b>High Bit:</b> {md.get('HighBit','N/A')}<br>"
                f"<b>Pixel Representation:</b> {md.get('PixelRepresentation','N/A')}<br>"
                f"<b>Dose Units:</b> {md.get('DoseUnits','N/A')}<br>"
                f"<b>Normalization Point:</b> {md.get('NormalizationPoint','N/A')}<br>"
                f"<b>Dose Grid Scaling:</b> {md.get('DoseGridScaling','N/A')}<br>"
                f"<b>Isocenter Position:</b> {md.get('IsocenterPosition','N/A')}"
            )
            self.info_label.setText(text)
        
        elif tab_type == "isodose":
            file_path = md.get("file_path", "N/A")
            text = (
                f"<b>Isodose Map:</b> {os.path.basename(file_path)}<br>"
                f"<b>Type:</b> {os.path.splitext(file_path)[1][1:].upper()} file"
            )
            self.info_label.setText(text)

        elif tab_type == "profiles":
            file_path = md.get("file_path", "N/A")
            text = (
                f"<b>Dose Map:</b> {os.path.basename(file_path)}<br>"
                f"<b>Type:</b> {os.path.splitext(file_path)[1][1:].upper()} file"
            )
            self.info_label.setText(text)

        elif tab_type == "gamma":
            ref = os.path.basename(md.get("reference", "N/A"))
            eval_ = os.path.basename(md.get("evaluation", "N/A"))
            dd = md.get("dose_diff", "N/A")
            dta = md.get("dta", "N/A")
            pr = md.get("pass_rate", "N/A")

            text = (
                f"<b>Reference:</b> {ref}<br>"
                f"<b>Evaluation:</b> {eval_}<br>"
                f"<b>Dose Difference:</b> {dd:.1f}%<br>"
                f"<b>DTA:</b> {dta:.1f} mm<br>"
                f"<b>Pass rate:</b> {pr:.3f}%"
            )
            self.info_label.setText(text)

        elif tab_type == "crop_tm":
            full_path = md.get('file_path', 'N/A')
            p = Path(full_path)
            # Obtener los últimos 3 elementos de la ruta
            last_parts = Path(*p.parts[-3:])
            text = (
                f"<b>Crop with Template Matching</b><br>"
                f"<b>TPS:</b> {md.get('tps', 'N/A')}<br>"
                f"<b>Film:</b> {md.get('film', 'N/A')}<br>"
                f"<b>Saved to:</b><br>{last_parts.as_posix()}"
            )
            self.info_label.setText(text)

        elif tab_type == "calibration":
            # Nombre de la película usada
            film = md.get("file", "N/A")
            calib_type = md.get("calibration_type", "N/A")
            text = (
                f"<b>Calibration film:</b> {film}<br>"
                    f"<b>Type:</b> {calib_type}"
            )
            # Si ya se ha guardado, mostrar también la ruta del JSON
            if md.get("json_path"):
                saved = os.path.basename(md["json_path"])
                text += f"<br><b>Saved to:</b> {saved}"
            self.info_label.setText(text)


        else:
            # Unknown or unhandled file
            file_path = md.get("file_path", "N/A")
            self.info_label.setText(f"Could not load file: {file_path}")

    def close_tab(self, index):
        """Close the tab at the given index."""
        self.tab_widget.removeTab(index)

    def toggle_fullscreen(self):
        """Toggle fullscreen mode with F11."""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 12))
    window = FilmQADoseMainWindow()
    window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
