# widgets/new_calibration.py

import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph import RectROI
from pyqtgraph.Qt import QtCore, QtWidgets
from PIL import Image
from datetime import datetime
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QWidget, QMessageBox
)
from calibration.calibration import FilmCalibration


class CalibrationDialog(QDialog):
    """
    Diálogo para seleccionar una sola película (.tif)
    y elegir el tipo de calibración: 'polynomial' o 'rational'.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.default_dir = os.path.join(os.path.dirname(__file__), "..", "media")
        self.setWindowTitle("New Calibration")
        self.calibration_path = ""
        self.calib_type = ""

        layout = QVBoxLayout(self)

        # Selector de un solo archivo
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Calibration film (.tif):"))
        self.path_edit = QLineEdit(self)
        file_layout.addWidget(self.path_edit)
        browse_btn = QPushButton("Browse…", self)
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(browse_btn)
        layout.addLayout(file_layout)

        # Combobox tipo
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Calibration type:"))
        self.type_combo = QComboBox(self)
        self.type_combo.addItems(["polynomial", "rational"])
        type_layout.addWidget(self.type_combo)
        layout.addLayout(type_layout)

        # Botones
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def browse_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Calibration Film",
            self.default_dir,
            "TIFF File (*.tif *.tiff)"
        )
        if file:
            self.calibration_path = file
            self.path_edit.setText(file)

    def _on_accept(self):
        if not self.calibration_path:
            QMessageBox.warning(self, "No file", "Please select a calibration film.")
            return
        self.calib_type = self.type_combo.currentText()
        self.accept()



class CalibrationWidget(QWidget):
    """
    Interactive widget to mark ROIs on the selected .tif film
    and assign a dose to each, with an embedded calibration curve display.
    """
    def __init__(self, tif_path, calib_type, parent=None):
        super().__init__(parent)
        self.tif_path = tif_path
        self.calib_type = calib_type
        self.rois = []
        self.labels = []
        self.roi_data = []

        main_layout = QHBoxLayout(self)
        view_layout = QVBoxLayout()
        ctrl_layout = QVBoxLayout()

        # Graphics view for image + ROIs
        self.gv = pg.GraphicsView()
        self.vb = pg.ViewBox()
        self.gv.setCentralItem(self.vb)
        view_layout.addWidget(self.gv)

        # Calibrate button
        self.calibrate_btn = QPushButton("Calibrate", self)
        self.calibrate_btn.clicked.connect(self.on_calibrate)
        ctrl_layout.addWidget(self.calibrate_btn)
        ctrl_layout.addStretch()

        main_layout.addLayout(view_layout, stretch=3)
        main_layout.addLayout(ctrl_layout, stretch=1)

        # Configure ViewBox
        self.vb.setAspectLocked(True)
        self.vb.invertY()
        self.vb.setMouseEnabled(False, False)
        self.vb.setMenuEnabled(False)
        self.vb.wheelEvent = lambda ev: None

        # Load TIFF
        arr = np.flipud(np.rot90(np.asarray(Image.open(self.tif_path))))
        self.image_item = pg.ImageItem(arr)
        self.vb.addItem(self.image_item)

        # ROI click handler
        self.vb.scene().sigMouseClicked.connect(self.on_click)

    def ask_dose(self, default_value=100):
        dose, ok = QtWidgets.QInputDialog.getInt(
            self, "Enter dose", "Associated dose (cGy):",
            value=default_value, minValue=0
        )
        return ok, int(dose)

    def refresh_roi_data(self):
        self.roi_data = []
        for roi in self.rois:
            p, s = roi.pos(), roi.size()
            self.roi_data.append((
                int(p.x()), int(p.y()),
                int(s.x()), int(s.y()), roi.dose
            ))
        print("ROI data:", self.roi_data)

    def add_roi(self, pos, size=(100, 100)):
        ok, dose = self.ask_dose()
        if not ok:
            return
        bounds = self.image_item.boundingRect()
        roi = RectROI(pos, size, pen=pg.mkPen('r', width=2), maxBounds=bounds)
        roi.addScaleHandle([1, 1], [0, 0])
        roi.addScaleHandle([0, 0], [1, 1])
        self.vb.addItem(roi)
        roi.dose = dose

        label = pg.TextItem(text=str(dose), color=(255, 0, 0), anchor=(0, 1))
        label.setPos(*pos)
        self.vb.addItem(label)

        roi.sigRegionChanged.connect(lambda: label.setPos(*roi.pos()))
        roi.sigRegionChangeFinished.connect(lambda: self.on_roi_modified(roi, label))

        self.rois.append(roi)
        self.labels.append(label)
        self.refresh_roi_data()

    def on_roi_modified(self, roi, label):
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("ROI dose")
        msg.setText("The ROI has changed. What would you like to do?")
        change_btn = msg.addButton("Change dose", QtWidgets.QMessageBox.ActionRole)
        delete_btn = msg.addButton("Delete ROI", QtWidgets.QMessageBox.DestructiveRole)
        keep_btn   = msg.addButton("Keep",     QtWidgets.QMessageBox.AcceptRole)
        msg.exec()

        if msg.clickedButton() == change_btn:
            ok, new_dose = self.ask_dose(default_value=roi.dose)
            if ok:
                roi.dose = new_dose
                label.setText(str(new_dose))
        elif msg.clickedButton() == delete_btn:
            self.vb.removeItem(roi)
            self.vb.removeItem(label)
            self.rois.remove(roi)
            self.labels.remove(label)
        self.refresh_roi_data()

    def on_click(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            view_pos = self.vb.mapSceneToView(event.scenePos())
            rect = self.image_item.boundingRect()
            w, h = 100, 100
            x0 = min(max(view_pos.x(), rect.left()),  rect.right()  - w)
            y0 = min(max(view_pos.y(), rect.top()),   rect.bottom() - h)
            self.add_roi((x0, y0), size=(w, h))

    def on_calibrate(self):
        # Instantiate calibration
        calibration = FilmCalibration(self.tif_path, fitting_function_name=self.calib_type)
        # Add each ROI
        for x, y, w, h, dose in self.roi_data:
            calibration.add_roi(dose, x, y, w, h)
        print(calibration.get_rois_by_dose())
        # Perform calibration
        calibration.calibrate()
        # Generate embedded plot
        fig = calibration.graph_calibration_curve()
        # Save JSON with current date
        date_str = datetime.now().strftime("%Y%m%d")
        json_path = os.path.join(os.path.dirname(self.tif_path), "outputs", f"calibration_{date_str}.json")
        calibration.to_json(json_path)
        # Notify user
        QMessageBox.information(self, "Calibration Saved", f"Calibration saved to {json_path}")
        # Replace view with matplotlib canvas
        # Clear old items
        for roi in self.rois:
            self.vb.removeItem(roi)
        for lbl in self.labels:
            self.vb.removeItem(lbl)
        self.vb.removeItem(self.image_item)
        # Embed figure
        canvas = FigureCanvas(fig)
        # Replace GraphicsView with canvas
        parent_layout = self.layout()
        # Remove widgets from first position
        old_widget = parent_layout.itemAt(0).widget()
        parent_layout.replaceWidget(old_widget, canvas)
        old_widget.setParent(None)
        self.vb = None
        self.gv = None
        canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        parent_layout.insertWidget(0, canvas, 3)
