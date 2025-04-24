from PIL import Image
import numpy as np
import pyqtgraph as pg
from pyqtgraph import RectROI
from pyqtgraph.Qt import QtCore, QtWidgets
import os
import sys

# --- Qt Application ---------------------------------------------------------
app = pg.mkQApp()

# --- GraphicsView + ViewBox -------------------------------------------------
gv = pg.GraphicsView()
vb = pg.ViewBox()
gv.setCentralItem(vb)
gv.setWindowTitle("ROI editor with dose")
gv.show()

vb.setAspectLocked(True)
vb.invertY()
vb.setMouseEnabled(False, False)
vb.setMenuEnabled(False)
vb.wheelEvent = lambda ev: None       # block zoom with mouse wheel

# --- Image ------------------------------------------------------------------
path = os.path.join('media', 'Dosis0a10.tif')
arr = np.flipud(np.rot90(np.asarray(Image.open(path))))
image = pg.ImageItem(arr)
vb.addItem(image)

# --- Global structures ------------------------------------------------------
rois     = []  # RectROI objects
labels   = []  # TextItem showing the current dose
roi_data = []  # [(x, y, w, h, dose)]


# ---------------------------------------------------------------------------#

def ask_dose(default_value: int = 100) -> tuple[bool, int]:
    """Open a QInputDialog and return (ok, dose)."""
    dose, ok = QtWidgets.QInputDialog.getInt(
        gv,                       # parent window
        "Enter dose",            # dialog title
        "Associated dose (cGy):",  # label text
        value=default_value,      # default value (dynamic)
        minValue=0                # minimum allowed
    )
    return ok, int(dose)


def refresh_roi_data() -> None:
    """Rebuild the internal list and print it to the console."""
    global roi_data
    roi_data = []
    for roi in rois:
        p, s = roi.pos(), roi.size()
        roi_data.append((int(p.x()), int(p.y()),
                         int(s.x()), int(s.y()), roi.dose))
    print("ROI data:", roi_data)


def add_roi(pos, size=(100, 100)):
    """Create the RectROI, its label, and store its dose attribute."""
    # 1. Ask for the dose
    ok, dose = ask_dose()
    if not ok:  # user cancelled
        return None

    # 2. Create ROI
    bounds = image.boundingRect()
    roi = RectROI(pos, size,
                  pen=pg.mkPen('r', width=2),
                  maxBounds=bounds)
    roi.addScaleHandle([1, 1], [0, 0])
    roi.addScaleHandle([0, 0], [1, 1])
    vb.addItem(roi)

    # 3. Extra attribute
    roi.dose = dose

    # 4. Visual label (shows dose)
    label = pg.TextItem(text=str(roi.dose), color=(255, 0, 0),
                        anchor=(0, 1))  # top‑left corner of ROI
    label.setPos(pos[0], pos[1])
    vb.addItem(label)

    # 5. Connections and storage
    roi.sigRegionChanged.connect(lambda: label.setPos(*roi.pos()))
    roi.sigRegionChangeFinished.connect(lambda: on_roi_modified(roi, label))

    rois.append(roi)
    labels.append(label)
    refresh_roi_data()
    return roi


def on_roi_modified(roi, label):
    """Triggered after moving/scaling a ROI."""
    msg = QtWidgets.QMessageBox(gv)
    msg.setWindowTitle("ROI dose")
    msg.setText("The ROI has changed. What would you like to do?")
    change_btn = msg.addButton("Change dose", QtWidgets.QMessageBox.ActionRole)
    delete_btn = msg.addButton("Delete ROI",  QtWidgets.QMessageBox.DestructiveRole)
    keep_btn   = msg.addButton("Keep",        QtWidgets.QMessageBox.AcceptRole)
    msg.exec()

    pressed = msg.clickedButton()
    if pressed == change_btn:
        ok, new_dose = ask_dose(default_value=roi.dose)
        if ok:
            roi.dose = new_dose
            label.setText(str(roi.dose))
    elif pressed == delete_btn:
        # remove from the scene and from the lists
        vb.removeItem(roi)
        vb.removeItem(label)
        rois.remove(roi)
        labels.remove(label)
    # In any case, refresh the table
    refresh_roi_data()


def on_click(event):
    if event.button() == QtCore.Qt.LeftButton:
        view_pos = vb.mapSceneToView(event.scenePos())
        rect     = image.boundingRect()
        w, h     = 100, 100
        x0 = min(max(view_pos.x(), rect.left()),  rect.right()  - w)
        y0 = min(max(view_pos.y(), rect.top()),   rect.bottom() - h)
        add_roi((x0, y0), size=(w, h))


# Connect the click
tvb_scene = vb.scene()
tvb_scene.sigMouseClicked.connect(on_click)

# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    sys.exit(app.exec())
