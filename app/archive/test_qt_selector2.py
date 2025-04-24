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
gv.setWindowTitle("ROI editor con dosis")
gv.show()

vb.setAspectLocked(True)
vb.invertY()
vb.setMouseEnabled(False, False)
vb.setMenuEnabled(False)
vb.wheelEvent = lambda ev: None       # bloquea zoom con rueda

# --- Imagen -----------------------------------------------------------------
path = os.path.join('media', 'mama100cropped.tif')
arr  = np.rot90(np.asarray(Image.open(path)))
image = pg.ImageItem(arr)
vb.addItem(image)

# --- Estructuras globales ---------------------------------------------------
rois       = []           # RectROI
labels     = []           # TextItem con el número
roi_data   = []           # [(id, x, y, w, h, dose)]
_next_id   = 1            # contador global


# ---------------------------------------------------------------------------#
def pedir_dosis(valor_defecto=100):
    """Abre QInputDialog y devuelve (ok, dosis)."""

    # 2) con nombre correcto en PySide6
    dosis, ok = QtWidgets.QInputDialog.getInt(
        gv, "Ingresar dosis",
        "Dosis asociada (cGy):",
        value=100,
        minValue=0
    )
    return ok, int(dosis)


def actualizar_roi_data():
    """Regenera la tabla interna e imprime en consola."""
    global roi_data
    roi_data = []
    for roi in rois:
        p, s = roi.pos(), roi.size()
        roi_data.append((roi.uid, int(p.x()), int(p.y()),
                         int(s.x()), int(s.y()), roi.dose))
    print("ROI data:", roi_data)


def add_roi(pos, size=(100, 100)):
    """Crea el RectROI + etiqueta + atributos uid/dose."""
    global _next_id

    # 1. Preguntar dosis
    ok, dosis = pedir_dosis()
    if not ok:  # usuario canceló
        return None

    # 2. Crear ROI
    bounds = image.boundingRect()
    roi = RectROI(pos, size,
                  pen=pg.mkPen('r', width=2),
                  maxBounds=bounds)
    roi.addScaleHandle([1, 1], [0, 0])
    roi.addScaleHandle([0, 0], [1, 1])
    vb.addItem(roi)

    # 3. Atributos extras
    roi.uid  = _next_id
    roi.dose = dosis
    _next_id += 1

    # 4. Etiqueta visual
    label = pg.TextItem(text=str(roi.uid), color=(255, 0, 0),
                        anchor=(0, 1))  # arriba‑izquierda
    label.setPos(pos[0], pos[1])
    vb.addItem(label)

    # 5. Conexiones y almacenamiento
    roi.sigRegionChanged.connect(lambda: label.setPos(*roi.pos()))
    roi.sigRegionChangeFinished.connect(
        lambda: on_roi_modified(roi, label))
    rois.append(roi)
    labels.append(label)
    actualizar_roi_data()
    return roi


def on_roi_modified(roi, label):
    """Se dispara al terminar de mover/escalar una ROI."""
    msg = QtWidgets.QMessageBox(gv)
    msg.setWindowTitle(f"ROI #{roi.uid}")
    msg.setText("La ROI ha cambiado. ¿Qué desea hacer?")
    cambiar = msg.addButton("Cambiar dosis",  QtWidgets.QMessageBox.ActionRole)
    eliminar = msg.addButton("Eliminar ROI",  QtWidgets.QMessageBox.DestructiveRole)
    mantener = msg.addButton("Mantener",      QtWidgets.QMessageBox.AcceptRole)
    msg.exec()

    pressed = msg.clickedButton()
    if pressed == cambiar:
        ok, dosis_nueva = pedir_dosis(valor_defecto=roi.dose)
        if ok:
            roi.dose = dosis_nueva
    elif pressed == eliminar:
        # quitar de la escena y de las listas
        vb.removeItem(roi)
        vb.removeItem(label)
        rois.remove(roi)
        labels.remove(label)
    # En cualquier caso, refrescar tabla
    actualizar_roi_data()


def on_click(event):
    if event.button() == QtCore.Qt.LeftButton:
        view_pos = vb.mapSceneToView(event.scenePos())
        rect     = image.boundingRect()
        w, h     = 100, 100
        x0 = min(max(view_pos.x(), rect.left()),  rect.right()  - w)
        y0 = min(max(view_pos.y(), rect.top()),   rect.bottom() - h)
        add_roi((x0, y0), size=(w, h))


# Conectar el click
vb.scene().sigMouseClicked.connect(on_click)

# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    sys.exit(app.exec())
