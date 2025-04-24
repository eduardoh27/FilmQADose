from PIL import Image
import numpy as np
import pyqtgraph as pg
from pyqtgraph import RectROI
from pyqtgraph.Qt import QtCore
import os

app = pg.mkQApp()

# Ventana + ViewBox
gv = pg.GraphicsView()
vb = pg.ViewBox()
gv.setCentralItem(vb)
gv.show()

# Bloquear pan/zoom pero permitir clicks
vb.setAspectLocked(True)
vb.invertY()
vb.setMouseEnabled(x=False, y=False)
vb.setMenuEnabled(False)
vb.wheelEvent = lambda ev: None

# Carga y rota 90°
path = os.path.join('media', 'mama100cropped.tif')
img = Image.open(path)
arr = np.rot90(np.asarray(img))
image = pg.ImageItem(arr)
vb.addItem(image)

# Listas de ROIs y datos
rois = []
roi_data = []

def update_roi_data():
    global roi_data
    roi_data = []
    for roi in rois:
        p = roi.pos()
        s = roi.size()
        roi_data.append((int(p.x()), int(p.y()), int(s.x()), int(s.y())))
    print("ROI data:", roi_data)

def add_roi(pos, size=(100,100)):
    """
    Crea una ROI en `pos` con `size` restringida a image.boundingRect()
    usando el parámetro maxBounds.
    """
    bounds = image.boundingRect()
    roi = RectROI(
        pos, size,
        pen=pg.mkPen('r', width=2),
        maxBounds=bounds
    )
    # manejadores de escalado
    roi.addScaleHandle([1,1], [0,0])
    roi.addScaleHandle([0,0], [1,1])
    vb.addItem(roi)
    rois.append(roi)
    roi.sigRegionChangeFinished.connect(update_roi_data)
    update_roi_data()
    return roi

def on_click(event):
    if event.button() == QtCore.Qt.LeftButton:
        scene_pos = event.scenePos()
        view_pos = vb.mapSceneToView(scene_pos)

        rect = image.boundingRect()
        xMin, yMin = rect.left(), rect.top()
        xMax, yMax = rect.right(), rect.bottom()
        default_w, default_h = 100, 100

        # Clamp inicial para que no nazca fuera
        x0 = min(max(view_pos.x(), xMin), xMax - default_w)
        y0 = min(max(view_pos.y(), yMin), yMax - default_h)

        add_roi((x0, y0), size=(default_w, default_h))

vb.scene().sigMouseClicked.connect(on_click)

app.exec()
