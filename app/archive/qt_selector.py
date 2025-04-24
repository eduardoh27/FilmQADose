from PIL import Image
import numpy as np
import pyqtgraph as pg
from pyqtgraph import RectROI
from pyqtgraph.Qt import QtCore
import os

# 1) Inicializa la app
app = pg.mkQApp()

# 2) Ventana + ViewBox
gv = pg.GraphicsView()
vb = pg.ViewBox()
gv.setCentralItem(vb)
gv.show()

# 3) Bloquear pan/zoom pero permitir click para añadir ROI
vb.setAspectLocked(True)
vb.invertY()
vb.setMouseEnabled(x=False, y=False)
vb.setMenuEnabled(False)
vb.wheelEvent = lambda ev: None

# 4) Carga y rota tu imagen 90°
path = os.path.join('media', 'i.jpg')
img = Image.open(path)
arr = np.rot90(np.asarray(img))
image = pg.ImageItem(arr)
vb.addItem(image)

# 5) Estructuras para ROI
rois = []
roi_data = []  # [(x,y,w,h), ...]

def update_roi_data():
    """Actualiza roi_data con la posición y tamaño de cada ROI."""
    global roi_data
    roi_data = []
    for roi in rois:
        p = roi.pos()
        s = roi.size()
        roi_data.append((int(p.x()), int(p.y()), int(s.x()), int(s.y())))
    print("ROI data:", roi_data)

def add_roi(pos, size=(100, 100)):
    """
    Crea una nueva ROI en `pos` (x,y) con `size` y la añade al ViewBox.
    """
    roi = RectROI(pos, size, pen=pg.mkPen('r', width=2))
    roi.addScaleHandle([1,1], [0,0])
    roi.addScaleHandle([0,0], [1,1])
    vb.addItem(roi)
    rois.append(roi)
    roi.sigRegionChangeFinished.connect(update_roi_data)
    update_roi_data()
    return roi

# 6) Conecta el click en la escena para crear nuevas ROI
def on_click(event):
    if event.button() == QtCore.Qt.LeftButton:
        # convierte coordenadas de escena a coordenadas de imagen
        scene_pos = event.scenePos()
        view_pos = vb.mapSceneToView(scene_pos)
        add_roi((view_pos.x(), view_pos.y()))

vb.scene().sigMouseClicked.connect(on_click)

# 7) Arranca la aplicación
app.exec()
