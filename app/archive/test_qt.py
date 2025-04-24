from PIL import Image
import numpy as np
import pyqtgraph as pg
import os

app = pg.mkQApp()

# Ventana + ViewBox
gv = pg.GraphicsView()
vb = pg.ViewBox()
gv.setCentralItem(vb)
gv.show()

# Bloquear interacciones pero mantener la rotación previa
vb.setAspectLocked(True)
vb.invertY()
vb.setMouseEnabled(x=False, y=False)   # sin arrastre/zoom
vb.setMenuEnabled(False)               # sin menú contextual
vb.wheelEvent = lambda ev: None        # ignora rueda de ratón

# Carga, rota 90° y muestra
path = os.path.join('media', 'i.jpg')
#path = os.path.join('media', 'Dosis0a10.tif')
img = Image.open(path)
arr = np.rot90(np.asarray(img))        # <<< rotación fija de 90°
arr = np.flipud(arr)                  # <<< reflejar en eje Y
image = pg.ImageItem(arr)
vb.addItem(image)

app.exec()
