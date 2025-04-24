"""
selector.py  –  display an image in its true EXIF orientation,
                allow the user to place & edit rectangular ROIs,
                and save them as (x, y, width, height) tuples.
"""

import sys, json
import numpy as np
from PIL import Image, ImageOps                 # Pillow ≥ 6.0
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
import os


class ROISelector(QtWidgets.QMainWindow):
    def __init__(self, image_path):
        super().__init__()
        self.setWindowTitle("ROI selector (click to add boxes)")
        self.rois = []                           # all RectROI objects

        # 1 ── READ + CORRECT ORIENTATION ──────────────────────────────────────
        pil = Image.open(image_path)
        pil = ImageOps.exif_transpose(pil)       # honour orientation tag
        img = np.array(pil)                      # → ndarray (H, W, C)
        h, w = img.shape[:2]

        # 2 ── BUILD VIEW ───────────────────────────────────────────────────────
        self.view = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.view)
        self.vb   = self.view.addViewBox(lockAspect=True)
        self.vb.setMouseEnabled(x=False, y=False)   # no pan / zoom
        self.vb.invertY()                           # (0,0) top‑left
        self.vb.setLimits(xMin=0, xMax=w, yMin=0, yMax=h)

        # 3 ── SHOW IMAGE ───────────────────────────────────────────────────────
        self.vb.addItem(pg.ImageItem(img))
        self.vb.autoRange()                         # fit the whole frame

        # 4 ── CLICK TO ADD ROIs ───────────────────────────────────────────────
        self.vb.scene().sigMouseClicked.connect(self._new_roi)

        # 5 ── SAVE BUTTON ─────────────────────────────────────────────────────
        btn = QtWidgets.QPushButton("Save ROIs")
        btn.clicked.connect(self._save)
        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(btn)
        proxy.setPos(10, 10)                       # screen coords
        self.vb.addItem(proxy)

    # ── helpers ────────────────────────────────────────────────────────────────
    def _new_roi(self, ev):
        if ev.button() not in (QtCore.Qt.LeftButton, QtCore.Qt.RightButton):
            return
        pt = self.vb.mapSceneToView(ev.scenePos())
        self._add_roi(int(pt.x()), int(pt.y()), 100, 100)

    def _add_roi(self, x, y, w, h):
        roi = pg.RectROI((x, y), (w, h), pen=pg.mkPen("y", width=2))
        roi.sigRegionChanged.connect(lambda r=roi: self._changed(r))
        self.vb.addItem(roi)
        self.rois.append(roi)

    def _changed(self, roi):
        print("ROI:", *self._tuple(roi))

    @staticmethod
    def _tuple(roi):
        x, y = roi.pos()
        w, h = roi.size()
        return int(x), int(y), int(w), int(h)

    def _save(self):
        data = [self._tuple(r) for r in self.rois]
        with open("rois.json", "w") as f:
            json.dump(data, f, indent=2)
        QtWidgets.QMessageBox.information(self, "Saved", f"{len(data)} ROI(s) written to rois.json")


if __name__ == "__main__":
    pg.setConfigOption('background', (30, 30, 30))   # dark background, optional
    app  = QtWidgets.QApplication(sys.argv)

    IMAGE = os.path.join('media', 'i.jpg')
    win   = ROISelector(IMAGE)
    win.resize(900, 650)
    win.show()

    sys.exit(app.exec())
