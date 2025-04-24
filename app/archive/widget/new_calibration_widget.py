class CalibrationWidgetOld(QWidget):
    """
    Widget interactivo para marcar ROIs sobre la primera película de calibración
    y asociarles un valor de dosis.
    """
    def __init__(self, tif_path, calib_type, parent=None):
        super().__init__(parent)
        self.tif_path = tif_path
        self.calib_type = calib_type
        self.rois = []
        self.labels = []
        self.roi_data = []

        # Layout principal
        layout = QVBoxLayout(self)

        # GraphicsView + ViewBox
        self.gv = pg.GraphicsView()
        self.vb = pg.ViewBox()
        self.gv.setCentralItem(self.vb)
        layout.addWidget(self.gv)

        # Configuración del ViewBox
        self.vb.setAspectLocked(True)
        self.vb.invertY()
        self.vb.setMouseEnabled(False, False)
        self.vb.setMenuEnabled(False)
        self.vb.wheelEvent = lambda ev: None

        # Cargar la primera imagen de calibración
        arr = np.flipud(np.rot90(np.asarray(Image.open(self.tif_path))))
        self.image_item = pg.ImageItem(arr)
        self.vb.addItem(self.image_item)

        # Conectar clic para añadir ROI
        self.vb.scene().sigMouseClicked.connect(self.on_click)

    def ask_dose(self, default_value=100):
        dose, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Enter dose",
            "Associated dose (cGy):",
            value=default_value,
            minValue=0
        )
        return ok, int(dose)

    def refresh_roi_data(self):
        """
        Reconstruye self.roi_data y lo imprime (se puede adaptar para mostrar en UI).
        """
        self.roi_data = []
        for roi in self.rois:
            p, s = roi.pos(), roi.size()
            self.roi_data.append((
                int(p.x()), int(p.y()),
                int(s.x()), int(s.y()),
                roi.dose
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

        # Conexiones
        roi.sigRegionChanged.connect(lambda: label.setPos(*roi.pos()))
        roi.sigRegionChangeFinished.connect(lambda: self.on_roi_modified(roi, label))

        self.rois.append(roi)
        self.labels.append(label)
        self.refresh_roi_data()
        return roi

    def on_roi_modified(self, roi, label):
        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("ROI dose")
        msg.setText("The ROI has changed. What would you like to do?")
        change_btn = msg.addButton("Change dose", QtWidgets.QMessageBox.ActionRole)
        delete_btn = msg.addButton("Delete ROI", QtWidgets.QMessageBox.DestructiveRole)
        keep_btn   = msg.addButton("Keep", QtWidgets.QMessageBox.AcceptRole)
        msg.exec()

        pressed = msg.clickedButton()
        if pressed == change_btn:
            ok, new_dose = self.ask_dose(default_value=roi.dose)
            if ok:
                roi.dose = new_dose
                label.setText(str(new_dose))
        elif pressed == delete_btn:
            self.vb.removeItem(roi)
            self.vb.removeItem(label)
            self.rois.remove(roi)
            self.labels.remove(label)
        # siempre refrescar
        self.refresh_roi_data()

    def on_click(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            view_pos = self.vb.mapSceneToView(event.scenePos())
            rect = self.image_item.boundingRect()
            w, h = 100, 100
            x0 = min(max(view_pos.x(), rect.left()),  rect.right()  - w)
            y0 = min(max(view_pos.y(), rect.top()),   rect.bottom() - h)
            self.add_roi((x0, y0), size=(w, h))
