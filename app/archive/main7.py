import sys
import os
import numpy as np
import pydicom
import pymedphys
import matplotlib
os.environ["QT_API"] = "PySide6"
matplotlib.use('QtAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtSvg import QSvgRenderer
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QPixmap, QImage, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QLineEdit,
    QDialogButtonBox,
    QTabWidget,
    QSizePolicy
)

# Tools widgets
from dose_analysis.isodoses3 import IsodoseWidget
from dose_analysis.dose_profile2 import DoseProfileWidget
from dose_analysis.gamma2 import GammaWidget, GammaDialog

class ScaledLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.metadata = {}
        self._pixmap = None
    def setPixmap(self, pixmap: QPixmap):
        self._pixmap = pixmap; self._update()
    def resizeEvent(self, event): super().resizeEvent(event); self._update()
    def _update(self):
        if self._pixmap and not self._pixmap.isNull():
            sp = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            super().setPixmap(sp)
        else:
            super().setPixmap(QPixmap())

class FilmQADoseMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FilmQADose")
        men = self.menuBar()
        fm = men.addMenu("File"); fm.addAction(QAction("Open...", self, triggered=self.open_file))
        men.addMenu("Calibration")
        tm = men.addMenu("Tools")
        tm.addAction(QAction("Isodoses", self, triggered=self.open_isodoses))
        tm.addAction(QAction("Profiles", self, triggered=self.open_profiles))
        tm.addAction(QAction("Gamma", self, triggered=self.open_gamma))
        c = QWidget(); self.setCentralWidget(c)
        lo = QHBoxLayout(c)
        self.info = QLabel("Welcome to FilmQADose")
        left = QVBoxLayout(); left.addWidget(self.info); lo.addLayout(left,1)
        self.tabs = QTabWidget(); self.tabs.setTabsClosable(True)
        self.tabs.tabCloseRequested.connect(lambda i: self.tabs.removeTab(i))
        self.tabs.currentChanged.connect(self.refresh_info)
        lo.addWidget(self.tabs,3)
        self._add_welcome_tab(); self.resize(800,600)

    def _add_welcome_tab(self):
        lbl=ScaledLabel(); img=os.path.join(os.path.dirname(__file__),'media','logo full without bg.png')
        px=QPixmap(img); lbl.setPixmap(px) if not px.isNull() else lbl.setText('logo not found.')
        lbl.metadata={'type':'welcome'}; self.tabs.addTab(lbl,'Welcome')

    def open_file(self):
        d=QFileDialog(self,'Open File'); d.setNameFilter('Images (*.png *.jpg *.jpeg *.tiff *.tif);;DICOM (*.dcm);;All Files (*)')
        if d.exec(): self._add_image_tab(d.selectedFiles()[0])
    def _add_image_tab(self,path):
        lbl=ScaledLabel(); px=QPixmap(path)
        if not px.isNull() and not path.lower().endswith('.dcm'):
            lbl.setPixmap(px); lbl.metadata={'type':'image','file':path,'dims':f"{px.width()}x{px.height()}"}
        else:
            pm,info=self._load_dicom(path)
            if not pm.isNull(): lbl.setPixmap(pm); m={'type':'dicom','file':path}; m.update(info); lbl.metadata=m
            else: lbl.setText(f"Cannot load: {os.path.basename(path)}"); lbl.metadata={'type':'unknown','file':path}
        self.tabs.addTab(lbl,os.path.basename(path)); self.tabs.setCurrentIndex(self.tabs.count()-1)
    def open_isodoses(self):
        d=QFileDialog(self,'Select Dose Map for Isodoses',filter='Dose Maps (*.dcm *.npy)')
        if d.exec(): self._add_isodose_tab(d.selectedFiles()[0])
    def _add_isodose_tab(self,path):
        w=IsodoseWidget(path); w.setFocusPolicy(Qt.ClickFocus); w.setFocus()
        self.tabs.addTab(w,f"Isodose: {os.path.basename(path)}"); self.tabs.setCurrentIndex(self.tabs.count()-1)
    def open_profiles(self):
        d=QFileDialog(self,'Select Dose Map for Profiles',filter='Dose Maps (*.dcm *.npy)')
        if d.exec(): self._add_profiles_tab(d.selectedFiles()[0])
    def _add_profiles_tab(self,path):
        w=DoseProfileWidget(path); w.setFocusPolicy(Qt.ClickFocus); w.setFocus()
        self.tabs.addTab(w,f"Profiles: {os.path.basename(path)}"); self.tabs.setCurrentIndex(self.tabs.count()-1)
    def open_gamma(self):
        dialog = GammaDialog(self)
        if dialog.exec() == QDialog.Accepted and dialog.ref_path and dialog.eval_path:
            ref = dialog.ref_path
            eval_ = dialog.eval_path
            dd = dialog.spin_diff.value()
            dta = dialog.spin_dta.value()
            self._add_gamma_tab(ref, eval_, dd, dta)

    def _add_gamma_tab(self, ref_path, eval_path, dose_diff, dta):
        # Mostrar mensaje de espera en la pestaña
        placeholder = QWidget()
        box = QVBoxLayout(placeholder)
        wait_lbl = QLabel("Calculando gamma, por favor espere...")
        wait_lbl.setAlignment(Qt.AlignCenter)
        box.addWidget(wait_lbl)
        tab_label = f"Gamma: {os.path.basename(ref_path)} vs {os.path.basename(eval_path)}"
        index = self.tabs.addTab(placeholder, tab_label)
        self.tabs.setCurrentIndex(index)

        # Ejecutar cálculo tras breve retardo para renderizar mensaje
        from PySide6.QtCore import QTimer
        QTimer.singleShot(100, lambda: self._replace_gamma_tab(
            index, ref_path, eval_path, dose_diff, dta
        ))

    def _replace_gamma_tab(self, index, ref_path, eval_path, dose_diff, dta):
        # Crear y mostrar el widget definitivo
        gamma_widget = GammaWidget(ref_path, eval_path, dose_diff, dta)
        gamma_widget.setFocusPolicy(Qt.ClickFocus)
        gamma_widget.setFocus()
        tab_label = f"Gamma: {os.path.basename(ref_path)} vs {os.path.basename(eval_path)}"
        # Reemplazar la pestaña de espera
        self.tabs.removeTab(index)
        self.tabs.insertTab(index, gamma_widget, tab_label)
        self.tabs.setCurrentIndex(index)

    def _load_dicom(self,path):
        from pydicom.filereader import InvalidDicomError
        try: ds=pydicom.dcmread(path)
        except (InvalidDicomError,FileNotFoundError): return QPixmap(),{}
        info={k:str(ds.get(k,'N/A')) for k in['PatientName','PatientID','Modality','Rows','Columns','DoseGridScaling']}
        arr=ds.pixel_array; norm=((arr.astype(float)-arr.min())/(arr.max()-arr.min())*255 if arr.max()!=arr.min() else np.zeros_like(arr))
        img=QImage(norm.astype(np.uint8).data, norm.shape[1],norm.shape[0],norm.shape[1],QImage.Format_Grayscale8)
        return QPixmap.fromImage(img),info
    def refresh_info(self,i):
        w=self.tabs.widget(i); md=getattr(w,'metadata',{}); t=md.get('type','')
        if t=='welcome': self.info.setText('Welcome to FilmQADose')
        elif t=='image': self.info.setText(f"Image: {os.path.basename(md['file'])}\nDims: {md['dims']}")
        elif t=='dicom': self.info.setText('DICOM: '+os.path.basename(md['file']))
        elif t=='gamma': self.info.setText(f"Gamma: {os.path.basename(md['reference'])} vs {os.path.basename(md['evaluation'])}\nDoseDiff: {md['dose_diff']}% DTA: {md['dta']}mm")
        else: self.info.setText('')

if __name__=='__main__':
    app=QApplication(sys.argv); win=FilmQADoseMainWindow(); win.show(); sys.exit(app.exec())
