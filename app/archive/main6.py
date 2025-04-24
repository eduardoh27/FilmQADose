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

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QPixmap, QImage
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

class GammaDialog(QDialog):
    """
    Dialog to configure Gamma analysis parameters.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Gamma Analysis")
        self.setMinimumWidth(500)
        self.ref_path = ''
        self.eval_path = ''

        layout = QVBoxLayout(self)

        # Reference
        hl = QHBoxLayout()
        self.ref_edit = QLineEdit()
        self.ref_edit.setPlaceholderText('Reference (.dcm or .npy)')
        btn = QPushButton('Browse Reference')
        btn.clicked.connect(self.browse_reference)
        hl.addWidget(self.ref_edit)
        hl.addWidget(btn)
        layout.addLayout(hl)

        # Evaluation
        hl = QHBoxLayout()
        self.eval_edit = QLineEdit()
        self.eval_edit.setPlaceholderText('Evaluation (.dcm or .npy)')
        btn = QPushButton('Browse Evaluation')
        btn.clicked.connect(self.browse_evaluation)
        hl.addWidget(self.eval_edit)
        hl.addWidget(btn)
        layout.addLayout(hl)

        # Dose Diff (%)
        hl = QHBoxLayout()
        lbl = QLabel('Dose Diff.:')
        self.spin_diff = QSpinBox()
        self.spin_diff.setMinimum(1)
        self.spin_diff.setValue(1)
        self.spin_diff.setSuffix('%')
        self.spin_diff.setFixedWidth(60)
        hl.addWidget(lbl)
        hl.addWidget(self.spin_diff)
        layout.addLayout(hl)

        # DTA (mm)
        hl = QHBoxLayout()
        lbl = QLabel('DTA:')
        self.spin_dta = QSpinBox()
        self.spin_dta.setMinimum(1)
        self.spin_dta.setValue(1)
        self.spin_dta.setSuffix(' mm')
        self.spin_dta.setFixedWidth(60)
        hl.addWidget(lbl)
        hl.addWidget(self.spin_dta)
        layout.addLayout(hl)

        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def browse_reference(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Reference", filter="Dose Maps (*.dcm *.npy)")
        if path:
            self.ref_path = path
            self.ref_edit.setText(os.path.basename(path))

    def browse_evaluation(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Evaluation", filter="Dose Maps (*.dcm *.npy)")
        if path:
            self.eval_path = path
            self.eval_edit.setText(os.path.basename(path))

class GammaWidget(QWidget):
    """
    QWidget embedding side-by-side Gamma matrix and histogram.
    """
    def __init__(self, ref_path, eval_path, dd, dta, parent=None):
        super().__init__(parent)
        # Load
        ds = pydicom.dcmread(ref_path)
        tps = ds.pixel_array * ds.DoseGridScaling
        ver = np.load(eval_path)
        # Normalize
        ref = tps / np.max(tps)
        eval_ = ver / np.max(ver)
        # axes
        dx, dy = ds.PixelSpacing
        x1 = np.linspace(0, tps.shape[1]*dx, tps.shape[1])
        y1 = np.linspace(0, tps.shape[0]*dy, tps.shape[0])
        ejes1 = (x1, y1)
        ejes2 = ejes1
        # options
        opt = {
            'dose_percent_threshold': dd,
            'distance_mm_threshold': dta,
            'lower_percent_dose_cutoff': 5,
            'interp_fraction': 10,
            'max_gamma': 5,
            'random_subset': None,
            'local_gamma': False,
            'ram_available': 5*(2**29)
        }
        # compute
        gamma = pymedphys.gamma(ejes1, ref, ejes2, eval_, **opt)
        # build figure
        fig = Figure(figsize=(10, 5))
        canvas = FigureCanvas(fig)
        ax1 = fig.add_subplot(1,2,1)
        im = ax1.imshow(gamma, cmap='inferno')
        fig.colorbar(im, ax=ax1)
        ax1.set_title(f"Gamma Matrix {dta}mm/{dd}%")
        # histogram
        ax2 = fig.add_subplot(1,2,2)
        vals = gamma[~np.isnan(gamma)]
        bins = np.linspace(0, opt['max_gamma'], opt['interp_fraction']*opt['max_gamma']+1)
        ax2.hist(vals, bins=bins, alpha=0.7)
        ax2.axvline(1, color='red', linestyle='--')
        pass_rate = np.sum(vals<=1)/len(vals)*100
        ax2.text(1.1, 0.9*ax2.get_ylim()[1], f"Pass: {pass_rate:.2f}%")
        ax2.set_title(f"Gamma Histogram {dta}mm/{dd}%")
        ax2.set_xlabel('Gamma')
        ax2.set_ylabel('Count')
        # layout
        v = QVBoxLayout(self)
        v.addWidget(canvas)
        self.metadata = {'type':'gamma', 'reference':ref_path, 'evaluation':eval_path, 'dose_diff':dd, 'dta':dta}

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
        lbl=ScaledLabel(); img=os.path.join(os.path.dirname(__file__),'media','Uniandes.png')
        px=QPixmap(img); lbl.setPixmap(px) if not px.isNull() else lbl.setText('Uniandes.png not found.')
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
        dlg=GammaDialog(self)
        if dlg.exec()==QDialog.Accepted and dlg.ref_path and dlg.eval_path:
            self._add_gamma_tab(dlg.ref_path,dlg.eval_path,dlg.spin_diff.value(),dlg.spin_dta.value())
    def _add_gamma_tab(self,ref,ev,dd,dta):
        w=GammaWidget(ref,ev,dd,dta); w.setFocusPolicy(Qt.ClickFocus); w.setFocus()
        self.tabs.addTab(w,f"Gamma: {os.path.basename(ref)} vs {os.path.basename(ev)}"); self.tabs.setCurrentIndex(self.tabs.count()-1)
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
