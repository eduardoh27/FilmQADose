import os
import numpy as np
from pydicom import dcmread
from pymedphys import gamma
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QSpinBox,
    QDialogButtonBox,
    QFileDialog,
)

class GammaDialog(QDialog):
    """
    Dialog to configure Gamma analysis parameters.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.default_dir = os.path.join(os.path.dirname(__file__), "..", "media")
        self.setWindowTitle("Configure Gamma Analysis")
        self.setMinimumWidth(500)
        self.ref_path = ''
        self.eval_path = ''

        layout = QVBoxLayout(self)

        # Reference
        hl = QHBoxLayout()
        self.ref_edit = QLineEdit()
        self.ref_edit.setPlaceholderText('Reference (.dcm)')
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
        self.spin_diff.setFixedWidth(80)
        hl.addWidget(lbl)
        hl.addWidget(self.spin_diff)
        layout.addLayout(hl)

        # DTA (mm)
        hl = QHBoxLayout()
        lbl = QLabel('DTA:')
        self.spin_dta = QSpinBox()
        self.spin_dta.setMinimum(1)
        self.spin_dta.setValue(1)
        self.spin_dta.setSuffix('mm')
        self.spin_dta.setFixedWidth(80)
        hl.addWidget(lbl)
        hl.addWidget(self.spin_dta)
        layout.addLayout(hl)

        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def browse_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Reference", 
            self.default_dir,
            filter="Dose Maps (*.dcm)"
        )
        if path:
            self.ref_path = path
            self.ref_edit.setText(os.path.basename(path))

    def browse_evaluation(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Evaluation", 
            self.default_dir,
            filter="Dose Maps (*.dcm *.npy)"
        )
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
        ds = dcmread(ref_path)
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
        gamma_matrix = gamma(ejes1, ref, ejes2, eval_, **opt)
        # figure
        fig = Figure(figsize=(10, 5))
        canvas = FigureCanvas(fig)
        fig.subplots_adjust(left=0.05, right=0.95, wspace=0.25)
        # matrix
        ax1 = fig.add_subplot(1,2,1)
        im = ax1.imshow(gamma_matrix, cmap='inferno')
        fig.colorbar(im, ax=ax1)
        ax1.set_title(f"Gamma Matrix {dta}mm/{dd}%")
        # histogram
        ax2 = fig.add_subplot(1,2,2)
        vals = gamma_matrix[~np.isnan(gamma_matrix)]
        bins = np.linspace(0, opt['max_gamma'], opt['interp_fraction']*opt['max_gamma']+1)
        ax2.hist(vals, bins=bins, alpha=0.7)
        ax2.axvline(1, color='red', linestyle='--', linewidth=2, label="Pass Threshold")
        pass_rate = np.sum(vals<=1)/len(vals)*100
        ax2.text(1.1, 0.9*ax2.get_ylim()[1], f"Pass rate: {pass_rate:.2f}%")
        ax2.set_title(f"Gamma Histogram {dta}mm/{dd}%")
        ax2.set_xlabel(r'$\gamma$ index')
        ax2.set_ylabel('Number of reference points')
        ax2.legend()
        # layout
        v = QVBoxLayout(self)
        v.addWidget(canvas)
        self.metadata = {'type':'gamma', 'reference':ref_path, 'evaluation':eval_path, 'dose_diff':dd, 'dta':dta, 'pass_rate':pass_rate}
