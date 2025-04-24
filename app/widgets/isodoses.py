# dose_analysis/isodoses.py
import os, pydicom, numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import Slider
from PySide6.QtWidgets import QWidget, QVBoxLayout

class IsodoseWidget(QWidget):
    def __init__(self, path, save_fig=False, parent=None):
        super().__init__(parent)

        # --- carga del mapa ---
        if path.endswith('.dcm'):
            ds = pydicom.dcmread(path)
            dose_map = ds.pixel_array * ds.DoseGridScaling
        elif path.endswith('.npy'):
            dose_map = np.load(path)
        else:
            raise ValueError("Unsupported file format. Use .dcm or .npy")

        max_dose = dose_map.max()

        # --- figura y canvas ---
        fig = Figure()
        canvas = FigureCanvas(fig)
        fig.canvas = canvas

        # Used to avoid overlapping of the sliders and the image
        fig.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.25)

        # --- ejes ---
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(dose_map, alpha=0.2, cmap='viridis_r')
        colors = ['darkorange','turquoise','blueviolet']
        init = [30,60,90]
        self.contours = [
            ax.contour(dose_map, levels=[max_dose*p/100], colors=col)
            for p,col in zip(init, colors)
        ]

        # --- sliders ---
        axcolor = 'lightgoldenrodyellow'
        s_axes = [fig.add_axes([0.15, 0.10 - i*0.04, 0.65, 0.03], facecolor=axcolor)
                  for i in range(3)]
        alpha_ax = fig.add_axes([0.90, 0.25, 0.03, 0.65], facecolor=axcolor)

        self.sliders = [
            Slider(ax, '', 0,100, valinit=p, valstep=1, valfmt='%d%%', facecolor=col)
            for ax,p,col in zip(s_axes, init, colors)
        ]
        self.alpha_slider = Slider(alpha_ax, 'Opacity', 0,1, valinit=0.2,
                                   orientation='vertical', valstep=0.01)

        # Hide the lines on the sliders, if they exist
        for slider in self.sliders:
            if hasattr(slider, 'vline'):
                slider.vline.set_visible(False)

        if hasattr(self.alpha_slider, 'hline'):
            self.alpha_slider.hline.set_visible(False)

        def update(_):
            for cs in self.contours:
                for coll in cs.collections:
                    coll.remove()
            self.contours = [
                ax.contour(dose_map, levels=[max_dose*sl.val/100], colors=col)
                for sl,col in zip(self.sliders, colors)
            ]
            fig.canvas.draw_idle()

        def update_alpha(val):
            im.set_alpha(val)
            fig.canvas.draw_idle()

        for sl in self.sliders:
            sl.on_changed(update)
        self.alpha_slider.on_changed(update_alpha)

        # --- montaje en Qt ---
        layout = QVBoxLayout(self)
        layout.addWidget(canvas)

        if save_fig:
            name = os.path.splitext(os.path.basename(path))[0]
            fig.savefig(f"{name}_isodoses.png", dpi=300)
