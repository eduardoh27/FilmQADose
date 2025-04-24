import os
import numpy as np
import pydicom
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
from PySide6.QtWidgets import QWidget, QVBoxLayout

class DoseProfilesWidget(QWidget):
    """
    QWidget that embeds an interactive dose profile plot:
      - Dose map with reference lines
      - Horizontal and vertical dose profiles with sliders
    """
    def __init__(self, path, x=None, y=None, save_fig=False, parent=None):
        super().__init__(parent)

        # Load dose map
        if path.lower().endswith('.dcm'):
            ds = pydicom.dcmread(path)
            dose_map = ds.pixel_array * ds.DoseGridScaling
        elif path.lower().endswith('.npy'):
            dose_map = np.load(path)
        else:
            raise ValueError("Unsupported file format. Use .npy or .dcm")

        # Coordinates
        nx, ny = dose_map.shape
        if x is None:
            x = np.arange(ny)
        if y is None:
            y = np.arange(nx)

        max_dose = dose_map.max()
        default_row = nx // 2
        default_col = ny // 2
        default_slider_row = (nx - 1) - default_row

        # Create figure and canvas
        fig = Figure(figsize=(8, 6))
        canvas = FigureCanvas(fig)
        fig.canvas = canvas

        # Layout adjustments to make room for sliders
        #fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.16)

        # Main grid: map on left, profiles on right
        gs_main = gridspec.GridSpec(1, 2, width_ratios=[3, 2],
                                    left=0.1, right=0.95, top=0.9, bottom=0.15, wspace=0.2)

        # Dose map with reference lines
        ax_map = fig.add_subplot(gs_main[0, 0])
        im = ax_map.imshow(dose_map, cmap='inferno')
        ax_map.set_title('Dose Map')
        ax_map.set_xlabel('X')
        ax_map.set_ylabel('Y')
        line_h = ax_map.axhline(y=y[default_row], color='cyan', lw=2)
        line_v = ax_map.axvline(x=x[default_col], color='lime', lw=2)

        # Profiles panel
        gs_profiles = gridspec.GridSpecFromSubplotSpec(2, 1,
                            subplot_spec=gs_main[0, 1], hspace=0.4)
        # Horizontal profile
        ax_h = fig.add_subplot(gs_profiles[0, 0])
        h_line, = ax_h.plot(x, dose_map[default_row, :], lw=2)
        ax_h.set_title(f'Horizontal Profile (row {default_row})')
        ax_h.set_xlabel('X')
        ax_h.set_ylabel('Dose (Gy)')
        ax_h.grid(True)
        ax_h.set_ylim(0, max_dose)

        # Vertical profile
        ax_v = fig.add_subplot(gs_profiles[1, 0])
        v_line, = ax_v.plot(y, dose_map[:, default_col], lw=2, color='green')
        ax_v.set_title(f'Vertical Profile (column {default_col})')
        ax_v.set_xlabel('Y')
        ax_v.set_ylabel('Dose (Gy)')
        ax_v.grid(True)
        ax_v.set_ylim(0, max_dose)

        # Sliders axes
        axcolor = 'lightgoldenrodyellow'
        ax_slider_row = fig.add_axes([0.02, 0.3, 0.03, 0.6], facecolor=axcolor)
        ax_slider_col = fig.add_axes([0.135, 0.05, 0.375, 0.03], facecolor=axcolor)

        # Sliders
        slider_row = Slider(ax_slider_row, 'Row (Y)', 0, nx-1,
                            valinit=default_slider_row, valstep=1,
                            orientation='vertical', color='dimgray')
        slider_col = Slider(ax_slider_col, 'Column (X)', 0, ny-1,
                            valinit=default_col, valstep=1,
                            color='dimgray')

        # Update callbacks
        def update_row(val):
            row = int((nx - 1) - slider_row.val)
            line_h.set_ydata([y[row], y[row]])
            h_line.set_ydata(dose_map[row, :])
            ax_h.set_title(f'Horizontal Profile (row {row})')
            slider_row.valtext.set_text(str(row))
            fig.canvas.draw_idle()

        def update_col(val):
            col = int(slider_col.val)
            line_v.set_xdata([x[col], x[col]])
            v_line.set_ydata(dose_map[:, col])
            ax_v.set_title(f'Vertical Profile (column {col})')
            fig.canvas.draw_idle()

        slider_row.on_changed(update_row)
        slider_col.on_changed(update_col)

        # Optional save
        if save_fig:
            name = os.path.splitext(os.path.basename(path))[0]
            fig.savefig(f"{name}_dose_profiles.png", dpi=300, bbox_inches='tight')

        # Embed in widget layout
        layout = QVBoxLayout(self)
        layout.addWidget(canvas)

        # Keep references
        self._fig = fig
        self._canvas = canvas
        self._sliders = (slider_row, slider_col)

        self.metadata = {'type': 'profiles', 'file': path}
