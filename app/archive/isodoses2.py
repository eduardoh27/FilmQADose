# dose_analysis/isodoses.py
import os
import pydicom
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import Slider

def create_isodose_canvas(path, save_fig=False):
    # --- carga ---
    if path.endswith('.dcm'):
        ds = pydicom.dcmread(path)
        dose_map = ds.pixel_array * ds.DoseGridScaling
    elif path.endswith('.npy'):
        dose_map = np.load(path)
    else:
        raise ValueError("Formato no soportado")

    max_dose = dose_map.max()

    # --- crea Figure + Canvas explícitamente ---
    fig = Figure()
    canvas = FigureCanvas(fig)
    fig.canvas = canvas  # <-- fundamental

    # --- ejes y contornos iniciales ---
    ax = fig.add_subplot(1,1,1)
    ax.set_title("Isodoses")
    im = ax.imshow(dose_map, alpha=0.2, cmap='viridis_r')
    colors = ['darkorange','turquoise','blueviolet']
    percentages = [30,60,90]
    contours = [
        ax.contour(dose_map, levels=[max_dose*p/100], colors=col)
        for p, col in zip(percentages, colors)
    ]

    # --- sliders sobre el mismo fig ---
    axcolor = 'lightgoldenrodyellow'
    s_axes = [
        fig.add_axes([0.15, 0.10 - i*0.04, 0.65, 0.03], facecolor=axcolor)
        for i in range(3)
    ]
    alpha_ax = fig.add_axes([0.90, 0.25, 0.03, 0.65], facecolor=axcolor)

    sliders = [
        Slider(ax, '', 0, 100, valinit=p, valstep=1, valfmt='%d%%', facecolor=col)
        for ax, p, col in zip(s_axes, percentages, colors)
    ]
    alpha_slider = Slider(alpha_ax, 'Opacity', 0, 1, valinit=0.2,
                          orientation='vertical', valstep=0.01)

    def update(_):
        nonlocal contours
        # borra viejos
        for cs in contours:
            for coll in cs.collections:
                coll.remove()
        # dibuja nuevos
        contours = [
            ax.contour(dose_map, levels=[max_dose*sl.val/100], colors=col)
            for sl, col in zip(sliders, colors)
        ]
        fig.canvas.draw_idle()

    def update_alpha(val):
        im.set_alpha(val)
        fig.canvas.draw_idle()

    for sl in sliders:
        sl.on_changed(update)
    alpha_slider.on_changed(update_alpha)

    if save_fig:
        name = os.path.splitext(os.path.basename(path))[0]
        fig.savefig(f"{name}_isodoses.png", dpi=300)

    return fig, canvas
