import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pydicom 

ds = pydicom.dcmread('mama_TPS.dcm')
dose_map = ds.pixel_array * ds.DoseGridScaling
max_dose = dose_map.max()


fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
im = ax.imshow(dose_map, cmap='viridis_r', alpha=0.2)
ax.set_title("Isodoses")

# Valores iniciales de los sliders (en porcentaje)
initial_values = [50, 70, 90]
initial_levels = [max_dose * p/100 for p in initial_values]

# Trazar inicialmente cada contorno por separado con su color
cs1 = ax.contour(dose_map, levels=[initial_levels[0]], colors=['darkorange'])
cs2 = ax.contour(dose_map, levels=[initial_levels[1]], colors=['turquoise'])
cs3 = ax.contour(dose_map, levels=[initial_levels[2]], colors=['blueviolet'])

# Guardamos los contornos en una lista para poder removerlos después
contours = [cs1, cs2, cs3]

# Crear los ejes para los sliders
axcolor = 'lightgoldenrodyellow'
slider_ax1 = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
slider_ax2 = plt.axes([0.15, 0.06, 0.65, 0.03], facecolor=axcolor)
slider_ax3 = plt.axes([0.15, 0.02, 0.65, 0.03], facecolor=axcolor)

# Crear los sliders:
# - label='' elimina el texto a la izquierda
# - valfmt='%1.1f%%' muestra el valor con 1 decimal y un símbolo '%'
# - facecolor asigna el color de la barra para cada slider
slider1 = Slider(slider_ax1, label='', valmin=0, valmax=100, valinit=30,
                 valfmt='%1.1f%%', facecolor='darkorange')
slider2 = Slider(slider_ax2, label='', valmin=0, valmax=100, valinit=60,
                 valfmt='%1.1f%%', facecolor='turquoise')
slider3 = Slider(slider_ax3, label='', valmin=0, valmax=100, valinit=90,
                 valfmt='%1.1f%%', facecolor='blueviolet')

def update(val):
    global contours
    # Remover los contornos previos
    for cs in contours:
        cs.remove()
    contours.clear()
    
    # Calcular nuevos niveles en Gy basados en el valor del slider
    level1 = max_dose * slider1.val / 100
    level2 = max_dose * slider2.val / 100
    level3 = max_dose * slider3.val / 100

    # Trazar los contornos de cada nivel con su color
    cs1 = ax.contour(dose_map, levels=[level1], colors=['darkorange'])
    cs2 = ax.contour(dose_map, levels=[level2], colors=['turquoise'])
    cs3 = ax.contour(dose_map, levels=[level3], colors=['blueviolet'])
    contours = [cs1, cs2, cs3]

    fig.canvas.draw_idle()

slider1.on_changed(update)
slider2.on_changed(update)
slider3.on_changed(update)

plt.show()
