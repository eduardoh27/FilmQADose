import os
import pydicom 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def plot_isodose_map(path, save_fig=False):
    """
    Plots an isodose map from a given file and provides interactive sliders to adjust the contour levels and image opacity.

    The input file can be either a DICOM (.dcm) file or a NumPy binary (.npy) file.
    In the case of a DICOM file, the function multiplies the pixel array by the DoseGridScaling attribute.
    The isodose levels are initially set at 30%, 60%, and 90% of the maximum dose.

    Parameters
    ----------
    path : str
        Path to the dose map file (supported formats: .dcm or .npy).
    save_fig : bool, optional
        If True, saves the resulting figure as a PNG file with a name based on the input file.

    Raises
    ------
    ValueError
        If the file format is not supported.
    """
    # Define colors for the isodose contours
    curves_colors = ['darkorange', 'turquoise', 'blueviolet']
    # Get the base name (without extension) of the file to use for saving the figure
    output_name = os.path.splitext(os.path.basename(path))[0]

    # Load the file based on its extension
    if path.endswith('.dcm'):
        ds = pydicom.dcmread(path)
        dose_map = ds.pixel_array * ds.DoseGridScaling
    elif path.endswith('.npy'):
        dose_map = np.load(path)
    else:
        raise ValueError("Unsupported file format. Use .dcm or .npy")

    max_dose = dose_map.max()

    # Create a matplotlib figure and axis for displaying the dose map
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25, right=0.85)
    im = ax.imshow(dose_map, alpha=0.2, cmap='viridis_r')
    ax.set_title("Isodoses")

    # Set initial contour levels as percentages of the maximum dose
    initial_values = [30, 60, 90]
    initial_levels = [max_dose * p / 100 for p in initial_values]

    # Draw the initial contours using the preset levels and colors
    cs1 = ax.contour(dose_map, levels=[initial_levels[0]], colors=curves_colors[0])
    cs2 = ax.contour(dose_map, levels=[initial_levels[1]], colors=curves_colors[1])
    cs3 = ax.contour(dose_map, levels=[initial_levels[2]], colors=curves_colors[2])
    contours = [cs1, cs2, cs3]

    # Define the color for the slider background
    axcolor = 'lightgoldenrodyellow'
    # Create axes for the three sliders that control the contour levels
    slider_ax1 = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
    slider_ax2 = plt.axes([0.15, 0.06, 0.65, 0.03], facecolor=axcolor)
    slider_ax3 = plt.axes([0.15, 0.02, 0.65, 0.03], facecolor=axcolor)

    # Create sliders for each contour level (with no label)
    slider1 = Slider(slider_ax1, '', 0, 100, valinit=30, valstep=1,
                     valfmt='%d%%', facecolor=curves_colors[0])
    slider2 = Slider(slider_ax2, '', 0, 100, valinit=60, valstep=1,
                     valfmt='%d%%', facecolor=curves_colors[1])
    slider3 = Slider(slider_ax3, '', 0, 100, valinit=90, valstep=1,
                     valfmt='%d%%', facecolor=curves_colors[2])

    # Hide the vertical lines on the sliders, if they exist
    for slider in [slider1, slider2, slider3]:
        if hasattr(slider, 'vline'):
            slider.vline.set_visible(False)

    # Create an axis for the opacity slider (vertical slider)
    slider_ax_alpha = plt.axes([0.9, 0.25, 0.03, 0.65], facecolor=axcolor)
    alpha_slider = Slider(slider_ax_alpha, 'Opacity', 0, 1, valinit=0.2,
                          orientation='vertical', valfmt='%1.2f', valstep=0.01)

    # Hide the horizontal line of the opacity slider, if it exists
    if hasattr(alpha_slider, 'hline'):
        alpha_slider.hline.set_visible(False)

    def update(val):
        """
        Update callback for the contour level sliders.
        Removes old contour lines and draws new ones based on slider values.
        """
        nonlocal contours
        # Remove existing contours
        for cs in contours:
            cs.remove()
        contours.clear()

        # Compute new levels based on current slider values
        levels = [max_dose * slider.val / 100 for slider in [slider1, slider2, slider3]]
        # Draw new contours with updated levels and colors
        cs1 = ax.contour(dose_map, levels=[levels[0]], colors=curves_colors[0])
        cs2 = ax.contour(dose_map, levels=[levels[1]], colors=curves_colors[1])
        cs3 = ax.contour(dose_map, levels=[levels[2]], colors=curves_colors[2])
        contours = [cs1, cs2, cs3]
        fig.canvas.draw_idle()

    def update_alpha(val):
        """
        Update callback for the opacity slider.
        Adjusts the alpha transparency of the displayed dose map.
        """
        im.set_alpha(val)
        fig.canvas.draw_idle()

    # Register the update callbacks with the sliders
    slider1.on_changed(update)
    slider2.on_changed(update)
    slider3.on_changed(update)
    alpha_slider.on_changed(update_alpha)

    # Save the figure if requested
    if save_fig:
        fig.savefig(f"{output_name}_isodoses.png", dpi=300)
    plt.show()
    #return fig

# Uncomment the line below to test the function with an example file.
plot_isodose_map('media/mama_TPS.dcm', save_fig=False)
