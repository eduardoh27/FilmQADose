import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
import pydicom
import os

def display_dose_profile(path, x=None, y=None, save_fig=False):
    """
    Loads a dose map from a file (.npy or .dcm) and displays interactive profiles:
      - Dose map image with reference lines.
      - Horizontal (row) and vertical (column) profiles with interactive sliders.
    
    Parameters
    ----------
    path : str
        File path to the dose map (.npy or .dcm).
    x : array-like, optional
        Horizontal coordinates; if None, pixel indices are used.
    y : array-like, optional
        Vertical coordinates; if None, pixel indices are used.
    save_fig : bool, optional
        If True, saves the figure as a PNG file.
    """
    
    # Load the file based on its extension
    if path.endswith('.dcm'):
        ds = pydicom.dcmread(path)
        dose_map = ds.pixel_array * ds.DoseGridScaling
    elif path.endswith('.npy'):
        dose_map = np.load(path)
    else:
        raise ValueError("Unsupported file format. Use .npy or .dcm")
    
    # Get the base name of the file for output naming
    output_name = os.path.splitext(os.path.basename(path))[0]
    
    # Dimensions of the dose map
    nx, ny = dose_map.shape
    if x is None:
        x = np.arange(ny)
    if y is None:
        y = np.arange(nx)
        
    max_dose = dose_map.max()
    default_row = nx // 2
    default_col = ny // 2

    # For the inverted slider, adjust the initial value:
    # Since the slider value is inverted, the real index is (nx - 1) - slider value.
    default_slider_value = (nx - 1) - default_row

    # Create the main figure with a custom layout
    fig = plt.figure(figsize=(12, 8))
    gs_main = gridspec.GridSpec(1, 2, width_ratios=[3, 2],
                                 left=0.1, right=0.9, top=0.9, bottom=0.3, wspace=0.1)
    
    # Dose map panel: display the dose map with reference lines
    ax_map = fig.add_subplot(gs_main[0, 0])
    extent = [x[0], x[-1], y[0], y[-1]]
    im = ax_map.imshow(dose_map, cmap='inferno')  # You may add extent=extent if required
    ax_map.set_title("Dose Map")
    ax_map.set_xlabel("X")
    ax_map.set_ylabel("Y")
    # Initial reference lines using default_row and default_col
    line_h = ax_map.axhline(y=y[default_row], color='cyan', lw=2)
    line_v = ax_map.axvline(x=x[default_col], color='lime', lw=2)

    # Profiles panel: subplots for horizontal and vertical profiles
    gs_profiles = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0, 1], hspace=0.4)
    
    # Horizontal profile (row): using default_row index
    ax_hprofile = fig.add_subplot(gs_profiles[0, 0])
    h_profile_line, = ax_hprofile.plot(x, dose_map[default_row, :], lw=2)
    ax_hprofile.set_title(f"Horizontal Profile (row {default_row})")
    ax_hprofile.set_xlabel("X")
    ax_hprofile.set_ylabel("Dose (Gy)")
    ax_hprofile.grid(True)
    ax_hprofile.set_ylim(0, max_dose)

    # Vertical profile (column): using default_col index
    ax_vprofile = fig.add_subplot(gs_profiles[1, 0])
    v_profile_line, = ax_vprofile.plot(y, dose_map[:, default_col], lw=2, color='green')
    ax_vprofile.set_title(f"Vertical Profile (column {default_col})")
    ax_vprofile.set_xlabel("Y")
    ax_vprofile.set_ylabel("Dose (Gy)")
    ax_vprofile.grid(True)
    ax_vprofile.set_ylim(0, max_dose)

    # Create slider for the horizontal reference (row)
    ax_slider_row = fig.add_axes([0.02, 0.3, 0.03, 0.6])
    # Note: instead of using invert_yaxis(), we map the value in the callback
    slider_row = Slider(ax_slider_row, 'Row (Y)', 0, nx - 1,
                        valinit=default_slider_value, valstep=1, 
                        orientation='vertical', color='dimgray')

    # Create slider for the vertical reference (column)
    ax_slider_col = fig.add_axes([0.13, 0.20, 0.4, 0.03])
    slider_col = Slider(ax_slider_col, 'Column (X)', 0, ny - 1,
                        valinit=default_col, valstep=1, color='dimgray')

    # Update function for the row slider:
    def update_row(val):
        # Apply the inverse transformation: the actual row index is (nx - 1) - slider value
        row = int((nx - 1) - slider_row.val)
        line_h.set_ydata([y[row], y[row]])
        h_profile_line.set_ydata(dose_map[row, :])
        ax_hprofile.set_title(f"Horizontal Profile (row {row})")
        # Manually update the slider text to show the actual row index
        slider_row.valtext.set_text(str(row))
        fig.canvas.draw_idle()

    # Update function for the column slider:
    def update_col(val):
        col = int(slider_col.val)
        line_v.set_xdata([x[col], x[col]])
        v_profile_line.set_ydata(dose_map[:, col])
        ax_vprofile.set_title(f"Vertical Profile (column {col})")
        fig.canvas.draw_idle()

    # Connect the sliders to the update functions
    slider_row.on_changed(update_row)
    slider_col.on_changed(update_col)

    # Save the figure if requested
    if save_fig:
        fig.savefig(f"{output_name}_dose_profiles.png", bbox_inches='tight', dpi=300)

    plt.show()


# Example usage:
display_dose_profile('dose_map_multi_copy.npy', save_fig=True)
# display_dose_profile('mama_TPS.dcm')
