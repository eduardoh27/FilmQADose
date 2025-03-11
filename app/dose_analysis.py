import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec

def display_dose_profile(dose_map, x=None, y=None):
    """
    Interactively displays the dose map and its profiles (horizontal and vertical)
    organized as follows:
      - On the left, the dose map is shown.
      - On the right, two graphs are displayed:
          the horizontal profile (top) and the vertical profile (bottom).
      - The slider to select the row (for the horizontal profile) is shown on the left side (vertical orientation).
      - The slider to select the column (for the vertical profile) is shown at the bottom (horizontal orientation).
    
    Additionally, the dose axis in the profiles is fixed between 0 and the maximum dose in the map.
    
    Parameters:
      - dose_map: 2D numpy array containing the dose map.
      - x: (optional) 1D array with horizontal coordinates. If None, indices are used.
      - y: (optional) 1D array with vertical coordinates. If None, indices are used.
    """
    # Dimensions of the dose map
    nx, ny = dose_map.shape
    if x is None:
        x = np.arange(ny)
    if y is None:
        y = np.arange(nx)
    
    max_dose = dose_map.max()
    
    fig = plt.figure(figsize=(12, 8))
    
    # Main grid for the plots (from y=0.3 to y=0.9) with reduced horizontal space (wspace)
    gs_main = gridspec.GridSpec(1, 2, width_ratios=[3, 2],
                                 left=0.1, right=0.9, top=0.9, bottom=0.3, wspace=0.1)
    
    # Left panel: dose map
    ax_map = fig.add_subplot(gs_main[0, 0])
    extent = [x[0], x[-1], y[0], y[-1]]
    im = ax_map.imshow(dose_map, cmap='inferno', origin='lower', extent=extent)
    # Add colorbar with label
    cbar = plt.colorbar(im, ax=ax_map)
    cbar.set_label('Dose (Gy)')
    
    ax_map.set_title("Dose Map")
    ax_map.set_xlabel("X")
    ax_map.set_ylabel("Y")
    
    # Initial values: central row and central column
    default_row = nx // 2
    default_col = ny // 2
    
    # Draw reference lines on the map: horizontal and vertical
    line_h = ax_map.axhline(y=y[default_row], color='cyan', lw=2)
    line_v = ax_map.axvline(x=x[default_col], color='lime', lw=2)
    
    # Right panel: profiles in two rows (vertically separated)
    gs_profiles = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[0, 1], hspace=0.4)
    
    # Horizontal profile: Dose (Gy) vs X (selected row)
    ax_hprofile = fig.add_subplot(gs_profiles[0, 0])
    h_profile_line, = ax_hprofile.plot(x, dose_map[default_row, :], lw=2)
    ax_hprofile.set_title(f"Horizontal Profile (row {default_row})")
    ax_hprofile.set_xlabel("X")
    ax_hprofile.set_ylabel("Dose (Gy)")
    ax_hprofile.grid(True)
    ax_hprofile.set_ylim(0, max_dose)
    
    # Vertical profile: Dose (Gy) vs Y (selected column)
    ax_vprofile = fig.add_subplot(gs_profiles[1, 0])
    v_profile_line, = ax_vprofile.plot(y, dose_map[:, default_col], lw=2, color='green')
    ax_vprofile.set_title(f"Vertical Profile (column {default_col})")
    ax_vprofile.set_xlabel("Y")
    ax_vprofile.set_ylabel("Dose (Gy)")
    ax_vprofile.grid(True)
    ax_vprofile.set_ylim(0, max_dose)
    
    # Vertical slider for the row (horizontal profile) placed to the left of the dose map
    ax_slider_row = fig.add_axes([0.02, 0.35, 0.03, 0.5])  # [left, bottom, width, height]
    slider_row = Slider(ax_slider_row, 'Row (Y)', 0, nx - 1, 
                        valinit=default_row, valstep=1, orientation='vertical', color='dimgray')
    
    # Horizontal slider for the column (vertical profile), repositioned at an intermediate point
    ax_slider_col = fig.add_axes([0.15, 0.20, 0.7, 0.03])
    slider_col = Slider(ax_slider_col, 'Column (X)', 0, ny - 1, 
                        valinit=default_col, valstep=1, color='dimgray')
    
    # Update function for the vertical slider (row)
    def update_row(val):
        row = int(slider_row.val)
        line_h.set_ydata([y[row], y[row]])
        h_profile_line.set_ydata(dose_map[row, :])
        ax_hprofile.set_title(f"Horizontal Profile (row {row})")
        # Fix the dose axis
        ax_hprofile.set_ylim(0, max_dose)
        fig.canvas.draw_idle()
    
    # Update function for the horizontal slider (column)
    def update_col(val):
        col = int(slider_col.val)
        line_v.set_xdata([x[col], x[col]])
        v_profile_line.set_ydata(dose_map[:, col])
        ax_vprofile.set_title(f"Vertical Profile (column {col})")
        # Fix the dose axis
        ax_vprofile.set_ylim(0, max_dose)
        fig.canvas.draw_idle()
    
    slider_row.on_changed(update_row)
    slider_col.on_changed(update_col)
    
    plt.show()

    # save the figure
    #fig.savefig('dose_profile.png', bbox_inches='tight')

# Example usage:
if __name__ == "__main__":
    # Load the dose map from a .npy file
    dose_map = np.load('dose_map_channel_0.npy')
    display_dose_profile(dose_map)
