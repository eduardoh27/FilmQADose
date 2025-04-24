import pymedphys
import numpy as np
import matplotlib.pyplot as plt
from pydicom import dcmread


def calculate_gamma(TPS_map_path, verification_map_path, dd, mm, plot):
    """
    Calculate the gamma index between two dose maps.
    """

    ds1 = dcmread(TPS_map_path)
    TPS_map = ds1.pixel_array * ds1.DoseGridScaling

    verification_map = np.load(verification_map_path)

    reference = TPS_map / np.max(TPS_map)
    evaluation = verification_map / np.max(verification_map)

    # Define spatial axes
    x1 = np.linspace(0, 512 * ds1.PixelSpacing[0], 512)
    y1 = np.linspace(0, 512 * ds1.PixelSpacing[1], 512)

    x2 = np.linspace(0, 512 * ds1.PixelSpacing[0], 512)
    y2 = np.linspace(0, 512 * ds1.PixelSpacing[1], 512)

    ejes1 = (x1, y1)
    ejes2 = (x2, y2)

    # Gamma analysis parameters
    gamma_options = {
        'dose_percent_threshold': dd,
        'distance_mm_threshold': mm,
        'lower_percent_dose_cutoff': 5,
        'interp_fraction': 10,
        'max_gamma': 5,
        'random_subset': None,
        'local_gamma': False,
        'ram_available': 5 * (2 ** 29)
    }

    # Compute Gamma index
    gamma = pymedphys.gamma(
        ejes1, reference,
        ejes2, evaluation,
        **gamma_options
    )

    if plot:
        # Plot gamma matrix
        plot_gamma_matrix(gamma, gamma_options)
        # Plot gamma histogram
        plot_gamma_histogram(gamma, gamma_options)

    valid_gamma = gamma[~np.isnan(gamma)]
    pass_ratio = np.sum(valid_gamma <= 1) / len(valid_gamma)
    
    return pass_ratio



def plot_gamma_matrix(gamma_matrix, gamma_options, title = None):
    """
    Plot the gamma matrix.
    """
    if title is None:
        title = f'Gamma Matrix {gamma_options["distance_mm_threshold"]}mm/{gamma_options["dose_percent_threshold"]}\%'
    #print(gamma_matrix)
    plt.imshow(gamma_matrix, cmap='inferno')
    plt.title(title, fontsize=14)
    plt.colorbar()
    valid_gamma = gamma_matrix[~np.isnan(gamma_matrix)]
    #print(valid_gamma)
    pass_ratio = np.sum(valid_gamma <= 1) / len(valid_gamma)
    print("El porcentaje de aprobación es", pass_ratio)

def plot_gamma_histogram(gamma_matrix, gamma_options, title = None):
    """
    Plot the histogram of gamma values.
    """
    if title is None:
        title = f'Gamma Histogram {gamma_options["distance_mm_threshold"]}mm/{gamma_options["dose_percent_threshold"]}\%'
        
    valid_gamma = gamma_matrix[~np.isnan(gamma_matrix)]
    num_bins = (gamma_options['interp_fraction'] * gamma_options['max_gamma'])
    bins = np.linspace(0, gamma_options['max_gamma'], num_bins + 1)

    plt.figure(figsize=(8, 6))
    plt.hist(valid_gamma, bins, density=False, alpha=0.7, color='blue')

    # Add a vertical red dashed line at x=1 (pass threshold)
    plt.axvline(x=1, color='red', linestyle='--', linewidth=2, label="Pass Threshold")

    # Display pass percentage as text
    pass_ratio = np.sum(valid_gamma <= 1) / len(valid_gamma)
    plt.text(1.1, 0.9 * plt.ylim()[1], f"Pass rate: {pass_ratio * 100:.2f}\%", color='black', fontsize=12)

    plt.title(title, fontsize=14)
    plt.xlabel(r'$\gamma$ index')
    plt.ylabel('Number of reference points')
    plt.legend()
    plt.show()


