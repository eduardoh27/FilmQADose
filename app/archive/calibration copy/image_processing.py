"""
image_processing.py

This module provides functions for processing images, including reading TIFF and DICOM images,
applying filters, displaying images, cropping regions of interest (ROIs), and performing template matching.
"""

import os
import cv2
import pydicom
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.restoration import unsupervised_wiener, wiener
from scipy.ndimage import median_filter
from scipy.signal import wiener as wiener_scipy
from scipy.signal.windows import gaussian
from numpy.fft import fft2, ifft2
from PIL import Image
from PIL.TiffTags import TAGS


def tif_bits_per_channel(image_path):
    """
    Retrieves the number of bits per channel for a TIFF image.

    Parameters
    ----------
    image_path : str
        Path to the TIFF image.

    Returns
    -------
    int
        The number of bits per channel.

    Raises
    ------
    ValueError
        If the BitsPerSample tag is not found or if its values are not identical.
    """
    with Image.open(image_path) as img:
        meta_dict = {TAGS[key]: img.tag[key] for key in img.tag.keys()}
        bits_per_sample = meta_dict.get('BitsPerSample', None)
    
        if bits_per_sample is None:
            raise ValueError('BitsPerSample tag not found in the image.')
        # If there is more than one value, ensure they are all identical.
        elif len(set(bits_per_sample)) != 1:
            raise ValueError('BitsPerSample values are not identical.')
        return bits_per_sample[0]


def read_image(image_path): 
    """
    Reads an image from the specified file path using skimage.io.
    For TIFF images, a custom reader is used.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    image : ndarray
        The loaded image as a NumPy array.
    """
    if image_path.lower().endswith('.tif') or image_path.lower().endswith('.tiff'):
        image = read_image_tif(image_path)
    else:
        image = io.imread(image_path)
    
    return image


def read_image_tif(image_path):
    """
    Reads a TIFF image from the specified file path using tifffile.
    Normalizes the image to the range [0, 1] based on its bit depth.

    Parameters
    ----------
    image_path : str
        Path to the TIFF image file.

    Returns
    -------
    image : ndarray
        The loaded and normalized image as a NumPy array.
    """
    # Read the TIFF image using tifffile
    image = tifffile.imread(image_path)
    # Print the number of bits per channel (e.g., 8, 16, or 32)
    print(f"Bits per channel: {tif_bits_per_channel(image_path)}")
    image = image / (2 ** tif_bits_per_channel(image_path) - 1)
    return image


def show_image(image, title=None, show_labels=True, show_axis=True):
    """
    Displays the provided image along with an optional coordinate system
    to facilitate region of interest (ROI) selection.

    Parameters
    ----------
    image : ndarray
        The image to display.
    title : str, optional
        Title for the displayed image.
    show_labels : bool, optional
        If True, display axis labels.
    show_axis : bool, optional
        If True, display the axis.

    Returns
    -------
    image : ndarray
        The same image that was displayed.
    """
    # Create a figure and display the image (x corresponds to columns, y to rows)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    if show_labels:
        plt.xlabel("X Coordinate (column)")
        plt.ylabel("Y Coordinate (row)")
    if show_axis:
        plt.axis('on')
    plt.show()

    return image


def save_image(image, output_path):
    """
    Saves an image (NumPy array) to the specified file path.

    Parameters
    ----------
    image : ndarray
        NumPy array representing the image to be saved.
    output_path : str
        Full path of the output file (including the file format, e.g., .tiff or .png).
    """
    io.imsave(output_path, image)


def filter_image(image: np.ndarray, filter_type: str = None, kernel_size: int = 3) -> np.ndarray:
    """
    Applies a filtering or preprocessing operation to the provided image.

    Parameters
    ----------
    image : np.ndarray
        Input image data (grayscale).
    filter_type : str, optional
        Type of filter to apply. Possible values include:
          "none", "gaussian", "median", "sobel", etc.
        This function currently demonstrates placeholder options.
    kernel_size : int, optional
        The size of the filter kernel.

    Returns
    -------
    np.ndarray
        The filtered image as a NumPy array.
    """

    def wiener_filter_manual(img, K=30):
        kernel_size = 3
        h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
        h = np.dot(h, h.transpose())
        h /= np.sum(h)
        kernel = h
        kernel /= np.sum(kernel)
        transformed = fft2(np.copy(img))
        kernel = fft2(kernel, s = img.shape)
        kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
        transformed = transformed * kernel
        wiener = np.abs(ifft2(transformed))
        return wiener

    # Ensure the image has only one channel (grayscale)
    if image.ndim > 2 and image.shape[-1] > 1:
        raise ValueError("The input image must be single-channel (grayscale).")

    if filter_type is None:
        filtered = image

    elif filter_type == "median":
        filtered = median_filter(image, size=kernel_size)

    elif filter_type == "wiener-scipy":
        filtered = wiener_scipy(image, mysize=(kernel_size, kernel_size))
    
    elif filter_type == "wiener-manual":
        filtered = wiener_filter_manual(image)

    elif filter_type == "wiener-skimage-1":
        k = kernel_size
        psf = np.ones((k, k)) / (k * k)
        filtered, _ = unsupervised_wiener(image, psf)

    elif filter_type == "wiener-skimage-2" or filter_type == "wiener":
        k = kernel_size
        psf = np.ones((k, k)) / (k * k)
        filtered = wiener(image, psf, balance=0.35)
    
    else:
        raise ValueError(f"Unknown filter type: {filter_type}.")

    return filtered


def reflect_image(image, mode='horizontal'):
    """
    Reflects (flips) an image along the specified axis.

    Parameters
    ----------
    image : ndarray
        The input image as a NumPy array.
    mode : str, optional
        'horizontal' to flip left to right, 'vertical' to flip top to bottom.

    Returns
    -------
    ndarray
        The reflected image.
    """
    if mode.lower() == 'horizontal':
        # Horizontal flip: reflect left <-> right
        reflected_image = np.fliplr(image)
    elif mode.lower() == 'vertical':
        # Vertical flip: reflect top <-> bottom
        reflected_image = np.flipud(image)
    else:
        raise ValueError("Mode must be 'horizontal' or 'vertical'.")
    
    return reflected_image


def rotate_image(image, times=1, direction='clockwise'):
    """
    Rotates an image by 90-degree increments.

    Parameters
    ----------
    image : ndarray
        NumPy array representing the image.
    times : int, optional
        Number of 90° rotations (e.g., 1 for 90°, 2 for 180°, etc.).
    direction : str, optional
        'clockwise' for clockwise rotation, or 'counterclockwise' for counterclockwise rotation.

    Returns
    -------
    ndarray
        The rotated image.
    """
    # Limit the number of rotations to the range [0, 3]
    times = times % 4

    if direction.lower() == 'clockwise':
        # np.rot90 rotates counterclockwise by default;
        # rotate by a negative number of times to rotate clockwise.
        k = -times
    elif direction.lower() == 'counterclockwise':
        k = times
    else:
        raise ValueError("The 'direction' parameter must be 'clockwise' or 'counterclockwise'.")

    rotated_image = np.rot90(image, k=k)
    return rotated_image


def crop_square_roi(image, x, y, side_length):
    """
    Crops a square region of interest (ROI) from the image.

    Parameters
    ----------
    image : ndarray
        Input image as a NumPy array.
    x : int
        X-coordinate (column) of the top-left corner of the ROI.
    y : int
        Y-coordinate (row) of the top-left corner of the ROI.
    side_length : int
        Length of one side of the square ROI.

    Returns
    -------
    ndarray
        The cropped ROI as a NumPy array.

    Raises
    ------
    ValueError
        If the ROI exceeds the image boundaries.
    """
    # Get image dimensions (height and width)
    max_y, max_x = image.shape[:2]
    if y + side_length > max_y or x + side_length > max_x:
        raise ValueError("The ROI exceeds the image boundaries. Adjust (x, y) or 'side_length'.")
    roi = image[y:y+side_length, x:x+side_length]
    return roi


def get_real_dimensions(image_path):
    """
    Calculates the physical dimensions (in centimeters) of an image from its metadata.
    Supports TIFF and DICOM image formats.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    tuple
        (width_cm, height_cm) - the width and height in centimeters.

    Raises
    ------
    ValueError
        If required metadata is missing or the format is unsupported.
    """
    ext = os.path.splitext(image_path)[-1].lower()

    if ext in [".tif", ".tiff"]:
        # Load the TIFF image using PIL
        img = Image.open(image_path)
        # Get the resolution in DPI (dots per inch)
        dpi = img.info.get("dpi", (300, 300))  # Assume 300 dpi if not specified
        # Get image dimensions in pixels
        width_px, height_px = img.size
        # Convert dimensions to centimeters (1 inch = 2.54 cm)
        width_cm = (width_px / dpi[0]) * 2.54
        height_cm = (height_px / dpi[1]) * 2.54
        return width_cm, height_cm

    elif ext in [".dcm"]:
        # Load the DICOM image
        ds = pydicom.dcmread(image_path)
        rows = ds.Rows
        cols = ds.Columns
        # Check for PixelSpacing information
        if hasattr(ds, "PixelSpacing"):
            pixel_spacing = ds.PixelSpacing  # [row spacing, column spacing] in mm
            width_cm = (cols * float(pixel_spacing[0])) / 10
            height_cm = (rows * float(pixel_spacing[1])) / 10
            return width_cm, height_cm
        else:
            raise ValueError("DICOM - PixelSpacing information not found.")
    else:
        raise ValueError("Unsupported format. Use TIFF or DICOM.")


def template_matching(TPS_map_path, film_tif_path, output_path):
    """
    Performs template matching between a dose map (TPS map) from a DICOM file and an original film TIFF image.
    The function aligns the images using template matching, crops the matching region,
    applies transformations to standardize orientation, and saves the result.

    Parameters
    ----------
    TPS_map_path : str
        File path to the TPS map DICOM file.
    film_tif_path : str
        File path to the original film TIFF image.
    output_path : str
        File path where the processed (cropped and transformed) image will be saved.
    """
    # Define file paths
    imageA_path = TPS_map_path
    film_tif_full_path = film_tif_path

    # Load the dose map (Image A) from the DICOM file
    imageA = pydicom.dcmread(imageA_path).pixel_array
    imageA = imageA.astype(np.float32)

    # Load the original film TIFF image (preserving its properties)
    film_orig = tifffile.imread(film_tif_full_path)

    # For processing, use the green channel if the image is multichannel; otherwise, use a copy of the original image
    if film_orig.ndim == 3:
        imageB = film_orig[:, :, 1]
    else:
        imageB = film_orig.copy()
    imageB = imageB.astype(np.float32)

    # Verify that the images have been loaded correctly
    if imageA is None or imageA.size == 0:
        print(f'Error loading Image A from {imageA_path}')
    if imageB is None or imageB.size == 0:
        print(f'Error loading Image B from {film_tif_full_path}')

    # Assume the images are 2D arrays
    heightA, widthA = imageA.shape
    heightB, widthB = imageB.shape

    # Obtain real dimensions in centimeters for both images
    widthA_cm, heightA_cm = get_real_dimensions(imageA_path)
    print("Image A dimensions (cm):", widthA_cm, heightA_cm)
    widthB_cm, heightB_cm = get_real_dimensions(film_tif_full_path)
    print("Image B dimensions (cm):", widthB_cm, heightB_cm)

    # Calculate the resolution (pixels per cm) for each image
    resolutionA_x = widthA / widthA_cm
    resolutionA_y = heightA / heightA_cm
    resolutionB_x = widthB / widthB_cm
    resolutionB_y = heightB / heightB_cm

    print(f'Image A resolution: {resolutionA_x:.2f} px/cm x {resolutionA_y:.2f} px/cm')
    print(f'Image B resolution: {resolutionB_x:.2f} px/cm x {resolutionB_y:.2f} px/cm')

    # If the resolutions differ, rescale Image A to match the physical scale of Image B
    if abs(resolutionA_x - resolutionB_x) > 1e-2 or abs(resolutionA_y - resolutionB_y) > 1e-2:
        new_widthA = int(widthA_cm * resolutionB_x)
        new_heightA = int(heightA_cm * resolutionB_y)
        imageA = cv2.resize(imageA, (new_widthA, new_heightA), interpolation=cv2.INTER_LINEAR)
        print(f'Rescaled Image A to: {new_widthA}x{new_heightA} pixels')

    # Normalize both images to the range [0, 1]
    imageA = cv2.normalize(imageA, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    imageB = cv2.normalize(imageB, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Transform Image B for template matching:
    # Apply horizontal flip and 180° rotation (equivalent to a vertical flip)
    imageB_trans = cv2.flip(imageB, 1)
    imageB_trans = cv2.rotate(imageB_trans, cv2.ROTATE_180)
    # Invert intensities (dark becomes white and vice versa)
    imageB_trans = 1.0 - imageB_trans

    # Display the transformed images for template matching
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    im1 = axs[0].imshow(imageA, cmap='gray')
    axs[0].set_title("Image A (Template)")
    axs[0].axis("off")
    fig.colorbar(im1, ax=axs[0])
    im2 = axs[1].imshow(imageB_trans, cmap='gray')
    axs[1].set_title("Transformed and Inverted Image B")
    axs[1].axis("off")
    fig.colorbar(im2, ax=axs[1])
    #plt.show()

    # Apply template matching using the TM_CCOEFF_NORMED method
    result = cv2.matchTemplate(imageB_trans, imageA, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print(f'Maximum correlation value: {max_val:.3f}')
    print(f'Top-left location in transformed image: {max_loc}')

    # Get the template size (dimensions of Image A)
    template_height, template_width = imageA.shape

    # Given that the transformation applied to imageB is equivalent to a vertical flip,
    # the coordinate mapping from imageB_trans to the original film image is:
    #   (x, y)_orig = (x, H - y - template_height)
    # where H is the height of the original Image B.
    H = heightB  
    col_t, row_t = max_loc
    orig_top = H - row_t - template_height
    orig_left = col_t
    orig_bottom = H - row_t
    orig_right = col_t + template_width

    print('Coordinates in the original image for the crop:')
    print(f'  Top-left: ({orig_left}, {orig_top})')
    print(f'  Bottom-right: ({orig_right}, {orig_bottom})')

    # To visualize the bounding box on the original image:
    # Convert film_orig to 8-bit for display (preserving channels if present)
    if film_orig.ndim == 3:
        if film_orig.dtype != np.uint8:
            film_disp = cv2.normalize(film_orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            film_disp = film_orig.copy()
    else:
        film_disp = cv2.normalize(film_orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        film_disp = cv2.cvtColor(film_disp, cv2.COLOR_GRAY2BGR)
        
    # Draw the bounding box on the original image
    cv2.rectangle(film_disp, (orig_left, orig_top), (orig_right, orig_bottom), (0, 0, 255), 2)
    plt.figure(figsize=(6,6))
    # Convert BGR to RGB for matplotlib if the image is in color
    if film_disp.ndim == 3:
        film_disp_rgb = cv2.cvtColor(film_disp, cv2.COLOR_BGR2RGB)
        plt.imshow(film_disp_rgb)
    else:
        plt.imshow(film_disp, cmap='gray')
    plt.title("Original Image with Bounding Box")
    plt.axis("off")
    #plt.show()

    # Crop the identified region from the original image (without transformation)
    cropped = film_orig[orig_top:orig_bottom, orig_left:orig_right].copy()
    
    # To match the orientation of the transformed image, apply the same transformation:
    # horizontal flip and 180° rotation.
    cropped_trans = cv2.flip(cropped, 1)
    cropped_trans = cv2.rotate(cropped_trans, cv2.ROTATE_180)

    # Save the transformed crop as a TIFF file (preserving image properties as much as possible)
    tifffile.imwrite(output_path, cropped_trans)
    print(f'Cropped (flipped and rotated) image saved as {output_path}')
