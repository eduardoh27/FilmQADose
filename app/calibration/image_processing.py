import numpy as np
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
from skimage import io


def read_image(image_path):
    """
    Reads an image from the specified file path using skimage.io.
    Uses the 'pil' plugin for TIFF images.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file.
    
    Returns:
    --------
    image : ndarray
        The loaded image as a NumPy array.
    """
    if image_path.lower().endswith('.tif') or image_path.lower().endswith('.tiff'):
        image = io.imread(image_path, plugin='pil')
    else:
        image = io.imread(image_path)
    
    return image


def show_image(image, title=None, show_labels=True, show_axis=True):
    """
    Displays the original image with a coordinate system to facilitate 
    ROI selection. Returns the loaded image for further use.
    """

    # Display the image with axes (x corresponds to columns, y to rows)
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
    Saves an image (NumPy array) to the specified path.
    
    Parameters:
    -----------
    image : ndarray
        NumPy array representing the image to be saved.
    output_path : str
        Full path of the output file (including format, e.g., .tiff or .png).
    """
    io.imsave(output_path, image)


def filter_image(image: np.ndarray, filter_type: str = None) -> np.ndarray:
    """
    Apply a filtering or preprocessing operation to the given image.

    Parameters
    ----------
    image : np.ndarray
        The input image data as a NumPy array. This can be grayscale or color.
    filter_type : str, optional
        The type of filter to apply. Possible values might be:
          "none", "gaussian", "median", "sobel", etc.
        Currently, this is just a placeholder demonstrating how you might
        structure such a function.

    Returns
    -------
    np.ndarray
        The filtered or processed image as a NumPy array.
    """

    if filter_type is None:
        filtred = image

    elif filter_type == "median":
        filtered = median_filter(image, size=3)
    
    return filtered


def reflect_image(image, mode='horizontal'):
    """
    Reflects (flips) an image along the specified axis: horizontal or vertical.
    
    Parameters:
    -----------
    image : ndarray
        Image loaded as a NumPy array.
    mode : str, optional
        'horizontal' to flip from left to right (horizontal flip).
        'vertical' to flip from top to bottom (vertical flip).
    
    Returns:
    --------
    reflected_image : ndarray
        Resulting image after applying the reflection.
    """

    if mode.lower() == 'horizontal':
        # Horizontal flip = reflect left ↔ right
        reflected_image = np.fliplr(image)
    elif mode.lower() == 'vertical':
        # Vertical flip = reflect top ↕ bottom
        reflected_image = np.flipud(image)
    else:
        raise ValueError("Mode must be 'horizontal' or 'vertical'.")
    
    return reflected_image


def rotate_image(image, times=1, direction='clockwise'):
    """
    Rotates an image in 90-degree increments.
    
    Parameters:
    -----------
    image : ndarray
        NumPy array representing the image.
    times : int, optional
        Number of times to rotate the image in 90° increments.
        1 = 90°, 2 = 180°, 3 = 270°, 4 = 360° (equivalent to the original image).
    direction : str, optional
        'clockwise' (rotation in the clockwise direction) or 'counterclockwise' (rotation in the counterclockwise direction).
    
    Returns:
    --------
    rotated_image : ndarray
        Image after applying the rotation.
    """
    # Ensure the number of rotations stays within [0..3]
    # (4 rotations of 90° equal 360°, which leaves the image unchanged)
    times = times % 4

    if direction.lower() == 'clockwise':
        # np.rot90 rotates the image 90° counterclockwise by default.
        # To rotate clockwise, we can rotate -k times instead of k.
        k = -times
    elif direction.lower() == 'counterclockwise':
        k = times
    else:
        raise ValueError("The 'direction' parameter must be 'clockwise' or 'counterclockwise'.")

    rotated_image = np.rot90(image, k=k)
    return rotated_image


def crop_square_roi(image, x, y, side_length):
    """
    Crops a square ROI from the image.
    
    Parameters:
    -----------
    image : ndarray
        Loaded image (NumPy array format).
    x : int
        X-coordinate (column) of the top-left corner of the ROI.
    y : int
        Y-coordinate (row) of the top-left corner of the ROI.
    side_length : int
        Size of the square ROI's side.

    Returns:
    --------
    roi : ndarray
        Cropped image corresponding to the square ROI.
    """
    # Validation to ensure the ROI does not exceed image boundaries
    # (row = y, column = x)
    max_y, max_x = image.shape[:2]  # First two values: image height and width
    if y + side_length > max_y or x + side_length > max_x:
        raise ValueError("The ROI exceeds the image boundaries. Adjust (x, y) or 'side_length'.")

    roi = image[y:y+side_length, x:x+side_length]
    return roi
