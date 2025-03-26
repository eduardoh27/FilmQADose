import numpy as np
import matplotlib.pyplot as plt
import tifffile
import cv2
import pydicom
import os
from skimage import io
from skimage.util import img_as_float, img_as_ubyte, img_as_uint
from scipy.ndimage import median_filter
from scipy.signal import convolve2d, wiener
from scipy.signal.windows import gaussian
from numpy.fft import fft2, ifft2
from PIL import Image
from PIL.TiffTags import TAGS


def tif_bits_per_channel(image_path):

    with Image.open(image_path) as img:
        meta_dict = {TAGS[key] : img.tag[key] for key in img.tag.keys()}
        bits_per_sample = meta_dict.get('BitsPerSample', None)
    
        if bits_per_sample is None:
            raise ValueError('No se encontró el tag BitsPerSample en la imagen.')
        # si tiene más de un valor, verificar que todos sean iguales
        elif len(set(bits_per_sample)) != 1:
            raise ValueError('Los valores de BitsPerSample no son iguales.')
        return bits_per_sample[0]

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
        image = read_image_tif(image_path)
    else:
        image = io.imread(image_path)
    
    return image

def read_image_tif(image_path):
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
    #image = io.imread(image_path, plugin='pil')

    image = tifffile.imread(image_path)
    image = image / (2**tif_bits_per_channel(image_path) - 1)
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


def filter_image(image: np.ndarray, filter_type: str = None, kernel_size: int = 3) -> np.ndarray:
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

    # Ensure the image has only one channel
    if image.ndim > 2 and image.shape[-1] > 1:
        raise ValueError("The input image must be single-channel (grayscale).")

    def wiener_filter1(img, K=30):
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


    if filter_type is None:
        filtered = image

    elif filter_type == "median":
        filtered = median_filter(image, size=kernel_size)

    elif filter_type == "wiener":
        filtered = wiener(image, mysize=(kernel_size, kernel_size))
    
    elif filter_type == "wiener-manual":
        filtered = wiener_filter1(image)

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


def get_real_dimensions(image_path):

    ext = os.path.splitext(image_path)[-1].lower()

    if ext in [".tif", ".tiff"]:
        # Cargar la imagen TIFF
        img = Image.open(image_path)

        # Obtener resolución en DPI (dots per inch)
        dpi = img.info.get("dpi", (300, 300))  # Si no hay dpi, asumir 300 ppp por defecto

        # Obtener dimensiones en píxeles
        width_px, height_px = img.size

        # Convertir a centímetros (1 pulgada = 2.54 cm)
        width_cm = (width_px / dpi[0]) * 2.54
        height_cm = (height_px / dpi[1]) * 2.54

        return width_cm, height_cm

    elif ext in [".dcm"]:
        # Cargar la imagen DICOM
        ds = pydicom.dcmread(image_path)

        # Obtener el tamaño de la imagen en píxeles
        rows = ds.Rows
        cols = ds.Columns

        # Obtener el tamaño del píxel en mm (PixelSpacing contiene [espaciado fila, espaciado columna])
        if hasattr(ds, "PixelSpacing"):
            pixel_spacing = ds.PixelSpacing  
            width_cm = (cols * float(pixel_spacing[0])) / 10
            height_cm = (rows * float(pixel_spacing[1])) / 10

            return width_cm, height_cm
        else:
            raise ValueError("DICOM - No se encontró información de PixelSpacing.")

    else:
        raise ValueError("Formato no soportado. Usa TIFF o DICOM.")


def template_matching(TPS_map_path, film_tif_path, output_path):

    # Rutas a los archivos
    imageA_path = TPS_map_path
    film_tif_full_path = film_tif_path

    # Cargar el mapa de dosis (imagen A) desde DICOM
    imageA = pydicom.dcmread(imageA_path).pixel_array
    imageA = imageA.astype(np.float32)

    # Cargar la imagen TIFF original (se preservan sus propiedades)
    film_orig = tifffile.imread(film_tif_full_path)

    # Para el procesamiento se utiliza el canal verde si es multicanal;
    # de lo contrario se usa una copia de la imagen original
    if film_orig.ndim == 3:
        imageB = film_orig[:, :, 1]
    else:
        imageB = film_orig.copy()
    imageB = imageB.astype(np.float32)

    # Verificar que los arrays se hayan cargado correctamente
    if imageA is None or imageA.size == 0:
        print(f'Error al cargar la imagen A desde {imageA_path}')
    if imageB is None or imageB.size == 0:
        print(f'Error al cargar la imagen B desde {film_tif_full_path}')

    # Asumir que las imágenes son matrices 2D
    heightA, widthA = imageA.shape
    heightB, widthB = imageB.shape

    # Obtener dimensiones reales en cm (se asume que get_real_dimensions está definida)
    widthA_cm, heightA_cm = get_real_dimensions(imageA_path)
    print("Dimensiones imagen A (cm):", widthA_cm, heightA_cm)
    widthB_cm, heightB_cm = get_real_dimensions(film_tif_full_path)
    print("Dimensiones imagen B (cm):", widthB_cm, heightB_cm)

    # Calcular la resolución en píxeles por cm para cada imagen
    resolutionA_x = widthA / widthA_cm
    resolutionA_y = heightA / heightA_cm
    resolutionB_x = widthB / widthB_cm
    resolutionB_y = heightB / heightB_cm

    print(f'Resolución de la imagen A: {resolutionA_x:.2f} px/cm x {resolutionA_y:.2f} px/cm')
    print(f'Resolución de la imagen B: {resolutionB_x:.2f} px/cm x {resolutionB_y:.2f} px/cm')

    # Si las resoluciones difieren, reescalar la imagen A para igualar la escala física
    if abs(resolutionA_x - resolutionB_x) > 1e-2 or abs(resolutionA_y - resolutionB_y) > 1e-2:
        new_widthA = int(widthA_cm * resolutionB_x)
        new_heightA = int(heightA_cm * resolutionB_y)
        imageA = cv2.resize(imageA, (new_widthA, new_heightA), interpolation=cv2.INTER_LINEAR)
        print(f'Reescalada de la imagen A a: {new_widthA}x{new_heightA} píxeles')

    # Normalización de ambas imágenes al rango [0, 1]
    imageA = cv2.normalize(imageA, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    imageB = cv2.normalize(imageB, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Transformar la imagen B para template matching:
    # Se aplica: 1) reflejo horizontal y 2) rotación 180°.
    # Estas operaciones son equivalentes a un volteo vertical.
    imageB_trans = cv2.flip(imageB, 1)
    imageB_trans = cv2.rotate(imageB_trans, cv2.ROTATE_180)
    # Invertir intensidades (lo oscuro pasa a ser blanco y viceversa)
    imageB_trans = 1.0 - imageB_trans

    # Mostrar las imágenes transformadas (para template matching) 
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    im1 = axs[0].imshow(imageA, cmap='gray')
    axs[0].set_title("Imagen A (Template)")
    axs[0].axis("off")
    fig.colorbar(im1, ax=axs[0])
    im2 = axs[1].imshow(imageB_trans, cmap='gray')
    axs[1].set_title("Imagen B transformada e invertida")
    axs[1].axis("off")
    fig.colorbar(im2, ax=axs[1])
    plt.show()

    # Aplicar template matching (método TM_CCOEFF_NORMED)
    result = cv2.matchTemplate(imageB_trans, imageA, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print(f'Valor máximo de correlación: {max_val:.3f}')
    print(f'Ubicación en la imagen transformada (esquina superior izquierda): {max_loc}')

    # Tamaño del template (imagen A)
    template_height, template_width = imageA.shape

    # Dado que la transformación aplicada a imageB es equivalente a un volteo vertical,
    # la relación entre las coordenadas en imageB_trans y la imagen original (film_orig) es:
    #   (x, y)_orig = (x, H - y - template_height)  para la esquina superior izquierda.
    # Se utiliza H = heightB (la altura de la imagen B original).
    H = heightB  
    col_t, row_t = max_loc
    orig_top = H - row_t - template_height
    orig_left = col_t
    orig_bottom = H - row_t
    orig_right = col_t + template_width

    print(f'Coordenadas en la imagen original para el recorte:')
    print(f'  Esquina superior izquierda: ({orig_left}, {orig_top})')
    print(f'  Esquina inferior derecha: ({orig_right}, {orig_bottom})')

    # Para visualizar el bounding box sobre la imagen original:
    # Convertir film_orig a 8 bits para visualización (manteniendo canales si existen)
    if film_orig.ndim == 3:
        # Si la imagen ya es de 8 bits, se usa directamente; de lo contrario se normaliza
        if film_orig.dtype != np.uint8:
            film_disp = cv2.normalize(film_orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            film_disp = film_orig.copy()
    else:
        film_disp = cv2.normalize(film_orig, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        film_disp = cv2.cvtColor(film_disp, cv2.COLOR_GRAY2BGR)
        
    # Dibujar el bounding box sobre la imagen original
    cv2.rectangle(film_disp, (orig_left, orig_top), (orig_right, orig_bottom), (0, 0, 255), 2)
    plt.figure(figsize=(6,6))
    # Convertir de BGR a RGB para matplotlib (si es a color)
    if film_disp.ndim == 3:
        film_disp_rgb = cv2.cvtColor(film_disp, cv2.COLOR_BGR2RGB)
        plt.imshow(film_disp_rgb)
    else:
        plt.imshow(film_disp, cmap='gray')
    plt.title("Imagen original con bounding box")
    plt.axis("off")
    plt.show()

    # Recortar la región encontrada en la imagen original (sin transformación)
    cropped = film_orig[orig_top:orig_bottom, orig_left:orig_right].copy()
    
    # Ahora, para que el recorte tenga la misma orientación que la imagen transformada,
    # se aplica el mismo proceso: reflejo horizontal y rotación 180°.
    cropped_trans = cv2.flip(cropped, 1)
    cropped_trans = cv2.rotate(cropped_trans, cv2.ROTATE_180)

    # Guardar el recorte transformado como TIFF (se preservan las propiedades en la medida de lo posible)
    tifffile.imwrite(output_path, cropped_trans)
    print(f'Imagen recortada (reflejada y rotada) guardada como {output_path}')
