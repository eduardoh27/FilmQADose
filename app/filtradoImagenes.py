import numpy as np
import matplotlib.pyplot as plt
import cv2 
#import despike # EDO
import tifffile as tiff
#from skimage import color, data, restoration # EDO

from scipy.signal import convolve2d


def wiener_filter(img, kernel, K):
	kernel /= np.sum(kernel)
	dummy = np.copy(img)
	dummy = fft2(dummy)
	kernel = fft2(kernel, s = img.shape)
	kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
	dummy = dummy * kernel
	dummy = np.abs(ifft2(dummy))
	return dummy
	
def filtrar_imagen(img,tipoFiltro):
	result=img
	
	if 'mediana' in tipoFiltro:
		result=cv2.medianBlur(img.astype('float32'),5)
	if 'promedio' in tipoFiltro:
		kernel = np.ones((5,5),np.float64)/25
		result = cv2.filter2D(result,-1,kernel)
		
	return result


	
	
#img=tiff.imread('Dosis0a10.tif')
#img=tiff.imread('FondoNegro-1.tif')
#img2=tiff.imread('Dosis0a10.tif')
#img=img2-img
#img=(img/2**16)
#plt.imshow(img)
#plt.figure()

#kernel = np.ones((5,5),np.float64)/25
#dst = cv2.filter2D(img,-1,kernel)
#tiff.imsave('dosis0-10Filtradas.tif',dst)
#plt.imshow(dst)

#plt.figure()

#median=median*2**16

#plt.imshow(median)

#plt.show()
