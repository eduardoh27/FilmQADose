import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
def invertir_imagenes(nombres):
	for nombre in nombres:
		ima=tiff.imread(nombre)
		imaM=np.flip(ima,axis=0)
		nomFin=nombre+'-invertido'
		tiff.imsave(nomFin,imaM)


