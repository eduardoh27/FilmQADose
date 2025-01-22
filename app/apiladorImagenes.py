import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
"""
nombres=["/home/carlos/Escritorio/MedidasH100/0Gy.tif",
"/home/carlos/Escritorio/MedidasH100/02GyVolteada.tif",
"/home/carlos/Escritorio/MedidasH100/05Gy.tif",
"/home/carlos/Escritorio/MedidasH100/1Gy.tif",
"/home/carlos/Escritorio/MedidasH100/2Gy.tif",
"/home/carlos/Escritorio/MedidasH100/4Gy.tif",
"/home/carlos/Escritorio/MedidasH100/6GySegunda.tif",
"/home/carlos/Escritorio/MedidasH100/8Gy.tif",
"/home/carlos/Escritorio/MedidasH100/10Gy.tif",
"/home/carlos/Escritorio/MedidasH100/12Gy.tif",
"/home/carlos/Escritorio/MedidasH100/15Gy.tif",
"/home/carlos/Escritorio/MedidasH100/20Gy.tif"]
"""
def apilar_imagenes(nombres, nombre_final):
	arreglos=[]
	forma=(3,4)
	xmax=0
	ymax=0
	for nombre in nombres:
		arr=tiff.imread(nombre)
		x,y,z=arr.shape
		if x>xmax:
			xmax=x
		if y>ymax:
			ymax=y
		arreglos.append(arr)
	arrfinal=np.zeros(shape=(forma[0]*xmax,forma[1]*ymax,3))    

	i=0
	j=0
	for arr in arreglos:
		arrfinal[i*xmax:i*xmax+arr.shape[0],j*ymax:j*ymax+arr.shape[1],:]=arr
		i+=1
		if i>forma[0]-1:
			i=0
			j+=1
	esa=arrfinal/2**16        
	tiff.imsave(nombre_final,arrfinal)    

