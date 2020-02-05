#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 18:10:57 2020

@author: araweli

Obtener los rangos para filtrar los colores, analizando las imagenes.

El siguiente paso es filtrar las imagenes, y obtener por un lado solo el color azul, y por otro lado solo el color rojo, dado que las señales de tráfico que queremos identificar son de estos dos colores. Se Debe realizar un analisis para saber por qué valores filtrar.

Para ello se separan los canales de la señal, y se analiza qué valores toman.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
    
#################################################################################################################

# Por otro lado, se hace un recorte de las señales, para intentar obtener así los valores de azul y rojo que se necesita filtrar.

#Señales azules
# Se leen imagenes de señales recortadas, para ver su histograma y ver en que rango esta el azul y el rojo
colorAzul=['/home/arawe/Dropbox/Grado/VA/P2/7Senales/color/1_indicacion.png',
           '/home/arawe/Dropbox/Grado/VA/P2/7Senales/color/1_obligacion2.png',
           '/home/arawe/Dropbox/Grado/VA/P2/7Senales/color/40_obligacion.png']

# Se definen tres array donde iremos guardando los valores de cada canal de cada señal.
L = np.array([])
A = np.array([])
B = np.array([])

#Iiterar sobre el array que tiene las direcciones de las imagenes
for i in range(len(colorAzul)): 
    
    img=cv2.imread(colorAzul[i]) # Para la imagen i
    
    imglab=cv2.cvtColor(img, cv2.COLOR_RGB2Lab) # Convertir a lab
    
    # Separarlos canales
    l = imglab[:,:,0] #De la 3ª dimension, cojo la primera columa. [ancho, alto, canales]
    a = imglab[:,:,1]
    b = imglab[:,:,2] 
    
    # Convertir la matriz con el canal, a un array
    # Se obtuvo una matriz del mismo ancho y alto que la imagen leida, que en lugar de tener tres canales (800,1360,3), ahora solo tiene uno (800,1360). Asi que remodelamos esta matriz para convertirla a un array que contenga todos los datos.

    l = l.reshape(l.shape[0]*l.shape[1])
    a = a.reshape(a.shape[0]*a.shape[1])
    b = b.reshape(b.shape[0]*b.shape[1])
    
    # Se añaden los valores de cada imagen a los array L,A y B. Con los datos de todas las imagenes en un mismo array, se hace un histograma para ver en que valores caen los datos, y asi saber como filtrar los colores.
    L = np.append(L,l)
    A = np.append(A,a)
    B = np.append(B,b)
        
# Creamos un histograma con L, A y B, para ver en que rango se encuentra el color azul.

plt.title('Canal A - Azul')
plt.hist(A, bins = 256, range=(0,255)) # Quitar el flag range para ver solo el rango donde hay valores.
plt.grid(True)
plt.show()
plt.clf()
# Los valores se concentran en: 153-162 y 175-185. Que haya dos intervalos, podria deberse a zonas blancas al recortar las señales.

plt.title('Canal B - Azul')
plt.hist(B, bins =256, range=(0,255))
plt.grid(True)
plt.show()
plt.clf()
# Los valores se concentran en: 130-140

plt.title('Canal L - Azul')
plt.hist(L, bins =256, range=(0,255))
plt.grid(True)
plt.show()
plt.clf()
# Se ve como los valores se reparten entre 40 y 175. Con un pico en 100 y otro en 45.

'''
Estas serían las mascaras:
'''
# Azul, es necesario crear dos mascaras, ya que en el canal A, hay dos rangos donde se encuentra este color:
# azul_bajo1 = np.array([l,a,b])    
azul_bajo1 = np.array([40, 153, 130])
azul_alto1 = np.array([175,162, 140])

azul_bajo2 = np.array([40, 175, 130])
azul_alto2 = np.array([175,185, 140])

# Se crean las máscaras de la forma normal con la función inRange():

mascara_azul1 = cv2.inRange(imglab, azul_bajo1, azul_alto1)
mascara_azul2 = cv2.inRange(imglab, azul_bajo2, azul_alto2)

# Se juntan las máscaras en una sola máscara final que llamaremos ‘mask’. La función cv2.add sólo puede recibir dos argumentos, por lo que habrá que aplicarla varias veces.
mask = cv2.add(mascara_azul1, mascara_azul2)

# cv2.imshow('Finale', mask)
# cv2.imshow('senal', imglab)
# cv2.imshow('mascara azul', mascara_azul1)

# cv2.waitKey( )
# cv2.destroyAllWindows()




# Intento 2   
azul_bajo = np.array([30, 100, 100])
azul_alto = np.array([200,200, 150])
mascara_azul = cv2.inRange(imglab, azul_bajo, azul_alto)     

# cv2.imshow('Imagen', mascara_azul)
# cv2.waitKey( )
# cv2.destroyAllWindows() 
 
    
    
#################################################################################################################

# Se realiza el mismo procedimiento con los recortes de las señales rojas:
colorRojo=['/home/arawe/Dropbox/Grado/VA/P2/7Senales/color/1_peligro.png',
           '/home/arawe/Dropbox/Grado/VA/P2/7Senales/color/23_peligro2.png',
           '/home/arawe/Dropbox/Grado/VA/P2/7Senales/color/23_prohibicion.png',
           '/home/arawe/Dropbox/Grado/VA/P2/7Senales/color/40_prohibicion.png']

 
  
   
# Se definen tres array donde iremos guardando los valores de cada canal de cada señal.
L = np.array([])
A = np.array([])
B = np.array([])

#Iiterar sobre el array que tiene las direcciones de las imagenes
for i in range(len(colorRojo)): 
    
    img=cv2.imread(colorRojo[i]) # Para la imagen i
    
    imglab=cv2.cvtColor(img, cv2.COLOR_RGB2Lab) # Convertir a lab
    
    # Separarlos canales
    l = imglab[:,:,0] 
    a = imglab[:,:,1]
    b = imglab[:,:,2] 
    
    # Convertir la matriz con el canal, a un array
    l = l.reshape(l.shape[0]*l.shape[1])
    a = a.reshape(a.shape[0]*a.shape[1])
    b = b.reshape(b.shape[0]*b.shape[1])
    

    # Guardar los datos, se añaden los valores de cada imagen al final del array.
    L = np.append(L,l)
    A = np.append(A,a)
    B = np.append(B,b)
        
# Creamos un histograma con L, A y B, para ver en que rango se encuentra el color azul.

plt.title('Canal A - Rojo')
plt.hist(A, bins = 256, range=(0,255))
plt.grid(True)
plt.show()
plt.clf()
#Los valores se reparten desde el valor 120 hasta 235 aproximadamente. Habiendo dos picos en 182-195 y 205-215.

plt.title('Canal B - Rojo')
plt.hist(B, bins = 256, range=(0,255))
plt.grid(True)
plt.show()
plt.clf()
# Los valores se reparten desde 20 hasta 130. Habiendo un pico en 19-25, y otro en 35-50

plt.title('Canal L - Rojo')
plt.hist(L, bins = 256, range=(0,255))
plt.grid(True)
plt.show()
plt.clf()
# Se ve como los valores se reparten a lo largo del histograma.

 
 ##################################################################################################################
# Buscar el color azul, para ello se analiza una imagen que tiene diferentes tonos de azul.

'''

azules= cv2.imread('/home/arawe/Dropbox/Grado/VA/P2/7Senales/azules.png')

cv2.imshow("Azules", azules)
cv2.waitKey( )
cv2.destroyAllWindows()

# Se convierte al modelo de color lab
azuleslab= cv2.cvtColor(azules, cv2.COLOR_RGB2Lab)
   

# Se separan los tres canales, y se convierte cada matriz resultante, a un array que será mas facil de manejar
l = azuleslab[:,:,0]
l = l.reshape(l.shape[0]*l.shape[1])  # Da una nueva forma a una matriz sin cambiar sus datos. Se cambia a un array de 1 dimension donde len(l)=ancho x alto.
a = azuleslab[:,:,1]
a = a.reshape(a.shape[0]*a.shape[1])
b = azuleslab[:,:,2]
b = b.reshape(b.shape[0]*b.shape[1])

# Se buscan los maximos y minimos obtenidos. Que podrían ser utilizados para filtrar por color.
min(a) # 117
max(a) #208
min(b) #136
max(b) #207

# En principio el canal l no se filtraría, ya que variará en funcion de la luz de la imagen.
min(l) #48
max(l) #198

# También se podria hacer un histograma con estos valores, y asi saber en que rango se encuentra el azul que se busca para filtrar. 


plt.title('Canal A - Azules')
plt.hist(a, bins = 256, range=(0,255))
plt.grid(True)
plt.show()
plt.clf()

plt.title('Canal B - Azules')
plt.hist(b, bins = 256, range=(0,255))
plt.grid(True)
plt.show()
plt.clf()

plt.title('Canal L - Azules')
plt.hist(l, bins = 256, range=(0,255))
plt.grid(True)
plt.show()
plt.clf()


'''