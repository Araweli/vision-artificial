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

# Se hace un recorte de las señales, para intentar obtener así los valores de azul y rojo que se necesita filtrar.

#Señales azules
# Se leen imagenes de señales recortadas, para ver su histograma y ver en que rango esta el azul y el rojo
colorAzul=['/home/arawe/Dropbox/Grado/VA/P2/7Senales/color/1_indicacion.png',
           '/home/arawe/Dropbox/Grado/VA/P2/7Senales/color/1_obligacion2.png',
           '/home/arawe/Dropbox/Grado/VA/P2/7Senales/color/40_obligacion.png']

# Se definen tres array donde iremos guardando los valores de cada canal de cada señal.
H = np.array([])
S = np.array([])
V = np.array([])

#Iiterar sobre el array que tiene las direcciones de las imagenes
for i in range(len(colorAzul)): 
    
    img=cv2.imread(colorAzul[i]) # Para la imagen i
    
    #  FALLO SOLUCIONADO: utilizar RGB2HSV en lugar de BGR2HSV 
    imghsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convertir a HSV
    
    # Separarlos canales
    h = imghsv[:,:,0] #De la 3ª dimension, cojo la primera columa. [ancho, alto, canales]
    s = imghsv[:,:,1]
    v = imghsv[:,:,2] 
    
    # Convertir la matriz con el canal, a un array
    # Se obtuvo una matriz del mismo ancho y alto que la imagen leida, que en lugar de tener tres canales (800,1360,3), ahora solo tiene uno (800,1360). Asi que remodelamos esta matriz para convertirla a un array que contenga todos los datos.

    h = h.reshape(h.shape[0]*h.shape[1])
    s = s.reshape(s.shape[0]*s.shape[1])
    v = v.reshape(v.shape[0]*v.shape[1])
    
    # Se añaden los valores de cada imagen a los array H, S y V. Con los datos de todas las imagenes en un mismo array, se hace un histograma para ver en que valores caen los datos, y asi saber como filtrar los colores.
    H = np.append(H,h)
    S = np.append(S,s)
    V = np.append(V,v)
        
# Se crea un histograma con H, S y V, para ver en que rango se encuentra el color azul de las señales recortadas.

plt.title('Canal H - Azul')
# plt.hist(H, bins = 256, range=(0,255)) # Quitar el flag range para ver solo el rango donde hay valores.
plt.hist(H, bins = 256)
plt.grid(True)
plt.show()
plt.clf()
# Los valores se concentran cerca del cero, probablemente por las zonas blancas en las imágenes recortadas, y entre 100-135

plt.title('Canal S - Azul')
plt.hist(S, bins =256, range=(0,255))
# plt.hist(S, bins = 256)
plt.grid(True)
plt.show()
plt.clf()
# Los valores se concentran en: 40 y 240

plt.title('Canal V - Azul')
# plt.hist(V, bins =256, range=(0,255))
plt.hist(V, bins = 256)
plt.grid(True)
plt.show()
plt.clf()
# Se ve como los valores se reparten entre 70 y 255.


''' ------------------------------------------------------------------ '''
''' ----------------------------ROJO---------------------------------- '''
''' ------------------------------------------------------------------ '''

# Se realiza el mismo procedimiento con los recortes de las señales rojas:

rojo=cv2.imread('/home/arawe/Dropbox/Grado/VA/P2/7Senales/color/RecprtesRojos.png',)
rojohsv=cv2.cvtColor(rojo, cv2.COLOR_BGR2HSV)

h = rojohsv[:,:,0]
h = h.reshape(h.shape[0]*h.shape[1])
s = rojohsv[:,:,1]
s = s.reshape(s.shape[0]*s.shape[1])
v = rojohsv[:,:,2] 
v = v.reshape(v.shape[0]*v.shape[1])   

# Histogramas
plt.title('Canal H - Rojo')
# plt.hist(H, bins = 256, range=(0,255)) # Quitar el flag range para ver solo el rango donde hay valores.
plt.hist(h, bins = 256)
plt.grid(True)
plt.show()
plt.clf()

plt.title('Canal S - Rojo')
# plt.hist(S, bins =256, range=(0,255))
plt.hist(s, bins = 256)
plt.grid(True)
plt.show()
plt.clf()
# Los valores se concentran en: 150-255

plt.title('Canal V - Rojo')
# plt.hist(V, bins =256, range=(0,255))
plt.hist(v, bins = 256)
plt.grid(True)
plt.show()
plt.clf()

    # rojo_bajo1 = np.array([0, 150, 60])
    # rojo_alto1 = np.array([12, 200, 255])
    
    # rojo_bajo2 = np.array([170, 250, 180])
    # rojo_alto2 = np.array([180, 255, 255])
    
    # rojo_bajo1 = np.array([0, 140, 60])
    # rojo_alto1 = np.array([12, 200, 255])
    
    # rojo_bajo2 = np.array([170, 250, 60])
    # rojo_alto2 = np.array([180, 255, 255])  
    
# NO - esta coge el naranja
    # rojo_bajo1 = np.array([0, 140, 60])    
    # rojo_alto1 = np.array([20, 200, 255])
    
    # rojo_bajo2 = np.array([240, 250, 60])
    # rojo_alto2 = np.array([255, 255, 255])  
# 

    # rojo_bajo4 = np.array([170, 125, 25]) #bajando aqui el 70 obtiene los oscuros
    # rojo_alto4 = np.array([180, 255, 48]) # Si subo esto aqui salen muchismos claritos
 

# Recortes de zonas con hojas anaranjadas, luces de semaforo, coches, etc.
rojo=cv2.imread('/home/arawe/Dropbox/Grado/VA/P2/7Senales/color/RojoNo.png',)
rojohsv=cv2.cvtColor(rojo, cv2.COLOR_BGR2HSV)

h = rojohsv[:,:,0]
h = h.reshape(h.shape[0]*h.shape[1])
s = rojohsv[:,:,1]
s = s.reshape(s.shape[0]*s.shape[1])
v = rojohsv[:,:,2] 
v = v.reshape(v.shape[0]*v.shape[1])   

# Histogramas
plt.title('Canal H - RojoNo')
# plt.hist(H, bins = 256, range=(0,255)) # Quitar el flag range para ver solo el rango donde hay valores.
plt.hist(h, bins = 256)
plt.grid(True)
plt.show()
plt.clf()

plt.title('Canal S - RojoNo')
# plt.hist(S, bins =256, range=(0,255))
plt.hist(s, bins = 256)
plt.grid(True)
plt.show()
plt.clf()
# Los valores se concentran en: 150-255

plt.title('Canal V - RojoNo')
# plt.hist(V, bins =256, range=(0,255))
plt.hist(v, bins = 256)
plt.grid(True)
plt.show()
plt.clf()

















'''
# Se definen tres array donde iremos guardando los valores de cada canal de cada señal.
H = np.array([])
S = np.array([])
V = np.array([])

#Iiterar sobre el array que tiene las direcciones de las imagenes
for i in range(len(colorRojo)): 
    
    img=cv2.imread(colorRojo[i]) # Para la imagen i
    imghsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convertir a HSV
    
    # Separarlos canales
    h = imghsv[:,:,0] #De la 3ª dimension, cojo la primera columa. [ancho, alto, canales]
    s = imghsv[:,:,1]
    v = imghsv[:,:,2] 
    
    # Convertir la matriz con el canal, a un array
    h = h.reshape(h.shape[0]*h.shape[1])
    s = s.reshape(s.shape[0]*s.shape[1])
    v = v.reshape(v.shape[0]*v.shape[1])
    
    # Se añaden los valores de cada imagen a los array H, S y V. 
    H = np.append(H,h)
    S = np.append(S,s)
    V = np.append(V,v)


# Histogramas
plt.title('Canal H - Rojo')
# plt.hist(H, bins = 256, range=(0,255)) # Quitar el flag range para ver solo el rango donde hay valores.
plt.hist(H, bins = 256)
plt.grid(True)
plt.show()
plt.clf()
# Hay dos intervalos para el rojo, 0-20 y 150-180

plt.title('Canal S - Rojo')
# plt.hist(S, bins =256, range=(0,255))
plt.hist(S, bins = 256)
plt.grid(True)
plt.show()
plt.clf()
# Los valores se concentran en: 150-255

plt.title('Canal V - Rojo')
# plt.hist(V, bins =256, range=(0,255))
plt.hist(V, bins = 256)
plt.grid(True)
plt.show()
plt.clf()
# Se ve como los valores se reparten entre 175 y 255.


# Como hay dos intervalos para el canal H, se crean dos máscaras:
# rojo_bajo1 = np.array([0, 150, 175])
# rojo_alto1 = np.array([20, 255, 255])

# rojo_bajo2 = np.array([150, 150, 175])
# rojo_alto2 = np.array([180, 255, 255])


# #Bajar el V porque no deja pasar los rojos oscuros
# rojo_bajo1 = np.array([0, 100, 175])
# rojo_alto1 = np.array([20, 255, 255])

# rojo_bajo2 = np.array([150, 100, 175])
# rojo_alto2 = np.array([180, 255, 255])

# #Bajar el S para los rojos oscuros
# rojo_bajo1 = np.array([0, 150, 150])
# rojo_alto1 = np.array([20, 255, 255])

# rojo_bajo2 = np.array([150, 150, 150])
# rojo_alto2 = np.array([180, 255, 255])

# Bajar S y V
rojo_bajo1 = np.array([0, 75, 75])
rojo_alto1 = np.array([20, 255, 255])

rojo_bajo2 = np.array([150, 75, 75])
rojo_alto2 = np.array([180, 255, 255])

for i in range(len(colorRojo)): 
    
    img=cv2.imread(colorRojo[i]) # Para la imagen i 
    imghsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convertir a HSV

    # Se crea la máscara de la forma normal con la función inRange():
 
    mascara_roja1 = cv2.inRange(imghsv, rojo_bajo1, rojo_alto1)
    mascara_roja2 = cv2.inRange(imghsv, rojo_bajo2, rojo_alto2)
    
    # Se juntan las máscaras en una sola máscara final que llamaremos mascara_roja. La función cv2.add sólo puede recibir dos argumentos, por lo que habrá que aplicarla varias veces.
    mascara_roja = cv2.add(mascara_roja1, mascara_roja2)
    
    
    cv2.imshow('senal', img)
    cv2.imshow('mascara roja', mascara_roja)
    cv2.waitKey( )
    cv2.destroyAllWindows()
 
'''    
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