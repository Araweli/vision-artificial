#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 18:49:53 2019

@author: araweli

P2 Visión Artificial: Detección y reconocimiento de señales de tráfico.

Se describe el proceso en los comentarios. Asi como en la memoria en pdf.

"""


import cv2
import numpy as np

# Ruta de las imagenes
ruta="/home/arawe/Dropbox/Grado/VA/P2/7Senales/00"


imagenes = [
             ruta + "001.ppm" ,ruta + "023.ppm", ruta + "040.ppm" ,ruta + "115.ppm" 
            ,ruta + "159.ppm" ,ruta + "171.ppm" ,ruta + "177.ppm" ,ruta + "214.ppm" 
            ,ruta + "219.ppm" ,ruta + "235.ppm" ,ruta + "262.ppm" ,ruta + "446.ppm"
           #  # Extras
            # ,ruta + "024.ppm" ,ruta + "075.ppm" ,ruta + "080.ppm" ,ruta + "082.ppm" 
           # # Extra Rojas
           # ,ruta + "140.ppm",ruta + "185.ppm" ,ruta + "197.ppm" 
            ]
# Para cada imagen ejecutamos todo el código
for i in range(len(imagenes)):
    # Leer imagen
    senal= cv2.imread(imagenes[i]) 
    
    # Convertir imagen un modelo de color que aisle el brillo para dar robustez ante él.   
        # LAB. L Lightness. a Green to Magenta. b Blue to Yellow.   -DESCARTADO.
        # lab = cv2.cvtColor(senal, cv2.COLOR_RGB2Lab) 
        
    #HSV: H - Hue, matiz. S - Saturation. V - Value, valor, brillo.
    hsv = cv2.cvtColor(senal, cv2.COLOR_BGR2HSV)
    
    # Se obtiene una matriz de tamaño anchoxalto de la imagen, para cada posicion de la matriz img[i][j] están los tres canales asociados al pixel(i,j).
    
    # # Mostrar los 3 canales
    # cv2.imshow("img original", senal)
    # cv2.imshow("hsv", hsv) 
    # cv2.imshow("canal h", hsv[:,:,0])
    # cv2.imshow("canal s", hsv[:,:,1])
    # cv2.imshow("canal v", hsv[:,:,2])
    # cv2.waitKey( )
    # cv2.destroyAllWindows()   
    
    ''' FILTRAR POR COLOR es el siguiente paso, y obtener por un lado solo el color azul, y por otro lado solo el color rojo, dado que las señales de tráfico que queremos identificar son de estos dos colores. Se realiza un análisis para saber por qué valores filtrar. '''
    
    # Revisar archivo 'HSV_ObtenerRangoColor.py' donde se analizan las imagenes para obtener el rango de color. En un primer momento se intento filtrar con el modelo de color lab, pero no dio resultados satisfactorios, asi que opté por el modelo de color hsv, se filtra el color en el canal h (con pequeños ajustes en s y v) y además es el modelo de color para el que hay más documentación. También esta el archivo para LAB.
    
# Para aplicar detección de color a la imagen, se crea una máscara para cada color que quiera detectar.
    '''    ---------- AZUL ---------- '''
    
    # Los valores en este caso están repartidos en un solo intervalo, por lo que con una sola máscara será suficiente:   

    # azul = np.array([h,s,v])    
    azul_bajo = np.array([100, 140,  70])
    azul_alto = np.array([135, 240, 255])
    
    # Se crea la máscara de la forma normal con la función inRange():
    mascara_azul = cv2.inRange(hsv, azul_bajo, azul_alto)
       
    # #Los pixeles blancos es lo que la máscara ha identificado como color azul.
    # cv2.imshow('senal', senal)
    # cv2.imshow('mascara azul', mascara_azul)
    # cv2.waitKey( )
    # cv2.destroyAllWindows()
    
    ''' ELIMINAR RUIDO una vez detectado el color azul. '''
    
    # GAUSSIANA en lugar de la apertura. Despues filtrar con un umbral del nivel de blanco para eliminar el ruido difuminado. Esto filtra mucho mejor en menos pasos. No es necesario realizar una apertura porque se elimina casi todo el ruido. No se identifican algunas señales pequeñas muy al fondo, a cambio se reducen practicamente en su totalidad las zonas identificadas como señales que eran ruido.
    #gaussiana = cv2.GaussianBlur(imagen, (n, n), σ) # (n,n) tamaño del kernel
    gaussianaAzul = cv2.GaussianBlur(mascara_azul, (7, 7), 3)
    t, umbralAzul = cv2.threshold(gaussianaAzul, 180, 255, cv2.THRESH_BINARY)
    # Realizar una pequeña dilatacion ya que los objetos se hacen mas pequeños con el paso anterior, así se recuperan las señales pequeñas que dejaban de detectarse.
    umbralAzul = cv2.dilate(umbralAzul,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations = 1)
    
    # cv2.imshow("senal", senal)
    # cv2.imshow("dst",umbralAzul)
    # cv2.waitKey( )
    # cv2.destroyAllWindows()
    
        # DESCARTADO- APERTURA para eliminar las pequenas zonas blancas que se detectan. Es adecuada esta operacion porque primero con un cierre, al crecer las regiones negras, se eliminarán las regiones blancas pequeñas, y con la dilatación que viene después se volveran los objetos a su tamaño original.
        
        # Tamaño 2x4 para evitar que los bordes laterales de las señales se pierdan, por ejemplo en la parte baja de los triangulos blancos de los pasos de peatones y en los laterales de las flechas de indicacion.
        # kernelAzul= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,4))   
        # kernelAzul= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))    
        # aperturaAzul = cv2.morphologyEx(mascara_azul, cv2.MORPH_OPEN, kernelAzul)    
        # cierreAzul = cv2.morphologyEx(aperturaAzul, cv2.MORPH_CLOSE, kernelAzulC,iterations=5)  
        # kernelAzulC = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))   
        
        # cv2.imshow('senal', senal)
        # cv2.imshow("opening", aperturaAzul)
        # cv2.imshow("Cierre", cierreAzul)
        # cv2.waitKey( )
        # cv2.destroyAllWindows()
    

    # CIERRE para unificar las zonas detectadas, ya que las señales tienen los huecos de las indicaciones dentro, si estan divididas en varias regiones conexas, se identifican como más de un objeto. Es necesario un kernel más grande para que una las regiones, y varias iteraciones. Por ejemplo las señales cuadradas que indican paso de peatones, se identifican como dos señales si no estan lo suficientemente cerradas, aunque por otro lado, cerrar completamente las más grandes hace que se confundan las formas.
    
    kernelAzulC = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))   
    # Tambien funciona bien con el kernel (3,3)
    cierreUmbralAzul = cv2.morphologyEx(umbralAzul, cv2.MORPH_CLOSE, kernelAzulC,iterations=8)  
    
    # cv2.imshow("Cierre", cierreUmbralAzul )
    # cv2.waitKey( )
    # cv2.destroyAllWindows()
    

    ''' CLASIFICAR LAS SEÑALES. Una vez que las señales de trafico estan filtradas sin ruido. '''

    # Para las señales azules, hay dos posibilidades: cuadrada (indicación) y circular (obligación). Se detectan los contornos. Como la imagen que se tiene es binaria, se utilizará la funcion cv2.findContour:
    
    # Obtener los contornos. contours es una lista con los contornos detectados. Cada contorno se almacena como un vector de puntos.
    # contours, _ = cv2.findContours(cierreAzul, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursUmbral, _ = cv2.findContours(cierreUmbralAzul, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # dibujar los contornos obtenidos, sobre la imagen original
    # cv2.drawContours(senal, contours, -1, (0, 0, 255), 2, cv2.LINE_AA) #Esto asi dibuja todos los contornos,incluso los del ruido que no se haya filtrado
    for c in contoursUmbral:
        areaContorno = cv2.contourArea(c)
        (x, y, w, h) = cv2.boundingRect(c) # Rectángulo mínimo para el conjunto de puntos especificado. (x,y) es donde empieza el rectangulo, w-ancho, h-alto de los lados. 
        areaRectangulo=w*h
        # Si el area del contorno es igual al del cuadrado, podriamos decir que es una señal cuadrada. Si el area del contorno es menor al cuadrado, podriamos decir que el contorno es un circulo.
        if areaContorno < (areaRectangulo*0.8): # Es circular, dibuja circulo de color azul
            centro = (int(x+w/2), int(y+h/2))          
            radio = int((w+h)/4)
            cv2.circle(senal, centro, radio, (255, 0, 0), 2)
        elif areaContorno > (areaRectangulo*0.8): # Es cuadrada, dibuja cuadrado de color verde
            cv2.rectangle(senal, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)
    
    # cv2.imshow("Contornos", senal)
    # cv2.waitKey( )
    # cv2.destroyAllWindows()

    '''
     --------------------------
     ---------- ROJO ---------- 
     --------------------------
     Se crea la máscara para el rojo. Son necesarias dos máscaras, ya que el color rojo se encuentra en dos intervalos del canal H.
      
    ''' 
    
    # No detectan las señales que estan muy oscuras, y las triangulares detecta huecos en las partes rojas, asi que podria hacer un cierre para quitar las zonas negras primero, y luego la apertura para eliminar el ruido (las hojas que se detectan)
    
# Tambien se selecciona el v en dos rangos. V mide rojo intenso o rojo oscuro. Los rojos no salen muy bien, y sale mucho naranja.
     
    rojo_bajo1 = np.array([0, 75, 25])    
    rojo_alto1 = np.array([12, 225, 85]) # Con la V a 60, no se detectan los dos stop al final.
    
    rojo_bajo2 = np.array([0, 75, 120])    
    rojo_alto2 = np.array([12, 225, 255])
     
    rojo_bajo3 = np.array([170, 125, 60])
    rojo_alto3 = np.array([180, 225, 255])
    
    # Se crea la máscara de la forma normal con la función inRange():
    mascara_roja1 = cv2.inRange(hsv, rojo_bajo1, rojo_alto1)
    mascara_roja2 = cv2.inRange(hsv, rojo_bajo2, rojo_alto2)
    mascara_roja3 = cv2.inRange(hsv, rojo_bajo3, rojo_alto3)
        
    # Se juntan las máscaras en una sola máscara final. La función cv2.add sólo puede recibir dos argumentos.
    mascara_roja_i = cv2.add(mascara_roja1, mascara_roja2)
    mascara_roja = cv2.add(mascara_roja_i, mascara_roja3)
      
    # Mostrar el resultado, pixeles blancos es lo que se ha identificado como color rojo.
    
    # cv2.imshow('mascara roja', mascara_roja)
    # cv2.waitKey( )
    # cv2.destroyAllWindows()    
    
    ''' ELIMINAR RUIDO una vez detectado el color rojo. '''
    
    # Las hojas, detectadas como puntitos blancos, son ruido 'sal', por lo que un filtro de mediana irá bien, tambien se prueba uno de  minimos.
    medianaRojo= cv2.medianBlur(mascara_roja, 3)
    # minRojo = cv2.erode(mascara_roja,  np.ones((2, 2),dtype = int));
    
    # cv2.imshow('medianaRojo', medianaRojo)
    # cv2.imshow('minRojo', minRojo)
    # cv2.waitKey( )
    # cv2.destroyAllWindows()
    
        #######Otras opciones
        # Apertura
        # kernelRojo= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))     
        # aperturaRojo = cv2.morphologyEx(mascara_roja, cv2.MORPH_OPEN, kernelRojo)    
        # Mostrar el resultado de la apertura
        # cv2.imshow('mascara roja', medianaRojo)
        # cv2.imshow('Apertura rojo', aperturaRojo)
        # cv2.waitKey( )
        # cv2.destroyAllWindows()
          
        # Gaussiana
        # gaussianaRojo = cv2.GaussianBlur(mascara_roja, (5, 5), 3)   
        # # cv2.imshow('gaussiana Roja', gaussianaRojo)
        # # cv2.waitKey( )
        # # cv2.destroyAllWindows()      
        # t, umbralRojo = cv2.threshold(gaussianaRojo, 150, 255, cv2.THRESH_BINARY)   
        # cv2.imshow('umbralizado', umbralRojo)
        # cv2.waitKey( )
        # cv2.destroyAllWindows()
    
        ##################################################
        # Gaussiana y canny directamnte. 
        
        # Canny: Detección de bordes con Sobel, Supresión de píxeles fuera del borde y Aplicar umbral por histéresis
        # canny = cv2.Canny(imagen, umbral_minimo, umbral_maximo). Si el valor del píxel es mayor que el umbral máximo, el píxel se considera parte del borde. Un píxel se considera que no es borde si su valor es menor que el umbral mínimo, Si está entre el máximo y el mínimo, será parte del borde si está conectado con un píxel que forma ya parte del borde.
    
        # gaussianaRoja = cv2.GaussianBlur(mascara_roja, (7, 7), 0)
        # bordesCanny = cv2.Canny(gaussianaRoja,110,150)
        # contornos, jerarquia = cv2.findContours(bordesCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
         
        # for c in contornos:
        #     (x, y, w, h) = cv2.boundingRect(c)
        #     cv2.rectangle(senal, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)
         
        # Encuentra muchisimos, tendria que Umbralizar para coger solo los mayores, y aun asi detectaria algunos grandes que no fuesen señales.         
        # # cv2.drawContours (bordesCanny, contornos, -1, (0,150,0), 1)
        # cv2.imshow('canny', senal)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        ###################################################
        
               
    # Un problema de esta máscara es que no detecta completamente las zonas rojas, por lo que se hará un cierre para terminar de rellenar estas zonas.
    kernelCierreRojo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        # cierreRojo = cv2.morphologyEx(aperturaRojo, cv2.cv2.MORPH_CLOSE, kernelRojo) #Este kernel es muy pequeño
        # En este caso se harán más iteraciones, para rellenar mejor las señales.
        # cierreRojoMediana = cv2.morphologyEx(minRojo, cv2.cv2.MORPH_CLOSE, kernelCierreRojo, iterations=2)
    cierreRojoMed     = cv2.morphologyEx(medianaRojo, cv2.cv2.MORPH_CLOSE, kernelCierreRojo, iterations=3)
    
    # Se aplica cani para tener solo los bordes
    bordesCanny = cv2.Canny(cierreRojoMed,100,100)
    
    # cv2.imshow('canny', bordesCanny)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    
    ''' Despues de probar con una Gaussiana, filtro de minimos, de mediana y apertura, el mejor resultado es la madiana con cierre y Canny'''
        
            #  Una vez hecho el filtrado, voy a buscar circulos con Hough        
            # circles = cv2.HoughCircles(cierreRojoMin,cv2.HOUGH_GRADIENT,1,20, param1=50,param2=30,minRadius=0,maxRadius=0)        
            # circles = np.uint16(np.around(circles))
            # for i in circles[0,:]:
            #     # draw the outer circle
            #     cv2.circle(cierreRojoMin,(i[0],i[1]),i[2],(0,255,0),2)
            #     # draw the center of the circle
            #     cv2.circle(cierreRojoMin,(i[0],i[1]),2,(0,0,255),3)        
            # cv2.imshow('detected circles',cierreRojoMin)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        
    '''
    Clasificar Roja
    '''                                    
    # Una vez que estan bastante bien segmentadas, se debe clasificar entre prohibicion (stop, ceda, prohibido el paso o limitacion de velocidad) o peligro (triangulares y rojas)    
    contornos, jerarquia = cv2.findContours(bordesCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.RETR_EXTERNAL Para intentar coger el contorno externo (en las señales circulares y triangulares hay dos)
    # cv2.CHAIN_APPROX_SIMPLE guarda sololos puntos necesarios del contorno para representarlo
    # contours	Detected contours. Each contour is stored as a vector of points (e.g. std::vector<std::vector<cv::Point> >).
    for c in contornos:
        areaContorno = cv2.contourArea(c) 
        # Elimino las areas muy pequeñas en relacion al tamaño de la imagen (invariante a escalado)
        # if areaContorno >200: # Se cambia 200 por una relacion entre el area y el tamaño de la imagen, por eso el 0.0184
        if areaContorno > (senal.shape[0]*senal.shape[1]* 0.0184/100):
            (x, y, w, h) = cv2.boundingRect(c)
            if ((0.85*h) < w) & ((1.15*h) > w): # Se eliminan los contornos rectangulares, porque las señales son igual de anchas que de altas
                    # Analizar los vertices de los contornos
                    # print('imagen: ', imagenes[i],'. len(c)', len(c)) 
                    # el numero de vertices cambia mucho de un contorno a otro asi que no sirve para clasificarlas.
                
                # Analizar momento de los contornos: comparar el centroide del contorno, con el centro del minimo circulo que lo envuelve
                momentos = cv2.moments(c)
                cx = int(momentos['m10']/momentos['m00'])
                cy = int(momentos['m01']/momentos['m00'])
                
                #Calculo el menor circulo que enmarca el contorno
                (xminCir,yminCir),radiusminCir = cv2.minEnclosingCircle(c)
                centerminCir = (int(xminCir),int(yminCir))
                radiusminCir = int(radiusminCir)
                # La distancia entre dos puntos (x1, y1) y (x2, y2) es raíz de ((x1 - x2) al cuadrado + (y1 - y2) al cuadrado)
                distancia=(int(xminCir)-cx)**2 + (int(yminCir)-cy)**2   
                # Filtrar por un numero concreto de distancia, lo hara dependiente a escala, asi que se busca una relacion entre la distancia y el radio
                if (radiusminCir == 0 or distancia == 0):   
                     relacion=0
                else: 
                     relacion=radiusminCir/distancia
                # print('distancia', distancia, 'radio', radiusminCir, 'relacion', relacion)
                # Se observa que si la relacion es muy cercana a 0, significa que los centros del contorno y del circulo minimo que lo rodea, estan muy lejos, por lo que es ruido. Se pierde una señal y a cambio se eliminan muchos falsos positivos.
                if ( (relacion>=2.2 and relacion<=28) or relacion==0): 
                # Pintar en amarillo los falsos positivos que se eliminan
                # if (( (relacion<2.2) & (relacion>0)) or relacion>28): 
                # cv2.rectangle(senal, (x, y), (x + w, y + h), (0, 255, 255), 2, cv2.LINE_AA)                    
                    cv2.rectangle(senal, (x, y), (x + w, y + h), (0, 0, 255), 2, cv2.LINE_AA)   
                    

    cv2.imshow(imagenes[i], senal)
    # cv2.imshow('mascara roja', mascara_roja)    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
