# -*- coding: utf-8 -*-
"""
Created on Wed May 20 21:05:23 2026

@author: ECyT
"""

#Pruebo distintos anchos, para ver cómo varían cambiando el nperseg

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
import scipy.io as sio #Importo esta librería para los .mat 

#  Leo la señal del Electrocardiograma (ECG)
fs_ecg = 1000 #Definición de la frecuencia de muestreo del ECG.
mat_struct = sio.loadmat('./ECG_TP4.mat') #Lee el archivo binario de Matlab, y devuelve un diccionario en Python. 
ecg_lead = mat_struct['ecg_lead'].flatten() #Accede al diccionario  bsucando ecg_lead. 
#Esto no me funciona sin el flatten, no me andaba, se lo puse a la ia y solo le agregó eso,
#entendí que tiene que ver con que las funciones que vamos a usar esperan un vector plano y no una matriz, flatten hace eso, me lo aplana en un 
#solo array de una sola fila. 

# CALCULO LA PSD DE LA ECG 
f_ecg, Pxx_ecg = sig.welch(ecg_lead, fs_ecg, nperseg=1024)
#Aplico el método de Welch, tiene tres elementos de entrada: ecg_lead es el vector temporal de datos, fs_ecg  la frecuencia de muestreo
#para que pueda ajustar el eje de frecuencias y nperseg=1024: Longitud de cada segmento en cantidad de muestras. 
#Al no especificar el tipo de ventana ni la superposición se aplica por defecto una ventana de Hann y un solapamiento del 50% 
#Devuelve dos salidas, f_ecg es un vector que contiene los componentes de frcuencias, va desde 0 hatsa nyquist (fs/2) 
# y Pxx_ecg es el vector que contiene los valores estimados de la Densidad Espectral de Potencia.
#Si Nperseg es parecido a N, se tiene un solo promedio, si quiero mucha varianza hago segmento grandes.

Pxx_ecg_db = 10 * np.log10(Pxx_ecg) #hago una transformación logarítmica a los valores de la PSD.

#------------------GRAFICO-------------------------------#

plt.figure(figsize=(10, 12))

plt.plot(f_ecg, Pxx_ecg_db, color='orchid')
plt.title('PSD - Electrocardiograma (ECG con Ruido)', fontweight='bold') # fontweight='bold' me lo pone en negrita, porque entre tanto texto me pierdo.
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim([0, 70])
plt.ylim([0, 70])
plt.grid(True, alpha=0.3)
plt.show

# CALCULO LA PSD DE LA ECG 
f_ecg, Pxx_ecg1 = sig.welch(ecg_lead, fs_ecg, nperseg=1400)
Pxx_ecg_db1 = 10 * np.log10(Pxx_ecg1) #hago una transformación logarítmica a los valores de la PSD.
plt.plot(f_ecg, Pxx_ecg_db1, color='mediumvioletred')

# CALCULO LA PSD DE LA ECG 
f_ecg, Pxx_ecg2 = sig.welch(ecg_lead, fs_ecg, nperseg=400)
Pxx_ecg_db2 = 10 * np.log10(Pxx_ecg2) #hago una transformación logarítmica a los valores de la PSD.
plt.plot(f_ecg, Pxx_ecg_db2, color='pink')

# CALCULO LA PSD DE LA ECG 
f_ecg, Pxx_ecg2 = sig.welch(ecg_lead, fs_ecg, nperseg=2400)
Pxx_ecg_db2 = 10 * np.log10(Pxx_ecg2) #hago una transformación logarítmica a los valores de la PSD.
plt.plot(f_ecg, Pxx_ecg_db2, color='black')
plt.legend()


# Calculo del ancho de banda fijando un porcentaje de la potencia 99 o 95

def calcular_mi_bw_99(f, Pxx): #Calculo la suma acumulada de la potencia espectral usando np.cumsum. 
    # El otro 1% es ruido, fijo unumbral de la potencia en 99%
    potencia_acumulada = np.cumsum(Pxx)
    potencia_total = potencia_acumulada[-1]
    
    # Establezco el umbral calculando cuánto es el 99% de esa energía .
    umbral_99 = 0.99 * potencia_total
    
    #Si es pasabajo acumulo el porcentaje que quiero y lo de la derecha es ruido, como en este caso.
    #Si es pasabanada agarro una franja, me paro en el medio y me corro tipo 2.5 para un lado y 2.5 para el otro
    #Busco donde se pasa ese punto ya freno, quiero encontrar un índice 
    # Busco el primer lugar donde la suma acumulada empata o supera mi umbral del 99%.
    indice_99 = np.where(potencia_acumulada >= umbral_99)[0][0]  
    
    #Quiero buscar posicionar a mi vector ahí, para que 
    
    return f_ecg[indice_99]

    #Para que la potencia sea unitaria, debo escalar por std(x), o sea desvío estandar de la sñeal
    #varianza.
    
anchobanda = calcular_mi_bw_99(f_ecg, Pxx_ecg1)
print(f'el ancho de banda es {anchobanda}')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    