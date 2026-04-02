# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:06:22 2026

@author: jazmin muñoz 
"""
#%%
import numpy as np
import matplotlib.pyplot as plt  


    

nn = np.arange(0, 8) #linspace : muchos tiempos entre medio xdistantes 
    #arange_: los naturales entre 0 y 7 
xx = 4 +  3 * np.sin((np.pi)/2 * nn ) #funcion senoidal original , base 

# plt.stem(nn,xx) #para ver los puntos del grafico y no una continua 
# plt.show()


    #%% HAGO AHORA LA FFT 
DFT = np.fft.fft(xx)     
magnitud = np.abs(DFT) 
fase = np.angle(DFT)
plt.figure()
plt.stem(nn,magnitud) #para ver los puntos del grafico y no una continua 
plt.title("Modulo de la FFT ")
plt.grid(True)
plt.show()
plt.figure()
plt.stem(nn,fase) #para ver los puntos del grafico y no una continua 
plt.title("Fase de la FFT ")
plt.grid(True)
plt.show()

espectpot = (1/(8)**2) * (magnitud)**2 #8 es la cantidad de muestras

plt.figure()
plt.stem(espectpot)
plt.title("Espectro de potencia ")
plt.grid(True)
plt.show() #muestra el grafico en pantalla

#%%Ahora probamos con 3/2Pi

nn = np.arange(0, 8) #linspace : muchos tiempos entre medio xdistantes 
    #arange_: los naturales entre 0 y 7 
xx = 4 +  3 * np.sin((3*np.pi)/2 * nn ) #funcion senoidal original , base 
DFT = np.fft.fft(xx)     
magnitud = np.abs(DFT) 
fase = np.angle(DFT)
plt.figure()
plt.stem(nn,magnitud) #para ver los puntos del grafico y no una continua 
plt.title("Modulo de la FFT ")
plt.grid(True)
plt.show()
plt.figure()
plt.stem(nn,fase) #para ver los puntos del grafico y no una continua 
plt.title("Fase de la FFT con 3/2pi ")
plt.grid(True)
plt.show()

espectpot = (1/(8)**2) * (magnitud)**2

plt.figure()
plt.stem(espectpot)
plt.title("Espectro de potencia con 3/2pi")
plt.grid(True)
plt.show() #muestra el grafico en pantalla

#%%Ahora probamos con 1/3Pi

nn = np.arange(0, 8) #linspace : muchos tiempos entre medio xdistantes 
    #arange_: los naturales entre 0 y 7 
xx = 4 +  3 * np.sin((np.pi)/3 * nn ) #funcion senoidal original , base 
DFT = np.fft.fft(xx)     
magnitud = np.abs(DFT) 
fase = np.angle(DFT)
plt.figure()
plt.stem(nn,magnitud) #para ver los puntos del grafico y no una continua 
plt.title("Modulo de la FFT ")
plt.grid(True)
plt.show()
plt.figure()
plt.stem(nn,fase) #para ver los puntos del grafico y no una continua 
plt.title("Fase de la FFT con 1/3pi ")
plt.grid(True)
plt.show()

espectpot = (1/(8)**2) * (magnitud)**2

plt.figure()
plt.stem(espectpot)
plt.title("Espectro de potencia con 1/3pi")
plt.grid(True)
plt.show() #muestra el grafico en pantalla