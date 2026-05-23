# -*- coding: utf-8 -*-
"""
Created on Wed May 20 10:20:27 2026

@author: jazbv
"""

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

# Leo la señal Pletismografía (PPG) 
fs_ppg = 400 # Hz
ppg_raw = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1) #Es una función de NumPy diseñada para leer archivos de texto  que contienen 
# datos numéricos, skip_header=1 salta la primera línea que suele ser un título, así no se rompe al intentar leerlo como número.  
#delimiter=',' avisa que el carácter que separa un número del siguiente en el archivo es una coma.


# Leo los archivos de Audio.
fs_audio1, wav_cucaracha = sio.wavfile.read('la cucaracha.wav') 
fs_audio2, wav_prueba    = sio.wavfile.read('prueba psd.wav')
fs_audio3, wav_silbido   = sio.wavfile.read('silbido.wav')
# sio.wavfile.read es una función  para decodificar archivos de audio  .wav , devuelve una tupla, la frecuencia de muestreo y las amplitudes de onda.

#------------------Calculo las PSDs CON EL MÉTODO DE WELCH-------------------------------#

# 1. ECG 
f_ecg, Pxx_ecg = sig.welch(ecg_lead, fs_ecg, nperseg=1024)
#Aplico el método de Welch, tiene tres elementos de entrada: ecg_lead es el vector temporal de datos, fs_ecg  la frecuencia de muestreo
#para que pueda ajustar el eje de frecuencias y nperseg=1024: Longitud de cada segmento en cantidad de muestras. 
#Al no especificar el tipo de ventana ni la superposición se aplica por defecto una ventana de Hann y un solapamiento del 50% 
#Devuelve dos salidas, f_ecg es un vector que contiene los componentes de frcuencias, va desde 0 hatsa nyquist (fs/2) 
# y Pxx_ecg es el vector que contiene los valores estimados de la Densidad Espectral de Potencia.
#Si Nperseg es parecido a N, se tiene un solo promedio, si quiero mucha varianza hago segmento grandes.

Pxx_ecg_db = 10 * np.log10(Pxx_ecg) #hago una transformación logarítmica a los valores de la PSD.

# 2. PPG
f_ppg, Pxx_ppg = sig.welch(ppg_raw, fs_ppg, nperseg=512)
Pxx_ppg_db = 10 * np.log10(Pxx_ppg)

#  3. Audio: La Cucaracha
f_cuca, Pxx_cuca = sig.welch(wav_cucaracha, fs_audio1, nperseg=4096)
Pxx_cuca_db = 10 * np.log10(Pxx_cuca)

# 4. Audio: Prueba PSD
f_prue, Pxx_prue = sig.welch(wav_prueba, fs_audio2, nperseg=4096)
Pxx_prue_db = 10 * np.log10(Pxx_prue)

# 5. Audio: Silbido 
f_silb, Pxx_silb = sig.welch(wav_silbido, fs_audio3, nperseg=4096)
Pxx_silb_db = 10 * np.log10(Pxx_silb)


#------------------GRAFICO-------------------------------#

plt.figure(figsize=(12, 16))

# ECG
plt.subplot(5, 1, 1)
plt.plot(f_ecg, Pxx_ecg_db, color='orchid')
plt.title('PSD - Electrocardiograma (ECG con Ruido)', fontweight='bold') # fontweight='bold' me lo pone en negrita, porque entre tanto texto me pierdo.
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim([0, 150])
plt.grid(True, alpha=0.3)

# PPG
plt.subplot(5, 1, 2)
plt.plot(f_ppg, Pxx_ppg_db, color='lightpink')
plt.title('PSD - Fotopletismografía (PPG Cruda)', fontweight='bold')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim([0, 30])
plt.grid(True, alpha=0.3)

# Audio 1: La Cucaracha
plt.subplot(5, 1, 3)
plt.plot(f_cuca, Pxx_cuca_db, color='purple')
plt.title('PSD - Audio: La Cucaracha', fontweight='bold')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim([0, 6000])
plt.grid(True, alpha=0.3)

# Audio 2: Prueba PSD
plt.subplot(5, 1, 4)
plt.plot(f_prue, Pxx_prue_db, color='mediumvioletred')
plt.title('PSD - Audio: Prueba PSD', fontweight='bold')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim([0, 6000])
plt.grid(True, alpha=0.3)

# Audio 3: Silbido
plt.subplot(5, 1, 5)
plt.plot(f_silb, Pxx_silb_db, color='crimson')
plt.title('PSD - Audio: Silbido', fontweight='bold')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim([0, 6000])
plt.grid(True, alpha=0.3)

plt.tight_layout() #Me quedaban todos los títulos del gráfico superpuestos, Gemini me dijo que le agregue esto que lo que hace es acivar un 
#algoritmo de matplotlib que reajusta automáticamente las posiciones y los espacios entre los gráficoos, y ahí me quedó lindo.
plt.show()



#===========================AHORA REPITO CON EL MÉTODO DE BLACKMAN-TUKEY===============================#


 #Me perdí un montón, porque busqué en spicy .signal cómo usarlo y terminé viendo la implementación en un foro, que usaba un millón  de cosas
    #que no vimos, así que hice medio un **Frankestein** entre eso y lo que me corrigió la IA. 

 #Me perdí un montón, porque busqué en spicy .signal cómo usarlo y terminé viendo la implementación en un foro, que usaba un millón  de cosas
    #que no vimos, así que hice medio un **Frankestein** entre eso y lo que me corrigió la IA. 

def blackman_tukey_psd(x, fs, max_lag): #Defino una función que me calcule la Psd usando Blackman-Tukey, así no tengo que hacer 5 veces lo mismo.
#  x es el vector que contiene las muestras de la señal ( ECG,  PPG , audio).
#  fs es la frecuencia de muestreo de esa señal en particular.
#  max_lag  es el parámetro que representa la cantidad máxima de desplazamientos en muestras, que se va inspeccionar en la autocorrelación.
    
    N = len(x) # Cueno cuántas muestras totales tiene el vector x y guardo ese número entero en N, para después dividir por N. 
    x_detrend = x - np.mean(x) # np.mean(x) Calcula el promedio de todos los valores de la señal y se lo resta a cada una de las muestras de x, 
    # guardando el resultado en  x_detrend, esto sirve para eliminar la componente de continua (DC) y que no me tape la info que quiero ver. 
    
    # AHORA VOY A USAR UN TRUCO PARA CALCULAR LA AUTOCORRELACIÓN USANDO LA FFT Y NO MOVIENDO LA SEÑAL MUESTRA A MUESTRA EN EL TIEMPO.
    
    # Llevamos a frecuencias completando con ceros (zero-padding) para evitar solapamiento circular
    n_fft = 2**int(np.ceil(np.log2(2 * N - 1)))
    X = np.fft.fft(x_detrend, n=n_fft)
    
    # La autocorrelación es la FFT inversa del módulo al cuadrado (Teorema de Wiener-Khinchin)
    r_full = np.fft.ifft(X * np.conj(X)).real / N
    
    # Acomodamos el resultado para tener los lags negativos a la izquierda y positivos a la derecha
    # Esto equivale al orden del 'mode=full' de np.correlate
    r = np.concatenate((r_full[-(N-1):], r_full[:N]))
    centro = len(r) // 2
    # ---------------------------------------------
    
    # Acotamos a los lags de interés (-max_lag a +max_lag)
    r_acotada = r[centro - max_lag : centro + max_lag + 1]
    
    # Ventaneamos la autocorrelación con una ventana de Hann
    w = sig.windows.hann(len(r_acotada))
    r_ventaneada = r_acotada * w
    
    # Transformada de Fourier final sobre la autocorrelación podada
    Pxx = np.abs(np.fft.fft(r_ventaneada))
    f = np.fft.fftfreq(len(r_ventaneada), d=1/fs)
    
    # Nos quedamos solo con las frecuencias positivas (espectro unilateral)
    pos_indices = np.where(f >= 0)
    return f[pos_indices], Pxx[pos_indices]


#Llamo a la funcion para cada señal
# =============================================================================
# LLAMADA A LA FUNCIÓN - CÁLCULO DE LAS 5 SEÑALES 
# =============================================================================
# Usamos la función que creamos arriba pasándole a cada una su vector, su fs y su ventana (max_lag)
f_bt_ecg, Pxx_bt_ecg   = blackman_tukey_psd(ecg_lead, fs_ecg, max_lag=512)
f_bt_ppg, Pxx_bt_ppg   = blackman_tukey_psd(ppg_raw, fs_ppg, max_lag=256)
f_bt_cuca, Pxx_bt_cuca = blackman_tukey_psd(wav_cucaracha, fs_audio1, max_lag=2048)
f_bt_prue, Pxx_bt_prue = blackman_tukey_psd(wav_prueba, fs_audio2, max_lag=2048)
f_bt_silb, Pxx_bt_silb = blackman_tukey_psd(wav_silbido, fs_audio3, max_lag=2048)


# 2. Armamos la figura con los 5 gráficos
plt.figure(figsize=(12, 16))

# ECG
plt.subplot(5, 1, 1)
plt.plot(f_bt_ecg, 10 * np.log10(Pxx_bt_ecg), color='crimson')
plt.title('PSD - Electrocardiograma (Blackman-Tukey)', fontweight='bold')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim([0, 150])
plt.grid(True, alpha=0.3)

# PPG
plt.subplot(5, 1, 2)
plt.plot(f_bt_ppg, 10 * np.log10(Pxx_bt_ppg), color='teal')
plt.title('PSD - Fotopletismografía (Blackman-Tukey)', fontweight='bold')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim([0, 30])
plt.grid(True, alpha=0.3)

# Audio 1: La Cucaracha
plt.subplot(5, 1, 3)
plt.plot(f_bt_cuca, 10 * np.log10(Pxx_bt_cuca), color='purple')
plt.title('PSD - Audio: La Cucaracha (Blackman-Tukey)', fontweight='bold')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim([0, 6000])
plt.grid(True, alpha=0.3)

# Audio 2: Prueba PSD
plt.subplot(5, 1, 4)
plt.plot(f_bt_prue, 10 * np.log10(Pxx_bt_prue), color='darkblue')
plt.title('PSD - Audio: Prueba PSD (Blackman-Tukey)', fontweight='bold')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim([0, 6000])
plt.grid(True, alpha=0.3)

# Audio 3: Silbido
plt.subplot(5, 1, 5)
plt.plot(f_bt_silb, 10 * np.log10(Pxx_bt_silb), color='forestgreen')
plt.title('PSD - Audio: Silbido (Blackman-Tukey)', fontweight='bold')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim([0, 6000])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# =============================================================================
# MÉTODO DEL PERIODOGRAMA MODIFICADO / VENTANEADO
# =============================================================================

def periodograma_ventaneado_psd(x, fs):
    # Defino una función que me calcule la PSD usando el Periodograma Modificado.
    # A diferencia de Welch, acá NO cortamos la señal en pedacitos; se procesa el registro completo de golpe.
    # x es el vector que contiene las muestras de la señal (ECG, PPG, audio).
    # fs es la frecuencia de muestreo de esa señal en particular.
    
    x_detrend = x - np.mean(x) # Volvemos a restar el promedio para eliminar la componente de continua (DC)
    # Esto evita que el pico gigante de 0 Hz se desparrame y nos tape las frecuencias bajas útiles.
    
    # Calculamos el periodograma usando SciPy. Le pasamos la ventana 'hann' (o Hamming) 
    # para suavizar los bordes de la señal en el tiempo y reducir el desparramo espectral (Spectral Leakage).
    # Al no pasarle un tamaño de segmento corto (nperseg), usa el largo total de la señal.
    f, Pxx = sig.periodogram(x_detrend, fs=fs, window='hann')
    
    return f, Pxx


# =============================================================================
# LLAMADA A LA FUNCIÓN - CÁLCULO DE LAS 5 SEÑALES POR PERIODOGRAMA
# =============================================================================
# Llamamos a nuestra función para cada señal. Spyder va a guardar los resultados
# en variables distintas (con el sufijo '_per') para no pisar lo que calculamos con Blackman-Tukey.
f_per_ecg, Pxx_per_ecg   = periodograma_ventaneado_psd(ecg_lead, fs_ecg)
f_per_ppg, Pxx_per_ppg   = periodograma_ventaneado_psd(ppg_raw, fs_ppg)
f_per_cuca, Pxx_per_cuca = periodograma_ventaneado_psd(wav_cucaracha, fs_audio1)
f_per_prue, Pxx_per_prue = periodograma_ventaneado_psd(wav_prueba, fs_audio2)
f_per_silb, Pxx_per_silb = periodograma_ventaneado_psd(wav_silbido, fs_audio3)


# =============================================================================
# GRÁFICOS DEL PERIODOGRAMA
# =============================================================================
# Armamos una nueva figura independiente con los 5 gráficos del Periodograma
plt.figure(figsize=(12, 16))

# ECG
plt.subplot(5, 1, 1)
plt.plot(f_per_ecg, 10 * np.log10(Pxx_per_ecg), color='crimson')
plt.title('PSD - Electrocardiograma (Periodograma Ventaneado)', fontweight='bold')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim([0, 150])
plt.grid(True, alpha=0.3)

# PPG
plt.subplot(5, 1, 2)
plt.plot(f_per_ppg, 10 * np.log10(Pxx_per_ppg), color='teal')
plt.title('PSD - Fotopletismografía (Periodograma Ventaneado)', fontweight='bold')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim([0, 30])
plt.grid(True, alpha=0.3)

# Audio 1: La Cucaracha
plt.subplot(5, 1, 3)
plt.plot(f_per_cuca, 10 * np.log10(Pxx_per_cuca), color='purple')
plt.title('PSD - Audio: La Cucaracha (Periodograma Ventaneado)', fontweight='bold')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim([0, 6000])
plt.grid(True, alpha=0.3)

# Audio 2: Prueba PSD
plt.subplot(5, 1, 4)
plt.plot(f_per_prue, 10 * np.log10(Pxx_per_prue), color='darkblue')
plt.title('PSD - Audio: Prueba PSD (Periodograma Ventaneado)', fontweight='bold')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [dB/Hz]')
plt.xlim([0, 6000])
plt.grid(True, alpha=0.3)

# =============================================================================
# 3) MI FUNCIÓN EN PRIMERA PERSONA PARA EL ANCHO DE BANDA (CRITERIO DEL 99%)
# =============================================================================

def calcular_mi_bw_99(f, Pxx):
    # Calculo la suma acumulada de la potencia espectral usando np.cumsum. 
    # Lo hago en escala lineal (no en dB) porque necesito ir sumando la energía de verdad paso a paso.
    potencia_acumulada = np.cumsum(Pxx)
    
    # El último casillero de mi suma acumulada representa el 100% de la potencia total de la señal.
    potencia_total = potencia_acumulada[-1]
    
    # Establezco mi línea de llegada (el umbral) calculando cuánto es el 99% de esa energía total.
    umbral_99 = 0.99 * potencia_total
    
    # Busco el primer casillero en el vector donde la suma acumulada empata o supera mi umbral del 99%.
    indice_99 = np.where(potencia_acumulada >= umbral_99)[0][0]
    
    # Uso ese índice para ubicar en mi vector de frecuencias 'f' cuál es el valor exacto en Hz donde se plantó el 99%.
    return f[indice_99]


# =============================================================================
# 4) EJECUCIÓN: LLAMO A MIS FUNCIONES PARA PROCESAR LAS 5 SEÑALES
# =============================================================================

# --- Ejecuto el Método de Blackman-Tukey ---
# Adapto el max_lag a ojo según la dinámica de cada señal (menos muestras en biomédicas, más en audio).
f_bt_ecg, Pxx_bt_ecg   = blackman_tukey_psd(ecg_lead, fs_ecg, max_lag=512)
f_bt_ppg, Pxx_bt_ppg   = blackman_tukey_psd(ppg_raw, fs_ppg, max_lag=256)
f_bt_cuca, Pxx_bt_cuca = blackman_tukey_psd(wav_cucaracha, fs_audio1, max_lag=2048)
f_bt_prue, Pxx_bt_prue = blackman_tukey_psd(wav_prueba, fs_audio2, max_lag=2048)
f_bt_silb, Pxx_bt_silb = blackman_tukey_psd(wav_silbido, fs_audio3, max_lag=2048)

# --- Ejecuto el Método del Periodograma Ventaneado ---
# Guardo los resultados en variables con el sufijo '_per' para no pisar lo anterior.
f_per_ecg, Pxx_per_ecg   = periodograma_ventaneado_psd(ecg_lead, fs_ecg)
f_per_ppg, Pxx_per_ppg   = periodograma_ventaneado_psd(ppg_raw, fs_ppg)
f_per_cuca, Pxx_per_cuca = periodograma_ventaneado_psd(wav_cucaracha, fs_audio1)
f_per_prue, Pxx_per_prue = periodograma_ventaneado_psd(wav_prueba, fs_audio2)
f_per_silb, Pxx_per_silb = periodograma_ventaneado_psd(wav_silbido, fs_audio3)


# =============================================================================
# 5) EXTRACCIÓN AUTOMÁTICA DE TODOS LOS ANCHOS DE BANDA
# =============================================================================

# Primero extraigo los anchos de banda de Welch (usando las variables que ya tenías de la primera parte)
bw_welch_ecg  = calcular_mi_bw_99(f_ecg, Pxx_ecg)
bw_welch_ppg  = calcular_mi_bw_99(f_ppg, Pxx_ppg)
bw_welch_cuca = calcular_mi_bw_99(f_cuca, Pxx_cuca)
bw_welch_prue = calcular_mi_bw_99(f_prue, Pxx_prue)
bw_welch_silb = calcular_mi_bw_99(f_silb, Pxx_silb)

# Ahora extraigo los anchos de banda para el método de Blackman-Tukey
bw_bt_ecg  = calcular_mi_bw_99(f_bt_ecg, Pxx_bt_ecg)
bw_bt_ppg  = calcular_mi_bw_99(f_bt_ppg, Pxx_bt_ppg)
bw_bt_cuca = calcular_mi_bw_99(f_bt_cuca, Pxx_bt_cuca)
bw_bt_prue = calcular_mi_bw_99(f_bt_prue, Pxx_bt_prue)
bw_bt_silb = calcular_mi_bw_99(f_bt_silb, Pxx_bt_silb)

# Y finalmente extraigo los del Periodograma Ventaneado
bw_per_ecg  = calcular_mi_bw_99(f_per_ecg, Pxx_per_ecg)
bw_per_ppg  = calcular_mi_bw_99(f_per_ppg, Pxx_per_ppg)
bw_per_cuca = calcular_mi_bw_99(f_per_cuca, Pxx_per_cuca)
bw_per_prue = calcular_mi_bw_99(f_per_prue, Pxx_per_prue)
bw_per_silb = calcular_mi_bw_99(f_per_silb, Pxx_per_silb)


# =============================================================================
# 6) MI BLOQUE PARA IMPRIMIR LA TABLA COMPARATIVA GIGANTE EN CONSOLA
# =============================================================================

# Imprimo el encabezado acomodando los anchos de las columnas con espacios para que quede estético.
print(f"\n{'Señal Analizada':<26} | {'BW Welch (99%)':<16} | {'BW Blackman-T. (99%)':<22} | {'BW Periodog. Vent (99%)':<24}")
print("-" * 101) # Meto una línea separadora.

# Escupo los resultados numéricos formateando todo a dos decimales (.2f) para que sea un informe de ingeniería serio.
print(f"{'Electrocardiograma (ECG)':<26} | {bw_welch_ecg:<12.2f} Hz | {bw_bt_ecg:<19.2f} Hz | {bw_per_ecg:<21.2f} Hz")
print(f"{'Fotopletismografía (PPG)':<26} | {bw_welch_ppg:<12.2f} Hz | {bw_bt_ppg:<19.2f} Hz | {bw_per_ppg:<21.2f} Hz")
print(f"{'Audio: La Cucaracha':<26} | {bw_welch_cuca:<12.2f} Hz | {bw_bt_cuca:<19.2f} Hz | {bw_per_cuca:<21.2f} Hz")
print(f"{'Audio: Prueba PSD':<26} | {bw_welch_prue:<12.2f} Hz | {bw_bt_prue:<19.2f} Hz | {bw_per_prue:<21.2f} Hz")
print(f"{'Audio: Silbido':<26} | {bw_welch_silb:<12.2f} Hz | {bw_bt_silb:<19.2f} Hz | {bw_per_silb:<21.2f} Hz")