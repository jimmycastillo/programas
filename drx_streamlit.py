# correr con: streamlit run drx_streamlit.py


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
import os
import io

# --- Funciones de procesamiento de datos ---
def read_xrd_data(file_object):
    """
    Lee datos de un espectro XRD desde un objeto de archivo cargado.
    """
    try:
        file_object.seek(0)
        df = pd.read_csv(
            file_object,
            sep='\t',
            skiprows=2,
            header=None,
            names=['2-theta', 'Intensidad'],
            on_bad_lines='skip'
        )
        if not df.dropna().empty:
            return df['2-theta'].to_numpy(), df['Intensidad'].to_numpy()
    except Exception:
        pass

    try:
        file_object.seek(0)
        df = pd.read_csv(
            file_object,
            sep='\s+',
            header=None,
            names=['2-theta', 'Intensidad'],
            on_bad_lines='skip'
        )
        if not df.dropna().empty:
            return df['2-theta'].to_numpy(), df['Intensidad'].to_numpy()
    except Exception as e:
        st.warning(f"Error al leer el archivo: {e}")

    return None, None

def correct_baseline(intensidad):
    """
    Realiza una corrección simple de la línea base restando el valor mínimo.
    """
    min_intensity = np.min(intensidad)
    return intensidad - min_intensity

def smooth_curve(intensidad, window_length, polyorder):
    """Suaviza una curva de datos usando el filtro Savitzky-Golay."""
    if len(intensidad) < window_length:
        st.warning(f"Advertencia: La longitud de los datos ({len(intensidad)}) es menor que la longitud de la ventana ({window_length}). No se pudo suavizar.")
        return intensidad
    
    return savgol_filter(intensidad, window_length, polyorder)

def find_prominent_peaks(theta_2, intensidad_suavizada, prominence):
    """Encuentra los picos más prominentes en el espectro."""
    peaks, _ = find_peaks(intensidad_suavizada, prominence=prominence)
    
    if len(peaks) == 0:
        return np.array([]), np.array([])
    
    peak_theta = theta_2[peaks]
    peak_intensity = intensidad_suavizada[peaks]
    
    return peak_theta, peak_intensity

def plot_xrd_spectra(files_data, offset, prominence, mode):
    """
    Grafica uno o múltiples espectros XRD.
    
    Args:
        files_data (list): Lista de tuplas (objeto de archivo, nombre de archivo).
        offset (float): Desplazamiento vertical para apilar.
        prominence (int): Prominencia de los picos.
        mode (str): 'Apilado' o 'Superpuesto'.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    peak_info = {}

    if mode == 'Apilado':
        reversed_files_data = files_data[::-1]
        for i, (file_object, file_name) in enumerate(reversed_files_data):
            theta_2, intensidad = read_xrd_data(file_object)
            if theta_2 is not None and intensidad is not None:
                intensidad_corregida = correct_baseline(intensidad)
                smoothed_intensity = smooth_curve(intensidad_corregida, 11, 3)
                
                peak_theta, peak_intensity = find_prominent_peaks(theta_2, smoothed_intensity, prominence=prominence)
                
                normalized_intensity = smoothed_intensity / np.max(smoothed_intensity)
                ax.plot(theta_2, normalized_intensity + i * offset, label=file_name)
                
                peak_info[file_name] = {'2-theta': peak_theta, 'Intensidad': peak_intensity}
                
                normalized_peaks = peak_intensity / np.max(smoothed_intensity)
                ax.plot(peak_theta, normalized_peaks + i * offset, 'ro', markersize=6)
                
    elif mode == 'Superpuesto':
        for file_object, file_name in files_data:
            theta_2, intensidad = read_xrd_data(file_object)
            if theta_2 is not None and intensidad is not None:
                intensidad_corregida = correct_baseline(intensidad)
                smoothed_intensity = smooth_curve(intensidad_corregida, 11, 3)
                
                peak_theta, peak_intensity = find_prominent_peaks(theta_2, smoothed_intensity, prominence=prominence)
                
                normalized_intensity = smoothed_intensity / np.max(smoothed_intensity)
                ax.plot(theta_2, normalized_intensity, label=file_name)
                
                peak_info[file_name] = {'2-theta': peak_theta, 'Intensidad': peak_intensity}
                
                normalized_peaks = peak_intensity / np.max(smoothed_intensity)
                ax.plot(peak_theta, normalized_peaks, 'ro', markersize=6)

    ax.set_xlabel('Ángulo 2θ (°)')
    ax.set_ylabel('Intensidad (Normalizada)')
    ax.set_title(f'Comparación de Espectros XRD ({mode})')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(10, 70)
    ax.legend()
    
    return fig, peak_info

# --- Lógica principal de la aplicación Streamlit ---
st.set_page_config(page_title="Análisis de XRD", layout="wide")
st.title('Aplicación Interactiva para Difracción de Rayos-X (XRD)')

st.sidebar.header('Parámetros de Gráfico')
mode = st.sidebar.radio('Modo de Visualización', ('Apilado', 'Superpuesto'))

if mode == 'Apilado':
    offset_value = st.sidebar.slider('Desplazamiento Vertical', min_value=0.0, max_value=2.0, value=1.2, step=0.1)
    prominence_value = st.sidebar.number_input('Prominencia del Pico', min_value=1, value=20, step=1)
    
elif mode == 'Superpuesto':
    offset_value = 0 # No se usa en este modo
    prominence_value = st.sidebar.number_input('Prominencia del Pico', min_value=1, value=20, step=1)

uploaded_files = st.file_uploader(
    "Sube uno o más archivos de datos XRD (.txt o .xy)",
    type=['txt', 'xy'],
    accept_multiple_files=True
)

if uploaded_files:
    files_data_list = [(io.BytesIO(file.getvalue()), file.name) for file in uploaded_files]
    
    # Generar y mostrar el gráfico
    fig, peak_data = plot_xrd_spectra(files_data_list, offset_value, prominence_value, mode)
    st.pyplot(fig)
    
    # Botón para descargar el gráfico
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    st.download_button(
        label="Descargar gráfico",
        data=buf.getvalue(),
        file_name="espectros_xrd.png",
        mime="image/png"
    )

    # Mostrar la tabla de picos
    st.subheader('Picos Detectados')
    for file_name, data in peak_data.items():
        if len(data['2-theta']) > 0:
            st.write(f"**Archivo: {file_name}**")
            df_peaks = pd.DataFrame({
                'Ángulo 2θ (°)' : data['2-theta'],
                'Intensidad': data['Intensidad']
            })
            st.dataframe(df_peaks, hide_index=True)