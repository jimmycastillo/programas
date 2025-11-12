# correr con: streamlit run hplc_streamlit.py


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.ndimage import gaussian_filter1d
import os
import re
from io import BytesIO

# Ignorar warnings de optimización, ya que se gestionan manualmente
import warnings
warnings.simplefilter('ignore', OptimizeWarning)

# Configuración de la página de Streamlit
st.set_page_config(layout="wide", page_title="Analizador de Cromatogramas")

# -----------------------------
# 1. Funciones de procesamiento
# -----------------------------
def suma_gaussianas(x, *params):
    n = len(params) // 3
    y = np.zeros_like(x)
    for i in range(n):
        A, mu, sigma = params[3*i:3*i+3]
        y += A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return y

def corregir_linea_base(x, y, orden_polinomio=1, n_puntos_extremos=10):
    """
    Corrige la línea base usando un ajuste polinómico solo en los puntos extremos.
    """
    if len(x) < 2 * n_puntos_extremos:
        return y, np.zeros_like(y)
    
    x_fit = np.concatenate((x[:n_puntos_extremos], x[-n_puntos_extremos:]))
    y_fit = np.concatenate((y[:n_puntos_extremos], y[-n_puntos_extremos:]))
    
    z = np.polyfit(x_fit, y_fit, orden_polinomio)
    p = np.poly1d(z)
    
    linea_base = p(x)
    y_corregida = y - linea_base
    
    return y_corregida, linea_base

def cargar_y_suavizar(uploaded_file, sigma):
    df = pd.read_csv(uploaded_file, sep='\t', comment='#', skiprows=4,
                     names=['time_min', 'intensity_mV'])
    y_suav = gaussian_filter1d(df['intensity_mV'], sigma=sigma)
    return df['time_min'].values, y_suav

def ajustar_gaussianas_compuestas(x, y, p0_params):
    try:
        popt, pcov = curve_fit(
            suma_gaussianas, x, y, p0=p0_params,
            maxfev=20000,
            ftol=1e-6, xtol=1e-6
        )
    except RuntimeError as e:
        raise RuntimeError(f"Error en el ajuste de curva: {e}")
    ajuste = suma_gaussianas(x, *popt)
    areas_gauss = [popt[3*i] * abs(popt[3*i+2]) * np.sqrt(2*np.pi)
                   for i in range(len(popt)//3)]
    return ajuste, popt, areas_gauss

def guardar_resultados_pico_buffers(x, y, ajuste, popt, rango, areas_gauss, pico_num):
    t_start, t_end = rango
    
    # 1. Gráfico
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, label="Pico original", color='gray')
    ax.plot(x, ajuste, color='black', label="Ajuste Total")
    for i in range(len(areas_gauss)):
        A, mu, sigma = popt[3*i:3*i+3]
        g = A * np.exp(-(x - mu)**2 / (2 * sigma**2))
        ax.plot(x, g, '--', label=f"Gauss {i+1}")
    ax.set(title=f"Ajuste Pico {pico_num} ({t_start:.2f}-{t_end:.2f})",
           xlabel="Tiempo (min)", ylabel="Intensidad (mV)")
    ax.legend(); ax.grid(True)
    fig.tight_layout()
    
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300)
    plt.close(fig)
    img_buffer.seek(0)

    # 2. Resumen de áreas en CSV
    resumen = {}
    area_orig = np.trapz(y, x)
    resumen[f"Área Original Pico {pico_num} ({t_start:.2f}-{t_end:.2f})"] = round(area_orig, 3)
    resumen[f"Máximo Original Pico {pico_num}"] = round(np.max(y), 3)
    for i, a in enumerate(areas_gauss):
        resumen[f"Área Gaussiana {i+1}"] = round(a, 3)
    
    csv_buffer = BytesIO()
    pd.DataFrame([resumen]).to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    return img_buffer, csv_buffer

def guardar_grafico_original(x, y, rangos, filename):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, y, color='black', label="Datos Corregidos")
    
    for i, (t_start, t_end) in enumerate(rangos):
        mask = (x >= t_start) & (x <= t_end)
        if np.any(mask):
            area = np.trapz(y[mask], x[mask])
            max_y = np.max(y[mask])
            max_x = x[mask][np.argmax(y[mask])]
            
            ax.fill_between(x, 0, y, where=mask, alpha=0.3, label=f"Área Pico {i+1}")
            ax.annotate(
                f'Máx: {max_y:.2f}',
                xy=(max_x, max_y),
                xytext=(max_x, max_y + 0.1 * max_y),
                arrowprops=dict(facecolor='black', shrink=0.05),
                ha='center',
                fontsize=8,
                color='red'
            )
            
    ax.set(title="Picos Originales Sombreados", xlabel="Tiempo (min)", ylabel="Intensidad (mV)")
    ax.legend(); ax.grid(True)
    
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300)
    plt.close(fig)
    img_buffer.seek(0)
    return img_buffer

# --------------------------------
# 2. Interfaz de usuario (Streamlit)
# --------------------------------
st.title("👨‍🔬 Analizador de Cromatogramas HPLC")
st.markdown("---")

uploaded_file = st.file_uploader("Sube tu archivo de datos de cromatografía (.txt)", type="txt")

if uploaded_file:
    # -----------------------------
    # Sección de Análisis Inicial
    # -----------------------------
    st.header("Análisis Inicial")

    col1, col2 = st.columns(2)
    with col1:
        smoothing_sigma = st.slider("Factor de Suavizado (σ)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    with col2:
        baseline_order = st.number_input("Orden del Polinomio de Línea Base", min_value=0, max_value=5, value=2, step=1)
    
    # Cargar y pre-procesar los datos
    try:
        x_suavizado, y_suavizado = cargar_y_suavizar(uploaded_file, smoothing_sigma)
        y_corregida, linea_base = corregir_linea_base(x_suavizado, y_suavizado, baseline_order)

        # Gráfico inicial
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(x_suavizado, y_suavizado, color='gray', label="Datos Originales Suavizados")
        ax.plot(x_suavizado, y_corregida, color='black', label="Datos Corregidos")
        ax.plot(x_suavizado, linea_base, color='red', linestyle='--', label="Línea Base Polinómica")
        ax.set(title=f"Línea Base Corregida de {uploaded_file.name}", xlabel="Tiempo (min)", ylabel="Intensidad (mV)")
        ax.legend(); ax.grid(True)
        st.pyplot(fig)

        st.markdown("---")
        st.header("Análisis de Picos")
        
        # -----------------------------
        # Sección de Definición de Picos
        # -----------------------------
        
        rangos_str = st.text_area(
            "Define los rangos de tiempo para los picos (ej: `(1.2, 2.0),(2.5, 3.2)`)"
        )

        try:
            if rangos_str:
                rangos = [tuple(map(float, re.findall(r"[\d.]+", r))) for r in rangos_str.split('),')]
                
                # Calcular y mostrar áreas originales y sombrear en el gráfico
                st.subheader("Áreas bajo la curva original")
                fig_orig, ax_orig = plt.subplots(figsize=(12, 6))
                ax_orig.plot(x_suavizado, y_corregida, color='black', label="Datos Corregidos")
                
                areas_originales_str = ""
                for i, (t_start, t_end) in enumerate(rangos):
                    mask = (x_suavizado >= t_start) & (x_suavizado <= t_end)
                    if np.any(mask):
                        area = np.trapz(y_corregida[mask], x_suavizado[mask])
                        max_y = np.max(y_corregida[mask])
                        max_x = x_suavizado[mask][np.argmax(y_corregida[mask])]

                        areas_originales_str += f"Pico {i+1} ({t_start:.2f}-{t_end:.2f}): Área={area:.2f}, Máx={max_y:.2f}\n"
                        ax_orig.fill_between(x_suavizado, 0, y_corregida, where=mask, alpha=0.3, label=f"Área Pico {i+1}")
                        ax_orig.annotate(
                            f'Máx: {max_y:.2f}',
                            xy=(max_x, max_y),
                            xytext=(max_x, max_y + 0.1 * max_y),
                            arrowprops=dict(facecolor='black', shrink=0.05),
                            ha='center',
                            fontsize=8,
                            color='red'
                        )
                        
                ax_orig.set(title="Picos Originales Sombreados", xlabel="Tiempo (min)", ylabel="Intensidad (mV)")
                ax_orig.legend(); ax_orig.grid(True)
                st.pyplot(fig_orig)
                
                st.code(areas_originales_str)
                
                # Botón de descarga para el gráfico original
                if rangos:
                    img_buffer_orig = guardar_grafico_original(x_suavizado, y_corregida, rangos, uploaded_file.name)
                    st.download_button(
                        label="Descargar Gráfico de Áreas Originales PNG",
                        data=img_buffer_orig,
                        file_name=f"{os.path.splitext(uploaded_file.name)[0]}_areas_originales.png",
                        mime="image/png"
                    )

                st.markdown("---")
                st.subheader("Ajuste y Optimización de Picos")
                
                # -----------------------------
                # Sección de Picos Individuales (usando expander)
                # -----------------------------
                for i, rango in enumerate(rangos):
                    t_start, t_end = rango
                    
                    with st.expander(f"Pico {i+1} ({t_start:.2f} - {t_end:.2f} min)"):
                        col_params, col_graph = st.columns(2)
                        
                        mask = (x_suavizado >= t_start) & (x_suavizado <= t_end)
                        x_pico = x_suavizado[mask]
                        y_pico = y_corregida[mask]
                        
                        if len(x_pico) < 3:
                            st.warning(f"El rango para el pico {i+1} es muy pequeño para el ajuste.")
                            continue

                        # Inicializar o recuperar el estado de la sesión
                        p0_default = []
                        mu_pico_base = np.mean(rango)
                        idx_base = np.abs(x_pico - mu_pico_base).argmin()
                        A0 = y_pico[idx_base]
                        p0_default.extend([A0, mu_pico_base, 0.1])
                        p0_default.extend([A0, mu_pico_base, 0.1])
                        
                        if f'p0_input_{i}' not in st.session_state:
                            st.session_state[f'p0_input_{i}'] = p0_default

                        # Controles de parámetros en la columna izquierda
                        with col_params:
                            st.markdown("**Parámetros de Ajuste Manual**")
                            num_gauss = len(st.session_state[f'p0_input_{i}']) // 3
                            for j in range(num_gauss):
                                st.markdown(f"**Gaussiana {j+1}**")
                                st.session_state[f'p0_input_{i}'][3*j] = st.number_input(f"Amplitud (A) {j+1}", value=st.session_state[f'p0_input_{i}'][3*j], key=f"A_{i}_{j}")
                                st.session_state[f'p0_input_{i}'][3*j+1] = st.number_input(f"Media (μ) {j+1}", value=st.session_state[f'p0_input_{i}'][3*j+1], key=f"mu_{i}_{j}")
                                st.session_state[f'p0_input_{i}'][3*j+2] = st.number_input(f"Sigma (σ) {j+1}", value=abs(st.session_state[f'p0_input_{i}'][3*j+2]), key=f"sigma_{i}_{j}")

                            st.button("Añadir Gaussiana", key=f"add_gauss_{i}", on_click=lambda: st.session_state.update({f'p0_input_{i}': st.session_state[f'p0_input_{i}'] + [A0, mu_pico_base, 0.1]}))

                            st.markdown("---")
                            
                            # Botones de ajuste y descarga
                            if st.button("Ajustar", key=f"btn_ajustar_{i}"):
                                try:
                                    ajuste, popt, areas_g = ajustar_gaussianas_compuestas(x_pico, y_pico, st.session_state[f'p0_input_{i}'])
                                    st.session_state[f'ajuste_pico_{i}'] = (ajuste, popt, areas_g)
                                    st.success("Ajuste realizado con éxito.")
                                except RuntimeError as e:
                                    st.error(e)
                            
                        # Gráfico en la columna derecha
                        with col_graph:
                            if f'ajuste_pico_{i}' in st.session_state:
                                ajuste, popt, areas_g = st.session_state[f'ajuste_pico_{i}']
                                fig_pico, ax_pico = plt.subplots(figsize=(8, 4))
                                ax_pico.plot(x_pico, y_pico, color='gray', label="Pico original")
                                ax_pico.plot(x_pico, ajuste, color='black', label="Ajuste Total")
                                
                                for j in range(len(areas_g)):
                                    A, mu, sigma = popt[3*j:3*j+3]
                                    g = A * np.exp(-(x_pico - mu)**2 / (2 * sigma**2))
                                    ax_pico.plot(x_pico, g, '--', label=f"Gauss {j+1}")
                                
                                ax_pico.set(title=f"Ajuste Pico {i+1}", xlabel="Tiempo (min)", ylabel="Intensidad (mV)")
                                ax_pico.legend(); ax_pico.grid(True)
                                st.pyplot(fig_pico)

                                # Resultados numéricos y botones de descarga
                                st.markdown("---")
                                st.subheader("Resultados")
                                
                                max_y_pico = np.max(y_pico)
                                st.text(f"Máximo del Pico Original: {max_y_pico:.2f} mV")

                                area_orig_pico = np.trapz(y_pico, x_pico)
                                st.text(f"Área Original: {area_orig_pico:.2f}")

                                for j, a in enumerate(areas_g):
                                    st.text(f"Área Gaussiana {j+1}: {a:.2f}")

                                img_buffer, csv_buffer = guardar_resultados_pico_buffers(x_pico, y_pico, ajuste, popt, rango, areas_g, i + 1)
                                
                                st.download_button(
                                    label="Descargar Gráfico de Ajuste PNG",
                                    data=img_buffer,
                                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_pico_{i+1}_ajuste.png",
                                    mime="image/png"
                                )
                                st.download_button(
                                    label="Descargar Áreas CSV",
                                    data=csv_buffer,
                                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_pico_{i+1}_areas.csv",
                                    mime="text/csv"
                                )
                                
        except (ValueError, IndexError):
            st.error("Formato de rangos de picos incorrecto. Por favor, usa el formato: `(t_inicio, t_fin),(t_inicio, t_fin)`")
            
    except Exception as e:
        st.error(f"Ocurrió un error al procesar el archivo: {e}")