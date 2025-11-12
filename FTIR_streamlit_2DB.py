import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
from scipy.signal import savgol_filter, find_peaks

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Visualizador y Pre-procesador de Espectros FTIR Avanzado",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Funciones de Pre-procesamiento Espectral ---

def corregir_linea_base_polinomial(x_data, y_data, degree):
    """
    Aplica una corrección de línea base mediante ajuste y sustracción polinomial.
    
    Ajusta un polinomio de grado 'degree' a todos los datos y lo resta.
    """
    if not isinstance(x_data, np.ndarray):
        x_data = np.array(x_data)
    if not isinstance(y_data, np.ndarray):
        y_data = np.array(y_data)
    
    # Ajuste Polinomial (Polynomial Fitting)
    coeffs = np.polyfit(x_data, y_data, degree)
    
    # Generar la línea base (Base line)
    baseline = np.polyval(coeffs, x_data)
    
    # Sustracción
    y_corrected = y_data - baseline
    
    # Para que la línea base corregida quede en cero, sumamos el valor mínimo 
    # de la línea base corregida (si los datos están en Absorbancia).
    y_corrected = y_corrected - np.min(y_corrected)
    
    return y_corrected

def normalizar_a_100_transm(y_data):
    """
    Normaliza la señal para que el valor más alto (máximo de Transmitancia) sea 100.
    """
    y_array = np.array(y_data)
    
    if y_array.max() != 0:
        y_normalized = (y_array / y_array.max()) * 100
        return y_normalized
    return y_data

# --- Función para Correlación 2D (2D-COS) ---

def calcular_correlacion_2d(espectros_matrix, x_data):
    """
    Calcula las matrices de Correlación Síncrona y Asíncrona (2D-COS)
    
    Args:
        espectros_matrix (np.ndarray): Matriz N x M (N = número de espectros, M = puntos de onda).
        x_data (np.ndarray): Puntos del eje X.
        
    Returns:
        tuple: (matriz_sincrona, matriz_asincrona, rango_onda)
    """
    N_espectros = espectros_matrix.shape[0]
    
    # 1. Normalización del Factor de Perturbación (Centrado de los espectros)
    espectros_centrados = espectros_matrix - np.mean(espectros_matrix, axis=0)

    # 2. Cálculo de la Matriz Síncrona (Synchronous)
    # G = (1 / (N-1)) * (Y^T * Y)
    matriz_sincrona = np.dot(espectros_centrados.T, espectros_centrados) / (N_espectros - 1)
    
    # 3. Cálculo de la Matriz Asíncrona (Asynchronous)
    # Aproximación del operador de Noda (A)
    A = np.zeros((N_espectros, N_espectros))
    for i in range(N_espectros):
        for j in range(N_espectros):
            if i != j:
                # La simplificación del operador de Noda
                A[i, j] = 1.0 / (np.pi * (j - i)) 
    
    # W = (1 / (N-1)) * (Y^T * A * Y)
    matriz_asincrona = np.dot(np.dot(espectros_centrados.T, A), espectros_centrados) / (N_espectros - 1)
    
    return matriz_sincrona, matriz_asincrona, x_data

# --- Función Auxiliar para la Descarga (Mejorada) ---
def get_file_download_link(fig, filename, file_format="png"):
    """Crea un enlace de descarga para la figura de Matplotlib en PNG (300dpi) o PDF (vectorial)."""
    buf = io.BytesIO()
    if file_format == "png":
        # PNG con alta resolución para web/presentaciones
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    elif file_format == "pdf":
        # PDF (vectorial, ideal para publicación)
        fig.savefig(buf, format="pdf", bbox_inches='tight')
    return buf.getvalue()

# --- Función Principal para Procesar y Graficar ---
def graficar_espectros_ftir(uploaded_files, header_row, x_label, y_label, invert_x, corregir_base, normalizar_100, poly_degree, hacer_2dcos):
    """Procesa los archivos subidos, genera la gráfica Matplotlib y la muestra con opciones de pre-procesamiento."""
    
    espectros_para_2d = []
    x_data_2d = None
    
    # Crea la figura de Matplotlib para 1D
    fig_1d, ax_1d = plt.subplots(figsize=(10, 5))
    datos_graficados = False
    
    for file in uploaded_files:
        try:
            # 1. Lectura del archivo
            df = pd.read_csv(io.StringIO(file.getvalue().decode("utf-8")), header=header_row)
            
            col_x = df.columns[0]
            col_y = df[df.columns[1]].values
            
            x_data = df[col_x].values
            y_data = col_y.copy()
            
            # 2. Pre-procesamiento
            y_procesada = y_data
            
            if corregir_base:
                st.info(f"Aplicando Corrección de Línea Base Polinomial (grado {poly_degree}) a '{file.name}'.")
                y_procesada = corregir_linea_base_polinomial(x_data, y_procesada, poly_degree)
                
            if normalizar_100:
                st.info(f"Aplicando Normalización a 100% de Transmitancia a '{file.name}'.")
                y_procesada = normalizar_a_100_transm(y_procesada)
            
            # 3. Preparar datos para 2D-COS
            if hacer_2dcos:
                if x_data_2d is None:
                    x_data_2d = x_data
                
                if len(x_data) != len(x_data_2d):
                    st.warning(f"El archivo '{file.name}' tiene un número diferente de puntos de onda. Se omitirá para el 2D-COS.")
                else:
                    espectros_para_2d.append(y_procesada)

            # 4. Graficar los datos procesados en 1D
            ax_1d.plot(x_data, y_procesada, label=file.name)
            datos_graficados = True
            
        except Exception as e:
            st.error(f"Error al procesar el archivo '{file.name}'. Detalle: {e}")
            continue

    if datos_graficados:
        # Personalización del gráfico 1D
        ax_1d.set_title("Espectros FTIR Pre-procesados (1D)")
        ax_1d.set_xlabel(x_label)
        ax_1d.set_ylabel(y_label)
        ax_1d.grid(True, linestyle='--', alpha=0.6)
        ax_1d.legend(title="Muestra", loc='best')

        if invert_x:
            ax_1d.invert_xaxis()

        # Muestra la figura 1D
        st.subheader("Gráfico de Espectros FTIR (1D)")
        st.pyplot(fig_1d)
        
        # Opciones de descarga 1D
        st.download_button(
            label="Descargar 1D (PNG, 300 dpi)",
            data=get_file_download_link(fig_1d, 'espectros_ftir_1D.png', "png"),
            file_name="espectros_ftir_1D.png",
            mime="image/png"
        )
        st.download_button(
            label="Descargar 1D (PDF, Publicación)",
            data=get_file_download_link(fig_1d, 'espectros_ftir_1D.pdf', "pdf"),
            file_name="espectros_ftir_1D.pdf",
            mime="application/pdf"
        )
        
        # --- LÓGICA DE CORRELACIÓN 2D ---
        if hacer_2dcos and len(espectros_para_2d) >= 2:
            st.markdown("---")
            st.subheader("Análisis de Correlación 2D (2D-COS)")
            
            espectros_matrix = np.array(espectros_para_2d)
            espectro_media = np.mean(espectros_matrix, axis=0)
            
            try:
                sincrona, asincrona, onda = calcular_correlacion_2d(espectros_matrix, x_data_2d)

                # --- Graficación Síncrona con marginales ---
                fig_sync = plt.figure(figsize=(10, 8))
                # Define la rejilla para los subplots: 4x4. El mapa 2D ocupará 3x3
                gs_sync = fig_sync.add_gridspec(4, 4, hspace=0.05, wspace=0.05) 

                # 1. Mapa 2D Síncrono (Principal)
                ax_sync = fig_sync.add_subplot(gs_sync[1:, :3])
                c_sync = ax_sync.contourf(onda, onda, sincrona, cmap='jet', levels=20)
                ax_sync.set_title("Mapa de Correlación Síncrona", loc='left')
                ax_sync.set_xlabel(x_label)
                ax_sync.set_ylabel(x_label)
                
                # 2. Gráfico Marginal Superior (Eje X)
                ax_top = fig_sync.add_subplot(gs_sync[0, :3], sharex=ax_sync)
                ax_top.plot(onda, espectro_media, color='black')
                ax_top.set_title("Mapa de Correlación Síncrona (con espectro promedio)", fontsize=10)
                ax_top.tick_params(axis="x", labelbottom=False) # Ocultar etiquetas X
                ax_top.tick_params(axis="y", labelleft=False) # Ocultar etiquetas Y
                ax_top.set_yticks([]) # Limpiar ticks Y
                ax_top.spines['right'].set_visible(False)
                ax_top.spines['top'].set_visible(False)
                
                # 3. Gráfico Marginal Derecho (Eje Y)
                ax_right = fig_sync.add_subplot(gs_sync[1:, 3], sharey=ax_sync)
                ax_right.plot(espectro_media, onda, color='black')
                ax_right.tick_params(axis="y", labelleft=False) # Ocultar etiquetas Y
                ax_right.tick_params(axis="x", labelbottom=False) # Ocultar etiquetas X
                ax_right.set_xticks([]) # Limpiar ticks X
                ax_right.spines['right'].set_visible(False)
                ax_right.spines['top'].set_visible(False)

                # 4. Barra de Color
                cbar_ax = fig_sync.add_subplot(gs_sync[1:3, 3]) # Posiciona la barra de color
                fig_sync.colorbar(c_sync, ax=cbar_ax, label='Intensidad de Correlación')
                cbar_ax.set_visible(False) # Ocultar el subplot usado para posicionar la barra

                if invert_x:
                    ax_sync.invert_xaxis()
                    ax_sync.invert_yaxis()
                    ax_top.invert_xaxis()
                    ax_right.invert_yaxis()
                
                st.pyplot(fig_sync)
                
                # Opciones de descarga Síncrona
                st.download_button(
                    label="Descargar Síncrona (PNG, 300 dpi)",
                    data=get_file_download_link(fig_sync, 'mapa_sincrono.png', "png"),
                    file_name="mapa_sincrono.png",
                    mime="image/png"
                )
                st.download_button(
                    label="Descargar Síncrona (PDF, Publicación)",
                    data=get_file_download_link(fig_sync, 'mapa_sincrono.pdf', "pdf"),
                    file_name="mapa_sincrono.pdf",
                    mime="application/pdf"
                )

                # --- Graficación Asíncrona con marginales ---
                fig_async = plt.figure(figsize=(10, 8))
                gs_async = fig_async.add_gridspec(4, 4, hspace=0.05, wspace=0.05) 

                # 1. Mapa 2D Asíncrono (Principal)
                ax_async = fig_async.add_subplot(gs_async[1:, :3])
                c_async = ax_async.contourf(onda, onda, asincrona, cmap='seismic', levels=20)
                ax_async.set_title("Mapa de Correlación Asíncrona", loc='left')
                ax_async.set_xlabel(x_label)
                ax_async.set_ylabel(x_label)

                # 2. Gráfico Marginal Superior (Eje X)
                ax_top_async = fig_async.add_subplot(gs_async[0, :3], sharex=ax_async)
                ax_top_async.plot(onda, espectro_media, color='black')
                ax_top_async.set_title("Mapa de Correlación Asíncrona (con espectro promedio)", fontsize=10)
                ax_top_async.tick_params(axis="x", labelbottom=False)
                ax_top_async.tick_params(axis="y", labelleft=False)
                ax_top_async.set_yticks([])
                ax_top_async.spines['right'].set_visible(False)
                ax_top_async.spines['top'].set_visible(False)

                # 3. Gráfico Marginal Derecho (Eje Y)
                ax_right_async = fig_async.add_subplot(gs_async[1:, 3], sharey=ax_async)
                ax_right_async.plot(espectro_media, onda, color='black')
                ax_right_async.tick_params(axis="y", labelleft=False)
                ax_right_async.tick_params(axis="x", labelbottom=False)
                ax_right_async.set_xticks([])
                ax_right_async.spines['right'].set_visible(False)
                ax_right_async.spines['top'].set_visible(False)

                # 4. Barra de Color
                cbar_ax_async = fig_async.add_subplot(gs_async[1:3, 3]) 
                fig_async.colorbar(c_async, ax=cbar_ax_async, label='Intensidad de Correlación')
                cbar_ax_async.set_visible(False)

                if invert_x:
                    ax_async.invert_xaxis()
                    ax_async.invert_yaxis()
                    ax_top_async.invert_xaxis()
                    ax_right_async.invert_yaxis()
                
                st.pyplot(fig_async)
                
                # Opciones de descarga Asíncrona
                st.download_button(
                    label="Descargar Asíncrona (PNG, 300 dpi)",
                    data=get_file_download_link(fig_async, 'mapa_asincrono.png', "png"),
                    file_name="mapa_asincrono.png",
                    mime="image/png"
                )
                st.download_button(
                    label="Descargar Asíncrona (PDF, Publicación)",
                    data=get_file_download_link(fig_async, 'mapa_asincrono.pdf', "pdf"),
                    file_name="mapa_asincrono.pdf",
                    mime="application/pdf"
                )
                
            except Exception as e:
                st.error(f"Error al calcular o graficar el 2D-COS. Detalle: {e}. Asegúrate de que los espectros son compatibles.")
                
        elif hacer_2dcos and len(espectros_para_2d) < 2:
             st.warning("Se necesitan al menos 2 espectros con el mismo eje X para calcular el 2D-COS.")
             
    else:
        st.info("Sube uno o varios archivos CSV para comenzar.")

# --- Interfaz de Usuario de Streamlit (sin cambios en los widgets) ---

st.title("Espectros FTIR: Visualizador y Pre-procesador Avanzado 🔬✨")
st.markdown("Ahora con **Corrección de Línea Base Polinomial**, **2D-COS con marginales** y **descarga PDF**.")

# --- BARRA LATERAL CON WIDGETS ---
with st.sidebar:
    st.header("1. Cargar y Configurar")
    uploaded_files = st.file_uploader(
        "Selecciona uno o más archivos CSV:",
        type=["csv"],
        accept_multiple_files=True
    )
    
    header_default = 3
    header_row = st.number_input(
        "Número de Fila del Encabezado (Comenzando en 0):", 
        min_value=0, 
        value=header_default,
        help="La línea donde se encuentran los nombres de las columnas (ej. cm-1, %T)."
    )
    st.markdown("---")

    st.header("2. Pre-procesamiento Espectral")
    
    # WIDGET DE CORRECCIÓN POLINOMIAL
    corregir_base = st.checkbox(
        "Corregir Línea Base (Ajuste Polinomial)", 
        value=False,
        help="Aplica un ajuste polinomial a los datos y lo resta para aplanar el fondo."
    )
    if corregir_base:
        poly_degree = st.slider(
            "Grado del Polinomio (1=lineal, 2=cuadrático, etc.):",
            min_value=1,
            max_value=10,
            value=2,
            help="Un grado bajo (1-3) suele ser suficiente para corregir pendientes suaves."
        )
    else:
        poly_degree = 0 
        
    normalizar_100 = st.checkbox(
        "Normalizar a 100% Transmitancia", 
        value=False,
        help="Ajusta el valor máximo de la señal a 100."
    )
    st.markdown("---")
    
    st.header("3. Análisis de Correlación 2D")
    hacer_2dcos = st.checkbox(
        "Calcular Correlación 2D (2D-COS)",
        value=False,
        help="Requiere 2 o más espectros (matriz de datos) con el mismo eje X."
    )
    st.markdown("---")
    
    st.header("4. Personalizar Gráfica")
    x_label = st.text_input("Etiqueta del Eje X:", value="Número de Onda ($\mathbf{cm^{-1}}$)")
    y_label = st.text_input("Etiqueta del Eje Y:", value="Intensidad Procesada")
    invert_x = st.checkbox("Invertir Eje X", value=True)
    st.markdown("---")

# 5. Llamar a la función principal
if uploaded_files:
    graficar_espectros_ftir(
        uploaded_files, 
        header_row, 
        x_label, 
        y_label, 
        invert_x, 
        corregir_base, 
        normalizar_100,
        poly_degree, 
        hacer_2dcos 
    )
else:
    st.warning("Por favor, sube tus archivos CSV en la barra lateral para generar la gráfica.")