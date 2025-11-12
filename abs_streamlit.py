import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import os
import plotly.express as px
from scipy.signal import savgol_filter

# --- Funciones de procesamiento de datos ---
def read_spectrum_data(file_object):
    """
    Lee y parsea los datos de un espectro desde un archivo.
    """
    try:
        file_object.seek(0)
        lines = file_object.readlines()
        lines = [line.decode("latin-1") for line in lines]

        data_start_line = None
        for i, line in enumerate(lines):
            if ">>>>>Begin Processed Spectral Data<<<<<" in line:
                data_start_line = i + 1
                break
        
        if data_start_line is None:
            st.error("No se encontró la sección de datos '>>>>>Begin Processed Spectral Data<<<<<' en el archivo.")
            return None

        df = pd.read_csv(
            io.StringIO(''.join(lines[data_start_line:])),
            sep='\t',
            header=None,
            names=['Longitud de onda', 'Valor'],
            on_bad_lines='skip'
        )
        
        df = df.dropna().apply(pd.to_numeric, errors='coerce').dropna()
        
        if df.empty:
            st.error("Los datos procesados están vacíos o no son válidos.")
            return None

        return df

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        return None

def calculate_tauc(df, band_type):
    """
    Calcula los datos para el gráfico de Tauc.
    """
    hc = 1240.0
    
    df['Energía del fotón (eV)'] = hc / df['Longitud de onda']
    
    n = 2.0 if band_type == "directo" else 0.5
    
    # Asume que el DF ya tiene la columna 'Absorbancia (u.a)'
    df['(Abs*hv)^(1/n)'] = (df['Absorbancia (u.a)'] * df['Energía del fotón (eV)'])**(1/n)
    
    return df

def perform_linear_fit(df, start_eV, end_eV):
    """
    Realiza un ajuste lineal a los datos de Tauc en un rango de energía específico.
    """
    filtered_df = df[(df['Energía del fotón (eV)'] >= start_eV) & (df['Energía del fotón (eV)'] <= end_eV)]
    
    if filtered_df.empty or len(filtered_df) < 2:
        return None, None, "No hay suficientes puntos en el rango seleccionado para realizar el ajuste."
    
    x = filtered_df['Energía del fotón (eV)'].values
    y = filtered_df['(Abs*hv)^(1/n)'].values
    
    m, b = np.polyfit(x, y, 1)
    
    # Cálculo de Bandgap: intercepto en el eje X cuando y=0
    bandgap = -b / m
    
    # Generar la línea de ajuste para el gráfico
    x_fit = np.array([x.min(), bandgap, x.max()])
    y_fit = m * x_fit + b
    
    return bandgap, pd.DataFrame({'x': x_fit, 'y': y_fit}), None

def smooth_spectrum(data, window_length, poly_order):
    """
    Aplica el filtro de Savitzky-Golay a los datos.
    """
    if window_length % 2 == 0:
        window_length += 1
    if window_length <= poly_order:
        window_length = poly_order + 2

    return savgol_filter(data, window_length, poly_order)

def create_download_button(fig, filename_prefix):
    """
    Crea un botón de descarga para una figura de Plotly con calidad de publicación (scale=3).
    """
    # Usar scale=3 para aumentar la resolución de la imagen PNG
    buf = fig.to_image(format="png", scale=3) 
    st.download_button(
        label=f"Descargar Gráfico ({filename_prefix})",
        data=buf,
        file_name=f"{filename_prefix}.png",
        mime="image/png"
    )

# --- Lógica principal de la aplicación Streamlit ---
st.set_page_config(page_title="Análisis de Espectros", layout="wide")
st.title('Análisis de Espectros UV-Vis y Fluorescencia 🧪')

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = {}

# --- Barra Lateral (Opciones Globales) ---
st.sidebar.header("Opciones de Archivo y Tipo")
spectrum_type_selection = st.sidebar.radio("Los archivos cargados son de:", ("Absorbancia", "Fluorescencia"))

st.sidebar.header("Opciones de Suavizado")
use_smoothing = st.sidebar.checkbox("Aplicar suavizado a los datos")
if use_smoothing:
    window_length = st.sidebar.slider("Longitud de la ventana", 3, 51, 11, step=2)
    poly_order = st.sidebar.slider("Orden del polinomio", 1, 5, 2, step=1)
else:
    window_length = None
    poly_order = None

st.markdown("""
Sube uno o más archivos. La aplicación procesará y organizará los resultados en las pestañas correspondientes.
Los gráficos se descargan con **calidad de publicación**.
""")

uploaded_files = st.file_uploader(
    "Selecciona tus archivos de datos",
    accept_multiple_files=True
)

# --- Lógica de Reprocesamiento y Cacheo ---
if uploaded_files:
    # Definir el estado actual de la configuración que impacta el procesamiento
    current_config = (
        spectrum_type_selection,
        use_smoothing,
        window_length,
        poly_order,
        tuple(sorted([f.name for f in uploaded_files])) # Usar una tupla de nombres para comparación estable
    )

    # Si la configuración ha cambiado, borrar el caché y forzar el reprocesamiento de todos los archivos
    if st.session_state.get('last_config') != current_config:
        st.session_state.processed_files = {}
        st.session_state.last_config = current_config

    # --- 1. Procesamiento de Datos ---
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        
        # Solo procesar si el archivo no está en el caché
        if file_name not in st.session_state.processed_files:
            df = read_spectrum_data(uploaded_file)
            
            if df is not None and not df.empty:
                df['Archivo'] = os.path.splitext(file_name)[0]
                
                if spectrum_type_selection == "Absorbancia":
                    y_label = 'Absorbancia (u.a)'
                else:
                    y_label = 'Intensidad de Fluorescencia (u.a)'
                
                df.rename(columns={'Valor': y_label}, inplace=True)
                
                df_to_plot = df.copy() 
                
                if use_smoothing:
                    df_to_plot[y_label] = smooth_spectrum(df_to_plot[y_label].values, window_length, poly_order)
                
                st.session_state.processed_files[file_name] = {
                    'data': df_to_plot, 
                    'y_label': y_label, 
                    'is_smoothed': use_smoothing
                }

# --- 2. Creación de Pestañas ---
if st.session_state.processed_files:
    tab_individual, tab_comparacion = st.tabs(["Gráficos Individuales", "Gráfico de Comparación"])

    # --- Lógica para la Pestaña de Gráficos Individuales ---
    with tab_individual:
        st.header(f"Resultados Individuales de {spectrum_type_selection}")
        
        for file_name, data in st.session_state.processed_files.items():
            # Asegurarse de que solo se muestren los archivos que coinciden con la última selección de tipo de espectro
            if data['y_label'] == ('Absorbancia (u.a)' if spectrum_type_selection == "Absorbancia" else 'Intensidad de Fluorescencia (u.a)'):
                with st.container():
                    st.subheader(f"Archivo: {file_name}")
                    key_suffix = os.path.splitext(file_name)[0].replace(' ', '_').replace('.', '')
                    
                    df_plot = data['data']
                    y_label = data['y_label']
                    is_smoothed = data['is_smoothed']
                    
                    download_prefix = f"{'suavizado' if is_smoothed else 'original'}_{key_suffix}"
                    title_suffix = " (Suavizado)" if is_smoothed else ""

                    if spectrum_type_selection == "Absorbancia":
                        # --- Gráfico de Absorbancia ---
                        fig_abs = px.line(df_plot, 
                                        x='Longitud de onda', 
                                        y=y_label,
                                        title=f"Espectro de Absorbancia para {file_name}{title_suffix}")
                        
                        st.plotly_chart(fig_abs, use_container_width=True, key=f"abs_chart_{key_suffix}")
                        create_download_button(fig_abs, f"absorbancia_{download_prefix}")

                        # --- Análisis de Tauc ---
                        with st.expander("Análisis de Tauc y Cálculo del Bandgap"):
                            st.markdown("### Análisis de Tauc")
                            
                            band_type = st.radio(
                                "Selecciona el tipo de transición electrónica:",
                                ("directo", "indirecto"),
                                key=f"band_type_{key_suffix}"
                            )
                            
                            df_tauc = calculate_tauc(df_plot.copy(), band_type)

                            fig_tauc = px.line(df_tauc, 
                                            x='Energía del fotón (eV)', 
                                            y='(Abs*hv)^(1/n)',
                                            title=f"Gráfico de Tauc para {file_name} (Transición {band_type}){title_suffix}")
                            st.plotly_chart(fig_tauc, use_container_width=True, key=f"tauc_chart_{key_suffix}")

                            st.markdown("**Selecciona el rango para el ajuste lineal**")
                            col1, col2 = st.columns(2)
                            with col1:
                                min_ev = float(df_tauc['Energía del fotón (eV)'].min())
                                max_ev = float(df_tauc['Energía del fotón (eV)'].max())
                                
                                fit_start = st.number_input("Energía de inicio (eV)", 
                                                            min_value=min_ev, 
                                                            max_value=max_ev,
                                                            value=min(2.0, max_ev - 0.1), 
                                                            step=0.05,
                                                            key=f"fit_start_{key_suffix}")
                            with col2:
                                fit_end = st.number_input("Energía de fin (eV)", 
                                                        min_value=min_ev, 
                                                        max_value=max_ev,
                                                        value=min(3.0, max_ev), 
                                                        step=0.05,
                                                        key=f"fit_end_{key_suffix}")

                            if st.button("Calcular Bandgap", key=f"calc_button_{key_suffix}"):
                                bandgap, fit_line_df, error = perform_linear_fit(df_tauc, fit_start, fit_end)
                                if error:
                                    st.error(error)
                                else:
                                    st.success(f"Bandgap calculado: **{bandgap:.3f} eV**")
                                    
                                    fig_tauc_fit = px.line(df_tauc, 
                                                        x='Energía del fotón (eV)', 
                                                        y='(Abs*hv)^(1/n)',
                                                        title=f"Gráfico de Tauc con Ajuste para {file_name} (Transición {band_type}){title_suffix}")
                                    fig_tauc_fit.add_scatter(x=fit_line_df['x'], y=fit_line_df['y'], mode='lines', name='Ajuste Lineal', line=dict(color='red', width=3))
                                    fig_tauc_fit.add_vline(x=bandgap, line_width=2, line_dash="dash", line_color="green", annotation_text=f"Eg = {bandgap:.3f} eV", annotation_position="bottom right")
                                    
                                    st.plotly_chart(fig_tauc_fit, use_container_width=True, key=f"tauc_chart_fit_{key_suffix}")
                                    create_download_button(fig_tauc_fit, f"tauc_plot_bandgap_{download_prefix}")
                
                    elif spectrum_type_selection == "Fluorescencia":
                        # --- Gráfico de Fluorescencia ---
                        st.markdown("### Espectro de Fluorescencia")
                        
                        fig_fluor = px.line(df_plot, 
                                            x='Longitud de onda', 
                                            y=y_label,
                                            title=f"Espectro de Fluorescencia para {file_name}{title_suffix}")
                        
                        st.plotly_chart(fig_fluor, use_container_width=True, key=f"fluor_chart_{key_suffix}")
                        create_download_button(fig_fluor, f"fluorescencia_{download_prefix}")
                    
                    st.markdown("---")

    # --- Lógica para la Pestaña de Comparación ---
    with tab_comparacion:
        if len(st.session_state.processed_files) > 1:
            st.header("Gráfico de Comparación de Espectros")
            st.markdown(f"**Nota:** El gráfico de comparación usa la versión **{'suavizada' if use_smoothing else 'original'}** de los datos para la muestra.")
            
            combined_df = pd.DataFrame()
            y_label_combined = ''
            
            # Filtra y combina solo los archivos del tipo de espectro seleccionado
            files_to_compare = {k: v for k, v in st.session_state.processed_files.items() if v['y_label'] == ('Absorbancia (u.a)' if spectrum_type_selection == "Absorbancia" else 'Intensidad de Fluorescencia (u.a)')}

            if not files_to_compare:
                st.info(f"Sube archivos de **{spectrum_type_selection}** para la comparación.")
            elif len(files_to_compare) < 2:
                st.info("Necesitas al menos dos archivos del mismo tipo para generar el gráfico de comparación.")
            else:
                y_label_combined = next(iter(files_to_compare.values()))['y_label']
                
                for file_name, data in files_to_compare.items():
                    combined_df = pd.concat([combined_df, data['data']], ignore_index=True)
                
                title_comp = f'Comparación de Espectros de {spectrum_type_selection} ({len(files_to_compare)} muestras)'
                if use_smoothing:
                     title_comp += " (Suavizados)"

                fig_comp = px.line(combined_df, 
                                   x='Longitud de onda', 
                                   y=y_label_combined, 
                                   color='Archivo',
                                   title=title_comp,
                                   labels={y_label_combined: y_label_combined})
                
                st.plotly_chart(fig_comp, use_container_width=True)
                create_download_button(fig_comp, f"comparacion_espectros_{spectrum_type_selection.lower()}_{'suavizados' if use_smoothing else 'originales'}")
        else:
            st.info("Sube al menos dos archivos para generar un gráfico de comparación.")