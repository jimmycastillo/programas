import streamlit as st
import pandas as pd
import re
import io
import os
import plotly.express as px

# --- Funciones de procesamiento de datos ---
def read_dls_data(file_object):
    """
    Lee y parsea los datos de autocorrelación y distribución de tamaño 
    desde un objeto de archivo DLS cargado.
    """
    try:
        file_object.seek(0)
        lines = file_object.readlines()
        lines = [line.decode("latin-1") for line in lines]

        try:
            auto_start = lines.index(next(line for line in lines if "Autocorrelacion" in line)) + 1
            size_start = lines.index(next(line for line in lines if "Distribuci" in line)) + 1
        except StopIteration:
            st.error("No se encontraron las secciones 'Autocorrelacion' o 'Distribución' en el archivo.")
            return None, None
        
        auto_data = []
        for line in lines[auto_start:size_start - 1]:
            parts = re.split(r'\s+', line.strip())
            if len(parts) == 3:
                try:
                    t, g1, fit = map(float, parts)
                    auto_data.append((t, g1, fit))
                except ValueError:
                    continue

        auto_df = pd.DataFrame(auto_data, columns=["Tiempo (s)", "g1(t)", "Ajuste"])

        size_data = []
        for line in lines[size_start:]:
            parts = re.split(r'\s+', line.strip())
            if len(parts) == 3:
                try:
                    d, dist, accum = map(float, parts)
                    size_data.append((d, dist, accum))
                except ValueError:
                    continue

        size_df = pd.DataFrame(size_data, columns=["Diámetro (nm)", "Distribución", "Acumulada"])
        
        return auto_df, size_df

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
        return None, None

def plot_autocorrelation(df_auto, file_name, key_suffix):
    """
    Grafica la función de autocorrelación y su ajuste con Plotly.
    """
    df_auto_melted = df_auto.melt(id_vars=["Tiempo (s)"], var_name="Curva", value_name="Valor")
    
    fig = px.line(df_auto_melted, 
                  x="Tiempo (s)", 
                  y="Valor", 
                  color="Curva",
                  title=f"Función de Autocorrelación para {file_name}",
                  labels={"Valor": "Intensidad Normalizada"})

    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, key=f"autocorr_chart_{key_suffix}")
    return fig

# --- NUEVA FUNCIÓN para detectar máximos ---
def find_local_maxima(df_size):
    """
    Identifica los índices de los máximos locales en la columna 'Distribución'.
    """
    maxima_indices = []
    distribution = df_size['Distribución'].values
    
    # Asegurarse de que el array tenga al menos 3 elementos
    if len(distribution) < 3:
        return maxima_indices

    # Iterar sobre los elementos intermedios
    for i in range(1, len(distribution) - 1):
        if distribution[i] > distribution[i-1] and distribution[i] > distribution[i+1]:
            maxima_indices.append(i)
    
    # También considerar los extremos si son máximos
    if distribution[0] > distribution[1]:
        maxima_indices.append(0)
    if distribution[-1] > distribution[-2]:
        maxima_indices.append(len(distribution) - 1)
        
    return maxima_indices

def plot_size_distribution(df_size, file_name, key_suffix):
    """
    Grafica la distribución de tamaño y la frecuencia acumulada con Plotly,
    incluyendo el valor del diámetro en los máximos.
    """
    fig = px.scatter(df_size, 
                     x="Diámetro (nm)", 
                     y="Distribución", 
                     title=f"Distribución de Tamaño de Partícula para {file_name}",
                     labels={"Distribución": "Distribución (u.a)"})

    fig.update_traces(marker=dict(size=8), mode='lines+markers')
    fig.update_layout(xaxis_title="Diámetro (nm)", yaxis_title="Distribución (u.a)", hovermode="x unified")

    fig.add_trace(px.line(df_size, x="Diámetro (nm)", y="Acumulada").data[0])
    fig.data[-1].update(name="Acumulada", yaxis="y2")
    fig.layout.yaxis2 = dict(overlaying='y', side='right', title="Acumulada (u.a)")

    # --- Lógica para mostrar los máximos ---
    maxima_indices = find_local_maxima(df_size)
    
    for i in maxima_indices:
        diametro = df_size.loc[i, "Diámetro (nm)"]
        distribucion = df_size.loc[i, "Distribución"]
        
        # Agrega una anotación para cada pico
        fig.add_annotation(
            x=diametro,
            y=distribucion,
            text=f'{diametro:.2f} nm',
            showarrow=True,
            arrowhead=2,
            ax=10,
            ay=-20
        )
    # --- FIN de la lógica para los máximos ---

    st.plotly_chart(fig, key=f"distrib_chart_{key_suffix}")
    return fig

def create_download_button(fig, filename_prefix):
    """
    Crea un botón de descarga para una figura de Plotly.
    """
    buf = fig.to_image(format="png")
    st.download_button(
        label=f"Descargar Gráfico ({filename_prefix})",
        data=buf,
        file_name=f"{filename_prefix}.png",
        mime="image/png"
    )

# --- Lógica principal de la aplicación Streamlit ---
st.set_page_config(page_title="Análisis de DLS", layout="wide")
st.title('Aplicación Interactiva para Análisis de DLS')

st.markdown("""
Sube uno o más archivos de texto de DLS. La aplicación generará 
automáticamente los gráficos de función de autocorrelación, 
distribución de tamaño y frecuencia acumulada para cada archivo.
""")

uploaded_files = st.file_uploader(
    "Selecciona tus archivos de datos DLS",
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        
        # Genera un key único para cada archivo
        key_suffix = os.path.splitext(file_name)[0].replace(' ', '_')
        
        st.subheader(f"Resultados para: {file_name}")
        
        auto_df, size_df = read_dls_data(uploaded_file)
        
        if auto_df is not None and not auto_df.empty:
            st.markdown("### Función de Autocorrelación")
            fig_auto = plot_autocorrelation(auto_df, file_name, key_suffix)
            create_download_button(fig_auto, f"autocorrelacion_{os.path.basename(file_name).replace('.', '_')}")
        
        if size_df is not None and not size_df.empty:
            st.markdown("### Distribución de Tamaño")
            fig_size = plot_size_distribution(size_df, file_name, key_suffix)
            create_download_button(fig_size, f"distribucion_{os.path.basename(file_name).replace('.', '_')}")

    
        st.markdown("---")