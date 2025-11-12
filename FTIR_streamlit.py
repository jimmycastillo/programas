import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Visualizador de Espectros FTIR",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Función Principal para Procesar y Graficar ---
def graficar_espectros_ftir(uploaded_files, header_row, x_label, y_label, invert_x):
    """Procesa los archivos subidos, genera la gráfica Matplotlib y la muestra."""
    
    # Crea la figura de Matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    
    datos_graficados = False
    
    for file in uploaded_files:
        try:
            # Lee el archivo CSV. El parámetro 'header' maneja las líneas de metadatos.
            df = pd.read_csv(io.StringIO(file.getvalue().decode("utf-8")), header=header_row)
            
            # Asumimos que la primera columna es X (Número de Onda) y la segunda es Y (Transmitancia/Absorbancia)
            col_x = df.columns[0]
            col_y = df.columns[1]
            
            # Graficar los datos
            ax.plot(df[col_x], df[col_y], label=file.name)
            datos_graficados = True
            
        except Exception as e:
            st.error(f"Error al procesar el archivo '{file.name}'. Asegúrate de que el formato (CSV) y el número de encabezado son correctos. Detalle: {e}")
            continue

    if datos_graficados:
        # Personalización del gráfico
        ax.set_title("Espectros FTIR")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(title="Muestra", loc='best')

        # Opción para invertir el eje X (común en FTIR)
        if invert_x:
            ax.invert_xaxis()

        # Muestra la figura de Matplotlib en Streamlit
        st.pyplot(fig)
        
        # Permite al usuario descargar el gráfico
        st.download_button(
            label="Descargar Gráfico (PNG)",
            data=get_image_download_link(fig, 'espectros_ftir.png'),
            file_name="espectros_ftir.png",
            mime="image/png"
        )
    else:
        st.info("Sube uno o varios archivos CSV para comenzar.")

# --- Función Auxiliar para la Descarga ---
def get_image_download_link(fig, filename):
    """Crea un enlace de descarga para la figura de Matplotlib."""
    buf = io.BytesIO()
    # Guardar con alta resolución
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    return buf.getvalue()

# --- Interfaz de Usuario de Streamlit ---

st.title("Espectros FTIR: Visualizador Interactivo 🔬")
st.markdown("Sube tus archivos CSV de espectros y personaliza la visualización.")

# 1. Zona de Subida de Archivos
with st.sidebar:
    st.header("1. Cargar Archivos")
    uploaded_files = st.file_uploader(
        "Selecciona uno o más archivos CSV:",
        type=["csv"],
        accept_multiple_files=True
    )
    st.markdown("---")
    
# 2. Configuración de Archivo (Asumiendo que el archivo de ejemplo tiene 3 líneas de metadatos)
with st.sidebar:
    st.header("2. Configuración de Datos")
    # Para tu archivo 'JC 1.csv', la línea 4 es el encabezado (índice 3)
    header_default = 3
    header_row = st.number_input(
        "Número de Fila del Encabezado (Comenzando en 0):", 
        min_value=0, 
        value=header_default,
        help="La línea donde se encuentran los nombres de las columnas (ej. cm-1, %T)."
    )
    st.markdown("---")

# 3. Personalización de Ejes (Widgets para cambiar las etiquetas)
with st.sidebar:
    st.header("3. Personalizar Gráfica")
    
    # Campo para cambiar el Eje X
    x_label = st.text_input(
        "Etiqueta del Eje X (Número de Onda):", 
        value="Número de Onda ($\mathbf{cm^{-1}}$)",
        help="Usa notación LaTeX para superíndices, como en el valor por defecto."
    )
    
    # Campo para cambiar el Eje Y
    y_label = st.text_input(
        "Etiqueta del Eje Y (Señal):", 
        value="Transmitancia (%)"
    )

    # Opción para invertir el eje X
    invert_x = st.checkbox("Invertir Eje X", value=True)
    st.markdown("---")


# 4. Llamar a la función de graficación si hay archivos
if uploaded_files:
    graficar_espectros_ftir(uploaded_files, header_row, x_label, y_label, invert_x)
else:
    st.warning("Por favor, sube tus archivos CSV en la barra lateral para generar la gráfica.")