import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

st.set_page_config(page_title="Analizador SpectraSuite", layout="wide")

st.title("🧪 Análisis de Espectros y Stern-Volmer")

uploaded_files = st.sidebar.file_uploader("Cargar archivos .txt", accept_multiple_files=True)

def limpiar_y_leer(file):
    """Busca el inicio de los datos y lee ignorando pies de página erróneos"""
    content = file.getvalue().decode("utf-8").splitlines()
    
    # 1. Encontrar el inicio de los datos
    data_start = 0
    for i, line in enumerate(content):
        if ">>>>>Begin" in line:
            data_start = i + 1
            break
    
    # 2. Leer los datos
    # Usamos on_bad_lines='skip' para que ignore el footer de SpectraSuite
    df = pd.read_csv(
        io.StringIO("\n".join(content[data_start:])),
        sep=r'\s+', 
        names=['Wavelength', 'Intensity'],
        decimal='.',
        engine='python',
        on_bad_lines='skip' 
    )
    
    # 3. Limpieza de seguridad: Forzar que ambas columnas sean numéricas 
    # (por si acaso el footer empezó con números pero luego tuvo texto)
    df['Wavelength'] = pd.to_numeric(df['Wavelength'], errors='coerce')
    df['Intensity'] = pd.to_numeric(df['Intensity'], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    
    return df

if uploaded_files:
    st.sidebar.subheader("Configuración de Muestras")
    file_names = [f.name for f in uploaded_files]
    
    # Tabla editable para factores
    df_config = pd.DataFrame({
        "Archivo": file_names,
        "Factor_Dilucion": [1.0] * len(file_names),
        "Conc_Quencher [Q]": [0.0] * len(file_names)
    })
    
    st.write("### 1. Ajusta los factores de dilución y concentraciones")
    edited_config = st.data_editor(df_config, hide_index=True, use_container_width=True)

    espectros_procesados = []
    
    for _, row in edited_config.iterrows():
        file = next(f for f in uploaded_files if f.name == row["Archivo"])
        try:
            df = limpiar_y_leer(file)
            
            # --- SUAVIZADO ---
            # Ventana de 11 puntos para eliminar ruido electrónico
            df['Intensity_Smooth'] = df['Intensity'].rolling(window=11, center=True).mean()
            
            # --- CORRECCIÓN ---
            df['Intensity_Corr'] = df['Intensity_Smooth'] * row["Factor_Dilucion"]
            
            # --- CÁLCULO DE MÁXIMO PROMEDIADO ---
            # Buscamos el índice del máximo en la zona visible (ej. > 300nm para evitar ruido UV)
            df_filtered = df[df['Wavelength'] > 450]
            idx_max = df_filtered['Intensity_Corr'].idxmax()
            
            # Promediamos 10 puntos alrededor del máximo
            rango_max = df.iloc[max(0, idx_max-5) : min(len(df), idx_max+6)]
            max_val = rango_max['Intensity_Corr'].mean()
            wvl_max = rango_max['Wavelength'].mean()
            
            espectros_procesados.append({
                "nombre": row["Archivo"],
                "df": df,
                "max_val": max_val,
                "wvl_max": wvl_max,
                "quencher": row["Conc_Quencher [Q]"]
            })
        except Exception as e:
            st.error(f"Error en {row['Archivo']}: {e}")

    # --- GRÁFICAS ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Espectros Suavizados y Corregidos")
        fig1, ax1 = plt.subplots()
        for esp in espectros_procesados:
            ax1.plot(esp["df"]["Wavelength"], esp["df"]["Intensity_Corr"], label=esp["nombre"], alpha=0.8)
            ax1.scatter(esp["wvl_max"], esp["max_val"], s=30, edgecolors='black', zorder=5)
            
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Intensidad Corregida")
        ax1.legend(loc='upper right', fontsize='x-small')
        st.pyplot(fig1)

    with col2:
        st.subheader("Análisis Stern-Volmer")
        # El blanco es el que tiene [Q] = 0
        puros = [e for e in espectros_procesados if e["quencher"] == 0]
        
        if puros and len(espectros_procesados) > 1:
            I0 = puros[0]["max_val"]
            sv_list = [{"Q": e["quencher"], "I0_I": I0 / e["max_val"]} for e in espectros_procesados]
            df_sv = pd.DataFrame(sv_list).sort_values("Q")
            
            fig2, ax2 = plt.subplots()
            ax2.scatter(df_sv["Q"], df_sv["I0_I"], color='red', label='Datos')
            
            m, b = np.polyfit(df_sv["Q"], df_sv["I0_I"], 1)
            ax2.plot(df_sv["Q"], m*df_sv["Q"] + b, 'k--', label=f'R²={np.corrcoef(df_sv["Q"], df_sv["I0_I"])[0,1]**2:.3f}')
            
            ax2.set_xlabel("[Quencher]")
            ax2.set_ylabel("$I_0 / I$")
            ax2.legend()
            st.pyplot(fig2)
            
            st.metric("Constante Stern-Volmer (Ksv)", f"{m:.4f}")
            st.latex(f"I_0/I = {m:.4f}[Q] + {b:.4f}")
        else:
            st.warning("Asigna '0' a la muestra pura para calcular Stern-Volmer.")
else:
    st.info("Sube archivos para comenzar.")