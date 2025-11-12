import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.signal import find_peaks, savgol_filter 
import io

# --- PALETA DE COLORES PARA ELEMENTOS IDENTIFICADOS ---
COLORES_ELEMENTOS = {
    'H': 'red', 'He': 'orange', 'Li': 'darkgreen', 'Na': 'yellow', 'K': 'purple',
    'Ca': 'lime', 'Fe': 'brown', 'Mg': 'teal', 'Al': 'magenta', 'Si': 'black',
    'C': 'darkred', 'N': 'navy', 'O': 'olive', 'Cu': 'darkcyan', 'Zn': 'darkblue',
    'Pb': 'gray', 'Cd': 'gold'
}

# Base de datos de líneas atómicas comunes (NIST Strong Lines)
LINEAS_ATOMICAS = {
    'H': [656.28, 486.13, 434.05],
    'He': [587.56, 447.15, 402.62],
    'Li': [670.78, 610.36],
    'Na': [589.00, 589.59],
    'K': [766.49, 769.90],
    'Ca': [422.67, 393.37, 396.85],
    'Fe': [358.12, 373.49, 385.99, 404.58],
    'Mg': [285.21, 279.55],
    'Al': [396.15, 394.40],
    'Si': [288.16, 251.61],
    'C': [247.86, 193.09],
    'N': [746.83, 821.63],
    'O': [777.19, 844.63],
    'Cu': [324.754, 327.396],
    'Zn': [213.856, 481.053],
    'Pb': [405.782, 368.347],
    'Cd': [228.802, 326.106],
}

def cargar_archivo_asc(contenido):
    """Carga archivo ASC y devuelve DataFrame con datos del espectro"""
    try:
        texto = contenido.getvalue().decode('utf-8')
        lineas = texto.strip().split('\n')
        
        datos = []
        for linea in lineas:
            partes = linea.split()
            if len(partes) >= 2:
                try:
                    # Convertir formato europeo (coma decimal) a float
                    longitud_onda = float(partes[0].replace(',', '.'))
                    intensidad = float(partes[1].replace(',', '.'))
                    datos.append([longitud_onda, intensidad])
                except ValueError:
                    continue
        
        df = pd.DataFrame(datos, columns=['Longitud_Onda', 'Intensidad_Original'])
        return df.sort_values('Longitud_Onda').reset_index(drop=True)
    
    except Exception as e:
        st.error(f"Error al cargar archivo: {e}")
        return None

def corregir_fondo(intensidades, window_length=201, polyorder=3):
    """
    Calcula la corrección de línea base usando el Filtro Savitzky-Golay.
    """
    try:
        if window_length >= len(intensidades):
            window_length = len(intensidades) - 1
            if window_length % 2 == 0:
                window_length -= 1
            if window_length < 3:
                st.warning("Datos insuficientes para el filtro SG. No se aplica corrección.")
                return np.zeros_like(intensidades), intensidades

        # Aplicar el filtro Savitzky-Golay (suavizado agresivo para el fondo)
        baseline = savgol_filter(intensidades, window_length=window_length, polyorder=polyorder)
        
        # Corregir el espectro
        intensidad_corregida = intensidades - baseline
        intensidad_corregida[intensidad_corregida < 0] = 0
        
        return baseline, intensidad_corregida
    except Exception as e:
        st.warning(f"Error en la corrección de fondo (SG): {e}. Se usará la intensidad original.")
        return np.zeros_like(intensidades), intensidades

def encontrar_picos(espectro, columna_intensidad, altura_minima=0.1, distancia=5):
    """Encuentra picos en el espectro (usando la columna especificada)"""
    intensidades = espectro[columna_intensidad].values
    max_intensidad = np.max(intensidades) if np.max(intensidades) > 0 else 1
    
    picos, propiedades = find_peaks(intensidades, height=altura_minima * max_intensidad, distance=distancia)
    
    resultados = []
    for pico in picos:
        resultados.append({
            'Longitud_Onda': espectro.iloc[pico]['Longitud_Onda'],
            'Intensidad': espectro.iloc[pico][columna_intensidad],
            'Indice': pico
        })
    
    return pd.DataFrame(resultados)

def buscar_coincidencias(picos, elemento, tolerancia=0.5):
    """Busca coincidencias entre picos y líneas atómicas"""
    if elemento not in LINEAS_ATOMICAS:
        return pd.DataFrame()
    
    lineas_elemento = LINEAS_ATOMICAS[elemento]
    coincidencias = []
    
    for pico_idx, pico in picos.iterrows():
        longitud_onda_pico = pico['Longitud_Onda']
        
        for linea in lineas_elemento:
            diferencia = abs(longitud_onda_pico - linea)
            if diferencia <= tolerancia:
                coincidencias.append({
                    'Longitud_Onda_Pico': longitud_onda_pico,
                    'Intensidad_Pico': pico['Intensidad'],
                    'Linea_Atomica': linea,
                    'Elemento': elemento,
                    'Diferencia': diferencia
                })
    
    return pd.DataFrame(coincidencias)

def generar_grafico(df, y_col, title, picos_df=None, coincidencias_df=None, height=550):
    """Genera un gráfico Plotly interactivo."""
    fig = go.Figure()
    
    # 1. Trazar el Espectro
    fig.add_trace(go.Scatter(
        x=df['Longitud_Onda'], 
        y=df[y_col], 
        mode='lines', 
        line=dict(width=1.5, color='blue'), 
        name=title.split('(')[0].strip() # Nombre corto para leyenda
    ))

    # 2. Añadir picos detectados (sin identificación)
    if picos_df is not None and not picos_df.empty:
        fig.add_trace(go.Scatter(
            x=picos_df['Longitud_Onda'], 
            y=picos_df['Intensidad'], 
            mode='markers', 
            name='Picos detectados',
            marker=dict(color='red', size=6, symbol='circle'),
            hoverinfo='text',
            hovertext=picos_df.apply(lambda row: f"Pico: {row['Longitud_Onda']:.3f} nm<br>I: {row['Intensidad']:.2f}", axis=1),
        ))

    # 3. Añadir las coincidencias (Identificación por Elemento y Etiqueta)
    if coincidencias_df is not None and not coincidencias_df.empty:
        
        # Agrupar las coincidencias para procesar por elemento
        for elemento in coincidencias_df['Elemento'].unique():
            # Usar .copy() para evitar SettingWithCopyWarning
            df_elem = coincidencias_df[coincidencias_df['Elemento'] == elemento].copy()
            color = COLORES_ELEMENTOS.get(elemento, 'black') # Asignar color

            # Crear etiqueta de texto
            df_elem.loc[:, 'Etiqueta_Linea'] = df_elem.apply(
                lambda row: f"{row['Elemento']} ({row['Linea_Atomica']:.3f})", axis=1
            )
            
            # Traza de los marcadores de coincidencia
            fig.add_trace(go.Scatter(
                x=df_elem['Longitud_Onda_Pico'], 
                y=df_elem['Intensidad_Pico'], 
                mode='markers+text', 
                name=f'{elemento} Identificado',
                marker=dict(size=10, symbol='star', color=color, line=dict(width=1, color='white')),
                text=df_elem['Etiqueta_Linea'], # Mostrar etiqueta
                textposition="top center",
                textfont=dict(color=color, size=10),
                hoverinfo='text',
                hovertext=df_elem.apply(
                    lambda row: f"Pico: {row['Longitud_Onda_Pico']:.3f} nm<br>Línea NIST: {row['Linea_Atomica']:.3f} nm<br>Elemento: {row['Elemento']}", 
                    axis=1
                ),
                showlegend=True
            ))
            
            # Líneas verticales (referencia de la línea NIST)
            for _, row in df_elem.iterrows():
                fig.add_vline(x=row['Linea_Atomica'], line_width=1, line_dash="dash", line_color=color, opacity=0.4)

    # 4. Configuración del Layout
    fig.update_layout(
        title=title,
        xaxis_title='Longitud de Onda (nm)',
        yaxis_title='Intensidad',
        hovermode="x unified", 
        height=height,
        legend_title="Elementos/Trazas"
    )
    return fig

def main():
    st.set_page_config(page_title="Analizador LIBS", layout="wide")
    
    st.title("🔬 Analizador de Espectros LIBS")
    st.markdown("Análisis de espectros con corrección de fondo (Filtro Savitzky-Golay), detección de picos y líneas NIST.")
    
    # Sidebar para controles
    with st.sidebar:
        st.header("Configuración")
        
        archivo_subido = st.file_uploader("Subir archivo ASC", type=['asc'])
        
        if archivo_subido:
            st.success(f"Archivo cargado: {archivo_subido.name}")
        
            st.subheader("Corrección de Fondo (Savitzky-Golay)")
            window_length = st.slider(
                "Longitud de Ventana (impar): Suavizado", 
                21, 501, 201, 2, # Empieza en 21, termina en 501, default 201, paso de 2 (impar)
                help="Tamaño de la ventana del filtro SG (debe ser impar). Un valor más alto suaviza más agresivamente, dejando solo la curva de fondo."
            )
            polyorder = st.slider("Orden del Polinomio", 1, 5, 3, 1)
            
            st.subheader("Detección de Picos")
            altura_minima = st.slider("Altura mínima de picos (relativa a Imax Corregida)", 0.0, 1.0, 0.1, 0.01)
            distancia_picos = st.slider("Distancia entre picos (en puntos de datos)", 1, 20, 5)
            
            st.subheader("Búsqueda de Elementos")
            default_elements = ['H', 'Na', 'Ca', 'Cd']
            default_selection = [e for e in default_elements if e in LINEAS_ATOMICAS]
            
            elementos_seleccionados = st.multiselect(
                "Elementos a buscar",
                list(LINEAS_ATOMICAS.keys()),
                default=default_selection
            )
            tolerancia = st.slider("Tolerancia (nm)", 0.1, 2.0, 0.5, 0.1)
    
    # Contenido principal
    if archivo_subido:
        espectro = cargar_archivo_asc(archivo_subido)
        
        if espectro is not None:
            # 1. Aplicar corrección de fondo (Savitzky-Golay)
            baseline, intensidad_corregida = corregir_fondo(
                espectro['Intensidad_Original'].values, 
                window_length,
                polyorder
            )
            espectro = espectro.copy()
            espectro['Intensidad_Corregida'] = intensidad_corregida
            espectro['Fondo_Calculado'] = baseline
            
            # 2. Detección de picos
            picos_corregidos = encontrar_picos(espectro, 'Intensidad_Corregida', altura_minima, distancia_picos)
            picos_originales = encontrar_picos(espectro, 'Intensidad_Original', altura_minima, distancia_picos)
            
            # 3. Búsqueda de elementos (siempre con picos corregidos)
            df_coincidencias = pd.DataFrame()
            if elementos_seleccionados and len(picos_corregidos) > 0:
                todas_coincidencias = []
                for elemento in elementos_seleccionados:
                    coincidencias = buscar_coincidencias(picos_corregidos, elemento, tolerancia)
                    if not coincidencias.empty:
                        todas_coincidencias.append(coincidencias)
                
                if todas_coincidencias:
                    df_coincidencias = pd.concat(todas_coincidencias, ignore_index=True)
            
            
            # --- SECCIÓN DE VISUALIZACIÓN DE GRÁFICOS (ANCHO COMPLETO) ---
            st.subheader("Visualización de Espectros")
            
            # Gráfico 1: Espectro Original
            st.markdown("### 1. Espectro Original (Picos sin Corregir)")
            fig_orig = generar_grafico(
                espectro, 
                'Intensidad_Original', 
                'Espectro Original', 
                picos_df=picos_originales, 
                coincidencias_df=None 
            )
            st.plotly_chart(fig_orig, width='stretch')
            
            # Gráfico 2: Espectro Corregido
            st.markdown("### 2. Espectro Procesado (Fondo Corregido Savitzky-Golay) y Análisis")
            fig_corr = generar_grafico(
                espectro, 
                'Intensidad_Corregida', 
                'Espectro Corregido (Savitzky-Golay)', 
                picos_df=picos_corregidos, 
                coincidencias_df=df_coincidencias
            )
            st.plotly_chart(fig_corr, width='stretch')
            
            
            # --- SECCIÓN DE RESULTADOS Y ESTADÍSTICAS ---
            st.subheader("Resultados de Análisis")
            col_stats, col_resumen = st.columns([1, 2])
            
            with col_stats:
                st.metric("Picos Detectados (Corregido)", len(picos_corregidos))
                st.metric("Intensidad Máxima Original", f"{espectro['Intensidad_Original'].max():.2f}")
                st.metric("Intensidad Máxima Corregida", f"{espectro['Intensidad_Corregida'].max():.2f}")
                
                if len(picos_corregidos) > 0:
                    st.markdown("**Top 5 Picos (Corregidos)**")
                    st.dataframe(picos_corregidos[['Longitud_Onda', 'Intensidad']].nlargest(5, 'Intensidad').round(3), width='stretch')

            with col_resumen:
                if not df_coincidencias.empty:
                    st.markdown("### Resumen por Elemento")
                    resumen = df_coincidencias.groupby('Elemento').agg({
                        'Longitud_Onda_Pico': 'count',
                        'Intensidad_Pico': 'mean'
                    }).rename(columns={'Longitud_Onda_Pico': 'Número_Líneas_Coincidencia', 
                                     'Intensidad_Pico': 'Intensidad_Promedio_Pico_Corregida'})
                    st.dataframe(resumen.round(3), width='stretch')
                    
                    st.markdown("### Coincidencias Detalladas")
                    df_mostrar = df_coincidencias.sort_values(['Elemento', 'Longitud_Onda_Pico'])
                    st.dataframe(df_mostrar[['Elemento', 'Longitud_Onda_Pico', 'Intensidad_Pico', 'Linea_Atomica', 'Diferencia']].round(4), width='stretch')
                else:
                    st.warning("No se encontraron coincidencias con los elementos seleccionados en el espectro corregido.")
            
            # --- SECCIÓN DE DESCARGA ---
            st.subheader("💾 Exportar Resultados")
            if len(picos_corregidos) > 0:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    espectro[['Longitud_Onda', 'Intensidad_Original', 'Fondo_Calculado', 'Intensidad_Corregida']].to_excel(writer, sheet_name='Espectro_Completo', index=False)
                    picos_corregidos.to_excel(writer, sheet_name='Picos_Detectados', index=False)
                    if not df_coincidencias.empty:
                        df_coincidencias.to_excel(writer, sheet_name='Elementos_Identificados', index=False)
                
                st.download_button(
                    label="📥 Descargar Resultados (Excel)",
                    data=output.getvalue(),
                    file_name=f"analisis_libs_{archivo_subido.name.split('.')[0]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    else:
        # Página de bienvenida
        st.info("👆 Sube un archivo ASC en la barra lateral para comenzar el análisis")
        st.subheader("Elementos disponibles:")
        elementos = list(LINEAS_ATOMICAS.keys())
        cols_per_row = 4
        col_elems = st.columns(cols_per_row)
        for i, elem in enumerate(elementos):
            with col_elems[i % cols_per_row]:
                st.metric(f"{elem}", f"{len(LINEAS_ATOMICAS[elem])} líneas")

if __name__ == "__main__":
    main()