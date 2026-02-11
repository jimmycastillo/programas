import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from scipy.signal import savgol_filter

st.set_page_config(page_title="CQD Nano-Analyzer Pro", layout="wide")

# --- FUNCIONES DE SOPORTE ---
def leer_spectra(file):
    try:
        content = file.getvalue().decode("utf-8").splitlines()
        data_start = 0
        for i, line in enumerate(content):
            if ">>>>>Begin" in line:
                data_start = i + 1
                break
        df = pd.read_csv(io.StringIO("\n".join(content[data_start:])), 
                         sep=r'\s+', names=['W', 'I'], engine='python', on_bad_lines='skip')
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        return df
    except Exception as e:
        st.error(f"Error al leer {file.name}: {e}")
        return None

def save_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    return buf.getvalue()

# --- INTERFAZ ---
st.title("🧪 CQD Nano-Analyzer Pro")
st.markdown("Herramienta avanzada para el análisis de Carbon Quantum Dots y Quenching.")

tabs = st.tabs(["📉 Fluorescencia & S-V", "🔍 Inspección Libre", "🌓 Abs vs Fluo"])

# ==========================================
# PESTAÑA 1: FLUORESCENCIA Y STERN-VOLMER
# ==========================================
with tabs[0]:
    st.sidebar.header("Parámetros Stern-Volmer")
    v_ini = st.sidebar.number_input("Volumen inicial CQD (uL)", value=2000)
    c_sto = st.sidebar.number_input("Conc. Stock Quencher (ppm)", value=100)
    
    col_f1, col_f2 = st.columns([1, 2])
    
    with col_f1:
        st.subheader("1. Carga de Datos")
        fl_files = st.file_uploader("Subir archivos .txt (Fluo)", accept_multiple_files=True, key="flu_sv")
        v_adds = {}
        if fl_files:
            for f in fl_files:
                v_adds[f.name] = st.number_input(f"uL añadidos en {f.name}", value=0, key=f"v_sv_{f.name}")

    if fl_files:
        espectros = []
        for f in fl_files:
            df = leer_spectra(f)
            if df is not None:
                # Procesamiento
                df['I_smooth'] = savgol_filter(df['I'], 31, 3)
                v_tot = v_ini + v_adds[f.name]
                f_corr = v_tot / v_ini
                df['I_corr'] = df['I_smooth'] * f_corr
                conc = (c_sto * v_adds[f.name]) / v_tot
                espectros.append({"name": f.name, "df": df, "conc": conc})

        with col_f2:
            st.subheader("2. Visualización y Ajuste de Escalas")
            
            # Controles de escala
            c1, c2, c3 = st.columns(3)
            with c1:
                xlim = st.slider("Escala X (Wavelength)", 200.0, 900.0, (400.0, 650.0), key="xlim_sv")
            with c2:
                ylim = st.slider("Escala Y (Intensidad)", 0, 10000, (0, 5000), key="ylim_sv")
            with c3:
                w_target = st.number_input("Wavelength de trabajo (nm)", value=450.0)

            fig_sv, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            datos_sv = []
            for e in espectros:
                ax1.plot(e['df']['W'], e['df']['I_corr'], label=f"{e['conc']:.1f} ppm")
                # Valor para SV
                val = e['df'].iloc[(e['df']['W'] - w_target).abs().argsort()[:5]]['I_corr'].mean()
                datos_sv.append({"Q": e['conc'], "I": val})

            ax1.set_xlim(xlim); ax1.set_ylim(ylim)
            ax1.set_title("Espectros Corregidos"); ax1.legend(fontsize='small')
            
            # Stern-Volmer
            df_sv = pd.DataFrame(datos_sv).sort_values("Q")
            f0 = df_sv.iloc[0]['I']
            df_sv['f0_f'] = f0 / df_sv['I']
            
            ax2.scatter(df_sv['Q'], df_sv['f0_f'], color='red', s=100, edgecolors='k')
            m, b = np.polyfit(df_sv['Q'], df_sv['f0_f'], 1)
            ax2.plot(df_sv['Q'], m*df_sv['Q'] + b, 'r--', label=f'Ksv: {m:.4f}\nR²: {np.corrcoef(df_sv["Q"], df_sv["I0_I" if "I0_I" in df_sv else "f0_f"])[0,1]**2:.4f}')
            ax2.set_title("Gráfica de Stern-Volmer"); ax2.set_xlabel("Conc (ppm)"); ax2.set_ylabel("F0/F"); ax2.legend()
            
            st.pyplot(fig_sv)
            st.download_button("💾 Guardar Análisis Completo", save_plot(fig_sv), "analisis_sv.png")

# ==========================================
# PESTAÑA 2: INSPECCIÓN LIBRE (COMPARATIVOS)
# ==========================================
with tabs[1]:
    st.subheader("Inspección de Espectros Independientes")
    st.info("Carga cualquier archivo para compararlos sin corrección de volumen.")
    
    col_i1, col_i2 = st.columns([1, 3])
    
    with col_i1:
        ins_files = st.file_uploader("Cargar archivos para comparar", accept_multiple_files=True, key="ins_files")
    
    if ins_files:
        fig_ins, ax_ins = plt.subplots(figsize=(10, 5))
        for f in ins_files:
            df_ins = leer_spectra(f)
            if df_ins is not None:
                df_ins['I_s'] = savgol_filter(df_ins['I'], 31, 3)
                ax_ins.plot(df_ins['W'], df_ins['I_s'], label=f.name)
        
        with col_i2:
            ix = st.slider("Escala X", 200.0, 900.0, (350.0, 750.0), key="ix")
            iy = st.slider("Escala Y", -100, 10000, (0, 4000), key="iy")
            ax_ins.set_xlim(ix); ax_ins.set_ylim(iy)
            ax_ins.legend(loc='upper right', fontsize='x-small')
            st.pyplot(fig_ins)
            st.download_button("💾 Guardar Comparativa", save_plot(fig_ins), "comparativa_libre.png")

# ==========================================
# PESTAÑA 3: DUAL (ABSORBANCIA VS FLUORESCENCIA)
# ==========================================
with tabs[2]:
    st.subheader("Relación de Propiedades Ópticas")
    ca, cf = st.columns(2)
    with ca: f_abs = st.file_uploader("Cargar Absorbancia", key="dual_abs")
    with cf: f_flu = st.file_uploader("Cargar Fluorescencia", key="dual_flu")
    
    if f_abs and f_flu:
        df_a = leer_spectra(f_abs)
        df_f = leer_spectra(f_flu)
        
        fig_dual, ax_a = plt.subplots(figsize=(10, 5))
        
        # Eje Absorbancia (Izquierda)
        ax_a.plot(df_a['W'], df_a['I'], color='blue', lw=2, label='Absorbancia')
        ax_a.set_ylabel("Absorbancia", color='blue', fontsize=12)
        ax_a.tick_params(axis='y', labelcolor='blue')
        
        # Eje Fluorescencia (Derecha)
        ax_f = ax_a.twinx()
        ax_f.plot(df_f['W'], df_f['I'], color='red', lw=2, label='Fluorescencia')
        ax_f.set_ylabel("Fluorescencia (u.a.)", color='red', fontsize=12)
        ax_f.tick_params(axis='y', labelcolor='red')
        
        # Escalas Duales
        st.divider()
        c_d1, c_d2, c_d3 = st.columns(3)
        with c_d1: dx = st.slider("Rango Wavelength", 200.0, 900.0, (300.0, 700.0), key="dx")
        with c_d2: dy_a = st.slider("Escala Absorbancia", 0.0, 3.0, (0.0, 1.0), key="dya")
        with c_d3: dy_f = st.slider("Escala Fluorescencia", 0, 10000, (0, 5000), key="dyf")
        
        ax_a.set_xlim(dx); ax_a.set_ylim(dy_a)
        ax_f.set_ylim(dy_f)
        
        plt.title("Caracterización Óptica: Absorbancia vs Emisión")
        st.pyplot(fig_dual)
        st.download_button("💾 Guardar Gráfico Dual", save_plot(fig_dual), "caracterizacion_dual.png")