import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import ast
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO

# Fungsi evaluasi aman untuk input user dengan penanganan error yang lebih baik
def safe_eval(expr, var='p', t=0):
    try:
        allowed_names = {
            'np': np,
            'sqrt': np.sqrt,
            'log': np.log,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'exp': np.exp,
            'pi': np.pi,
            var: None, 
            't': None
        }
        
        # Parse expression menjadi AST
        code = compile(expr, "<string>", "eval", flags=ast.PyCF_ONLY_AST)
        
        # Validasi node yang diizinkan
        for node in ast.walk(code):
            if isinstance(node, ast.Name):
                if node.id not in allowed_names:
                    raise ValueError(f"Nama variabel/fungsi tidak diizinkan: {node.id}")
            elif isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in allowed_names:
                    raise ValueError(f"Fungsi tidak diizinkan")
                    
        # Buat fungsi dengan penanganan error
        def f(p, t_val=0):
            try:
                return eval(expr, {
                    "np": np, 
                    "p": p, 
                    "t": t_val, 
                    "sqrt": np.sqrt, 
                    "log": np.log, 
                    "sin": np.sin, 
                    "cos": np.cos, 
                    "tan": np.tan,
                    "exp": np.exp,
                    "pi": np.pi
                })
            except Exception as e:
                st.error(f"Error saat mengevaluasi fungsi: {e}")
                return np.nan
                
        return f
    except Exception as e:
        st.error(f"Error dalam ekspresi: {e}")
        return lambda p, t_val=0: np.nan

# Metode Newton-Raphson yang lebih robust
def newton_raphson(f, df, p0, t=0, tol=1e-6, max_iter=100):
    p = p0
    for _ in range(max_iter):
        try:
            fp = f(p, t)
            if np.isnan(fp):
                return None
                
            dfp = df(p, t)
            if np.isnan(dfp):
                return None
                
            if abs(dfp) < 1e-12:  # Tolerance lebih kecil untuk derivatif
                return None
                
            p_new = p - fp / dfp
            
            # Penanganan nilai divergen
            if abs(p_new) > 1e10:
                return None
                
            if abs(p_new - p) < tol:
                return p_new
                
            p = p_new
        except:
            return None
    return None

# --- Streamlit UI ---
st.set_page_config(page_title="Simulasi Dinamika Pasar", page_icon="📈", layout="wide")
st.title("📈 Simulasi Dinamika Pasar Multi-Produk")

# Sidebar untuk input utama
with st.sidebar:
    st.header("⚙️ Parameter Simulasi")
    num_products = st.number_input("Jumlah Produk", 1, 5, 2, help="Pilih jumlah produk yang akan disimulasikan (1-5)")
    sim_time = st.slider("Durasi Simulasi (waktu)", 1, 50, 10, help="Durasi simulasi dalam satuan waktu")
    time_step = st.slider("Resolusi Waktu", 0.1, 1.0, 0.5, help="Interval waktu antara titik simulasi")
    st.markdown("---")
    st.info("ℹ️ Gunakan fungsi matematika standar (sin, cos, exp, log, sqrt) dengan variabel 'p' untuk harga dan 't' untuk waktu.")

# Input produk
products = []
for i in range(num_products):
    with st.expander(f"🛒 Produk {i+1}", expanded=(i==0)):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input(f"Nama Produk {i+1}", value=f"Produk {chr(65+i)}", key=f"name_{i}")
            d_expr = st.text_input(
                f"Fungsi Permintaan {name}", 
                value="100 - 10*p + 5*sin(t)", 
                key=f"d_expr_{i}",
                help="Contoh: 100 - 10*p + 5*sin(t)"
            )
            tax = st.slider(f"Pajak {name}", 0.0, 20.0, 1.0, key=f"tax_{i}")
            
        with col2:
            s_expr = st.text_input(
                f"Fungsi Penawaran {name}", 
                value="20 + 5*p", 
                key=f"s_expr_{i}",
                help="Contoh: 20 + 5*p + 2*cos(t)"
            )
            subsidy = st.slider(f"Subsidi {name}", 0.0, 20.0, 0.0, key=f"subsidy_{i}")
            shock_amp = st.slider(
                f"Amplitudo Shock Permintaan {name}", 
                0.0, 50.0, 10.0, 
                key=f"shock_{i}",
                help="Amplitudo shock musiman pada permintaan"
            )
        
        # Parse fungsi
        d_func = safe_eval(d_expr)
        s_func = safe_eval(s_expr)
        
        products.append({
            "name": name,
            "demand": d_func,
            "supply": s_func,
            "tax": tax,
            "subsidy": subsidy,
            "shock": shock_amp,
            "demand_expr": d_expr,
            "supply_expr": s_expr
        })

# Simulasi waktu
time_points = np.arange(0, sim_time, time_step)

# Jalankan simulasi untuk setiap produk
for prod in products:
    prices = []
    demands = []
    supplies = []
    tax_effects = []
    
    for t in time_points:
        # Fungsi ekuilibrium dengan shock dan kebijakan
        def f(p, t_val=0):
            demand = prod["demand"](p, t_val) + prod["shock"] * np.sin(t_val)
            supply = prod["supply"](p + prod["tax"] - prod["subsidy"], t_val)
            return demand - supply
            
        # Derivatif numerik
        def df(p, t_val=0):
            h = 1e-5
            return (f(p + h, t_val) - f(p - h, t_val)) / (2 * h)
            
        # Cari ekuilibrium
        p_eq = newton_raphson(f, df, 5.0, t)
        
        if p_eq is not None:
            demand_val = prod["demand"](p_eq, t) + prod["shock"] * np.sin(t)
            supply_val = prod["supply"](p_eq + prod["tax"] - prod["subsidy"], t)
            tax_effect = p_eq + prod["tax"] - prod["subsidy"]
        else:
            demand_val = np.nan
            supply_val = np.nan
            tax_effect = np.nan
            
        prices.append(p_eq)
        demands.append(demand_val)
        supplies.append(supply_val)
        tax_effects.append(tax_effect)
    
    # Simpan hasil
    prod.update({
        "prices": prices,
        "demands": demands,
        "supplies": supplies,
        "tax_effects": tax_effects
    })

# Visualisasi hasil
tab1, tab2, tab3 = st.tabs(["📊 Grafik Harga", "📈 Permintaan & Penawaran", "📋 Data"])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for prod in products:
        ax1.plot(time_points, prod["prices"], '.-', label=f'{prod["name"]} (Harga Konsumen)')
        ax1.plot(time_points, prod["tax_effects"], ':', label=f'{prod["name"]} (Harga Produsen)')
    
    ax1.set_title("Dinamika Harga Ekuilibrium")
    ax1.set_xlabel("Waktu")
    ax1.set_ylabel("Harga")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for prod in products:
        ax2.plot(time_points, prod["demands"], '.-', label=f'Permintaan {prod["name"]}')
        ax2.plot(time_points, prod["supplies"], '.-', label=f'Penawaran {prod["name"]}')
    
    ax2.set_title("Dinamika Permintaan dan Penawaran")
    ax2.set_xlabel("Waktu")
    ax2.set_ylabel("Kuantitas")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig2)

with tab3:
    st.subheader("Data Simulasi")
    for prod in products:
        with st.expander(f"Data {prod['name']}"):
            df = pd.DataFrame({
                'Waktu': time_points,
                'Harga Konsumen': prod["prices"],
                'Harga Produsen': prod["tax_effects"],
                'Permintaan': prod["demands"],
                'Penawaran': prod["supplies"],
                'Pajak': [prod["tax"]] * len(time_points),
                'Subsidi': [prod["subsidy"]] * len(time_points)
            })
            st.dataframe(df.style.format("{:.2f}"), use_container_width=True)

# Fungsi untuk generate PDF yang lebih informatif
def generate_pdf(products, time_points):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=(595.2756, 841.8898))  # Ukuran A4 dalam points (210x297mm)
    styles = getSampleStyleSheet()
    elements = []
    
    # Judul
    elements.append(Paragraph("Laporan Simulasi Dinamika Pasar", styles['Title']))
    elements.append(Spacer(1, 24))
    
    # Ringkasan parameter
    elements.append(Paragraph("Parameter Simulasi", styles['Heading2']))
    param_data = [
        ["Jumlah Produk", str(len(products))],
        ["Durasi Simulasi", f"{float(time_points[-1]):.1f} satuan waktu"],
        ["Resolusi Waktu", f"{float(time_points[1]-time_points[0]):.2f}"]
    ]
    param_table = Table(param_data, colWidths=[200, 100])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    elements.append(param_table)
    elements.append(Spacer(1, 24))
    
    # Detail produk
    for prod in products:
        elements.append(Paragraph(f"Produk: {prod['name']}", styles['Heading3']))
        
        # Fungsi
        func_data = [
            ["Fungsi Permintaan", prod["demand_expr"]],
            ["Fungsi Penawaran", prod["supply_expr"]],
            ["Pajak", f"{float(prod['tax']):.2f}"],
            ["Subsidi", f"{float(prod['subsidy']):.2f}"],
            ["Shock Permintaan", f"{float(prod['shock']):.2f}"]
        ]
        func_table = Table(func_data, colWidths=[150, 300])
        func_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        elements.append(func_table)
        
        # Data 
        sample_data = [["Waktu", "Harga", "Permintaan", "Penawaran"]]
        for t, p, d, s in zip(time_points[:5], prod["prices"][:5], prod["demands"][:5], prod["supplies"][:5]):
            sample_data.append([
                f"{float(t):.1f}", 
                f"{float(p):.2f}" if not np.isnan(p) else "NaN", 
                f"{float(d):.2f}" if not np.isnan(d) else "NaN", 
                f"{float(s):.2f}" if not np.isnan(s) else "NaN"
            ])
        
        sample_table = Table(sample_data, colWidths=[80, 80, 80, 80])
        sample_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Data:", styles['Normal']))
        elements.append(sample_table)
        elements.append(Spacer(1, 24))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Unduh laporan
st.subheader("📤 Ekspor Hasil Simulasi")
col1, col2 = st.columns(2)

with col1:
    if st.button("🔄 Jalankan Simulasi Ulang"):
        st.rerun()

with col2:
    pdf_buffer = generate_pdf(products, time_points)
    st.download_button(
        "📥 Unduh Laporan PDF", 
        data=pdf_buffer, 
        file_name="laporan_simulasi_pasar.pdf", 
        mime="application/pdf",
        help="Unduh laporan lengkap dalam format PDF"
    )