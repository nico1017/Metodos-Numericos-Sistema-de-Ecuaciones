import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.linalg import solve, lu, norm

st.set_page_config(page_title="Data Center Optimizer - UMSA", layout="wide")

st.title("💻 Optimización de Recursos en Data Center")
st.write("Análisis de estabilidad y eficiencia en sistemas de ecuaciones lineales de alta disponibilidad.")

# --- 1. CONFIGURACIÓN LATERAL ---
st.sidebar.header("⚙️ Configuración")
escenario = st.sidebar.selectbox("1. Seleccione Escenario", ["Ideal", "Bajo Estrés", "Mal Condicionado"])
metodo_view = st.sidebar.selectbox("2. Seleccione Método a Visualizar", 
                                  ["Sistema Original", "Factorización LU", "Jacobi", "Gauss-Seidel", "SOR (Sobrerelajación)", "Gradiente Conjugado"])

# Parámetro omega exclusivo para SOR
omega = 1.0
if metodo_view == "SOR (Sobrerelajación)":
    omega = st.sidebar.slider("Parámetro de Relajación (ω)", 0.1, 1.9, 1.2, step=0.05)

# --- 2. DEFINICIÓN DE MATRICES ---
if escenario == "Ideal":
    A = np.array([[10, 2, 1], [1, 10, 2], [2, 3, 10]], dtype=float)
    b = np.array([13, 13, 15], dtype=float)
elif escenario == "Bajo Estrés":
    A = np.array([[100, 50, 20], [50, 200, 30], [20, 30, 150]], dtype=float)
    b = np.array([500, 800, 400], dtype=float)
else: # Mal Condicionado
    A = np.array([[1, 0.99, 0.99], [0.99, 1, 0.99], [0.99, 0.99, 1]], dtype=float)
    b = np.array([2.98, 2.98, 2.98], dtype=float)

# Cálculo del número de condición para la sección de análisis
cond_A = np.linalg.cond(A)

# --- 3. SECCIÓN DE RESOLUCIÓN ---
st.header(f"🔍 Resolución Numérica: {metodo_view}")

if metodo_view == "Sistema Original":
    st.latex(r"Ax = b")
    col_mat1, col_mat2 = st.columns(2)
    with col_mat1:
        st.write("**Matriz de Coeficientes (A):**")
        st.dataframe(pd.DataFrame(A, columns=["x1 (IA)", "x2 (Web)", "x3 (DB)"]))
    with col_mat2:
        st.write("**Vector de Términos Independientes (b):**")
        st.dataframe(pd.DataFrame(b, columns=["b"]))
    
    # Muestra el rigor matemático del condicionamiento pedido por la rúbrica
    st.metric(label="Número de Condición κ(A)", value=f"{cond_A:.4f}")
    if cond_A > 100:
        st.error("⚠️ El sistema está mal condicionado. Pequeñas variaciones en los datos causarán grandes cambios en la solución.")
    else:
        st.success("✅ El sistema está bien condicionado. Los métodos iterativos convergerán rápidamente.")

elif metodo_view == "Factorización LU":
    P, L, U = lu(A)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Matriz L (Triangular Inferior):**")
        st.table(pd.DataFrame(L))
    with col2:
        st.write("**Matriz U (Triangular Superior):**")
        st.table(pd.DataFrame(U))
    st.success(f"🎯 Solución Exacta Calculada: {solve(A, b).tolist()}")

elif metodo_view in ["Jacobi", "Gauss-Seidel", "SOR (Sobrerelajación)"]:
    st.write(f"### Primeras 5 iteraciones del método")
    x = np.zeros(len(b))
    historial = []
    
    for k in range(5):
        x_prev = x.copy()
        for i in range(len(b)):
            if metodo_view == "Jacobi":
                suma = sum(A[i][j] * x_prev[j] for j in range(len(b)) if i != j)
                x[i] = (b[i] - suma) / A[i][i]
            elif metodo_view == "Gauss-Seidel":
                suma = sum(A[i][j] * x[j] for j in range(i)) + sum(A[i][j] * x_prev[j] for j in range(i + 1, len(b)))
                x[i] = (b[i] - suma) / A[i][i]
            elif metodo_view == "SOR (Sobrerelajación)":
                suma = sum(A[i][j] * x[j] for j in range(i)) + sum(A[i][j] * x_prev[j] for j in range(i + 1, len(b)))
                x_GS = (b[i] - suma) / A[i][i]
                x[i] = (1 - omega) * x_prev[i] + omega * x_GS
        historial.append(x.copy())
    
    st.table(pd.DataFrame(historial, columns=["x1 (IA)", "x2 (Web)", "x3 (DB)"]))
    st.info("La tabla detalla la aproximación paso a paso hacia el vector de estabilidad óptimo.")

elif metodo_view == "Gradiente Conjugado":
    st.write("### Desarrollo de Direcciones Conjugadas (Espacio de Krylov)")
    x = np.zeros(len(b))
    r = b - np.dot(A, x)
    p = r.copy()
    pasos = []
    
    for k in range(len(b)):
        alpha = np.dot(r, r) / np.dot(p, np.dot(A, p))
        x = x + alpha * p
        r_new = r - alpha * np.dot(A, p)
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
        pasos.append(x.copy())
        
    st.table(pd.DataFrame(pasos, columns=["x1 (IA)", "x2 (Web)", "x3 (DB)"]))

# --- 4. GRÁFICO 3D ---
st.divider()
st.header("📊 Análisis Geométrico de Hiperplanos 3D")
sol = solve(A, b)
x_range = np.linspace(sol[0]-2, sol[0]+2, 10)
y_range = np.linspace(sol[1]-2, sol[1]+2, 10)
X, Y = np.meshgrid(x_range, y_range)

fig = go.Figure()
for i in range(3):
    Z = (b[i] - A[i,0]*X - A[i,1]*Y) / A[i,2]
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, opacity=0.6, name=f"Ecuación {i+1}", showscale=False))

fig.update_layout(
    scene=dict(xaxis_title='x1 (IA)', yaxis_title='x2 (Web)', zaxis_title='x3 (DB)'),
    margin=dict(l=0, r=0, b=0, t=30)
)
st.plotly_chart(fig, use_container_width=True)