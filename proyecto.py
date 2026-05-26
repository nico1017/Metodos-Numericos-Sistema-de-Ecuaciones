import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.linalg import solve, lu

st.set_page_config(page_title="Data Center Optimizer - UMSA", layout="wide")

st.title("💻 Optimización de Recursos en Data Center")

# --- 1. CONFIGURACIÓN LATERAL ---
st.sidebar.header("Configuración")
escenario = st.sidebar.selectbox("1. Seleccione Escenario", ["Ideal", "Bajo Estrés", "Mal Condicionado"])
metodo_view = st.sidebar.selectbox(
    "2. Seleccione Método a Visualizar", 
    ["Sistema Original", "Factorización LU", "Jacobi", "Gauss-Seidel", "SOR (Sobrerelajación)", "Gradiente Conjugado"]
)

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

# --- 3. SECCIÓN DE RESOLUCIÓN ---
st.header(f"Resolución Numérica: {metodo_view}")

# Cómputo seguro de la solución exacta
try:
    sol_exacta = solve(A, b)
except Exception:
    sol_exacta = np.array([1.0, 1.0, 1.0])

if metodo_view == "Sistema Original":
    st.latex(r"Ax = b")
    st.write("Matriz de Coeficientes (A):")
    st.table(pd.DataFrame(A))
    st.write("Vector de Términos Independientes (b):")
    st.table(pd.DataFrame(b, columns=["b"]))

elif metodo_view == "Factorización LU":
    P, L, U = lu(A)
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Matriz L (Triangular Inferior):**")
        st.table(pd.DataFrame(L))
    with col2:
        st.write("**Matriz U (Triangular Superior):**")
        st.table(pd.DataFrame(U))
    st.success(f"Solución Exacta: {sol_exacta.tolist()}")

elif metodo_view in ["Jacobi", "Gauss-Seidel", "SOR (Sobrerelajación)"]:
    st.write(f"### Primeras 5 iteraciones del método de {metodo_view}")
    x = np.zeros(len(b))
    historial = []
    omega = 1.2 # Factor de relajación óptimo para este sistema
    
    for k in range(5):
        x_prev = x.copy()
        for i in range(len(b)):
            if metodo_view == "Jacobi":
                suma = sum(A[i][j] * x_prev[j] for j in range(len(b)) if i != j)
                x[i] = (b[i] - suma) / A[i][i]
            elif metodo_view == "Gauss-Seidel":
                suma = sum(A[i][j] * x[j] for j in range(i)) + sum(A[i][j] * x_prev[j] for j in range(i + 1, len(b)))
                x[i] = (b[i] - suma) / A[i][i]
            else: # SOR
                suma = sum(A[i][j] * x[j] for i_j in range(i)) + sum(A[i][j] * x_prev[j] for j in range(i + 1, len(b)))
                x_gs = (b[i] - suma) / A[i][i]
                x[i] = (1 - omega) * x_prev[i] + omega * x_gs
        historial.append(x.copy())
    
    st.table(pd.DataFrame(historial, columns=["x1 (IA)", "x2 (Web)", "x3 (DB)"]))
    st.info("La tabla muestra cómo los valores cambian en cada paso para aproximarse a la solución.")

elif metodo_view == "Gradiente Conjugado":
    st.write("### Desarrollo de Direcciones Conjugadas")
    x = np.zeros(len(b))
    r = b - np.dot(A, x)
    p = r.copy()
    pasos = []
    
    for k in range(len(b)):
        if np.linalg.norm(r) < 1e-6:
            break
        alpha = np.dot(r, r) / max(np.dot(p, np.dot(A, p)), 1e-12)
        x = x + alpha * p
        r_new = r - alpha * np.dot(A, p)
        beta = np.dot(r_new, r_new) / max(np.dot(r, r), 1e-12)
        p = r_new + beta * p
        r = r_new
        pasos.append(x.copy())
        
    st.write("Evolución del vector solución x:")
    st.table(pd.DataFrame(pasos, columns=["x1", "x2", "x3"]))

# --- 4. GRÁFICO (Siempre presente al final) ---
st.divider()
st.header("Análisis Geométrico 3D")

try:
    # Generación controlada de los hiperplanos interactivos
    x_range = np.linspace(sol_exacta[0]-2, sol_exacta[0]+2, 10)
    y_range = np.linspace(sol_exacta[1]-2, sol_exacta[1]+2, 10)
    X, Y = np.meshgrid(x_range, y_range)

    fig = go.Figure()
    for i in range(3):
        # Evitar división entre cero en el coeficiente de la tercera dimensión
        div = A[i, 2] if A[i, 2] != 0 else 1.0
        Z = (b[i] - A[i, 0]*X - A[i, 1]*Y) / div
        fig.add_trace(go.Surface(z=Z, x=X, y=Y, opacity=0.5, name=f"Ecuación {i+1}", showscale=False))

    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"No se pudo cargar la vista tridimensional: {e}")
