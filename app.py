"""
Aplicación Streamlit - Optimización TSP
Empresa de Distribución de Paquetes - San José, Costa Rica

Caso: Redes - Camino más Corto y TSP
Universidad de Costa Rica - Ingeniería Industrial
II-1122 Optimización Industrial
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os

# Importar solver
from tsp_solver import (
    load_distance_matrix,
    solve_tsp_mtz,
    solve_tsp_nearest_neighbor,
    format_route,
)

# ─────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TSP Optimizer - UCR",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS personalizado
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 5px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card h2 { font-size: 2rem; margin: 0; font-weight: 700; }
    .metric-card p  { font-size: 0.85rem; opacity: 0.85; margin: 4px 0 0; }

    .route-box {
        background: #f0f7ff;
        border-left: 5px solid #1e6eb5;
        border-radius: 8px;
        padding: 15px 20px;
        font-size: 1rem;
        font-family: monospace;
        word-break: break-all;
        margin-top: 10px;
    }
    .section-title {
        color: #1e3a5f;
        border-bottom: 2px solid #2d6a9f;
        padding-bottom: 6px;
        margin-top: 30px;
    }
    .info-box {
        background: #e8f4fd;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #1a3a5c;
        font-size: 0.92rem;
    }
    .badge-optimal  { color: #155724; background: #d4edda; padding: 3px 10px; border-radius: 20px; font-weight: 600; }
    .badge-heuristic{ color: #856404; background: #fff3cd; padding: 3px 10px; border-radius: 20px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Carga de datos
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    path = os.path.join(base, "distance_matrix.json")
    nodes, distances = load_distance_matrix(path)
    return nodes, distances

nodes, distances = load_data()
all_customers = sorted([n for n in nodes if n != 0])

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://www.ucr.ac.cr/medios/fotos/2022/logoucr-1663268481.png", width=160)
    st.markdown("## ⚙️ Configuración del modelo")
    st.markdown("---")

    st.markdown("### 📍 Selección de clientes")
    num_customers = st.slider(
        "Número de clientes (aleatorio)",
        min_value=4, max_value=20, value=8, step=1,
        help="Se seleccionarán N clientes aleatoriamente de la red."
    )

    random_seed = st.number_input("Semilla aleatoria", min_value=0, max_value=999, value=42)

    st.markdown("---")
    st.markdown("### 🎯 Clientes manuales (opcional)")
    manual_mode = st.checkbox("Seleccionar clientes manualmente")
    selected_customers = []

    if manual_mode:
        selected_customers = st.multiselect(
            "Nodos cliente (excluye depósito 0)",
            options=all_customers,
            default=all_customers[:8],
            help="Seleccione los nodos a visitar."
        )
    else:
        rng = np.random.RandomState(int(random_seed))
        selected_customers = sorted(rng.choice(all_customers, size=num_customers, replace=False).tolist())
        st.info(f"Clientes seleccionados:\n`{selected_customers}`")

    st.markdown("---")
    st.markdown("### ⏱️ Solver")
    time_limit = st.slider("Tiempo límite CBC (s)", 30, 600, 120, step=30)
    run_heuristic = st.checkbox("Ejecutar también heurística NN", value=True)

    st.markdown("---")
    solve_btn = st.button("🚀 Resolver TSP", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
# Encabezado principal
# ─────────────────────────────────────────────
st.title("🚚 Optimización de Rutas — TSP")
st.markdown(
    "**Empresa de Distribución · San José, Costa Rica** &nbsp;|&nbsp; "
    "Problema del Agente Viajero (TSP) · Formulación MTZ con PuLP + CBC"
)

tab_model, tab_result, tab_analysis, tab_formulation = st.tabs([
    "📊 Modelo", "✅ Resultados", "📈 Análisis", "📐 Formulación"
])

# ─────────────────────────────────────────────
# TAB: MODELO
# ─────────────────────────────────────────────
with tab_model:
    st.markdown("<h3 class='section-title'>Red de clientes seleccionados</h3>", unsafe_allow_html=True)

    if not selected_customers:
        st.warning("Seleccione al menos un cliente en la barra lateral.")
    else:
        all_sel = [0] + selected_customers
        n = len(all_sel)

        col1, col2, col3 = st.columns(3)
        col1.metric("Depósito", "Nodo 0 (San José)")
        col2.metric("Clientes a visitar", len(selected_customers))
        col3.metric("Nodos totales", n)

        st.markdown("#### Matriz de distancias (subconjunto seleccionado)")
        mat = pd.DataFrame(index=all_sel, columns=all_sel, dtype=float)
        for i in all_sel:
            for j in all_sel:
                if i == j:
                    mat.loc[i, j] = 0
                else:
                    mat.loc[i, j] = distances.get((i, j), distances.get((j, i), None))

        st.dataframe(
            mat.astype("Int64"),
            use_container_width=True, height=350
        )

        st.markdown("#### Tamaño del modelo MTZ")
        n_vars_bin = n * (n - 1)
        n_vars_mtz = len(selected_customers)
        n_cons_out = n
        n_cons_in  = n
        n_cons_mtz = len(selected_customers) * (len(selected_customers) - 1)

        data_model = {
            "Componente": [
                "Variables binarias x_ij",
                "Variables continuas u_i (MTZ)",
                "Total variables",
                "Rest. salida única",
                "Rest. entrada única",
                "Rest. subciclos (MTZ)",
                "Total restricciones"
            ],
            "Cantidad": [
                n_vars_bin,
                n_vars_mtz,
                n_vars_bin + n_vars_mtz,
                n_cons_out,
                n_cons_in,
                n_cons_mtz,
                n_cons_out + n_cons_in + n_cons_mtz
            ]
        }
        st.table(pd.DataFrame(data_model))

# ─────────────────────────────────────────────
# TAB: RESULTADOS
# ─────────────────────────────────────────────
with tab_result:
    if not solve_btn:
        st.info("👈 Configure los parámetros en la barra lateral y presione **Resolver TSP**.")
    elif not selected_customers:
        st.error("Debe seleccionar al menos 2 clientes.")
    else:
        # ── Resolver MTZ ──
        with st.spinner("⏳ Resolviendo TSP (formulación MTZ)…"):
            result_mtz = solve_tsp_mtz(selected_customers, distances, time_limit)

        # ── Resolver heurística ──
        result_nn = None
        if run_heuristic:
            with st.spinner("⏳ Ejecutando heurística del vecino más cercano…"):
                result_nn = solve_tsp_nearest_neighbor(selected_customers, distances)

        # ── Guardar en session state ──
        st.session_state["result_mtz"] = result_mtz
        st.session_state["result_nn"]  = result_nn

        # ── Métricas MTZ ──
        st.markdown("<h3 class='section-title'>Solución óptima (MTZ)</h3>", unsafe_allow_html=True)

        badge = (
            '<span class="badge-optimal">✅ ÓPTIMO</span>'
            if result_mtz["status"] == "Optimal"
            else '<span class="badge-heuristic">⚠️ ' + result_mtz["status"] + '</span>'
        )
        st.markdown(f"**Estado:** {badge}", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""<div class="metric-card"><h2>{result_mtz['objective']:.1f}</h2><p>Distancia total (km)</p></div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class="metric-card"><h2>{result_mtz['solve_time']:.2f}s</h2><p>Tiempo de solución</p></div>""", unsafe_allow_html=True)
        c3.markdown(f"""<div class="metric-card"><h2>{result_mtz['num_variables']}</h2><p>Variables del modelo</p></div>""", unsafe_allow_html=True)
        c4.markdown(f"""<div class="metric-card"><h2>{result_mtz['num_constraints']}</h2><p>Restricciones</p></div>""", unsafe_allow_html=True)

        st.markdown("#### 🗺️ Ruta óptima")
        st.markdown(f'<div class="route-box">{format_route(result_mtz["route"])}</div>', unsafe_allow_html=True)

        # ── Detalle por tramo ──
        st.markdown("#### 📋 Detalle de tramos")
        if result_mtz["route_detail"]:
            df_detail = pd.DataFrame(result_mtz["route_detail"])
            df_detail.columns = ["Desde", "Hasta", "Distancia (km)"]
            df_detail.index = range(1, len(df_detail) + 1)
            st.dataframe(df_detail, use_container_width=True)

        # ── Comparación con heurística ──
        if result_nn:
            st.markdown("<h3 class='section-title'>Comparación MTZ vs Heurística NN</h3>", unsafe_allow_html=True)
            gap = (result_nn["objective"] - result_mtz["objective"]) / result_mtz["objective"] * 100 if result_mtz["objective"] else 0
            df_cmp = pd.DataFrame({
                "Método": ["MTZ (Óptimo)", "Vecino más cercano"],
                "Distancia total (km)": [result_mtz["objective"], result_nn["objective"]],
                "Tiempo (s)": [f"{result_mtz['solve_time']:.3f}", f"{result_nn['solve_time']:.4f}"],
                "Gap vs óptimo (%)": ["0.00%", f"{gap:.2f}%"],
                "Ruta": [format_route(result_mtz["route"]), format_route(result_nn["route"])]
            })
            st.dataframe(df_cmp, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# TAB: ANÁLISIS
# ─────────────────────────────────────────────
with tab_analysis:
    st.markdown("<h3 class='section-title'>Análisis de resultados</h3>", unsafe_allow_html=True)

    result_mtz = st.session_state.get("result_mtz")
    result_nn  = st.session_state.get("result_nn")

    if not result_mtz:
        st.info("Ejecute el solver primero en la pestaña **Resultados**.")
    else:
        st.markdown("### 1. Secuencia óptima de visitas")
        if result_mtz.get("route"):
            for i, (frm, to) in enumerate(zip(result_mtz["route"], result_mtz["route"][1:]), 1):
                seg_d = distances.get((frm, to), distances.get((to, frm), "?"))
                st.markdown(
                    f"**Paso {i}:** Nodo `{frm}` → Nodo `{to}` &nbsp; ({seg_d} km)"
                )

        st.markdown("---")
        st.markdown("### 2. Distancia total recorrida")
        st.markdown(f'<div class="info-box">🚚 <b>Distancia total MTZ:</b> {result_mtz["objective"]:.1f} km</div>', unsafe_allow_html=True)
        if result_nn:
            gap = (result_nn["objective"] - result_mtz["objective"]) / result_mtz["objective"] * 100
            st.markdown(f'<div class="info-box">🔶 <b>Heurística NN:</b> {result_nn["objective"]:.1f} km &nbsp;|&nbsp; Gap: {gap:.2f}% más largo que el óptimo</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 3. Tamaño del modelo generado")
        n = len(selected_customers) + 1
        st.markdown(f"""
| Componente | Valor |
|---|---|
| Nodos totales (incluyendo depósito) | **{n}** |
| Variables binarias x_ij | **{n*(n-1)}** |
| Variables MTZ u_i | **{n-1}** |
| Total variables | **{n*(n-1) + n - 1}** |
| Restricciones salida única | **{n}** |
| Restricciones entrada única | **{n}** |
| Restricciones MTZ | **{(n-1)*(n-2)}** |
| **Total restricciones** | **{2*n + (n-1)*(n-2)}** |
| Tiempo de solución | **{result_mtz['solve_time']:.2f} s** |
""")

        st.markdown("---")
        st.markdown("### 4. Limitaciones para instancias más grandes")
        st.markdown("""
- **Explosión combinatoria:** el TSP pertenece a la clase NP-hard; el número de rutas posibles crece como *(n-1)!/2*.
- **MTZ escala polinomialmente en restricciones** (O(n²)) pero el relajado LP puede ser débil para n > 20-30.
- **Tiempo de solución:** el CBC puede tardar horas o no converger para n > 15-20 nodos sin buenas cotas.
- **Memoria:** para n = 50, se tienen ≈ 2,450 variables binarias y ≈ 2,450 restricciones MTZ.
- **Simetría:** sin romper simetrías, el branch-and-bound explora demasiados nodos equivalentes.
""")

        st.markdown("---")
        st.markdown("### 5. Mejoras posibles")
        st.markdown("""
- **Formulación DFJ (Dantzig-Fulkerson-Johnson):** cotas LP más fuertes que MTZ, aunque requiere generación dinámica de restricciones.
- **Branch-and-Cut:** añadir cortes Subtour-Elimination dinámicamente solo cuando aparecen subciclos.
- **Metaheurísticas:** algoritmos genéticos, Simulated Annealing, Ant Colony Optimization para instancias grandes.
- **2-opt / 3-opt:** mejora local post-solución heurística; reduce el gap respecto al óptimo.
- **Column Generation / VRPTW:** extensiones al problema real con ventanas de tiempo y múltiples vehículos.
- **Solvers comerciales:** Gurobi o CPLEX ofrecen heurísticas internas y cortes más agresivos.
""")

# ─────────────────────────────────────────────
# TAB: FORMULACIÓN MATEMÁTICA
# ─────────────────────────────────────────────
with tab_formulation:
    st.markdown("<h3 class='section-title'>Formulación Matemática del TSP (MTZ)</h3>", unsafe_allow_html=True)

    st.markdown("""
## Parámetros

| Símbolo | Descripción |
|---|---|
| $N$ | Conjunto de todos los nodos $\\{0, 1, \\ldots, n-1\\}$ |
| $0$ | Depósito (centro logístico en San José) |
| $C = N \\setminus \\{0\\}$ | Conjunto de clientes |
| $d_{ij}$ | Distancia del arco $(i,j)$, $\\forall i,j \\in N, i \\neq j$ |
| $n$ | Número total de nodos |

---

## Variables de decisión

$$
x_{ij} \\in \\{0, 1\\}, \\quad \\forall i, j \\in N, \\; i \\neq j
$$

$x_{ij} = 1$ si el vehículo viaja directamente del nodo $i$ al nodo $j$; $0$ en caso contrario.

$$
u_i \\in [1, n-1], \\quad \\forall i \\in C
$$

$u_i$ es el orden de visita del cliente $i$ en la ruta (variable auxiliar MTZ para eliminar subciclos).

---

## Función objetivo

$$
\\min \\sum_{i \\in N} \\sum_{j \\in N, j \\neq i} d_{ij} \\cdot x_{ij}
$$

Minimizar la distancia total recorrida.

---

## Restricciones

### (1) Salida única de cada nodo

$$
\\sum_{j \\in N, j \\neq i} x_{ij} = 1, \\quad \\forall i \\in N
$$

Cada nodo tiene exactamente un arco saliente activo.

### (2) Entrada única a cada nodo

$$
\\sum_{i \\in N, i \\neq j} x_{ij} = 1, \\quad \\forall j \\in N
$$

Cada nodo tiene exactamente un arco entrante activo.

### (3) Eliminación de subciclos — MTZ

$$
u_i - u_j + n \\cdot x_{ij} \\leq n - 1, \\quad \\forall i, j \\in C, \\; i \\neq j
$$

Garantiza que no se formen subciclos en la solución (Miller, Tucker & Zemlin, 1960).

### (4) Dominio de variables

$$
x_{ij} \\in \\{0, 1\\}, \\quad \\forall i,j \\in N, \\; i \\neq j
$$

$$
1 \\leq u_i \\leq n - 1, \\quad u_i \\in \\mathbb{R}^+, \\quad \\forall i \\in C
$$

---

## Verificación de correctitud de MTZ

Si existiera un subciclo $S \\subset C$ que no incluya el depósito, sumando las restricciones MTZ
sobre ese ciclo se obtendría $0 \\leq -|S|$, una contradicción. Por tanto, **la formulación MTZ
elimina todos los subciclos correctamente**.
""")
