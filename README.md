# TSP Optimizer — Distribución de Paquetes, San José CR

**Caso:** Redes — Camino más Corto y TSP  
**Curso:** II-1122 Optimización Industrial  
**Universidad:** Universidad de Costa Rica — Ingeniería Industrial

---

## Descripción

Aplicación para resolver el Problema del Agente Viajero (TSP) sobre la red vial de San José,
Costa Rica. El depósito central corresponde al nodo 0; el vehículo debe visitar un conjunto
de clientes exactamente una vez y regresar al depósito minimizando la distancia total.

### Formulación utilizada

**MTZ (Miller-Tucker-Zemlin)**  
- Variables binarias xᵢⱼ ∈ {0,1}: 1 si se recorre el arco (i→j)  
- Variables de posición uᵢ ∈ [1, n-1]: orden de visita (elimina subciclos)

**Restricciones:**
1. Salida única de cada nodo: Σⱼ xᵢⱼ = 1 ∀i
2. Entrada única a cada nodo: Σᵢ xᵢⱼ = 1 ∀j
3. Eliminación de subciclos (MTZ): uᵢ - uⱼ + n·xᵢⱼ ≤ n-1 ∀i,j ∈ Clientes, i≠j

**Solver:** PuLP + CBC (gratuito y de código abierto)

---

## Estructura del proyecto

```
tsp_project/
├── app.py                    # Aplicación Streamlit
├── tsp_solver.py             # Solver MTZ + heurística NN
├── extract_matrix.py         # Extrae matriz desde .xlsx
├── distance_matrix.json      # Matriz de distancias (generada)
├── requirements.txt          # Dependencias Python
└── README.md                 # Este archivo
```

---

## Instalación y uso

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Extraer la matriz de distancias (si se parte del .xlsx)
```bash
python extract_matrix.py --input Matriz_distancias_Desde-Hasta.xlsx
```

### 3. Ejecutar la aplicación Streamlit
```bash
streamlit run app.py
```

### 4. Usar el solver directamente (desde Python)
```python
from tsp_solver import load_distance_matrix, solve_tsp_mtz

nodes, distances = load_distance_matrix("distance_matrix.json")
customers = [21, 41, 65, 76, 107, 146, 196, 250]  # ejemplo

result = solve_tsp_mtz(customers, distances, time_limit=120)
print(f"Distancia total: {result['objective']} km")
print(f"Ruta: {' → '.join(str(n) for n in result['route'])}")
```

---

## Despliegue en Streamlit Community Cloud

1. Subir el proyecto a un repositorio GitHub (público o privado).
2. Ir a [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Seleccionar repositorio, rama `main`, archivo `app.py`.
4. Asegurarse de que `distance_matrix.json` esté incluido en el repo.
5. Click en **Deploy**.

---

## Datos de entrada

La red incluye **90 nodos** (nodo 0 = depósito San José) con distancias
precalculadas mediante caminos más cortos (Dijkstra) sobre la red vial real.

Rango de distancias: 2 – 86 km  
Fuente: Matriz proporcionada por el curso II-1122

---

## Resultados típicos (8 clientes)

| Métrica | Valor |
|---|---|
| Variables binarias | 72 |
| Variables MTZ | 8 |
| Restricciones | 74 |
| Tiempo solución | < 2 s |
| Calidad | Óptimo global |

---

## Referencias

- Miller, C. E., Tucker, A. W., & Zemlin, R. A. (1960). Integer Programming Formulation of Traveling Salesman Problems. *Journal of the ACM*, 7(4), 326–329.
- PuLP documentation: https://coin-or.github.io/pulp/
- Streamlit documentation: https://docs.streamlit.io
