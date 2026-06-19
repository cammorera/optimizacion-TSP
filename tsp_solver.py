"""
TSP Solver - Traveling Salesman Problem
Formulación MTZ (Miller-Tucker-Zemlin) para eliminación de subciclos.

Caso: Optimización de Rutas para una Empresa de Distribución
Universidad de Costa Rica - Ingeniería Industrial
II-1122 Optimización Industrial
"""

import json
import time
from itertools import permutations
import pulp
import pandas as pd
import numpy as np


def load_distance_matrix(json_path: str) -> tuple[list, dict]:
    """Carga la matriz de distancias desde un archivo JSON."""
    with open(json_path) as f:
        data = json.load(f)
    nodes = data["nodes"]
    raw = data["distances"]
    distances = {eval(k): v for k, v in raw.items()}
    return nodes, distances


def build_subproblem(nodes: list, distances: dict, selected_nodes: list) -> dict:
    """
    Extrae la submatriz de distancias para los nodos seleccionados
    (depósito 0 + clientes).
    """
    sub = {}
    for i in selected_nodes:
        for j in selected_nodes:
            if i != j:
                key = (i, j)
                if key in distances:
                    sub[key] = distances[key]
                else:
                    sub[key] = 9999  # penalización si no existe
    return sub


def solve_tsp_mtz(selected_customers: list, distances: dict, time_limit: int = 300) -> dict:
    """
    Resuelve el TSP con formulación MTZ (Miller-Tucker-Zemlin).

    Parámetros del modelo:
    ----------------------
    - N  : conjunto de nodos {0, 1, ..., n-1} donde 0 es el depósito
    - d_ij: distancia/costo del arco (i, j)

    Variables de decisión:
    ----------------------
    - x_ij ∈ {0, 1}: 1 si se recorre el arco de i a j
    - u_i ∈ [1, n]: variable de orden de visita (MTZ), solo para clientes

    Función objetivo:
    -----------------
    Minimizar Σ_i Σ_j d_ij * x_ij

    Restricciones:
    --------------
    1. Salida única de cada nodo:   Σ_j x_ij = 1   ∀ i ∈ N
    2. Entrada única a cada nodo:   Σ_i x_ij = 1   ∀ j ∈ N
    3. Eliminación de subciclos (MTZ):
       u_i - u_j + n * x_ij ≤ n - 1   ∀ i,j ∈ N\{0}, i ≠ j
    """
    depot = 0
    all_nodes = [depot] + selected_customers
    n = len(all_nodes)

    # Índices locales (0-based para MTZ)
    node_idx = {node: i for i, node in enumerate(all_nodes)}
    idx_node = {i: node for node, i in node_idx.items()}

    # Submatriz de distancias
    d = {}
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                d[(i, j)] = distances.get((i, j), distances.get((j, i), 9999))

    # -----------------------------------------------
    # Modelo PuLP
    # -----------------------------------------------
    model = pulp.LpProblem("TSP_MTZ", pulp.LpMinimize)

    # Variables binarias x_ij
    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
        for i in all_nodes
        for j in all_nodes
        if i != j
    }

    # Variables de posición MTZ (solo clientes, no depósito)
    u = {
        i: pulp.LpVariable(f"u_{i}", lowBound=1, upBound=n - 1, cat="Continuous")
        for i in selected_customers
    }

    # -----------------------------------------------
    # Función objetivo: minimizar distancia total
    # -----------------------------------------------
    model += pulp.lpSum(d[(i, j)] * x[(i, j)] for i in all_nodes for j in all_nodes if i != j), "Total_Distance"

    # -----------------------------------------------
    # Restricción 1: Salida única de cada nodo
    # -----------------------------------------------
    for i in all_nodes:
        model += (
            pulp.lpSum(x[(i, j)] for j in all_nodes if i != j) == 1,
            f"Salida_unica_{i}"
        )

    # -----------------------------------------------
    # Restricción 2: Entrada única a cada nodo
    # -----------------------------------------------
    for j in all_nodes:
        model += (
            pulp.lpSum(x[(i, j)] for i in all_nodes if i != j) == 1,
            f"Entrada_unica_{j}"
        )

    # -----------------------------------------------
    # Restricción 3: Eliminación de subciclos (MTZ)
    # u_i - u_j + n * x_ij <= n - 1   ∀ i,j clientes, i≠j
    # -----------------------------------------------
    for i in selected_customers:
        for j in selected_customers:
            if i != j:
                model += (
                    u[i] - u[j] + n * x[(i, j)] <= n - 1,
                    f"MTZ_{i}_{j}"
                )

    # -----------------------------------------------
    # Resolver
    # -----------------------------------------------
    start = time.time()
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    status = model.solve(solver)
    elapsed = time.time() - start

    # -----------------------------------------------
    # Extraer solución
    # -----------------------------------------------
    result = {
        "status": pulp.LpStatus[model.status],
        "objective": pulp.value(model.objective),
        "solve_time": elapsed,
        "num_variables": len(model.variables()),
        "num_constraints": len(model.constraints),
        "route": [],
        "route_detail": [],
        "x_values": {}
    }

    if model.status == 1:  # Optimal
        # Reconstruir ruta desde el depósito
        active_arcs = {
            (i, j): 1
            for (i, j) in x
            if pulp.value(x[(i, j)]) is not None and pulp.value(x[(i, j)]) > 0.5
        }

        # Reconstruir secuencia
        route = [depot]
        current = depot
        visited = {depot}
        for _ in range(n - 1):
            for j in all_nodes:
                if j not in visited and (current, j) in active_arcs:
                    route.append(j)
                    visited.add(j)
                    current = j
                    break
        route.append(depot)  # regresar al depósito

        # Calcular distancias tramo por tramo
        route_detail = []
        for k in range(len(route) - 1):
            seg_dist = d.get((route[k], route[k + 1]), 0)
            route_detail.append({
                "desde": route[k],
                "hasta": route[k + 1],
                "distancia": seg_dist
            })

        result["route"] = route
        result["route_detail"] = route_detail
        result["x_values"] = {str(k): v for k, v in active_arcs.items()}

    return result


def solve_tsp_nearest_neighbor(selected_customers: list, distances: dict) -> dict:
    """
    Heurística del vecino más cercano (solución rápida de referencia).
    Útil para comparar con el óptimo.
    """
    depot = 0
    all_nodes = [depot] + selected_customers
    d = {}
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                d[(i, j)] = distances.get((i, j), distances.get((j, i), 9999))

    start = time.time()
    route = [depot]
    unvisited = set(selected_customers)
    current = depot

    while unvisited:
        nearest = min(unvisited, key=lambda j: d.get((current, j), 9999))
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    route.append(depot)
    elapsed = time.time() - start

    total_dist = sum(d.get((route[k], route[k + 1]), 0) for k in range(len(route) - 1))
    route_detail = [
        {"desde": route[k], "hasta": route[k + 1], "distancia": d.get((route[k], route[k + 1]), 0)}
        for k in range(len(route) - 1)
    ]

    return {
        "status": "Heuristic",
        "objective": total_dist,
        "solve_time": elapsed,
        "route": route,
        "route_detail": route_detail,
        "num_variables": 0,
        "num_constraints": 0
    }


def format_route(route: list) -> str:
    """Formatea la ruta como cadena legible."""
    return " → ".join(str(n) for n in route)


if __name__ == "__main__":
    # Prueba rápida con los primeros 8 clientes
    nodes, distances = load_distance_matrix("distance_matrix.json")
    customers = [n for n in nodes if n != 0][:8]
    print(f"Resolviendo TSP con depósito 0 y clientes: {customers}")

    result = solve_tsp_mtz(customers, distances, time_limit=120)
    print(f"\nEstado: {result['status']}")
    print(f"Distancia total: {result['objective']:.1f} km")
    print(f"Tiempo: {result['solve_time']:.2f} s")
    print(f"Variables: {result['num_variables']}")
    print(f"Restricciones: {result['num_constraints']}")
    print(f"Ruta: {format_route(result['route'])}")
