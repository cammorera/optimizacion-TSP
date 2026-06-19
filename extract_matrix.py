"""
Script auxiliar: Extrae la matriz de distancias del archivo Excel
y la guarda como JSON para ser usada por el solver y la app.

Uso:
    python extract_matrix.py --input Matriz_distancias_Desde-Hasta.xlsx
"""

import argparse
import json
import openpyxl


def extract_matrix(xlsx_path: str, output_path: str = "distance_matrix.json"):
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))

    # Cabecera: primer elemento es etiqueta, resto son IDs de nodo
    header = list(rows[0])
    nodes = [h for h in header[1:] if h is not None]

    distances = {}
    for row in rows[1:]:
        if row[0] is None:
            continue
        from_node = row[0]
        for j, node in enumerate(nodes):
            val = row[j + 1]
            if val is not None:
                distances[(from_node, node)] = int(val)

    dist_serializable = {str(k): v for k, v in distances.items()}
    data = {"nodes": nodes, "distances": dist_serializable}

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Matriz extraída: {len(nodes)} nodos, {len(distances)} distancias")
    print(f"   Guardada en: {output_path}")
    return nodes, distances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="Matriz_distancias_Desde-Hasta.xlsx")
    parser.add_argument("--output", default="distance_matrix.json")
    args = parser.parse_args()
    extract_matrix(args.input, args.output)
