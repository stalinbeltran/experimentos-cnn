#!/usr/bin/env python3
"""Las pruebas de `dim-nist` que no entrenan nada. Sin pytest: `python nn/probar.py` y un codigo."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos                                   # noqa: E402
import modelo                                  # noqa: E402

PARAMETROS_ESPERADOS = {8: 1258, 7: 1258, 6: 754, 5: 754, 4: 394}   # C=8, calculados el 2026-10-02
MAPAS_ESPERADOS = {8: [5, 2], 7: [4, 1], 6: [4, 2], 5: [3, 1], 4: [3, 2]}
fallos = 0


def prueba(que, bien, det=""):
    global fallos
    fallos += 0 if bien else 1
    print(f"  [{'ok' if bien else 'FALLA':>5}] {que}" + (f"  {det}" if det else ""))


def main() -> int:
    print("dato")
    d = datos.crudo()
    x, y = d["imagenes"], d["etiquetas"]
    for W in datos.W_TODOS:
        A = datos.matriz(W)
        prueba(f"W={W}: filas de A suman 8 y huella congelada",
               bool((A.sum(1) == 8).all()) and datos._huella_i8(datos.sumas(x, W)) == datos.HUELLAS[W])
    prueba("W=4 es la media de bloques 2x2", bool((datos.matriz(4) == np.repeat(np.eye(4, dtype=np.int64), 2, axis=1) * 4).all()))
    prueba("W=8 es la imagen / 16", np.allclose(datos.fraccion(datos.sumas(x, 8))[:, 0], x / 16.0))
    prueba("fraccion en [0,1]", float(datos.fraccion(datos.sumas(x, 5)).max()) <= 1.0)
    # un peso equivocado (p. ej. truncar en vez de redondear el solape) tiene que dar OTRA huella
    A_mal = datos.matriz(7).copy(); A_mal[0, 0] -= 1; A_mal[0, 1] += 1
    S_mal = np.einsum("ij,njk,lk->nil", A_mal, x.astype(np.int64), A_mal)
    prueba("una matriz de pesos distinta se detecta (otra huella a W=7)", datos._huella_i8(S_mal) != datos.HUELLAS[7])
    c = datos.cargar(6)
    prueba("cargar(6): 180 train / 1617 val, formas", c["x_train"].shape == (180, 1, 6, 6) and c["x_val"].shape == (1617, 1, 6, 6))
    cc = datos.cargar(8, control=True)
    prueba("cargar(8, control): forma de 8 y huella de 4", cc["x_train"].shape == (180, 1, 8, 8) and cc["huella_x"] == datos.HUELLAS[4])
    prueba("18 por clase en train", all((c["y_train"] == k).sum() == 18 for k in range(10)))
    prueba("piso = 1/10", datos.piso(c["y_val"]) == 0.1)
    prueba("la media se conserva en cada W (suma S = W²·suma x)",
           all(int(datos.sumas(x, W).sum()) == W * W * int(x.astype(np.int64).sum()) for W in datos.W_TODOS))

    print("modelo")
    for fila in modelo.tabla():
        W = fila["W"]
        prueba(f"W={W}: mapas {MAPAS_ESPERADOS[W]}", fila["mapas"] == MAPAS_ESPERADOS[W], str(fila["mapas"]))
        prueba(f"W={W}: parametros = tabla", fila["parametros"] == PARAMETROS_ESPERADOS[W], str(fila["parametros"]))
        prueba(f"W={W}: campo receptivo >= W-1", fila["campo_receptivo"] >= W - 1, str(fila["campo_receptivo"]))
    prueba("parsear w8-de4-s3", modelo.parsear("w8-de4-s3") == ("w8-de4", 8, True, 3))
    for malo in ("w3-s1", "w8", "w8-s", "w16-s1", "w8-de4"):
        try:
            modelo.parsear(malo); bien = False
        except ValueError:
            bien = True
        prueba(f"parsear se niega con '{malo}'", bien)
    import torch
    r1, r2, r3 = modelo.construir(6, 1), modelo.construir(6, 1), modelo.construir(6, 2)
    prueba("misma semilla -> misma inicializacion", all(torch.equal(p, q) for p, q in zip(r1.parameters(), r2.parameters())))
    prueba("otra semilla -> otra", not all(torch.equal(p, q) for p, q in zip(r1.parameters(), r3.parameters())))
    prueba("salida (N,10) en los cinco W", all(tuple(modelo.construir(W, 1)(torch.rand(3, 1, W, W)).shape) == (3, 10) for W in modelo.W_TODOS))
    print(f"\n{'todo en orden' if not fallos else f'✗ {fallos} fallo(s)'}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    raise SystemExit(main())
