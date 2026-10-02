#!/usr/bin/env python3
"""Las pruebas de `dim-gen` que no entrenan nada (R14: lo que puede fallar en silencio, se
prueba). Sin pytest a proposito: `python nn/probar.py` y un codigo de salida.

  - el dato: huellas congeladas, suma conservada, W=128 == recorte, etiquetas en [0,1],
    un acumulador uint16 CAE, el control tiene la forma y los bloques que dice
  - el modelo: mapa final L x L en los cinco W, parametros = la tabla de ESPECIFICACION.md §3.2,
    el parseo de brazos se niega con lo que no es un brazo
  - el IoU: identicas 1, disjuntas 0, invertida 0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos                                   # noqa: E402
import modelo                                  # noqa: E402

# La tabla de ESPECIFICACION.md §3.2 (f=0,25, C=8), calculada el 2026-10-01.
PARAMETROS_ESPERADOS = {128: 205348, 64: 51748, 32: 13348, 16: 3748, 8: 1348}

fallos = 0


def prueba(que: str, bien: bool, det: str = "") -> None:
    global fallos
    fallos += 0 if bien else 1
    print(f"  [{'ok' if bien else 'FALLA':>5}] {que}" + (f"  {det}" if det else ""))


def main() -> int:
    print("dato")
    d = datos.crudo()
    c = datos.recortar(d["imagenes"])
    for W in datos.W_TODOS:
        s = datos.sumas(c, W)
        prueba(f"huella congelada W={W}", datos._huella_i8(s) == datos.HUELLAS[W])
    prueba("W=128 es el recorte", bool((datos.sumas(c, 128) == c).all()))
    y = datos.etiquetas_norm(d["etiquetas"])
    prueba("etiquetas en [0,1] y huella", float(y.min()) >= 0 and float(y.max()) <= 1
           and datos._huella_f8(y) == datos.HUELLA_ETIQUETAS)
    b = 128 // 16
    mal = c.astype(np.uint16).reshape(len(c), 16, b, 16, b).sum(axis=(2, 4), dtype=np.uint16)
    prueba("un acumulador uint16 a W=16 da OTRA huella (se detecta)",
           datos._huella_i8(mal.astype(np.int64)) != datos.HUELLAS[16])
    x16 = datos.fraccion(datos.sumas(c, 16), 16)
    ctl = datos.control_de16(x16)
    prueba("control: forma (N,1,128,128) y bloques 8x8 uniformes",
           ctl.shape == (len(c), 1, 128, 128) and bool((ctl[:, :, 3::8, 5::8] == x16).all()))
    prueba("fraccion de tinta en [0,1]", float(x16.min()) >= 0 and float(x16.max()) <= 1)
    cargado = datos.cargar(32)
    prueba("cargar(32): 100 train / 900 val, formas", cargado["x_train"].shape == (100, 1, 32, 32)
           and cargado["x_val"].shape == (900, 1, 32, 32) and cargado["y_val"].shape == (900, 4))
    cargado_c = datos.cargar(128, control=True)
    prueba("cargar(128, control): forma de 128 y huella de 16",
           cargado_c["x_train"].shape == (100, 1, 128, 128) and cargado_c["huella_x"] == datos.HUELLAS[16])
    piso = datos.piso_caja_media(cargado["y_train"], cargado["y_val"])
    prueba("piso de caja media = 0,2464 (medido 2026-10-01)", abs(piso - 0.2464) < 0.001, f"{piso:.4f}")

    print("IoU")
    a = np.array([[0.1, 0.5, 0.2, 0.6]]); b2 = np.array([[0.6, 0.9, 0.7, 0.9]])
    inv = np.array([[0.5, 0.1, 0.2, 0.6]])
    prueba("identicas -> 1", abs(datos.iou(a, a)[0] - 1) < 1e-9)
    prueba("disjuntas -> 0", datos.iou(a, b2)[0] == 0)
    prueba("invertida -> 0", datos.iou(inv, a)[0] == 0)
    prueba("fuera de [0,1] se recorta", datos.iou(np.array([[-1, 2, -1, 2]]), np.array([[0, 1, 0, 1]]))[0] == 1)

    print("modelo")
    for fila in modelo.tabla():
        W = fila["W"]
        prueba(f"W={W}: mapa final {modelo.L}x{modelo.L}", fila["mapas"][-1] == modelo.L, str(fila["mapas"]))
        prueba(f"W={W}: parametros = tabla", fila["parametros"] == PARAMETROS_ESPERADOS[W],
               f"{fila['parametros']} (esperados {PARAMETROS_ESPERADOS[W]})")
    prueba("parsear w128-de16-s3", modelo.parsear("w128-de16-s3") == ("w128-de16", 128, True, 3))
    prueba("parsear w008-s5", modelo.parsear("w008-s5") == ("w008", 8, False, 5))
    for malo in ("w096-s1", "w128", "w128-s", "x128-s1", "w128-de16"):
        try:
            modelo.parsear(malo); bien = False
        except ValueError:
            bien = True
        prueba(f"parsear se niega con '{malo}'", bien)
    import torch
    r1, r2 = modelo.construir(16, 1), modelo.construir(16, 1)
    prueba("misma semilla -> misma inicializacion",
           all(torch.equal(p, q) for p, q in zip(r1.parameters(), r2.parameters())))
    r3 = modelo.construir(16, 2)
    prueba("otra semilla -> otra inicializacion",
           not all(torch.equal(p, q) for p, q in zip(r1.parameters(), r3.parameters())))

    print(f"\n{'todo en orden' if not fallos else f'✗ {fallos} fallo(s)'}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    raise SystemExit(main())
