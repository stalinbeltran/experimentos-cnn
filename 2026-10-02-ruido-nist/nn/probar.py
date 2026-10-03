#!/usr/bin/env python3
"""Las pruebas de `ruido-nist` que no entrenan de verdad. Sin pytest: `python nn/probar.py` y un código.
(La de «ruido nulo da los pesos de limpio bit a bit» entrena 2 épocas: está en `entrenar_local.py --comprobar`.)"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos                                   # noqa: E402
import modelo                                  # noqa: E402
import ruido                                   # noqa: E402

PARAMETROS_ESPERADOS = 1338     # 80 + 584 + 584 + 90, calculado el 2026-10-02
MAPAS_ESPERADOS = [6, 4, 2]
fallos = 0


def prueba(que, bien, det=""):
    global fallos
    fallos += 0 if bien else 1
    print(f"  [{'ok' if bien else 'FALLA':>5}] {que}" + (f"  {det}" if det else ""))


def main() -> int:
    print("ruido: nombres y semillas")
    prueba("9 tipos, 5 niveles cada uno, el medio es el índice 2",
           len(ruido.TIPOS) == 9 and all(len(ruido.NIVELES[t]) == 5 for t in ruido.TIPOS) and ruido.INDICE_MEDIO == 2)
    prueba("niveles crecientes en todos los tipos", all(list(ruido.NIVELES[t]) == sorted(ruido.NIVELES[t]) for t in ruido.TIPOS))
    prueba("semilla 1000 + 10·t + i: borrado@0.05 → 1000, sal-pimienta@0.3 → 1084, oblicua@0.6-r2 → 6042",
           ruido.semilla("borrado", 0.05) == 1000 and ruido.semilla("sal-pimienta", 0.3) == 1084 and ruido.semilla("oblicua", 0.6, 2) == 6042)
    semillas = {ruido.semilla(t, n, r) for t in ruido.TIPOS for n in ruido.NIVELES[t] for r in (1, 2)}
    prueba("las 90 semillas de ruido son distintas", len(semillas) == 90)
    prueba("parsear('oblicua@0.6-r2') y el nombre canónico ida y vuelta",
           ruido.parsear("oblicua@0.6-r2") == ("oblicua", 0.6, 2) and ruido.escenario("oblicua", 0.6, 2) == "oblicua@0.6-r2"
           and ruido.escenario("limpio", None) == "limpio" and ruido.parsear("limpio") == ("limpio", None, 1))
    prueba("-linea: parsear, es_linea, fijo_de y el nombre canónico",
           ruido.parsear("recorte@0.6-linea") == ("recorte", 0.6, 1) and ruido.es_linea("recorte@0.6-linea") and not ruido.es_linea("recorte@0.6")
           and ruido.fijo_de("recorte@0.6-linea") == "recorte@0.6" and ruido.escenario("recorte", 0.6, 1, linea=True) == "recorte@0.6-linea")
    for malo in ("horizontal", "horizontal@0.7", "limpio-linea", "recorte@0.6-linea-r2", "limpio@0.6", "limpio-r2", "nada@0.6", "oblicua@0.6-r0", "Horizontal@0.6"):
        try:
            ruido.parsear(malo); bien = False
        except ValueError:
            bien = True
        prueba(f"parsear se niega con '{malo}'", bien)

    print("ruido: qué hace cada tipo")
    d = datos.limpio()
    cu = d["cuentas_train"]
    x = cu.astype(np.float32) / 16
    for tipo in ruido.TIPOS:
        s = ruido.semilla(tipo, ruido.NIVELES[tipo][0])
        nulo = ruido.aplicar(tipo, 0.0, cu, np.random.default_rng(s))
        prueba(f"{tipo:<13} nivel 0 == original, bit a bit", np.array_equal(nulo, x))
        a = ruido.aplicar(tipo, ruido.NIVELES[tipo][ruido.INDICE_MEDIO], cu, np.random.default_rng(s))
        b = ruido.aplicar(tipo, ruido.NIVELES[tipo][ruido.INDICE_MEDIO], cu, np.random.default_rng(s))
        c = ruido.aplicar(tipo, ruido.NIVELES[tipo][ruido.INDICE_MEDIO], cu, np.random.default_rng(s + 1))
        prueba(f"{tipo:<13} misma semilla → misma copia; otra semilla → otra; en [0,1] float32",
               np.array_equal(a, b) and not np.array_equal(a, c) and a.dtype == np.float32 and a.min() >= 0 and a.max() <= 1)
    rng = np.random.default_rng(0)
    bor = ruido.aplicar("borrado", 0.4, cu, rng)
    prueba("borrado sólo QUITA tinta (nunca sube un píxel) y deja el fondo a 0", bool((bor <= x + 1e-7).all()) and bool((bor[x == 0] == 0).all()))
    ext = ruido.aplicar("externos", 0.2, cu, rng)
    prueba("externos sólo AÑADE tinta (nunca baja un píxel)", bool((ext >= x - 1e-7).all()))
    prueba("externos en media añade ≈ p·(1 − x): p = 0,2 → +0,139 ± 0,01 sobre tinta media 0,305",
           abs(float((ext - x).mean()) - 0.2 * float((1 - x).mean())) < 0.01, f"{float((ext - x).mean()):.4f}")
    rec = ruido.aplicar("recorte", 1.0, cu, rng)
    prueba("recorte con α = 1 borra del todo un cuadrado (hay píxeles de tinta que pasan a 0) y no añade",
           bool((rec <= x + 1e-7).all()) and int(((x > 0) & (rec == 0)).sum()) > 0)
    hor = ruido.aplicar("horizontal", 1.0, cu, rng)
    dif = hor - x
    filas_tocadas = (np.abs(dif) > 1e-6).any(axis=2)                       # (N, 8): qué filas cambia
    prueba("horizontal toca 1–2 filas (a 32 px, 1–2 px de grosor: ≤ 4 filas de 8) y en ellas no baja la tinta",
           bool((filas_tocadas.sum(1) <= 4).all()) and bool((filas_tocadas.sum(1) >= 1).all()) and bool((dif >= -1e-6).all()))
    ver = ruido.aplicar("vertical", 1.0, cu, rng)
    cols = (np.abs(ver - x) > 1e-6).any(axis=1)
    prueba("vertical toca 1–4 columnas", bool((cols.sum(1) <= 4).all()) and bool((cols.sum(1) >= 1).all()))
    m = ruido.mascaras("oblicua", 50, np.random.default_rng(3))
    prueba("oblicua: la máscara a 32 px cubre ≤ 20 % y no es ni fila ni columna pura",
           float(m.mean()) < 0.2 and not any((mm.sum(0) == 32).any() or (mm.sum(1) == 32).any() for mm in m))
    cob = ruido.cobertura(m)
    prueba("cobertura = bits del bloque / 16 (múltiplos de 1/16, en [0,1])",
           cob.shape == (50, 8, 8) and bool(np.allclose(cob * 16, np.round(cob * 16))) and float(cob.max()) <= 1)
    gau = ruido.aplicar("gaussiano", 0.3, cu, rng)
    prueba("gaussiano recortado a [0,1] y con sd ≈ σ en los píxeles no saturados",
           gau.min() >= 0 and gau.max() <= 1 and 0.2 < float((gau - x)[(x > 0.2) & (x < 0.8)].std()) < 0.4)
    sp = ruido.aplicar("sal-pimienta", 0.1, cu, rng)
    cambiados = (np.abs(sp - x) > 1e-6).reshape(len(x), -1).sum(1)
    prueba("sal-pimienta cambia ≤ round(p·64) = 6 píxeles por imagen, a 0 ó 1",
           bool((cambiados <= 6).all()) and bool(np.isin(sp[np.abs(sp - x) > 1e-6], [0.0, 1.0]).all()))
    prueba("componer: α·c·(v − x) con α = 1, c = 1, v = 1 da 1; con v = 0 da 0",
           np.allclose(ruido.componer(x[:2], np.ones_like(x[:2]), 1.0, 1.0), 1) and np.allclose(ruido.componer(x[:2], np.ones_like(x[:2]), 1.0, 0.0), 0))

    print("datos: escenarios")
    e = datos.escenario("curva@0.6")
    prueba("escenario: (360,1,8,8) float32, y duplicadas, val (1617,1,8,8)",
           e["x_train"].shape == (360, 1, 8, 8) and e["x_train"].dtype == np.float32 and np.array_equal(e["y_train"][:180], e["y_train"][180:])
           and e["x_val"].shape == (1617, 1, 8, 8))
    prueba("las 180 primeras son las originales, las 180 siguientes la copia", np.array_equal(e["x_train"][:180], e["x_train_limpio"]) and e["huella_copia"] != e["huella_x_train_limpio"])
    prueba("val no pasa por el ruido (huella congelada en todos los escenarios)",
           all(datos.escenario(ruido.escenario(t, ruido.NIVELES[t][4]))["huella_x_val"] == datos.HUELLA_X_VAL for t in ruido.TIPOS[:3]))
    e2 = datos.escenario("curva@0.6-r2")
    prueba("la realización 2 es OTRA copia del mismo escenario", e2["huella_copia"] != e["huella_copia"] and e2["semilla_ruido"] == e["semilla_ruido"] + 5000)
    el = datos.escenario("recorte@0.6-linea")
    c1, c2, c3 = el["regenerar"](), el["regenerar"](), datos.escenario("recorte@0.6-linea")["regenerar"]()
    prueba("en línea: época 1 == copia fija; cada regenerar da otra copia; determinista entre llamadas",
           el["huella_copia"] == e["huella_copia"] if False else el["huella_copia"] == datos.escenario("recorte@0.6")["huella_copia"]
           and not np.array_equal(c1, el["x_train"][180:]) and not np.array_equal(c1, c2) and np.array_equal(c1, c3)
           and c1.shape == (180, 1, 8, 8) and c1.dtype == np.float32)
    prueba("escenario fijo no trae regenerar", e["regenerar"] is None and not e["linea"])
    prueba("piso = 1/10", datos.piso() == 0.1)

    print("modelo")
    import torch
    red = modelo.Red()
    prueba(f"mapas {MAPAS_ESPERADOS} y {PARAMETROS_ESPERADOS} parámetros", red.lados == MAPAS_ESPERADOS and red.n_parametros() == PARAMETROS_ESPERADOS, f"{red.lados} {red.n_parametros()}")
    prueba("salida (N,10)", tuple(red(torch.rand(3, 1, 8, 8)).shape) == (3, 10))
    r1, r2, r3 = modelo.nueva(1), modelo.nueva(1), modelo.nueva(2)
    prueba("nueva(1) dos veces: misma huella; nueva(2): otra", modelo.huella_pesos(r1) == modelo.huella_pesos(r2) != modelo.huella_pesos(r3))
    if modelo.HUELLAS_INIT.is_file():
        for s in modelo.SEMILLAS:
            try:
                ini = modelo.cargar_inicial(s)
                prueba(f"init-s{s}.pt carga y casa su huella ({modelo.huella_pesos(ini)})", True)
            except RuntimeError as err:
                prueba(f"init-s{s}.pt carga y casa su huella", False, str(err))
    else:
        prueba("nn/init/huellas.json existe (python nn/modelo.py --inicializar)", False)
    try:
        modelo.cargar_inicial(9); bien = False
    except RuntimeError:
        bien = True
    prueba("cargar_inicial se niega sin fichero", bien)
    print(f"\n{'todo en orden' if not fallos else f'✗ {fallos} fallo(s)'}")
    return 0 if not fallos else 1


if __name__ == "__main__":
    raise SystemExit(main())
