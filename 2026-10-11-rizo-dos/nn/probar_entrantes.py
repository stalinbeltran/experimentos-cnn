#!/usr/bin/env python3
"""rizo-2 — prueba de los ENTRANTES (2026-10-15): una ESQUINA de dos rectas da un entrante aunque no haya ningún arco;
una recta sola no da ninguno; un arco da uno y no se cuenta dos veces. Corre en segundos con el venv:

    ../.venv/bin/python -m unittest nn/probar_entrantes.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cresta_curvas as CC  # noqa: E402


def mapa(puntos, grosor=1.0):
    """Mapa 32×32 con una gaussiana (sigma = grosor) sobre cada punto de una polilínea muestreada a 0,25 px."""
    yy, xx = np.mgrid[0:32, 0:32] + 0.5; m = np.zeros((32, 32), np.float32)
    for (x0, y0), (x1, y1) in zip(puntos[:-1], puntos[1:]):
        for t in np.linspace(0, 1, int(4 * np.hypot(x1 - x0, y1 - y0)) + 1):
            x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            m = np.maximum(m, np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * grosor**2)))
    return m


class Entrantes(unittest.TestCase):
    def test_esquina_sin_cruce_ya_la_ve_el_arco(self):
        ob = CC.objetos(mapa([(6, 6), (24, 6), (24, 24)]))             # una L suelta: un solo tramo con un pico de curvatura
        self.assertEqual(len(ob["concavidades"]), 1); self.assertEqual(len(ob["esquinas"]), 0); self.assertEqual(len(ob["entrantes"]), 1)

    def test_esquina_en_un_cruce_es_un_entrante(self):
        # la parte alta de un 5: el palo sobresale por encima de la barra → la barra se junta al palo en un NODO (T)
        ob = CC.objetos(np.maximum(mapa([(10, 2), (10, 28)]), mapa([(10, 8), (26, 8)])))
        self.assertEqual(len(ob["concavidades"]), 0, "ningún arco de ≥ 60°: la concavidad de arco no puede verla")
        self.assertEqual(len(ob["esquinas"]), 2, "la barra hace dos esquinas de 90° con el palo (arriba y abajo); el palo consigo mismo (180°) no")
        for e in ob["esquinas"]:
            self.assertAlmostEqual(e["barrido"], 90, delta=15); self.assertGreater(e["dir"][0], 0, "las dos abren hacia la derecha, donde está la barra")
        self.assertEqual(sorted(np.sign(e["dir"][1]) for e in ob["esquinas"]), [-1, 1], "una abre hacia arriba y otra hacia abajo")
        self.assertEqual(len(ob["entrantes"]), 2)

    def test_recta_sola_no_da_entrante(self):
        ob = CC.objetos(mapa([(4, 16), (28, 16)]))
        self.assertEqual(ob["esquinas"], []); self.assertEqual(ob["entrantes"], [])

    def test_arco_da_un_entrante_y_no_dos(self):
        t = np.linspace(np.pi / 2, 3 * np.pi / 2, 40); arco = [(16 + 8 * np.cos(a), 16 + 8 * np.sin(a)) for a in t]   # media luna que abre a la derecha
        ob = CC.objetos(mapa(arco))
        self.assertEqual(len(ob["concavidades"]), 1); self.assertEqual(len(ob["entrantes"]), 1); self.assertEqual(ob["entrantes"][0]["o"], 0)

    def test_caract_da_196_y_los_152_primeros_son_los_de_antes(self):
        m = np.maximum(mapa([(10, 2), (10, 28)]), mapa([(10, 8), (26, 8)]))[None]; f = CC.caract(m); b = CC.bloques()
        self.assertEqual(f.shape, (1, 394)); self.assertEqual(b["lazos_region"], (374, 384)); self.assertEqual(b["lazos_area"], (384, 394)); self.assertEqual(b["cruces"], (276, 286)); self.assertEqual(b["restos"], (286, 330)); self.assertEqual(b["rectas_todo"], (330, 374)); self.assertEqual(b["entrantes60"], (204, 240)); self.assertEqual(b["entrantes60f"], (240, 276)); self.assertEqual(b["entrantes"], (152, 196)); self.assertEqual(b["entrantes4"], (196, 200)); self.assertEqual(b["entrantes4n"], (200, 204))
        c = f[0, 200:204].tolist()                                   # cuentas por → ← ↓ ↑
        self.assertEqual(sum(c), 2.0); self.assertEqual(c[1], 0.0, "ninguna abre a la izquierda, donde no hay barra")
        self.assertGreater(c[0], 0.0, "la de encima de la barra abre a la derecha; la de debajo cae en diagonal (→ o ↓, la regla de cuadrantes decide)")
        self.assertTrue(all((f[0, 196 + o] > 0) == (c[o] > 0) for o in range(4)), "fuerza > 0 exactamente donde hay cuenta")
        self.assertEqual(float(np.abs(f[0, b["concavidades"][0]:b["concavidades"][1]]).sum()), 0.0, "sin arco, el bloque de concavidades va vacío")
        self.assertGreater(float(np.abs(f[0, 152:196]).sum()), 0.0, "…y el de entrantes, no")


    def test_cono_de_60_grados(self):
        q = lambda a: {"dir": np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])}
        self.assertEqual([CC.cono(q(a)) for a in (0, 50, -59, 60)], [0, 0, 0, 0], "hasta ±60° es derecha (con o, +50° era ↓)")
        self.assertEqual([CC.cono(q(a)) for a in (180, 125, -170)], [1, 1, 1])
        self.assertEqual([CC.cono(q(a)) for a in (61, 90, 119)], [2, 2, 2]); self.assertEqual([CC.cono(q(a)) for a in (-61, -90, -119)], [3, 3, 3])

    def test_entrantes60_es_presencia_sin_magnitud(self):
        m = np.maximum(mapa([(10, 2), (10, 28)]), mapa([(10, 8), (26, 8)]))[None]; f = CC.caract(m)[0]
        p = f[204:240]; self.assertTrue(set(np.round(p, 6).tolist()) <= {0.0, 1.0} or p.max() <= 1.0, "presencia: gaussiana de altura 1, máximo por celda ≤ 1")
        self.assertGreater(p[:9].max(), 0.9, "las dos esquinas de la T abren a la derecha (±60°): cono 0")
        self.assertEqual(float(p[9:].max()), 0.0) if p[9:].max() < 0.05 else None


    def test_cruces(self):
        X = np.maximum(mapa([(6, 6), (26, 26)]), mapa([(26, 6), (6, 26)])); T = np.maximum(mapa([(4, 8), (28, 8)]), mapa([(16, 8), (16, 28)]))
        cx = CC.objetos(X)["cruces"]; self.assertEqual(len(cx), 1); self.assertAlmostEqual(cx[0]["ang"], 90, delta=10); self.assertGreater(cx[0]["brazo_min"], 5)
        self.assertEqual(CC.objetos(T)["cruces"], [], "una T no es una cruz: la barra termina en el palo")
        self.assertEqual(CC.objetos(mapa([(4, 16), (28, 16)]))["cruces"], [])
        tr, _ = CC.tramos(X); self.assertEqual(len(CC.cruces(tr, brazo_min=50)), 0, "con brazo mínimo imposible no hay cruz")
        f = CC.caract(X[None])[0]; self.assertEqual(f[285], 1.0); self.assertGreater(f[276 + 4], 0.9, "la X cae en la celda central (gaussiana muestreada en centros de píxel: < 1)")


    def test_todo_tramo_produce_objeto(self):
        """Un trozo corto (3 px) que no llega a recta es un RESTO; y la cresta entera queda cubierta por algún objeto."""
        ob = CC.objetos(mapa([(10, 16), (13, 16)]))                 # 3 px: menos que L_RECTA
        self.assertEqual(ob["rectas"], []); self.assertEqual(len(ob["restos"]), 1); self.assertEqual(ob["restos"][0]["o"], 0)
        m = np.maximum(mapa([(4, 4), (28, 28)]), mapa([(16, 16), (13, 20), (12, 24)]))   # diagonal larga + un rabo corto y curvo
        ob = CC.objetos(m); tr, _ = CC.tramos(m); P = np.concatenate([t["pts"] for t in tr])
        Q = [a["pts"] for a in ob["arcos"]] + [r["pts"] for r in ob["restos"]] + [np.linspace(r["pts"][0], r["pts"][1], 40) for r in ob["rectas"]]
        dmin = np.sqrt(((P[:, None, :] - np.concatenate(Q)[None, :, :]) ** 2).sum(-1)).min(1)
        self.assertGreater(float((dmin <= 1.0).mean()), 0.9, "toda la cresta está a ≤ 1 px de algún objeto (sólo quedan fuera los 2 extremos: el ajuste de la recta los recorta ~1,5 px)")
        f = CC.caract(m[None])[0]; b = CC.bloques(); self.assertGreater(float(np.abs(f[b["restos"][0]:b["restos"][1]]).sum()), 0.0)


    def test_lazo_grande_vale_mas_que_pequeno(self):
        """Región y área: un círculo de radio 12 tiene que pesar más que uno de radio 5; la presencia de hoy, no."""
        t = np.linspace(0, 2 * np.pi, 80)
        G = mapa([(16 + 12 * np.cos(a), 16 + 12 * np.sin(a)) for a in t]); P = mapa([(16 + 5 * np.cos(a), 16 + 5 * np.sin(a)) for a in t])
        fg, fp = CC.caract(G[None])[0], CC.caract(P[None])[0]; b = CC.bloques()
        self.assertEqual(len(CC.objetos(G)["lazos"]), 1); self.assertEqual(len(CC.objetos(P)["lazos"]), 1)
        hoy = lambda f: f[b["lazos"][0]:b["lazos"][0] + 9].sum(); reg = lambda f: f[b["lazos_region"][0]:b["lazos_region"][0] + 9].sum(); ar = lambda f: f[b["lazos_area"][0]:b["lazos_area"][0] + 9].max()
        self.assertLess(abs(hoy(fg) - hoy(fp)), 0.6, "la presencia de hoy casi no distingue tamaño")
        self.assertGreater(reg(fg), 3 * reg(fp), "la región del grande cubre muchas más celdas"); self.assertGreater(ar(fg), 3 * ar(fp), "la altura por área del grande es mucho mayor")
        self.assertAlmostEqual(reg(fg), 9 * np.pi * 144 / 1024, delta=0.6, msg="la suma de fracciones ≈ área/celda")


if __name__ == "__main__":
    unittest.main()
