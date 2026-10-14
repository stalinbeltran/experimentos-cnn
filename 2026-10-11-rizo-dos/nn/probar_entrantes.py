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
        self.assertEqual(f.shape, (1, 204)); self.assertEqual(b["entrantes"], (152, 196)); self.assertEqual(b["entrantes4"], (196, 200)); self.assertEqual(b["entrantes4n"], (200, 204))
        c = f[0, 200:204].tolist()                                   # cuentas por → ← ↓ ↑
        self.assertEqual(sum(c), 2.0); self.assertEqual(c[1], 0.0, "ninguna abre a la izquierda, donde no hay barra")
        self.assertGreater(c[0], 0.0, "la de encima de la barra abre a la derecha; la de debajo cae en diagonal (→ o ↓, la regla de cuadrantes decide)")
        self.assertTrue(all((f[0, 196 + o] > 0) == (c[o] > 0) for o in range(4)), "fuerza > 0 exactamente donde hay cuenta")
        self.assertEqual(float(np.abs(f[0, b["concavidades"][0]:b["concavidades"][1]]).sum()), 0.0, "sin arco, el bloque de concavidades va vacío")
        self.assertGreater(float(np.abs(f[0, 152:196]).sum()), 0.0, "…y el de entrantes, no")


if __name__ == "__main__":
    unittest.main()
