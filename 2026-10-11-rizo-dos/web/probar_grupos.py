#!/usr/bin/env python3
"""Pruebas de la pantalla separadora (`grupos.py`, § 106). Sin dataset y sin entrenar: trazos sintéticos y un «datos» de pega.

    cd 2026-10-11-rizo-dos && ../.venv/bin/python -m unittest web/probar_grupos.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "web")); sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
os.environ["REVISOR_SIN_ALMACEN"] = "1"                         # nada de lo que haga una prueba llega al almacén
import grupos as GR                                             # noqa: E402

_YY, _XX = np.mgrid[0:32, 0:32] + 0.5


def trazo(*segmentos, r=0.6):
    """Un «delgado» sintético: segmentos con perfil suave de radio ~r (como la salida de delg-cnn)."""
    M = np.zeros((32, 32), np.float32)
    for (x0, y0), (x1, y1) in segmentos:
        t = np.clip(((_XX - x0) * (x1 - x0) + (_YY - y0) * (y1 - y0)) / ((x1 - x0) ** 2 + (y1 - y0) ** 2), 0, 1)
        M = np.maximum(M, np.clip(1.2 - np.hypot(_XX - (x0 + t * (x1 - x0)), _YY - (y0 + t * (y1 - y0))) / (2 * r), 0, 1))
    return M


def datos_de_pega():
    """idx 0 y 1 son unos, idx 2 es un 4: lo justo para marcar y negarse."""
    M = np.stack([trazo(((16, 3), (16, 29))), trazo(((16, 3), (16, 29)), ((16, 3), (8, 11))), trazo(((8, 3), (8, 20)), ((8, 20), (24, 20)))])
    return types.SimpleNamespace(y=np.array([1, 1, 4]), M=M, tr=np.array([0]), va=np.array([1]), ex=np.array([2]))


class Regla(unittest.TestCase):
    """La regla del 1l, escrita antes de mirar (R13): un palo recto y casi vertical, sin nada más."""

    def test_un_palo_vertical_es_1l(self):
        self.assertEqual(GR.regla_1(trazo(((16, 3), (16, 29)))), "1l")

    def test_un_palo_poco_inclinado_es_1l_y_uno_tumbado_no(self):
        self.assertEqual(GR.regla_1(trazo(((19, 3), (13, 29)))), "1l")     # ~13° de la vertical
        self.assertEqual(GR.regla_1(trazo(((26, 4), (6, 28)))), "1a")      # ~40°

    def test_con_bandera_o_con_base_es_1a(self):
        self.assertEqual(GR.regla_1(trazo(((16, 3), (16, 29)), ((16, 3), (9, 10)))), "1a")      # bandera
        self.assertEqual(GR.regla_1(trazo(((16, 3), (16, 28)), ((10, 28), (22, 28)))), "1a")   # base

    def test_limitacion_conocida_una_bandera_corta_se_pierde(self):
        """⚠ Fija una LIMITACIÓN, no un acierto: el esqueleto se come la esquina y una bandera de 8,5 px queda como espolón.
        Si un cambio la arregla, este test falla y hay que actualizar el comentario de grupos.py (y alegrarse)."""
        self.assertEqual(GR.regla_1(trazo(((16, 3), (16, 29)), ((16, 3), (10, 9)))), "1l")

    def test_curvo_es_1a(self):
        th = np.linspace(-1.2, 1.2, 30); pts = list(zip(10 + 9 * np.cos(th), 16 + 13 * np.sin(th)))
        self.assertEqual(GR.regla_1(trazo(*zip(pts[:-1], pts[1:]))), "1a")


class Marcar(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); os.environ["REVISOR_ETIQUETAS_DIR"] = self.tmp.name; GR._REGLA.clear()

    def tearDown(self):
        os.environ.pop("REVISOR_ETIQUETAS_DIR", None); self.tmp.cleanup()

    def test_marcar_escribe_y_auto_quita(self):
        D = datos_de_pega(); f = GR.marcar(D, "1", 1, "1l")
        self.assertEqual(f, Path(self.tmp.name) / "etiquetas-1.json"); self.assertEqual(GR.manuales("1"), {1: "1l"})
        els = {e["idx"]: e for e in GR.elementos(D, "1")}
        self.assertEqual(set(els), {0, 1}); self.assertEqual(els[0]["regla"], "1l"); self.assertEqual(els[1]["manual"], "1l")
        self.assertEqual(els[1]["regla"], "1a")                                 # la bandera: la regla dice 1a, la mano manda
        GR.marcar(D, "1", 1, "auto"); self.assertEqual(GR.manuales("1"), {})

    def test_se_niega(self):
        D = datos_de_pega()
        for d, i, c in (("4", 2, "4l"), ("7", 0, "7a"), ("1", 2, "1l"), ("1", 0, "1x"), ("1", 99, "1l")):
            with self.assertRaises(ValueError, msg=(d, i, c)):
                GR.marcar(D, d, i, c)
        with self.assertRaisesRegex(ValueError, "/etiquetar"):
            GR.grupo("4")

    def test_la_copia_del_almacen_vale_si_falta_la_local(self):
        D = datos_de_pega(); f = GR.marcar(D, "1", 0, "1a")
        with tempfile.TemporaryDirectory() as a:
            empujes = []; GR.respaldar(f, Path(a), empujes.append); f.unlink()
            self.assertEqual(GR.manuales("1", Path(a)), {0: "1a"}); self.assertEqual(len(empujes), 1)


class Huella(unittest.TestCase):
    def test_separar_los_1_no_cambia_la_huella_de_ningun_compositor(self):
        """El fichero del 4 entra en la huella (Datos.hash_manuales); el del 1 NO: marcar un 1 no puede invalidar unos pesos."""
        import servidor as S
        with tempfile.TemporaryDirectory() as d:
            os.environ["REVISOR_ETIQUETAS_DIR"] = d
            try:
                D4 = types.SimpleNamespace(hash="abc", hash_manuales=lambda: S.Datos.hash_manuales(None))
                c = S.Compositor("x", ["lazos_region", "entrantes60"], sub4=True); antes = c.huella(D4)
                GR.marcar(datos_de_pega(), "1", 0, "1a"); despues = c.huella(D4)
                self.assertEqual(antes, despues); self.assertFalse(any("1" in k for k in despues if k not in ("semillas",)))
            finally:
                os.environ.pop("REVISOR_ETIQUETAS_DIR", None)


if __name__ == "__main__":
    unittest.main()
