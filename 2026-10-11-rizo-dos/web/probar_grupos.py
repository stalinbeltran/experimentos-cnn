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


class ClaseNueva(unittest.TestCase):
    """§ 107: el 1l es una CLASE del entrenamiento (el dueño: «que ésta sea la nueva clase en el app y para el entrenamiento»)."""

    def test_clases_y_salidas_en_las_cuatro_combinaciones(self):
        import servidor as S
        casos = {(False, False): (10, None, None), (True, False): (11, 10, None), (False, True): (11, None, 10), (True, True): (12, 10, 11)}
        for (s4, s1), (n, i4l, i1l) in casos.items():
            c = S.Compositor("x", ["rectas"], sub4=s4, sub1=s1); K = c.clases_finas()
            self.assertEqual((c.salidas, len(K)), (n, n), (s4, s1))
            self.assertEqual(K.index("4l") if "4l" in K else None, i4l); self.assertEqual(K.index("1l") if "1l" in K else None, i1l)
            self.assertEqual(K[4], "4a" if s4 else "4"); self.assertEqual(K[1], "1a" if s1 else "1")
            todas = np.arange(n); self.assertEqual(c.juntar(todas).tolist(), list(range(10)) + ([4] if s4 else []) + ([1] if s1 else []))
        self.assertEqual(S.Compositor("x", ["rectas"], sub4=True).clases_finas(), S.Compositor.CLASES_FINAS)   # el 4 solo: como siempre

    def _datos(self):
        import servidor as S
        D = datos_de_pega(); D.subgrupo4 = lambda i: True; D.manuales = lambda: {}; D._finas_c = {}
        return S, D

    def test_etiquetas_finas_el_4_solo_no_cambia_y_el_1_entra_detras(self):
        with tempfile.TemporaryDirectory() as d:
            os.environ["REVISOR_ETIQUETAS_DIR"] = d; GR._REGLA.clear()
            try:
                S, D = self._datos(); GR.marcar(D, "1", 1, "1l")       # idx 0: palo (1l por regla), 1: bandera (1l a mano), 2: un 4
                calc = lambda a, b: S.Datos._finas_calc(D, a, b).tolist()
                self.assertEqual(calc(True, False), [1, 1, 10])         # = lo de siempre: el 4 partido, el 1 intacto
                self.assertEqual(calc(True, True), [11, 11, 10])        # 1l detrás del 4l
                self.assertEqual(calc(False, True), [10, 10, 4])        # sin partir el 4, el 1l es la 10
                GR.marcar(D, "1", 0, "1a"); self.assertEqual(calc(True, True), [1, 11, 10])   # la mano manda sobre la regla
            finally:
                os.environ.pop("REVISOR_ETIQUETAS_DIR", None)

    def test_la_huella_del_1_solo_la_lleva_quien_lo_parte(self):
        import servidor as S
        with tempfile.TemporaryDirectory() as d:
            os.environ["REVISOR_ETIQUETAS_DIR"] = d; GR._REGLA.clear()
            try:
                D = types.SimpleNamespace(hash="abc", hash_manuales=lambda: "h4", hash_manuales1=lambda: S.Datos.hash_manuales1(None))
                solo4 = S.Compositor("a", ["rectas"], sub4=True); con1 = S.Compositor("b", ["rectas"], sub4=True, sub1=True)
                h4, h1 = solo4.huella(D), con1.huella(D)
                self.assertNotIn("etiquetas1", h4); self.assertNotIn("sub1", h4)          # LazosEntrantes y compañía: sin cambios
                GR.marcar(datos_de_pega(), "1", 0, "1a")
                self.assertEqual(solo4.huella(D), h4); self.assertNotEqual(con1.huella(D)["etiquetas1"], h1["etiquetas1"])
            finally:
                os.environ.pop("REVISOR_ETIQUETAS_DIR", None)


if __name__ == "__main__":
    unittest.main()
