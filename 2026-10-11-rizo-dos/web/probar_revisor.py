#!/usr/bin/env python3
"""rizo-2 — pruebas del REVISOR (2026-10-15). Las que no necesitan el dato corren siempre; las que sí (reproducir el
README, leer el 782) se saltan si no está el dataset publicado. Con el venv:

    ../.venv/bin/python -m unittest web/probar_revisor.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "web")); sys.path.insert(0, str(AQUI / "nn"))
import servidor as S  # noqa: E402
import cresta_curvas as CC  # noqa: E402

HAY_DATO = S.DELGADOS.is_file() or True
try:
    from expcnn import ruta_dataset
    HAY_DATO = ruta_dataset(S.DATASET) is not None
except Exception:  # noqa: BLE001
    HAY_DATO = False


class Columnas(unittest.TestCase):
    def test_cada_bloque_nombra_exactamente_sus_columnas(self):
        """Los nombres de las columnas (lo que la app enseña) tienen que casar con el ancho real del bloque en caract."""
        for celdas in (3, 4):
            B = S.rangos(celdas)   # los de caract y los que calcula la app (vertice_1)
            for b in S.BLOQUES:
                self.assertEqual(len(S.columnas(b, celdas)), B[b][1] - B[b][0], f"{b} con {celdas}×{celdas}")

    def test_los_sin_posicion_se_declaran(self):
        """Regla del dueño (2026-10-15): un detector sin posición se DICE. Los de 4 números son los únicos sin rejilla."""
        for b in S.BLOQUES:
            tiene_rejilla = any(c["tipo"] == "rejilla" for c in S.columnas(b, 3))
            self.assertEqual(tiene_rejilla, b not in S.SIN_POSICION, b)

    def test_entrenar_es_el_de_rastrear(self):
        """El MLP copiado da los MISMOS logits que `rastrear.entrenar` con la misma semilla (dos copias divergen)."""
        src = (AQUI / "nn" / "rastrear.py").read_text()
        cuerpo = src[src.index("def entrenar("):src.index("def logits(")]
        mio = (AQUI / "web" / "servidor.py").read_text(); mio = mio[mio.index("def entrenar("):mio.index("def _limpio(")]
        for linea in ("torch.manual_seed(sem); mu, sd = xtr.mean(0), xtr.std(0) + 1e-6", "lr=3e-3, weight_decay=1e-3", "for _ in range(400):", "torch.nn.Dropout(0.3)"):
            self.assertIn(linea, cuerpo); self.assertIn(linea, mio)

    def test_piezas_de_un_bloque(self):
        """2026-10-16, el dueño: la rejilla se elige aparte de la cuenta. Todas las piezas = el bloque a secas (misma huella que
        las recetas viejas); una pieza desconocida se NIEGA; y `corta` saca exactamente esas columnas, en su orden."""
        self.assertEqual(S.partes("vertice"), ["rejilla", "cuenta"])
        self.assertEqual(S.partes("rectas_todo"), ["rejilla", "cuenta", "largo"])
        self.assertEqual(S.partes("entrantes"), ["rejilla", "cuenta", "abertura"])
        self.assertEqual(S.partes("entrantes60"), ["rejilla"])
        self.assertEqual(S.normalizar("vertice:cuenta+rejilla"), "vertice")
        self.assertEqual(S.normalizar("vertice:rejilla"), "vertice:rejilla")
        self.assertEqual(S.normalizar("rectas_todo:largo+rejilla"), "rectas_todo:rejilla+largo")
        for malo in ("vertice:largo", "nada:rejilla", "vertice:"):
            with self.assertRaises(ValueError, msg=malo):
                S.normalizar(malo)
        B = CC.bloques(3); W = max(z for _, z in B.values()); X = np.arange(2 * W, dtype=float).reshape(2, W)
        c = S.Compositor("p", ["vertice:rejilla", "rectas_todo:cuenta"], 3)
        a, z = B["vertice"]; r = B["rectas_todo"][0]
        esperado = list(range(a, a + 9)) + [r + 9, r + 20, r + 31, r + 42]
        self.assertEqual(c.corta(X)[0].tolist(), esperado)
        self.assertEqual(c.cortes(), {"vertice:rejilla": (0, 9), "rectas_todo:cuenta": (9, 13)})
        entero = S.Compositor("e", ["vertice"], 3)
        self.assertEqual(entero.corta(X)[0].tolist(), list(range(a, z)))   # el bloque entero no cambia

    def test_vertice_en_una_celda(self):
        """2026-10-16, el dueño: el vértice en UNA celda. Un centro justo en la frontera (x = 21, la del dígito 77) cae en la
        celda que lo contiene, sin repartirse; la cuenta es la de siempre; y va al final de la matriz, sin mover los demás."""
        import unittest.mock as um
        with um.patch.object(S.CC, "vertices", return_value=[{"cx": 21.0, "cy": 0.5}]):
            g = S.bloque_vertice_1(np.zeros((32, 32)), 3)
        self.assertEqual(g, [0, 1, 0, 0, 0, 0, 0, 0, 0, 1.0])
        with um.patch.object(S.CC, "vertices", return_value=[]):
            self.assertEqual(S.bloque_vertice_1(np.zeros((32, 32)), 3), [0.0] * 10)
        B = S.rangos(3); fin = max(z for _, z in S.CC.bloques(3).values())   # justo detrás de caract (desde § 105 le sigue vertice_libre)
        self.assertEqual(B["vertice_1"], (fin, fin + 10))
        self.assertEqual(S.partes("vertice_1"), ["rejilla", "cuenta"])

    def test_los_de_tinta_no_se_ofrecen_pero_siguen_existiendo(self):
        """El dueño, 2026-10-17: los features de TINTA no aparecen como opción; los compositores ya entrenados con ellos cargan."""
        ofrecidos = S.bloques_ofrecidos()
        for b in ("vertice", "vertice_1"):
            self.assertIn(b, S.RETIRADOS); self.assertNotIn(b, ofrecidos); self.assertNotIn(b, S.archivados())
            self.assertIn(b, S.BLOQUES)                               # siguen en el código: LazosConcavidades/LazosEntrantes los usan
        self.assertEqual(S.normalizar("vertice_1:rejilla"), "vertice_1:rejilla")
        self.assertEqual(set(ofrecidos) | set(S.RETIRADOS), set(S.BLOQUES))
        self.assertFalse([b for b in S.BLOQUES_DEFECTO if S.pieza(b)[0] in S.RETIRADOS], "el defecto no puede llevar un retirado")

    def test_un_compositor_nuevo_con_un_retirado_se_niega(self):
        with self.assertRaisesRegex(ValueError, "TINTA"):
            S.comprobar_nuevo(["lazos_region:rejilla", "vertice_1:rejilla"])
        S.comprobar_nuevo(["rectas", "lazos_region:rejilla"])       # sin retirados: pasa

    @staticmethod
    def _lambda(abajo=False):
        """Un Λ de trazo de 3 px (un «delgado» sintético): punta en (16, 5), patas a (7, 27) y (25, 27). Con abajo, una V."""
        M = np.zeros((32, 32), np.float32); yy, xx = np.mgrid[0:32, 0:32] + 0.5
        for (x0, y0), (x1, y1) in (((16, 5), (7, 27)), ((16, 5), (25, 27))):
            t = np.clip(((xx - x0) * (x1 - x0) + (yy - y0) * (y1 - y0)) / ((x1 - x0) ** 2 + (y1 - y0) ** 2), 0, 1)
            M[np.hypot(xx - (x0 + t * (x1 - x0)), yy - (y0 + t * (y1 - y0))) <= 1.5] = 1.0
        return M[::-1].copy() if abajo else M

    def test_vertice_libre_columnas_casan_con_el_bloque(self):
        """§ 105: las columnas que nombra el revisor son exactamente los números que calcula nn/vertice_libre.py (R5)."""
        self.assertEqual(S.partes("vertice_libre"), ["rejilla", "amplitud", "despegue", "cerca", "radio", "cuenta"])
        for celdas in (3, 4):
            self.assertEqual(len(S.columnas("vertice_libre", celdas)), len(S.VLB.bloque(self._lambda(), celdas)))
        B = S.rangos(3); self.assertEqual(B["vertice_libre"][0], B["vertice_1"][1])          # va DESPUÉS: los guardados no se mueven
        subs = {}
        for c in S.columnas("vertice_libre", 3):
            if c["celda"] is not None:
                subs[c["sub"]] = subs.get(c["sub"], 0) + 1
        self.assertEqual(set(subs.values()), {9}); self.assertEqual(len(subs), 4 * 5)        # 4 direcciones × 5 rejillas 3×3
        self.assertIn("vertice_libre", S.bloques_ofrecidos())

    def test_vertice_libre_ve_la_punta_y_su_direccion(self):
        for abajo, d in ((False, "N"), (True, "S")):
            b = np.asarray(S.VLB.bloque(self._lambda(abajo), 3)); cols = S.columnas("vertice_libre", 3)
            val = {(c["sub"], c["celda"]): b[k] for k, c in enumerate(cols)}
            fila = 0 if d == "N" else 2
            self.assertEqual(val[(d, fila * 3 + 1)], 1.0, f"{d}: presencia en la celda del centro de su fila")
            # ⚠ La amplitud de hoy (cuerda a 6 px desde la punta de la CRESTA) SOBREESTIMA en trazos gruesos: el esqueleto se curva
            # en una punta aguda. Medido el 2026-10-17 con Λ sintéticos: 44° reales salen 44–68° según el grosor; el ángulo entre
            # rectas ajustadas a los brazos (3–9 px) queda en ±6° (propuesto al dueño, § 105). Aquí se fija lo que SÍ se cumple:
            # nunca se queda corta en más de 5° y nunca pasa del límite de 90°.
            amp = val[(f"{d} · amplitud", fila * 3 + 1)] * 180
            self.assertGreaterEqual(amp, 44 - 5); self.assertLess(amp, 90)
            self.assertEqual(sum(v for (sub, celda), v in val.items() if celda is None and sub == d), 1.0)   # cuenta 1
            otras = [x for x in S.VLB.ORDEN if x != d]
            self.assertEqual(sum(v for (sub, celda), v in val.items() if sub.split(" ")[0] in otras), 0.0)

    def test_la_huella_solo_cambia_para_quien_usa_el_vertice_libre(self):
        class D:
            hash = "abc"
            def hash_manuales(self):
                return "sin"
        sin = S.Compositor("a", ["lazos_region", "vertice_1:rejilla"]).huella(D()); con = S.Compositor("b", ["vertice_libre:rejilla+radio"]).huella(D())
        self.assertNotIn("vertice_libre", sin)                       # los compositores guardados conservan su huella y sus pesos
        self.assertEqual(con["vertice_libre"], S.VLB.huella_modulo())

    @staticmethod
    def _tres(fin=60, espejo=False, ese=False):
        """Un 3 sintético de trazo de 3 px: dos arcos de radio 6 (centros (15, 10) y (15, 22)) que se encuentran en la cintura, cada
        uno pasando `fin` grados de su punto más a la derecha. Con fin = 90 llegan en paralelo y el esqueleto los parte en un NUDO.
        Con `ese`, el de abajo abre al otro lado (una S): + − sin el segundo +, no es cúspide."""
        M = np.zeros((32, 32), np.float32); yy, xx = np.mgrid[0:32, 0:32] + 0.5
        def arco(c, t0, t1):
            t = np.radians(np.linspace(t0, t1, 40)); p = np.stack([c[0] + 6 * np.cos(t), c[1] + 6 * np.sin(t)], 1)
            for (x0, y0), (x1, y1) in zip(p[:-1], p[1:]):
                u = np.clip(((xx - x0) * (x1 - x0) + (yy - y0) * (y1 - y0)) / ((x1 - x0) ** 2 + (y1 - y0) ** 2 + 1e-9), 0, 1)
                M[np.hypot(xx - (x0 + u * (x1 - x0)), yy - (y0 + u * (y1 - y0))) <= 1.5] = 1.0
        if ese:
            arco((15, 10), 30, 270); arco((15, 22), 270, 570)
        else:
            arco((15, 10), 200, 360 + fin); arco((15, 22), 360 - fin, 520)
        return M[:, ::-1].copy() if espejo else M

    def _cusp(self, M):
        b = np.asarray(S.CUS.bloque(M, 3)); cols = S.columnas("cuspide", 3)
        return {(c["sub"], c["celda"]): b[k] for k, c in enumerate(cols)}

    def test_cuspide_columnas_casan_con_el_bloque(self):
        """§ 111: las columnas que nombra el revisor son exactamente los números de nn/cuspide.py (R5), y va la ÚLTIMA."""
        self.assertEqual(S.partes("cuspide"), ["rejilla", "giro", "curvas", "cuenta"])
        for celdas in (3, 4):
            self.assertEqual(len(S.columnas("cuspide", celdas)), len(S.CUS.bloque(self._tres(), celdas)))
        B = S.rangos(3); self.assertEqual(B["cuspide"][0], B["vertice_libre"][1])             # detrás: los guardados no se mueven
        self.assertEqual(B["cuspide"][1] - B["cuspide"][0], 4 * (3 * 9 + 1))
        self.assertIn("cuspide", S.bloques_ofrecidos())

    def test_cuspide_ve_la_cintura_del_3_y_hacia_donde_apunta(self):
        """El 3 apunta al O, su espejo (Ɛ) al E, en la celda del centro, con cuenta 1 y nada en las demás direcciones; también
        cuando la punta cae en un NUDO (fin = 90: el esqueleto deja un hueco y unir los brazos diluía el giro, 2026-10-17)."""
        for fin in (60, 90):
            if fin == 90:
                self.assertGreaterEqual(len(S.CC.tramos(self._tres(90))[0]), 2, "el caso del nudo tiene que tener un nudo")
            for espejo, d in ((False, "O"), (True, "E")):
                v = self._cusp(self._tres(fin, espejo))
                self.assertEqual(v[(d, 4)], 1.0, f"fin {fin}: presencia {d} en la celda del centro")
                self.assertEqual(v[(d, None)], 1.0, f"fin {fin}: cuenta {d} = 1")
                self.assertGreater(v[(f"{d} · giro", 4)], 15 / 90 - 1e-6); self.assertGreater(v[(f"{d} · curvas", 4)], 45 / 180 - 1e-6)
                self.assertEqual(sum(x for (sub, _), x in v.items() if sub.split(" ")[0] != d), 0.0, f"fin {fin}: sólo {d}")

    def test_cuspide_no_la_dan_una_s_ni_un_vertice(self):
        """Una S (2, 5) es + − sin el segundo +; un Λ es UN giro entre dos rectas: ninguno es cúspide."""
        for nombre, M in (("S", self._tres(ese=True)), ("Λ", self._lambda()), ("V", self._lambda(True))):
            self.assertEqual(sum(self._cusp(M).values()), 0.0, nombre)

    def test_la_huella_solo_cambia_para_quien_usa_la_cuspide(self):
        class D:
            hash = "abc"
            def hash_manuales(self):
                return "sin"
        sin = S.Compositor("a", ["lazos_region", "vertice_libre:rejilla"]).huella(D()); con = S.Compositor("b", ["cuspide:rejilla"]).huella(D())
        self.assertNotIn("cuspide", sin)                              # LEvertice y los demás conservan su huella y sus pesos
        self.assertEqual(con["cuspide"], S.CUS.huella_modulo())

    @staticmethod
    def _ovalo(cy, ry, rx=5.0, cx=16.0, palo=None, M=None):
        """Un óvalo de trazo de 3 px (un «delgado» sintético) con centro (cx, cy) y semiejes (rx, ry); `palo` = (y0, y1) añade un palo
        vertical a la derecha (un 9 o un 6). Se puede pintar encima de otro M."""
        M = np.zeros((32, 32), np.float32) if M is None else M; yy, xx = np.mgrid[0:32, 0:32] + 0.5
        t = np.linspace(0, 2 * np.pi, 200); p = np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1)
        segs = list(zip(p[:-1], p[1:])) + ([((cx + rx, palo[0]), (cx + rx, palo[1]))] if palo else [])
        for (x0, y0), (x1, y1) in segs:
            u = np.clip(((xx - x0) * (x1 - x0) + (yy - y0) * (y1 - y0)) / ((x1 - x0) ** 2 + (y1 - y0) ** 2 + 1e-9), 0, 1)
            M[np.hypot(xx - (x0 + u * (x1 - x0)), yy - (y0 + u * (y1 - y0))) <= 1.5] = 1.0
        return M

    def _alt(self, M):
        b = np.asarray(S.LAL.bloque(M, 3)); return b[:9].reshape(3, 3), b[9]

    def test_lazos_altura_columnas_casan_con_el_bloque(self):
        """§ 128: las columnas que nombra el revisor son los números de nn/lazo_altura.py (R5), y va la ÚLTIMA (detrás de la cúspide)."""
        self.assertEqual(S.partes("lazos_altura"), ["rejilla", "cuenta"])
        for celdas in (3, 4):
            self.assertEqual(len(S.columnas("lazos_altura", celdas)), len(S.LAL.bloque(self._ovalo(16, 11), celdas)))
        B = S.rangos(3); self.assertEqual(B["lazos_altura"][0], B["cuspide"][1]); self.assertEqual(B["lazos_altura"][1] - B["lazos_altura"][0], 10)
        self.assertIn("lazos_altura", S.bloques_ofrecidos())

    def test_lazos_altura_un_cero_vale_uno_y_un_nueve_menos(self):
        """Un 0 (el lazo ES el dígito) da ≈ 1 en la celda del centro; un 9 (lazo de 10 px arriba, palo hasta abajo) da ≈ 0,4 arriba."""
        g, n = self._alt(self._ovalo(16, 11)); self.assertEqual(n, 1.0); self.assertGreater(g[1, 1], 0.9); self.assertAlmostEqual(g[1, 1], g.max())
        g, n = self._alt(self._ovalo(9, 5, palo=(9, 28))); self.assertEqual(n, 1.0)
        self.assertEqual(int(np.argmax(g)) // 3, 0, "el pico va en la fila de ARRIBA")
        self.assertTrue(0.3 < g.max() < 0.55, f"altura relativa {g.max():.2f}")

    def test_lazos_altura_no_depende_del_tamano_en_px(self):
        """El dueño: «relativa a 1, así es independiente del tamaño en px». El mismo 9 a dos escalas da la misma altura (± 0,05)."""
        a = self._alt(self._ovalo(9, 5, palo=(9, 28)))[0].max(); b = self._alt(self._ovalo(11, 3.5, rx=3.5, palo=(11, 24)))[0].max()
        self.assertAlmostEqual(a, b, delta=0.05)
        self.assertEqual(self._alt(np.zeros((32, 32), np.float32))[1], 0.0, "sin cresta: cuenta 0 y no se cae")

    def test_la_huella_solo_cambia_para_quien_usa_lazos_altura(self):
        class D:
            hash = "abc"
            def hash_manuales(self):
                return "sin"
        sin = S.Compositor("a", ["lazos_region", "cuspide:rejilla"]).huella(D()); con = S.Compositor("b", ["lazos_altura"]).huella(D())
        self.assertNotIn("lazos_altura", sin); self.assertEqual(con["lazos_altura"], S.LAL.huella_modulo())

    def test_la_fuente_no_cambia_la_huella_de_nist_y_entra_en_la_de_las_demas(self):
        """§ 124: un compositor sin fuente es «nist» y su huella es la de siempre (los guardados se restauran igual); con otra fuente,
        la huella dice el dataset, la fuente y el delgado (unos pesos de una fuente nunca sirven para otra). Una fuente desconocida, no."""
        class D:
            hash = "abc"
            def hash_manuales(self):
                return "sin"
        viejo = S.Compositor("a", ["lazos_region", "entrantes60"], sub4=True).huella(D())
        self.assertEqual(viejo["dataset"], S.DATASET); self.assertNotIn("fuente", viejo); self.assertNotIn("delgado", viejo)
        nuevo = S.Compositor("a", ["lazos_region", "entrantes60"], sub4=True, fuente="alto32x64").huella(D())
        self.assertEqual(nuevo["dataset"], S.FUENTES["alto32x64"]["dataset"]); self.assertEqual(nuevo["fuente"], "alto32x64")
        self.assertEqual(nuevo["delgado"], S.FUENTES["alto32x64"]["delgado"].name); self.assertNotEqual(viejo, nuevo)
        with self.assertRaises(ValueError):
            S.Compositor("a", ["rectas"], fuente="no-existe")

    def test_modo_desconocido_se_niega(self):
        r = subprocess.run([sys.executable, str(AQUI / "web" / "servidor.py"), "lo", "que", "sea"], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 2); self.assertIn("no sé qué es", r.stdout)

    def test_entrenar_seco_lista_y_se_niega(self):
        """`entrenar … --seco` dice qué entrenaría (ID · nombre) sin entrenar, y ante una receta desconocida SE NIEGA (rc 2)."""
        srv = str(AQUI / "web" / "servidor.py")
        r = subprocess.run([sys.executable, srv, "entrenar", "todas", "--seco"], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr); self.assertIn("SECO", r.stdout)
        ids = re.findall(r"^  ([0-9a-f]{8}) · ", r.stdout, re.M); self.assertGreaterEqual(len(ids), 1)
        r2 = subprocess.run([sys.executable, srv, "entrenar", ids[0], "--seco"], capture_output=True, text=True, timeout=120)
        self.assertEqual(r2.returncode, 0); self.assertEqual(len(re.findall(r"^  [0-9a-f]{8} · ", r2.stdout, re.M)), 1)
        r3 = subprocess.run([sys.executable, srv, "entrenar", "no-existe", "--seco"], capture_output=True, text=True, timeout=120)
        self.assertEqual(r3.returncode, 2); self.assertIn("no conozco", r3.stdout)

    def test_lanzador_paralelo_seco_y_se_niega(self):
        """`entrenar_paralelo.sh seco` imprime recetas, máquinas y la ORDEN sin lanzar; un modo desconocido y una receta
        desconocida se niegan (2026-09-08: el último caso del despacho nunca es una acción por defecto)."""
        sh = str(AQUI / "web" / "entrenar_paralelo.sh")
        r = subprocess.run(["sh", sh, "seco", "todas"], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr); self.assertIn("SECO", r.stdout); self.assertIn("Orden: ", r.stdout)
        self.assertIn("desacoplar-persistente.sh rizo2-entrenar-paralelo", r.stdout); self.assertIn("--dentro", r.stdout)
        self.assertRegex(r.stdout, r"rizo2-[0-9a-f]{8}")
        r2 = subprocess.run(["sh", sh, "seco", "no-existe"], capture_output=True, text=True, timeout=120)
        self.assertEqual(r2.returncode, 2); self.assertIn("desconocida", r2.stdout)
        r3 = subprocess.run(["sh", sh, "lo-que-sea"], capture_output=True, text=True, timeout=60)
        self.assertEqual(r3.returncode, 2); self.assertIn("no sé qué es", r3.stdout)

    def test_url_sin_token_dice_que_no_ha_arrancado(self):
        env = {**os.environ, "REVISOR_TOKEN_FILE": str(Path(tempfile.mkdtemp()) / "no-existe.token")}
        r = subprocess.run([sys.executable, str(AQUI / "web" / "servidor.py"), "url"], capture_output=True, text=True, timeout=120, env=env)
        self.assertEqual(r.returncode, 1); self.assertIn("no hay token", r.stdout)


@unittest.skipUnless(HAY_DATO, "sin dataset publicado")
class ConDato(unittest.TestCase):
    """Un servidor de verdad en un puerto libre, con token propio: reproduce el README y contesta las cuatro pantallas."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(); cls.puerto = 8900 + os.getpid() % 100
        env = {**os.environ, "REVISOR_TOKEN_FILE": f"{cls.tmp}/t.token", "REVISOR_SIN_ALMACEN": "1"}   # nunca al almacén de verdad
        cls.p = subprocess.Popen([sys.executable, str(AQUI / "web" / "servidor.py"), "--puerto", str(cls.puerto), "--host", "127.0.0.1"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for _ in range(600):                                     # caract la primera vez tarda ~1 min; la caché, segundos
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{cls.puerto}/salud", timeout=2); break
            except Exception:  # noqa: BLE001
                time.sleep(1)
        cls.tok = Path(f"{cls.tmp}/t.token").read_text().strip()
        for _ in range(900):                                     # los cuatro compositores base: ~30 s cada uno
            e = cls.api("/api/estado")
            if all(c["estado"] in ("listo", "error") for c in e["compositores"]):
                break
            time.sleep(2)

    @classmethod
    def tearDownClass(cls):
        cls.p.terminate(); cls.p.wait(timeout=30)

    @classmethod
    def api(cls, ruta, **q):
        from urllib.parse import urlencode
        with urllib.request.urlopen(f"http://127.0.0.1:{cls.puerto}{ruta}?" + urlencode({"t": cls.tok, **q}), timeout=120) as r:
            return json.loads(r.read())

    def test_sin_token_403(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(f"http://127.0.0.1:{self.puerto}/api/estado", timeout=10)
        self.assertEqual(cm.exception.code, 403)

    def test_reproduce_las_corridas_del_readme(self):
        """Cada receta base cae en el val_rango y da los MISMOS fallos (semilla 0) que el JSON de su corrida."""
        e = self.api("/api/estado"); comps = {c["nombre"]: c for c in e["compositores"]}
        for nombre, fichero in S.CORRIDA_DE.items():
            j = json.loads((AQUI / "resultados" / fichero).read_text()); c = comps[nombre]
            self.assertEqual(c["estado"], "listo", c.get("error"))
            self.assertEqual(c["val"], j["val"], nombre); self.assertEqual(c["fallos_s0"], j["fallos_s0"], nombre)

    def test_lista_fallos_y_filtros(self):
        n = "aprobada: lazo región (§ 79)"
        L = self.api("/api/lista", c=n); self.assertEqual(L["total"], 105)
        self.assertTrue(all(f["real"] != f["predicho"] for f in L["filas"]))
        L6 = self.api("/api/lista", c=n, conf="6→0"); self.assertIn(782, [f["idx"] for f in L6["filas"]])
        self.assertTrue(all(f["real"] == f["predicho"] for f in self.api("/api/lista", c=n, modo="aciertos", clase=3)["filas"]))

    def test_leer_782_es_la_figura_82(self):
        """El 6 idx 782 leído como 0: logit 0 ≈ +4,3, logit 6 ≈ +4,0, y el bloque del lazo región empuja ≈ +3,0 (README § 82)."""
        L = self.api("/api/leer", c="aprobada: lazo región (§ 79)", idx=782)
        self.assertEqual((L["real"], L["predicho"]), (6, 0))
        self.assertAlmostEqual(L["logits"][0], 4.3, delta=0.15); self.assertAlmostEqual(L["logits"][6], 4.0, delta=0.15)
        self.assertAlmostEqual(L["suma_por_bloque"]["lazos_region"], 3.0, delta=0.2)
        self.assertEqual(sum(len(S.columnas(b["bloque"], 3)) for b in L["bloques"]), 144)
        self.assertEqual(L["pesan"][0]["nombre"], "lazos REGIÓN · centro")

    def test_digito_y_comparar(self):
        d = self.api("/api/digito", idx=782); self.assertEqual(d["real"], 6); self.assertEqual(len(d["M"]), 1024)
        self.assertAlmostEqual(d["objetos"]["lazos"][0]["area"], 182, delta=1); self.assertIn("cono", d["objetos"]["entrantes"][0])
        R = self.api("/api/comparar", idx=782, cs="aprobada: lazo región (§ 79)|lazo área (§ 79)")
        self.assertEqual(sorted(R["bloques_que_difieren"]), ["lazos_area", "lazos_region"])
        self.assertEqual(R["lecturas"]["lazo área (§ 79)"]["predicho"], 6)             # README § 83: con área se lee bien
        m = self.api("/api/miniaturas", ids="782,101"); self.assertEqual([x["idx"] for x in m], [782, 101])

    def test_etiquetar_a_mano(self):
        """/etiquetar: marcar un 4 a mano manda sobre la regla, se guarda en web/etiquetas-4.json y 'auto' lo quita."""
        f = AQUI / "web" / "etiquetas-4.json"; antes = f.read_text() if f.is_file() else None
        try:
            c = {x["idx"]: x for x in self.api("/api/cuatros")["cuatros"]}; self.assertEqual(len(c), 568); self.assertIsNone(c[110]["manual"])
            r = self.api("/api/marcar", idx=110, clase="4a"); self.assertEqual(r["manuales"], {"110": "4a"})
            self.assertEqual(json.loads(f.read_text())["4a"], [110])
            self.assertEqual({x["idx"]: x for x in self.api("/api/cuatros")["cuatros"]}[110]["manual"], "4a")
            with self.assertRaises(urllib.error.HTTPError):
                self.api("/api/marcar", idx=782, clase="4l")                 # el 782 es un 6
            self.api("/api/marcar", idx=110, clase="auto"); self.assertEqual(json.loads(f.read_text())["4a"], [])
        finally:
            if antes is None:
                f.unlink(missing_ok=True)
            else:
                f.write_text(antes)

    def test_nuevo_compositor_y_borrar(self):
        r = self.api("/api/nuevo", nombre="prueba-golpes", bloques="golpes,lazos", semillas=1)
        self.assertEqual(r["nombre"], "prueba-golpes")
        for _ in range(120):
            c = {x["nombre"]: x for x in self.api("/api/estado")["compositores"]}["prueba-golpes"]
            if c["estado"] == "listo":
                break
            time.sleep(1)
        self.assertEqual(c["estado"], "listo"); self.assertEqual(c["n"], 20); self.assertLess(c["val"], 0.7)
        self.api("/api/borrar", nombre="prueba-golpes")
        self.assertNotIn("prueba-golpes", [x["nombre"] for x in self.api("/api/estado")["compositores"]])
        with self.assertRaises(urllib.error.HTTPError):
            self.api("/api/nuevo", bloques="no-existe")


if __name__ == "__main__":
    unittest.main()


class Pesos(unittest.TestCase):
    """Los pesos de un compositor sobreviven al dev (el dueño, 2026-10-16). Sin dato: redes entrenadas sobre números al azar."""

    def _listo(self, nombre="receta · prueba", semillas=(0, 1)):
        rng = np.random.default_rng(0); X = rng.normal(size=(60, 12)).astype(np.float32); y = rng.integers(0, 10, 60)
        c = S.Compositor(nombre, ["rectas"], 3, semillas); c.n = 12; c.redes = []
        for s in semillas:
            red, mu, sd = S.entrenar(X, y, s); c.redes.append((red, mu, sd))
        c.preds = np.zeros((len(semillas), 5), int); c.veces = np.zeros(5, int); c.finos = c.preds.copy(); c.logits_va = np.zeros((5, 10))
        c.segundos = 1.0; c.resumen = {"nombre": nombre, "val": 0.5}; c.estado = "listo"
        return c, X

    def test_borrar_quita_sus_pesos_y_limpiar_quita_los_huerfanos(self):
        """El dueño, 2026-10-17: «si los borro es porque ya no son necesarios»: borrar un compositor quita su .pt del almacén, y
        `limpiar-pesos` quita los de compositores borrados antes (cuando borrar no los tocaba). Los de recetas vivas se quedan."""
        with tempfile.TemporaryDirectory() as d:
            alm = Path(d); vivo, muerto, viejo = (S.Compositor(n, ["rectas"]) for n in ("Vivo", "Muerto", "Viejo de ayer"))
            for c in (vivo, muerto, viejo):
                c.fichero(alm).write_bytes(b"x")
            real = (S.almacen_revisor, S.recetas_guardadas, S.empujar_almacen, S.CACHE); empujes = []
            S.almacen_revisor = lambda: alm; S.recetas_guardadas = lambda: {"Vivo": {}}; S.empujar_almacen = empujes.append
            S.CACHE = alm / "cache-local"                             # la caché del test, nunca la de la app viva
            self.assertEqual(S.limpiar_pesos(seco=True), 2)           # sin ningún recetas.json legible se NIEGA (no adivina cuáles viven)
            (alm / "recetas.json").write_text('{"Vivo": {}}')
            try:
                self.assertEqual(S.borrar_pesos([muerto]), [muerto.fichero(alm).name])
                self.assertFalse(muerto.fichero(alm).exists()); self.assertTrue(vivo.fichero(alm).exists())
                self.assertEqual([f.name for f in S.pesos_huerfanos(alm, {"Vivo": {}})], [viejo.fichero(alm).name])
                self.assertEqual(S.limpiar_pesos(seco=True), 0); self.assertTrue(viejo.fichero(alm).exists()); self.assertEqual(empujes, [])
                self.assertEqual(S.limpiar_pesos(seco=False), 0)
                self.assertFalse(viejo.fichero(alm).exists()); self.assertTrue(vivo.fichero(alm).exists()); self.assertEqual(len(empujes), 1)
            finally:
                S.almacen_revisor, S.recetas_guardadas, S.empujar_almacen, S.CACHE = real

    def test_guardar_y_restaurar_da_los_mismos_logits(self):
        c, X = self._listo(); huella = {"cresta": "abc", "etiquetas4": "sin", "bloques": ["rectas"]}
        with tempfile.TemporaryDirectory() as d:
            f = c.guardar(Path(d), huella); self.assertTrue(f.is_file())
            c2 = S.Compositor(c.nombre, ["rectas"], 3, (0, 1))
            self.assertTrue(c2.restaurar(Path(d), huella)); self.assertEqual(c2.estado, "listo"); self.assertIn("restaurado", c2.resumen)
            for (r1, mu, sd), (r2, mu2, sd2) in zip(c.redes, c2.redes):
                import torch
                z = torch.from_numpy((X - mu) / sd).float()
                with torch.no_grad():
                    np.testing.assert_allclose(r1(z).numpy(), r2(z).numpy(), atol=1e-6)

    def test_huella_distinta_no_restaura(self):
        """Otra versión de cresta_curvas.py, otras etiquetas a mano u otra receta: se reentrena, nunca se sirve con pesos ajenos."""
        c, _ = self._listo(); huella = {"cresta": "abc", "etiquetas4": "sin"}
        with tempfile.TemporaryDirectory() as d:
            c.guardar(Path(d), huella)
            for otra in ({"cresta": "zzz", "etiquetas4": "sin"}, {"cresta": "abc", "etiquetas4": "1234"}, {}):
                c2 = S.Compositor(c.nombre, ["rectas"], 3, (0, 1))
                self.assertFalse(c2.restaurar(Path(d), otra)); self.assertEqual(c2.estado, "pendiente"); self.assertEqual(c2.redes, [])
            self.assertFalse(S.Compositor("otra receta", ["rectas"], 3).restaurar(Path(d), huella))

    def test_un_fichero_roto_se_reentrena_en_vez_de_caer(self):
        c, _ = self._listo()
        with tempfile.TemporaryDirectory() as d:
            c.fichero(Path(d)).write_bytes(b"basura"); self.assertFalse(c.restaurar(Path(d), {}))

    def test_la_cache_de_features_viene_del_almacen_si_falta_en_local(self):
        """Las features (web/cache/*.npz) también sobreviven al dev (el dueño, 2026-10-16): si no están en local se traen del
        almacén; si tampoco están allí, se calculan; y nunca se pisa lo local."""
        with tempfile.TemporaryDirectory() as d:
            alm = Path(d) / "alm"; (alm / "cache").mkdir(parents=True); loc = Path(d) / "local" / "caract-x-3x3.npz"
            real = S.almacen_revisor; S.almacen_revisor = lambda: alm
            try:
                self.assertFalse(S.traer_cache(loc)); self.assertFalse(loc.exists())           # en ningún lado: se calcula
                (alm / "cache" / loc.name).write_bytes(b"del almacen")
                self.assertTrue(S.traer_cache(loc)); self.assertEqual(loc.read_bytes(), b"del almacen")
                loc.write_bytes(b"local"); self.assertTrue(S.traer_cache(loc)); self.assertEqual(loc.read_bytes(), b"local")
                (alm / "cache" / loc.name).unlink(); S.empujar_almacen = lambda m: None     # en local y no en el almacén: se sube (sin git aquí)
                self.assertTrue(S.traer_cache(loc)); self.assertEqual((alm / "cache" / loc.name).read_bytes(), b"local")
                self.assertEqual(S.cache_del_almacen(loc), alm / "cache" / loc.name)
            finally:
                S.almacen_revisor = real
        S_real = S.almacen_revisor; S.almacen_revisor = lambda: None
        try:
            self.assertIsNone(S.cache_del_almacen(Path("/x/caract.npz"))); self.assertFalse(S.traer_cache(Path("/x/no-existe.npz")))
        finally:
            S.almacen_revisor = S_real

    def test_reentrenar_uno_solo_o_todos_los_sub4(self):
        """`reentrenar(nombre)` encola SÓLO ése (para pisar unos pesos emulados sin pagar los demás); sin nombre, todos los sub4."""
        import threading
        app = S.App.__new__(S.App); app.comps = {}; app.cola = []; app._lock = threading.Lock()
        a = S.Compositor("a", ["rectas"], sub4=True); b = S.Compositor("b", ["rectas"], sub4=True); c = S.Compositor("c", ["rectas"])
        for x in (a, b, c):
            x.estado = "listo"; app.comps[x.nombre] = x
        self.assertEqual(app.reentrenar("c"), 1); self.assertEqual(app.cola, [c]); self.assertEqual(c.estado, "pendiente"); self.assertTrue(c.reentrenar)
        self.assertEqual(app.reentrenar("c"), 0)                                  # ya espera: no se duplica
        self.assertEqual(app.reentrenar(), 2); self.assertEqual(app.cola, [c, a, b])   # sin nombre: los sub4 (c no es sub4 y ya estaba)
        with self.assertRaises(KeyError):
            app.reentrenar("no existe")

    def test_no_se_guarda_lo_que_no_esta_listo(self):
        c = S.Compositor("x", ["rectas"])
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                c.guardar(Path(d), {})
