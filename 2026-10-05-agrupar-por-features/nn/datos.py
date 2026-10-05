#!/usr/bin/env python3
"""Los dígitos de `feat-agr`: los 5620 de `uci-optdigits-orig-32px-r20261005`, leídos por su nombre (no se regeneran).

⚠ LA ETIQUETA NO ENTRA AL AGRUPAR, y eso lo garantiza este módulo, no el cuidado: `cargar()` NO devuelve etiquetas.
Para tenerlas hay que llamar a `etiquetas()`, y eso sólo lo hacen la lectura (nn/leer.py), las figuras y las comprobaciones.
`--comprobar` falla si el código que agrupa (nn/representar.py, nn/entrenar_local.py) las pide.

    python nn/datos.py --comprobar     huellas del dataset; windep reducido = el 8×8 publicado; la `particion()` copiada da la
                                       `huella_particiones` de feat-ind32 y contiene los 67 fallos de C; los 14 bloques de
                                       windep siguen una sola plantilla; y que el agrupador no toca la etiqueta
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
sys.path.insert(0, str(EXP.parent))                       # el repo: donde vive `expcnn`
from expcnn import exigir_dataset, por_id                 # noqa: E402

DATASET = "uci-optdigits-orig-32px-r20261005"
DIGITOS_8PX = "uci-optdigits-8px-r20261002"               # sólo para COMPROBAR que la reducción 4×4 es la de NIST
# Los 14 bloques de windep, por su primer índice DENTRO de windep. Medidos el 2026-10-05 sobre las etiquetas: los 14 siguen
# la misma plantilla de 130 casillas (13 por clase), así que son, PROBABLEMENTE, un formulario por persona. Son 14 bloques
# para 13 escritores: «bloque = escritor» NO está comprobado. Los usa sólo la lectura; `--comprobar` los vuelve a medir.
BLOQUES_WINDEP = (0, 130, 256, 386, 516, 646, 776, 906, 1029, 1157, 1287, 1415, 1545, 1667)
# Lo que agrupa no puede pedir la etiqueta (ver arriba).
AGRUPADORES = ("representar.py", "entrenar_local.py")
PIDE_ETIQUETA = re.compile(r"datos\.etiquetas|\[[\"']etiquetas[\"']\]|evaluaciones_c|errores_c")

# COPIADO de `feat-ind32` (nn/ganancia.py): los mismos T, p y semillas, en el mismo orden, para reproducir su huella.
T_TAMANOS = (500, 1000, 2000, 4000)
FRACCIONES = (0.02, 0.04, 0.10, 0.20, 0.50)
SEMILLAS_GANANCIA = (1, 2, 3)
# Lo que evaluó C (`feat-ind32`, nn/errores_c.py): T = 4000, p = 50 %, semillas 1–3.
C_T, C_P, C_SEMILLAS = 4000, 0.5, (1, 2, 3)


def huella(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _crudo() -> dict:
    raiz = exigir_dataset(DATASET)
    man = json.loads((raiz / "manifiesto.json").read_text(encoding="utf-8"))
    z = dict(np.load(raiz / "datos.npz"))
    for k, h in man["huellas"].items():
        if huella(z[k]) != h:
            raise RuntimeError(f"{DATASET}/datos.npz: '{k}' no casa con su manifiesto. Me niego.")
    return z


def reducir(x32: np.ndarray) -> np.ndarray:
    """(N,1,32,32) 0/1 → (N,1,8,8) = cuentas por bloque 4×4 ÷ 16: la reducción de NIST (la de `feat-ind32`, curvas_cnn.dato)."""
    return (x32.reshape(-1, 1, 8, 4, 8, 4).sum((3, 5)) / 16).astype(np.float32)


def cargar() -> dict:
    """Lo que ve el AGRUPADOR, SIN etiquetas: x32 (N,1,32,32) 0/1, x8 (N,1,8,8) y el origen (windep, tra, cv, wdep)."""
    z = _crudo()
    x32 = z["imagenes"].astype(np.float32)[:, None]
    return {"x32": x32, "x8": reducir(x32), "origen": z["origen"].astype(str)}


def etiquetas() -> np.ndarray:
    """Las etiquetas de NIST. SÓLO para LEER los grupos y para comprobar: nunca para formarlos."""
    return _crudo()["etiquetas"].astype(np.int64)


def bloques(origen: np.ndarray) -> np.ndarray:
    """(N,) el bloque de windep de cada dígito (0..13), −1 fuera de windep. Posible formulario por persona, NO comprobado."""
    w = np.flatnonzero(origen == "windep")
    out = np.full(len(origen), -1, np.int64)
    lim = list(BLOQUES_WINDEP) + [len(w)]
    for b, (a, z) in enumerate(zip(lim, lim[1:])):
        out[w[a:z]] = b
    return out


def particion(y: np.ndarray, T: int, p: float, sem: int) -> tuple[np.ndarray, np.ndarray]:
    """COPIA LITERAL de `feat-ind32` (nn/ganancia.py), que a su vez la copió de `feat-ind`:
    (train, test) de un dataset BALANCEADO de T dígitos del pool (T/10 por clase); p·T/10 por clase a train."""
    por = T // 10; n_tr = int(round(p * por))
    if T % 10 or abs(n_tr - p * por) > 1e-9 or n_tr < 1:
        raise ValueError(f"T={T}, p={p}: p·T/10 tiene que ser un entero ≥ 1")
    rng = np.random.default_rng(1_000_000 * sem + T)
    tr, te = [], []
    for c in range(10):
        idx = rng.permutation(np.flatnonzero(y == c))[:por]
        tr.append(idx[:n_tr]); te.append(idx[n_tr:])
    return np.concatenate(tr), np.concatenate(te)


def huella_particiones(y: np.ndarray) -> str:
    """El mismo recorrido que `feat-ind32` (nn/ganancia.py) usa para su `huella_particiones`."""
    hp = hashlib.sha256()
    for T in T_TAMANOS:
        for p in FRACCIONES:
            for sem in SEMILLAS_GANANCIA:
                tr, te = particion(y, T, p, sem)
                hp.update(tr.tobytes()); hp.update(te.tobytes())
    return hp.hexdigest()[:16]


def errores_c() -> dict:
    """`resultados/errores-c.json` de feat-ind32, pedido al registro por su id: los dígitos que falla C."""
    return json.loads((por_id("feat-ind32").carpeta / "resultados" / "errores-c.json").read_text(encoding="utf-8"))


def evaluaciones_c(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Las 6000 evaluaciones de C (índice del dígito, con repetición: 3 semillas × 2000) y sus 67 fallos, POR EVALUACIÓN.
    Se niega si un fallo no cae en el test de su semilla, o si un test no tiene 2000."""
    e = errores_c()
    if (e["T"], e["p"]) != (C_T, C_P):
        raise SystemExit(f"✗ errores-c.json es de T={e['T']}, p={e['p']}, no de {C_T}/{C_P}. Me niego.")
    evals, fallos = [], []
    for s in e["semillas"]:
        _, te = particion(y, C_T, C_P, s["sem"])
        if len(te) != s["n_test"]:
            raise SystemExit(f"✗ semilla {s['sem']}: el test regenerado tiene {len(te)}, no {s['n_test']}. Me niego.")
        dentro = set(te.tolist())
        idx = [x["indice"] for x in s["errores"]]
        if not set(idx) <= dentro or any(y[i] != x["real"] for i, x in zip(idx, s["errores"])):
            raise SystemExit(f"✗ semilla {s['sem']}: hay fallos de C fuera de su test o con otra etiqueta. Me niego.")
        evals.append(te); fallos += idx
    return np.concatenate(evals), np.array(fallos, np.int64)


def _lcs(a: np.ndarray, b: np.ndarray) -> int:
    m = np.zeros((len(a) + 1, len(b) + 1), np.int64)
    for i in range(len(a)):
        for j in range(len(b)):
            m[i + 1, j + 1] = m[i, j] + 1 if a[i] == b[j] else max(m[i, j + 1], m[i + 1, j])
    return int(m[-1, -1])


def comprobar() -> int:
    ok = True
    z = _crudo(); y = etiquetas(); origen = z["origen"].astype(str)
    print(f"  [   ok] {DATASET}: huellas del manifiesto · {len(y)} dígitos · " +
          ", ".join(f"{o} {int((origen == o).sum())}" for o in ("windep", "tra", "cv", "wdep")))
    w = origen == "windep"
    z8 = np.load(exigir_dataset(DIGITOS_8PX) / "datos.npz")
    igual = bool((reducir(z["imagenes"].astype(np.float32)[:, None][w])[:, 0] * 16 == z8["imagenes"]).all())
    print(f"  [{'ok' if igual else 'FALLA':>5}] windep reducido 4×4 = {DIGITOS_8PX} (lo que ve el banco de 8×8)"); ok &= igual
    hp = huella_particiones(y)
    suya = json.loads((por_id("feat-ind32").carpeta / "resultados" / "ganancia.json").read_text(encoding="utf-8"))["huella_particiones"]
    print(f"  [{'ok' if hp == suya else 'FALLA':>5}] particion() copiada: huella {hp} = la de feat-ind32 ({suya})"); ok &= hp == suya
    ev, fa = evaluaciones_c(y)
    print(f"  [   ok] C evaluó {len(ev)} veces ({len(set(ev.tolist()))} dígitos distintos) y falló {len(fa)} "
          f"({len(set(fa.tolist()))} distintos); todos los fallos caen en el test de su semilla")
    yw = y[w]; lim = list(BLOQUES_WINDEP) + [len(yw)]
    bl = [yw[a:b] for a, b in zip(lim, lim[1:])]
    plantillas = [b for b in bl if len(b) == 130]
    vals, cuenta = np.unique(np.stack(plantillas), axis=0, return_counts=True)
    tpl = vals[cuenta.argmax()]
    lcs = [_lcs(b, tpl) for b in bl]
    bien = all(l >= len(b) - 1 for l, b in zip(lcs, bl)) and np.bincount(tpl, minlength=10).tolist() == [13] * 10
    print(f"  [{'ok' if bien else 'FALLA':>5}] windep: 14 bloques ({', '.join(str(len(b)) for b in bl)}) que siguen una plantilla"
          f" de 130 (13 por clase): {int(cuenta.max())} idénticos, LCS {lcs}"); ok &= bien
    malos = [f for f in AGRUPADORES if (AQUI / f).is_file() and PIDE_ETIQUETA.search((AQUI / f).read_text(encoding="utf-8"))]
    print(f"  [{'ok' if not malos else 'FALLA':>5}] el código que agrupa no pide la etiqueta"
          + (f": la piden {malos}" if malos else f" ({', '.join(AGRUPADORES)})")); ok &= not malos
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comprobar", action="store_true")
    a = p.parse_args()
    if a.comprobar:
        return comprobar()
    p.print_help(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
