#!/usr/bin/env python3
"""El dato de `ruido-nist`: el dataset `uci-optdigits-8px-r20261002` PUBLICADO por `dim-nist` (se lee
por su nombre, no se regenera), y el train de cada ESCENARIO = las 180 originales + 180 copias.

    python nn/datos.py --comprobar     huellas del manifiesto y congeladas; que val no pasa por el ruido

Lo que es fijo: val = las 1617 limpias, SIEMPRE las mismas (huella congelada, se comprueba en cada
carga). Train de un escenario = concat(originales/16, copia), 360 imágenes; en `limpio` la copia es
la original tal cual (las 180 duplicadas), así todos los escenarios ven el mismo número de pasos e
imágenes y lo único distinto es la copia. Las copias ocupan los índices 180–359: el orden de lotes
(que lo fija la semilla de lotes, fuera de aquí) se aplica igual en todos los escenarios.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
sys.path.insert(0, str(EXP.parent))                       # el repo: donde vive `expcnn`
sys.path.insert(0, str(AQUI))
from expcnn import exigir_dataset  # noqa: E402
import ruido                       # noqa: E402

DATASET = "uci-optdigits-8px-r20261002"
BITS = ruido.BITS
N_TRAIN, N_VAL = 180, 1617
# Huellas CONGELADAS el 2026-10-02 (sha256, 16 hex) del dato publicado tal cual lo da el manifiesto
# y de lo que ESTE experimento deriva de él. `limpio()` SE NIEGA si no casan.
HUELLA_IMAGENES = "8f26b2bd9d135c25"      # uint8 (1797, 8, 8)
HUELLA_ETIQUETAS = "8ba4f891220f5e4c"     # uint8 (1797,)
HUELLA_PARTICION = "abbaa6fc4132c445"     # <U5 (1797,)
HUELLA_X_VAL = "57a6cb6086e0f956"           # float32 (1617, 1, 8, 8) = cuentas/16 de val: lo que NUNCA cambia
HUELLA_X_TRAIN = "dc7573f9099d5072"       # float32 (180, 1, 8, 8): las originales de train


def _huella(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def crudo() -> dict:
    raiz = exigir_dataset(DATASET)
    man = json.loads((raiz / "manifiesto.json").read_text(encoding="utf-8"))
    d = np.load(raiz / "datos.npz")
    x, y, part = d["imagenes"], d["etiquetas"], d["particion"]
    esperadas = {"imagenes": HUELLA_IMAGENES, "etiquetas": HUELLA_ETIQUETAS, "particion": HUELLA_PARTICION}
    for nombre, a in (("imagenes", x), ("etiquetas", y), ("particion", part)):
        h = _huella(a)
        if h != man["huellas"][nombre] or h != esperadas[nombre]:
            raise RuntimeError(f"{DATASET}/datos.npz: '{nombre}' da {h}; el manifiesto dice {man['huellas'][nombre]} "
                               f"y lo congelado aquí {esperadas[nombre]}: no es el dato publicado. Me niego.")
    return {"imagenes": x, "etiquetas": y, "particion": part, "manifiesto": man}


def fraccion(cuentas: np.ndarray) -> np.ndarray:
    """(N, 8, 8) uint8 0..16 -> (N, 1, 8, 8) float32 en [0, 1]: fracción de TINTA."""
    return (cuentas.astype(np.float32) / BITS)[:, None]


def limpio() -> dict:
    """Las 180 de train y las 1617 de val, limpias, como CUENTAS (uint8) y etiquetas int64."""
    d = crudo()
    tr = d["particion"] == "train"
    va = ~tr
    if tr.sum() != N_TRAIN or va.sum() != N_VAL:
        raise RuntimeError(f"reparto {tr.sum()}/{va.sum()}: el dataset ya no es 180/1617")
    y = d["etiquetas"].astype(np.int64)
    out = {"cuentas_train": d["imagenes"][tr], "y_train": y[tr], "cuentas_val": d["imagenes"][va], "y_val": y[va]}
    hv, ht = _huella(fraccion(out["cuentas_val"])), _huella(fraccion(out["cuentas_train"]))
    if hv != HUELLA_X_VAL or ht != HUELLA_X_TRAIN:
        raise RuntimeError(f"val da {hv} (congelada {HUELLA_X_VAL}) y train {ht} (congelada {HUELLA_X_TRAIN}): "
                           f"no es el mismo dato. Me niego.")
    return out


def escenario(nombre: str) -> dict:
    """El dato de UN escenario: x_train (360, 1, 8, 8) = originales + copia; x_val limpio; huellas.
    `limpio`: la copia es la original (las 180 duplicadas)."""
    tipo, nivel, r = ruido.parsear(nombre)
    d = limpio()
    orig = fraccion(d["cuentas_train"])
    if tipo == ruido.LIMPIO:
        cop, s_ruido = orig.copy(), None
    else:
        c, s_ruido = ruido.copia(tipo, nivel, d["cuentas_train"], r)
        cop = c[:, None]
    x_train = np.concatenate([orig, cop], axis=0)
    y_train = np.concatenate([d["y_train"], d["y_train"]], axis=0)
    x_val = fraccion(d["cuentas_val"])
    assert x_train.shape == (2 * N_TRAIN, 1, 8, 8) and x_val.shape == (N_VAL, 1, 8, 8)
    assert x_train.dtype == np.float32 and float(x_train.min()) >= 0 and float(x_train.max()) <= 1
    return {"escenario": nombre, "tipo": tipo, "nivel": nivel, "realizacion": r, "semilla_ruido": s_ruido,
            "dataset": DATASET,
            "x_train": x_train, "y_train": y_train,
            "x_train_limpio": orig, "y_train_limpio": d["y_train"],
            "x_val": x_val, "y_val": d["y_val"],
            "huella_copia": _huella(cop), "huella_x_val": _huella(x_val), "huella_x_train_limpio": _huella(orig),
            "tinta_media_copia": float(cop.mean()), "tinta_media_original": float(orig.mean())}


def piso() -> float:
    """Adivinar al azar entre 10 clases (18 por clase en train: la mayoritaria no existe)."""
    return 0.1


def comprobar() -> int:
    ok = True

    def linea(que, bien, det=""):
        nonlocal ok
        ok &= bool(bien)
        print(f"  [{'ok' if bien else 'FALLA':>5}] {que}" + (f"  {det}" if det else ""))

    d = crudo()
    print(f"dataset {DATASET}: {len(d['etiquetas'])} imagenes, manifiesto casado")
    l = limpio()
    linea("180 train / 1617 val, 18 por clase en train", all((l["y_train"] == c).sum() == 18 for c in range(10)))
    linea("cuentas uint8 en 0..16", l["cuentas_train"].dtype == np.uint8 and int(l["cuentas_train"].max()) <= 16)
    linea("huella de val y de train limpio congeladas", True, f"val {_huella(fraccion(l['cuentas_val']))} · train {_huella(fraccion(l['cuentas_train']))}")
    e0 = escenario("limpio")
    linea("limpio: 360 = 180 + las mismas 180", bool((e0["x_train"][:180] == e0["x_train"][180:]).all()) and e0["huella_copia"] == e0["huella_x_train_limpio"])
    for tipo in ruido.TIPOS:
        nivel = ruido.NIVELES[tipo][ruido.INDICE_MEDIO]
        e = escenario(ruido.escenario(tipo, nivel))
        e2 = escenario(ruido.escenario(tipo, nivel))
        cambia = not np.array_equal(e["x_train"][180:], e["x_train"][:180])
        linea(f"{e['escenario']:<18} val intacta · originales intactas · copia distinta · determinista",
              e["huella_x_val"] == HUELLA_X_VAL and np.array_equal(e["x_train"][:180], e0["x_train"][:180])
              and cambia and e["huella_copia"] == e2["huella_copia"],
              f"copia {e['huella_copia']} · tinta {e['tinta_media_original']:.3f} → {e['tinta_media_copia']:.3f} · semilla {e['semilla_ruido']}")
    print("\nel dato esta en orden." if ok else "\n✗ ALGO NO CASA: no se entrena sobre esto.")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comprobar", action="store_true")
    p.add_argument("--huellas", action="store_true", help="imprime las huellas de val/train limpio (para congelarlas)")
    a = p.parse_args()
    if a.huellas:
        d = crudo(); tr = d["particion"] == "train"
        print("HUELLA_X_VAL", _huella(fraccion(d["imagenes"][~tr])), "HUELLA_X_TRAIN", _huella(fraccion(d["imagenes"][tr])))
        return 0
    if a.comprobar:
        return comprobar()
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
