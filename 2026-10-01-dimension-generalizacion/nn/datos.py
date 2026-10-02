#!/usr/bin/env python3
"""El dato de `dim-gen`: el dataset publicado, recortado a 128 y reducido a `W` por suma EXACTA
de bloques. NADA se publica: las reducciones son pipeline (REGLAS.md § Entradas), y lo que las
hace «el mismo dato» es que su huella está CONGELADA aquí y se exige al cargar.

    python nn/datos.py --comprobar     huellas del manifiesto, huellas congeladas de cada W,
                                       aserciones, tamaños de caja por W y el piso de caja media

Qué se transforma, en orden (y por qué, en REGLAS.md):
  1. recorte central 146 -> 128 (9 px por lado): el marco de las etiquetas, y 128 = 2^7
  2. suma de bloques b x b, b = 128/W, acumulando en int64 (a W <= 16 no cabe en uint16)
  3. x = 1 - suma / (255 * 16 * b^2): fraccion de tinta por pixel, igual a todo W
  4. etiqueta = ((c/4) - 9) / 128, la MISMA para todo W
  5. el control w128-de16: la matriz de W = 16 repetida 8x8 hasta 128
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
from expcnn import exigir_dataset  # noqa: E402

DATASET = "parrafos1000-584px-r4-r20260908b"
MARCO = 146            # como se guarda: 584 / 4
DESCARTE = 9           # por lado, para llegar al marco de las etiquetas
FINAL = 128            # MARCO - 2 * DESCARTE
SUMA_GUARDADA = 16     # cada pixel guardado es la suma de un bloque 4x4 del lienzo (max 4080)
W_TODOS = (128, 64, 32, 16, 8)
PARTES = ("train", "monitor", "eval")

# Huellas CONGELADAS el 2026-10-01 sobre los .npz publicados: sha256 de la matriz int64 de sumas
# (1000, W, W) en el orden train, monitor, eval tal como vienen, bytes little-endian ('<i8'),
# 16 hex. Si una no casa, `cargar` SE NIEGA (R2): entrenar sobre otro dato sin saberlo es el
# fallo silencioso que estas lineas existen para evitar.
HUELLAS = {
    128: "a53caf6cf1d7527c",
    64: "6ba56079c36e8089",
    32: "0d9dd6a70fcf496e",
    16: "db69ac6f091f289b",
    8: "27e1978cf167f46b",
}
# Las etiquetas normalizadas (float64 '<f8'), mismo orden.
HUELLA_ETIQUETAS = "36ca489668552b90"


def _huella(a: np.ndarray) -> str:
    """La del manifiesto del dataset (la misma receta con la que se publico)."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _huella_i8(a: np.ndarray) -> str:
    """La CONGELADA de arriba: bytes little-endian de int64."""
    return hashlib.sha256(np.ascontiguousarray(a).astype("<i8").tobytes()).hexdigest()[:16]


def _huella_f8(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).astype("<f8").tobytes()).hexdigest()[:16]


def crudo() -> dict:
    """Las tres particiones publicadas, apiladas en orden train, monitor, eval, con la
    huella de cada una casada contra el manifiesto. Se niega si algo no casa."""
    raiz = exigir_dataset(DATASET)
    man = json.loads((raiz / "manifiesto.json").read_text(encoding="utf-8"))
    imgs, etiq, idx, partes = [], [], [], []
    for p in PARTES:
        d = np.load(raiz / f"{p}.npz")
        esperada = man["particiones"][p]
        hi, he = _huella(d["imagenes"]), _huella(d["etiquetas"])
        if hi != esperada["sha256_16"] or he != esperada["sha256_16_etiquetas"]:
            raise RuntimeError(
                f"{DATASET}/{p}.npz no casa con su manifiesto: imagenes {hi} (esperada "
                f"{esperada['sha256_16']}), etiquetas {he} (esperada "
                f"{esperada['sha256_16_etiquetas']}). No es el dato publicado.")
        imgs.append(d["imagenes"]); etiq.append(d["etiquetas"]); idx.append(d["indices"])
        partes += [p] * len(d["imagenes"])
    meta = json.loads(str(np.load(raiz / "meta.npz")["meta"]))
    return {"imagenes": np.concatenate(imgs), "etiquetas": np.concatenate(etiq),
            "indices": np.concatenate(idx), "partes": np.array(partes), "meta": meta,
            "manifiesto": man}


def recortar(img146: np.ndarray) -> np.ndarray:
    """(N,146,146) uint16 -> (N,128,128) int64. El recorte central: 9 px por lado."""
    assert img146.shape[-2:] == (MARCO, MARCO), img146.shape
    return img146[..., DESCARTE:DESCARTE + FINAL, DESCARTE:DESCARTE + FINAL].astype(np.int64)


def sumas(c128: np.ndarray, W: int) -> np.ndarray:
    """(N,128,128) int64 -> (N,W,W) int64: la suma EXACTA de cada bloque b x b."""
    if FINAL % W:
        raise ValueError(f"W={W} no divide a {FINAL}: no es una reduccion exacta")
    b = FINAL // W
    n = c128.shape[0]
    s = c128.reshape(n, W, b, W, b).sum(axis=(2, 4), dtype=np.int64)
    assert s.sum() == c128.sum(), "la suma no se conserva: la reduccion no es exacta"
    return s


def fraccion(sum_w: np.ndarray, W: int) -> np.ndarray:
    """La fraccion de TINTA por pixel, (N,1,W,W) float32 en [0,1]: papel 0, tinta 1."""
    b = FINAL // W
    x = 1.0 - sum_w.astype(np.float64) / (255.0 * SUMA_GUARDADA * b * b)
    return x.astype(np.float32)[:, None]


def etiquetas_norm(e584: np.ndarray) -> np.ndarray:
    """(N,4) int32 en el marco de 584 -> (N,4) float32 en [0,1] del marco de 128.
    Columnas: izq, der, sup, inf."""
    return (((e584.astype(np.float64) / 4.0) - DESCARTE) / FINAL).astype(np.float32)


def control_de16(x16: np.ndarray) -> np.ndarray:
    """(N,1,16,16) -> (N,1,128,128): cada pixel de 16 es un bloque uniforme de 8x8."""
    assert x16.shape[-2:] == (16, 16), x16.shape
    return np.repeat(np.repeat(x16, 8, axis=-2), 8, axis=-1)


def factores(d: dict, mascara: np.ndarray) -> dict:
    """Los factores del generador para las imagenes de `mascara`, por `indices` -> meta[i].
    Solo para el desglose del resultado: nunca entran en el entrenamiento."""
    meta = d["meta"]
    ids = d["indices"][mascara]
    e = etiquetas_norm(d["etiquetas"][mascara])
    area = (e[:, 1] - e[:, 0]) * (e[:, 3] - e[:, 2])
    return {
        "fuente": np.array([meta[i]["fuente"] for i in ids]),
        "cuerpo": np.array([meta[i]["cuerpo"] for i in ids], dtype=np.float64),
        "gris_nivel": np.array([meta[i]["gris_nivel"] for i in ids], dtype=np.float64),
        "area": area.astype(np.float64),
    }


def iou(pred: np.ndarray, real: np.ndarray) -> np.ndarray:
    """IoU por fila entre cajas (izq, der, sup, inf) normalizadas. La prediccion se recorta a
    [0,1]; una caja invertida (der <= izq o inf <= sup) vale 0."""
    p = np.clip(pred.astype(np.float64), 0.0, 1.0)
    r = real.astype(np.float64)
    valida = (p[:, 1] > p[:, 0]) & (p[:, 3] > p[:, 2])
    ix = np.maximum(0.0, np.minimum(p[:, 1], r[:, 1]) - np.maximum(p[:, 0], r[:, 0]))
    iy = np.maximum(0.0, np.minimum(p[:, 3], r[:, 3]) - np.maximum(p[:, 2], r[:, 2]))
    inter = ix * iy
    ap = (p[:, 1] - p[:, 0]) * (p[:, 3] - p[:, 2])
    ar = (r[:, 1] - r[:, 0]) * (r[:, 3] - r[:, 2])
    out = inter / np.maximum(ap + ar - inter, 1e-12)
    return np.where(valida, out, 0.0)


def piso_caja_media(y_train: np.ndarray, y_val: np.ndarray) -> float:
    """Predecir siempre la caja MEDIA de train: el piso del criterio."""
    m = np.broadcast_to(y_train.mean(0), y_val.shape)
    return float(iou(m, y_val).mean())


def cargar(W: int, control: bool = False) -> dict:
    """Todo lo que un brazo necesita, con las huellas EXIGIDAS. `control=True` es
    `w128-de16`: la informacion de 16 con la forma de 128."""
    if W not in W_TODOS:
        raise ValueError(f"W={W} no esta en {W_TODOS}")
    if control and W != 128:
        raise ValueError("el control es w128-de16: W tiene que ser 128")
    d = crudo()
    c = recortar(d["imagenes"])
    w_dato = 16 if control else W
    s = sumas(c, w_dato)
    h = _huella_i8(s)
    if h != HUELLAS[w_dato]:
        raise RuntimeError(f"la reduccion a W={w_dato} da {h} y la huella congelada es "
                           f"{HUELLAS[w_dato]}: NO es el mismo dato reducido. Me niego.")
    x = fraccion(s, w_dato)
    if control:
        x = control_de16(x)
    y = etiquetas_norm(d["etiquetas"])
    hy = _huella_f8(y)
    if hy != HUELLA_ETIQUETAS:
        raise RuntimeError(f"las etiquetas normalizadas dan {hy}, congelada {HUELLA_ETIQUETAS}")
    tr = d["partes"] == "train"
    va = ~tr
    assert tr.sum() == 100 and va.sum() == 900, (tr.sum(), va.sum())
    return {
        "W": W, "control": control, "dataset": DATASET,
        "x_train": x[tr], "y_train": y[tr], "x_val": x[va], "y_val": y[va],
        "partes_val": d["partes"][va], "factores_val": factores(d, va),
        "huella_x": h, "huella_y": hy,
    }


def comprobar() -> int:
    ok = True

    def linea(que: str, bien: bool, det: str = "") -> None:
        nonlocal ok
        ok &= bien
        print(f"  [{'ok' if bien else 'FALLA':>5}] {que}" + (f"  {det}" if det else ""))

    d = crudo()
    print(f"dataset {DATASET}: {len(d['imagenes'])} imagenes, manifiesto casado (3 particiones)")
    c = recortar(d["imagenes"])
    linea("recorte 146 -> 128 y etiquetas dentro del marco", True,
          f"etiquetas@128 en [{(d['etiquetas'] / 4 - 9).min():.1f}, {(d['etiquetas'] / 4 - 9).max():.2f}]")
    for W in W_TODOS:
        s = sumas(c, W)
        h = _huella_i8(s)
        linea(f"W={W:>3}: huella congelada", h == HUELLAS[W], f"{h} (max suma {s.max()})")
    s128 = sumas(c, 128)
    linea("W=128 es exactamente el recorte", bool((s128 == c).all()))
    y = etiquetas_norm(d["etiquetas"])
    linea("etiquetas normalizadas: huella y rango [0,1]",
          _huella_f8(y) == HUELLA_ETIQUETAS and y.min() >= 0 and y.max() <= 1,
          f"{_huella_f8(y)} rango [{y.min():.4f}, {y.max():.4f}]")
    # ⚠ un acumulador uint16 tiene que dar OTRA huella a W<=16 (R14: lo que puede fallar en
    # silencio se prueba). Si diera la misma, la comprobacion no protegeria de nada.
    b = FINAL // 16
    mal = c.astype(np.uint16).reshape(len(c), 16, b, 16, b).sum(axis=(2, 4), dtype=np.uint16)
    linea("un acumulador uint16 a W=16 se detecta (otra huella)",
          _huella_i8(mal.astype(np.int64)) != HUELLAS[16])
    x16 = fraccion(sumas(c, 16), 16)
    ctl = control_de16(x16)
    linea("control w128-de16: (N,1,128,128) y bloques 8x8 uniformes",
          ctl.shape == (len(c), 1, 128, 128) and bool((ctl[:, :, ::8, ::8] == x16).all()))
    tr = d["partes"] == "train"
    piso = piso_caja_media(y[tr], y[~tr])
    linea("piso de caja media sobre las 900", abs(piso - 0.2464) < 0.001, f"{piso:.4f} (escrito: 0,2464)")
    print("\ntamano de las cajas por W (px):")
    w_px = (y[:, 1] - y[:, 0]); h_px = (y[:, 3] - y[:, 2])
    for W in W_TODOS:
        a, al = w_px * W, h_px * W
        print(f"  W={W:>3}: ancho {a.min():5.1f} · {np.median(a):5.1f} · {a.max():5.1f}   "
              f"alto {al.min():5.1f} · {np.median(al):5.1f} · {al.max():5.1f}   "
              f"alto<2px {100 * (al < 2).mean():4.1f} %")
    print("\nel dato esta en orden." if ok else "\n✗ ALGO NO CASA: no se entrena sobre esto.")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comprobar", action="store_true")
    a = p.parse_args()
    if a.comprobar:
        return comprobar()
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
