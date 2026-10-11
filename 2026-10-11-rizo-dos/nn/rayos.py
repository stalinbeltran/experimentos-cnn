#!/usr/bin/env python3
"""rizo-2 — estrategia R: RAYOS para rizos abiertos y cerrados.

Un píxel de fondo es centro de un rizo si, de 36 rayos (cada 10°), una fracción ≥ F choca con tinta antes de RMAX px
Y las distancias al choque son parejas (p90 − p10 ≤ DISP): un anillo alrededor, no tinta suelta; y el radio mediano
≥ 1,25 px (una muesca de un píxel en un trazo no es un rizo). Los píxeles que lo
cumplen se agrupan (vecindad 8); cada grupo es un rizo: centro = el píxel de mejor puntuación del grupo (fracción que
choca, y a igualdad menor dispersión), radio = mediana del punto medio del trazo a lo largo de cada rayo (del choque a la salida), cierre = (fracción − 0,5)/0,5,
abertura = dirección media de los rayos que no chocan. Sin morfología: es una medida geométrica sobre la tinta.

    python nn/rayos.py   → resultados/cal-rayos.json (la calibración de F, RMAX y DISP con sintéticos, semilla 101)
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import warnings

import numpy as np
from scipy.ndimage import label

warnings.filterwarnings("ignore", "All-NaN")

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn"))
import sinteticos as SN                                             # noqa: E402

ANG = np.deg2rad(np.arange(36) * 10 + 5); CA, SA = np.cos(ANG), np.sin(ANG)
REJILLA = {"f": (0.6, 0.7, 0.8, 0.9), "rmax": (6.0, 8.0), "disp": (1.5, 2.5, 4.0)}
DEFECTO = {"f": 0.7, "rmax": 8.0, "disp": 2.5}
RADIO_MIN = 1.25          # un hueco de un píxel (una muesca en el trazo) no es un rizo: el radio mediano tiene que pasar de esto


def _rayos(x, ys, xs, rmax):
    ps = np.arange(0.5, rmax + 0.01, 0.25)
    cx, cy = xs + 0.5, ys + 0.5
    px = cx[:, None, None] + CA[None, :, None] * ps[None, None, :]
    py = cy[:, None, None] + SA[None, :, None] * ps[None, None, :]
    ix, iy = np.floor(px).astype(int), np.floor(py).astype(int)
    ok = (ix >= 0) & (ix < x.shape[1]) & (iy >= 0) & (iy < x.shape[0])
    hit = np.zeros(ix.shape, bool); hit[ok] = x[iy[ok], ix[ok]]
    choca = hit.any(2)
    k1 = np.argmax(hit, axis=2); dist = np.where(choca, ps[k1], np.nan)
    # dónde SALE del trazo el rayo (primer paso sin tinta después del choque): el radio al centro del trazo es la media
    despues = np.cumsum(hit, axis=2) > 0                      # True desde el choque en adelante
    sale = despues & ~hit; k2 = np.where(sale.any(2), np.argmax(sale, axis=2), hit.shape[2] - 1)
    medio = np.where(choca, (ps[k1] + ps[k2]) / 2, np.nan)
    return choca, dist, medio


def rizos(x: np.ndarray, f: float = DEFECTO["f"], rmax: float = DEFECTO["rmax"], disp: float = DEFECTO["disp"]) -> list[dict]:
    x = np.asarray(x) > 0
    ys, xs = np.nonzero(~x)
    if len(ys) == 0:
        return []
    choca, dist, _ = _rayos(x, ys, xs, rmax)
    frac = choca.mean(1)
    with np.errstate(all="ignore"):
        p10, p90 = np.nanpercentile(dist, 10, axis=1), np.nanpercentile(dist, 90, axis=1)
    with np.errstate(all="ignore"):
        med = np.nanmedian(dist, axis=1)
    ok = (frac >= f) & ((p90 - p10) <= disp) & (np.nan_to_num(med) >= RADIO_MIN)
    m = np.zeros(x.shape, bool); m[ys[ok], xs[ok]] = True
    punt = np.zeros(x.shape); punt[ys, xs] = frac - 0.01 * np.nan_to_num(p90 - p10)
    lab, n = label(m, structure=np.ones((3, 3)))
    out = []
    for z in range(1, n + 1):
        gy, gx = np.nonzero(lab == z)
        k = int(np.argmax(punt[gy, gx])); cy, cx = gy[k] + 0.5, gx[k] + 0.5
        ch, di, me = _rayos(x, np.array([gy[k]]), np.array([gx[k]]), rmax)
        fr = float(ch[0].mean()); libres = ANG[~ch[0]]
        ori = float(np.degrees(np.arctan2(np.sin(libres).mean(), np.cos(libres).mean())) % 360) if len(libres) else None
        out.append({"centro": (float(cx), float(cy)), "radio": float(np.nanmedian(me[0])), "radio_hueco": float(np.nanmedian(di[0])),
                    "cierre": float(np.clip((fr - 0.5) / 0.5, 0, 1)), "orientacion": ori, "puntuacion": fr, "pixeles": int(len(gy))})
    return sorted(out, key=lambda r: -r["puntuacion"])


def puntuar(conj: list[dict], det) -> dict:
    """Puntuación de calibración: media de % encontrados (≤ 2,5 px) y % negativos limpios; y rizos de más por positivo."""
    enc, lim, extra = [], [], []
    for f in conj:
        r = det(f["x"])
        if f["tipo"] == "rizo":
            d = [np.hypot(c["centro"][0] - f["centro"][0], c["centro"][1] - f["centro"][1]) for c in r]
            enc.append(any(e <= 2.5 for e in d)); extra.append(sum(e > 2.5 for e in d))
        else:
            lim.append(not r)
    return {"encontrados": round(100 * float(np.mean(enc)), 1), "negativos_limpios": round(100 * float(np.mean(lim)), 1),
            "extra_por_positivo": round(float(np.mean(extra)), 2),
            "puntos": round(50 * float(np.mean(enc)) + 50 * float(np.mean(lim)) - 2 * float(np.mean(extra)), 2)}


def calibrar(conj: list[dict]) -> dict:
    out, mejor = [], None
    for f, rmax, disp in itertools.product(*REJILLA.values()):
        r = puntuar(conj, lambda x: rizos(x, f, rmax, disp)); r.update(f=f, rmax=rmax, disp=disp); out.append(r)
        if mejor is None or r["puntos"] > mejor["puntos"]:
            mejor = r
    return {"rejilla": out, "elegida": {"f": mejor["f"], "rmax": mejor["rmax"], "disp": mejor["disp"]}, "mejor": mejor}


def cargar_calibracion() -> dict:
    p = AQUI / "resultados" / "cal-rayos.json"
    return json.loads(p.read_text())["elegida"] if p.exists() else dict(DEFECTO)


def main() -> int:
    t0 = time.time(); cal = SN.conjunto(150, 150, 101); r = calibrar(cal)
    (AQUI / "resultados").mkdir(exist_ok=True)
    (AQUI / "resultados" / "cal-rayos.json").write_text(json.dumps(r, indent=1) + "\n")
    print(f"rayos: elegida {r['elegida']} → {r['mejor']}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
