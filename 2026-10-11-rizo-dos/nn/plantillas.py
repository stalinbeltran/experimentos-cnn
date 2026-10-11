#!/usr/bin/env python3
"""rizo-2 — estrategia P: correlación con PLANTILLAS de rizo (un «compositor» de un solo nivel: el rizo entero como
patrón, con signo).

Banco: radio R ∈ {2,5 … 7} × grosor g ∈ {3, 4, 5} × cobertura ∈ {0,55; 0,7; 0,85; 1,0} × abertura cada 30°. Cada
plantilla son tres máscaras: el TRAZO (el arco), el HUECO (el interior) y el VANO (lo que el arco deja sin cubrir).
Puntuación en cada posición = fracción del trazo con tinta − fracción del hueco con tinta − ½ · fracción del vano con
tinta; se toma la mejor plantilla por posición, los máximos locales ≥ S que no caen sobre tinta, y se quitan los
candidatos a menos de R/2 de otro mejor. Todo con una convolución de torch (1152 plantillas, 21×21).

    python nn/plantillas.py   → resultados/cal-plantillas.json (la calibración de S con sintéticos, semilla 101)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn"))
import sinteticos as SN                                             # noqa: E402

K = 21
RADIOS, GROSORES, COBS, ABERTURAS = (2.5, 3, 3.5, 4, 4.5, 5, 6, 7), (3, 4, 5), (0.55, 0.7, 0.85, 1.0), tuple(range(0, 360, 30))
REJILLA_S = (0.5, 0.6, 0.7, 0.8)
DEFECTO_S = 0.7
_YY, _XX = np.mgrid[0:K, 0:K] + 0.5 - K / 2


def _plantilla(R, g, cob, ab):
    d = np.hypot(_XX, _YY); a = np.degrees(np.arctan2(_YY, _XX))
    anillo = np.abs(d - R) <= g / 2
    lejos = np.abs((a - ab + 180) % 360 - 180) > (1 - cob) * 180
    trazo = anillo & lejos; vano = anillo & ~lejos; hueco = d < max(R - g / 2 - 0.3, 0.5)
    return trazo, hueco, vano


def banco():
    tr, hu, va, meta = [], [], [], []
    for R in RADIOS:
        for g in GROSORES:
            if R - g / 2 < 0.5:                                # el mismo hueco mínimo que los sintéticos y que R
                continue
            for cob in COBS:
                for ab in (ABERTURAS if cob < 0.99 else (0,)):
                    t, h, v = _plantilla(R, g, cob, ab)
                    tr.append(t / max(t.sum(), 1)); hu.append(h / max(h.sum(), 1)); va.append(v / max(v.sum(), 1))
                    meta.append({"radio": R, "grosor": g, "cierre": (cob - 0.5) / 0.5, "abertura": ab if cob < 0.99 else None})
    w = np.stack(tr) - np.stack(hu) - 0.5 * np.stack(va)
    return torch.from_numpy(w.astype(np.float32))[:, None], meta


BANCO, META = banco()


@torch.no_grad()
def mapa(x: np.ndarray):
    r = F.conv2d(torch.from_numpy((np.asarray(x) > 0).astype(np.float32))[None, None], BANCO, padding=K // 2)[0]
    best, idx = r.max(0)
    return best.numpy(), idx.numpy()


def rizos(x: np.ndarray, s: float = DEFECTO_S) -> list[dict]:
    x = np.asarray(x) > 0
    best, idx = mapa(x)
    pico = (best == maximum_filter(best, size=3)) & (best >= s) & ~x
    ys, xs = np.nonzero(pico)
    out = []
    for i in np.argsort(-best[ys, xs]):
        cx, cy = xs[i] + 0.5, ys[i] + 0.5; m = META[idx[ys[i], xs[i]]]
        if any(np.hypot(cx - o["centro"][0], cy - o["centro"][1]) < o["radio"] / 2 for o in out):
            continue
        out.append({"centro": (float(cx), float(cy)), "radio": float(m["radio"]), "cierre": float(m["cierre"]),
                    "orientacion": None if m["abertura"] is None else float(m["abertura"]), "puntuacion": float(best[ys[i], xs[i]])})
    return out


def cargar_calibracion() -> float:
    p = AQUI / "resultados" / "cal-plantillas.json"
    return json.loads(p.read_text())["elegida"]["s"] if p.exists() else DEFECTO_S


def main() -> int:
    from rayos import puntuar
    t0 = time.time(); cal = SN.conjunto(150, 150, 101); out, mejor = [], None
    for s in REJILLA_S:
        r = puntuar(cal, lambda x: rizos(x, s)); r["s"] = s; out.append(r)
        if mejor is None or r["puntos"] > mejor["puntos"]:
            mejor = r
    res = {"rejilla": out, "elegida": {"s": mejor["s"]}, "mejor": mejor, "plantillas": int(BANCO.shape[0])}
    (AQUI / "resultados").mkdir(exist_ok=True)
    (AQUI / "resultados" / "cal-plantillas.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"plantillas ({BANCO.shape[0]}): elegida S {mejor['s']} → {mejor}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
