#!/usr/bin/env python3
"""La SONDA de grosor, SIN entrenar nada (2026-10-06): ¿el borde es independiente del grosor? Cada feature se dibuja con
la MISMA geometría (la misma instancia de features.py) a grosor 3 y a 6, 9 y 12 px, y se compara la representación de 3 px
con la de cada grosor: coseno tras suavizar con una gaussiana de σ = 1,5 px,

  · sin desplazar  (¿son los mismos píxeles, ± 1–2 px?)
  · admitiendo un desplazamiento de hasta ± 6 px (lo que una red convolucional absorbe); en `signo`, cada lado busca el suyo

y cuántos píxeles enciende cada representación. Es la medida que decidió entrenar las DOS variantes de borde: el contorno
sin signo pierde la forma al engrosar (un trazo grueso tiene dos bordes) y el borde con signo la conserva lado a lado.

    python nn/sonda_grosor.py      → resultados/sonda-grosor.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import bordes as B                              # noqa: E402
import features as F                            # noqa: E402

GROSORES = (3, 6, 9, 12)
POR_FAMILIA = {"sin desplazar": 40, "con desplazamiento": 15}     # el segundo es 169 veces más caro por par
SIGMA, RADIO = 1.5, 6
_K = np.exp(-np.arange(-4, 5) ** 2 / (2 * SIGMA ** 2)); _K /= _K.sum()


def suave(a: np.ndarray) -> np.ndarray:
    a = np.apply_along_axis(lambda v: np.convolve(v, _K, "same"), 0, a.astype(np.float64))
    return np.apply_along_axis(lambda v: np.convolve(v, _K, "same"), 1, a)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / n) if n > 0 else 0.0


def cos_desplazado(a: np.ndarray, b: np.ndarray, r: int = RADIO) -> float:
    A = np.pad(suave(a), r); Bs = suave(b); mejor = 0.0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            mejor = max(mejor, cos(A[r + dy:r + dy + 32, r + dx:r + dx + 32], Bs))
    return mejor


def medir(desplazar: bool, n: int) -> dict:
    F.usar_perfil("grueso")                        # para que quepan los gruesos; el grosor se fija a mano
    rng = np.random.default_rng(1)
    sim = {rep: {w: [] for w in GROSORES[1:]} for rep in B.REPRESENTACIONES}
    pix = {rep: {w: [] for w in GROSORES} for rep in B.REPRESENTACIONES}
    for fam in F.CON_TRAZO:
        hechos = 0
        while hechos < n:
            inst = F.instancia(rng, fam)
            if inst is None:
                continue
            ims = {}
            for w in GROSORES:
                inst["grosor"] = w; ims[w] = F.rasterizar(inst)
            if any(v.sum() == 0 for v in ims.values()):
                continue
            for rep in B.REPRESENTACIONES:
                r = {w: B.bordes(ims[w][None], rep)[0] for w in GROSORES}           # (C, 32, 32)
                for w in GROSORES:
                    pix[rep][w].append(float(r[w].sum()))
                for w in GROSORES[1:]:
                    if desplazar:      # cada canal con su desplazamiento: cada lado se mueve hacia su lado
                        s = [cos_desplazado(r[3][k], r[w][k]) for k in range(len(r[w])) if r[3][k].sum() and r[w][k].sum()]
                    else:
                        s = [cos(np.concatenate([suave(c).ravel() for c in r[3]]), np.concatenate([suave(c).ravel() for c in r[w]]))]
                    sim[rep][w].append(float(np.mean(s)))
            hechos += 1
    F.usar_perfil("fino")
    return {"instancias": n * len(F.CON_TRAZO),
            "coseno_con_3px": {rep: {str(w): round(float(np.mean(v)), 3) for w, v in d.items()} for rep, d in sim.items()},
            "pixeles_encendidos": {rep: {str(w): round(float(np.mean(v)), 1) for w, v in d.items()} for rep, d in pix.items()}}


def main() -> int:
    out = {"sigma_px": SIGMA, "desplazamiento_max_px": RADIO,
           "sin desplazar": medir(False, POR_FAMILIA["sin desplazar"]),
           "con desplazamiento": medir(True, POR_FAMILIA["con desplazamiento"])}
    destino = AQUI.parent / "resultados" / "sonda-grosor.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for modo in ("sin desplazar", "con desplazamiento"):
        print(f"{modo} ({out[modo]['instancias']} instancias): coseno con la misma geometría a 3 px")
        for rep in B.REPRESENTACIONES:
            c = out[modo]["coseno_con_3px"][rep]; p = out[modo]["pixeles_encendidos"][rep]
            print(f"  {rep:<9} w=6 {c['6']:.3f} · w=9 {c['9']:.3f} · w=12 {c['12']:.3f}   píxeles w=3/12: {p['3']:.0f} / {p['12']:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
