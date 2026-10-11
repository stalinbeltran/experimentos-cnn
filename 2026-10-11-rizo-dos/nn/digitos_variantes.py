#!/usr/bin/env python3
"""rizo-2 — el compositor con VARIAS salidas del detector de curvas cerradas, cada una como feature aparte (pedido del
dueño, 2026-10-11: «alimentar el compositor con cada uno de ellos como un feature separado, pues cada uno tiene su
especialidad»).

Los candidatos del compuesto (plantillas rodeadas ∪ rayos) se calculan UNA vez por dígito, con el hueco medido en su
centro (como `cerradas_hueco.py`) y el encierro en ese centro con rayos de 8, 14 y 18 px. Cada VARIANTE es un filtro
(h mínimo, alcance de rayos, encierro mínimo) que deja sus candidatos, y da dos mapas 32×32 (campana σ 1,5 de altura =
encierro en cada candidato, y otra sólo con el mejor) → celdas 8×8 = 128 características. Se entrena el compositor
lineal del boceto (180 train / 1617 val / 3823 a ciegas, 3 semillas) con: otros · otros + cada variante · otros + todas.

    python nn/digitos_variantes.py   → resultados/15-variantes.json · 15-variantes-fallos.png · 15-variantes-mapas.png  (~25 min)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as edt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
BOCETO = AQUI.parent / "docs" / "bocetos" / "2026-10-09-curvas-gabor"
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent)); sys.path.insert(0, str(BOCETO))
import compuesto2 as C2                                             # noqa: E402
import evaluar as EV                                                # noqa: E402
import rayos as RY                                                  # noqa: E402
import delgadas as DL                                               # noqa: E402
import digitos as D                                                 # noqa: E402
import digitos_bordes as DB                                         # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

_YY, _XX = np.mgrid[0:32, 0:32] + 0.5
ALCANCES = (8, 14, 18)
# nombre: (h mínimo, alcance de rayos, encierro mínimo)
VARIANTES = {"rizo (h0, 8px)": (0.0, 8, 0.85), "medio (h1.5, 14px)": (1.5, 14, 0.85), "lazo (h2.5, 14px)": (2.5, 14, 0.85),
             "amplia (h3.5, 18px)": (3.5, 18, 0.85), "amplia floja (h3.5, 18px, enc 0.7)": (3.5, 18, 0.7)}
CACHE = AQUI / "resultados" / "15-candidatos.pkl"


def candidatos(x) -> list[dict]:
    dt = edt(np.asarray(x) == 0); cs = C2.compuesto2(x)
    for c in cs:
        cerca = np.hypot(_XX - c["centro"][0], _YY - c["centro"][1]) <= 2.0
        k = int(np.argmax(np.where(cerca, dt, -1))); c["h"] = float(dt.flat[k]); c["centro_hueco"] = (k % 32 + 0.5, k // 32 + 0.5)
    if cs:
        ys = np.array([int(c["centro_hueco"][1]) for c in cs]); xs = np.array([int(c["centro_hueco"][0]) for c in cs])
        for a in ALCANCES:
            ch, _, _ = RY._rayos(np.asarray(x) > 0, ys, xs, float(a))
            for c, f in zip(cs, ch.mean(1)):
                c[f"r{a}"] = float(f)
    return [{k: c[k] for k in ("centro_hueco", "h", *[f"r{a}" for a in ALCANCES])} for c in cs]


def filtrar(cs, hmin, alcance, rmin):
    ok = sorted([c for c in cs if c["h"] >= hmin and c[f"r{alcance}"] >= rmin], key=lambda c: -c[f"r{alcance}"])
    out = []
    for c in ok:
        if not any(EV._dist(c["centro_hueco"], o["centro_hueco"]) <= 2.0 for o in out):
            out.append(c)
    return out


def mapas(cs, alcance) -> np.ndarray:
    todos = np.zeros((32, 32), np.float32); mejor = np.zeros((32, 32), np.float32)
    for k, c in enumerate(cs):
        g = (np.exp(-((_XX - c["centro_hueco"][0]) ** 2 + (_YY - c["centro_hueco"][1]) ** 2) / (2 * 1.5 ** 2)) * c[f"r{alcance}"]).astype(np.float32)
        todos = np.maximum(todos, g)
        if k == 0:
            mejor = g
    return np.stack([todos, mejor])


@torch.no_grad()
def caract(C, hmin, alcance, rmin):
    M = np.stack([mapas(filtrar(cs, hmin, alcance, rmin), alcance) for cs in C])
    return F.adaptive_max_pool2d(torch.from_numpy(M), 8).flatten(1).numpy(), M


def main() -> int:
    torch.set_num_threads(2); t0 = time.time()
    DL.configurar(**json.loads((BOCETO / "resultados-delgadas.json").read_text())["elegida"])
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va, ex = np.flatnonzero(part == "train"), np.flatnonzero(part == "val"), np.flatnonzero(part == "extra")
    XA = DB.caract(img, borde=lambda z: z.astype(np.float32))["bordes"]
    print(f"otros: {XA.shape} ({time.time() - t0:.0f} s)", flush=True)
    if CACHE.exists():
        C = pickle.loads(CACHE.read_bytes())
    else:
        C = [candidatos(x) for x in img]; CACHE.write_bytes(pickle.dumps(C))
    print(f"candidatos: {np.mean([len(c) for c in C]):.1f} por dígito ({time.time() - t0:.0f} s)", flush=True)
    XV, MV = {}, {}
    for n, (hmin, al, rmin) in VARIANTES.items():
        XV[n], MV[n] = caract(C, hmin, al, rmin)
    conjuntos = {"otros (delgado)": XA}
    for n in VARIANTES:
        conjuntos[f"{n} sola"] = XV[n]; conjuntos[f"otros + {n}"] = np.concatenate([XA, XV[n]], 1)
    conjuntos["otros + rizo + lazo"] = np.concatenate([XA, XV["rizo (h0, 8px)"], XV["lazo (h2.5, 14px)"]], 1)
    conjuntos["otros + rizo + lazo + amplia"] = np.concatenate([XA, XV["rizo (h0, 8px)"], XV["lazo (h2.5, 14px)"], XV["amplia (h3.5, 18px)"]], 1)
    conjuntos["otros + las cinco"] = np.concatenate([XA, *XV.values()], 1)
    out = {"variantes": {n: {"h_min": v[0], "rayos_px": v[1], "encierro_min": v[2],
                             "candidatos_por_digito_val": round(float(np.mean([len(filtrar(C[i], *v)) for i in va])), 2),
                             "por_clase_val_pc_con_alguno": {str(k): round(100 * float(np.mean([bool(filtrar(C[i], *v)) for i in va[y[va] == k]])), 1) for k in range(10)}}
                       for n, v in VARIANTES.items()}, "resultados": {}}
    preds = {}
    for n, X in conjuntos.items():
        v, c = [], []
        for sem in D.SEMILLAS:
            pr = DB.entrenar(X[tr], y[tr], sem); pv, pe = pr(X[va]), pr(X[ex])
            v.append(float((pv == y[va]).mean())); c.append(float((pe == y[ex]).mean()))
            if sem == 0:
                preds[n] = (pv, pe)
        out["resultados"][n] = {"caracteristicas": int(X.shape[1]), "val": round(float(np.mean(v)), 4), "val_rango": [round(min(v), 4), round(max(v), 4)],
                                "ciega": round(float(np.mean(c)), 4), "fallos_val_s0": int((preds[n][0] != y[va]).sum())}
        print(f"{n:45s} val {out['resultados'][n]['val']:.4f} {out['resultados'][n]['val_rango']} · ciega {out['resultados'][n]['ciega']:.4f} · {X.shape[1]} car.", flush=True)
    mejor = max((n for n in conjuntos if n.startswith("otros +")), key=lambda n: out["resultados"][n]["val"]); out["mejor_combinacion"] = mejor
    pa0, pv0 = preds["otros (delgado)"][0], preds[mejor][0]
    out["mejor_vs_otros_s0"] = {"arregla": int(((pa0 != y[va]) & (pv0 == y[va])).sum()), "rompe": int(((pa0 == y[va]) & (pv0 != y[va])).sum())}
    f = np.flatnonzero(pv0 != y[va]); out["indices_fallos_mejor_s0"] = [int(va[j]) for j in f]
    (AQUI / "resultados" / "15-variantes.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in ("variantes", "mejor_combinacion", "mejor_vs_otros_s0")}, indent=1, ensure_ascii=False), flush=True)

    # figura: tres dígitos por clase, con la variante en columnas (los candidatos de cada una)
    V = list(VARIANTES); fig, axs = plt.subplots(len(V), 10, figsize=(14, 1.55 * len(V) + 0.8))
    for r, n in enumerate(V):
        for k in range(10):
            i = va[np.flatnonzero(y[va] == k)[0]]; cs = filtrar(C[i], *VARIANTES[n]); ax = axs[r, k]
            ax.imshow(np.where(img[i] > 0, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#b9b8b4"]), vmin=0, vmax=1)
            for q, c in enumerate(cs):
                ax.plot(c["centro_hueco"][0] - 0.5, c["centro_hueco"][1] - 0.5, "+", color=EV.AZU if q == 0 else "#9fc2ec", ms=8 if q == 0 else 5, mew=1.6 if q == 0 else 1)
            ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(n, fontsize=7)
            ax.set_title(f"{k} · {len(cs)}", fontsize=7, pad=2)
    fig.suptitle("rizo-2 · 15 · las variantes del detector de curvas cerradas (filas) en un dígito de cada clase: cruz azul = el mejor candidato", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(AQUI / "resultados" / "15-variantes-mapas.png", dpi=95); plt.close(fig)

    orden = f[np.lexsort((pv0[f], y[va][f]))]; n_max = 72; cols = 12; filas = int(np.ceil(min(len(orden), n_max) / cols))
    fig, axs = plt.subplots(max(filas, 1), cols, figsize=(cols * 1.3, max(filas, 1) * 1.5 + 0.8)); axs = np.atleast_2d(axs)
    for k, ax in enumerate(axs.flat):
        ax.axis("off")
        if k < min(len(orden), n_max):
            j = orden[k]; i = va[j]
            ax.imshow(np.where(img[i] > 0, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#b9b8b4"]), vmin=0, vmax=1)
            for n, col in (("rizo (h0, 8px)", EV.NAR), ("lazo (h2.5, 14px)", EV.AZU), ("amplia (h3.5, 18px)", EV.VERDE)):
                cs = filtrar(C[i], *VARIANTES[n])
                if cs:
                    ax.plot(cs[0]["centro_hueco"][0] - 0.5, cs[0]["centro_hueco"][1] - 0.5, "+", color=col, ms=8, mew=1.6)
            ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5)
            ax.set_title(f"{y[i]}→{pv0[j]} · idx {i}" + ("" if pa0[j] != y[va][j] else " (otros acertaba)"), color=EV.NAR, fontsize=6.5, pad=2)
    fig.suptitle(f"rizo-2 · 15 · los {len(f)} dígitos de val que FALLA «{mejor}» (s0, val {out['resultados'][mejor]['val']:.4f}) · "
                 "mejor candidato de: rizo (naranja) · lazo (azul) · amplia (verde)" + (f" · se enseñan {n_max}" if len(f) > n_max else ""), fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(AQUI / "resultados" / "15-variantes-fallos.png", dpi=95)
    print(f"→ resultados/15-variantes.json · 15-variantes-mapas.png · 15-variantes-fallos.png · {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
