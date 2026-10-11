#!/usr/bin/env python3
"""rizo-2 — plantillas MÁS RODEADAS (pedido por el dueño, 2026-10-11): (a) enseñar el banco actual; (b) probar bancos que
exigen más vuelta (sin la de media vuelta, cob 0,55) y hueco mayor, con los mismos 47 doses: % encontrados, candidatos
por dígito y top-1.

    python nn/plantillas_rodeadas.py   → resultados/11-banco-actual.png · 11-bancos-rodeados.png · 11-bancos.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import evaluar as EV                                                # noqa: E402
import plantillas as PL                                             # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

BANCOS = {
    "actual (cob 0,55–1; hueco ≥ 0,5)": {"cobs": (0.55, 0.7, 0.85, 1.0), "hueco": 0.5},
    "cob ≥ 0,7; hueco ≥ 0,5":           {"cobs": (0.7, 0.85, 1.0), "hueco": 0.5},
    "cob ≥ 0,7; hueco ≥ 1":             {"cobs": (0.7, 0.85, 1.0), "hueco": 1.0},
    "cob ≥ 0,85; hueco ≥ 0,5":          {"cobs": (0.85, 1.0), "hueco": 0.5},
    "cob ≥ 0,85; hueco ≥ 1":            {"cobs": (0.85, 1.0), "hueco": 1.0},
    "cob 0,6–0,8 (como los 2); hueco ≥ 0,5": {"cobs": (0.6, 0.7, 0.8), "hueco": 0.5},
}


def banco(cobs, hueco):
    tr, hu, va, meta = [], [], [], []
    for R in PL.RADIOS:
        for g in PL.GROSORES:
            if R - g / 2 < hueco:
                continue
            for cob in cobs:
                for ab in (PL.ABERTURAS if cob < 0.99 else (0,)):
                    t, h, v = PL._plantilla(R, g, cob, ab)
                    tr.append(t / max(t.sum(), 1)); hu.append(h / max(h.sum(), 1)); va.append(v / max(v.sum(), 1))
                    meta.append({"radio": R, "grosor": g, "cierre": (cob - 0.5) / 0.5, "abertura": ab if cob < 0.99 else None})
    w = np.stack(tr) - np.stack(hu) - 0.5 * np.stack(va)
    return torch.from_numpy(w.astype(np.float32))[:, None], meta


@torch.no_grad()
def rizos(x, B, META, s):
    x = np.asarray(x) > 0
    r = F.conv2d(torch.from_numpy(x.astype(np.float32))[None, None], B, padding=PL.K // 2)[0]
    best, idx = r.max(0); best, idx = best.numpy(), idx.numpy()
    pico = (best == maximum_filter(best, size=3)) & (best >= s) & ~x
    ys, xs = np.nonzero(pico); out = []
    for i in np.argsort(-best[ys, xs]):
        cx, cy = xs[i] + 0.5, ys[i] + 0.5; m = META[idx[ys[i], xs[i]]]
        if any(np.hypot(cx - o["centro"][0], cy - o["centro"][1]) < o["radio"] / 2 for o in out):
            continue
        out.append({"centro": (float(cx), float(cy)), "radio": float(m["radio"]), "cierre": float(m["cierre"]),
                    "orientacion": None if m["abertura"] is None else float(m["abertura"]), "puntuacion": float(best[ys[i], xs[i]])})
    return out


def main() -> int:
    # (a) el banco actual: todas las coberturas × aberturas para R 4 g 4, y todos los R × g para cob 0,7 ab 120
    fig, axs = plt.subplots(2, 1, figsize=(15, 7.2)); axs = axs.ravel()
    filas = []
    for cob in PL.COBS:
        fila = [np.where(PL._plantilla(4, 4, cob, ab)[0], 1.0, np.where(PL._plantilla(4, 4, cob, ab)[1], -1.0, np.where(PL._plantilla(4, 4, cob, ab)[2], -0.5, 0)))
                for ab in (PL.ABERTURAS if cob < 0.99 else (0,) * 12)]
        filas.append(np.concatenate(fila, 1))
    axs[0].imshow(np.concatenate(filas, 0), cmap="RdBu", vmin=-1, vmax=1); axs[0].set_xticks([]); axs[0].set_yticks([PL.K * (i + 0.5) for i in range(4)]); axs[0].set_yticklabels([f"cob {c}" for c in PL.COBS])
    axs[0].set_title("el banco ACTUAL (851 plantillas): filas = cobertura 0,55 · 0,7 · 0,85 · 1,0; columnas = abertura cada 30° (R 4, g 4). Azul = trazo (+1), rojo = hueco (−1), rosa = vano (−½)", fontsize=9)
    filas = []
    for g in PL.GROSORES:
        fila = []
        for R in PL.RADIOS:
            if R - g / 2 < 0.5:
                fila.append(np.zeros((PL.K, PL.K))); continue
            t, h, v = PL._plantilla(R, g, 0.7, 120); fila.append(np.where(t, 1.0, np.where(h, -1.0, np.where(v, -0.5, 0))))
        filas.append(np.concatenate(fila, 1))
    axs[1].imshow(np.concatenate(filas, 0), cmap="RdBu", vmin=-1, vmax=1); axs[1].set_yticks([PL.K * (i + 0.5) for i in range(3)]); axs[1].set_yticklabels([f"g {g}" for g in PL.GROSORES])
    axs[1].set_xticks([PL.K * (i + 0.5) for i in range(len(PL.RADIOS))]); axs[1].set_xticklabels([f"R {r}" for r in PL.RADIOS])
    axs[1].set_title("y los radios × grosores (cob 0,7, abertura 120°): el hueco (rojo) va de 1 píxel a 5 px de radio", fontsize=9)
    fig.tight_layout(); fig.savefig(AQUI / "resultados" / "11-banco-actual.png", dpi=95); plt.close(fig)

    # (b) los bancos, sobre los 47 doses
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz")); IM = (d["imagenes"] > 0).astype(np.uint8)
    E = [e for e in EV.etiquetas() if not e["dudoso"]]; X2 = [IM[e["i"]] for e in E]
    res = {}
    for nombre, cfg in BANCOS.items():
        B, META = banco(cfg["cobs"], cfg["hueco"]); res[nombre] = {"plantillas": int(B.shape[0]), "umbrales": []}
        for s in (0.6, 0.7, 0.8, 0.9):
            R2 = [rizos(x, B, META, s) for x in X2]
            arriba = [[c for c in r if c["centro"][1] < 16] for r in R2]
            enc = np.mean([any(EV._dist(c["centro"], e["centro"]) <= EV.TOL for c in r) for r, e in zip(R2, E)])
            top1 = np.mean([bool(a) and EV._dist(a[0]["centro"], e["centro"]) <= EV.TOL for a, e in zip(arriba, E)])
            res[nombre]["umbrales"].append({"s": s, "encontrados_pc": round(100 * float(enc), 1), "top1_pc": round(100 * float(top1), 1),
                                            "candidatos_por_dos": round(float(np.mean([len(r) for r in R2])), 1)})
        print(nombre, res[nombre], flush=True)
    (AQUI / "resultados" / "11-bancos.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4)); cols = ["#52514e", "#2a78d6", "#1b7f3b", "#c2410c", "#7c3aed", "#be185d"]
    for (n, r), col in zip(res.items(), cols):
        u = r["umbrales"]
        axs[0].plot([z["candidatos_por_dos"] for z in u], [z["encontrados_pc"] for z in u], "o-", color=col, ms=4, label=f"{n} ({r['plantillas']})")
        axs[1].plot([z["s"] for z in u], [z["top1_pc"] for z in u], "o-", color=col, ms=4)
        for z in u:
            axs[0].annotate(f"{z['s']:g}", (z["candidatos_por_dos"], z["encontrados_pc"]), fontsize=6, color=col, xytext=(3, 2), textcoords="offset points")
    axs[0].set_xlabel("candidatos por dígito"); axs[0].set_ylabel("% rizos etiquetados encontrados"); axs[0].set_ylim(0, 102); axs[0].legend(frameon=False, fontsize=7); axs[0].grid(alpha=.3)
    axs[1].set_xlabel("umbral S"); axs[1].set_ylabel("% top-1 de la mitad superior acierta"); axs[1].set_ylim(0, 102); axs[1].grid(alpha=.3)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("rizo-2 · 11 · bancos de plantillas MÁS RODEADAS contra el actual, en los 47 doses etiquetados", fontsize=9.5)
    fig.tight_layout(); fig.savefig(AQUI / "resultados" / "11-bancos-rodeados.png", dpi=100)
    print("→ resultados/11-banco-actual.png · 11-bancos-rodeados.png · 11-bancos.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
