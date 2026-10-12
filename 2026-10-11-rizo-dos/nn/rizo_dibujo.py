#!/usr/bin/env python3
"""rizo-2 — CÓMO se obtienen los rizos, etapa a etapa sobre un dígito (pedido por el dueño el 2026-10-12). La cadena es la
de `digitos_variantes.candidatos` → `filtrar` → `mapas` → `compositores.compacta`:

  1 dígito · 2 plantillas rodeadas (banco de anillos con cob ≥ 0,85: mejor correlación por píxel y sus picos, P')
  3 rayos (36 rayos de 8 px desde cada píxel de fondo: fracción que choca con tinta; regiones con ≥ 0,5 y poca dispersión, R)
  4 unión P' ∪ R ordenada por encierro y sin duplicados a 2 px (los candidatos CRUDOS)
  5 el centro del HUECO (máximo de la distancia al trazo a ≤ 2 px del candidato) y su altura h
  6 encierro en ese centro con rayos de 8 px; filtro h ≥ 0 y encierro ≥ 0,85; sin duplicados → los RIZOS
  7 mapa: campana σ 1,5 por rizo con altura = encierro (todos · sólo el mejor)
  8 lo que entra al compositor: 2×2 máximos del mapa del mejor + la cuenta = 5 números

    python nn/rizo_dibujo.py --idx 784   → resultados/25-rizo-784.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as edt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import compuesto as CP                                              # noqa: E402
import compuesto2 as C2                                             # noqa: E402
import digitos_variantes as DV                                      # noqa: E402
import evaluar as EV                                                # noqa: E402
import plantillas as PL                                             # noqa: E402
import plantillas_rodeadas as PR                                    # noqa: E402
import rayos as RY                                                  # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

FONDO = matplotlib.colors.ListedColormap(["#c9c8c4"])


def fondo(ax, x, titulo):
    ax.imshow(np.where(x > 0, 1.0, np.nan), cmap=FONDO, vmin=0, vmax=1, zorder=2); ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(titulo, fontsize=8.5)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--idx", type=int, default=784); a = ap.parse_args()
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz")); x = (d["imagenes"][a.idx] > 0); real = int(d["etiquetas"][a.idx])
    best, idx = C2.mapa_rodeado(x); cP = PR.rizos(x, C2.B_ROD, C2.META_ROD, C2.S_ROD)
    ys, xs = np.nonzero(~x); ch, _, _ = RY._rayos(x, ys, xs, C2.RMAX); r = np.full(x.shape, np.nan); r[ys, xs] = ch.mean(1); cR = RY.rizos(x, CP.F_R, 8.0, 99.0)
    crudos = C2.compuesto2(x); cand = DV.candidatos(x); dt = edt(~x)
    hmin, al, rmin = DV.VARIANTES["rizo (h0, 8px)"]; rizos = DV.filtrar(cand, hmin, al, rmin); M = DV.mapas(rizos, al)
    comp = F.adaptive_max_pool2d(torch.from_numpy(M[1:2]), 2)[0].numpy(); feat = np.concatenate([comp.ravel(), [len(rizos)]])
    fig, axs = plt.subplots(2, 4, figsize=(16, 8.6)); axs = axs.ravel()
    fondo(axs[0], x, f"1 · idx {a.idx} (un {real})")
    # 2 plantillas
    ax = axs[1]; im = ax.imshow(np.where(x, np.nan, best), cmap="magma", vmin=0, zorder=1); fondo(ax, x, f"2 · plantillas RODEADAS: mejor correlación por píxel\n{len(C2.META_ROD)} anillos (cob ≥ 0,85); picos ≥ {C2.S_ROD} → P' ({len(cP)})")
    for c in cP:
        ax.add_patch(plt.Circle((c["centro"][0] - 0.5, c["centro"][1] - 0.5), c["radio"], fill=False, color="#2a78d6", lw=1.2, zorder=3)); ax.plot(c["centro"][0] - 0.5, c["centro"][1] - 0.5, "+", color="#2a78d6", ms=7, mew=1.5, zorder=4)
    t, h, v = PL._plantilla(4, 3, 0.85, 90); ins = ax.inset_axes([0.70, 0.02, 0.28, 0.28]); ins.imshow(t * 1.0 - h * 1.0 - 0.5 * v, cmap="RdBu_r", vmin=-1, vmax=1); ins.set_xticks([]); ins.set_yticks([]); ins.set_title("una plantilla (R 4, g 3, cob 0,85)", fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    # 3 rayos
    ax = axs[2]; im = ax.imshow(r, cmap="magma", vmin=0, vmax=1, zorder=1); fondo(ax, x, f"3 · RAYOS: 36 rayos de {C2.RMAX:.0f} px desde cada píxel de fondo,\nfracción que choca con tinta; regiones ≥ {CP.F_R} → R ({len(cR)})")
    for c in cR:
        ax.plot(c["centro"][0] - 0.5, c["centro"][1] - 0.5, "x", color="#1b7f3b", ms=7, mew=1.6, zorder=4)
    c0 = cR[0] if cR else None
    if c0:
        cx, cy = c0["centro"]; ch1, di1, _ = RY._rayos(x, np.array([int(cy)]), np.array([int(cx)]), C2.RMAX)
        for k, (ca, sa) in enumerate(zip(RY.CA, RY.SA)):
            L = di1[0, k] if ch1[0, k] else C2.RMAX; ax.plot([cx - 0.5, cx - 0.5 + ca * L], [cy - 0.5, cy - 0.5 + sa * L], color="#1b7f3b" if ch1[0, k] else "#c2410c", lw=0.7, alpha=0.9, zorder=3)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    # 4 unión
    ax = axs[3]; fondo(ax, x, f"4 · UNIÓN P' ∪ R, ordenada por encierro, sin duplicados a 2 px\n→ {len(crudos)} candidatos CRUDOS (azul P', verde R)")
    for c in crudos:
        ax.plot(c["centro"][0] - 0.5, c["centro"][1] - 0.5, "+" if c["origen"] == "P'" else "x", color="#2a78d6" if c["origen"] == "P'" else "#1b7f3b", ms=6, mew=1.3, alpha=0.4 + 0.6 * c["r"], zorder=4)
    # 5 hueco
    ax = axs[4]; im = ax.imshow(np.where(x, np.nan, dt), cmap="Blues", vmin=0, zorder=1); fondo(ax, x, "5 · el HUECO: distancia al trazo; cada candidato salta al\nmáximo a ≤ 2 px (centro_hueco) y toma su altura h")
    for c0, c in zip(crudos, cand):
        ax.annotate("", xy=(c["centro_hueco"][0] - 0.5, c["centro_hueco"][1] - 0.5), xytext=(c0["centro"][0] - 0.5, c0["centro"][1] - 0.5), arrowprops=dict(arrowstyle="->", color="#52514e", lw=0.7), zorder=4)
        ax.plot(c["centro_hueco"][0] - 0.5, c["centro_hueco"][1] - 0.5, ".", color="#c2410c", ms=5, zorder=5)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    # 6 encierro + filtro
    ax = axs[5]; fondo(ax, x, f"6 · ENCIERRO en el centro del hueco (rayos de {al} px)\nfiltro h ≥ {hmin:g} · encierro ≥ {rmin} · sin duplicados → {len(rizos)} RIZOS")
    for c in cand:
        pasa = c["h"] >= hmin and c[f"r{al}"] >= rmin
        ax.plot(c["centro_hueco"][0] - 0.5, c["centro_hueco"][1] - 0.5, "o", color="#1b7f3b" if pasa else "#c2410c", ms=3 + 6 * c[f"r{al}"], alpha=0.5, mew=0, zorder=4)
    for k, c in enumerate(rizos):
        ax.plot(c["centro_hueco"][0] - 0.5, c["centro_hueco"][1] - 0.5, "+", color=EV.AZU if k == 0 else "#9fc2ec", ms=12 if k == 0 else 8, mew=2, zorder=5)
        ax.annotate(f"h {c['h']:.1f} · enc {c[f'r{al}']:.2f}", (c["centro_hueco"][0] - 0.5, c["centro_hueco"][1] - 0.5), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=EV.AZU, zorder=6)
    ax.text(0.02, 0.02, "tamaño = encierro · verde pasa · rojo no · cruz grande = el mejor", transform=ax.transAxes, fontsize=6.5, color="#52514e")
    # 7 mapas
    ax = axs[6]; im = ax.imshow(M[0], cmap="Oranges", vmin=0, vmax=1, zorder=1); fondo(ax, x, "7 · MAPA: campana σ 1,5 en cada rizo, altura = encierro\n(todos; el compositor usa el del MEJOR, abajo a la derecha)")
    ins = ax.inset_axes([0.70, 0.02, 0.28, 0.28]); ins.imshow(M[1], cmap="Oranges", vmin=0, vmax=1); ins.set_xticks([]); ins.set_yticks([]); ins.set_title("sólo el mejor", fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    # 8 compactado
    ax = axs[7]; ax.imshow(comp, cmap="Oranges", vmin=0, vmax=1); ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["izq", "der"]); ax.set_yticklabels(["arriba", "abajo"], rotation=90, va="center")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{comp[i, j]:.2f}", ha="center", va="center", fontsize=12, color="k" if comp[i, j] < 0.6 else "w")
    ax.set_title(f"8 · lo que ENTRA al compositor: máximo 2×2 del mapa del mejor\n+ la cuenta ({len(rizos)}) = 5 números: {np.round(feat, 2).tolist()}", fontsize=8.5)
    fig.suptitle(f"rizo-2 · 25 · cómo se obtiene la feature del RIZO, etapa a etapa (idx {a.idx}): plantillas ∪ rayos → hueco → encierro → filtro → mapa → 2×2 + cuenta", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(AQUI / "resultados" / f"25-rizo-{a.idx}.png", dpi=100); print(f"→ resultados/25-rizo-{a.idx}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
