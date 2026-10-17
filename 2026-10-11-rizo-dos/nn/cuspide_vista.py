#!/usr/bin/env python3
"""rizo-2 § 110 — VISTA PREVIA: la CÚSPIDE, dos curvas que abren hacia el mismo lado y se encuentran en una punta (el dueño,
2026-10-17: «dejemos el vértice como está… lo que busco es un cambio en la derivada: 2 curvas abiertas en la misma dirección que
se encuentran en una punta. ¿Podemos detectar esa forma?»). Prototipo: no toca `cresta_curvas.py` ni el revisor; nada entrenado.

Sobre la cresta del DELGADO, con la MISMA curvatura con signo κ de `cresta_curvas` (suavizado σ 1,5, K_MIN). Al recorrer el centro
de un 3, la tangente gira hacia un lado en el arco de arriba (+), da un giro corto AL REVÉS en la punta (−) y vuelve a girar hacia
el primer lado en el arco de abajo (+). Esa firma + − + es la cúspide:

  · un run de κ de un signo con |giro| ≥ G_PUNTA y ≤ L_PUNTA px de largo (la punta),
  · con ≥ G_ARCO de giro del signo CONTRARIO en los VENT px de cada lado (las dos curvas, que por eso abren hacia el mismo lado).
  · Una S (2, 5) es + −, sin el segundo +: no la da. Un vértice de la cresta (§ 102) es UN solo giro, sin curvas a los lados.

Se mira en cada tramo empalmado y, para la punta que cae en un nudo (el 3 con lengüeta), en cada par de brazos del nudo unidos.
Dirección = hacia dónde apunta la punta (contraria al centro de su giro), en 4 sectores de 90° (N E S O, la regla del repo).

    ../.venv/bin/python nn/cuspide_vista.py   → resultados/110-cuspide.{png,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import cresta_curvas as CC                                               # noqa: E402
from expcnn import exigir_dataset                                         # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

DELGADOS = AQUI.parent / "2026-10-15-delgado-cnn" / "resultados" / "delgados-k3x3.npz"
G_ARCO = 45.0      # giro mínimo de cada curva, en sus VENT px junto a la punta
G_PUNTA = 15.0     # giro mínimo de la punta (al revés)
L_PUNTA = 12.0     # largo máximo de la punta (px de cresta); con 8 px el 3 baja del 80 al 70 %
VENT = 12.0        # px de cada lado en los que se suma el giro de las curvas
SUPR = 3.0
FLECHA = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "O": (-1, 0)}


def _rumbo(b):
    return float(np.degrees(np.arctan2(b[0], -b[1])) % 360)               # brújula: 0 N, 90 E (y crece hacia abajo)


def en_camino(p, junta=None):
    if len(p) < 5:
        return []
    ps = CC._suavizar(p, False); k, nrm = CC.curvatura(ps, False); paso = np.hypot(*np.gradient(ps, axis=0).T)
    s = np.concatenate([[0], np.cumsum(paso)]); cl = np.where(np.abs(k) >= CC.K_MIN, np.sign(k), 0).astype(int)
    out = []; i = 0; n = len(ps)
    while i < n:
        j = i
        while j < n and cl[j] == cl[i]:
            j += 1
        c = cl[i]
        if c != 0:
            giro = np.degrees(np.sum(k[i:j] * paso[i:j])); largo = s[j] - s[i]
            izq = s[i] - s[:i] <= VENT; der = s[j:n] - s[j] <= VENT
            gi = -c * np.degrees(np.sum((k[:i] * paso[:i])[izq])); gd = -c * np.degrees(np.sum((k[j:] * paso[j:])[der]))
            if abs(giro) >= G_PUNTA and largo <= L_PUNTA and gi >= G_ARCO and gd >= G_ARCO:
                m = i + int(np.argmax(np.abs(k[i:j])))
                if junta is None or np.hypot(*(ps[m] - junta)) <= SUPR:
                    r = _rumbo(-nrm[i:j].mean(0))
                    out.append({"cx": float(ps[m, 0]), "cy": float(ps[m, 1]), "giro": float(abs(giro)), "curvas": (float(gi), float(gd)),
                                "rumbo": r, "dir": "NESO"[int(((r + 45) % 360) // 90)]})
        i = j
    return out


def cuspides(M) -> list[dict]:
    tr, _ = CC.tramos(M); cand = []
    for t in CC.empalmar(tr):
        cand += en_camino(np.asarray(t["pts"], float))
    ext = {}
    for t in tr:
        p = np.asarray(t["pts"], float)
        if t["a"] > 0:
            ext.setdefault(t["a"], []).append(p)
        if t["b"] > 0:
            ext.setdefault(t["b"], []).append(p[::-1])
    for br in ext.values():                                               # la punta EN un nudo: cada par de brazos, unidos
        for x in range(len(br)):
            for y in range(x + 1, len(br)):
                cand += en_camino(np.concatenate([br[x][::-1], br[y][1:]]), junta=br[x][0])
    out = []
    for c in sorted(cand, key=lambda c: -c["giro"]):
        if all(np.hypot(c["cx"] - o["cx"], c["cy"] - o["cy"]) >= SUPR for o in out):
            out.append(c)
    return out


def pinta(a, M, i, y, cs, titulo=None):
    a.imshow(M, cmap="Greys", vmin=0, vmax=1, alpha=0.4, extent=(0, 32, 32, 0))
    tr, _ = CC.tramos(M)
    for t in CC.empalmar(tr):
        p = np.asarray(t["pts"], float)
        if len(p) < 3:
            continue
        ps = CC._suavizar(p, False); k, _ = CC.curvatura(ps, False)
        col = np.where(k >= CC.K_MIN, 1, np.where(k <= -CC.K_MIN, -1, 0))
        for q in range(len(ps) - 1):
            a.plot(ps[q:q + 2, 0], ps[q:q + 2, 1], "-", lw=2.2, color={1: "#c2410c", -1: "#2a78d6", 0: "#999"}[int(col[q])])
    for c in cs:
        dx, dy = FLECHA[c["dir"]]
        a.annotate("", (c["cx"] + 3 * dx, c["cy"] + 3 * dy), (c["cx"], c["cy"]), arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=2.2))
        a.plot(c["cx"], c["cy"], "o", ms=8, color="#7c3aed", mec="k")
        a.text(c["cx"] + 3.5 * dx + (0.5 if dx >= 0 else -0.5), c["cy"] + 3.5 * dy, f"{c['dir']}\n{c['giro']:.0f}°", fontsize=8, color="#7c3aed",
               fontweight="bold", ha="left" if dx >= 0 else "right", va="center", bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.5))
    a.set_xlim(0, 32); a.set_ylim(32, 0); a.set_xticks([]); a.set_yticks([])
    a.set_title(titulo or f"#{i} (un {y}) · {len(cs)} cúspide(s)", fontsize=9)


def main():
    d = np.load(exigir_dataset("uci-optdigits-orig-32px-r20261005") / "datos.npz")
    M = np.load(DELGADOS)["M"].astype(np.float32); y = d["etiquetas"]; va = np.flatnonzero(d["particion"] == "val")
    C = {int(i): cuspides(M[i]) for i in va}; cnt = np.bincount(y[va], minlength=10)
    alguna = np.array([sum(bool(C[i]) for i in va if y[i] == c) for c in range(10)])
    medio_o = np.array([sum(any(x["dir"] == "O" and 32 / 3 <= x["cy"] < 64 / 3 for x in C[i]) for i in va if y[i] == c) for c in range(10)])
    dirs = {c: {d_: sum(x["dir"] == d_ for i in va if y[i] == c for x in C[i]) for d_ in "NESO"} for c in (3, 8)}
    ok3 = [i for i in va if y[i] == 3 and C[int(i)] and i != 409][:5]; no3 = [i for i in va if y[i] == 3 and not C[int(i)]][:6]
    si8 = [i for i in va if y[i] == 8 and C[int(i)]][:4]; los5 = [i for i in va if y[i] == 5][:4]

    fig = plt.figure(figsize=(17, 15)); g = fig.add_gridspec(5, 6, height_ratios=[1.35, 1, 1, 1, 1])
    a = fig.add_subplot(g[0, 0:2]); pinta(a, M[409], 409, 3, C[409], "#409 (el 3 de § 108): rojo κ +, azul κ −, gris recta")
    b = fig.add_subplot(g[0, 2:4])
    tr, _ = CC.tramos(M[409]); p = np.asarray(CC.empalmar(tr)[0]["pts"], float); ps = CC._suavizar(p, False); k, _ = CC.curvatura(ps, False)
    sx = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(ps, axis=0).T))])
    b.fill_between(sx, 0, k, where=k >= 0, color="#c2410c", alpha=0.5, step="mid"); b.fill_between(sx, 0, k, where=k < 0, color="#2a78d6", alpha=0.7, step="mid")
    b.axhline(0, color="k", lw=0.8); b.axhspan(-CC.K_MIN, CC.K_MIN, color="#ddd", alpha=0.6)
    for x0, t in ((7, "curva de arriba\n+126°"), (16, "PUNTA\n−23°"), (32, "curva de abajo\n+201°")):
        b.text(x0, 0.33 if "PUNTA" not in t else -0.27, t, ha="center", fontsize=10, fontweight="bold", color="#2a78d6" if "PUNTA" in t else "#c2410c")
    b.set_xlabel("px de cresta, de la punta de arriba a la de abajo"); b.set_ylabel("curvatura κ (con signo)"); b.set_ylim(-0.35, 0.45)
    b.set_title("la firma + − +: el giro corto AL REVÉS entre dos curvas que giran igual", fontsize=10)
    c = fig.add_subplot(g[0, 4:6]); x = np.arange(10)
    c.bar(x - 0.2, 100 * alguna / cnt, 0.4, color=["#7c3aed" if q == 3 else "#bbb" for q in x], label="con alguna cúspide")
    c.bar(x + 0.2, 100 * medio_o / cnt, 0.4, color=["#c4b5fd" if q == 3 else "#e5e5e5" for q in x], label="con cúspide O en la fila del medio")
    for q in x:
        c.text(q - 0.2, 100 * alguna[q] / cnt[q] + 1.5, f"{100 * alguna[q] / cnt[q]:.0f}", ha="center", fontsize=8)
    c.set_xticks(x); c.set_ylabel("% de los dígitos de val"); c.set_ylim(0, 100); c.legend(fontsize=8, loc="upper left")
    c.set_title("val, por clase · hacia dónde apunta: " + " · ".join(f"{q}: " + ", ".join(f"{d_} {n}" for d_, n in dirs[q].items() if n) for q in (3, 8)), fontsize=9)
    filas = []
    for fila, (ids, nombre) in enumerate(((ok3, "3 CON cúspide"), (no3, "3 SIN cúspide"), (si8, "8: la cintura"), (los5, "5: ninguno"))):
        for col, i in enumerate(ids):
            ax = fig.add_subplot(g[fila + 1, col]); pinta(ax, M[i], int(i), y[i], C[int(i)])
            if col == 0:
                filas.append((ax, nombre))
    fig.suptitle(f"La CÚSPIDE: dos curvas que abren al mismo lado y se encuentran en una punta (prototipo, cresta del delgado)\n"
                 f"punta = giro al revés ≥ {G_PUNTA:.0f}° en ≤ {L_PUNTA:.0f} px · cada curva ≥ {G_ARCO:.0f}° en sus {VENT:.0f} px · flecha = hacia dónde apunta la punta",
                 fontsize=13)
    fig.tight_layout(rect=(0.015, 0, 1, 0.96))
    for ax, nombre in filas:                                              # la etiqueta, a la altura REAL de su fila
        q = ax.get_position(); fig.text(0.006, q.y0 + q.height / 2, nombre, rotation=90, fontsize=11, fontweight="bold", va="center")
    out = AQUI / "resultados" / "110-cuspide.png"; fig.savefig(out, dpi=95)
    res = {"parametros": {"G_ARCO": G_ARCO, "G_PUNTA": G_PUNTA, "L_PUNTA": L_PUNTA, "VENT": VENT, "K_MIN": CC.K_MIN},
           "pct_con_cuspide": np.round(100 * alguna / cnt, 1).tolist(), "pct_O_en_el_medio": np.round(100 * medio_o / cnt, 1).tolist(),
           "direcciones": dirs, "409": C[409], "tres_sin_cuspide": [int(i) for i in va if y[i] == 3 and not C[int(i)]]}
    out.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=1)); print(out); print(json.dumps({k: v for k, v in res.items() if k != "tres_sin_cuspide"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
