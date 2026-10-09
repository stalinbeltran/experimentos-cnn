#!/usr/bin/env python3
"""¿Por qué el compositor COMBINADO lee 9 en un 5 nítido? (pedido del dueño, 2026-10-09)

El dígito: el fallo nº 37 (fila 4, columna 1) de `imagenes/6-fallos-combinado.png`, o sea el índice 36 de la lista de
fallos de la semilla 0. Se reproduce el MISMO compositor (misma semilla, mismo entrenamiento de `digitos.py`) y se abre:

    7-cinco-1-entrada.png    lo que ve: la imagen → 4 Gabor → integrado 15 px → max en celdas 8×8, en 2 escalas;
                             y los 3 mapas del detector de curvas
    7-cinco-2-votos.png      quién vota 9 y quién 5: logit = W·z + b, y (W9 − W5)·z celda a celda, sobre el dígito
    7-cinco-3-borrar.png     qué tinta empuja al 9 (borrando parches 4×4) y qué hay que borrar para que diga 5
    7-cinco-4-ejemplos.png   con qué aprendió: los 18 cincos y 18 nueves de entrenamiento, y los vecinos de este
    7-cinco-5-rasgos.png     los dos rasgos (barra media alta, lado derecho recto), contra los 5 y 9 de entrenamiento

    python por_que_5.py [--k 36]   → imagenes/7-cinco-*.png · resultados-cinco.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parents[2]))
import curvas as C                                                  # noqa: E402
import digitos as D                                                 # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

IMG = AQUI / "imagenes"
SUP, T1, T2, NAR, AZU = "#fcfcfb", "#0b0b0b", "#52514e", "#eb6834", "#2a78d6"
VOTO = LinearSegmentedColormap.from_list("voto", [AZU, "#ffffff", NAR])      # azul = vota 5 · naranja = vota 9
ORI = ("— 0°", "\\ 45°", "| 90°", "/ 135°")
CANALES = [f"rectas {o} · esc. 1" for o in ORI] + [f"rectas {o} · esc. ½" for o in ORI] + \
          ["curvas: recto", "curvas: curvo |κ|", "curvas: golpe"]
A, B = 5, 9          # real, predicho


def compositor(xtr, ytr, sem):
    """El de `digitos.py`, línea a línea, pero devolviendo los pesos (misma semilla → mismo resultado)."""
    torch.manual_seed(sem)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    W = torch.nn.Linear(xtr.shape[1], 10); opt = torch.optim.Adam(W.parameters(), lr=D.LR, weight_decay=D.L2)
    xt, yt = torch.from_numpy((xtr - mu) / sd).float(), torch.from_numpy(ytr).long()
    for _ in range(D.EPOCAS):
        opt.zero_grad(); F.cross_entropy(W(xt), yt).backward(); opt.step()
    return W.weight.detach().numpy().astype(np.float64), W.bias.detach().numpy().astype(np.float64), mu, sd


def caract(x: np.ndarray) -> np.ndarray:
    return np.concatenate([D.caract_rectas(x), D.caract_curvas(x)], 1)


def estilo():
    plt.rcParams.update({"font.size": 8, "figure.facecolor": SUP, "axes.facecolor": SUP, "axes.titlecolor": T1,
                         "axes.edgecolor": T2})


def limpio(ax):
    ax.set_xticks([]); ax.set_yticks([])


def sobre_digito(ax, x, rejilla, norm, cmap=VOTO, alpha=0.8):
    """Una rejilla 8×8 (celdas de 4×4 px) encima del dígito."""
    ax.imshow(x, cmap="gray_r", vmin=0, vmax=3)
    ax.imshow(rejilla, cmap=cmap, norm=norm, extent=(-0.5, 31.5, 31.5, -0.5), alpha=alpha, interpolation="nearest")
    ax.contour(x.astype(float), levels=[0.5], colors=[T1], linewidths=0.6)
    limpio(ax)


# ─── 1 · lo que ve ────────────────────────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def figura_entrada(x, fila_x):
    estilo()
    fig = plt.figure(figsize=(15, 11.5))
    gs = fig.add_gridspec(5, 9, hspace=0.55, wspace=0.25)
    xt = torch.from_numpy(x.astype(np.float32))[None, None]
    pooled = []
    for e in range(2):
        xe = xt if e == 0 else F.avg_pool2d(xt, 2)
        g = F.relu(F.conv2d(xe, D.GABOR4, padding=4))
        m = F.conv2d(g, D.INTEG, padding=D.L_INT // 2, groups=4)
        p = F.adaptive_max_pool2d(m, 8)[0].numpy(); pooled.append(p)
        ax = fig.add_subplot(gs[e, 0]); ax.imshow(xe[0, 0], cmap="gray_r", vmin=0, vmax=1); limpio(ax)
        ax.set_title("imagen 32×32" if e == 0 else "a escala ½ (16×16)")
        ax.set_ylabel("escala 1" if e == 0 else "escala ½", fontsize=10, color=T1)
        gm, mm = g[0].numpy().max(), m[0].numpy().max()
        for j in range(4):
            a1 = fig.add_subplot(gs[e, 1 + j]); a1.imshow(g[0, j], cmap="Oranges", vmin=0, vmax=gm); limpio(a1)
            a1.set_title(f"Gabor {ORI[j]}\nmax {g[0, j].max():.2f}")
            a2 = fig.add_subplot(gs[e, 5 + j]); a2.imshow(m[0, j], cmap="Oranges", vmin=0, vmax=mm); limpio(a2)
            a2.set_title(f"integr. 15 px {ORI[j]}\nmax {m[0, j].max():.2f}")
    # lo que recibe el compositor: las 8 rejillas 8×8 de rectas
    top = max(p.max() for p in pooled)
    for e in range(2):
        for j in range(4):
            ax = fig.add_subplot(gs[2, e * 4 + j + (1 if e else 0)])
            sobre_digito(ax, x, pooled[e][j], plt.Normalize(0, top), cmap="Oranges", alpha=0.85)
            ax.set_title(f"→ 8×8 {ORI[j]} esc. {'1' if e == 0 else '½'}")
    ax = fig.add_subplot(gs[2, 4]); ax.axis("off")
    ax.text(0.5, 0.5, "max en\nceldas\n4×4 px\n→ 64 números\npor mapa", ha="center", va="center", color=T2, fontsize=8)
    # curvas
    mc = D.mapas_curvas(x); pc = F.adaptive_max_pool2d(torch.from_numpy(mc)[None], 8)[0].numpy()
    c = C.campo(x)
    ax = fig.add_subplot(gs[3, 0]); ax.imshow(x, cmap="gray_r", vmin=0, vmax=1, alpha=0.2); C._ticks(ax, c, c["Emax"] > C.TAU)
    ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); limpio(ax); ax.set_title("orientación local θ"); ax.set_ylabel("curvas", fontsize=10, color=T1)
    nombres = ("recto (|κ| < 1)", "curvo |κ| (°/px)", "golpe (coh. baja)")
    for j in range(3):
        a1 = fig.add_subplot(gs[3, 1 + 2 * j]); a1.imshow(mc[j], cmap="Oranges", vmin=0, vmax=max(mc[j].max(), 1e-6)); limpio(a1)
        a1.contour(x.astype(float), levels=[0.5], colors=[T2], linewidths=0.4); a1.set_title(nombres[j])
        a2 = fig.add_subplot(gs[3, 2 + 2 * j]); sobre_digito(a2, x, pc[j], plt.Normalize(0, max(pc[j].max(), 1e-6)), cmap="Oranges")
        a2.set_title(f"→ 8×8 {nombres[j].split(' ')[0]}")
    ax = fig.add_subplot(gs[4, :]); ax.axis("off")
    ax.text(0, 0.9, "El compositor recibe 11 rejillas de 8×8 = 704 números (8 de rectas, 3 de curvas), cada uno el MÁXIMO de su mapa en una "
            "celda de 4×4 px.\nNo ve la imagen ni el orden de los trazos: ve DÓNDE hay algo de cada orientación, ya estirado 15 px a lo largo "
            "de esa orientación por la integración.\n" + fila_x, va="top", fontsize=9, color=T1)
    fig.suptitle("1 · Lo que VE el compositor del 5 que lee como 9", color=T1, fontsize=12)
    fig.savefig(IMG / "7-cinco-1-entrada.png", dpi=100, bbox_inches="tight"); plt.close(fig)
    print("→", IMG / "7-cinco-1-entrada.png")
    return np.concatenate([np.concatenate([p.reshape(-1) for p in pooled]), pc.reshape(-1)])


# ─── 2 · los votos ────────────────────────────────────────────────────────────────────────────────────────────────
def figura_votos(x, z, W, b, logits):
    estilo()
    dW = W[B] - W[A]; voto = dW * z
    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(4, 11, hspace=0.5, wspace=0.15, height_ratios=[1, 1, 1, 1.25])
    mW, mz, mv = np.abs(dW).max(), np.abs(z).max(), np.abs(voto).max()
    for k in range(11):
        sl = slice(64 * k, 64 * (k + 1))
        a = fig.add_subplot(gs[0, k]); a.imshow(dW[sl].reshape(8, 8), cmap=VOTO, norm=TwoSlopeNorm(0, -mW, mW)); limpio(a)
        a.set_title(CANALES[k].replace(" · ", "\n"), fontsize=7)
        a = fig.add_subplot(gs[1, k]); sobre_digito(a, x, z[sl].reshape(8, 8), TwoSlopeNorm(0, -mz, mz), cmap="PRGn", alpha=0.85)
        a = fig.add_subplot(gs[2, k]); sobre_digito(a, x, voto[sl].reshape(8, 8), TwoSlopeNorm(0, -mv, mv))
        a.set_xlabel(f"{voto[sl].sum():+.2f}", color=NAR if voto[sl].sum() > 0 else AZU, fontsize=9)
    fig.text(0.09, 0.80, "W₉ − W₅\nlo que el compositor\naprendió que separa\n9 de 5 (naranja → 9)", ha="right", va="center", fontsize=8, color=T1)
    fig.text(0.09, 0.61, "z del dígito\n(estandarizado contra\nlos 180 de entreno)\nverde = más que lo normal\nmorado = menos", ha="right", va="center", fontsize=8, color=T1)
    fig.text(0.09, 0.42, "VOTO = (W₉ − W₅)·z\nsobre el dígito\nnaranja → 9 · azul → 5\n(suma del mapa abajo)", ha="right", va="center", fontsize=8, color=T1)
    # barras: logits y votos por canal
    a = fig.add_subplot(gs[3, 0:4])
    col = [NAR if c == B else (AZU if c == A else "#c9c8c4") for c in range(10)]
    a.bar(range(10), logits, color=col); a.set_xticks(range(10)); a.set_title("logits de las 10 clases", loc="left")
    for c in range(10):
        a.text(c, logits[c], f"{logits[c]:.1f}", ha="center", va="bottom" if logits[c] >= 0 else "top", fontsize=7)
    a.spines[["top", "right"]].set_visible(False)
    a = fig.add_subplot(gs[3, 5:11])
    sumas = [voto[64 * k:64 * (k + 1)].sum() for k in range(11)] + [b[B] - b[A]]
    nombres = [n.replace("rectas ", "").replace(" · ", " ") for n in CANALES] + ["sesgo b₉ − b₅"]
    a.barh(range(12), sumas, color=[NAR if s > 0 else AZU for s in sumas]); a.set_yticks(range(12)); a.set_yticklabels(nombres, fontsize=7)
    a.invert_yaxis(); a.axvline(0, color=T2, lw=0.8); a.spines[["top", "right"]].set_visible(False)
    pres, aus = voto[z > 0].sum(), voto[z <= 0].sum()
    a.set_title(f"logit₉ − logit₅ = {logits[B] - logits[A]:+.2f}, por canal.   De ello: donde HAY más trazo de lo normal (z > 0) "
                f"{pres:+.2f} · donde FALTA (z ≤ 0) {aus:+.2f}", loc="left", fontsize=8)
    fig.suptitle(f"2 · Quién vota 9 y quién vota 5.  El compositor es lineal: logit_c = Σ W_c·z + b_c, así que logit₉ − logit₅ es la suma "
                 "de (W₉ − W₅)·z sobre los 704 números", color=T1, fontsize=11)
    fig.savefig(IMG / "7-cinco-2-votos.png", dpi=100, bbox_inches="tight"); plt.close(fig)
    print("→", IMG / "7-cinco-2-votos.png")
    return {"margen_9_menos_5": round(float(logits[B] - logits[A]), 3),
            "por_canal": {n: round(float(s), 3) for n, s in zip(nombres, sumas)},
            "presencia_z_pos": round(float(pres), 3), "ausencia_z_neg": round(float(aus), 3)}


# ─── 3 · borrar ───────────────────────────────────────────────────────────────────────────────────────────────────
def figura_borrar(x, f_logits):
    estilo()
    P, S = 4, 2
    base = f_logits(x[None])[0]; m0 = base[B] - base[A]
    pos = [(i, j) for i in range(0, 32 - P + 1, S) for j in range(0, 32 - P + 1, S) if x[i:i + P, j:j + P].any()]
    lote = np.repeat(x[None], len(pos), 0)
    for n, (i, j) in enumerate(pos):
        lote[n, i:i + P, j:j + P] = 0
    L = f_logits(lote); caida = m0 - (L[:, B] - L[:, A])
    mapa, cuenta = np.zeros((32, 32)), np.zeros((32, 32))
    for (i, j), c in zip(pos, caida):
        mapa[i:i + P, j:j + P] += c; cuenta[i:i + P, j:j + P] += 1
    mapa = np.where(cuenta > 0, mapa / np.maximum(cuenta, 1), np.nan) * (x > 0)
    # borrado codicioso de parches hasta que deje de decir 9 (máx. 6)
    pasos, xc = [(x.copy(), base, None)], x.copy()
    for _ in range(6):
        cand = [(i, j) for i in range(0, 32 - P + 1, S) for j in range(0, 32 - P + 1, S) if xc[i:i + P, j:j + P].any()]
        lote = np.repeat(xc[None], len(cand), 0)
        for n, (i, j) in enumerate(cand):
            lote[n, i:i + P, j:j + P] = 0
        L = f_logits(lote); k = int(np.argmin(L[:, B] - L[:, A]))
        xc = lote[k]; pasos.append((xc.copy(), L[k], cand[k]))
        if L[k].argmax() != B:
            break
    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(2, 1 + len(pasos), hspace=0.35, wspace=0.25, height_ratios=[1, 0.55])
    a = fig.add_subplot(gs[:, 0]); mx = np.nanmax(np.abs(mapa))
    a.imshow(x, cmap="gray_r", vmin=0, vmax=3)
    a.imshow(mapa, cmap=VOTO, norm=TwoSlopeNorm(0, -mx, mx)); limpio(a)
    a.contour(x.astype(float), levels=[0.5], colors=[T1], linewidths=0.6)
    a.set_title("si BORRO esta tinta (parche 4×4),\n¿cuánto baja logit₉ − logit₅?\nnaranja: esa tinta empujaba al 9\nazul: empujaba al 5", fontsize=8)
    for n, (xi, Li, parche) in enumerate(pasos):
        a = fig.add_subplot(gs[0, 1 + n]); a.imshow(xi, cmap="gray_r", vmin=0, vmax=1); limpio(a)
        if parche:
            a.add_patch(plt.Rectangle((parche[1] - 0.5, parche[0] - 0.5), P, P, fill=False, ec=NAR, lw=1.6))
        pr = int(Li.argmax())
        a.set_title(("original" if n == 0 else f"borro {n}") + f"\n→ {pr}", color=NAR if pr == B else (AZU if pr == A else T1), fontsize=9)
        b = fig.add_subplot(gs[1, 1 + n]); b.bar(range(10), Li, color=["#c9c8c4"] * 5 + [AZU] + ["#c9c8c4"] * 3 + [NAR])
        b.set_xticks(range(10)); b.tick_params(labelsize=6); b.spines[["top", "right"]].set_visible(False)
        b.set_title(f"logit 9 − logit 5 = {Li[B] - Li[A]:+.1f}", fontsize=7)
    fig.suptitle("3 · Qué tinta lo hace 9: oclusión parche a parche, y el borrado mínimo (codicioso) hasta que deja de decir 9",
                 color=T1, fontsize=11)
    fig.savefig(IMG / "7-cinco-3-borrar.png", dpi=100, bbox_inches="tight"); plt.close(fig)
    print("→", IMG / "7-cinco-3-borrar.png")
    return {"pasos": [{"parche_yx": p, "pred": int(L.argmax()), "margen": round(float(L[B] - L[A]), 2)} for _, L, p in pasos]}


# ─── 4 · con qué aprendió ─────────────────────────────────────────────────────────────────────────────────────────
def figura_ejemplos(x, z, Z_tr, img, y, tr):
    estilo()
    dist = np.linalg.norm(Z_tr - z[None], axis=1); orden = np.argsort(dist)[:12]
    fig = plt.figure(figsize=(15, 6.4))
    gs = fig.add_gridspec(3, 19, hspace=0.75, wspace=0.15)
    gv = gs[2, :].subgridspec(1, 12, wspace=0.2)
    a = fig.add_subplot(gs[0:2, 0]); a.imshow(x, cmap="gray_r", vmin=0, vmax=1); limpio(a); a.set_title("ESTE 5", color=NAR, fontsize=10)
    for fila, clase in ((0, A), (1, B)):
        ids = tr[y[tr] == clase]
        for k, i in enumerate(ids):
            a = fig.add_subplot(gs[fila, 1 + k])
            a.imshow(img[i], cmap="gray_r", vmin=0, vmax=1); limpio(a)
            if k == 0:
                a.set_title(f"los 18 «{clase}» de entrenamiento →", fontsize=8, color=AZU if clase == A else NAR, loc="left")
    fig.text(0.5, 0.36, "Los 12 dígitos de ENTRENAMIENTO más parecidos a este en las 704 características (distancia en z)",
             ha="center", color=T1, fontsize=10)
    for k, i in enumerate(orden):
        a = fig.add_subplot(gv[0, k])
        a.imshow(img[tr[i]], cmap="gray_r", vmin=0, vmax=1); limpio(a)
        c = int(y[tr[i]]); a.set_title(f"{c}  (d {dist[i]:.1f})", color=AZU if c == A else (NAR if c == B else T1), fontsize=9)
    fig.suptitle("4 · Con qué aprendió: SÓLO 18 ejemplos por dígito (180 en total). Azul: 5 · naranja: 9", color=T1, fontsize=11)
    fig.savefig(IMG / "7-cinco-4-ejemplos.png", dpi=100, bbox_inches="tight"); plt.close(fig)
    print("→", IMG / "7-cinco-4-ejemplos.png")
    return {"vecinos": [{"etiqueta": int(y[tr[i]]), "dist": round(float(dist[i]), 2)} for i in orden]}


# ─── 5 · los dos rasgos, contra los 18 «5» y los 18 «9» ───────────────────────────────────────────────────────────
def figura_perfil(x, X, img, y, tr, i):
    estilo()
    def rej(XX, k):
        return XX[:, 64 * k:64 * (k + 1)].reshape(-1, 8, 8)
    t5, t9 = tr[y[tr] == A], tr[y[tr] == B]
    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 5, wspace=0.55, width_ratios=[1, 1, 1, 1.5, 1.5])
    for j, (nombre, ids) in enumerate((("media de los 18 «5»", t5), ("media de los 18 «9»", t9))):
        a = fig.add_subplot(gs[0, j]); a.imshow(img[ids].mean(0), cmap="gray_r", vmin=0, vmax=1); limpio(a)
        a.contour(x.astype(float), levels=[0.5], colors=[NAR], linewidths=1.0); a.set_title(nombre + "\n(naranja: el contorno de ESTE 5)")
    a = fig.add_subplot(gs[0, 2]); a.imshow(x, cmap="gray_r", vmin=0, vmax=1); limpio(a)
    a.add_patch(plt.Rectangle((-0.3, 7.7), 31.6, 8, fill=False, ec=NAR, lw=1.8))
    a.add_patch(plt.Rectangle((19.7, 11.7), 11.6, 19.6, fill=False, ec=AZU, lw=1.8, ls="--"))
    a.set_xlim(-0.5, 31.5); a.set_ylim(31.5, -0.5)
    a.set_xlabel("naranja: barra media ALTA (px 8–15)\nazul: lado derecho RECTO y largo", fontsize=8, color=T1)
    a.set_title(f"ESTE 5 (dígito {i})")
    filas = np.arange(8)
    a = fig.add_subplot(gs[0, 3])
    for ids, col, n in ((t5, AZU, "media «5» entreno"), (t9, NAR, "media «9» entreno")):
        v = rej(X[ids], 0).max(2); a.plot(v.mean(0), filas, "o-", color=col, label=n)
        a.fill_betweenx(filas, np.percentile(v, 25, 0), np.percentile(v, 75, 0), color=col, alpha=0.12)
    a.plot(rej(X[[i]], 0)[0].max(1), filas, "s-", color=T1, lw=2, label="este 5")
    a.set_yticks(filas); a.set_yticklabels([f"px {4 * r}–{4 * r + 3}" for r in filas]); a.invert_yaxis()
    a.set_xlabel("max de la fila de celdas"); a.set_title("trazo HORIZONTAL (— 0°, esc. 1), por altura", loc="left")
    a.legend(frameon=False, fontsize=7); a.spines[["top", "right"]].set_visible(False)
    a = fig.add_subplot(gs[0, 4])
    for ids, col, n in ((t5, AZU, "media «5» entreno"), (t9, NAR, "media «9» entreno")):
        v = rej(X[ids], 6)[:, :, 5:].max(2); a.plot(v.mean(0), filas, "o-", color=col, label=n)
        a.fill_betweenx(filas, np.percentile(v, 25, 0), np.percentile(v, 75, 0), color=col, alpha=0.12)
    a.plot(rej(X[[i]], 6)[0][:, 5:].max(1), filas, "s-", color=T1, lw=2, label="este 5")
    a.set_yticks(filas); a.set_yticklabels([f"px {4 * r}–{4 * r + 3}" for r in filas]); a.invert_yaxis()
    a.set_xlabel("max en las 3 columnas de celdas de la DERECHA"); a.set_title("trazo VERTICAL a la derecha (| 90°, esc. ½), por altura", loc="left")
    a.legend(frameon=False, fontsize=7); a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("5 · Los dos rasgos que lo hacen 9, contra lo que vio en entrenamiento (banda: cuartiles 25–75 de los 18)", color=T1, fontsize=11)
    fig.savefig(IMG / "7-cinco-5-rasgos.png", dpi=100, bbox_inches="tight"); plt.close(fig)
    print("→", IMG / "7-cinco-5-rasgos.png")
    return {"horizontal_por_fila": {"este": [round(float(v), 2) for v in rej(X[[i]], 0)[0].max(1)],
                                    "media_5": [round(float(v), 2) for v in rej(X[t5], 0).max(2).mean(0)],
                                    "media_9": [round(float(v), 2) for v in rej(X[t9], 0).max(2).mean(0)]},
            "vertical_derecha_por_fila": {"este": [round(float(v), 2) for v in rej(X[[i]], 6)[0][:, 5:].max(1)],
                                          "media_5": [round(float(v), 2) for v in rej(X[t5], 6)[:, :, 5:].max(2).mean(0)],
                                          "media_9": [round(float(v), 2) for v in rej(X[t9], 6)[:, :, 5:].max(2).mean(0)]}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=int, default=36, help="posición en la lista de fallos del combinado, semilla 0")
    a = p.parse_args()
    torch.set_num_threads(2); IMG.mkdir(exist_ok=True)
    d = dict(np.load(exigir_dataset(D.DIGITOS) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va = np.flatnonzero(part == "train"), np.flatnonzero(part == "val")
    X = caract(img)
    W, b, mu, sd = compositor(X[tr], y[tr], 0)
    pv = (((X[va] - mu) / sd) @ W.T + b).argmax(1)
    fallos = va[pv != y[va]]
    assert len(fallos) == 68, f"no reproduce el compositor de digitos.py: {len(fallos)} fallos, se esperaban 68"
    i = int(fallos[a.k]); x = img[i]
    assert (y[i], pv[np.flatnonzero(va == i)[0]]) == (A, B), "ese fallo no es un 5 leído como 9"
    z = (X[i] - mu) / sd; logits = z @ W.T + b

    def f_logits(xs):
        return ((caract(xs) - mu) / sd) @ W.T + b

    out = {"indice": i, "particion": "val", "real": A, "predicho": B, "logits": [round(float(v), 2) for v in logits]}
    fila = f"Dígito {i} de val (fallo nº {a.k + 1} del combinado, semilla 0). Falla igual con las semillas 1 y 2; el de rectas solo dice 9, el de curvas solo dice 6."
    f_ent = figura_entrada(x, fila)
    assert np.allclose(f_ent, X[i], atol=1e-5), "las rejillas dibujadas no son las características del compositor"
    out["votos"] = figura_votos(x, z, W, b, logits)
    out["borrar"] = figura_borrar(x, f_logits)
    out["ejemplos"] = figura_ejemplos(x, z, (X[tr] - mu) / sd, img, y, tr)
    out["rasgos"] = figura_perfil(x, X, img, y, tr, i)
    (AQUI / "resultados-cinco.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
