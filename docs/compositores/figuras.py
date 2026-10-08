#!/usr/bin/env python3
"""Las figuras de `docs/compositores/README.md`: cómo funciona un COMPOSITOR (pedido por el dueño el 2026-10-08).

No mide nada nuevo que cambie un veredicto. Usa los 13 detectores `fino` de `feat-ind` (corrida 2, sus `best.pt`) sobre
los dígitos `uci-optdigits-8px-r20261002`, entrena los dos compositores lineales de `feat-ind` exactamente como
`nn/compositor.py` (180 de train, semilla 1) y DIBUJA lo que hacen por dentro. Las cifras de los paneles de resultados
se leen de los JSON que ya están commiteados en `feat-ind/resultados/`, no se recalculan.

    python docs/compositores/figuras.py      → docs/compositores/img/*.png

Necesita torch + matplotlib (no hay un venv con los dos: se usó uno temporal con matplotlib que ve el torch del
venv de experimentos-cnn). Tarda ~10 s en CPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.colors import LinearSegmentedColormap             # noqa: E402
import numpy as np                                                # noqa: E402
import torch                                                      # noqa: E402
import torch.nn.functional as Fn                                  # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent))
from expcnn import por_id                                         # noqa: E402

FI = por_id("feat-ind").carpeta                                   # por id, nunca por ruta (CLAUDE.md del repo)
sys.path.insert(0, str(FI / "nn"))
import aplicar                                                    # noqa: E402
import compositor as C                                            # noqa: E402
import datos                                                      # noqa: E402
import features as F                                              # noqa: E402

IMG = AQUI / "img"
RES = FI / "resultados"
SUP, T1, T2, MUT = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
AZUL, NARANJA, ROJO = "#2a78d6", "#eb6834", "#e34948"
SEQ = LinearSegmentedColormap.from_list("azul", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#1c5aa6", "#6da7ec", "#f0efec", "#ef8a89", "#b52a2a"])
TINTA = LinearSegmentedColormap.from_list("tinta", ["#fcfcfb", "#0b0b0b"])
# arco-E tiene el CENTRO al Este, o sea la forma ⊂
NOMBRES = {"arco-E": "arco ⊂", "arco-W": "arco ⊃", "arco-N": "arco ∪", "arco-S": "arco ∩", "recta-V": "recta |",
           "recta-H": "recta —", "recta-S": "recta /", "recta-B": "recta \\", "lazo": "lazo ○", "esquina-NE": "esq. └",
           "esquina-NW": "esq. ┘", "esquina-SE": "esq. ┌", "esquina-SW": "esq. ┐"}
ETQ = [NOMBRES[f] for f in F.CON_TRAZO]
J = len(F.CON_TRAZO)

plt.rcParams.update({"figure.facecolor": SUP, "axes.facecolor": SUP, "savefig.facecolor": SUP, "font.size": 9,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": T2, "xtick.color": T2, "ytick.color": T2,
                     "text.color": T1, "axes.titlesize": 9})


def limpio(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#e1e0d9")


def ejes_ligeros(ax):
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", lw=0.6); ax.set_axisbelow(True)


def guardar(fig, nombre):
    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / nombre, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  {nombre}")


def entrenar(x, y, tr, semilla=1):
    """Igual que compositor.logistica (Linear, Adam lr 1e-2, L2 1e-3, 300 épocas), pero devuelve la capa."""
    torch.manual_seed(semilla)
    W = torch.nn.Linear(x.shape[1], 10)
    opt = torch.optim.Adam(W.parameters(), lr=C.LR, weight_decay=C.L2)
    xt, yt = torch.from_numpy(x[tr]), torch.from_numpy(y[tr])
    for _ in range(C.EPOCAS):
        opt.zero_grad(); Fn.cross_entropy(W(xt), yt).backward(); opt.step()
    return W


def probas(W, x):
    with torch.no_grad():
        return torch.softmax(W(torch.from_numpy(x)), 1).numpy()


def desplazar(s, dy, dx):
    """Mapas (..., 8, 8) movidos dy, dx celdas, relleno con ceros."""
    out = np.zeros_like(s)
    src = s[..., max(0, -dy):8 - max(0, dy), max(0, -dx):8 - max(0, dx)]
    out[..., max(0, dy):max(0, dy) + src.shape[-2], max(0, dx):max(0, dx) + src.shape[-1]] = src
    return out


def max3(s):
    return Fn.max_pool2d(torch.from_numpy(s.reshape(-1, J, 8, 8)), 3, 1, 1).numpy().reshape(s.shape)


# ── 1. la tubería ────────────────────────────────────────────────────────────────────────────────────────────────
def fig_tuberia(dx, s, p, i, y):
    fig = plt.figure(figsize=(12.5, 4.1))
    g = fig.add_gridspec(2, 1 + 7 + 1, width_ratios=[1.5] + [1] * 7 + [2.6], wspace=0.25, hspace=0.45)
    ax = fig.add_subplot(g[:, 0]); ax.imshow(dx[i, 0], cmap=TINTA, vmin=0, vmax=1); limpio(ax)
    ax.set_title(f"① el dígito\n(8×8, real: «{y[i]}»)", fontsize=9)
    for j in range(J):
        a = fig.add_subplot(g[j // 7, 1 + j % 7]); a.imshow(s[i, j], cmap=SEQ, vmin=0, vmax=1); limpio(a)
        a.set_title(f"{ETQ[j]}  {s[i, j].max():.2f}", fontsize=7.5, color=T1 if s[i, j].max() >= 0.5 else MUT,
                    fontweight="bold" if s[i, j].max() >= 0.5 else "normal")
    a = fig.add_subplot(g[1, 7]); a.axis("off")
    a.text(0.5, 0.5, "en negrita:\nlos que pasan\nde 0,5", ha="center", va="center", fontsize=8, color=T2)
    fig.text(0.42, 1.0, "② 13 detectores, cada uno da un mapa 8×8 (azul oscuro = «aquí está mi feature»); el número es su máximo",
             ha="center", fontsize=9)
    ab = fig.add_subplot(g[:, 8])
    cols = [AZUL if c == p[i].argmax() else "#cde2fb" for c in range(10)]
    ab.barh(range(10), p[i], color=cols, height=0.7); ab.invert_yaxis()
    ab.set_yticks(range(10)); ab.set_xlim(0, 1); ab.set_xlabel("probabilidad")
    for c in range(10):
        if p[i, c] > 0.02:
            ab.text(p[i, c] + 0.02, c, f"{p[i, c]:.2f}", va="center", fontsize=8, color=T1)
    for lado in ("top", "right"):
        ab.spines[lado].set_visible(False)
    ab.set_title("③ el COMPOSITOR lee los 13 mapas\n(13 × 64 = 832 números) y da\nuna probabilidad por dígito", fontsize=9)
    guardar(fig, "1-tuberia.png")


# ── 2. presencia contra posicional: el 6 y el 9 ─────────────────────────────────────────────────────────────────
def fig_presencia(dx, s, y, va):
    pres = s.reshape(len(s), J, -1).max(2)
    i6 = [k for k in va if y[k] == 6]; i9 = [k for k in va if y[k] == 9]
    # el par (6, 9) con la PRESENCIA más parecida: ahí es donde el compositor de presencia no tiene con qué separarlos
    d = ((pres[i6][:, None] - pres[i9][None]) ** 2).sum(2)
    a6, a9 = np.unravel_index(d.argmin(), d.shape); a6, a9 = i6[a6], i9[a9]
    ver = F.CON_TRAZO.index("lazo")                               # el hueco: lo que distingue al 6 del 9 es DÓNDE está
    fig = plt.figure(figsize=(12, 4.6))
    g = fig.add_gridspec(2, 3, width_ratios=[1, 1, 4.2], wspace=0.25, hspace=0.5)
    for r, (k, nom) in enumerate(((a6, "6"), (a9, "9"))):
        a = fig.add_subplot(g[r, 0]); a.imshow(dx[k, 0], cmap=TINTA, vmin=0, vmax=1); limpio(a); a.set_title(f"un «{nom}»")
        a = fig.add_subplot(g[r, 1]); a.imshow(s[k, ver], cmap=SEQ, vmin=0, vmax=1); limpio(a)
        a.set_title(f"mapa {ETQ[ver]}: ¿DÓNDE?", fontsize=8.5)
    ab = fig.add_subplot(g[:, 2]); w = 0.38; xs = np.arange(J)
    ab.bar(xs - w / 2, pres[a6], w, color=AZUL, label="el «6»"); ab.bar(xs + w / 2, pres[a9], w, color=NARANJA, label="el «9»")
    ab.set_xticks(xs); ab.set_xticklabels(ETQ, rotation=45, ha="right", fontsize=8); ab.set_ylim(0, 1.05)
    ab.set_ylabel("máximo del mapa"); ejes_ligeros(ab); ab.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    ab.set_title("Compositor de PRESENCIA: sólo ve estas 13 barras (el máximo de cada mapa)\n= QUÉ features hay, no DÓNDE")
    fig.text(0.01, -0.02, "Las barras del 6 y del 9 se parecen: con sólo «qué hay», los dos dígitos son casi el mismo. "
             "El compositor POSICIONAL ve el mapa entero (izquierda): el lazo del 6 está abajo y el del 9 arriba.",
             fontsize=8.5, color=T2)
    guardar(fig, "2-presencia-vs-posicional.png")


# ── 3. las plantillas: los pesos del compositor posicional ──────────────────────────────────────────────────────
def fig_plantillas(W):
    w = W.weight.detach().numpy().reshape(10, J, 8, 8); lim = np.percentile(np.abs(w), 99)
    fig, axs = plt.subplots(10, J, figsize=(J * 0.82, 10 * 0.86))
    for c in range(10):
        for j in range(J):
            a = axs[c, j]; a.imshow(w[c, j], cmap=DIV, vmin=-lim, vmax=lim); limpio(a)
            if c == 0:
                a.set_title(ETQ[j], fontsize=7.5, rotation=40, ha="left", va="bottom")
            if j == 0:
                a.set_ylabel(f"«{c}»", rotation=0, ha="right", va="center", fontsize=10, color=T1)
    fig.subplots_adjust(wspace=0.08, hspace=0.08)
    fig.text(0.5, 0.075, "Fila = un dígito · columna = un detector · cada cuadro = sus 64 pesos.  "
             "ROJO: «si esta feature aparece AQUÍ, suma para este dígito».  AZUL: «resta».  Gris: le da igual.",
             ha="center", fontsize=8.5, color=T2)
    guardar(fig, "3-plantillas.png")


# ── 4. por qué decide: la suma, detector a detector ─────────────────────────────────────────────────────────────
def fig_suma(W, s, i, y):
    w = W.weight.detach().numpy().reshape(10, J, 64); x = s[i].reshape(J, 64)
    aporte = (w * x[None]).sum(2)                                 # (10, J): lo que cada detector suma a cada dígito
    z = aporte.sum(1) + W.bias.detach().numpy(); top = np.argsort(-z)[:2]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for a, c, col in zip(axs, top, (AZUL, NARANJA)):
        a.barh(range(J), aporte[c], color=[col if v > 0 else "#c3c2b7" for v in aporte[c]], height=0.7)
        a.axvline(0, color=MUT, lw=0.8); a.set_yticks(range(J)); a.set_yticklabels(ETQ, fontsize=8); a.invert_yaxis()
        a.set_title(f"puntuación del «{c}» = {aporte[c].sum():+.1f} (detectores) {W.bias[c].item():+.1f} (sesgo) = {z[c]:+.1f}",
                    fontsize=9)
        a.set_xlabel("aporte = Σ celdas (mapa × peso)")
        for lado in ("top", "right"):
            a.spines[lado].set_visible(False)
    fig.suptitle(f"Por qué el compositor dice «{top[0]}» para este dígito (real: «{y[i]}»): gana el que suma más",
                 fontsize=10, y=1.0)
    guardar(fig, "4-la-suma.png")


# ── 5. el desplazamiento, y el máximo 3×3 ───────────────────────────────────────────────────────────────────────
def fig_max3(W, s, i, y):
    w = W.weight.detach().numpy().reshape(10, J, 8, 8); c = y[i]
    j = int(np.argmax((w[c] * s[i]).sum((1, 2))))                 # el detector que más aporta a su clase
    plantilla = np.clip(w[c, j], 0, None)
    m0 = s[i, j]; m1 = desplazar(s[i:i + 1], 0, 1)[0, j]; m2 = max3(desplazar(s[i:i + 1], 0, 1))[0, j]
    casos = [("A · tal cual", m0), ("A · dígito movido 1 celda →", m1), ("B · movido, tras el máximo 3×3", m2)]
    fig, axs = plt.subplots(1, 4, figsize=(11.5, 3.3))
    a = axs[0]; a.imshow(w[c, j], cmap=DIV, vmin=-np.abs(w[c, j]).max(), vmax=np.abs(w[c, j]).max()); limpio(a)
    a.set_title(f"plantilla: pesos de «{c}»\npara {ETQ[j]}")
    for a, (tit, m) in zip(axs[1:], casos):
        a.imshow(m, cmap=SEQ, vmin=0, vmax=1); limpio(a)
        a.contour(plantilla, levels=[plantilla.max() * 0.35], colors=[ROJO], linewidths=1.2)
        a.set_title(f"{tit}\naporte {(w[c, j] * m).sum():+.2f}")
    fig.text(0.01, -0.06, "Contorno rojo = donde la plantilla de la izquierda pone sus pesos positivos. El compositor lineal "
             "sólo suma si la evidencia cae DENTRO.\nMover el dígito una celda la saca y el aporte se hunde; el máximo 3×3 "
             "«ensancha» cada evidencia a sus 8 vecinas y la vuelve a meter.\n⚠ Ilustración: los tres aportes usan la plantilla de A. "
             "El compositor B de verdad se ENTRENA sobre los mapas ya ensanchados, así que su plantilla es otra.", fontsize=8.5, color=T2)
    guardar(fig, "5-desplazamiento-max3.png")


# ── 6. lo medido, leído de los JSON de feat-ind ─────────────────────────────────────────────────────────────────
def fig_resultados():
    comp = json.loads((RES / "compositores.json").read_text())
    comb = json.loads((RES / "compositor-comb.json").read_text())["bancos"]
    reg = json.loads((RES / "compositor-reg.json").read_text())["bancos"]
    desp = json.loads((RES / "desplazamiento.json").read_text())["casos"]["fino+grueso"]
    cur = json.loads((RES / "curva.json").read_text())
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.0), gridspec_kw={"width_ratios": [1.15, 1.1, 1.2], "wspace": 0.35})

    a = axs[0]
    filas = [("presencia\nfino", comp["presencia"]["acc_val_media"]),
             ("posicional\nfino", comp["posicional"]["acc_val_media"]),
             ("posicional\nfino+\ngrueso", comb["fino+grueso"]["lineal_val"]),
             ("regulari-\nzado (CV)\nf+g", reg["fino+grueso"]["acc_val"]),
             ("combi-\nnante\nf+g", comb["fino+grueso"]["comb_val"])]
    a.bar(range(len(filas)), [v for _, v in filas], color=[NARANJA] + [AZUL] * 4, width=0.65)
    for k, (_, v) in enumerate(filas):
        a.text(k, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
    a.set_xticks(range(len(filas))); a.set_xticklabels([n for n, _ in filas], fontsize=7.5); a.set_ylim(0.6, 1.0)
    a.set_ylabel("acierto en val (1617 dígitos)"); ejes_ligeros(a)
    a.set_title("Qué compositor (180 dígitos de train)")

    a = axs[1]; vs = ["A_posicional", "B_max3", "C_max2_4x4", "D_aumento"]
    nv = ["A\nposicional", "B\nmáx 3×3", "C\nmáx 2×2\n→ 4×4", "D ⚠\nentrenado\ncon movidos"]; w = 0.38; xs = np.arange(4)
    a.bar(xs - w / 2, [desp[v]["limpio"] for v in vs], w, color=AZUL, label="dígito tal cual")
    a.bar(xs + w / 2, [desp[v]["desplazar"] for v in vs], w, color=NARANJA, label="movido 1 celda")
    for k, v in enumerate(vs):
        a.text(k + w / 2, desp[v]["desplazar"] + 0.01, f"{desp[v]['desplazar']:.2f}", ha="center", fontsize=8)
    a.set_xticks(xs); a.set_xticklabels(nv, fontsize=7.5); a.set_ylim(0.4, 1.13); ejes_ligeros(a)
    a.legend(frameon=False, loc="upper center", ncol=2, fontsize=8); a.set_title("Tolerar el desplazamiento (fino+grueso)")

    a = axs[2]; ns = cur["tamanos"]; cv = cur["curva"]["fino+grueso"]
    a.plot(ns, [cv[str(n)]["lineal"] for n in ns], "-o", color=AZUL, lw=2, ms=5)
    a.plot(ns, [cv[str(n)]["combinante"] for n in ns], "-o", color=NARANJA, lw=2, ms=5)
    a.text(ns[-1] + 25, cv[str(ns[-1])]["lineal"] + 0.004, "lineal", color=T1, fontsize=8, va="bottom")
    a.text(ns[-1] + 25, cv[str(ns[-1])]["combinante"] - 0.004, "combinante", color=T1, fontsize=8, va="top")
    a.set_xlabel("dígitos de train del compositor"); a.set_xlim(0, 1250); a.set_ylim(0.85, 1.0); ejes_ligeros(a)
    a.set_title("El techo ~0,97 era de DATOS (test 717)")
    guardar(fig, "6-resultados.png")


def main() -> int:
    torch.manual_seed(0); torch.set_num_threads(2)
    dig = datos.digitos(); m = aplicar.mapas(dig)
    s, y, tr = m["sigma"], m["y"], m["train"]
    va = np.flatnonzero(~tr)
    x_pos = s.reshape(len(s), -1).astype(np.float32)
    W = entrenar(x_pos, y, tr)
    p = probas(W, x_pos)
    print(f"compositor posicional (semilla 1): acierto val {(p[va].argmax(1) == y[va]).mean():.4f} "
          f"(compositores.json dice {json.loads((RES / 'compositores.json').read_text())['posicional']['acc_val_por_semilla'][0]})")
    i = int(next(k for k in va if y[k] == 2 and p[k].argmax() == 2 and 0.6 < p[k, 2] < 0.95))
    fig_tuberia(dig["x"], s, p, i, y)
    fig_presencia(dig["x"], s, y, va)
    fig_plantillas(W)
    fig_suma(W, s, i, y)
    k = int(next(k for k in va if y[k] == 0 and p[k].argmax() == 0))
    fig_max3(W, s, k, y)
    fig_resultados()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
