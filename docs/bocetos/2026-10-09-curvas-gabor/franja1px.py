#!/usr/bin/env python3
"""Qué pasa con una franja central de 1 px (Gabor λ = 2), CALCULADO, con sus valores (pedido del dueño, 2026-10-10).

Tres figuras, de la causa a la consecuencia:
    12-franja-1-kernel.png   los kernels 9×9 con su valor en cada celda: λ = 2 a 0°, 30°, 45° y 60°, contra λ = 3 a 30°;
                             y el perfil de través con los desajustes reales de una recta de 1 px a 30°
    12-franja-2-recta.png    la respuesta píxel a píxel sobre esa recta (λ = 2, 3 y 6), con su valor
    12-franja-3-arco.png     un arco de 1 px (R = 12, el borde de la galería) con la configuración de bordes.py:
                             las 12 respuestas en un punto oblicuo, y a lo largo del arco la orientación, la
                             coherencia y el giro κ contra los reales

Nada se ajusta aquí: se llama a curvas.py / bordes.py y se pintan sus números.

    python franja1px.py      → imagenes/12-franja-*.png y resultados-franja1px.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import TwoSlopeNorm

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import curvas as cv                                                 # noqa: E402
import bordes as bd                                                 # noqa: E402
from curvas import plt                                              # noqa: E402

C2, C3, C6 = "#c2410c", "#2a78d6", "#8a8986"          # λ = 2 · λ = 3 · λ = 6
COLOR = {2: C2, 3: C3, 6: C6}


def _num(ax, m, fmt="{:.2f}", fs=6.5, umbral=None):
    """Escribe el valor de cada celda; en blanco sobre lo oscuro."""
    lim = umbral if umbral is not None else 0.55 * np.nanmax(np.abs(m))
    for (i, j), v in np.ndenumerate(m):
        if np.isfinite(v):
            ax.text(j, i, fmt.format(v).replace("-0.00", "0").replace("0.00", "0"), ha="center", va="center",
                    fontsize=fs, color="white" if abs(v) > lim else cv.T1)


def _limpio(ax):
    ax.set_xticks([]); ax.set_yticks([])


def figura_kernel(res: dict):
    cv._estilo()
    casos = [(2, 0), (2, 30), (2, 45), (2, 60), (3, 30)]
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1], hspace=0.32, wspace=0.12)
    res["kernels"] = {}
    for j, (lam, ang) in enumerate(casos):
        k = cv.kernel_gabor(cv.K, ang, lam)
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(k, cmap="RdBu_r", norm=TwoSlopeNorm(0, -0.5, 0.5))
        _num(ax, k, fs=6.3, umbral=0.3)
        ax.set_title(f"λ = {lam} · {ang}°" + ("  (franja 1 px)" if lam == 2 else "  (franja 1,5 px)"),
                     fontsize=10, color=COLOR[lam])
        _limpio(ax)
        res["kernels"][f"lam{lam}_{ang}"] = np.round(k, 3).tolist()
    # perfil de través: el kernel como función continua de la distancia v a su eje, y dónde caen los píxeles reales
    x = cv.recta(16, 16, 22, 30, 1); ys, xs = np.nonzero(x)
    t = np.deg2rad(30)
    v = -(xs + 0.5 - 16) * np.sin(t) + (ys + 0.5 - 16) * np.cos(t)
    vv = np.linspace(-4.5, 4.5, 400)
    ax = fig.add_subplot(gs[1, :3])
    for lam in (2, 3, 6):
        perfil = np.exp(-(vv ** 2) / (2 * (lam / 2) ** 2)) * np.cos(2 * np.pi * vv / lam)
        ax.plot(vv, perfil, color=COLOR[lam], lw=2, label=f"λ = {lam}")
        w = np.cos(2 * np.pi * v / lam) * np.exp(-(v ** 2) / (2 * (lam / 2) ** 2))
        ax.plot(v, w, "o", color=COLOR[lam], ms=6, mec=cv.SUP, mew=1)
    ax.axhline(0, color=cv.T2, lw=0.6); ax.axvspan(-0.5, 0.5, color="#e8e7e4", zorder=0)
    ax.set_xlabel("distancia v al eje del kernel (px), de través"); ax.set_ylabel("peso (sin normalizar)")
    ax.set_title("Perfil de través · puntos = los 22 píxeles de una recta de 1 px a 30°, a la distancia real a la que caen",
                 fontsize=9.5)
    ax.legend(frameon=False, fontsize=9, loc="upper right"); ax.set_xlim(-2.6, 2.6)
    for lam in (2, 3):
        wmin = float(np.cos(2 * np.pi * np.abs(v).max() / lam) * np.exp(-(np.abs(v).max() ** 2) / (2 * (lam / 2) ** 2)))
        res.setdefault("peso_en_el_peor_desajuste", {})[f"lam{lam}"] = round(wmin, 2)
    res["desajustes_v"] = np.round(np.sort(v), 2).tolist()
    ax2 = fig.add_subplot(gs[1, 3:]); ax2.axis("off")
    k0, k30, k45 = (cv.kernel_gabor(cv.K, a_, 2) for a_ in (0, 30, 45))
    d = float(np.abs(v).max())
    p2 = res["peso_en_el_peor_desajuste"]["lam2"]; p3 = res["peso_en_el_peor_desajuste"]["lam3"]
    ax2.text(0, 0.98, "\n".join([
        "Lo que se ve:",
        "",
        f"· λ = 2 a 0° y a 45° SÍ son una franja: la fila",
        f"  (o la diagonal) central +{k0[4, 4]:.2f}, sus vecinas {k0[3, 4]:+.2f}",
        f"  (a 45°: +{k45[4, 4]:.2f} y {k45[3, 4]:+.2f}). Son orientaciones",
        "  que la rejilla puede dibujar con 1 px.",
        "· A 30° y 60° la franja de 1 px NO cabe en la",
        "  rejilla: celdas sueltas, y la fila central",
        "  alterna " + " · ".join(f"{v:+.2f}" for v in k30[4, 4:]).replace("+0.00", "0").replace("-0.00", "0"),
        "  (del centro hacia fuera).",
        "",
        "· Una recta de 1 px a 30° es una escalera: sus",
        f"  píxeles caen hasta {d:.2f} px fuera del eje.",
        f"  Ahí el peso es {p2:+.2f} con λ = 2",
        f"  y {p3:+.2f} con λ = 3 (máximo: 1).",
        "  Con λ = 2, medio píxel de desajuste",
        "  ya es el cero del coseno.",
    ]), va="top", fontsize=10, color=cv.T1, family="monospace")
    fig.suptitle("1 · El kernel: con franja de 1 px, sólo 0°, 45°, 90° y 135° son una franja; las demás, celdas sueltas",
                 fontsize=12, color=cv.T1)
    fig.savefig(cv.IMG / "12-franja-1-kernel.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print("→ 12-franja-1-kernel.png")


def figura_recta(res: dict):
    cv._estilo()
    x = cv.recta(16, 16, 22, 30, 1); ys, xs = np.nonzero(x)
    t = np.deg2rad(30)
    a = (xs + 0.5 - 16) * np.cos(t) + (ys + 0.5 - 16) * np.sin(t); o = np.argsort(a)
    y0, y1, x0, x1 = ys.min() - 1, ys.max() + 2, xs.min() - 1, xs.max() + 2
    res["recta_30"] = {}
    mapas = {}
    for lam in (2, 3, 6):
        k = cv.kernel_gabor(cv.K, 30, lam)
        r = F.relu(F.conv2d(torch.from_numpy(x.astype(np.float32))[None, None], torch.from_numpy(k)[None, None],
                            padding=cv.K // 2))[0, 0].numpy()
        res["recta_30"][f"lam{lam}"] = np.round(r[ys, xs][o], 2).tolist()
        mapas[lam] = r
    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9], width_ratios=[1, 1], hspace=0.2, wspace=0.06)
    for j, lam in enumerate((2, 3)):              # λ = 6 sólo va en la curva: con tres mapas los números no caben
        r = mapas[lam]
        ax = fig.add_subplot(gs[0, j])
        m = np.where(x == 1, r, np.nan)[y0:y1, x0:x1]
        ax.imshow(r[y0:y1, x0:x1], cmap="Greys", vmin=0, vmax=2.4)
        ax.imshow(np.where(np.isfinite(m), 1, np.nan), cmap="Oranges", vmin=0, vmax=4, alpha=0.35)
        _num(ax, m, fmt="{:.2f}", fs=7.5, umbral=1.4)
        ax.set_title(f"λ = {lam}: respuesta del kernel de 30° en cada píxel de la recta", fontsize=10.5, color=COLOR[lam])
        _limpio(ax)
    ax = fig.add_subplot(gs[1, 0])
    for lam in (2, 3, 6):
        s = np.array(res["recta_30"][f"lam{lam}"])
        ax.plot(np.arange(len(s)), s, "-o", color=COLOR[lam], lw=2, ms=5,
                label=f"λ = {lam}:  {s[2:-2].min():.2f} – {s[2:-2].max():.2f} (sin los extremos)")
    ax.axhline(cv.TAU, color=cv.T1, lw=1, ls="--"); ax.text(21.3, cv.TAU + 0.04, "TAU 1,2", fontsize=8.5, ha="right")
    ax.set_xlabel("píxel de la recta, de un extremo al otro"); ax.set_ylabel("respuesta r(30°)")
    ax.set_ylim(0, 2.4); ax.legend(frameon=False, fontsize=9.5, loc="lower center")
    ax.set_title("La misma recta y la misma orientación: con λ = 2 la respuesta sube y baja según caiga cada escalón",
                 fontsize=9.5)
    s2, s3, s6 = (np.array(res["recta_30"][f"lam{lam}"])[2:-2] for lam in (2, 3, 6))
    ax2 = fig.add_subplot(gs[1, 1]); ax2.axis("off")
    ax2.text(0.04, 0.95, "\n".join([
        f"Sin contar los 2 píxeles de cada extremo ({len(s2)} píxeles):",
        "",
        f"λ = 2   {s2.min():.2f} – {s2.max():.2f}   (×{s2.max() / s2.min():.1f})   bajo TAU 1,2: {int((s2 < cv.TAU).sum())}",
        f"λ = 3   {s3.min():.2f} – {s3.max():.2f}   (×{s3.max() / s3.min():.1f})   bajo TAU 1,2: {int((s3 < cv.TAU).sum())}",
        f"λ = 6   {s6.min():.2f} – {s6.max():.2f}   (×{s6.max() / s6.min():.1f})   bajo TAU 1,2: {int((s6 < cv.TAU).sum())}",
        "",
        "Los mínimos de λ = 2 (0,63) son los cuatro píxeles",
        "que caen a 0,42–0,45 px del eje (el escalón):",
        "ahí la franja de 1 px casi no los ve (fig. 1).",
        "Un borde roto así deja huecos en la cadena.",
    ]), va="top", fontsize=10.5, color=cv.T1, family="monospace")
    fig.suptitle("2 · Una recta de 1 px a 30° (un borde oblicuo): la respuesta píxel a píxel", fontsize=12, color=cv.T1)
    fig.savefig(cv.IMG / "12-franja-2-recta.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print("→ 12-franja-2-recta.png")


def figura_arco(res: dict):
    cv._estilo()
    x = cv.arco(16, 10, 12, 0, 22, 1); ys, xs = np.nonzero(x)
    phi = np.degrees(np.arctan2(ys + 0.5 - 22, xs + 0.5 - 16)); o = np.argsort(phi)
    tang = (phi[o] + 90) % 180
    datos = {}
    for lam in (2, 3):
        bd.LAM_B = lam; bd.configurar()
        c, kappa, mask, vs = cv.detectar(x)
        curvos, rectos = bd.trozos(kappa, vs)
        datos[lam] = {"c": c, "kappa": kappa, "theta": c["theta"][ys, xs][o], "coh": c["coh"][ys, xs][o],
                      "k": kappa[ys, xs][o], "r": c["r"], "veredicto": [v["que"] for v in vs],
                      "curvos": [t["radio"] for t in curvos]}
    # tangente continua (sin el salto 180 → 0) para poder dibujarla
    def cont(t):
        return np.degrees(np.unwrap(np.deg2rad(2 * t)) / 2)
    i_obl = int(np.argmin(np.abs(tang - 30)))                 # el píxel cuya tangente real es la más cercana a 30°
    py, px = ys[o][i_obl], xs[o][i_obl]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 0.8, 0.8], hspace=0.38, wspace=0.18)
    for j, lam in enumerate((2, 3)):
        d = datos[lam]
        ax = fig.add_subplot(gs[0, j])
        ok = np.isfinite(d["kappa"]) & (x == 1)
        ax.imshow(np.where(x == 1, 1.0, np.nan), cmap="Greys", vmin=0, vmax=2.5)
        ax.imshow(np.where(ok, 1.0, np.nan), cmap="Oranges", vmin=0, vmax=1.6)
        for yy, xx in zip(ys, xs):
            ax.text(xx, yy, f"{d['c']['coh'][yy, xx]:.1f}".replace("0.", "."), ha="center", va="center", fontsize=6.3)
        ax.plot(px, py, "s", ms=13, mfc="none", mec=COLOR[lam], mew=2)
        ax.set_xlim(4.5, 27.5); ax.set_ylim(17.5, 7.5); _limpio(ax)
        ax.set_title(f"λ = {lam}: coherencia en cada píxel · naranja = giro medible\n"
                     f"veredicto: {', '.join(d['veredicto']) or 'nada'} · curvas: "
                     f"{', '.join(f'R≈{r:.0f}' for r in d['curvos']) or 'ninguna'}  (real R = 12)",
                     fontsize=9.5, color=COLOR[lam])
    ax = fig.add_subplot(gs[0, 2])
    w = 0.38
    for lam, dx in ((2, -w / 2), (3, w / 2)):
        ri = datos[lam]["r"][:, py, px]
        ax.bar(cv.THETAS / 15 + dx, ri, width=w, color=COLOR[lam], label=f"λ = {lam}")
        for i, v in enumerate(ri):
            if v > 0.05:
                ax.text(i + dx, v + 0.03, f"{v:.2f}", ha="center", fontsize=6.5, rotation=90, color=cv.T1)
    ax.set_xticks(range(12)); ax.set_xticklabels([f"{t:.0f}°" for t in cv.THETAS], fontsize=7.5)
    ax.axvline(tang[i_obl] / 15, color=cv.T1, ls="--", lw=1)
    ax.text(tang[i_obl] / 15 + 0.15, ax.get_ylim()[1] * 0.55, f"tangente real {tang[i_obl]:.0f}°", fontsize=8.5)
    ax.set_title(f"Las 12 respuestas en el píxel recuadrado\n"
                 f"coherencia: λ2 {datos[2]['coh'][i_obl]:.2f} · λ3 {datos[3]['coh'][i_obl]:.2f}  (mínimo {cv.COH_MIN})",
                 fontsize=9.5)
    ax.legend(frameon=False, fontsize=9)
    n = np.arange(len(tang))
    paneles = [("orientación θ (°) · continua: 200° = 20°", lambda d: cont(d["theta"]), cont(tang), "tangente real"),
               ("coherencia", lambda d: d["coh"], None, None),
               ("|κ| (°/px)", lambda d: np.abs(d["k"]), np.full(len(n), 57.3 / 12), "real 57,3/12 = 4,8")]
    for j, (nombre, f, real, etq_real) in enumerate(paneles):
        ax = fig.add_subplot((gs[1, :2], gs[1, 2], gs[2, :2])[j])
        for lam in (2, 3):
            ax.plot(n, f(datos[lam]), "-o", color=COLOR[lam], lw=2, ms=4.5, label=f"λ = {lam}")
        if real is not None:
            ax.plot(n, real, color=cv.T1, lw=1.2, ls="--", label=etq_real)
        if nombre == "coherencia":
            ax.axhline(cv.COH_MIN, color=cv.T1, lw=1, ls=":"); ax.text(0, cv.COH_MIN + 0.02, f"mínimo {cv.COH_MIN}", fontsize=8)
        if nombre.startswith("|κ|"):
            ax.axhline(bd.KAPPA_MIN_B, color=cv.T1, lw=1, ls=":")
            ax.text(0, bd.KAPPA_MIN_B + 0.12, f"curva desde {bd.KAPPA_MIN_B:g} °/px", fontsize=8)
            for lam in (2, 3):
                kk = np.abs(datos[lam]["k"])
                for i in np.nonzero(np.isfinite(kk))[0]:
                    ax.text(i, kk[i] + 0.25, f"{kk[i]:.1f}", ha="center", fontsize=7, color=COLOR[lam])
            ax.set_ylim(0, 6.5)
            ax.text(len(n) - 1, 0.3, "sin punto = no medible (coherencia baja o vecino fuera del trazo)",
                    ha="right", fontsize=8, color=cv.T2)
        ax.set_xlabel("píxel del arco, de un extremo al otro"); ax.set_title(nombre, fontsize=10)
        ax.legend(frameon=False, fontsize=8.5)
    axt = fig.add_subplot(gs[2, 2]); axt.axis("off")
    d2, d3 = datos[2], datos[3]
    axt.text(0, 0.98, "\n".join([
        "La cadena, con λ = 2:",
        "",
        "1 a 30° y 60° el kernel son celdas sueltas (fig. 1)",
        "2 responde a VARIAS orientaciones a la vez",
        "  → coherencia baja en los tramos oblicuos",
        f"  ({np.nanmin(d2['coh']):.2f}–{np.nanmax(d2['coh']):.2f}; λ = 3: "
        f"{np.nanmin(d3['coh']):.2f}–{np.nanmax(d3['coh']):.2f})",
        "3 bajo el mínimo, la orientación no se cree",
        "  → el giro sólo se mide arriba, donde la",
        "  tangente es casi 0° (la rejilla)",
        f"4 ahí sale |κ| {np.nanmin(np.abs(d2['k'])):.1f}–{np.nanmax(np.abs(d2['k'])):.1f}: la orientación",
        "  se APLANA hacia 0° (se pega a la rejilla;",
        "  panel θ, píxeles 9–12) y no llega a 2 °/px → «recta»",
        "",
        f"Con λ = 3: |κ| {np.nanmin(np.abs(d3['k'])):.1f}–{np.nanmax(np.abs(d3['k'])):.1f} en {int(np.isfinite(d3['k']).sum())} píxeles",
        f"→ curva R ≈ {d3['curvos'][0]:.0f}" if d3["curvos"] else "",
    ]), va="top", fontsize=9.5, color=cv.T1, family="monospace")
    fig.suptitle("3 · Un borde curvo de 1 px (arco R = 12), con la configuración de bordes.py: dónde se rompe", fontsize=12,
                 color=cv.T1)
    fig.savefig(cv.IMG / "12-franja-3-arco.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print("→ 12-franja-3-arco.png")
    res["arco_R12_1px"] = {f"lam{lam}": {"theta": np.round(d["theta"], 1).tolist(), "coherencia": np.round(d["coh"], 2).tolist(),
                                         "kappa": [None if not np.isfinite(k) else round(float(k), 2) for k in d["k"]],
                                         "veredicto": d["veredicto"], "curvas_R": d["curvos"]}
                           for lam, d in datos.items()}
    res["arco_R12_1px"]["tangente_real"] = np.round(tang, 1).tolist()


def main() -> int:
    res: dict = {}
    figura_kernel(res)
    figura_recta(res)
    figura_arco(res)
    (AQUI / "resultados-franja1px.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
