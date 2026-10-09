#!/usr/bin/env python3
"""Boceto: un detector de CURVAS con un banco de Gabor FIJO como entrada (2026-10-09, pedido del dueño).

La idea del dueño (2026-10-08): «una curva = cadena de detecciones de rectas cortas cuya orientación gira».
Aquí está montada entera, sin entrenar nada, y aplicada a rectas, curvas, esquinas, manchas y dígitos reales.

    1  BANCO     12 Gabor pares 9×9 (λ = 6 px, el de rect-lin), cada 15°; r_i = ReLU(G_i ⋆ x)
    2  CAMPO     en cada píxel, la orientación LOCAL por media vectorial en ángulo doble:
                 θ = ½·atan2(Σ r_i sin 2θ_i, Σ r_i cos 2θ_i), y su COHERENCIA c = |Σ r_i e^{2iθ_i}| / Σ r_i
                 (c ≈ 1: una sola orientación manda; c ≈ 0: responden todas por igual → mancha o cruce)
    3  GIRO      en cada píxel del trazo, la orientación a ±d px A LO LARGO de su propia tangente:
                 κ = (θ(p + d·u) − θ(p − d·u)) / 2d   [grados por píxel]   → radio ≈ 57,3 / |κ|
    4  VEREDICTO por componente conexa del trazo, con tres números: coherencia (c), giro total (Δθ) y κ mediana:
                 c baja → MANCHA · Δθ < 15° → RECTA · gira pero κ ≈ 0 en los píxeles (el giro es de golpe) → ESQUINA ·
                 si no → CURVA, con su radio; y si κ cambia de signo a mitad, S.

Convención de ángulo de rect-lin: x a la derecha, y hacia ABAJO; 0° = —, 45° = \\, 90° = |, 135° = /.
`kernel_gabor` está COPIADO de `nn/modelo.py` del experimento `rect-lin` (no se importa de otro experimento).

    python curvas.py              → imagenes/*.png y resultados.json
    python curvas.py --banco      → además pasa el banco de rect-lin entero y escribe resultados-banco.{json,txt}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, label, map_coordinates

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.colors import TwoSlopeNorm                          # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parents[2]))
from expcnn import exigir_dataset                                   # noqa: E402

IMG = AQUI / "imagenes"
LADO = 32
K, LAM, N_ORI = 9, 6.0, 12
THETAS = np.arange(N_ORI) * 180.0 / N_ORI
D_GIRO = 4           # px a cada lado, a lo largo de la tangente
SIGMA_CAMPO = 1.0    # suavizado del campo (px) antes de leer la orientación: sin él, una recta a 30° es una escalera
                     # y su orientación local oscila a lo largo (κ mediana 1,25 °/px en una recta, medido con σ = 0)
NUCLEO = 0.5         # para los números de una componente sólo cuentan los píxeles con energía ≥ NUCLEO × su máximo
                     # (el lomo del trazo), no la franja que el kernel 9×9 enciende alrededor
TAU = 1.2            # umbral de energía (max_i r_i) para «aquí hay trazo». Gabor de norma 1 sobre imagen binaria:
                     # una recta de 3 px da 2,8–3,3 (medido en rect-bor); una mancha, mediana 1,55
COH_MIN = 0.35       # por debajo, la orientación no se cree (mancha, cruce, esquina justa)
TURN_RECTA = 12.0    # giro total (grados) por debajo del cual es recta
KAPPA_MIN = 1.0      # °/px: por debajo, los píxeles «no giran» (radio > 57 px a 32×32 es una recta)
CONCENTRADO = 3.0    # esquina: el giro está en pocos píxeles → p90(|κ|) ≥ CONCENTRADO × mediana(|κ|), y p90 ≥ KAPPA_ESQ
KAPPA_ESQ = 3.0
MIN_TROZO = 4        # px: un trozo recto o curvo más corto no se cuenta
MIN_PX = 6
SUP, T1, T2, NAR, AZU = "#fcfcfb", "#0b0b0b", "#52514e", "#eb6834", "#2a78d6"
_YY, _XX = np.mgrid[0:LADO, 0:LADO] + 0.5


# ─── 1. el banco ──────────────────────────────────────────────────────────────────────────────────────────────────
def kernel_gabor(k: int, ang: float, lam: float = LAM) -> np.ndarray:
    """Gabor PAR (copiado de rect-lin): positivo en la línea media, negativo a los lados, alargado a lo largo."""
    r = (k - 1) / 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    t = np.deg2rad(ang)
    u = xx * np.cos(t) + yy * np.sin(t)
    v = -xx * np.sin(t) + yy * np.cos(t)
    g = np.exp(-(u ** 2) / (2 * (k / 3) ** 2) - (v ** 2) / (2 * (lam / 2) ** 2)) * np.cos(2 * np.pi * v / lam)
    g -= g.mean()
    return (g / np.linalg.norm(g)).astype(np.float32)


GABOR = torch.from_numpy(np.stack([kernel_gabor(K, t) for t in THETAS])[:, None])      # (12,1,9,9)
COS2 = np.cos(np.deg2rad(2 * THETAS))[:, None, None]
SIN2 = np.sin(np.deg2rad(2 * THETAS))[:, None, None]


@torch.no_grad()
def responder(x: np.ndarray) -> np.ndarray:
    """(H,W) → (12,H,W): ReLU(G_i ⋆ x)."""
    xt = torch.from_numpy(x.astype(np.float32))[None, None]
    return F.relu(F.conv2d(xt, GABOR, padding=K // 2))[0].numpy()


# ─── 2. el campo de orientación ───────────────────────────────────────────────────────────────────────────────────
def campo(x: np.ndarray) -> dict:
    r = responder(x)
    Emax = r.max(0)
    E = gaussian_filter(r.sum(0), SIGMA_CAMPO)
    Cx, Cy = gaussian_filter((r * COS2).sum(0), SIGMA_CAMPO), gaussian_filter((r * SIN2).sum(0), SIGMA_CAMPO)
    theta = (0.5 * np.degrees(np.arctan2(Cy, Cx))) % 180.0
    coh = np.hypot(Cx, Cy) / (E + 1e-6)
    return {"r": r, "E": E, "Emax": Emax, "Cx": Cx, "Cy": Cy, "theta": theta, "coh": coh}


# ─── 3. el giro a lo largo de la tangente ─────────────────────────────────────────────────────────────────────────
def _muestra(c: dict, ys: np.ndarray, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orientación y coherencia del campo en puntos fraccionarios (bilineal sobre Cx, Cy, E)."""
    cx = map_coordinates(c["Cx"], [ys, xs], order=1, mode="nearest")
    cy = map_coordinates(c["Cy"], [ys, xs], order=1, mode="nearest")
    e = map_coordinates(c["E"], [ys, xs], order=1, mode="nearest")
    return (0.5 * np.degrees(np.arctan2(cy, cx))) % 180.0, np.hypot(cx, cy) / (e + 1e-6)


def giro(c: dict, d: int = D_GIRO) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """κ (°/px) por píxel del trazo; NaN donde no se puede medir. Devuelve (kappa, mascara_trazo, energía mínima en ±d)."""
    mask = c["Emax"] > TAU
    kappa = np.full(mask.shape, np.nan); emin = np.zeros(mask.shape)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return kappa, mask, emin
    t = np.deg2rad(c["theta"][ys, xs]); ux, uy = np.cos(t), np.sin(t)
    th_p, co_p = _muestra(c, ys + d * uy, xs + d * ux)
    th_m, co_m = _muestra(c, ys - d * uy, xs - d * ux)
    e_p = map_coordinates(c["Emax"], [ys + d * uy, xs + d * ux], order=1, mode="constant")
    e_m = map_coordinates(c["Emax"], [ys - d * uy, xs - d * ux], order=1, mode="constant")
    dth = (th_p - th_m + 90.0) % 180.0 - 90.0                       # diferencia mínima, en (−90, 90]
    # medible sólo si los dos puntos a ±d siguen sobre el trazo y con orientación creíble: en los últimos d px de un
    # trazo, q± cae fuera y el giro saldría a la mitad (medido: R = 27 estimado 48 sin esta condición)
    ok = (c["coh"][ys, xs] > COH_MIN) & (co_p > COH_MIN) & (co_m > COH_MIN) & (e_p > TAU) & (e_m > TAU)
    kappa[ys[ok], xs[ok]] = dth[ok] / (2 * d)
    emin[ys, xs] = np.minimum(e_p, e_m)
    return kappa, mask, emin


# ─── 4. el veredicto por componente ───────────────────────────────────────────────────────────────────────────────
def _giro_total(theta: np.ndarray) -> float:
    """Amplitud (p2,5–p97,5) de las orientaciones alrededor de su media circular (ángulo doble)."""
    t2 = np.deg2rad(2 * theta)
    media = 0.5 * np.degrees(np.arctan2(np.sin(t2).mean(), np.cos(t2).mean()))
    dev = (theta - media + 90.0) % 180.0 - 90.0
    return float(np.percentile(dev, 97.5) - np.percentile(dev, 2.5))


def veredicto(c: dict, kappa: np.ndarray, mask: np.ndarray, emin: np.ndarray) -> list[dict]:
    lab, n = label(mask, structure=np.ones((3, 3)))
    out = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < MIN_PX:
            continue
        lomo = NUCLEO * c["Emax"][m].max()
        nucleo = m & (c["Emax"] >= lomo) & (emin >= lomo)   # el lomo del trazo, y a ≥ d px de sus extremos: en los
        coh = c["coh"][m & (c["Emax"] >= lomo)]              # últimos px la orientación se tuerce (medido: recta de 30°
        fiable = nucleo & (c["coh"] > COH_MIN)               # → θ 28–30° en el cuerpo y 16°/44° en las puntas)
        kk = kappa[nucleo]; kk = kk[np.isfinite(kk)]
        r = {"n_px": int(m.sum()), "n_nucleo": int(nucleo.sum()), "coh_med": round(float(np.median(coh)), 2),
             "giro_total": round(_giro_total(c["theta"][fiable]), 1) if fiable.sum() >= 3 else 0.0,
             "frac_medible": round(float(len(kk) / max(nucleo.sum(), 1)), 2),
             "kappa_med": round(float(np.median(np.abs(kk))), 2) if len(kk) else 0.0,
             "kappa_p90": round(float(np.percentile(np.abs(kk), 90)), 2) if len(kk) else 0.0,
             "signo_dominante": round(float(max((kk > 0).mean(), (kk < 0).mean())), 2) if len(kk) else 0.0}
        medible = nucleo & np.isfinite(kappa)
        r["trozos_rectos"] = int(sum(1 for z in range(1, label(medible & (np.abs(kappa) < KAPPA_MIN), structure=np.ones((3, 3)))[1] + 1)
                                     if (label(medible & (np.abs(kappa) < KAPPA_MIN), structure=np.ones((3, 3)))[0] == z).sum() >= MIN_TROZO))
        r["trozos_curvos"] = int(sum(1 for z in range(1, label(medible & (np.abs(kappa) >= KAPPA_MIN), structure=np.ones((3, 3)))[1] + 1)
                                     if (label(medible & (np.abs(kappa) >= KAPPA_MIN), structure=np.ones((3, 3)))[0] == z).sum() >= MIN_TROZO))
        # ⚠ el SIGNO de κ no se usa: la tangente está definida módulo 180°, así que el signo se da la vuelta donde
        # θ cruza 0°/180°. Distinguir una S de una C pide recorrer la cadena con un sentido (pendiente).
        if r["coh_med"] < COH_MIN:
            r["que"] = "mancha"
        elif len(kk) < MIN_TROZO:
            r["que"] = "corto"            # hay trazo, pero no da para medir si gira (menos de ~d·2 + 4 px útiles)
        elif r["giro_total"] < TURN_RECTA:
            r["que"] = "recta"
        elif r["kappa_med"] < KAPPA_MIN or (r["kappa_p90"] >= KAPPA_ESQ and r["kappa_p90"] >= CONCENTRADO * r["kappa_med"]):
            r["que"] = "esquina"          # gira en TOTAL, pero no en los píxeles (o sólo en unos pocos): el giro es de
        else:                             # golpe. Una esquina, o un cruce
            r["que"] = "curva"; r["radio"] = round(57.3 / r["kappa_med"], 1)
        r["mascara"] = m; r["nucleo"] = nucleo
        out.append(r)
    return sorted(out, key=lambda z: -z["n_px"])


def detectar(x: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
    c = campo(x); kappa, mask, emin = giro(c); return c, kappa, mask, veredicto(c, kappa, mask, emin)


# ─── las formas de prueba (dibujadas igual que en rect-lin) ───────────────────────────────────────────────────────
def _u(ang):
    t = np.deg2rad(ang); return np.array([np.cos(t), np.sin(t)])


def recta(cx, cy, largo, ang, grosor):
    u = _u(ang); px, py = _XX - cx, _YY - cy
    a = px * u[0] + py * u[1]; p = -px * u[1] + py * u[0]
    return ((np.abs(a) <= largo / 2) & (np.abs(p) <= grosor / 2)).astype(np.uint8)


def arco(mx, my, radio, ang, largo, grosor):
    u = _u(ang); n = np.array([-u[1], u[0]]); c = np.array([mx, my]) + radio * n
    dx, dy = _XX - c[0], _YY - c[1]; r = np.hypot(dx, dy)
    a_mid = np.arctan2(-n[1], -n[0]); da = np.angle(np.exp(1j * (np.arctan2(dy, dx) - a_mid)))
    medio = min(largo / radio, 2 * np.pi) / 2
    return ((np.abs(r - radio) <= grosor / 2) & (np.abs(da) <= medio)).astype(np.uint8)


def circulo(cx, cy, radio, grosor):
    r = np.hypot(_XX - cx, _YY - cy); return (np.abs(r - radio) <= grosor / 2).astype(np.uint8)


def discos(centros, diametro=2.0):
    img = np.zeros((LADO, LADO), bool)
    for x, y in centros:
        img |= (_XX - x) ** 2 + (_YY - y) ** 2 <= (diametro / 2) ** 2
    return img.astype(np.uint8)


def punteada(cx, cy, largo, ang, sep):
    u = _u(ang); n = int(np.floor(largo / sep)) + 1; t = (np.arange(n) - (n - 1) / 2) * sep
    return discos(np.stack([cx + t * u[0], cy + t * u[1]], 1))


def esquina(cx, cy, largo, ang1, ang2, grosor):
    """Dos semirrectas desde (cx,cy) hacia ang1 y ang2."""
    u1, u2 = _u(ang1), _u(ang2)
    a = recta(cx + u1[0] * largo / 2, cy + u1[1] * largo / 2, largo, ang1, grosor)
    b = recta(cx + u2[0] * largo / 2, cy + u2[1] * largo / 2, largo, ang2, grosor)
    return np.maximum(a, b)


def formas() -> list[tuple[str, np.ndarray, str]]:
    rng = np.random.default_rng(20261009)
    s_curva = np.maximum(arco(16, 10, 6, 0, 14, 3), arco(16, 22, 6, 180, 14, 3))
    return [
        ("recta 0° · 3 px", recta(16, 16, 22, 0, 3), "recta"),
        ("recta 30° · 3 px", recta(16, 16, 22, 30, 3), "recta"),
        ("recta 110° · 2 px", recta(16, 16, 22, 110, 2), "recta"),
        ("recta 0° · 8 px (gruesa)", recta(16, 16, 22, 0, 8), "recta"),
        ("la misma, a escala ½ (16×16)", recta(16, 16, 22, 0, 8).reshape(16, 2, 16, 2).mean((1, 3)), "recta"),
        ("arco R = 6 · 3 px", arco(16, 12, 6, 0, 20, 3), "curva R 6"),
        ("arco R = 9 · 3 px", arco(16, 12, 9, 0, 20, 3), "curva R 9"),
        ("arco R = 12 · 3 px", arco(16, 12, 12, 30, 20, 3), "curva R 12"),
        ("arco R = 18 · 3 px", arco(16, 12, 18, 60, 20, 3), "curva R 18"),
        ("arco R = 27 · 3 px", arco(16, 12, 27, 0, 20, 3), "curva R 27"),
        ("arco R = 40 · 3 px", arco(16, 12, 40, 0, 20, 3), "curva R 40"),
        ("arco R = 9 · 8 px (grueso)", arco(16, 12, 9, 0, 20, 8), "curva R 9"),
        ("el mismo, a escala ½ (16×16)", arco(16, 12, 9, 0, 20, 8).reshape(16, 2, 16, 2).mean((1, 3)), "curva R 4.5"),
        ("círculo R = 8 · 3 px", circulo(16, 16, 8, 3), "curva R 8"),
        ("esquina L (90°) · brazos 14", esquina(11, 11, 14, 0, 90, 3), "esquina"),
        ("esquina L (90°) · brazos 22", esquina(6, 6, 22, 0, 90, 3), "esquina"),
        ("V (60°)", esquina(16, 24, 14, 240, 300, 3), "esquina"),
        ("S (dos arcos R = 6)", s_curva, "curva S"),
        ("cruz +", np.maximum(recta(16, 16, 20, 0, 3), recta(16, 16, 20, 90, 3)), "esquina (cruce)"),
        ("punteada 20° · cada 4 px", punteada(16, 16, 22, 20, 4), "recta (punteada)"),
        ("la misma, a escala ½ (16×16)", punteada(16, 16, 22, 20, 4).reshape(16, 2, 16, 2).mean((1, 3)), "recta (punteada)"),
        ("mancha r = 4", (np.hypot(_XX - 16, _YY - 16) <= 4).astype(np.uint8), "mancha"),
        ("puntos sueltos", discos(rng.uniform(4, 28, size=(7, 2))), "nada"),
        ("ruido 1 %", (rng.random((LADO, LADO)) < 0.01).astype(np.uint8), "nada"),
    ]


# ─── figuras ──────────────────────────────────────────────────────────────────────────────────────────────────────
def _estilo():
    plt.rcParams.update({"font.size": 8, "figure.facecolor": SUP, "axes.facecolor": SUP, "axes.titlecolor": T1,
                         "axes.edgecolor": T2})


def _ticks(ax, c: dict, mask: np.ndarray, paso: int = 2):
    """La «cadena de rectas cortas»: un palito por píxel del trazo (cada `paso`), en su orientación local."""
    ys, xs = np.nonzero(mask)
    sel = ((ys % paso) == 0) & ((xs % paso) == 0)
    for y, x in zip(ys[sel], xs[sel]):
        t = np.deg2rad(c["theta"][y, x]); L = 0.9 * min(1.0, c["coh"][y, x] / 0.8)
        ax.plot([x - L * np.cos(t), x + L * np.cos(t)], [y - L * np.sin(t), y + L * np.sin(t)], color=AZU, lw=1.1)


def _fila(fig, gs_fila, nombre, x, esperado, c, kappa, mask, vs):
    axs = [fig.add_subplot(gs_fila[0, j]) for j in range(5)]
    axs[0].imshow(x, cmap="gray_r", vmin=0, vmax=1); axs[0].set_title("imagen")
    axs[0].set_ylabel(nombre, fontsize=8.5, color=T1)
    axs[1].imshow(c["Emax"], cmap="Oranges", vmin=0, vmax=max(3.3, c["Emax"].max())); axs[1].set_title("energía  max_i r_i")
    axs[1].contour(mask.astype(float), levels=[0.5], colors=[T2], linewidths=0.5)
    for v in vs:
        axs[4].contour(v["nucleo"].astype(float), levels=[0.5], colors=[AZU], linewidths=0.6)
    axs[2].imshow(x, cmap="gray_r", vmin=0, vmax=1, alpha=0.18); _ticks(axs[2], c, mask); axs[2].set_title("orientación local θ")
    axs[3].imshow(c["coh"] * mask, cmap="Greys", vmin=0, vmax=1); axs[3].set_title("coherencia c")
    kmax = max(4.0, np.nanmax(np.abs(kappa)) if np.isfinite(kappa).any() else 0)
    im = axs[4].imshow(np.where(mask, kappa, np.nan), cmap="RdBu_r", norm=TwoSlopeNorm(0, -kmax, kmax))
    axs[4].imshow(np.where(mask & ~np.isfinite(kappa), 1.0, np.nan), cmap="Greys", vmin=0, vmax=3)
    axs[4].set_title("giro κ (°/px)")
    for a in axs:
        a.set_xticks([]); a.set_yticks([]); a.set_xlim(-0.5, x.shape[1] - 0.5); a.set_ylim(x.shape[0] - 0.5, -0.5)
    axt = fig.add_subplot(gs_fila[0, 5]); axt.axis("off")
    lineas = [f"esperado: {esperado}", ""]
    if not vs:
        lineas.append("→ NADA (sin trazo)")
    for v in vs[:3]:
        extra = f" · R ≈ {v['radio']}" if "radio" in v else ""
        lineas.append(f"→ {v['que'].upper()}{extra}")
        lineas.append(f"   {v['n_nucleo']} px · c {v['coh_med']} · Δθ {v['giro_total']}°")
        lineas.append(f"   κ med {v['kappa_med']} · p90 {v['kappa_p90']}")
        lineas.append(f"   trozos: {v['trozos_rectos']} rectos · {v['trozos_curvos']} curvos")
    lineas.append(f"\nEmax {c['Emax'].max():.2f}")
    bien = (not vs and esperado == "nada") or (vs and vs[0]["que"].split(" ")[0] == esperado.split(" ")[0])
    axt.text(0, 0.95, "\n".join(lineas), va="top", fontsize=8, color=T1 if bien else NAR, family="monospace")
    return bool(bien)


def figura_formas(items, nombre_png, titulo) -> list[dict]:
    _estilo()
    fig = plt.figure(figsize=(13, 1.75 * len(items) + 0.8))
    gs = fig.add_gridspec(len(items), 1, hspace=0.45)
    res = []
    for i, (nombre, x, esperado) in enumerate(items):
        c, kappa, mask, vs = detectar(x)
        sub = gs[i].subgridspec(1, 6, wspace=0.25, width_ratios=[1, 1, 1, 1, 1, 1.6])
        bien = _fila(fig, sub, nombre, x, esperado, c, kappa, mask, vs)
        res.append({"forma": nombre, "esperado": esperado, "acierta": bien, "emax": round(float(c["Emax"].max()), 2),
                    "componentes": [{k: v for k, v in z.items() if k not in ("mascara", "nucleo")} for z in vs]})
    fig.suptitle(titulo, color=T1, fontsize=11, y=0.995)
    fig.subplots_adjust(top=0.975, bottom=0.01)
    fig.savefig(IMG / nombre_png, dpi=105, bbox_inches="tight"); plt.close(fig)
    print("→", IMG / nombre_png)
    return res


def figura_banco():
    """Los 12 kernels y sus 12 mapas sobre un arco: lo que entra al campo de orientación."""
    _estilo()
    x = arco(16, 12, 9, 0, 20, 3); r = responder(x)
    fig, axs = plt.subplots(3, N_ORI, figsize=(14, 4.2))
    for i in range(N_ORI):
        Kg = kernel_gabor(K, THETAS[i]); mx = abs(Kg).max()
        axs[0, i].imshow(Kg, cmap="RdBu_r", norm=TwoSlopeNorm(0, -mx, mx)); axs[0, i].set_title(f"{THETAS[i]:.0f}°")
        axs[1, i].imshow(r[i], cmap="Oranges", vmin=0, vmax=r.max())
        axs[1, i].contour(x.astype(float), levels=[0.5], colors=[T2], linewidths=0.5)
        axs[2, i].imshow(r[i] > TAU, cmap="Greys", vmin=0, vmax=1.5)
        for a in axs[:, i]:
            a.set_xticks([]); a.set_yticks([])
    axs[0, 0].set_ylabel("kernel 9×9", color=T1); axs[1, 0].set_ylabel("ReLU(G⋆x)\narco R = 9", color=T1)
    axs[2, 0].set_ylabel(f"> τ = {TAU}", color=T1)
    fig.suptitle("1 · El banco: 12 Gabor pares (λ = 6 px) cada 15°, y sus respuestas sobre un arco de radio 9. "
                 "Cada trozo del arco enciende el kernel de SU tangente: la orientación va girando a lo largo.", color=T1, fontsize=10)
    fig.savefig(IMG / "1-banco-gabor.png", dpi=105, bbox_inches="tight"); plt.close(fig)
    print("→", IMG / "1-banco-gabor.png")


def figura_radio() -> dict:
    """¿κ mediana sigue a 1/R? Arcos de radio 5–40 en 8 orientaciones, y rectas (R = ∞)."""
    _estilo()
    radios = [5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 22, 27, 33, 40]
    puntos = {"R": [], "ang": [], "kappa": [], "R_est": [], "que": []}
    for R in radios:
        for ang in range(0, 360, 45):
            c, kappa, mask, vs = detectar(arco(16, 16 - min(R, 10) + 6, R, ang, 20, 3))
            if not vs:
                continue
            v = vs[0]
            puntos["R"].append(R); puntos["ang"].append(ang); puntos["kappa"].append(v["kappa_med"])
            puntos["R_est"].append(57.3 / max(v["kappa_med"], 1e-3)); puntos["que"].append(v["que"])
    rectas_k = []
    for ang in range(0, 180, 15):
        _, _, _, vs = detectar(recta(16, 16, 20, ang, 3)); rectas_k.append(vs[0]["kappa_med"] if vs else np.nan)
    R = np.array(puntos["R"]); Re = np.array(puntos["R_est"]); kap = np.array(puntos["kappa"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    axs[0].plot([4, 45], [4, 45], color=T2, lw=1, ls="--", label="R estimado = R real")
    axs[0].scatter(R, Re, s=18, color=AZU, alpha=0.7, label="arcos de 20 px · 3 px · 8 orientaciones")
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("radio real (px)"); axs[0].set_ylabel("radio estimado 57,3 / κ (px)")
    axs[0].set_title("radio estimado contra real"); axs[0].legend(frameon=False, fontsize=8)
    axs[1].scatter(R, kap, s=18, color=AZU, alpha=0.7, label="arcos")
    axs[1].scatter([60] * len(rectas_k), rectas_k, s=18, color=NAR, alpha=0.8, label="rectas (R = ∞, puestas en 60)")
    axs[1].plot(np.array(radios), 57.3 / np.array(radios), color=T2, lw=1, ls="--", label="κ = 57,3 / R")
    axs[1].axhline(KAPPA_MIN, color=NAR, lw=0.8, ls=":"); axs[1].text(5, KAPPA_MIN + 0.15, f"κ_min = {KAPPA_MIN}", color=NAR, fontsize=8)
    axs[1].set_xscale("log"); axs[1].set_xlabel("radio real (px)"); axs[1].set_ylabel("κ mediana (°/px)"); axs[1].set_title("giro medido contra el teórico")
    axs[1].legend(frameon=False, fontsize=8)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("5 · ¿El giro medido sigue a 1/R? (arcos sintéticos de 3 px; sin entrenar nada)", color=T1, fontsize=10)
    fig.savefig(IMG / "5-radio-estimado.png", dpi=105, bbox_inches="tight"); plt.close(fig)
    print("→", IMG / "5-radio-estimado.png")
    err = np.abs(np.log(Re / R))
    por_R = {int(r): {"R_est_med": round(float(np.median(Re[R == r])), 1),
                      "curva": round(float(np.mean([q.startswith("curva") for q, rr in zip(puntos["que"], R) if rr == r])), 2)}
             for r in radios}
    return {"por_radio": por_R, "error_log_mediano": round(float(np.median(err)), 3),
            "kappa_rectas": [round(float(k), 2) for k in rectas_k]}


def figura_digitos() -> list[dict]:
    d = dict(np.load(exigir_dataset("uci-optdigits-orig-32px-r20261005") / "datos.npz"))
    img, y, part = d["imagenes"], d["etiquetas"].astype(int), d["particion"]
    tr = np.flatnonzero(part == "train")
    items = []
    for dig in (0, 1, 2, 3, 4, 5, 7, 8):
        i = tr[y[tr] == dig][0]
        items.append((f"dígito {dig} (train #{i})", (img[i] > 0).astype(np.uint8), {0: "curva", 1: "recta", 2: "curva", 3: "curva",
                                                                                 4: "esquina", 5: "esquina", 7: "esquina", 8: "curva"}[dig]))
    return figura_formas(items, "4-digitos.png", "4 · Dígitos reales (uci-optdigits 32×32), el mismo detector sin tocar. "
                         "«esperado» es lo que diría un humano del trazo mayor; aquí no hay etiqueta buena.")


def banco_rect_lin() -> dict:
    """Todo el banco de prueba de rect-lin (rectas cada 3°, grosor 2–14; arcos R 6–40, grosor 2/4/8; negativos)."""
    raiz = exigir_dataset("rect-lin-banco-r20261008")
    b = dict(np.load(raiz / "datos.npz"))
    nombres = ("continua", "punteada", "curva", "ruido", "puntos-sueltos", "mancha")
    tabla = {}
    for i in range(len(b["imagenes"])):
        _, _, _, vs = detectar(b["imagenes"][i])
        que = vs[0]["que"].split(" ")[0] if vs else "nada"
        t = nombres[int(b["tipo"][i])]
        if t == "continua":
            clave = f"recta · grosor {int(b['grosor'][i])}"
        elif t == "curva":
            clave = f"arco R {int(b['radio'][i])} · grosor {int(b['grosor'][i])}"
        elif t == "punteada":
            clave = f"punteada · sep {int(b['separacion'][i])}"
        else:
            clave = t
        tabla.setdefault(clave, {}); tabla[clave][que] = tabla[clave].get(que, 0) + 1
    return tabla


def tabla_banco_txt(tabla: dict) -> str:
    cols = ("recta", "curva", "esquina", "corto", "mancha", "nada")
    filas = ["| banco de rect-lin | n | " + " | ".join(cols) + " |", "|---|---:|" + "|".join(["---:"] * len(cols)) + "|"]

    def orden(k):
        if k.startswith("recta"): return (0, int(k.split()[-1]))
        if k.startswith("arco"): return (1, int(k.split()[2]), int(k.split()[-1]))
        if k.startswith("punteada"): return (2, int(k.split()[-1]))
        return (3, k)
    for k in sorted(tabla, key=orden):
        n = sum(tabla[k].values())
        filas.append(f"| {k} | {n} | " + " | ".join(f"{100 * tabla[k].get(c, 0) / n:.0f} %" if tabla[k].get(c, 0) else "·" for c in cols) + " |")
    return "\n".join(filas)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--banco", action="store_true", help="pasar también el banco entero de rect-lin")
    a = p.parse_args()
    torch.set_num_threads(2); IMG.mkdir(exist_ok=True)
    figura_banco()
    fs = formas()
    res = {"parametros": {"k": K, "lambda": LAM, "orientaciones": N_ORI, "d_giro": D_GIRO, "tau": TAU, "coh_min": COH_MIN,
                          "giro_recta": TURN_RECTA, "kappa_min": KAPPA_MIN}}
    res["rectas_y_curvas"] = figura_formas(fs[:14], "2-rectas-y-curvas.png",
                                           "2 · Rectas y curvas sintéticas (32×32). Palitos: la orientación local; κ: cuánto gira la orientación "
                                           "a lo largo de la tangente (gris: no medible).")
    res["otras_formas"] = figura_formas(fs[14:], "3-otras-formas.png",
                                        "3 · Otras formas: esquinas, S, cruce, punteada, mancha, puntos, ruido. Naranja en el texto: el "
                                        "veredicto no coincide con lo esperado.")
    res["digitos"] = figura_digitos()
    res["radio"] = figura_radio()
    for grupo in ("rectas_y_curvas", "otras_formas", "digitos"):
        n = len(res[grupo]); ok = sum(z["acierta"] for z in res[grupo])
        print(f"  {grupo:16s} {ok}/{n} coinciden con lo esperado")
        for z in res[grupo]:
            v = z["componentes"][0] if z["componentes"] else {"que": "nada"}
            det = (f"   c {v['coh_med']} Δθ {v['giro_total']:5} κ {v['kappa_med']:4} p90 {v['kappa_p90']:5} trozos {v['trozos_rectos']}r/{v['trozos_curvos']}c"
                   if "coh_med" in v else "") + f"  Emax {z['emax']}"
            print(f"    {'ok ' if z['acierta'] else 'NO '} {z['forma']:28s} → {v['que']:10s}" +
                  (f" R≈{v['radio']:5}" if "radio" in v else "        ") + f"  (esp. {z['esperado']:16s}){det}")
    print(f"  radio: error log mediano {res['radio']['error_log_mediano']} · κ rectas {res['radio']['kappa_rectas']}")
    (AQUI / "resultados.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if a.banco:
        tabla = banco_rect_lin(); txt = tabla_banco_txt(tabla)
        (AQUI / "resultados-banco.json").write_text(json.dumps(tabla, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        (AQUI / "resultados-banco.txt").write_text(txt + "\n", encoding="utf-8")
        print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
