#!/usr/bin/env python3
"""dig-delg — el dígito DELGADO a partir de las respuestas de Gabor (ver REGLAS.md e instrucciones/02-criterio.md).

Brazos (todos sobre el dígito binario 32×32, sin morfología salvo la referencia D):
    A        valle firmado del Gabor IMPAR de la figura 19 (λ 10, 9×9, 8 orientaciones): mínimo de |o| a lo largo de la
             normal, con la polaridad de un TRAZO a los lados (no la de un hueco entre trazos)
    B8, B12  cresta del Gabor PAR del ancho del trazo (λ_p 8 y 12, 9×9, 8 orientaciones): máximo a lo largo de la normal
    C        fase local: par e impar de la misma escala (λ 10, 9×9); cresta del par donde la fase |o|/e es pequeña
    D        REFERENCIA, no candidato: skimage.morphology.skeletonize (morfología)

Los umbrales de cada brazo (TAU_A, TAU_B, FASE_C) se fijaron ANTES de mirar resultados y no se tocan aquí; si se
cambian, es otra vuelta y se dice.

    python nn/delgado.py   → resultados/*.png · resultados/metricas.json  (~20 s)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, label, map_coordinates

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.colors import ListedColormap                        # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI.parent))
from expcnn import exigir_dataset                                   # noqa: E402

RES = AQUI / "resultados"
K, N_ORI = 9, 8
THETAS = np.arange(N_ORI) * 180.0 / N_ORI
LAM_IMPAR = 10.0             # el de la figura 19
TAU_A = 0.2                  # fuerza mínima del valle (respuesta normalizada por la de un escalón ideal)
TAU_B = 0.3                  # cresta: fracción del máximo de la imagen
FASE_C = 30.0                # grados: |fase| máxima para ser centro de trazo
H = 1.0                      # px: a qué distancia, a lo largo de la normal, se comparan los vecinos
UMBRALES = {"delgadez": 85.0, "dentro": 95.0, "cobertura": 95.0, "topologia": 90.0}
SUP, T1, T2, NAR, AZU, VERDE = "#fcfcfb", "#0b0b0b", "#52514e", "#c2410c", "#2a78d6", "#1b7f3b"


# ─── los bancos: COPIADOS de docs/bocetos/2026-10-09-curvas-gabor (curvas.kernel_gabor, bordes_gabor.kernel_impar) ──
def _coords(k, ang):
    r = (k - 1) / 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    t = np.deg2rad(ang)
    return xx * np.cos(t) + yy * np.sin(t), -xx * np.sin(t) + yy * np.cos(t)


def kernel_par(k: int, ang: float, lam: float) -> np.ndarray:
    u, v = _coords(k, ang)
    g = np.exp(-(u ** 2) / (2 * (k / 3) ** 2) - (v ** 2) / (2 * (lam / 2) ** 2)) * np.cos(2 * np.pi * v / lam)
    g -= g.mean()
    return (g / np.linalg.norm(g)).astype(np.float32)


def kernel_impar(k: int, ang: float, lam: float) -> np.ndarray:
    u, v = _coords(k, ang)
    g = np.exp(-(u ** 2) / (2 * (k / 3) ** 2) - (v ** 2) / (2 * (lam / 2) ** 2)) * np.sin(2 * np.pi * v / lam)
    return (g / np.linalg.norm(g)).astype(np.float32)


def banco(fn, lam) -> torch.Tensor:
    return torch.from_numpy(np.stack([fn(K, a, lam) for a in THETAS])[:, None])


@torch.no_grad()
def responder(x: np.ndarray, b: torch.Tensor) -> np.ndarray:
    return F.conv2d(torch.from_numpy(x.astype(np.float32))[None, None], b, padding=K // 2)[0].numpy()


IMPAR = banco(kernel_impar, LAM_IMPAR)
_paso = np.zeros((64, 64), np.float32); _paso[:, 32:] = 1
ESCALA_IMPAR = float(np.abs(responder(_paso, IMPAR)).max())       # la normalización de la figura 19
NX, NY = -np.sin(np.deg2rad(THETAS)), np.cos(np.deg2rad(THETAS))   # la normal de cada orientación (x, y)


def _vecinos(m: np.ndarray, i: int):
    """m (H,W) muestreada en p − H·n_i y p + H·n_i (bilineal)."""
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]].astype(float)
    a = map_coordinates(m, [yy - H * NY[i], xx - H * NX[i]], order=1, mode="constant")
    b = map_coordinates(m, [yy + H * NY[i], xx + H * NX[i]], order=1, mode="constant")
    return a, b


def _polaridad() -> float:
    """Signo de la respuesta impar ANTES del centro de un trazo (a lo largo de su normal). Se mide, no se supone."""
    x = np.zeros((32, 32), np.float32); x[13:19, :] = 1                # trazo horizontal: orientación 0, normal ↓
    o = responder(x, IMPAR)[0]
    return float(np.sign(o[13, 16]))                                   # fila 13: primer borde del trazo, mirando ↓


POL = _polaridad()


# ─── los brazos ───────────────────────────────────────────────────────────────────────────────────────────────────
def brazo_A(x: np.ndarray) -> np.ndarray:
    o = responder(x, IMPAR) / ESCALA_IMPAR
    mejor = np.zeros(x.shape)
    for i in range(N_ORI):
        a, b = _vecinos(o[i], i)
        fuerza = (POL * a - POL * b) / 2                              # trazo: +POL antes, −POL después
        absa, absb = _vecinos(np.abs(o[i]), i)
        es = (POL * a > 0) & (POL * b < 0) & (np.abs(o[i]) <= absa) & (np.abs(o[i]) <= absb)
        mejor = np.maximum(mejor, np.where(es, fuerza, 0.0))
    return mejor > TAU_A


def _cresta(R: np.ndarray, extra=None) -> np.ndarray:
    i_ = R.argmax(0); Rm = R.max(0)
    out = np.zeros(R.shape[1:], bool)
    for i in range(N_ORI):
        a, b = _vecinos(R[i], i)
        out |= (i_ == i) & (R[i] >= a) & (R[i] >= b)
    out &= Rm > TAU_B * Rm.max()
    if extra is not None:
        out &= extra
    return out


def brazo_B(x: np.ndarray, lam: float) -> np.ndarray:
    return _cresta(np.maximum(responder(x, banco(kernel_par, lam)), 0))


PAR10 = banco(kernel_par, LAM_IMPAR)


def brazo_C(x: np.ndarray) -> np.ndarray:
    e = responder(x, PAR10); o = responder(x, IMPAR)
    E = np.sqrt(e ** 2 + o ** 2); i_ = E.argmax(0)
    ee = np.take_along_axis(e, i_[None], 0)[0]; oo = np.take_along_axis(o, i_[None], 0)[0]
    fase = np.degrees(np.arctan2(np.abs(oo), ee))                      # 0 = par puro positivo = centro de trazo
    return _cresta(np.maximum(e, 0), extra=(ee > 0) & (fase < FASE_C))


def brazo_D(x: np.ndarray) -> np.ndarray:
    from skimage.morphology import skeletonize
    return skeletonize(x.astype(bool))


BRAZOS = {"A · valle firmado (impar λ 10)": brazo_A, "B · cresta par λ 8": lambda x: brazo_B(x, 8.0),
          "B · cresta par λ 12": lambda x: brazo_B(x, 12.0), "C · fase local (λ 10)": brazo_C,
          "D · esqueleto morfológico (REFERENCIA)": brazo_D}


ARCHIVO = ("A", "B8", "B12", "C", "D")                              # resultados/2-<esto>.png, en el orden de BRAZOS


# ─── las métricas del criterio (medir con morfología no es construir con ella) ────────────────────────────────────
def lazos_y_trozos(m: np.ndarray) -> tuple[int, int]:
    trozos = label(m, structure=np.ones((3, 3)))[1]
    fondo, n = label(~m)                                               # fondo 4-conexo, frente 8-conexo
    borde = set(np.unique(np.concatenate([fondo[0], fondo[-1], fondo[:, 0], fondo[:, -1]])))
    return sum(1 for z in range(1, n + 1) if z not in borde), trozos


def medir(x: np.ndarray, d: np.ndarray) -> dict:
    tinta = x > 0
    if d.sum() == 0:
        return {"delgadez": 0.0, "dentro": 0.0, "cobertura": 0.0, "topologia": False, "px": 0}
    vec = F.conv2d(torch.from_numpy(d.astype(np.float32))[None, None], torch.ones(1, 1, 3, 3), padding=1)[0, 0].numpy()
    dist = distance_transform_edt(~d)
    return {"delgadez": float(100 * (vec[d] <= 3).mean()), "dentro": float(100 * tinta[d].mean()),
            "cobertura": float(100 * (dist[tinta] <= 3).mean()),
            "topologia": lazos_y_trozos(d) == lazos_y_trozos(tinta), "px": int(d.sum())}


# ─── figuras ──────────────────────────────────────────────────────────────────────────────────────────────────────
def _estilo():
    plt.rcParams.update({"font.size": 8, "figure.facecolor": SUP, "axes.facecolor": SUP, "axes.titlecolor": T1})


def _pintar(ax, entrada, d, titulo, color=T1):
    ax.imshow(entrada, cmap="Greys", vmin=0, vmax=2.2)
    ax.imshow(np.where(d, 1.0, np.nan), cmap=ListedColormap([NAR]), vmin=0, vmax=1)
    ax.set_title(titulo, fontsize=8, pad=2, color=color); ax.set_xticks([]); ax.set_yticks([])


def main() -> int:
    torch.set_num_threads(2); RES.mkdir(exist_ok=True)
    sel = json.loads((AQUI / "entradas" / "digitos-64.json").read_text())
    dd = dict(np.load(exigir_dataset(sel["dataset"]) / "datos.npz"))
    idx = np.array(sel["indices"]); X = (dd["imagenes"][idx] > 0).astype(np.uint8); Y = dd["etiquetas"][idx].astype(int)
    entrada = [np.clip(np.abs(responder(x, IMPAR)).max(0) / ESCALA_IMPAR, 0, 1) for x in X]   # la de la figura 19
    print(f"{len(X)} dígitos · polaridad del trazo en el impar: {POL:+.0f}")
    salidas = {n: [f(x) for x in X] for n, f in BRAZOS.items()}
    met = {}
    for n, ds in salidas.items():
        m = [medir(x, d) for x, d in zip(X, ds)]
        r = {"delgadez": round(float(np.mean([z["delgadez"] for z in m])), 1),
             "dentro": round(float(np.mean([z["dentro"] for z in m])), 1),
             "cobertura": round(float(np.mean([z["cobertura"] for z in m])), 1),
             "topologia": round(float(100 * np.mean([z["topologia"] for z in m])), 1),
             "px_medio": round(float(np.mean([z["px"] for z in m])), 1), "por_digito": m}
        r["pasa"] = {k: bool(r[k] >= u) for k, u in UMBRALES.items()}
        r["pasa_los_cuatro"] = all(r["pasa"].values())
        met[n] = r
        print(f"  {n:40s} delgadez {r['delgadez']:5.1f} · dentro {r['dentro']:5.1f} · cobertura {r['cobertura']:5.1f}"
              f" · topología {r['topologia']:5.1f}  → {'PASA' if r['pasa_los_cuatro'] else 'no pasa'} {r['pasa']}")
    (RES / "metricas.json").write_text(json.dumps({"umbrales": UMBRALES, "parametros": {
        "K": K, "orientaciones": N_ORI, "lam_impar": LAM_IMPAR, "tau_A": TAU_A, "tau_B": TAU_B, "fase_C": FASE_C, "H": H},
        "indices": idx.tolist(), "brazos": met}, indent=1, ensure_ascii=False, default=bool) + "\n", encoding="utf-8")

    _estilo()
    # 1 · comparación: 16 dígitos, todos los brazos
    muestra = list(range(0, len(X), 4))[:16]
    cols = ["dígito", "entrada (fig. 19)"] + list(BRAZOS)
    fig, axs = plt.subplots(len(muestra), len(cols), figsize=(len(cols) * 1.6, len(muestra) * 1.6 + 0.8))
    for r_, k in enumerate(muestra):
        axs[r_, 0].imshow(X[k], cmap="Greys", vmin=0, vmax=1.6); axs[r_, 0].set_ylabel(f"{Y[k]}  (#{idx[k]})", fontsize=8.5)
        axs[r_, 1].imshow(entrada[k], cmap="gray_r", vmin=0, vmax=1)
        for c_, n in enumerate(BRAZOS):
            m = met[n]["por_digito"][k]
            ok = m["topologia"]
            _pintar(axs[r_, 2 + c_], X[k] * 0.6, salidas[n][k], "lazos ok" if ok else "lazos ✗", VERDE if ok else NAR)
        for c_ in range(len(cols)):
            axs[r_, c_].set_xticks([]); axs[r_, c_].set_yticks([])
            if r_ == 0:
                axs[r_, c_].text(0.5, 1.35, cols[c_].replace(" (", "\n("), transform=axs[r_, c_].transAxes, ha="center",
                                 fontsize=8.5, color=T1)
    fig.suptitle("dig-delg · 1 · El dígito DELGADO (naranja) sobre el original (gris), con cada alternativa · 16 de los 64",
                 fontsize=11, color=T1, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(RES / "1-comparacion.png", dpi=100); plt.close(fig)
    # 2 · cada brazo sobre los 64
    for j, n in enumerate(BRAZOS):
        fig, axs = plt.subplots(8, 8, figsize=(12, 13))
        for k, ax in enumerate(axs.flat):
            m = met[n]["por_digito"][k]
            _pintar(ax, X[k] * 0.6, salidas[n][k], f"{Y[k]} · {'ok' if m['topologia'] else 'lazos ✗'}",
                    VERDE if m["topologia"] else NAR)
        r = met[n]
        fig.suptitle(f"dig-delg · 2 · {n} · los 64 dígitos\ndelgadez {r['delgadez']} % · dentro {r['dentro']} % · "
                     f"cobertura {r['cobertura']} % · topología {r['topologia']} %  → "
                     f"{'PASA los cuatro' if r['pasa_los_cuatro'] else 'no pasa los cuatro'}", fontsize=11, color=T1)
        fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(RES / f"2-{ARCHIVO[j]}.png", dpi=95)
        plt.close(fig)
    # 3 · métricas
    fig, ax = plt.subplots(figsize=(12, 4.6))
    nombres = list(BRAZOS); w = 0.2; x_ = np.arange(len(nombres))
    for k, (clave, col) in enumerate(zip(UMBRALES, (AZU, VERDE, "#8a6bd1", NAR))):
        v = [met[n][clave] for n in nombres]
        ax.bar(x_ + (k - 1.5) * w, v, w, color=col, label=f"{clave} (umbral {UMBRALES[clave]:g} %)")
        for i, vv in enumerate(v):
            ax.text(i + (k - 1.5) * w, vv + 1, f"{vv:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(x_); ax.set_xticklabels([n.replace(" (", "\n(") for n in nombres], fontsize=8.5)
    ax.set_ylim(0, 112); ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("dig-delg · 3 · las cuatro métricas del criterio (media sobre los 64; topología = % de dígitos con los "
                 "mismos lazos y trozos)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(RES / "3-metricas.png", dpi=105); plt.close(fig)
    print("→ resultados/1-comparacion.png · 2-*.png · 3-metricas.png · metricas.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
