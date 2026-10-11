#!/usr/bin/env python3
"""lazos — SONDEO (no es una vuelta del criterio): ¿se pueden incluir los lazos MUY CERRADOS de radio pequeño (los rizos
de los 2)? Pedido del dueño, 2026-10-11: «Veo curvas de radio pequeño perderse. Tal vez esperamos una curva demasiado
redonda. Checa si se pueden incluir estos lazos muy cerrados. Toma unos 2 como ejemplo.»

Dos cosas, medidas:

1. POR QUÉ EL DETECTOR DE LAZOS NO LOS VE. El giro κ se mide comparando la orientación a ±D_GIRO px a lo largo de la
   tangente, y una orientación (módulo 180°) sólo puede girar ±90° sin ambigüedad: κ máximo = 90/(2·4) = 11,25 °/px,
   o sea **radio mínimo ≈ 5,1 px**. Un rizo de un 2 (trazo de 4 px alrededor de un hueco de 1–4 px) tiene radio ~2–3.
   Medir el giro más cerca (D_GIRO 2 o 3) se prueba aquí: ayuda en sintéticos pequeños y NO en los rizos reales, y mete
   lazos falsos (los 1 sin lazo caen a la mitad), porque en un rizo grueso el campo de orientación es incoherente.

2. UNA SEÑAL DIRECTA: RAYOS. Un píxel de fondo es el CENTRO DE UN RIZO si, de 36 rayos que salen de él (cada 10°), casi
   todos chocan con tinta antes de 5 px. Los píxeles que lo cumplen se agrupan (vecindad 8) y cada grupo es un rizo:
   centro = su centroide; radio = mediana de la distancia al primer choque; cierre = (fracción de rayos que chocan −
   0,5)/0,5; orientación = la dirección media de los rayos que NO chocan (si hay alguno). No es morfología: es una medida
   geométrica sobre la tinta. RAYOS_MIN 0,97 y RMAX 5 se eligieron con este mismo sondeo (los huecos de los 2 y lazos
   sintéticos pequeños): ⚠ elegidos mirando, no calibrados aparte.

    python nn/rizos.py   → resultados/4-rizos-doses.png · 5-rizos-clases.png · rizos.json  (~4 min)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import label

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import detector as DT                                               # noqa: E402
import lazos as LZ                                                  # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

RES = AQUI / "resultados"
RAYOS_MIN, RMAX = 0.97, 5.0
ANG = np.deg2rad(np.arange(36) * 10 + 5); CA, SA = np.cos(ANG), np.sin(ANG)
PS = np.arange(0.5, RMAX + 0.01, 0.25)


def _rayos(x: np.ndarray, ys: np.ndarray, xs: np.ndarray):
    """Para los píxeles (ys, xs): qué rayos chocan con tinta antes de RMAX, y a qué distancia (NaN si no chocan)."""
    cx, cy = xs + 0.5, ys + 0.5
    px = cx[:, None, None] + CA[None, :, None] * PS[None, None, :]
    py = cy[:, None, None] + SA[None, :, None] * PS[None, None, :]
    ix, iy = np.floor(px).astype(int), np.floor(py).astype(int)
    ok = (ix >= 0) & (ix < x.shape[1]) & (iy >= 0) & (iy < x.shape[0])
    hit = np.zeros(ix.shape, bool); hit[ok] = x[iy[ok], ix[ok]]
    choca = hit.any(2)
    dist = np.where(choca, PS[np.argmax(hit, axis=2)], np.nan)
    return choca, dist


def rizos(x: np.ndarray, rayos_min: float = RAYOS_MIN) -> list[dict]:
    x = np.asarray(x) > 0
    ys, xs = np.nonzero(~x)
    if len(ys) == 0:
        return []
    choca, _ = _rayos(x, ys, xs)
    m = np.zeros(x.shape, bool); m[ys, xs] = choca.mean(1) >= rayos_min
    lab, n = label(m, structure=np.ones((3, 3)))
    out = []
    for z in range(1, n + 1):
        gy, gx = np.nonzero(lab == z)
        cx, cy = gx.mean() + 0.5, gy.mean() + 0.5
        ch, di = _rayos(x, np.array([cy - 0.5]), np.array([cx - 0.5]))
        frac = float(ch[0].mean())
        libres = ANG[~ch[0]]
        ori = float(np.degrees(np.arctan2(np.sin(libres).mean(), np.cos(libres).mean())) % 360) if len(libres) else None
        out.append({"centro": (float(cx), float(cy)), "radio": float(np.nanmedian(di[0])), "cierre": float(np.clip((frac - 0.5) / 0.5, 0, 1)),
                    "orientacion": ori, "pixeles": int(len(gy))})
    return out


def huecos(x: np.ndarray) -> list[tuple[float, float]]:
    """Centros de los huecos cerrados del dígito (sólo para MEDIR: la verdad de los rizos de los 2)."""
    f, n = label(~(x > 0)); borde = set(np.unique(np.concatenate([f[0], f[-1], f[:, 0], f[:, -1]])))
    return [(float(np.nonzero(f == z)[1].mean() + 0.5), float(np.nonzero(f == z)[0].mean() + 0.5))
            for z in range(1, n + 1) if z not in borde]


def votos_d(x, ds):
    c = DT.campo(x); acc = np.zeros(x.shape)
    for d in ds:
        kappa = DT.giro(c, d)
        ys, xs = np.nonzero(np.isfinite(kappa) & (np.abs(np.nan_to_num(kappa)) >= DT.KAPPA_MIN))
        if not len(ys):
            continue
        k = kappa[ys, xs]; th = np.deg2rad(c["theta"][ys, xs]); kx, ky = -k * np.sin(th), k * np.cos(th)
        n = np.hypot(kx, ky); R = 57.3 / np.abs(k)
        ix, iy = np.floor(xs + 0.5 + R * kx / n).astype(int), np.floor(ys + 0.5 + R * ky / n).astype(int)
        ok = (ix >= 0) & (ix < 32) & (iy >= 0) & (iy < 32); np.add.at(acc, (iy[ok], ix[ok]), 1.0)
    return acc


def main() -> int:
    t0 = time.time(); RES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "figure.facecolor": LZ.SUP, "axes.facecolor": LZ.SUP})
    cal = json.loads((RES / "metricas.json").read_text())["elegida"]; sv, vm = cal["sigma_v"], cal["v_min"]
    d = dict(np.load(exigir_dataset(LZ.DATASET) / "datos.npz"))
    va = np.flatnonzero(d["particion"] == "val")
    X, Y = (d["imagenes"][va] > 0).astype(np.uint8), d["etiquetas"][va].astype(int)
    dos = np.flatnonzero(Y == 2)
    dos_h = [(i, h) for i in dos for h in huecos(X[i])]
    rng = np.random.default_rng(303); peq = []
    while len(peq) < 200:                                                    # lazos sintéticos pequeños (R 2–4,5)
        R, g, cob = rng.uniform(2, 4.5), int(rng.integers(1, 4)), rng.uniform(0.6, 1.0); cx, cy = rng.uniform(R + 3, 29 - R, 2)
        xx = LZ.lazo(cx, cy, R, cob, rng.uniform(0, 360), g)
        if xx.sum() >= 6:
            peq.append({"x": xx, "centro": (cx, cy), "cierre": (cob - 0.5) / 0.5})
    neg = [f for f in LZ.sinteticos(300, 202) if f["tipo"] == "negativo"]
    out = {"dos_val": int(len(dos)), "dos_con_hueco": int(len({i for i, _ in dos_h})), "huecos": len(dos_h),
           "radio_minimo_por_giro_px": round(57.3 / (90 / (2 * DT.D_GIRO)), 1)}

    # 1 · el detector de lazos midiendo el giro más cerca
    orig = DT.votos; res_d = {}
    for ds in ((4,), (3, 4), (2, 4)):
        DT.votos = lambda z, ds=ds: votos_d(z, ds)
        h = sum(any(np.hypot(l["centro"][0] - c[0], l["centro"][1] - c[1]) <= 3 for l in DT.lazos(X[i], sv, vm)) for i, c in dos_h)
        enc = np.mean([any(np.hypot(l["centro"][0] - f["centro"][0], l["centro"][1] - f["centro"][1]) <= 3
                           for l in DT.lazos(f["x"], sv, vm)) for f in peq])
        sin1 = np.mean([not DT.lazos(X[i], sv, vm) for i in np.flatnonzero(Y == 1)])
        res_d[str(ds)] = {"huecos_de_2": f"{h}/{len(dos_h)}", "sinteticos_pequenos_pc": round(100 * float(enc), 1),
                          "uno_sin_lazo_pc": round(100 * float(sin1), 1)}
        print(f"giro a ±{ds}: {res_d[str(ds)]}  ({time.time() - t0:.0f} s)", flush=True)
    DT.votos = orig
    out["giro_mas_cerca"] = res_d

    # 2 · los rizos por rayos, con tres umbrales
    res_r = {}
    for rmin in (0.94, 0.97, 1.0):
        h = sum(any(np.hypot(r["centro"][0] - c[0], r["centro"][1] - c[1]) <= 2 for r in rizos(X[i], rmin)) for i, c in dos_h)
        ec = np.mean([any(np.hypot(r["centro"][0] - f["centro"][0], r["centro"][1] - f["centro"][1]) <= 2 for r in rizos(f["x"], rmin))
                      for f in peq if f["cierre"] >= 0.8])
        fp = np.mean([bool(rizos(f["x"], rmin)) for f in neg])
        clase = {str(k): round(100 * float(np.mean([bool(rizos(X[i], rmin)) for i in np.flatnonzero(Y == k)])), 1) for k in range(10)}
        res_r[str(rmin)] = {"huecos_de_2": f"{h}/{len(dos_h)}", "sinteticos_pequenos_muy_cerrados_pc": round(100 * float(ec), 1),
                            "falsos_negativos_sinteticos_pc": round(100 * float(fp), 1), "con_rizo_por_clase_pc": clase}
        print(f"rayos ≥ {rmin}: {res_r[str(rmin)]}  ({time.time() - t0:.0f} s)", flush=True)
    out["rayos"] = res_r; out["elegido"] = {"rayos_min": RAYOS_MIN, "rmax": RMAX}
    (RES / "rizos.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # ── 4 · todos los 2 de val: lazos (detector de lazos) + rizos (rayos), con el dibujo pedido ──
    cols = 14; filas = int(np.ceil(len(dos) / cols))
    fig, axs = plt.subplots(filas, cols, figsize=(cols * 1.15, filas * 1.3 + 0.9))
    for k, ax in enumerate(axs.flat):
        if k >= len(dos):
            ax.axis("off"); continue
        i = dos[k]; L = DT.lazos(X[i], sv, vm); R = rizos(X[i])
        LZ.dibujar(ax, X[i], L + R, "", LZ.T1)
        for r in R:
            ax.plot(r["centro"][0] - 0.5, r["centro"][1] - 0.5, "o", mfc="none", mec=LZ.AZU, ms=11, mew=1.0)
        ax.set_title(f"{len(L)} lazo · {len(R)} rizo" + (" · hueco" if huecos(X[i]) else ""), fontsize=7.5,
                     color=LZ.NAR if R else LZ.T2, pad=1.5)
    fig.suptitle(f"lazos · 4 · TODOS los 2 de val ({len(dos)}) · gris = dígito · cruz azul = centro · arco = el lazo (hueco "
                 f"hacia la abertura) · círculo azul = RIZO (lazo muy cerrado, por rayos ≥ {RAYOS_MIN}, ≤ {RMAX:g} px)\n"
                 "título: lazos del detector de lazos · rizos · «hueco» = el dígito tiene un hueco cerrado (la verdad medible)",
                 fontsize=10, color=LZ.T1)
    fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(RES / "4-rizos-doses.png", dpi=90); plt.close(fig)

    # ── 5 · % con rizo por clase, para los tres umbrales ──
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for n, (rmin, col) in enumerate(zip(("0.94", "0.97", "1.0"), ("#b9b8b4", LZ.NAR, LZ.AZU))):
        v = [res_r[rmin]["con_rizo_por_clase_pc"][str(k)] for k in range(10)]
        ax.bar(np.arange(10) + (n - 1) * 0.27, v, 0.27, color=col, label=f"rayos ≥ {rmin}")
    ax.set_xticks(range(10)); ax.set_ylim(0, 105); ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("lazos · 5 · % de dígitos de val con algún RIZO, por clase (8: lazo bajo pequeño; 6 y 9: su lazo; 0: lazo "
                 "grande, no es rizo; 1 y 7: no deberían)", loc="left", fontsize=9.5)
    fig.tight_layout(); fig.savefig(RES / "5-rizos-clases.png", dpi=100); plt.close(fig)
    print(f"→ resultados/4-rizos-doses.png · 5-rizos-clases.png · rizos.json · total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
