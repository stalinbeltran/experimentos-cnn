#!/usr/bin/env python3
"""TANTEO (pedido del dueño, 2026-10-09): un clasificador de dígitos con el detector de CURVAS, contra el de RECTAS, y
los dos juntos. Mismo protocolo que `rect-bor/nn/prueba_digitos.py`: compositor LINEAL (Adam, 300 épocas a lote
completo, lr 1e-2, L2 1e-3), 180 train / 1617 val (`uci-optdigits-orig-32px-r20261005`), 3 semillas que sólo cambian la
inicialización del compositor; a ciegas, los 3823 de otros escritores. No hay criterio propio: es un tanteo.

Características, todas FIJAS (nada aprendido salvo el compositor):
    RECTAS   la variante C del tanteo de rect-bor, COPIADA: 4 Gabor 9×9, mapa → integrado 15 px a lo largo de su
             orientación → max en celdas → 8×8, en 2 escalas                                             512
    CURVAS   del detector de este boceto (curvas.py), tres mapas → max en celdas 8×8:
             · recto   (|κ| < 1 °/px, medible)   · curvo (|κ|, tope 12)   · golpe (trazo con coherencia baja: esquina,
               cruce, mancha)                                                                             192
    COMBINADO  las dos concatenadas                                                                       704

PREDICCIÓN (escrita antes de correrlo): rectas ≈ 0,955 (lo medido en rect-bor); curvas solas por debajo de 0,93,
porque en los dígitos la mitad del trazo sale «no medible» por el grosor (figura 4 del boceto); el combinado sube
poco o nada sobre rectas (≤ 0,96).

    python digitos.py   → resultados-digitos.json · imagenes/6-fallos-{curvas,rectas,combinado}.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parents[2]))
import curvas as C                                                  # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

IMG = AQUI / "imagenes"
DIGITOS = "uci-optdigits-orig-32px-r20261005"
SEMILLAS = (0, 1, 2)
EPOCAS, LR, L2 = 300, 1e-2, 1e-3
L_INT, KAPPA_TOPE = 15, 12.0
SUP, T1, T2, NAR, AZU = "#fcfcfb", "#0b0b0b", "#52514e", "#eb6834", "#2a78d6"

# ─── RECTAS: copia literal de rect-bor/nn/prueba_digitos.py (variante C) ─────────────────────────────────────────
GABOR4 = torch.from_numpy(np.stack([C.kernel_gabor(9, a) for a in (0, 45, 90, 135)])[:, None])


def linea(ang: float, largo: int = L_INT) -> np.ndarray:
    r = largo // 2; yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    t = np.deg2rad(ang); u = xx * np.cos(t) + yy * np.sin(t); v = -xx * np.sin(t) + yy * np.cos(t)
    k = ((np.abs(v) <= 0.5) & (np.abs(u) <= r)).astype(np.float32)
    return k / k.sum()


INTEG = torch.from_numpy(np.stack([linea(a) for a in (0, 45, 90, 135)])[:, None])


@torch.no_grad()
def caract_rectas(x: np.ndarray, escalas: int = 2) -> np.ndarray:
    xt = torch.from_numpy(x.astype(np.float32)[:, None])
    salidas = []
    for e in range(escalas):
        xe = xt if e == 0 else F.avg_pool2d(xt, 2 ** e)
        m = F.relu(F.conv2d(xe, GABOR4, padding=4))
        m = F.conv2d(m, INTEG, padding=L_INT // 2, groups=4)
        salidas.append(F.adaptive_max_pool2d(m, 8).flatten(1))
    return torch.cat(salidas, 1).numpy()


# ─── CURVAS: los mapas del detector de curvas.py ──────────────────────────────────────────────────────────────────
def mapas_curvas(x: np.ndarray) -> np.ndarray:
    """(32,32) → (3,32,32): recto · curvo (|κ|) · golpe."""
    c = C.campo(x); kappa, mask, _ = C.giro(c)
    medible = mask & np.isfinite(kappa)
    recto = (medible & (np.abs(kappa) < C.KAPPA_MIN)).astype(np.float32)
    curvo = np.where(medible, np.minimum(np.abs(np.nan_to_num(kappa)), KAPPA_TOPE), 0.0).astype(np.float32)
    golpe = (mask & (c["coh"] < C.COH_MIN)).astype(np.float32)
    return np.stack([recto, curvo, golpe])


@torch.no_grad()
def caract_curvas(x: np.ndarray) -> np.ndarray:
    m = torch.from_numpy(np.stack([mapas_curvas(xi) for xi in x]))
    return F.adaptive_max_pool2d(m, 8).flatten(1).numpy()


# ─── el compositor, copiado del tanteo ────────────────────────────────────────────────────────────────────────────
def compositor(xtr, ytr, sem):
    torch.manual_seed(sem)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    W = torch.nn.Linear(xtr.shape[1], 10); opt = torch.optim.Adam(W.parameters(), lr=LR, weight_decay=L2)
    xt, yt = torch.from_numpy((xtr - mu) / sd).float(), torch.from_numpy(ytr).long()
    for _ in range(EPOCAS):
        opt.zero_grad(); F.cross_entropy(W(xt), yt).backward(); opt.step()

    @torch.no_grad()
    def predecir(x):
        return W(torch.from_numpy((x - mu) / sd).float()).argmax(1).numpy()
    return predecir


# ─── la figura de los fallos ──────────────────────────────────────────────────────────────────────────────────────
def figura_fallos(nombre: str, img, y, pred, idx_val, acc, n_max=60):
    fallos = idx_val[pred != y[idx_val]]
    cols = 12; filas = int(np.ceil(min(len(fallos), n_max) / cols))
    plt.rcParams.update({"font.size": 8, "figure.facecolor": SUP, "axes.facecolor": SUP})
    fig, axs = plt.subplots(max(filas, 1), cols, figsize=(cols * 1.05, max(filas, 1) * 1.25 + 0.6))
    axs = np.atleast_2d(axs)
    for k, ax in enumerate(axs.flat):
        ax.axis("off")
        if k < min(len(fallos), n_max):
            i = fallos[k]; ax.imshow(img[i], cmap="gray_r", vmin=0, vmax=1)
            ax.set_title(f"{y[i]}→{pred[np.flatnonzero(idx_val == i)[0]]}", color=NAR, fontsize=8, pad=2)
    fig.suptitle(f"6 · Dígitos de val (1617) que FALLA el compositor «{nombre}» (semilla 0): {len(fallos)} fallos, "
                 f"acierto {acc:.4f}" + (f"; se enseñan los {n_max} primeros" if len(fallos) > n_max else "") +
                 ".  «real→predicho»", color=T1, fontsize=10)
    ruta = IMG / f"6-fallos-{nombre}.png"
    fig.savefig(ruta, dpi=105, bbox_inches="tight"); plt.close(fig); print("→", ruta)
    return fallos


def main() -> int:
    torch.set_num_threads(2); IMG.mkdir(exist_ok=True)
    d = dict(np.load(exigir_dataset(DIGITOS) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va, ex = np.flatnonzero(part == "train"), np.flatnonzero(part == "val"), np.flatnonzero(part == "extra")
    t0 = time.time()
    Xr = caract_rectas(img); Xc = caract_curvas(img)
    print(f"características: rectas {Xr.shape[1]} · curvas {Xc.shape[1]} ({time.time() - t0:.0f} s)")
    variantes = {"curvas": Xc, "rectas": Xr, "combinado": np.concatenate([Xr, Xc], 1)}
    out, pred0 = {}, {}
    for nombre, X in variantes.items():
        accs, por_digito = [], np.zeros((len(SEMILLAS), 10))
        for s_i, s in enumerate(SEMILLAS):
            predecir = compositor(X[tr], y[tr], s)
            pv, pe = predecir(X[va]), predecir(X[ex])
            accs.append([float((pv == y[va]).mean()), float((pe == y[ex]).mean())])
            por_digito[s_i] = [float((pv[y[va] == k] == k).mean()) for k in range(10)]
            if s == 0:
                pred0[nombre] = pv
        accs = np.array(accs)
        out[nombre] = {"caracteristicas": int(X.shape[1]), "val_1617": round(float(accs[:, 0].mean()), 4),
                       "val_rango": [round(float(accs[:, 0].min()), 4), round(float(accs[:, 0].max()), 4)],
                       "ciega_3823": round(float(accs[:, 1].mean()), 4),
                       "por_digito_val": [round(float(v), 3) for v in por_digito.mean(0)]}
        print(f"  {nombre:10s} {X.shape[1]:4d} caract. · val {out[nombre]['val_1617']:.4f} {out[nombre]['val_rango']} · "
              f"a ciegas {out[nombre]['ciega_3823']:.4f} · por dígito {out[nombre]['por_digito_val']}", flush=True)
    fallos = {n: set(figura_fallos(n, img, y, pred0[n], va, float((pred0[n] == y[va]).mean())).tolist()) for n in variantes}
    out["solape_semilla_0"] = {
        "falla_solo_curvas": len(fallos["curvas"] - fallos["rectas"]), "falla_solo_rectas": len(fallos["rectas"] - fallos["curvas"]),
        "fallan_los_dos": len(fallos["curvas"] & fallos["rectas"]),
        "rectas_falla_y_combinado_acierta": len(fallos["rectas"] - fallos["combinado"]),
        "combinado_falla_y_rectas_acierta": len(fallos["combinado"] - fallos["rectas"])}
    # confusiones más frecuentes (semilla 0)
    for n in variantes:
        conf = {}
        for i in va[pred0[n] != y[va]]:
            k = f"{y[i]}→{pred0[n][np.flatnonzero(va == i)[0]]}"; conf[k] = conf.get(k, 0) + 1
        out[n]["confusiones_semilla_0"] = dict(sorted(conf.items(), key=lambda z: -z[1])[:8])
    print("solape:", out["solape_semilla_0"])
    (AQUI / "resultados-digitos.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
