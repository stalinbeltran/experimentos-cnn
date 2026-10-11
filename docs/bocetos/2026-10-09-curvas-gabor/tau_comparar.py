#!/usr/bin/env python3
"""El detector DELGADO con TAU 0,3 · 0,6 · 0,9, comparado sobre los dígitos (pedido del dueño, 2026-10-10).

*«Probemos 0,6 y 0,9, muéstrame en gráficos comparativos con el 0,3 y con varios dígitos, todos si es posible.»*

TAU es el umbral de energía para «aquí hay trazo»; con 0,3 la máscara se sale de la tinta (`mascara.py`). Todo lo
demás, la calibración de `delgadas.py` sin tocar, y el compositor de siempre (3 semillas, 180 / 1617 / 3823 a ciegas).

    24-tau-1-estadisticas.png      acierto, desborde, por dígito, por grosor, errores y desplazamientos — TODOS los dígitos
    24-tau-2-detector-tau03.png    los mismos 100 dígitos (10 por clase, val) con lo que ve el detector, uno por TAU
    24-tau-2-detector-tau06.png
    24-tau-2-detector-tau09.png
    24-tau-3-discrepan.png         TODOS los dígitos de val que los tres TAU no leen igual (semilla 0), lado a lado

    python tau_comparar.py   → esas figuras · resultados-tau.json  (~4 min)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parents[2]))
import curvas as C                                                  # noqa: E402
import digitos as D                                                 # noqa: E402
import digitos_bordes as DB                                         # noqa: E402
import bordes_gabor as BG                                           # noqa: E402
import delgadas as DL                                               # noqa: E402
import digitos_delgadas as DD                                       # noqa: E402
from gemelos_5 import mover                                         # noqa: E402
from curvas import plt                                              # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

TAUS = (0.3, 0.6, 0.9)
COL = {0.3: "#8a8986", 0.6: "#c2410c", 0.9: "#2a78d6"}
VERDE, NAR = "#1b7f3b", "#c2410c"
DS = list(range(-4, 5))


def contorno(ax, x):
    """El borde de la tinta, en negro: lo que queda FUERA de esta línea es desborde de la máscara."""
    ax.contour(x.astype(float), levels=[0.5], colors=[C.T1], linewidths=0.7)


def main() -> int:
    torch.set_num_threads(2); t0 = time.time()
    cal = json.loads((AQUI / "resultados-delgadas.json").read_text())["elegida"]
    d = dict(np.load(exigir_dataset(D.DIGITOS) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va, ex = (np.flatnonzero(part == k) for k in ("train", "val", "extra"))
    anc = np.array([DD.grosor(x) for x in img])
    muestra = np.concatenate([va[y[va] == k][:10] for k in range(10)])         # 10 por clase, los primeros de val
    out, pv0, pe0 = {"calibracion_base": cal, "taus": {}}, {}, {}
    for tau in TAUS:
        DL.configurar(**{**cal, "tau": tau})
        X = DB.caract(img, borde=DD.tal_cual)["bordes"]
        v, c = [], []
        for sem in D.SEMILLAS:
            pr = DB.entrenar(X[tr], y[tr], sem); pv, pe = pr(X[va]), pr(X[ex])
            v.append(float((pv == y[va]).mean())); c.append(float((pe == y[ex]).mean()))
            if sem == 0:
                pred0, pv0[tau], pe0[tau] = pr, pv, pe
        fuera = []
        for i in va:
            x = img[i] > 0; k, m, _ = C.giro(C.campo(x.astype(np.float32)))
            fuera.append((m & ~x).sum() / max(m.sum(), 1))
        acc_d = []
        for k in DS:
            xs = np.stack([mover(img[i], 0, k) for i in va])
            acc_d.append(round(float((pred0(DB.caract(xs, borde=DD.tal_cual)["bordes"]) == y[va]).mean()), 4))
        r = {"val": round(float(np.mean(v)), 4), "val_rango": [round(min(v), 4), round(max(v), 4)],
             "ciega": round(float(np.mean(c)), 4), "ciega_rango": [round(min(c), 4), round(max(c), 4)],
             "mascara_fuera_pc": round(100 * float(np.mean(fuera)), 1),
             "por_digito_val": [round(float((pv0[tau][y[va] == k] == k).mean()), 3) for k in range(10)],
             "por_grosor": {n: {"val": round(float((pv0[tau][(anc[va] >= lo) & (anc[va] < hi)] ==
                                                     y[va][(anc[va] >= lo) & (anc[va] < hi)]).mean()), 4),
                                "ciega": round(float((pe0[tau][(anc[ex] >= lo) & (anc[ex] < hi)] ==
                                                       y[ex][(anc[ex] >= lo) & (anc[ex] < hi)]).mean()), 4)}
                            for n, lo, hi in DD.GRUPOS},
             "fallos_val_semilla_0": int((pv0[tau] != y[va]).sum()),
             "fallos_val_sin_muy_gruesos": int(((pv0[tau] != y[va]) & (anc[va] < DD.MUY_GRUESO)).sum()),
             "desplazamiento_horizontal": acc_d}
        out["taus"][f"{tau:g}"] = r
        print(f"TAU {tau}: val {r['val']} · a ciegas {r['ciega']} · fuera {r['mascara_fuera_pc']} % · fallos "
              f"{r['fallos_val_semilla_0']}  ({time.time() - t0:.0f} s)", flush=True)

        # figura 2 (por TAU): los mismos 100 dígitos
        C._estilo()
        fig, axs = plt.subplots(10, 10, figsize=(14, 16))
        pos = {int(i): n for n, i in enumerate(va)}
        for ax, i in zip(axs.flat, muestra):
            p = pv0[tau][pos[int(i)]]; ok = p == y[i]
            BG.dibujar_detector(ax, img[i].astype(np.float32), img[i].astype(float),
                                f"{y[i]} → lee {p}" + (" ✓" if ok else " ✗"), VERDE if ok else NAR)
            contorno(ax, img[i])
        fig.suptitle(f"24 · TAU {tau:g} · los mismos 100 dígitos de val (10 por clase) · máscara fuera de la tinta "
                     f"{r['mascara_fuera_pc']} % · val {r['val']:.3f} · a ciegas {r['ciega']:.3f}\nazul pálido = dígito · "
                     "línea negra = borde del dígito (lo de fuera es desborde) · gris = máscara sin giro medible · verde recto · naranja curvo · flecha al centro de curvatura",
                     fontsize=11, color=C.T1)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(DB.IMG / f"24-tau-2-detector-tau{str(tau).replace('.', '')}.png", dpi=90); plt.close(fig)

    # figura 3: los que discrepan, lado a lado (hay que redibujar cada uno con su TAU)
    dis = np.flatnonzero((pv0[0.3] != pv0[0.6]) | (pv0[0.3] != pv0[0.9]) | (pv0[0.6] != pv0[0.9]))
    dis = dis[np.argsort(y[va][dis], kind="stable")]
    out["discrepan_val"] = {"n": int(len(dis)), "indices": [int(va[j]) for j in dis]}
    trios = 6; filas = int(np.ceil(len(dis) / trios))
    fig, axs = plt.subplots(filas, 3 * trios, figsize=(3 * trios * 1.05, filas * 1.3 + 1.0))
    axs = np.atleast_2d(axs)
    for a in axs.flat:
        a.axis("off")
    for t_i, tau in enumerate(TAUS):
        DL.configurar(**{**cal, "tau": tau})
        for n, j in enumerate(dis):
            ax = axs[n // trios, 3 * (n % trios) + t_i]; ax.axis("on"); i = va[j]
            p = pv0[tau][j]; ok = p == y[i]
            BG.dibujar_detector(ax, img[i].astype(np.float32), img[i].astype(float),
                                f"{y[i]}·τ{tau:g}→{p}", VERDE if ok else NAR)
            contorno(ax, img[i])
    fig.suptitle(f"24 · Los {len(dis)} dígitos de val que los tres TAU NO leen igual (semilla 0) · de cada uno, tres celdas: "
                 "TAU 0,3 · 0,6 · 0,9 · «real·τ→leído», verde = acierta", fontsize=10.5, color=C.T1)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(DB.IMG / "24-tau-3-discrepan.png", dpi=90); plt.close(fig)

    # figura 1: estadísticas
    T = out["taus"]; ks = [f"{t:g}" for t in TAUS]
    fig, axs = plt.subplots(2, 3, figsize=(17, 9.4))
    a = axs[0, 0]; x_ = np.arange(3); w = 0.38
    a.bar(x_ - w / 2, [T[k]["val"] for k in ks], w, color=[COL[t] for t in TAUS], label="val (1617)")
    a.bar(x_ + w / 2, [T[k]["ciega"] for k in ks], w, color=[COL[t] for t in TAUS], alpha=0.5, label="a ciegas (3823)")
    for n, k in enumerate(ks):
        a.text(n - w / 2, T[k]["val"] + 0.001, f"{T[k]['val']:.3f}", ha="center", fontsize=8.5)
        a.text(n + w / 2, T[k]["ciega"] + 0.001, f"{T[k]['ciega']:.3f}", ha="center", fontsize=8.5)
    a.set_xticks(x_); a.set_xticklabels([f"TAU {k}" for k in ks]); a.set_ylim(0.88, 0.945); a.legend(frameon=False, fontsize=8)
    a.set_title("acierto (3 semillas): val sólido · a ciegas claro", loc="left", fontsize=9.5)
    a = axs[0, 1]
    a.bar(x_, [T[k]["mascara_fuera_pc"] for k in ks], color=[COL[t] for t in TAUS])
    for n, k in enumerate(ks):
        a.text(n, T[k]["mascara_fuera_pc"] + 0.5, f"{T[k]['mascara_fuera_pc']} %", ha="center", fontsize=9)
    a.set_xticks(x_); a.set_xticklabels([f"TAU {k}" for k in ks])
    a.set_title("% de la máscara FUERA de la tinta (los 1617 de val)", loc="left", fontsize=9.5)
    a = axs[0, 2]
    for n, (t, k) in enumerate(zip(TAUS, ks)):
        a.bar(np.arange(10) + (n - 1) * 0.27, T[k]["por_digito_val"], 0.27, color=COL[t], label=f"TAU {k}")
    a.set_xticks(range(10)); a.set_ylim(0.7, 1.02); a.legend(frameon=False, fontsize=8, ncol=3)
    a.set_title("acierto por dígito (val, semilla 0)", loc="left", fontsize=9.5)
    a = axs[1, 0]
    gn = [n for n, *_ in DD.GRUPOS]
    for n, (t, k) in enumerate(zip(TAUS, ks)):
        a.bar(np.arange(len(gn)) + (n - 1) * 0.27, [T[k]["por_grosor"][g]["ciega"] for g in gn], 0.27, color=COL[t],
              label=f"TAU {k}")
    a.set_xticks(range(len(gn))); a.set_xticklabels(gn, fontsize=8); a.set_ylim(0.75, 1.0); a.legend(frameon=False, fontsize=8)
    a.set_title("acierto A CIEGAS por grosor del trazo (semilla 0; 3823 dígitos)", loc="left", fontsize=9.5)
    a = axs[1, 1]
    for t, k in zip(TAUS, ks):
        a.plot(DS, T[k]["desplazamiento_horizontal"], "o-", color=COL[t], label=f"TAU {k}")
    a.set_xticks(DS); a.set_ylim(0.3, 1.0); a.legend(frameon=False, fontsize=8); a.set_xlabel("desplazamiento horizontal (px)")
    a.set_title("desplazamientos (val, semilla 0)", loc="left", fontsize=9.5)
    a = axs[1, 2]; a.axis("off")
    lin = [f"{'':10s} {'val':>6s} {'ciega':>6s} {'fuera':>6s} {'fallos':>7s} {'sin gruesos':>12s}"]
    for k in ks:
        r = T[k]
        lin.append(f"TAU {k:6s} {r['val']:6.3f} {r['ciega']:6.3f} {r['mascara_fuera_pc']:5.1f}% {r['fallos_val_semilla_0']:7d} "
                   f"{r['fallos_val_sin_muy_gruesos']:12d}")
    lin += ["", f"dígitos de val que los tres no leen igual: {len(dis)}", "(figura 24-3)", "",
            "val y a ciegas: media de 3 semillas;", "lo demás, semilla 0"]
    a.text(0, 1, "\n".join(lin), va="top", fontsize=9.5, family="monospace", color=C.T1)
    for a in axs.flat:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("24 · El detector DELGADO con TAU 0,3 · 0,6 · 0,9 sobre TODOS los dígitos (val 1617 + a ciegas 3823)",
                 fontsize=11.5, color=C.T1)
    fig.tight_layout(); fig.savefig(DB.IMG / "24-tau-1-estadisticas.png", dpi=100); plt.close(fig)
    (AQUI / "resultados-tau.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"discrepan {len(dis)} · → 24-tau-*.png · total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
