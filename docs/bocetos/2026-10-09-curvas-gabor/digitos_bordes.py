#!/usr/bin/env python3
"""TANTEO (pedido del dueño, 2026-10-10): el compositor de dígitos con SÓLO el detector de curvas sobre BORDES
(`bordes.py`: contorno de 1 px → Gabor λ = 3 → giro κ y orientación). REEMPLAZA a la variante «curvas» de `digitos.py`
(detector sobre el trazo, λ = 6, sin orientación: 0,867), que el dueño da por superada: «este está revisado, tiene
prioridad». `digitos.py` se conserva sin tocar porque las figuras 7–9 del boceto se calcularon con él.

Mismo protocolo que `digitos.py`, sin cambiar nada: compositor LINEAL (Adam, 300 épocas a lote completo, lr 1e-2,
L2 1e-3), 180 train / 1617 val de `uci-optdigits-orig-32px-r20261005`, 3 semillas que sólo cambian la inicialización,
y a ciegas los 3823 de otros escritores. Todas las características son FIJAS; sólo se entrena el compositor.

Características (mapas por píxel del borde → max en celdas 8×8 de 4 px):
    BORDES          6 mapas: recto (|κ| < 2 °/px, medible) · el VECTOR de curvatura k = κ·(−sin θ, cos θ) partido en
                    sus 4 sentidos (→ ← ↓ ↑, cada uno max(0, ·), |κ| con tope 12) · golpe (borde con coherencia
                    baja: esquina o cruce)                                                                384
Brazos de comparación, para saber de dónde sale lo que salga (re-ejecutados aquí, no copiados):
    BORDES SIN ORIENTACIÓN  recto · |κ| · golpe del mismo detector (la estructura de la variante vieja)      192
    CURVAS (ANTERIOR)       la variante «curvas» de digitos.py, tal cual                                       192
    RECTAS                  la variante C de rect-bor, tal cual (la referencia de la línea)                   512

PREDICCIÓN (escrita antes de correrlo, 2026-10-10):
    · bordes con orientación ≈ 0,90–0,93: por encima de la variante vieja (0,867), porque ahora el trazo grueso de un
      dígito no sale «no medible» (sus dos bordes son de 1 px) y porque la orientación separa curvas que con sólo |κ|
      son iguales (la panza del 6 abierta a la derecha y la del 9 a la izquierda);
    · sin orientación ≈ 0,86–0,89: sólo la ganancia de medir los bordes;
    · por debajo de rectas (0,955): una curva no dice nada de los tramos rectos largos (el 1, el 7, el 4).
    · desplazamientos: la misma forma que los demás (rejilla de celdas de 4 px): ±1 px apenas, ±2 px −0,05 a −0,10.

DESPLAZAMIENTOS (regla del dueño del 2026-10-09, CLAUDE.md del repo): sin re-entrenar, val movido d = −4…+4 px en
horizontal y en vertical, acierto y % que CAMBIA de lectura respecto de d = 0, con el % recortado (⚠ en vertical los
dígitos de UCI ocupan los 32 px de alto: esa curva mide el recorte). Y a nivel de FEATURE: el detector sobre el banco de
`rect-lin` movido igual. Lo que sale por un borde se pierde (`gemelos_5.mover`, sin np.roll).

    python digitos_bordes.py            → resultados-digitos-bordes.json · imagenes/14-digitos-bordes-*.png  (~5 min)
    python digitos_bordes.py --figura   → sólo redibuja la figura de resultados desde el JSON
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parents[2]))
import curvas as C                                                  # noqa: E402
import digitos as D                                                 # noqa: E402
import bordes as B                                                  # noqa: E402
from gemelos_5 import mover                                         # noqa: E402
from curvas import plt                                              # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

IMG = AQUI / "imagenes"
DS = list(range(-4, 5))
DIRS = {"horizontal": lambda a, d: mover(a, 0, d), "vertical": lambda a, d: mover(a, d, 0)}
NAR, AZU, GRIS, VERDE = "#c2410c", "#2a78d6", "#8a8986", "#1b7f3b"


# ─── las características del detector sobre bordes ────────────────────────────────────────────────────────────────
def mapas_bordes(x: np.ndarray, borde=None) -> np.ndarray:
    """(32,32) → (7,32,32): recto · k→ · k← · k↓ · k↑ · golpe · |κ|. (El último sólo lo usa el brazo sin orientación.)"""
    c = C.campo((borde or B.bordes)(x)); kappa, mask, _ = C.giro(c)   # `borde`: cómo se saca (bordes_gabor.py)
    medible = mask & np.isfinite(kappa)
    k = np.where(medible, np.clip(np.nan_to_num(kappa), -D.KAPPA_TOPE, D.KAPPA_TOPE), 0.0)
    th = np.deg2rad(c["theta"])
    kx, ky = -k * np.sin(th), k * np.cos(th)                # el vector de curvatura: apunta al centro (bordes.trozos)
    curvo = medible & (np.abs(k) >= C.KAPPA_MIN)
    kx, ky = np.where(curvo, kx, 0.0), np.where(curvo, ky, 0.0)
    recto = (medible & (np.abs(k) < C.KAPPA_MIN)).astype(np.float32)
    golpe = (mask & (c["coh"] < C.COH_MIN)).astype(np.float32)
    return np.stack([recto, np.maximum(kx, 0), np.maximum(-kx, 0), np.maximum(ky, 0), np.maximum(-ky, 0), golpe,
                     np.where(curvo, np.abs(k), 0.0)]).astype(np.float32)


@torch.no_grad()
def caract(x: np.ndarray, borde=None) -> dict[str, np.ndarray]:
    m = torch.from_numpy(np.stack([mapas_bordes(xi, borde) for xi in x]))
    p = F.adaptive_max_pool2d(m, 8).flatten(2)              # (n, 7, 64)
    return {"bordes": p[:, :6].flatten(1).numpy(), "bordes sin orientación": p[:, [0, 6, 5]].flatten(1).numpy()}


# ─── el compositor: el de digitos.py, devolviendo pesos para poder evaluarlo movido ───────────────────────────────
def entrenar(xtr, ytr, sem):
    torch.manual_seed(sem)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    W = torch.nn.Linear(xtr.shape[1], 10); opt = torch.optim.Adam(W.parameters(), lr=D.LR, weight_decay=D.L2)
    xt, yt = torch.from_numpy((xtr - mu) / sd).float(), torch.from_numpy(ytr).long()
    for _ in range(D.EPOCAS):
        opt.zero_grad(); F.cross_entropy(W(xt), yt).backward(); opt.step()
    Wn, bn = W.weight.detach().numpy().astype(np.float64), W.bias.detach().numpy().astype(np.float64)
    return lambda X: (((X - mu) / sd) @ Wn.T + bn).argmax(1)


def recortado(a, b) -> bool:
    return int(b.sum()) < int(a.sum())


# ─── las figuras ──────────────────────────────────────────────────────────────────────────────────────────────────
def figura_mapas(img, y, va):
    """Lo que ve el compositor: un dígito de cada clase, su borde, y cada trozo curvo con su flecha hacia el centro."""
    C._estilo()
    ids = [va[np.flatnonzero(y[va] == k)[0]] for k in range(10)]
    fig, axs = plt.subplots(3, 10, figsize=(16, 5.6))
    for j, i in enumerate(ids):
        b = B.bordes(img[i]); c = C.campo(b); kappa, mask, _ = C.giro(c)
        vs = C.veredicto(c, kappa, mask, C.giro(c)[2])
        curvos, rectos = B.trozos(kappa, vs, c["theta"])
        axs[0, j].imshow(img[i], cmap="gray_r", vmin=0, vmax=1); axs[0, j].set_title(str(y[i]), fontsize=11)
        axs[1, j].imshow(b, cmap="gray_r", vmin=0, vmax=1)
        a = axs[2, j]
        med = mask & np.isfinite(kappa)
        a.imshow(b, cmap="gray_r", vmin=0, vmax=1, alpha=0.15)
        a.imshow(np.where(med & (np.abs(kappa) < C.KAPPA_MIN), 1.0, np.nan), cmap="Greens", vmin=0, vmax=1.6)
        a.imshow(np.where(med & (np.abs(kappa) >= C.KAPPA_MIN), np.abs(kappa), np.nan), cmap="Oranges", vmin=0, vmax=12)
        for t in curvos:
            (cx, cy), d = t["centroide"], np.deg2rad(t["direccion"])
            a.annotate("", xy=(cx - 0.5 + 4 * np.cos(d), cy - 0.5 + 4 * np.sin(d)), xytext=(cx - 0.5, cy - 0.5),
                       arrowprops=dict(arrowstyle="-|>", color=AZU, lw=1.4, mutation_scale=8))
        a.set_xlabel(f"{len(curvos)} curvas · {rectos} rectos", fontsize=7.5)
        for ax in axs[:, j]:
            ax.set_xticks([]); ax.set_yticks([])
    for i, t in enumerate(["dígito", "bordes\n(lo que entra)", "detector: verde = recto\nnaranja = curvo · flecha\n→ centro de cada curva"]):
        axs[i, 0].set_ylabel(t, fontsize=8.5)
    fig.suptitle("14 · Lo que ve el compositor: el detector sobre los bordes de un dígito de cada clase (val)",
                 fontsize=11.5, color=C.T1)
    fig.tight_layout(); fig.savefig(IMG / "14-digitos-bordes-1-mapas.png", dpi=105); plt.close(fig)
    print("→ 14-digitos-bordes-1-mapas.png")


def figura_resultados(out):
    C._estilo()
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.6), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    nombres = ["bordes", "bordes sin orientación", "curvas (anterior)", "rectas"]
    colores = [NAR, "#f2a77f", GRIS, AZU]
    a = axs[0, 0]
    for k, (n, col) in enumerate(zip(nombres, colores)):
        r = out["acierto"][n]
        a.barh(k, r["val_1617"], color=col, height=0.6)
        a.text(r["val_1617"] + 0.003, k, f"{r['val_1617']:.3f}  (a ciegas {r['ciega_3823']:.3f})", va="center", fontsize=9)
    a.set_yticks(range(len(nombres))); a.set_yticklabels([f"{n}\n{out['acierto'][n]['caracteristicas']} caract." for n in nombres],
                                                         fontsize=8.5)
    a.invert_yaxis(); a.set_xlim(0.7, 1.05); a.set_title("acierto en val (1617), media de 3 semillas", loc="left", fontsize=10)
    a = axs[1, 0]
    pd = np.array(out["acierto"]["bordes"]["por_digito_val"]); pr = np.array(out["acierto"]["rectas"]["por_digito_val"])
    pc = np.array(out["acierto"]["curvas (anterior)"]["por_digito_val"])
    w = 0.27
    a.bar(np.arange(10) - w, pc, w, color=GRIS, label="curvas (anterior)")
    a.bar(np.arange(10), pd, w, color=NAR, label="bordes")
    a.bar(np.arange(10) + w, pr, w, color=AZU, label="rectas")
    a.set_xticks(range(10)); a.set_ylim(0.5, 1.0); a.legend(frameon=False, fontsize=8, loc="lower left")
    a.set_title("acierto por dígito (val)", loc="left", fontsize=10)
    for j, dire in enumerate(DIRS):
        r = out["desplazamientos"]["digito"][dire]
        a = axs[0, 1 + j]
        for n, col in zip(nombres, colores):
            a.plot(DS, r[n]["acierto"], "o-", color=col, label=n, lw=2 if n == "bordes" else 1.2)
        a.set_ylim(0.0, 1.0); a.set_title(f"DÍGITO movido en {dire} · acierto (semilla 0)", loc="left", fontsize=10)
        if dire == "vertical":
            a.text(0, 0.30, "⚠ en vertical casi todos se RECORTAN\n(ocupan los 32 px de alto):\nesta curva mide el recorte",
                   ha="center", fontsize=8, color=NAR)
        a = axs[1, 1 + j]
        if dire == "horizontal":
            for n, col in zip(nombres, colores):
                a.plot(DS, np.array(r[n]["cambian"]) * 100, "o-", color=col, label=n, lw=2 if n == "bordes" else 1.2)
            a.plot(DS, np.array(r["recortados"]) * 100, ":", color=C.T2, label="% recortados")
            a.set_ylim(0, 100); a.set_title("% que CAMBIA de lectura respecto de d = 0 · horizontal", loc="left", fontsize=10)
        else:
            rf = out["desplazamientos"]["feature"]
            for (g, v), col in zip(rf["horizontal"].items(), (VERDE, NAR, GRIS)):
                a.plot(DS, v["tasa"], "o-", color=col, label=f"{g} (n {v['n']}) · horiz.")
                a.plot(DS, rf["vertical"][g]["tasa"], "s--", color=col, lw=0.9, ms=3.5, label="… vertical")
            a.set_ylim(0, 1.0); a.set_title("FEATURE: el detector sobre el banco de rect-lin, movido", loc="left", fontsize=10)
    for a in list(axs[:, 1:].flat):
        a.set_xticks(DS); a.set_xlabel("desplazamiento d (px)"); a.axvline(0, color="#d8d7d3", lw=0.8)
        a.legend(frameon=False, fontsize=7)
    for a in axs.flat:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("14 · Compositor de dígitos con SÓLO el detector sobre bordes (λ = 3, con orientación), contra el anterior",
                 fontsize=11.5, color=C.T1)
    fig.tight_layout(); fig.savefig(IMG / "14-digitos-bordes-2-resultados.png", dpi=105); plt.close(fig)
    print("→ 14-digitos-bordes-2-resultados.png")


def main() -> int:
    torch.set_num_threads(2); IMG.mkdir(exist_ok=True)
    if "--figura" in sys.argv:                       # sólo redibuja la figura de resultados desde el JSON
        figura_resultados(json.loads((AQUI / "resultados-digitos-bordes.json").read_text(encoding="utf-8"))); return 0
    d = dict(np.load(exigir_dataset(D.DIGITOS) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va, ex = np.flatnonzero(part == "train"), np.flatnonzero(part == "val"), np.flatnonzero(part == "extra")
    t0 = time.time()
    # ⚠ ORDEN: las variantes viejas ANTES de configurar el modo bordes, que cambia los globales de curvas.py
    fe_viejas = {"curvas (anterior)": D.caract_curvas, "rectas": D.caract_rectas}
    X = {n: f(img) for n, f in fe_viejas.items()}
    movidas = {dire: {dd: np.stack([f(img[i], dd) for i in va]) for dd in DS} for dire, f in DIRS.items()}
    X_movidas = {dire: {dd: {n: fv(xs) for n, fv in fe_viejas.items()} for dd, xs in m.items()} for dire, m in movidas.items()}
    B.configurar()
    X.update(caract(img))
    print(f"características en {time.time() - t0:.0f} s: " + " · ".join(f"{n} {v.shape[1]}" for n, v in X.items()), flush=True)
    nombres = ["bordes", "bordes sin orientación", "curvas (anterior)", "rectas"]
    out = {"protocolo": {"train": int(len(tr)), "val": int(len(va)), "a_ciegas": int(len(ex)), "semillas": list(D.SEMILLAS),
                         "epocas": D.EPOCAS, "lr": D.LR, "l2": D.L2, "detector": {"lambda": B.LAM_B, "sigma_E": B.SIGMA_E,
                         "tau": B.TAU_B, "kappa_min": B.KAPPA_MIN_B}}, "acierto": {}}
    modelos0 = {}
    for n in nombres:
        accs, por = [], []
        for s in D.SEMILLAS:
            pred = entrenar(X[n][tr], y[tr], s)
            pv, pe = pred(X[n][va]), pred(X[n][ex])
            accs.append([float((pv == y[va]).mean()), float((pe == y[ex]).mean())])
            por.append([float((pv[y[va] == k] == k).mean()) for k in range(10)])
            if s == 0:
                modelos0[n] = pred
                conf = {}
                for i in np.flatnonzero(pv != y[va]):
                    kk = f"{y[va][i]}→{pv[i]}"; conf[kk] = conf.get(kk, 0) + 1
        accs = np.array(accs)
        out["acierto"][n] = {"caracteristicas": int(X[n].shape[1]), "val_1617": round(float(accs[:, 0].mean()), 4),
                             "val_rango": [round(float(accs[:, 0].min()), 4), round(float(accs[:, 0].max()), 4)],
                             "ciega_3823": round(float(accs[:, 1].mean()), 4),
                             "por_digito_val": [round(float(v), 3) for v in np.mean(por, 0)],
                             "confusiones_semilla_0": dict(sorted(conf.items(), key=lambda z: -z[1])[:8])}
        r = out["acierto"][n]
        print(f"  {n:24s} {r['caracteristicas']:4d} · val {r['val_1617']:.4f} {r['val_rango']} · a ciegas {r['ciega_3823']:.4f}",
              flush=True)

    # ── desplazamientos, nivel DÍGITO (semilla 0, sin re-entrenar) ──
    res = {}
    for dire in DIRS:
        res[dire] = {n: {"acierto": [], "cambian": []} for n in nombres}; res[dire]["recortados"] = []
        preds = {}
        for dd, xs in movidas[dire].items():
            fb = caract(xs)
            fb.update(X_movidas[dire][dd])
            preds[dd] = {n: modelos0[n](fb[n]) for n in nombres}
        for dd in DS:
            res[dire]["recortados"].append(
                round(float(np.mean([recortado(img[i], movidas[dire][dd][k]) for k, i in enumerate(va)])), 4))
            for n in nombres:
                res[dire][n]["acierto"].append(round(float((preds[dd][n] == y[va]).mean()), 4))
                res[dire][n]["cambian"].append(round(float((preds[dd][n] != preds[0][n]).mean()), 4))
        print(f"  dígito {dire:10s} " + " | ".join(f"{n}: " + " ".join(f"{v:.2f}" for v in res[dire][n]["acierto"])
                                                    for n in ("bordes", "curvas (anterior)")), flush=True)
    out["desplazamientos"] = {"d": DS, "digito": res}

    # ── desplazamientos, nivel FEATURE: el detector sobre el banco de rect-lin ──
    b = dict(np.load(exigir_dataset("rect-lin-banco-r20261008") / "datos.npz"))
    t, g, lg, rad = b["tipo"], b["grosor"], b["largo"], b["radio"]
    grupos = {"rectas finas → sólo rectos": (np.flatnonzero((t == 0) & (g <= 4) & (lg >= 16)), "recta"),
              "arcos R 6–27 → curva con su radio": (np.flatnonzero((t == 2) & (rad <= 27) & (g <= 4)), "curva"),
              "negativos → alguna curva (FP)": (np.flatnonzero(t >= 3)[::3], "fp")}
    rf = {}
    for dire, f in DIRS.items():
        rf[dire] = {}
        for nombre, (ids, que) in grupos.items():
            tasa, rec = [], []
            for dd in DS:
                hits = cort = 0
                for i in ids:
                    xm = f(b["imagenes"][i], dd); cort += recortado(b["imagenes"][i], xm)
                    _, kappa, _, vs = C.detectar(B.bordes(xm)); cur, nr = B.trozos(kappa, vs)
                    if que == "recta":
                        hits += bool(nr) and not cur
                    elif que == "curva":
                        hits += B.acierta("curva", int(rad[i]), int(g[i]), cur, vs)
                    else:
                        hits += bool(cur)
                tasa.append(round(hits / len(ids), 4)); rec.append(round(cort / len(ids), 4))
            rf[dire][nombre] = {"tasa": tasa, "recortados": rec, "n": int(len(ids))}
            print(f"  feature {dire:10s} {nombre:36s} " + " ".join(f"{v:.2f}" for v in tasa), flush=True)
    out["desplazamientos"]["feature"] = rf
    (AQUI / "resultados-digitos-bordes.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    figura_mapas(img, y, va)
    figura_resultados(out)
    print(f"total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
