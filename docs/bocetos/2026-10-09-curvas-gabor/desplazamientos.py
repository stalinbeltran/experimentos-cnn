#!/usr/bin/env python3
"""Primera aplicación de la regla del dueño del 2026-10-09 (CLAUDE.md del repo, § «la resistencia a los DESPLAZAMIENTOS
se mide SIEMPRE»): curvas de acierto contra desplazamiento, en los DOS niveles que pide la regla.

    DÍGITO ENTERO   los tres compositores de `digitos.py` (curvas · rectas · combinado, semilla 0) sobre los 1617 de val
                    movidos d px en horizontal y en vertical, d = −4…+4. Se mide el acierto y cuántos CAMBIAN de lectura
                    respecto de d = 0 (el acierto puede moverse poco con muchos cambios que se compensan).
    FEATURE         el detector de curvas de `curvas.py` sobre el banco de `rect-lin`: rectas finas (grosor 2–4,
                    largo 16/22) leídas «recta», arcos R 6–27 (grosor 2/4) leídos «curva», y negativos leídos como
                    recta o curva (falsos positivos), con la figura movida d px.

Lo que sale por un borde SE PIERDE (no se usa np.roll, que la haría reaparecer por el lado contrario), y se cuenta:
una curva sobre figuras recortadas mide el recorte, no la tolerancia al desplazamiento.

    python desplazamientos.py            → imagenes/9-desplazamientos.png · resultados-desplazamientos.json (≈ 4 min)
    python desplazamientos.py --figura   → sólo redibuja desde el JSON
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parents[2]))
import curvas as C                                                  # noqa: E402
import digitos as D                                                 # noqa: E402
import por_que_5 as P                                               # noqa: E402
from gemelos_5 import mover                                         # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

IMG = AQUI / "imagenes"
SUP, T1, T2, NAR, AZU = P.SUP, P.T1, P.T2, P.NAR, P.AZU
GRIS = "#8d8c88"
DS = list(range(-4, 5))
DIRS = {"horizontal": lambda a, d: mover(a, 0, d), "vertical": lambda a, d: mover(a, d, 0)}


def recortado(a: np.ndarray, b: np.ndarray) -> bool:
    return int(b.sum()) < int(a.sum())


def nivel_digito(out: dict) -> None:
    d = dict(np.load(exigir_dataset(D.DIGITOS) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va = np.flatnonzero(part == "train"), np.flatnonzero(part == "val")
    Xr, Xc = D.caract_rectas(img), D.caract_curvas(img)
    bloques = {"curvas": (Xc, lambda x: D.caract_curvas(x)), "rectas": (Xr, lambda x: D.caract_rectas(x)),
               "combinado": (np.concatenate([Xr, Xc], 1), None)}
    modelos = {}
    for n, (X, _) in bloques.items():
        W, b, mu, sd = P.compositor(X[tr], y[tr], 0); modelos[n] = (W, b, mu, sd)
    res = {}
    for dire, f in DIRS.items():
        res[dire] = {n: {"acierto": [], "cambian": []} for n in bloques}; res[dire]["recortados"] = []
        base = {}
        for dd in DS:
            xs = np.stack([f(img[i], dd) for i in va])
            res[dire]["recortados"].append(round(float(np.mean([recortado(img[i], xs[k]) for k, i in enumerate(va)])), 4))
            fr, fc = D.caract_rectas(xs), D.caract_curvas(xs)
            feats = {"curvas": fc, "rectas": fr, "combinado": np.concatenate([fr, fc], 1)}
            for n in bloques:
                W, b, mu, sd = modelos[n]; p = (((feats[n] - mu) / sd) @ W.T + b).argmax(1)
                if dd == 0:
                    base[n] = p
                res[dire][n]["acierto"].append(round(float((p == y[va]).mean()), 4))
                res[dire][n]["cambian"].append(round(float((p != base.get(n, p)).mean()), 4) if dd != 0 else 0.0)
            print(f"  dígito {dire:10s} d {dd:+d}  recortados {res[dire]['recortados'][-1]:.2f}  " +
                  "  ".join(f"{n} {res[dire][n]['acierto'][-1]:.3f}" for n in bloques), flush=True)
        # el «cambian» de d < 0 se calculó antes de tener la base de d = 0: se rehace
        for n in bloques:
            W, b, mu, sd = modelos[n]
            for k, dd in enumerate(DS):
                if dd < 0:
                    xs = np.stack([f(img[i], dd) for i in va])
                    fe = {"curvas": D.caract_curvas, "rectas": D.caract_rectas}
                    ft = np.concatenate([D.caract_rectas(xs), D.caract_curvas(xs)], 1) if n == "combinado" else fe[n](xs)
                    p = (((ft - mu) / sd) @ W.T + b).argmax(1)
                    res[dire][n]["cambian"][k] = round(float((p != base[n]).mean()), 4)
    out["digito"] = res


def nivel_feature(out: dict) -> None:
    b = dict(np.load(exigir_dataset("rect-lin-banco-r20261008") / "datos.npz"))
    t, g, lg, rad = b["tipo"], b["grosor"], b["largo"], b["radio"]
    grupos = {"rectas finas → «recta»": (np.flatnonzero((t == 0) & (g <= 4) & (lg >= 16)), ("recta",)),
              "arcos R 6–27 → «curva»": (np.flatnonzero((t == 2) & (rad <= 27) & (g <= 4)), ("curva",)),
              "negativos → recta o curva (FP)": (np.flatnonzero(t >= 3)[::3], ("recta", "curva"))}
    res = {}
    for dire, f in DIRS.items():
        res[dire] = {}
        for nombre, (ids, buenos) in grupos.items():
            tasa, rec = [], []
            for dd in DS:
                hits, cort = 0, 0
                for i in ids:
                    xm = f(b["imagenes"][i], dd); cort += recortado(b["imagenes"][i], xm)
                    vs = C.detectar(xm)[3]; hits += bool(vs) and vs[0]["que"].split(" ")[0] in buenos
                tasa.append(round(hits / len(ids), 4)); rec.append(round(cort / len(ids), 4))
            res[dire][nombre] = {"tasa": tasa, "recortados": rec, "n": int(len(ids))}
            print(f"  feature {dire:10s} {nombre:32s} " + " ".join(f"{v:.2f}" for v in tasa), flush=True)
    out["feature"] = res


def figura(out: dict) -> None:
    P.estilo()
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.4))
    cols = {"curvas": GRIS, "rectas": AZU, "combinado": NAR}
    for j, dire in enumerate(DIRS):
        r = out["digito"][dire]
        a = axs[0, j]
        for n, c in cols.items():
            a.plot(DS, r[n]["acierto"], "o-", color=c, label=n)
        a.set_title(f"DÍGITO ENTERO · movido en {dire}\nacierto en val (1617)", loc="left")
        a.set_ylim(0.2, 1.0)
        b2 = axs[1, j]
        for n, c in cols.items():
            b2.plot(DS, np.array(r[n]["cambian"]) * 100, "o-", color=c, label=n)
        b2.plot(DS, np.array(r["recortados"]) * 100, ":", color=T2, label="% recortados (tinta perdida)")
        b2.set_title(f"% que CAMBIAN de lectura respecto de d = 0 · {dire}", loc="left"); b2.set_ylim(0, 100)
        if dire == "vertical":
            a.text(0, 0.3, "⚠ en vertical casi todos se RECORTAN\n(ocupan los 32 px de alto): esta curva\nmide el recorte, no la tolerancia",
                   ha="center", fontsize=8, color=NAR)
    r = out["feature"]
    for j, dire in enumerate(DIRS):
        a = axs[j, 2]
        for (nombre, v), c in zip(r[dire].items(), (AZU, NAR, GRIS)):
            a.plot(DS, v["tasa"], "o-", color=c, label=f"{nombre} (n {v['n']})")
            if max(v["recortados"]) > 0:
                a.plot(DS, v["recortados"], ":", color=c, lw=0.8)
        a.set_ylim(0, 1.0); a.set_title(f"FEATURE · detector de curvas sobre el banco de rect-lin\nmovido en {dire} (punteado: recortados)", loc="left")
    for a in axs.flat:
        a.set_xticks(DS); a.set_xlabel("desplazamiento d (px)"); a.axvline(0, color="#d8d7d3", lw=0.8)
        a.spines[["top", "right"]].set_visible(False); a.legend(frameon=False, fontsize=7)
    fig.suptitle("9 · Resistencia a los DESPLAZAMIENTOS (regla del 2026-10-09): la misma figura movida d px, sin re-entrenar nada",
                 color=T1, fontsize=11)
    fig.tight_layout()
    fig.savefig(IMG / "9-desplazamientos.png", dpi=100, bbox_inches="tight"); plt.close(fig)
    print("→", IMG / "9-desplazamientos.png")


def main() -> int:
    torch.set_num_threads(2); IMG.mkdir(exist_ok=True)
    if "--figura" in sys.argv:
        figura(json.loads((AQUI / "resultados-desplazamientos.json").read_text(encoding="utf-8"))); return 0
    out = {"desplazamientos": DS}
    nivel_digito(out); nivel_feature(out); figura(out)
    (AQUI / "resultados-desplazamientos.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
