#!/usr/bin/env python3
"""rizo-2 — RESISTENCIA del compositor a los DESPLAZAMIENTOS, y cómo dársela (pedido del dueño, 2026-10-12: «dar mayor
resistencia al compositor, variando por varios métodos la posición de los píxeles de los detectores»).

Compositor: MLP 256 · rectas + gris + rizo (el mejor honesto, detectores.py). Se mide el acierto en val con el dígito
movido d = −4…+4 px, horizontal y vertical por separado (sólo los no recortados: los que no pierden tinta), y el % que
cambia de lectura respecto de d = 0. Cuatro métodos, acumulables:

  A  base: entrenar con los 180 tal cual (celdas 8×8)
  B  AUMENTO: entrenar también con copias movidas ±1, ±2, ±3 px en x y en y (180 × 13)
  C  CELDAS MÁS GRUESAS: 4×4 en vez de 8×8 (la lectura cambia menos al mover 1–2 px), con aumento
  D  CONJUNTO EN PRUEBA: promediar los logits de 9 copias movidas (0, ±1 px en x e y) del dígito, con B

    python nn/resistencia.py   → resultados/18-resistencia.json · 18-resistencia.png  (~2 h: el rizo se calcula una vez por copia movida de val, 18 copias)
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

AQUI = Path(__file__).resolve().parents[1]
BOCETO = AQUI.parent / "docs" / "bocetos" / "2026-10-09-curvas-gabor"
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent)); sys.path.insert(0, str(BOCETO))
import compositores as CO                                           # noqa: E402
import digitos_variantes as DV                                      # noqa: E402
import evaluar as EV                                                # noqa: E402
import bordes_gabor as BG                                           # noqa: E402
import digitos as D                                                 # noqa: E402
import digitos_bordes as DB                                         # noqa: E402
import gris_recalibrado as GR                                       # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

DS = list(range(-4, 5)); AUM = (-3, -2, -1, 1, 2, 3)
CAL_GRIS = {"tau": 0.1, "sigma_e": 0.5, "coh_min": 0.25, "lam": 3.0}
GRIS = BG.BordeGabor(10.0, k=9, binarizar=False)


@torch.no_grad()
def rectas(img, celdas):
    xt = torch.from_numpy(img.astype(np.float32)[:, None]); salidas = []
    for e in range(2):
        xe = xt if e == 0 else F.avg_pool2d(xt, 2 ** e)
        m = F.relu(F.conv2d(xe, D.GABOR4, padding=4)); m = F.conv2d(m, D.INTEG, padding=D.L_INT // 2, groups=4)
        salidas.append(F.adaptive_max_pool2d(m, celdas).flatten(1))
    return torch.cat(salidas, 1).numpy()


@torch.no_grad()
def gris(img, celdas):
    m = torch.from_numpy(np.stack([DB.mapas_bordes(x, GRIS) for x in img]))
    return F.adaptive_max_pool2d(m, celdas).flatten(2)[:, :6].flatten(1).numpy()


def candidatos(img):
    return [DV.candidatos(x) for x in img]


def desplazar_candidatos(C, ex, ey):
    """Los candidatos de la copia movida (ex, ey): los mismos, desplazados. Exacto si no se pierde tinta; cerca del
    borde los rayos podrían ver otra cosa, y eso se acepta (la alternativa eran 9 cálculos del rizo por copia: 14 h)."""
    return [[dict(c, centro_hueco=(c["centro_hueco"][0] + ex, c["centro_hueco"][1] + ey)) for c in cs
             if 0 <= c["centro_hueco"][0] + ex < 32 and 0 <= c["centro_hueco"][1] + ey < 32] for cs in C]


def feats(img, celdas, C):
    return np.concatenate([rectas(img, celdas), gris(img, celdas), CO.compacta(C, "rizo (h0, 8px)")], 1)


def mlp_logits(xtr, ytr, sem, oculta=256):
    torch.manual_seed(sem); mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    red = torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], oculta), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(oculta, 10))
    opt = torch.optim.Adam(red.parameters(), lr=3e-3, weight_decay=1e-3)
    xt, yt = torch.from_numpy((xtr - mu) / sd).float(), torch.from_numpy(ytr).long()
    for _ in range(400):
        red.train(); opt.zero_grad(); F.cross_entropy(red(xt), yt).backward(); opt.step()
    red.eval()

    @torch.no_grad()
    def lg(X):
        return red(torch.from_numpy((X - mu) / sd).float()).numpy()
    return lg


def mover(img, dx=0, dy=0):
    return np.stack([EV.mover(x, dx, dy) for x in img])


def main() -> int:
    torch.set_num_threads(2); t0 = time.time(); GR.configurar(**CAL_GRIS)
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va = np.flatnonzero(part == "train"), np.flatnonzero(part == "val")
    Itr, Iva, ytr, yva = img[tr], img[va], y[tr], y[va]
    # entrenamiento: original y aumentado, en 8×8 y 4×4
    aum = [(0, 0)] + [(k, 0) for k in AUM] + [(0, k) for k in AUM]
    Ctr = {m: candidatos(mover(Itr, *m)) for m in aum}
    Xtr = {c: {m: feats(mover(Itr, *m), c, Ctr[m]) for m in aum} for c in (8, 4)}
    print(f"entrenamiento: {len(aum)} copias × 2 rejillas ({time.time() - t0:.0f} s)", flush=True)
    # validación movida
    Xva, rec = {}, {}
    for eje in ("x", "y"):
        for dd in DS:
            m = (dd, 0) if eje == "x" else (0, dd); Im = mover(Iva, *m)
            rec[(eje, dd)] = np.array([a.sum() != b.sum() for a, b in zip(Im, Iva)])
            Cm = candidatos(Im)
            for c in (8, 4):
                Xva[(eje, dd, c)] = feats(Im, c, Cm)
            # para el conjunto en prueba (D): las 8 copias ±1 de cada copia movida (8×8); el rizo, desplazando los candidatos
            for ex in (-1, 0, 1):
                for ey in (-1, 0, 1):
                    if (ex, ey) != (0, 0):
                        Xva[(eje, dd, 8, ex, ey)] = feats(mover(Im, ex, ey), 8, desplazar_candidatos(Cm, ex, ey))
            print(f"val movida {eje} {dd:+d} ({time.time() - t0:.0f} s)", flush=True)
    metodos = {
        "A base (8×8)": {"celdas": 8, "aumento": False, "conjunto": False},
        "B aumento ±1–3 px": {"celdas": 8, "aumento": True, "conjunto": False},
        "C celdas 4×4 + aumento": {"celdas": 4, "aumento": True, "conjunto": False},
        "D conjunto en prueba (±1) + aumento": {"celdas": 8, "aumento": True, "conjunto": True},
    }
    out = {"d": DS, "recortados_pc": {eje: [round(100 * float(rec[(eje, dd)].mean()), 1) for dd in DS] for eje in ("x", "y")}, "metodos": {}}
    for n, cfg in metodos.items():
        c = cfg["celdas"]; copias = aum if cfg["aumento"] else [(0, 0)]
        X = np.concatenate([Xtr[c][m] for m in copias]); Y = np.concatenate([ytr] * len(copias))
        res = {"x": {"acierto": [], "cambian": []}, "y": {"acierto": [], "cambian": []}}
        for sem in D.SEMILLAS:
            lg = mlp_logits(X, Y, sem)
            def pred(eje, dd):
                L = lg(Xva[(eje, dd, c)])
                if cfg["conjunto"]:
                    L = L + sum(lg(Xva[(eje, dd, 8, ex, ey)]) for ex in (-1, 0, 1) for ey in (-1, 0, 1) if (ex, ey) != (0, 0))
                return L.argmax(1)
            p0 = pred("x", 0)
            for eje in ("x", "y"):
                acc, cam = [], []
                for dd in DS:
                    p = pred(eje, dd); ok = ~rec[(eje, dd)]
                    acc.append(float((p[ok] == yva[ok]).mean()) if ok.any() else np.nan); cam.append(float((p[ok] != p0[ok]).mean()) if ok.any() else np.nan)
                res[eje]["acierto"].append(acc); res[eje]["cambian"].append(cam)
        out["metodos"][n] = {**cfg, "n_train": int(len(X))}
        for eje in ("x", "y"):
            out["metodos"][n][eje] = {"acierto": np.nanmean(res[eje]["acierto"], 0).round(4).tolist(), "cambian_pc": (100 * np.nanmean(res[eje]["cambian"], 0)).round(1).tolist()}
        a0 = out["metodos"][n]["x"]["acierto"][4]
        print(f"{n:38s} d=0 {a0:.4f} · x {out['metodos'][n]['x']['acierto']} · y {out['metodos'][n]['y']['acierto']} · cambian x {out['metodos'][n]['x']['cambian_pc']}  ({time.time() - t0:.0f} s)", flush=True)
    (AQUI / "resultados" / "18-resistencia.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    fig, axs = plt.subplots(2, 2, figsize=(12, 7.6)); cols = ["#52514e", "#c2410c", "#2a78d6", "#1b7f3b"]
    for j, eje in enumerate(("x", "y")):
        for (n, m), col in zip(out["metodos"].items(), cols):
            axs[0, j].plot(DS, m[eje]["acierto"], "o-", color=col, label=n, ms=4); axs[1, j].plot(DS, m[eje]["cambian_pc"], "o-", color=col, ms=4)
        ax2 = axs[0, j].twinx(); ax2.bar(DS, out["recortados_pc"][eje], 0.5, color="#e5e3df", zorder=0); ax2.set_ylim(0, 100); ax2.set_ylabel("% recortados (gris)", fontsize=8)
        axs[0, j].set_zorder(ax2.get_zorder() + 1); axs[0, j].patch.set_visible(False)
        axs[0, j].set_title(f"acierto en val contra d ({'horizontal' if eje == 'x' else 'vertical'}), sin recortados", fontsize=9); axs[0, j].set_ylim(0.6, 1.0)
        axs[1, j].set_title("% que cambia de lectura respecto de d = 0", fontsize=9); axs[1, j].set_xlabel("d px"); axs[1, j].set_ylim(0, 60)
        for a in axs[:, j]:
            a.grid(alpha=.3); a.spines[["top"]].set_visible(False)
    axs[0, 0].legend(frameon=False, fontsize=7.5, loc="lower center")
    fig.suptitle("rizo-2 · 18 · resistencia a los DESPLAZAMIENTOS del compositor MLP 256 · rectas + gris + rizo, con cuatro métodos (3 semillas)", fontsize=10)
    fig.tight_layout(); fig.savefig(AQUI / "resultados" / "18-resistencia.png", dpi=100)
    print(f"→ resultados/18-resistencia.json · 18-resistencia.png · {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
