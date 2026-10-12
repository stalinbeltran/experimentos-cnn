#!/usr/bin/env python3
"""rizo-2 — COMPOSITORES NO LINEALES sobre las mismas características (pedido del dueño, 2026-10-11), para ver si las
salidas del detector de curvas cerradas valen más con un compositor que no sea la regresión lineal de 180 ejemplos.

Características (de `digitos_variantes.py`, candidatos cacheados): «otros» (detector delgado, 384) · + rizo compacto
(2×2 + cuenta, 5) · + las cinco compactas (25) · + rizo 8×8 (128). Compositores: lineal (la referencia, el del boceto) ·
MLP de una capa oculta (64 y 256 unidades, dropout, Adam, early stop en train) · SVM RBF · bosque aleatorio · gradient
boosting (HistGradientBoosting) · k vecinos. Mismo protocolo: 180 train / 1617 val / 3823 a ciegas, 3 semillas.

    python nn/compositores.py   → resultados/16-compositores.json · 16-compositores.png  (~10 min)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
BOCETO = AQUI.parent / "docs" / "bocetos" / "2026-10-09-curvas-gabor"
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent)); sys.path.insert(0, str(BOCETO))
import digitos_variantes as DV                                      # noqa: E402
import delgadas as DL                                               # noqa: E402
import digitos as D                                                 # noqa: E402
import digitos_bordes as DB                                         # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402


def compacta(C, n, celdas=2):
    hmin, al, rmin = DV.VARIANTES[n]; M = np.stack([DV.mapas(DV.filtrar(cs, hmin, al, rmin), al) for cs in C])
    p = F.adaptive_max_pool2d(torch.from_numpy(M[:, 1:2]), celdas).flatten(1).numpy()
    return np.concatenate([p, np.array([[len(DV.filtrar(cs, hmin, al, rmin))] for cs in C], np.float32)], 1)


def cerrado(C, n="rizo (h0, 8px)"):
    """Lo CERRADO de la curva (pedido del dueño, 2026-10-12): del mejor candidato, el cierre de la plantilla que casa
    (0 = media vuelta … 1 = vuelta entera), cos y sin de su abertura, y la media del cierre de todos; 4 números."""
    hmin, al, rmin = DV.VARIANTES[n]; out = []
    for cs in C:
        f = DV.filtrar(cs, hmin, al, rmin)
        if not f:
            out.append([0.0, 0.0, 0.0, 0.0]); continue
        m = f[0]; ab = np.deg2rad(m["orientacion"]) if m.get("orientacion") is not None else None
        out.append([m.get("cierre", 0.0), float(np.cos(ab)) if ab is not None else 0.0, float(np.sin(ab)) if ab is not None else 0.0,
                    float(np.mean([c.get("cierre", 0.0) for c in f]))])
    return np.array(out, np.float32)


def mlp(xtr, ytr, sem, oculta):
    torch.manual_seed(sem); mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    red = torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], oculta), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(oculta, 10))
    opt = torch.optim.Adam(red.parameters(), lr=3e-3, weight_decay=1e-3)
    xt, yt = torch.from_numpy((xtr - mu) / sd).float(), torch.from_numpy(ytr).long()
    for _ in range(400):
        red.train(); opt.zero_grad(); F.cross_entropy(red(xt), yt).backward(); opt.step()
    red.eval()

    @torch.no_grad()
    def pr(X):
        return red(torch.from_numpy((X - mu) / sd).float()).argmax(1).numpy()
    return pr


def sk(modelo):
    def entrenar(xtr, ytr, sem):
        sc = StandardScaler().fit(xtr); m = modelo(sem).fit(sc.transform(xtr), ytr)
        return lambda X: m.predict(sc.transform(X))
    return entrenar


COMPOSITORES = {
    "lineal (referencia)": DB.entrenar,
    "MLP 64": lambda x, y, s: mlp(x, y, s, 64),
    "MLP 256": lambda x, y, s: mlp(x, y, s, 256),
    "SVM RBF (C 10)": sk(lambda s: SVC(C=10, gamma="scale", random_state=s)),
    "bosque 500": sk(lambda s: RandomForestClassifier(500, random_state=s, n_jobs=2)),
    "gradient boosting": sk(lambda s: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=s)),
    "k vecinos (5)": sk(lambda s: KNeighborsClassifier(5)),
}


def main() -> int:
    torch.set_num_threads(2); t0 = time.time()
    DL.configurar(**json.loads((BOCETO / "resultados-delgadas.json").read_text())["elegida"])
    d = dict(np.load(exigir_dataset(DV.EV.DATASET) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va, ex = [np.flatnonzero(part == p) for p in ("train", "val", "extra")]
    XA = DB.caract(img, borde=lambda z: z.astype(np.float32))["bordes"]; C = pickle.loads(DV.CACHE.read_bytes())
    rizo = compacta(C, "rizo (h0, 8px)"); cinco = np.concatenate([compacta(C, n) for n in DV.VARIANTES], 1)
    rizo8, _ = DV.caract(C, *DV.VARIANTES["rizo (h0, 8px)"])
    conjuntos = {"otros": XA, "otros + rizo (5)": np.concatenate([XA, rizo], 1), "otros + las cinco (25)": np.concatenate([XA, cinco], 1),
                 "otros + rizo 8×8 (128)": np.concatenate([XA, rizo8], 1)}
    res = {}
    for cn, ent in COMPOSITORES.items():
        for xn, X in conjuntos.items():
            v, c = [], []
            for sem in D.SEMILLAS:
                pr = ent(X[tr], y[tr], sem); v.append(float((pr(X[va]) == y[va]).mean())); c.append(float((pr(X[ex]) == y[ex]).mean()))
            res[f"{cn} · {xn}"] = {"compositor": cn, "caracteristicas": xn, "val": round(float(np.mean(v)), 4), "val_rango": [round(min(v), 4), round(max(v), 4)],
                                   "ciega": round(float(np.mean(c)), 4)}
            print(f"{cn:22s} · {xn:26s} val {np.mean(v):.4f} [{min(v):.4f}–{max(v):.4f}] · ciega {np.mean(c):.4f}  ({time.time() - t0:.0f} s)", flush=True)
    (AQUI / "resultados" / "16-compositores.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    fig, ax = plt.subplots(figsize=(11, 4.6)); cols = {"otros": "#52514e", "otros + rizo (5)": "#c2410c", "otros + las cinco (25)": "#2a78d6", "otros + rizo 8×8 (128)": "#9fc2ec"}
    k = len(conjuntos); w = 0.8 / k
    for j, (xn, col) in enumerate(cols.items()):
        ax.bar(np.arange(len(COMPOSITORES)) + (j - (k - 1) / 2) * w, [res[f"{cn} · {xn}"]["val"] for cn in COMPOSITORES], w, color=col, label=xn)
        for i, cn in enumerate(COMPOSITORES):
            ax.plot(i + (j - (k - 1) / 2) * w, res[f"{cn} · {xn}"]["ciega"], "_", color="k", ms=8, mew=1.2)
    ax.set_xticks(range(len(COMPOSITORES))); ax.set_xticklabels(list(COMPOSITORES), fontsize=8); ax.set_ylim(0.8, 1.0); ax.set_ylabel("acierto en val (barra) · a ciegas (raya)")
    ax.legend(frameon=False, fontsize=8, ncol=4); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=.3)
    ax.set_title("rizo-2 · 16 · compositores NO lineales × características (180 train · 1617 val · 3823 a ciegas · 3 semillas)", loc="left", fontsize=9.5)
    fig.tight_layout(); fig.savefig(AQUI / "resultados" / "16-compositores.png", dpi=100)
    print(f"→ resultados/16-compositores.json · 16-compositores.png · {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
