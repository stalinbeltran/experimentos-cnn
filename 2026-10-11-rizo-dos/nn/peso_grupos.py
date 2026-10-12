#!/usr/bin/env python3
"""rizo-2 — ¿POR QUÉ el compositor (MLP 256 · rectas + gris + rizo, con aumento) pesa MÁS las rectas que el gris y el
rizo? Pedido por el dueño el 2026-10-12 tras el rastreo del idx 784. Cinco medidas, 3 semillas:

  1. cuánto peso de la primera capa se lleva cada grupo (‖W‖² por grupo, y por feature: el grupo pequeño no pierde por
     tener pocas features si cada una pesa mucho);
  2. sensibilidad media en val: |grad × entrada| por grupo, promediado sobre los 1617 dígitos (no sólo el 784);
  3. qué pierde el compositor ENTRENADO al quitar un grupo (a la media de train) y qué acierta con un grupo solo;
  4. cuánto separa cada grupo a 8 de 6 por sí solo (regresión logística 8 vs 6 sobre los 36 de train, acierto en val)
     y qué parte del margen 8−6 aporta cada grupo en los 8 y 6 de val (atribución media);
  5. cuánto RUIDO mete el aumento en cada grupo: varianza de una feature entre las 13 copias movidas del mismo dígito,
     frente a su varianza entre dígitos (señal/ruido por grupo). Un grupo que cambia mucho al mover 1 px es un grupo en
     el que el entrenamiento aprende a NO fiarse.

    python nn/peso_grupos.py   → resultados/21-peso-grupos.json · 21-peso-grupos.png
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent)); sys.path.insert(0, str(AQUI.parent / "docs" / "bocetos" / "2026-10-09-curvas-gabor"))
import evaluar as EV                                                # noqa: E402
import resistencia as R                                             # noqa: E402
import rastrear as RA                                               # noqa: E402
import digitos as D                                                 # noqa: E402
import gris_recalibrado as GR                                       # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

G = RA.GRUPOS


def main() -> int:
    torch.set_num_threads(2); GR.configurar(**R.CAL_GRIS)
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va = np.flatnonzero(part == "train"), np.flatnonzero(part == "val")
    cache = pickle.loads(R.CACHE.read_bytes())
    aum = [(0, 0)] + [(k, 0) for k in R.AUM] + [(0, k) for k in R.AUM]
    Xc = np.stack([R.feats(R.mover(img[tr], *m), 8, cache[("tr", m)]) for m in aum])   # (13, 180, 901)
    X = Xc.reshape(-1, Xc.shape[-1]); Y = np.concatenate([y[tr]] * len(aum)); X0 = Xc[0]
    Xva = R.feats(img[va], 8, cache[("va", "x", 0)]); yva = y[va]
    out = {"grupos": {g: b - a for g, (a, b) in G.items()}, "semillas": {}}
    # 5. señal/ruido del aumento (no depende de la semilla), sobre features estandarizadas con la mu/sd de X
    mu, sd = X.mean(0), X.std(0) + 1e-6; Zc = (Xc - mu) / sd
    ruido = Zc.var(0).mean(0)                                         # varianza entre copias, por feature (media sobre dígitos)
    senal = Zc.mean(0).var(0)                                         # varianza entre dígitos de la media de copias
    out["aumento"] = {g: {"var_entre_copias": round(float(ruido[a:b].mean()), 3), "var_entre_digitos": round(float(senal[a:b].mean()), 3),
                          "ruido/senal": round(float(ruido[a:b].mean() / senal[a:b].mean()), 3)} for g, (a, b) in G.items()}
    # 4a. 8 vs 6 por grupo, logística sobre los 36 de train (sin aumento y con él), acierto sobre los 8 y 6 de val
    m86tr, m86va = np.isin(Y, (6, 8)), np.isin(yva, (6, 8))
    out["8vs6_logistica"] = {}
    for g, (a, b) in G.items():
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(((X - mu) / sd)[m86tr][:, a:b], Y[m86tr])
        p = lr.predict(((Xva - mu) / sd)[m86va][:, a:b]); out["8vs6_logistica"][g] = round(float((p == yva[m86va]).mean()), 4)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(((X - mu) / sd)[m86tr], Y[m86tr]); out["8vs6_logistica"]["todo"] = round(float((lr.predict(((Xva - mu) / sd)[m86va]) == yva[m86va]).mean()), 4)
    for sem in D.SEMILLAS:
        red, mu, sd = RA.entrenar(X, Y, sem); W = red[0].weight.detach().numpy(); r = {}
        r["peso_capa1"] = {g: {"‖W‖²": round(float((W[:, a:b] ** 2).sum()), 2), "‖W‖² por feature": round(float((W[:, a:b] ** 2).sum() / (b - a)), 4)} for g, (a, b) in G.items()}
        # 2. sensibilidad media en val: |grad × entrada| del logit de la clase predicha menos el segundo
        z = torch.from_numpy((Xva - mu) / sd).float().requires_grad_(True); L = red(z); top2 = L.topk(2, 1).values; (top2[:, 0] - top2[:, 1]).sum().backward()
        A = (z.grad * z).detach().numpy(); r["sensibilidad_val"] = {g: round(float(np.abs(A[:, a:b]).sum(1).mean()), 3) for g, (a, b) in G.items()}
        # 4b. en los 8 y 6 de val: atribución media de logit8 − logit6 por grupo, por clase real
        z2 = torch.from_numpy((Xva - mu) / sd).float().requires_grad_(True); L2 = red(z2); (L2[:, 8] - L2[:, 6]).sum().backward(); A2 = (z2.grad * z2).detach().numpy()
        r["margen_8menos6"] = {str(k): {g: round(float(A2[yva == k][:, a:b].sum(1).mean()), 3) for g, (a, b) in G.items()} for k in (8, 6)}
        # 3. ablación en todo val
        base = RA.logits(red, mu, sd, Xva).argmax(1); r["acierto_val"] = round(float((base == yva).mean()), 4); r["sin"] = {}; r["solo"] = {}
        for g, (a, b) in G.items():
            xx = Xva.copy(); xx[:, a:b] = X.mean(0)[a:b]; r["sin"][g] = round(float((RA.logits(red, mu, sd, xx).argmax(1) == yva).mean()), 4)
            xx = np.tile(X.mean(0), (len(Xva), 1)); xx[:, a:b] = Xva[:, a:b]; r["solo"][g] = round(float((RA.logits(red, mu, sd, xx).argmax(1) == yva).mean()), 4)
        out["semillas"][str(sem)] = r; print(f"semilla {sem}: {json.dumps(r, ensure_ascii=False)}", flush=True)
    med = lambda f: {g: round(float(np.mean([f(out["semillas"][str(s)], g) for s in D.SEMILLAS])), 4) for g in G}   # noqa: E731
    out["media"] = {"peso_capa1_total": med(lambda r, g: r["peso_capa1"][g]["‖W‖²"]), "peso_capa1_por_feature": med(lambda r, g: r["peso_capa1"][g]["‖W‖² por feature"]),
                    "sensibilidad_val": med(lambda r, g: r["sensibilidad_val"][g]), "sin": med(lambda r, g: r["sin"][g]), "solo": med(lambda r, g: r["solo"][g]),
                    "acierto_val": round(float(np.mean([out["semillas"][str(s)]["acierto_val"] for s in D.SEMILLAS])), 4),
                    "margen_8menos6_en_los_8": med(lambda r, g: r["margen_8menos6"]["8"][g]), "margen_8menos6_en_los_6": med(lambda r, g: r["margen_8menos6"]["6"][g])}
    (AQUI / "resultados" / "21-peso-grupos.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in ("aumento", "8vs6_logistica", "media")}, indent=1, ensure_ascii=False))
    # figura
    gs = list(G); M = out["media"]; fig, axs = plt.subplots(2, 3, figsize=(15, 8)); col = ["#52514e", "#2a78d6", "#c2410c"]
    def barras(ax, vals, titulo, fmt="{:.3f}"):
        b = ax.bar(gs, vals, color=col); ax.set_title(titulo, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
        for bb, v in zip(b, vals):
            ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=8)
    barras(axs[0, 0], [M["peso_capa1_total"][g] for g in gs], "1. peso de la 1.ª capa: ‖W‖² por grupo (total)", "{:.1f}")
    barras(axs[0, 1], [M["peso_capa1_por_feature"][g] for g in gs], "1b. ‖W‖² POR FEATURE (rizo: 5 features · rectas: 512 · gris: 384)", "{:.4f}")
    barras(axs[0, 2], [M["sensibilidad_val"][g] for g in gs], "2. sensibilidad media en val: Σ|grad × entrada| por grupo\n(margen 1.º − 2.º logit, 1617 dígitos)")
    ax = axs[1, 0]; w = 0.38; ax.bar(np.arange(3) - w / 2, [M["sin"][g] for g in gs], w, color="#9a9a9a", label="SIN ese grupo (a la media)"); ax.bar(np.arange(3) + w / 2, [M["solo"][g] for g in gs], w, color=col, label="SÓLO ese grupo")
    ax.axhline(M["acierto_val"], ls="--", color="k", lw=1, label=f"los tres: {M['acierto_val']:.4f}"); ax.set_xticks(range(3)); ax.set_xticklabels(gs); ax.set_ylim(0, 1.02); ax.legend(fontsize=7, frameon=False, loc="lower left")
    for i, g in enumerate(gs):
        ax.text(i - w / 2, M["sin"][g], f"{M['sin'][g]:.3f}", ha="center", va="bottom", fontsize=7); ax.text(i + w / 2, M["solo"][g], f"{M['solo'][g]:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("3. acierto en val del compositor ENTRENADO, quitando o dejando un grupo", fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    ax = axs[1, 1]; L8 = out["8vs6_logistica"]; ax.bar(list(L8), [L8[k] for k in L8], color=col + ["#1b7f3b"]); ax.set_ylim(0.5, 1.02)
    for i, k in enumerate(L8):
        ax.text(i, L8[k], f"{L8[k]:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("4. 8 contra 6 con CADA grupo solo: logística sobre train (con aumento),\nacierto en los 8 y 6 de val", fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    ax = axs[1, 2]; A_ = out["aumento"]; ax.bar(np.arange(3) - w / 2, [A_[g]["var_entre_copias"] for g in gs], w, color="#9a9a9a", label="varianza ENTRE COPIAS movidas (ruido)"); ax.bar(np.arange(3) + w / 2, [A_[g]["var_entre_digitos"] for g in gs], w, color=col, label="varianza ENTRE DÍGITOS (señal)")
    for i, g in enumerate(gs):
        ax.text(i, max(A_[g]["var_entre_copias"], A_[g]["var_entre_digitos"]), f"ruido/señal {A_[g]['ruido/senal']:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(gs); ax.legend(fontsize=7, frameon=False); ax.set_title("5. lo que el AUMENTO (±1–3 px) mueve cada feature, por grupo\n(features estandarizadas)", fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("rizo-2 · 21 · por qué el compositor MLP 256 pesa más las rectas que el gris y el rizo (3 semillas)", fontsize=11); fig.tight_layout()
    fig.savefig(AQUI / "resultados" / "21-peso-grupos.png", dpi=100); print("→ resultados/21-peso-grupos.json · 21-peso-grupos.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
