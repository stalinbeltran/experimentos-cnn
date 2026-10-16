"""¿A qué OTROS dígitos se parece cada dígito, aun cuando el compositor lo reconoce bien? (el dueño, 2026-10-16:
«me dará una idea de qué tan desviado está el reconocimiento»).

Para cada clase real de val (11 clases: el 4 en dos, 4a abierto / 4l cerrado), sólo con los dígitos que el compositor
ACIERTA, la probabilidad media (softmax, media de las 3 semillas) que da a cada una de las OTRAS clases. La diagonal no se
mira: es el acierto. Sale una matriz 11×11 (`resultados/99-parecidos.json`) y su figura (`resultados/99-parecidos.png`).

    ../.venv/bin/python nn/parecidos.py ["<nombre del compositor>"]   # por defecto, el de mejor val del almacén
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "web")); sys.path.insert(0, str(AQUI / "nn"))
import servidor as S  # noqa: E402

NOMBRE = sys.argv[1] if len(sys.argv) > 1 else "aprobadas + vértice en UNA celda, sin golpes · 4 en dos · +extra"


def calcular(nombre: str = NOMBRE) -> dict:
    alm = S.almacen_revisor(); rec = S.recetas_guardadas()
    if alm is None:
        raise SystemExit("✗ sin repo de datos: no hay pesos que leer")
    import json as _j
    todas = {**_j.loads((alm / "recetas.json").read_text()), **rec}
    if nombre not in todas:
        raise SystemExit(f"✗ no conozco «{nombre}». Hay: {list(todas)}")
    bl = todas[nombre]; d = S.Datos()
    c = S.Compositor(nombre, bl["bloques"], bl.get("celdas", 3), bl.get("semillas", S.SEMILLAS), bl.get("sub4", False), bl.get("extra", False))
    if not c.restaurar(alm, c.huella(d)):
        raise SystemExit(f"✗ los pesos de «{nombre}» no casan con este código o no están: hay que entrenarlo")
    todos, _, _ = d.features(c.celdas); X = c.corta(todos[d.va])
    P = np.mean([torch.softmax(red(torch.from_numpy((X - mu) / sd).float()), 1).detach().numpy() for red, mu, sd in c.redes], 0)
    y = d.etiquetas_finas()[d.va] if c.sub4 else d.y[d.va]; K = c.CLASES_FINAS if c.sub4 else c.CLASES[:10]; n = len(K)
    ok = P.argmax(1) == y; M = np.full((n, n), np.nan); cuantos = []
    for r in range(n):
        sel = ok & (y == r); cuantos.append([int(sel.sum()), int((y == r).sum())])
        if sel.any():
            M[r] = P[sel].mean(0)
    return {"compositor": nombre, "val": c.resumen.get("val"), "clases": K, "aciertos_y_total": cuantos,
            "prob_media": [[None if np.isnan(v) else round(float(v), 6) for v in fila] for fila in M]}


def figura(R: dict, salida: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    K = R["clases"]; n = len(K); M = np.array([[np.nan if v is None else v for v in f] for f in R["prob_media"]], float) * 100
    propia = [M[i, i] for i in range(n)]
    for i in range(n):
        M[i, i] = np.nan                                           # la diagonal es el acierto: no se pinta
    cmap = LinearSegmentedColormap.from_list("azul", ["#f4f8fc", "#9ec5e8", "#3b82c4", "#123e72"]); cmap.set_bad("#ffffff")
    fig, ax = plt.subplots(figsize=(8.6, 7.4), dpi=150)
    im = ax.imshow(M, cmap=cmap, norm=LogNorm(vmin=0.01, vmax=max(1.0, np.nanmax(M))))
    for r in range(n):
        orden = np.argsort(-np.nan_to_num(M[r], nan=-1))[:3]       # los 3 más parecidos de la fila, en negrita
        for k in range(n):
            if r == k:
                ax.text(k, r, "·", ha="center", va="center", color="#bbbbbb", fontsize=10); continue
            v = M[r, k]
            if np.isnan(v):
                continue
            t = f"{v:.1f}" if v >= 0.1 else f"{v:.2f}" if v >= 0.01 else "0"
            ax.text(k, r, t, ha="center", va="center", fontsize=8.5, fontweight="bold" if k in orden else "normal",
                    color="#ffffff" if v >= 0.6 else "#1f2937")
    ax.set_xticks(range(n), K); ax.set_yticks(range(n), [f"{k}  ·  {a}/{t}  ·  {pr:.0f} %" for k, (a, t), pr in zip(K, R["aciertos_y_total"], propia)])
    ax.set_xlabel("se parece a… (probabilidad media, %)"); ax.set_ylabel("dígito real  ·  aciertos / total en val  ·  % en su propia clase")
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title(f"¿A qué se parece cada dígito BIEN reconocido?\n{R['compositor']}\nval {R['val']} · media de 3 semillas · negrita = sus 3 más parecidos",
                 fontsize=9.5, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("%  (escala logarítmica)", fontsize=8)
    fig.savefig(salida, bbox_inches="tight", pad_inches=0.15); plt.close(fig)


if __name__ == "__main__":
    R = calcular(); out = AQUI / "resultados"; out.mkdir(exist_ok=True)
    (out / "99-parecidos.json").write_text(json.dumps(R, ensure_ascii=False, indent=1))
    figura(R, out / "99-parecidos.png")
    K = R["clases"]; M = np.array([[np.nan if v is None else v for v in f] for f in R["prob_media"]], float)
    for r, k in enumerate(K):
        f = M[r].copy(); f[r] = np.nan; o = np.argsort(-np.nan_to_num(f, nan=-1))[:3]
        print(f"{k:>3}  acierta {R['aciertos_y_total'][r][0]:>3}/{R['aciertos_y_total'][r][1]:<3} · propia {M[r, r] * 100:5.1f} % · se parece a: " +
              ", ".join(f"{K[j]} {f[j] * 100:.2f} %" for j in o))
