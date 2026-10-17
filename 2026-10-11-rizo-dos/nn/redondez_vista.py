#!/usr/bin/env python3
"""rizo-2 — VISTA PREVIA de la REDONDEZ del vértice: tres definiciones para elegir, con ejemplos (el dueño, 2026-10-17: «el
delgado nos permite apreciar si el vértice es afilado o redondeado… muéstrame las imágenes de las definiciones de redondez con
varios ejemplos; vamos a definirlas ahora»). Nada se entrena ni entra en el revisor.

Sobre los vértices libres de § 102–103 (cresta del DELGADO), con sus dos brazos de cresta (hasta 9 px):

  (a) DESPEGUE — se ajusta una recta a cada brazo (su tramo de 3 a 9 px, lejos de la punta) y se mide la distancia de la
      punta de la cresta al punto donde esas dos rectas se cruzan. Una V afilada pasa por el cruce (≈ 0 px); una U redondeada
      «se despega» de él. Escala propuesta: 0 = 0 px, 1 = ≥ 4 px.
  (b) SE ABRE DE CERCA — el ángulo medido con cuerdas cortas (2 px) menos el medido con cuerdas largas (6 px). En una V los
      dos coinciden; en una U, de cerca se ve casi plano. Escala propuesta: 0 = 0°, 1 = ≥ 90°.
  (c) RADIO — el radio del círculo que mejor encaja en los puntos de la cresta a ≤ 3 px de la punta (ajuste por mínimos
      cuadrados, sin suavizar). Escala propuesta: 0 = 0 px, 1 = ≥ 6 px. ⚠ Con 3 px de brazo, una V muy cerrada también da un
      círculo pequeño pero no nulo: el radio mínimo que se puede leer depende del ángulo.

    ../.venv/bin/python nn/redondez_vista.py   → resultados/104-redondez.{png,json}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import cresta_curvas as CC                                               # noqa: E402
import vertice_libre_vista as VL                                          # noqa: E402
from expcnn import exigir_dataset                                         # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402
from matplotlib.patches import Circle                                     # noqa: E402

TOPE = {"a": 4.0, "b": 90.0, "c": 6.0}
NOMBRE = {"a": "(a) DESPEGUE: punta ↔ cruce (×) de las rectas de los brazos",
          "b": "(b) SE ABRE DE CERCA: ángulo a 2 px (verde) − a 6 px (azul)",
          "c": "(c) RADIO del círculo que encaja en la punta"}
UNIDAD = {"a": "px", "b": "°", "c": "px"}


def _largo(b):
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(b, axis=0).T))])


def _en(b, L):
    """El punto del brazo a L px de recorrido (interpolado)."""
    s = _largo(b)
    if L >= s[-1]:
        return b[-1]
    k = int(np.searchsorted(s, L)); t = (L - s[k - 1]) / (s[k] - s[k - 1] + 1e-9); return b[k - 1] + t * (b[k] - b[k - 1])


def _recta(b, desde=2.0, hasta=6.0):
    """Recta ajustada al tramo del brazo entre `desde` y `hasta` px de la punta: (punto, dirección unitaria hacia FUERA).
    Primera versión: de 3 a 9 px, y en brazos curvos (el lazo de un 6) las rectas se cruzaban a 112 px de la punta."""
    s = _largo(b); q = b[(s >= desde) & (s <= hasta)]
    if len(q) < 2:
        q = b[len(b) // 2:]
    m = q.mean(0); _, vec = np.linalg.eigh(np.cov((q - m).T)); d = vec[:, 1]
    if np.dot(d, b[-1] - b[0]) < 0:
        d = -d
    return m, d


def medir(v) -> dict:
    b1, b2 = v["brazos"]; P = (b1[0] + b2[0]) / 2
    # (a) despegue
    (m1, d1), (m2, d2) = _recta(b1), _recta(b2)
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-3:
        X, da = None, TOPE["a"]
    else:
        t = np.linalg.solve(A, m2 - m1); X = m1 + t[0] * d1; da = float(np.hypot(*(X - P)))
    # (b) se abre de cerca
    def ang(L):
        return VL._angulo(_en(b1, L) - b1[0], _en(b2, L) - b2[0])
    db = max(0.0, float(ang(2.0) - ang(6.0)))   # en un cruce raro salía negativo (de cerca MÁS cerrado): eso no es redondez
    # (c) radio: círculo ajustado (mínimos cuadrados, Kåsa) a los puntos de la cresta a ≤ 3 px de la punta por cada brazo, SIN
    # suavizar. Primera versión: 1/curvatura máxima de la cresta suavizada; el suavizado (σ 1,5) movía la punta y el radio
    # medía sobre todo el suavizado (casi todos entre 2 y 4 px, círculos que no tocaban la punta).
    q = np.concatenate([b1[_largo(b1) <= 3.0][::-1], b2[_largo(b2) <= 3.0][1:]])
    A3 = np.c_[2 * q, np.ones(len(q))]; rhs = (q ** 2).sum(1)
    try:
        sol, *_ = np.linalg.lstsq(A3, rhs, rcond=None); centro = sol[:2]; r = float(np.sqrt(max(sol[2] + (centro ** 2).sum(), 0)))
    except np.linalg.LinAlgError:
        centro, r = P, 99.0
    if not np.isfinite(r) or len(q) < 3:
        centro, r = P, 99.0
    return {"a": da, "b": db, "c": r, "cruce": X, "P": P, "rectas": ((m1, d1), (m2, d2)), "cerca": (_en(b1, 2.0), _en(b2, 2.0)),
            "lejos": (_en(b1, 6.0), _en(b2, 6.0)), "circulo": (centro, r), "puntos_c": q}


def escala(k, x):
    return float(np.clip(x / TOPE[k], 0, 1))


def panel(a, M, v, m, k, titulo):
    cx, cy = v["cx"], v["cy"]
    a.imshow(M, cmap="Greys", vmin=0, vmax=1, alpha=0.5, extent=(0, 32, 32, 0))
    for b in v["brazos"]:
        a.plot(b[:, 0], b[:, 1], "-", color="#222", lw=2.4)
    if k == "a":
        for (mm, d) in m["rectas"]:
            q = np.stack([mm - 9 * d, mm + 9 * d]); a.plot(q[:, 0], q[:, 1], "--", color="#c2410c", lw=1.3)
        if m["cruce"] is not None:
            a.plot(*m["cruce"], "x", color="#c2410c", ms=9, mew=2.2)
            a.plot([m["P"][0], m["cruce"][0]], [m["P"][1], m["cruce"][1]], "-", color="#c2410c", lw=2.5)
    elif k == "b":
        for q, col, lw in ((m["cerca"], "#1b7f3b", 2.6), (m["lejos"], "#2a78d6", 1.4)):
            for e in q:
                a.plot([m["P"][0], e[0]], [m["P"][1], e[1]], "-", color=col, lw=lw)
    else:
        (c0, r) = m["circulo"]
        a.plot(m["puntos_c"][:, 0], m["puntos_c"][:, 1], ".", color="#7c3aed", ms=5)
        a.add_patch(Circle(c0, min(r, 30), fill=False, ec="#7c3aed", lw=1.8)); a.plot(*c0, "+", color="#7c3aed", ms=7)
    a.plot(*m["P"], "o", ms=5, color="#e11d48")
    a.set_xlim(cx - 7, cx + 7); a.set_ylim(cy + 7, cy - 7); a.set_xticks([]); a.set_yticks([])
    a.set_title(titulo, fontsize=8.5)


def main() -> int:
    d = dict(np.load(exigir_dataset(VL.DATASET) / "datos.npz"))
    y = d["etiquetas"].astype(int); va = np.flatnonzero(d["particion"] == "val")
    M = np.load(VL.DELGADOS)["M"].astype(np.float32)                    # la entrada por defecto: el DELGADO
    todos = []
    for i in va:
        for v in VL.vertices_libres(M[i]):
            todos.append((int(i), v, medir(v)))
    X = {k: np.array([t[2][k] for t in todos]) for k in "abc"}
    rk = {k: np.argsort(np.argsort(X[k])) for k in "abc"}
    spear = {f"{p}-{q}": round(float(np.corrcoef(rk[p], rk[q])[0, 1]), 2) for p, q in (("a", "b"), ("a", "c"), ("b", "c"))}
    n_ej = 8; filas = 4
    fig, ax = plt.subplots(filas, n_ej, figsize=(2.45 * n_ej, 2.45 * filas + 1.6))
    for r, k in enumerate("abc"):
        orden = np.argsort(X[k]); pick = [orden[int(round(q * (len(orden) - 1)))] for q in np.linspace(0.02, 0.98, n_ej)]
        for c, j in enumerate(pick):
            i, v, m = todos[j]
            panel(ax[r, c], M[i], v, m, k, f"#{i} (un {y[i]}) · {v['dir']} {v['ang']:.0f}°\n{k}) {min(m[k], 99):.1f} {UNIDAD[k]} → {escala(k, m[k]):.2f}")
        ax[r, 0].set_ylabel(NOMBRE[k] + "\nafilado  →  redondeado", fontsize=9, rotation=0, ha="right", va="center", labelpad=8)
    # fila 4: los MISMOS vértices con las tres medidas, ordenados por la media de las tres escalas
    media = np.array([np.mean([escala(k, t[2][k]) for k in "abc"]) for t in todos]); orden = np.argsort(media)
    for c, j in enumerate([orden[int(round(q * (len(orden) - 1)))] for q in np.linspace(0.02, 0.98, n_ej)]):
        i, v, m = todos[j]
        panel(ax[3, c], M[i], v, m, "c", f"#{i} (un {y[i]}) · {v['dir']} {v['ang']:.0f}°\n"
              f"a {escala('a', m['a']):.2f} · b {escala('b', m['b']):.2f} · c {escala('c', m['c']):.2f}")
    ax[3, 0].set_ylabel("las TRES a la vez\n(mismo vértice)", fontsize=9, rotation=0, ha="right", va="center", labelpad=8)
    fig.suptitle("rizo-2 · 104 · VISTA PREVIA: la REDONDEZ del vértice, tres definiciones para elegir (sobre la cresta del DELGADO; nada entrenado)\n"
                 f"cada fila va de lo más AFILADO (izq.) a lo más REDONDEADO (der.), 8 vértices de val a cuantiles iguales · "
                 f"punto rojo = la punta · escalas 0–1 PROPUESTAS (a: 4 px, b: 90°, c: 6 px)\n"
                 f"{len(todos)} vértices en val · ¿miden lo mismo? correlación de rangos: a–b {spear['a-b']} · a–c {spear['a-c']} · b–c {spear['b-c']}",
                 fontsize=10)
    fig.tight_layout(rect=(0.09, 0, 1, 0.93))
    out = AQUI / "resultados" / "104-redondez"
    fig.savefig(out.with_suffix(".png"), dpi=100)
    res = {"vertices": len(todos), "spearman": spear, "topes": TOPE,
           "cuantiles": {k: [round(float(np.quantile(X[k], q)), 2) for q in (0.1, 0.25, 0.5, 0.75, 0.9)] for k in "abc"}}
    json.dump(res, open(out.with_suffix(".json"), "w"), indent=1); print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
