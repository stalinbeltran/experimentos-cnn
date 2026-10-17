#!/usr/bin/env python3
"""rizo-2 — VISTA PREVIA del vértice LIBRE: en cualquier sitio, en 4 direcciones, en UNA celda, y con su AMPLITUD (el dueño,
2026-10-17: «debe detectarlos donde estén… en posición única (si caen en medio de 2 celdas), pues nos sirve por si se ven varios
vértices. También otro vértice con la misma posición única que refleje el ángulo: 0 = 0°, 1 = 180°. Revisa si tiene sentido y
si es factible. Muéstrame cómo quedarían»). PROTOTIPO: no toca `cresta_curvas.vertices` ni nada que use el revisor. ENTRADA: el DELGADO de delg-cnn
(`delgados-k3x3.npz`), que es la entrada por defecto; la tinta no se usa ni para dibujar (el dueño, 2026-10-17).

Por qué en la CRESTA y no en la tinta: el detector de hoy (§ 95) mira los BORDES de la tinta desde la cima del dígito, y por
construcción sólo ve la punta del extremo (§ 101: N siempre en la fila de arriba). Para encontrar un vértice «donde esté» hay que
mirar el trazo por dentro, y la cresta (el esqueleto del delgado, `tramos`) ya es un grafo de polilíneas con sub-píxel.

Qué es un vértice aquí: un punto de la cresta del que salen DOS brazos de al menos BRAZO_MIN px que forman un ángulo < A_MAX,
medido con la cuerda a BRAZO px (amplitud) y, para no confundir una curva cerrada con una esquina, también < A_MAX_CORTO a
BRAZO_CORTO px (una U se abre al mirarla de cerca; una V no). Dos sitios: a mitad de un tramo (mínimo local del ángulo) y en un
racimo, entre cada par de tramos que salen de él (la punta de un Λ grueso suele dejar un racimo con un espolón corto, que no
cuenta como brazo). De los candidatos a < SUPR px se queda el más agudo.
  - dirección = hacia dónde APUNTA la punta (contraria a la bisectriz de los brazos), en conos de 90°: N S E O;
  - posición = la celda 3×3 que contiene el punto (int(c·3/32)), SIN mancha: una sola celda siempre (§ 98);
  - amplitud = ángulo / 180 en esa misma celda (si dos de la misma dirección caen en la misma celda, el más agudo).

⚠ Los parámetros de abajo son una PRIMERA elección, sin calibrar contra ninguna verdad.

    ../.venv/bin/python nn/vertice_libre_vista.py   → resultados/102-vertice-libre.{png,json}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import cresta_curvas as CC                                               # noqa: E402
from expcnn import exigir_dataset                                         # noqa: E402

import matplotlib                                                         # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                           # noqa: E402

DATASET = "uci-optdigits-orig-32px-r20261005"
DELGADOS = AQUI.parent / "2026-10-15-delgado-cnn" / "resultados" / "delgados-k3x3.npz"
# El núcleo (detector, parámetros, celda) vive en `vertice_libre.py` desde el 2026-10-17, que es lo que usa el revisor: dos
# copias de un detector divergen. Aquí sólo queda lo de la figura.
import vertice_libre as _VLB                                             # noqa: E402
from vertice_libre import (N, BRAZO, BRAZO_CORTO, BRAZO_MIN, A_MAX, A_MAX_CORTO, SUPR, ANG_SIGUE, ALCANCE,  # noqa: E402,F401
                           vertices_libres, celda)
_brazo, _camino, _angulo, _direccion, _rumbo = _VLB._brazo, _VLB._camino, _VLB._angulo, _VLB._direccion, _VLB._rumbo
DIRS = ["N", "S", "E", "O"]
CELDAS = ["NO", "N", "NE", "O", "C", "E", "SO", "S", "SE"]
COLOR = {"N": "#c2410c", "S": "#2a78d6", "E": "#1b7f3b", "O": "#7c3aed"}


def rejillas(vs) -> dict:
    """{dir: (presencia 3×3, amplitud 3×3, cuenta)}: lo que recibiría el compositor (4 × 19 = 76 números)."""
    out = {d: [np.zeros(9), np.zeros(9), 0] for d in DIRS}
    for v in vs:
        pr, am, _ = out[v["dir"]]; c = celda(v); out[v["dir"]][2] += 1
        am[c] = v["ang"] / 180 if pr[c] == 0 else min(am[c], v["ang"] / 180); pr[c] = 1.0
    return out


def mover(m, dx, dy):
    """Desplaza sin np.roll: lo que sale se pierde, lo que entra es cero."""
    o = np.zeros_like(m); H = m.shape[0]
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, H - dx), slice(dx, H)) if dx >= 0 else (slice(-dx, H), slice(0, H + dx))
    o[yd, xd] = m[ys, xs]; return o


def main() -> int:
    d = dict(np.load(exigir_dataset(DATASET) / "datos.npz"))
    y = d["etiquetas"].astype(int); va = np.flatnonzero(d["particion"] == "val")
    M = np.load(DELGADOS)["M"].astype(np.float32)   # ⚠ la entrada por defecto es el DELGADO (delg-cnn), NUNCA la tinta (el dueño, 2026-10-17)
    t0 = time.time(); V = {int(i): vertices_libres(M[i]) for i in va}; ms = (time.time() - t0) / len(va) * 1000
    por_clase = {k: [0] * 10 for k in DIRS}; por_celda = {k: [0] * 9 for k in DIRS}; angs = {k: [] for k in DIRS}
    n_vert = np.array([len(V[i]) for i in va]); borde = 0; choques = 0; total = 0
    for i in va:
        vistos = set()
        for v in V[i]:
            total += 1; por_clase[v["dir"]][y[i]] += 1; por_celda[v["dir"]][celda(v)] += 1; angs[v["dir"]].append(v["ang"])
            if min(abs(v["cx"] - b) for b in (32 / 3, 64 / 3)) < 1 or min(abs(v["cy"] - b) for b in (32 / 3, 64 / 3)) < 1:
                borde += 1
            k = (v["dir"], celda(v)); choques += k in vistos; vistos.add(k)
    # contra los 4 cerrados/abiertos del revisor (regla § 85 + las correcciones a mano de web/etiquetas-4.json)
    sys.path.insert(0, str(AQUI / "web")); import servidor as SV                            # noqa: E402
    fin = SV.Datos().etiquetas_finas()
    cuatro = {}
    for nom, cl in (("4l", 10), ("4a", 4)):
        ii = [int(i) for i in va if fin[i] == cl]
        cuatro[nom] = {"n": len(ii), "con_N": sum(any(v["dir"] == "N" for v in V[i]) for i in ii),
                       "con_N_arriba": sum(any(v["dir"] == "N" and celda(v) < 3 for v in V[i]) for i in ii)}
    # resistencia a desplazamientos de la PRESENCIA (regla del dueño del 2026-10-09): % de dígitos cuya rejilla cambia
    def firma(vs):
        r = rejillas(vs); return tuple(np.concatenate([r[k][0] for k in DIRS]).astype(int))
    base = {int(i): firma(V[int(i)]) for i in va}; despl = {}
    for dx, dy in [(-2, 0), (-1, 0), (1, 0), (2, 0), (0, -2), (0, -1), (0, 1), (0, 2)]:
        cambia = sum(firma(vertices_libres(mover(M[i], dx, dy))) != base[int(i)] for i in va)
        despl[f"{dx:+d},{dy:+d}"] = round(100 * cambia / len(va), 1)

    # la figura: 7 dígitos de val, uno por fila: la cresta con sus vértices, y las 8 rejillas (presencia y amplitud × 4)
    idx = [390, 613]                                                     # un 4 cerrado (el de § 101) y un 4 abierto
    for c, minimo in ((8, 0), (7, 2), (2, 2), (1, 1), (5, 2)):           # el 8: su cruce ya NO cuenta; los demás, con VARIOS vértices
        cand = [int(i) for i in va if y[i] == c and len(V[int(i)]) >= minimo]
        idx.append(cand[len(cand) // 2] if cand else int(va[y[va] == c][0]))
    fig = plt.figure(figsize=(17, 2.35 * len(idx) + 2.6))
    gs = fig.add_gridspec(len(idx) + 1, 9, width_ratios=[1.5] + [1] * 8, height_ratios=[1] * len(idx) + [1.05], hspace=0.55, wspace=0.25)
    for r, i in enumerate(idx):
        vs = V[i]; R = rejillas(vs)
        a = fig.add_subplot(gs[r, 0]); a.imshow(M[i], cmap="Greys", vmin=0, vmax=1, alpha=0.45, extent=(0, N, N, 0))   # el DELGADO: la entrada por defecto
        for t in CC.tramos(M[i])[0] + CC.tramos(M[i])[1]:
            a.plot(t["pts"][:, 0], t["pts"][:, 1], "-", color="#333", lw=1.2)
        for b in (32 / 3, 64 / 3):
            a.axhline(b, color="#999", lw=0.6, ls="--"); a.axvline(b, color="#999", lw=0.6, ls="--")
        flecha = {"N": (0, -4), "S": (0, 4), "E": (4, 0), "O": (-4, 0)}
        for v in vs:
            a.plot(v["cx"], v["cy"], "o", ms=8, mfc="none", mec=COLOR[v["dir"]], mew=2.2)
            a.annotate("", (v["cx"] + flecha[v["dir"]][0], v["cy"] + flecha[v["dir"]][1]), (v["cx"], v["cy"]),
                       arrowprops=dict(arrowstyle="->", color=COLOR[v["dir"]], lw=1.8))
            a.text(v["cx"] + 1.2, v["cy"] - 1.2, f"{v['dir']} {v['ang']:.0f}°", color=COLOR[v["dir"]], fontsize=8, weight="bold")
        a.set_xlim(0, N); a.set_ylim(N, 0); a.set_xticks([]); a.set_yticks([])
        a.set_title(f"val #{i}, un {y[i]} · {len(vs)} vértice(s)", fontsize=9)
        for col, (k, cual) in enumerate([(k, 0) for k in DIRS] + [(k, 1) for k in DIRS]):
            g = R[k][cual].reshape(3, 3); a = fig.add_subplot(gs[r, 1 + col])
            a.imshow(g, cmap="Reds" if cual == 0 else "Blues", vmin=0, vmax=1)
            for rr in range(3):
                for cc in range(3):
                    val = g[rr, cc]; txt = "·" if val == 0 else (f"{val:.0f}" if cual == 0 else f"{val:.2f}\n{val * 180:.0f}°")
                    a.text(cc, rr, txt, ha="center", va="center", fontsize=7.5, color="white" if val > 0.45 else "black")
            a.set_xticks([]); a.set_yticks([])
            if r == 0:
                a.set_title(f"{k} · {'¿está? dónde' if cual == 0 else 'amplitud'}\n", fontsize=9, color=COLOR[k], weight="bold")
            a.set_xlabel(f"cuenta {R[k][2]}" if cual == 0 else "", fontsize=7.5, labelpad=1)
    a = fig.add_subplot(gs[len(idx), :]); a.axis("off")
    filas = [f"VAL ({len(va)} dígitos) · {ms:.1f} ms por dígito · {total} vértices · dígitos con 0/1/2/≥3: "
             f"{(n_vert == 0).sum()}/{(n_vert == 1).sum()}/{(n_vert == 2).sum()}/{(n_vert >= 3).sum()}",
             "por dirección y clase (0…9):"]
    for k in DIRS:
        md = f"{np.median(angs[k]):.0f}°" if angs[k] else "—"
        filas.append(f"  {k}: {sum(por_clase[k]):>4}   " + " ".join(f"{c}:{por_clase[k][c]:<3}" for c in range(10)) + f"   amplitud mediana {md}")
    filas.append(f"el 4 CERRADO (4l, {cuatro['4l']['n']} en val): con vértice N {cuatro['4l']['con_N']} · N en la fila de arriba {cuatro['4l']['con_N_arriba']}   ·   "
                 f"el 4 ABIERTO (4a, {cuatro['4a']['n']}): con N {cuatro['4a']['con_N']} · arriba {cuatro['4a']['con_N_arriba']}      (el de hoy, § 95: 16 de 26 y 0 de 137)")
    filas.append(f"posición única: {borde} de {total} vértices ({100 * borde / max(total, 1):.0f} %) están a < 1 px de un borde de celda — "
                 f"con la mancha encenderían 2 celdas; aquí 1 · {choques} veces dos de la MISMA dirección caen en la misma celda (lo dice la cuenta)")
    filas.append("moviendo la entrada (sin np.roll), % de dígitos cuya rejilla de presencia cambia:  " + "  ".join(f"{k}:{v}%" for k, v in despl.items()))
    a.text(0, 1, "\n".join(filas), va="top", family="monospace", fontsize=9.5)
    fig.suptitle("rizo-2 · 102 · VISTA PREVIA: el vértice LIBRE (en cualquier sitio de la cresta) · 4 direcciones · UNA celda · amplitud = ángulo/180\n"
                 f"vértice = 2 brazos de cresta ≥ {BRAZO_MIN:.0f} px con < {A_MAX:.0f}° a {BRAZO:.0f} px (y < {A_MAX_CORTO:.0f}° a {BRAZO_CORTO:.0f} px: esquina, no curva), sin contar cruces · "
                 "dirección = hacia dónde APUNTA · celdas NO N NE / O C E / SO S SE · ENTRADA: el DELGADO (delg-cnn) · parámetros SIN calibrar", fontsize=10.5, y=0.995)
    out = AQUI / "resultados" / "102-vertice-libre"
    fig.savefig(out.with_suffix(".png"), dpi=100, bbox_inches="tight")
    resumen = {"val": int(len(va)), "ms_por_digito": round(ms, 2), "vertices": total, "por_clase": por_clase,
               "por_celda": {k: dict(zip(CELDAS, por_celda[k])) for k in DIRS}, "amplitud_mediana": {k: (float(np.median(angs[k])) if angs[k] else None) for k in DIRS},
               "digitos_0_1_2_3mas": [int((n_vert == 0).sum()), int((n_vert == 1).sum()), int((n_vert == 2).sum()), int((n_vert >= 3).sum())],
               "a_menos_de_1px_de_borde": borde, "misma_dir_misma_celda": choques, "cuatros": cuatro, "desplazamiento_pct_cambia": despl, "ejemplos": idx,
               "parametros": {"BRAZO": BRAZO, "BRAZO_CORTO": BRAZO_CORTO, "BRAZO_MIN": BRAZO_MIN, "A_MAX": A_MAX, "A_MAX_CORTO": A_MAX_CORTO, "SUPR": SUPR}}
    json.dump(resumen, open(out.with_suffix(".json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in resumen.items() if k not in ("por_celda",)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
