#!/usr/bin/env python3
"""El informe de `dim-gen`, del DISCO: lee `nn/pesos/*/summary.json`, aplica el criterio de
`instrucciones/02-criterio.md` tabla por tabla y escribe `resultados/RESULTADOS.md` y dos
figuras. Nunca se transcribe nada a mano.

    python nn/informe.py

Funciona con lo que haya: los brazos que faltan se dicen, no se inventan.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import modelo                                  # noqa: E402

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
PESOS = AQUI / "pesos"
RES = EXP / "resultados"
ORDEN = ["w128", "w064", "w032", "w016", "w008"]
CONTROL = "w128-de16"
DELTA = 0.01            # instrucciones/02-criterio.md § La resolucion del instrumento
PISO_ESCRITO = 0.2464   # medido el 2026-10-01 con las etiquetas


def leer() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for f in sorted(PESOS.glob("*/summary.json")):
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"⚠ {f}: no se puede leer ({e}); lo salto")
            continue
        out.setdefault(s["brazo"], []).append(s)
    return out


def _stats(v: list[float]) -> dict:
    a = np.array(v, dtype=np.float64)
    sd = float(a.std(ddof=1)) if len(a) > 1 else float("nan")
    return {"media": float(a.mean()), "sd": sd, "se": sd / math.sqrt(len(a)) if len(a) > 1 else float("nan"),
            "n": int(len(a))}


def agregar(lista: list[dict]) -> dict:
    return {"iou_val": _stats([s["iou_val"] for s in lista]),
            "iou_train": _stats([s["iou_train"] for s in lista]),
            "brecha": _stats([s["brecha"] for s in lista]),
            "parametros": lista[0]["parametros"], "W": lista[0]["W"], "n": lista[0]["n"],
            "semillas": sorted(s["semilla"] for s in lista),
            "segundos": float(np.mean([s["segundos"] for s in lista])),
            "piso": float(np.mean([s.get("piso_caja_media", PISO_ESCRITO) for s in lista]))}


def umbral(a: dict, b: dict) -> float:
    se = math.sqrt((a["iou_val"]["se"] or 0) ** 2 + (b["iou_val"]["se"] or 0) ** 2)
    return max(2 * se, DELTA)


def criterio(ag: dict[str, dict]) -> dict:
    """Las reglas de 02-criterio.md, en el mismo orden en que estan escritas."""
    c: dict = {"aprendio": {}, "clasificacion": {}, "avisos": []}
    piso = next(iter(ag.values()))["piso"] if ag else PISO_ESCRITO
    c["piso"] = piso
    presentes = [w for w in ORDEN if w in ag]
    for w in presentes:
        a = ag[w]
        se = a["iou_val"]["se"] if not math.isnan(a["iou_val"]["se"]) else 0.0
        c["aprendio"][w] = (a["iou_val"]["media"] - piso) > 2 * se
    aprendidos = [w for w in presentes if c["aprendio"][w]]
    if len([w for w in presentes if ag[w]["iou_val"]["n"] >= 2]) < len(presentes):
        c["avisos"].append("hay brazos con UNA sola semilla: su SE no existe y el umbral cae a δ")
    if not aprendidos:
        c["avisos"].append("ningun brazo aprendio (no supera el piso en 2·SE): no hay comparaciones")
        return c
    w_est = max(aprendidos, key=lambda w: ag[w]["iou_val"]["media"])
    c["W_estrella"] = w_est
    suficientes = [w for w in aprendidos
                   if ag[w_est]["iou_val"]["media"] - ag[w]["iou_val"]["media"] <= umbral(ag[w_est], ag[w])]
    c["suficientes"] = suficientes
    c["W_min_suficiente"] = min(suficientes, key=lambda w: ag[w]["W"])
    if "w128" in aprendidos and w_est != "w128":
        c["estorba"] = ag["w128"]["iou_val"]["media"] < ag[w_est]["iou_val"]["media"] - umbral(ag["w128"], ag[w_est])
    else:
        c["estorba"] = False
    wm = c["W_min_suficiente"]
    if "w128" in aprendidos and wm != "w128":
        c["brecha_crece"] = (ag["w128"]["brecha"]["media"] - ag[wm]["brecha"]["media"]) > umbral(ag["w128"], ag[wm])
    else:
        c["brecha_crece"] = False
    for w in presentes:
        if w == w_est:
            c["clasificacion"][w] = "W*"
            continue
        if not c["aprendio"][w]:
            c["clasificacion"][w] = "no aprendio (piso)"
            continue
        u = umbral(ag[w_est], ag[w])
        cae_val = ag[w_est]["iou_val"]["media"] - ag[w]["iou_val"]["media"] > u
        cae_train = ag[w_est]["iou_train"]["media"] - ag[w]["iou_train"]["media"] > u
        brecha_crece = ag[w]["brecha"]["media"] - ag[w_est]["brecha"]["media"] > u
        if not cae_val:
            c["clasificacion"][w] = "ninguna: dentro del umbral de W*"
        elif cae_train and not brecha_crece:
            c["clasificacion"][w] = "(a) menos informacion: train tambien cae, la brecha no crece"
        elif brecha_crece and not cae_train:
            # (b) y (c) solo los separa el control: si existe, se remite a su lectura; si no, queda
            # «sin cerrar (confundido)», que es la etiqueta de ESTADO.md para este mismo confound.
            c["clasificacion"][w] = ("(b) peor generalizacion — (b) o (c): lo separa el control, ver abajo"
                                     if CONTROL in ag else
                                     "(b) peor generalizacion — sin control: (b) o (c), sin cerrar")
        elif cae_train and brecha_crece:
            c["clasificacion"][w] = "mixta: cae train Y crece la brecha"
        else:
            c["clasificacion"][w] = "mixta: ninguna de las dos llega sola al umbral"
    if CONTROL in ag and "w016" in ag and "w128" in ag:
        ctl, w16, w128 = ag[CONTROL], ag["w016"], ag["w128"]
        cerca16 = abs(ctl["iou_val"]["media"] - w16["iou_val"]["media"]) <= umbral(ctl, w16)
        cerca128 = abs(ctl["iou_val"]["media"] - w128["iou_val"]["media"]) <= umbral(ctl, w128)
        den = w128["iou_val"]["media"] - w16["iou_val"]["media"]
        frac = (ctl["iou_val"]["media"] - w16["iou_val"]["media"]) / den if abs(den) > 1e-9 else float("nan")
        if cerca16 and cerca128:
            lectura = "indistinguible de los dos (w128 y w016 estan dentro del umbral entre si)"
        elif cerca16:
            lectura = "≈ w016: los parametros NO explican w128 − w016; es INFORMACION"
        elif cerca128:
            lectura = "≈ w128: lo explican los PARAMETROS, y 16 px de informacion bastaban"
        else:
            lectura = f"en medio: fraccion {frac:.2f} del camino de w016 a w128 (se reporta, no se redondea)"
        c["control"] = {"lectura": lectura, "fraccion": frac,
                        "brecha_ctl_vs_w016": ctl["brecha"]["media"] - w16["brecha"]["media"],
                        "umbral_brecha": umbral(ctl, w16)}
    return c


def figuras(ag: dict[str, dict]) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ sin matplotlib: no hay figuras")
        return []
    RES.mkdir(parents=True, exist_ok=True)
    presentes = [w for w in ORDEN if w in ag]
    xs = [ag[w]["W"] for w in presentes]
    hechas = []
    for clave, nombre, ylabel in (("iou", "iou-vs-w.png", "IoU"), ("brecha", "brecha-vs-w.png", "brecha = IoU_train − IoU_val")):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        if clave == "iou":
            for serie, etiqueta, marca in (("iou_val", "val (900, no vistas)", "o"), ("iou_train", "train (100)", "s")):
                ax.errorbar(xs, [ag[w][serie]["media"] for w in presentes],
                            yerr=[0 if math.isnan(ag[w][serie]["sd"]) else ag[w][serie]["sd"] for w in presentes],
                            marker=marca, capsize=3, label=etiqueta)
            if CONTROL in ag:
                ax.errorbar([128 * 1.18], [ag[CONTROL]["iou_val"]["media"]],
                            yerr=[0 if math.isnan(ag[CONTROL]["iou_val"]["sd"]) else ag[CONTROL]["iou_val"]["sd"]],
                            marker="^", capsize=3, label="control w128-de16 (val)")
            ax.axhline(next(iter(ag.values()))["piso"], ls=":", color="gray", label="piso: caja media")
        else:
            ax.errorbar(xs, [ag[w]["brecha"]["media"] for w in presentes],
                        yerr=[0 if math.isnan(ag[w]["brecha"]["sd"]) else ag[w]["brecha"]["sd"] for w in presentes],
                        marker="o", capsize=3, label="brecha")
            if CONTROL in ag:
                ax.errorbar([128 * 1.18], [ag[CONTROL]["brecha"]["media"]],
                            yerr=[0 if math.isnan(ag[CONTROL]["brecha"]["sd"]) else ag[CONTROL]["brecha"]["sd"]],
                            marker="^", capsize=3, label="control w128-de16")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel("W (lado de la imagen, px)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"dim-gen: {ylabel} contra la resolucion (media ± sd entre semillas)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(RES / nombre, dpi=130)
        plt.close(fig)
        hechas.append(nombre)
    return hechas


def md(ag: dict[str, dict], c: dict, figs: list[str]) -> str:
    def f4(x):
        return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"
    lineas = [f"# Resultados de `dim-gen` — generado por `nn/informe.py` el "
              f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", "",
              "**Generado del disco (`nn/pesos/*/summary.json`); no se edita a mano.** El criterio "
              "aplicado es el de `instrucciones/02-criterio.md`, escrito antes de entrenar.", "",
              f"Piso (caja media de train sobre las 900): **{c['piso']:.4f}**. δ = {DELTA}. "
              f"Umbral entre dos brazos = max(2·SE_dif, δ).", "",
              "| brazo | W | n | parámetros | semillas | IoU val (media ± sd) | IoU train | brecha | ¿aprendió? | clasificación |",
              "|---|---:|---:|---:|---:|---|---|---|---|---|"]
    for w in ORDEN + [CONTROL]:
        if w not in ag:
            lineas.append(f"| `{w}` | — | — | — | **falta** | — | — | — | — | — |")
            continue
        a = ag[w]
        lineas.append(
            f"| `{w}` | {a['W']} | {a['n']} | {a['parametros']:,} | {a['iou_val']['n']} ({','.join(map(str, a['semillas']))}) "
            f"| {f4(a['iou_val']['media'])} ± {f4(a['iou_val']['sd'])} | {f4(a['iou_train']['media'])} "
            f"| {a['brecha']['media']:+.4f} ± {f4(a['brecha']['sd'])} | "
            f"{'sí' if c['aprendio'].get(w) else ('no' if w in c['aprendio'] else '—')} | "
            f"{c['clasificacion'].get(w, 'control' if w == CONTROL else '—')} |")
    lineas += ["", "## Las dos preguntas del encargo", ""]
    if "W_estrella" in c:
        lineas += [f"- **W\\*** (mayor IoU val): `{c['W_estrella']}`.",
                   f"- **W mínimo suficiente**: `{c['W_min_suficiente']}` (suficientes: {', '.join(c['suficientes'])}).",
                   f"- **¿La resolución estorba?** {'**SÍ**' if c['estorba'] else 'no'}: "
                   f"{'w128 queda por debajo de W* más del umbral' if c['estorba'] else 'w128 no queda por debajo de W* más del umbral'}.",
                   f"- **¿La brecha crece con la resolución?** {'**SÍ**' if c['brecha_crece'] else 'no'} "
                   f"(brecha(128) − brecha(W mínimo suficiente) contra el umbral)."]
    else:
        lineas.append("- Sin brazos que superen el piso: no hay comparaciones.")
    if "control" in c:
        k = c["control"]
        lineas += ["", "## El control `w128-de16` (parámetros de w128, información de w016)", "",
                   f"- Lectura: **{k['lectura']}**.",
                   f"- Brecha del control − brecha de w016: {k['brecha_ctl_vs_w016']:+.4f} "
                   f"(umbral {k['umbral_brecha']:.4f}): "
                   f"{'más parámetros a igual información memorizan MÁS' if k['brecha_ctl_vs_w016'] > k['umbral_brecha'] else 'sin diferencia distinguible'}."]
    if c["avisos"]:
        lineas += ["", "## Avisos", ""] + [f"- ⚠ {a}" for a in c["avisos"]]
    faltan = [w for w in ORDEN + [CONTROL] if w not in ag]
    if faltan:
        lineas += ["", f"⚠ **Faltan brazos**: {', '.join(faltan)}. Lo de arriba es parcial."]
    if figs:
        lineas += ["", "## Figuras", ""] + [f"![{f}]({f})" for f in figs]
    lineas += ["", "## Por factor del generador (IoU val medio por brazo)", ""]
    for w in ORDEN + [CONTROL]:
        if w not in ag:
            continue
        # el desglose se promedia entre semillas
        lista = [s for s in leer().get(w, [])]
        if not lista:
            continue
        fu = {}
        for s in lista:
            for nombre, v in s.get("iou_val_por_factor", {}).get("fuente", {}).items():
                fu.setdefault(nombre, []).append(v["iou"])
        lineas.append(f"- `{w}`: " + " · ".join(f"{k} {np.mean(v):.3f}" for k, v in sorted(fu.items())))
    return "\n".join(lineas) + "\n"


def main() -> int:
    por_brazo = leer()
    if not por_brazo:
        print("no hay ningun summary.json en nn/pesos/: nada que informar")
        return 1
    ag = {w: agregar(l) for w, l in por_brazo.items()}
    c = criterio(ag)
    figs = figuras(ag)
    RES.mkdir(parents=True, exist_ok=True)
    texto = md(ag, c, figs)
    (RES / "RESULTADOS.md").write_text(texto, encoding="utf-8")
    (RES / "criterio-aplicado.json").write_text(json.dumps(c, indent=1, ensure_ascii=False, default=str) + "\n",
                                                 encoding="utf-8")
    print(texto)
    print(f"→ {RES / 'RESULTADOS.md'}" + (f", figuras: {', '.join(figs)}" if figs else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
