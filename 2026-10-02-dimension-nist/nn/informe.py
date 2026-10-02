#!/usr/bin/env python3
"""El informe de `dim-nist`, del DISCO: lee `nn/pesos/*/summary.json`, aplica el criterio de
`instrucciones/02-criterio.md` y escribe `resultados/RESULTADOS.md` y dos figuras.

    python nn/informe.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
PESOS = AQUI / "pesos"
RES = EXP / "resultados"
ORDEN = ["w8", "w7", "w6", "w5", "w4"]
CONTROL = "w8-de4"
DELTA = 0.01            # un punto porcentual de exactitud (02-criterio.md)
DELTA_CE = 0.05         # en entropia cruzada (nats), para la descomposicion que no satura
PISO_ESCRITO = 0.1      # azar entre 10 clases


def leer() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for f in sorted(PESOS.glob("*/summary.json")):
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"⚠ {f}: no se puede leer ({e}); lo salto"); continue
        out.setdefault(s["brazo"], []).append(s)
    return out


def _stats(v):
    a = np.array(v, dtype=np.float64)
    sd = float(a.std(ddof=1)) if len(a) > 1 else float("nan")
    return {"media": float(a.mean()), "sd": sd, "se": sd / math.sqrt(len(a)) if len(a) > 1 else float("nan"), "n": int(len(a))}


def agregar(lista):
    return {"acc_val": _stats([s["acc_val"] for s in lista]), "acc_train": _stats([s["acc_train"] for s in lista]),
            "brecha": _stats([s["brecha"] for s in lista]),
            "ce_val": _stats([s["ce_val"] for s in lista]), "ce_train": _stats([s["ce_train"] for s in lista]),
            "brecha_ce": _stats([s["brecha_ce"] for s in lista]),
            "atascadas": sorted(s["semilla"] for s in lista if s["ce_train"] >= 1.0),
            "parametros": lista[0]["parametros"], "W": lista[0]["W"],
            "n": lista[0]["n"], "semillas": sorted(s["semilla"] for s in lista),
            "segundos": float(np.mean([s["segundos"] for s in lista])),
            "piso": float(np.mean([s.get("piso", PISO_ESCRITO) for s in lista]))}


def umbral(a, b, clave="acc_val", delta=DELTA):
    se = math.sqrt((a[clave]["se"] or 0) ** 2 + (b[clave]["se"] or 0) ** 2)
    return max(2 * se, delta)


def umbral_ce(a, b):
    """Para la descomposicion: la exactitud de train SATURA en 1,0 (medido en el ensayo), asi que
    «cae train» y «crece la brecha» se leen en entropia cruzada, que no satura."""
    return max(umbral(a, b, "ce_train", DELTA_CE), umbral(a, b, "ce_val", DELTA_CE))


def criterio(ag):
    c = {"aprendio": {}, "clasificacion": {}, "avisos": []}
    piso = next(iter(ag.values()))["piso"] if ag else PISO_ESCRITO
    c["piso"] = piso
    presentes = [w for w in ORDEN if w in ag]
    for w in presentes:
        a = ag[w]; se = a["acc_val"]["se"] if not math.isnan(a["acc_val"]["se"]) else 0.0
        c["aprendio"][w] = (a["acc_val"]["media"] - piso) > 2 * se
    aprendidos = [w for w in presentes if c["aprendio"][w]]
    if any(ag[w]["acc_val"]["n"] < 2 for w in presentes):
        c["avisos"].append("hay brazos con UNA sola semilla: su SE no existe y el umbral cae a δ")
    if not aprendidos:
        c["avisos"].append("ningun brazo supera el piso en 2·SE"); return c
    w_est = max(aprendidos, key=lambda w: ag[w]["acc_val"]["media"]); c["W_estrella"] = w_est
    suf = [w for w in aprendidos if ag[w_est]["acc_val"]["media"] - ag[w]["acc_val"]["media"] <= umbral(ag[w_est], ag[w])]
    c["suficientes"] = suf
    c["W_min_suficiente"] = min(suf, key=lambda w: ag[w]["W"])
    c["estorba"] = ("w8" in aprendidos and w_est != "w8"
                    and ag["w8"]["acc_val"]["media"] < ag[w_est]["acc_val"]["media"] - umbral(ag["w8"], ag[w_est]))
    wm = c["W_min_suficiente"]
    c["brecha_crece"] = ("w8" in aprendidos and wm != "w8"
                         and (ag["w8"]["brecha_ce"]["media"] - ag[wm]["brecha_ce"]["media"]) > umbral_ce(ag["w8"], ag[wm]))
    c["atascadas"] = {w: ag[w]["atascadas"] for w in presentes + ([CONTROL] if CONTROL in ag else []) if ag[w]["atascadas"]}
    for w in presentes:
        if w == w_est:
            c["clasificacion"][w] = "W*"; continue
        if not c["aprendio"][w]:
            c["clasificacion"][w] = "no aprendio (piso)"; continue
        u = umbral(ag[w_est], ag[w])
        uce = umbral_ce(ag[w_est], ag[w])
        cae_val = ag[w_est]["acc_val"]["media"] - ag[w]["acc_val"]["media"] > u
        # en entropia cruzada: «cae train» = sube la CE de train; «crece la brecha» = sube ce_val - ce_train
        cae_train = ag[w]["ce_train"]["media"] - ag[w_est]["ce_train"]["media"] > uce
        brecha_crece = ag[w]["brecha_ce"]["media"] - ag[w_est]["brecha_ce"]["media"] > uce
        if not cae_val:
            c["clasificacion"][w] = "ninguna: dentro del umbral de W*"
        elif cae_train and not brecha_crece:
            c["clasificacion"][w] = "(a) menos informacion: train tambien cae, la brecha no crece"
        elif brecha_crece and not cae_train:
            c["clasificacion"][w] = ("(b) peor generalizacion — (b) o (c): lo separa el control, ver abajo"
                                     if CONTROL in ag else "(b) peor generalizacion — sin control: (b) o (c), sin cerrar")
        elif cae_train and brecha_crece:
            c["clasificacion"][w] = "mixta: cae train Y crece la brecha"
        else:
            c["clasificacion"][w] = "mixta: ninguna de las dos llega sola al umbral"
    if CONTROL in ag and "w4" in ag and "w8" in ag:
        ctl, w4, w8 = ag[CONTROL], ag["w4"], ag["w8"]
        cerca4 = abs(ctl["acc_val"]["media"] - w4["acc_val"]["media"]) <= umbral(ctl, w4)
        cerca8 = abs(ctl["acc_val"]["media"] - w8["acc_val"]["media"]) <= umbral(ctl, w8)
        den = w8["acc_val"]["media"] - w4["acc_val"]["media"]
        frac = (ctl["acc_val"]["media"] - w4["acc_val"]["media"]) / den if abs(den) > 1e-9 else float("nan")
        if cerca4 and cerca8:
            lectura = "indistinguible de los dos (w8 y w4 estan dentro del umbral entre si)"
        elif cerca4:
            lectura = "≈ w4: los parametros NO explican w8 − w4; es INFORMACION"
        elif cerca8:
            lectura = "≈ w8: lo explican los PARAMETROS, y 4 px de informacion bastaban"
        else:
            lectura = f"en medio: fraccion {frac:.2f} del camino de w4 a w8 (se reporta, no se redondea)"
        c["control"] = {"lectura": lectura, "fraccion": frac,
                        "brecha_ce_ctl_vs_w4": ctl["brecha_ce"]["media"] - w4["brecha_ce"]["media"], "umbral_brecha_ce": umbral_ce(ctl, w4)}
    return c


def figuras(ag):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ sin matplotlib: no hay figuras"); return []
    RES.mkdir(parents=True, exist_ok=True)
    pres = [w for w in ORDEN if w in ag]; xs = [ag[w]["W"] for w in pres]
    sd = lambda d: 0 if math.isnan(d["sd"]) else d["sd"]
    hechas = []
    for clave, nombre, ylabel in (("acc", "acc-vs-w.png", "exactitud"), ("brecha", "brecha-vs-w.png", "brecha = acc_train − acc_val")):
        fig, ax = plt.subplots(figsize=(7, 4.2))
        if clave == "acc":
            for serie, et, m in (("acc_val", "val (1617, no vistas)", "o"), ("acc_train", "train (180)", "s")):
                ax.errorbar(xs, [ag[w][serie]["media"] for w in pres], yerr=[sd(ag[w][serie]) for w in pres], marker=m, capsize=3, label=et)
            if CONTROL in ag:
                ax.errorbar([8.3], [ag[CONTROL]["acc_val"]["media"]], yerr=[sd(ag[CONTROL]["acc_val"])], marker="^", capsize=3, label="control w8-de4 (val)")
            ax.axhline(next(iter(ag.values()))["piso"], ls=":", color="gray", label="piso: clase mayoritaria")
        else:
            ax.errorbar(xs, [ag[w]["brecha"]["media"] for w in pres], yerr=[sd(ag[w]["brecha"]) for w in pres], marker="o", capsize=3, label="brecha")
            if CONTROL in ag:
                ax.errorbar([8.3], [ag[CONTROL]["brecha"]["media"]], yerr=[sd(ag[CONTROL]["brecha"])], marker="^", capsize=3, label="control w8-de4")
        ax.set_xticks(xs); ax.set_xlabel("W (lado de la imagen, px)"); ax.set_ylabel(ylabel)
        ax.set_title(f"dim-nist: {ylabel} contra la resolucion (media ± sd entre semillas)"); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(RES / nombre, dpi=130); plt.close(fig); hechas.append(nombre)
    return hechas


def md(ag, c, figs):
    f4 = lambda x: "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"
    L = [f"# Resultados de `dim-nist` — generado por `nn/informe.py` el {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", "",
         "**Generado del disco (`nn/pesos/*/summary.json`); no se edita a mano.** Criterio: `instrucciones/02-criterio.md`, escrito antes de entrenar.", "",
         f"Piso (clase mayoritaria de train sobre las 1617): **{c['piso']:.4f}**. δ = {DELTA}. Umbral entre dos brazos = max(2·SE_dif, δ).", "",
         f"Descomposición (a)/(b) en **entropía cruzada** (la exactitud de train satura): δ_ce = {DELTA_CE} nats.", "",
         "| brazo | W | n | parámetros | semillas | exactitud val (media ± sd) | exactitud train | brecha | CE train | CE val | atascadas | ¿aprendió? | clasificación |",
         "|---|---:|---:|---:|---:|---|---|---|---|---|---|---|---|"]
    for w in ORDEN + [CONTROL]:
        if w not in ag:
            L.append(f"| `{w}` | — | — | — | **falta** | — | — | — | — | — | — | — | — |"); continue
        a = ag[w]
        L.append(f"| `{w}` | {a['W']} | {a['n']} | {a['parametros']:,} | {a['acc_val']['n']} ({','.join(map(str, a['semillas']))}) "
                 f"| {f4(a['acc_val']['media'])} ± {f4(a['acc_val']['sd'])} | {f4(a['acc_train']['media'])} | {a['brecha']['media']:+.4f} ± {f4(a['brecha']['sd'])} "
                 f"| {f4(a['ce_train']['media'])} | {f4(a['ce_val']['media'])} | {','.join(map(str, a['atascadas'])) or '—'} | "
                 f"{'sí' if c['aprendio'].get(w) else ('no' if w in c['aprendio'] else '—')} | {c['clasificacion'].get(w, 'control' if w == CONTROL else '—')} |")
    L += ["", "## Las dos preguntas del encargo", ""]
    if "W_estrella" in c:
        L += [f"- **W\\*** (mayor exactitud val): `{c['W_estrella']}`.",
              f"- **W mínimo suficiente**: `{c['W_min_suficiente']}` (suficientes: {', '.join(c['suficientes'])}).",
              f"- **¿La resolución estorba?** {'**SÍ**' if c['estorba'] else 'no'}.",
              f"- **¿La brecha crece con la resolución?** {'**SÍ**' if c['brecha_crece'] else 'no'} (en entropía cruzada: brecha_ce(8) − brecha_ce(W mín. suficiente) contra el umbral)."]
        if c.get("atascadas"):
            L.append(f"- ⚠ **Semillas atascadas** (CE de train final ≥ 1,0): {c['atascadas']}.")
    else:
        L.append("- Sin brazos que superen el piso: no hay comparaciones.")
    if "control" in c:
        k = c["control"]
        L += ["", "## El control `w8-de4` (parámetros de w8, información de w4)", "", f"- Lectura: **{k['lectura']}**.",
              f"- Brecha (CE) del control − brecha (CE) de w4: {k['brecha_ce_ctl_vs_w4']:+.4f} (umbral {k['umbral_brecha_ce']:.4f}): "
              f"{'más parámetros a igual información memorizan MÁS' if k['brecha_ce_ctl_vs_w4'] > k['umbral_brecha_ce'] else 'sin diferencia distinguible'}."]
    if c["avisos"]:
        L += ["", "## Avisos", ""] + [f"- ⚠ {a}" for a in c["avisos"]]
    faltan = [w for w in ORDEN + [CONTROL] if w not in ag]
    if faltan:
        L += ["", f"⚠ **Faltan brazos**: {', '.join(faltan)}. Lo de arriba es parcial."]
    if figs:
        L += ["", "## Figuras", ""] + [f"![{f}]({f})" for f in figs]
    L += ["", "## Por dígito (exactitud val media entre semillas)", ""]
    por = leer()
    for w in ORDEN + [CONTROL]:
        if w not in por:
            continue
        cl = {}
        for s in por[w]:
            for k, v in s.get("acc_val_por_clase", {}).items():
                cl.setdefault(k, []).append(v["acc"])
        L.append(f"- `{w}`: " + " · ".join(f"{k} {np.mean(v):.3f}" for k, v in sorted(cl.items())))
    return "\n".join(L) + "\n"


def main() -> int:
    por = leer()
    if not por:
        print("no hay ningun summary.json en nn/pesos/: nada que informar"); return 1
    ag = {w: agregar(l) for w, l in por.items()}
    c = criterio(ag); figs = figuras(ag)
    RES.mkdir(parents=True, exist_ok=True)
    texto = md(ag, c, figs)
    (RES / "RESULTADOS.md").write_text(texto, encoding="utf-8")
    (RES / "criterio-aplicado.json").write_text(json.dumps(c, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(texto); print(f"→ {RES / 'RESULTADOS.md'}" + (f", figuras: {', '.join(figs)}" if figs else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
