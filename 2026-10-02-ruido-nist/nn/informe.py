#!/usr/bin/env python3
"""El informe de `ruido-nist`, del DISCO: lee `nn/pesos/*/summary.json`, aplica el criterio de
`instrucciones/02-criterio.md` (Δ PAREADO por semilla contra `limpio`) y escribe
`resultados/RESULTADOS.md`, `resultados/criterio-aplicado.json` y la figura `delta-por-tipo.png`.

    python nn/informe.py

Antes de comparar nada comprueba lo que el diseño promete: las tres semillas de un escenario
comparten la MISMA copia (huella), todos los escenarios de una semilla comparten los MISMOS pesos
iniciales, y la val es la congelada. Si algo de eso no casa, lo dice en voz alta y lo aparta.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos                                   # noqa: E402
import ruido                                   # noqa: E402

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
PESOS = AQUI / "pesos"
RES = EXP / "resultados"
BASE = ruido.LIMPIO
DELTA = 0.01            # un punto de exactitud = 16 imágenes de 1617 (02-criterio.md)
DELTA_CE = 0.05         # nats, para leer el mecanismo en entropía cruzada de train (no satura)
FIGURA = "delta-por-tipo.png"


def leer() -> dict[str, dict[int, dict]]:
    """{escenario: {semilla: summary}}."""
    out: dict[str, dict[int, dict]] = {}
    for f in sorted(PESOS.glob("*/summary.json")):
        try:
            s = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(f"⚠ {f}: no se puede leer ({e}); lo salto"); continue
        out.setdefault(s["escenario"], {})[int(s["semilla"])] = s
    return out


def _stats(v):
    a = np.array(v, dtype=np.float64)
    sd = float(a.std(ddof=1)) if len(a) > 1 else float("nan")
    return {"media": float(a.mean()), "sd": sd, "se": sd / math.sqrt(len(a)) if len(a) > 1 else float("nan"),
            "n": int(len(a)), "valores": [round(float(x), 5) for x in a]}


def _umbral(st, delta):
    se = 0.0 if math.isnan(st["se"]) else st["se"]
    return max(2 * se, delta)


def integridad(por) -> list[str]:
    """Lo que el diseño promete, comprobado contra lo que hay en disco."""
    avisos = []
    for esc, sems in por.items():
        copias = {s["huella_copia"] for s in sems.values()}
        if len(copias) > 1:
            avisos.append(f"`{esc}`: sus semillas NO comparten la copia ({', '.join(sorted(copias))}); el Δ pareado mezclaría ruidos")
        vals = {s["huella_x_val"] for s in sems.values()}
        if vals != {datos.HUELLA_X_VAL}:
            avisos.append(f"`{esc}`: la val no es la congelada ({', '.join(sorted(vals))})")
    por_semilla: dict[int, set] = {}
    for sems in por.values():
        for s, r in sems.items():
            por_semilla.setdefault(s, set()).add(r["huella_init"])
    for s, inits in sorted(por_semilla.items()):
        if len(inits) > 1:
            avisos.append(f"semilla {s}: hay escenarios con pesos iniciales DISTINTOS ({', '.join(sorted(inits))})")
    return avisos


def deltas(por) -> dict[str, dict]:
    """Por escenario ≠ limpio: Δ pareado por semilla contra limpio, en las cuatro medidas."""
    base = por.get(BASE, {})
    out = {}
    for esc, sems in por.items():
        if esc == BASE:
            continue
        comunes = sorted(set(sems) & set(base))
        if not comunes:
            out[esc] = {"n": 0, "semillas": sorted(sems), "sin_base": sorted(sems)}; continue
        d = {k: _stats([sems[s][k] - base[s][k] for s in comunes]) for k in ("acc_val", "ce_val", "acc_train", "ce_train")}
        tipo, nivel, r = ruido.parsear(esc)
        d.update({"tipo": tipo, "nivel": nivel, "realizacion": r, "linea": ruido.es_linea(esc), "semillas": comunes, "n": len(comunes),
                  "sin_base": sorted(set(sems) - set(base)),
                  "acc_val_abs": _stats([sems[s]["acc_val"] for s in comunes]),
                  "ce_val_abs": _stats([sems[s]["ce_val"] for s in comunes]),
                  "acc_train_abs": _stats([sems[s]["acc_train"] for s in comunes])})
        u = _umbral(d["acc_val"], DELTA)
        m = d["acc_val"]["media"]
        d["umbral"] = u
        d["veredicto"] = "ayuda" if m > u else ("perjudica" if m < -u else "indistinguible")
        uce = _umbral(d["ce_train"], DELTA_CE)
        mt = d["ce_train"]["media"]
        d["umbral_ce_train"] = uce
        d["mecanismo"] = ("regularización: ajusta PEOR el train limpio" if mt > uce
                          else ("facilitación: ajusta MEJOR el train limpio" if mt < -uce else "train limpio sin cambio distinguible"))
        out[esc] = d
    return out


def criterio(por, dl) -> dict:
    c = {"avisos": integridad(por), "base": {}, "fase2": [], "realizacion": {}}
    base = por.get(BASE, {})
    if not base:
        c["avisos"].append(f"falta `{BASE}`: sin base no hay Δ pareado; abajo sólo valores absolutos")
    else:
        c["base"] = {"acc_val": _stats([r["acc_val"] for r in base.values()]), "ce_val": _stats([r["ce_val"] for r in base.values()]),
                     "acc_train": _stats([r["acc_train"] for r in base.values()]), "ce_train": _stats([r["ce_train"] for r in base.values()]),
                     "semillas": sorted(base)}
    for esc, d in dl.items():
        if d["n"] == 0:
            c["avisos"].append(f"`{esc}`: ninguna de sus semillas {d['semillas']} tiene `limpio`: sin Δ")
        elif d["n"] < 3:
            c["avisos"].append(f"`{esc}`: Δ sobre {d['n']} semilla(s) ({d['semillas']}), no 3: el SE es {'inexistente' if d['n'] < 2 else 'de dos'}")
    # fase 1 → fase 2: los tipos al nivel MEDIO (realización 1) que ayudan o son indistinguibles con media > 0
    for tipo in ruido.TIPOS:
        esc = ruido.escenario(tipo, ruido.NIVELES[tipo][ruido.INDICE_MEDIO])
        d = dl.get(esc)
        if d and d["n"] and (d["veredicto"] == "ayuda" or (d["veredicto"] == "indistinguible" and d["acc_val"]["media"] > 0)):
            c["fase2"].append(tipo)
    c["veredictos"] = {esc: d["veredicto"] for esc, d in dl.items() if d["n"]}
    # fase 2: dentro de cada tipo con 2+ niveles (realización 1), el mejor nivel y la forma de la curva
    c["por_tipo"] = {}
    for tipo in ruido.TIPOS:
        niveles = sorted(((d["nivel"], esc) for esc, d in dl.items() if d["n"] and d.get("tipo") == tipo and d.get("realizacion") == 1 and not d.get("linea")))
        if len(niveles) < 2:
            continue
        medias = [dl[esc]["acc_val"]["media"] for _, esc in niveles]
        mejor = max(niveles, key=lambda ne: dl[ne[1]]["acc_val"]["media"])
        k = medias.index(max(medias))
        if k == 0:
            forma = "cae con la intensidad: el mejor es el más suave"
        elif k == len(medias) - 1:
            forma = "sube con la intensidad: el mejor es el más fuerte (el eje no está acotado por arriba)"
        else:
            forma = "pico interior"
        c["por_tipo"][tipo] = {"niveles": [{"nivel": n, "escenario": esc, "delta": dl[esc]["acc_val"]["media"], "se": dl[esc]["acc_val"]["se"],
                                           "veredicto": dl[esc]["veredicto"]} for n, esc in niveles],
                               "mejor": mejor[1], "mejor_nivel": mejor[0], "mejor_delta": dl[mejor[1]]["acc_val"]["media"],
                               "mejor_veredicto": dl[mejor[1]]["veredicto"], "forma": forma,
                               "amplitud": max(medias) - min(medias)}
    # fase 3: en línea contra su copia fija, pareado por semilla
    c["en_linea"] = {}
    for esc, sems in por.items():
        if not ruido.es_linea(esc):
            continue
        fijo = por.get(ruido.fijo_de(esc), {})
        comunes = sorted(set(sems) & set(fijo))
        if not comunes:
            c["avisos"].append(f"`{esc}`: no está su copia fija `{ruido.fijo_de(esc)}` en las mismas semillas"); continue
        st = _stats([sems[s]["acc_val"] - fijo[s]["acc_val"] for s in comunes])
        stce = _stats([sems[s]["ce_val"] - fijo[s]["ce_val"] for s in comunes])
        u = _umbral(st, DELTA)
        c["en_linea"][esc] = {"contra": ruido.fijo_de(esc), "delta_acc_val": st, "delta_ce_val": stce, "umbral": u,
                              "veredicto_vs_limpio": dl.get(esc, {}).get("veredicto"),
                              "lectura": ("en línea MEJOR que la copia fija" if st["media"] > u else
                                          ("en línea PEOR que la copia fija" if st["media"] < -u else "indistinguible de la copia fija"))}
    # la realización: un escenario con -r2 contra su -r1, pareado por semilla
    for esc, sems in por.items():
        tipo, nivel, r = ruido.parsear(esc)
        if r < 2:
            continue
        uno = por.get(ruido.escenario(tipo, nivel, 1), {})
        comunes = sorted(set(sems) & set(uno))
        if not comunes:
            c["avisos"].append(f"`{esc}`: no está su realización 1 en las mismas semillas"); continue
        st = _stats([sems[s]["acc_val"] - uno[s]["acc_val"] for s in comunes])
        u = _umbral(st, DELTA)
        medias = [d["acc_val"]["media"] for d in dl.values() if d["n"] and d.get("realizacion") == 1 and not d.get("linea")]
        amplitud = (max(medias) - min(medias)) if medias else float("nan")
        c["realizacion"][esc] = {"contra": ruido.escenario(tipo, nivel, 1), "delta_acc_val": st, "umbral": u,
                                 "amplitud_entre_tipos": amplitud,
                                 "pesa": abs(st["media"]) > u,
                                 "lectura": ("la REALIZACIÓN pesa tanto como un tipo: una copia fija no basta → ruido en línea (lo pendiente de S2)"
                                             if abs(st["media"]) > u else
                                             "la realización no se distingue (|Δ| dentro del umbral): una copia fija sirve para esta fase")}
    return c


def figura(dl) -> list[str]:
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt   # noqa: PLC0415
    except ImportError:
        print("⚠ sin matplotlib: no hay figura"); return []
    filas = [(esc, d) for esc, d in dl.items() if d["n"]]
    if not filas:
        return []
    varios = len({(d.get("tipo"), d.get("nivel")) for _, d in filas}) > len({d.get("tipo") for _, d in filas})
    if varios:   # fase 2: agrupado por tipo (orden de la tabla) y nivel, de arriba abajo
        filas.sort(key=lambda t: (-ruido.TIPOS.index(t[1]["tipo"]), -(t[1]["nivel"] or 0), -(t[1].get("realizacion") or 0)))
    else:        # fase 1: ordenado por Δ
        filas.sort(key=lambda t: t[1]["acc_val"]["media"])
    RES.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(10, 0.42 * len(filas) + 1.6), sharey=True)
    ys = np.arange(len(filas))
    for ax, clave, titulo in ((axs[0], "acc_val", "Δ exactitud de val (ruido − limpio)"), (axs[1], "ce_val", "Δ entropía cruzada de val")):
        med = [d[clave]["media"] for _, d in filas]
        err = [0 if math.isnan(d[clave]["se"]) else 2 * d[clave]["se"] for _, d in filas]
        ax.axvline(0, color="#6b7280", lw=1)
        if clave == "acc_val":
            ax.axvspan(-DELTA, DELTA, color="#e5e7eb", alpha=0.6, lw=0, label=f"±δ = {DELTA}")
        ax.errorbar(med, ys, xerr=err, fmt="o", ms=6, color="#2563eb", ecolor="#2563eb", elinewidth=1.5, capsize=3, label="media ± 2·SE (3 semillas, pareado)")
        for y, (esc, d) in zip(ys, filas):
            ax.scatter(d[clave]["valores"], [y] * d["n"], s=12, color="#2563eb", alpha=0.35, zorder=3)
        ax.set_title(titulo, fontsize=10); ax.grid(axis="x", alpha=0.3); ax.tick_params(labelsize=8)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="lower center", ncol=2, frameon=False)
    axs[0].set_yticks(ys); axs[0].set_yticklabels([esc for esc, _ in filas], fontsize=8)
    fig.suptitle("ruido-nist: Δ pareado por semilla contra `limpio` (los puntos tenues son las semillas)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06 if len(filas) < 15 else 0.025, 1, 1 - 1.2 / fig.get_figheight())); fig.savefig(RES / FIGURA, dpi=130); plt.close(fig)
    return [FIGURA]


def md(por, dl, c, figs) -> str:
    f4 = lambda x: "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"
    s4 = lambda x: "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.4f}"
    L = [f"# Resultados de `ruido-nist` — generado por `nn/informe.py` el {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", "",
         "**Generado del disco (`nn/pesos/*/summary.json`); no se edita a mano.** Criterio: `instrucciones/02-criterio.md`, escrito antes de entrenar.", "",
         f"Δ = medida(ruido, s) − medida(`limpio`, s), **pareado por semilla** (mismos pesos iniciales, mismo orden de lotes), media de las semillas y SE = sd/√n. "
         f"Umbral = max(2·SE, δ) con δ = {DELTA} en exactitud; mecanismo leído en la entropía cruzada de las 180 de train limpias con δ_ce = {DELTA_CE} nats.", ""]
    if c["base"]:
        b = c["base"]
        L += [f"**Base `limpio`** (semillas {b['semillas']}): exactitud val **{f4(b['acc_val']['media'])} ± {f4(b['acc_val']['sd'])}**, "
              f"CE val {f4(b['ce_val']['media'])}, exactitud train (180) {f4(b['acc_train']['media'])}, CE train {f4(b['ce_train']['media'])}.", ""]
    L += ["| escenario | tipo | nivel | n | acc val | **Δ acc val** ± SE | umbral | Δ CE val | Δ acc train | Δ CE train | veredicto | mecanismo |",
          "|---|---|---:|---:|---|---|---|---|---|---|---|---|"]
    orden = sorted(dl, key=lambda e: (ruido.TIPOS.index(dl[e]["tipo"]) if dl[e].get("tipo") in ruido.TIPOS else 99, dl[e].get("nivel") or 0, dl[e].get("realizacion") or 0))
    for esc in orden:
        d = dl[esc]
        if not d["n"]:
            L.append(f"| `{esc}` | — | — | 0 | — | **sin base** | — | — | — | — | — | — |"); continue
        L.append(f"| `{esc}` | {d['tipo']} | {d['nivel']:g} | {d['n']} | {f4(d['acc_val_abs']['media'])} ± {f4(d['acc_val_abs']['sd'])} "
                 f"| **{s4(d['acc_val']['media'])}** ± {f4(d['acc_val']['se'])} | {f4(d['umbral'])} | {s4(d['ce_val']['media'])} | {s4(d['acc_train']['media'])} "
                 f"| {s4(d['ce_train']['media'])} | **{d['veredicto']}** | {d['mecanismo']} |")
    ayudan = [e for e, d in dl.items() if d["n"] and d["veredicto"] == "ayuda"]
    perj = [e for e, d in dl.items() if d["n"] and d["veredicto"] == "perjudica"]
    L += ["", "## Lo que dice el criterio", ""]
    L.append(f"- **Ayudan** (media Δ > umbral): {', '.join(f'`{e}` ({s4(dl[e]['acc_val']['media'])})' for e in ayudan) if ayudan else '**ninguno**'}.")
    L.append(f"- **Perjudican**: {', '.join(f'`{e}` ({s4(dl[e]['acc_val']['media'])})' for e in perj) if perj else 'ninguno'}.")
    L.append(f"- **Pasan a la fase 2** (ayuda, o indistinguible con media > 0, al nivel medio): {', '.join(f'`{t}`' for t in c['fase2']) if c['fase2'] else '**ninguno** — con 180 imágenes y esta red, ningún ruido de la lista mejora la generalización'}.")
    if c.get("por_tipo"):
        L += ["", "## Fase 2: la intensidad dentro de cada tipo (Δ exactitud de val, pareado)", "",
              "| tipo | niveles → Δ | mejor | forma |", "|---|---|---|---|"]
        for tipo, pt in c["por_tipo"].items():
            celdas = " · ".join(f"{n['nivel']:g}: {s4(n['delta'])}{'*' if n['veredicto'] == 'ayuda' else ('†' if n['veredicto'] == 'perjudica' else '')}" for n in pt["niveles"])
            L.append(f"| `{tipo}` | {celdas} | `{pt['mejor']}` ({s4(pt['mejor_delta'])}, **{pt['mejor_veredicto']}**) | {pt['forma']}; amplitud {f4(pt['amplitud'])} |")
        L += ["", "\\* ayuda · † perjudica (por el umbral de cada escenario). «Mejor» es la mayor media; si no es «ayuda», no se distingue de `limpio`."]
        ganan = [(tipo, pt) for tipo, pt in c["por_tipo"].items() if pt["mejor_veredicto"] == "ayuda"]
        L += ["", f"- **Tipos con algún nivel que ayuda**: {', '.join(f'`{pt["mejor"]}` ({s4(pt["mejor_delta"])})' for _, pt in ganan) if ganan else '**ninguno**'}."]
    if c.get("en_linea"):
        L += ["", "## Fase 3: ruido EN LÍNEA (una copia nueva por época) contra la copia fija, pareado", "",
              "| escenario | Δ acc val vs `limpio` | veredicto vs `limpio` | **Δ acc val vs copia fija** ± SE | umbral | Δ CE val vs fija | lectura |",
              "|---|---|---|---|---|---|---|"]
        for esc, r in c["en_linea"].items():
            d = dl.get(esc, {})
            L.append(f"| `{esc}` | {s4(d['acc_val']['media']) if d.get('n') else '—'} | {r['veredicto_vs_limpio'] or '—'} | **{s4(r['delta_acc_val']['media'])}** ± {f4(r['delta_acc_val']['se'])} "
                     f"| {f4(r['umbral'])} | {s4(r['delta_ce_val']['media'])} | {r['lectura']} |")
        mejores = [e for e, r in c["en_linea"].items() if r["lectura"].startswith("en línea MEJOR")]
        peores = [e for e, r in c["en_linea"].items() if r["lectura"].startswith("en línea PEOR")]
        L += ["", f"- **En línea mejor que fija**: {', '.join(f'`{e}`' for e in mejores) if mejores else '**ninguno**'} · **peor**: {', '.join(f'`{e}`' for e in peores) if peores else 'ninguno'}."]
    for esc, r in c["realizacion"].items():
        L.append(f"- **La realización** (`{esc}` contra `{r['contra']}`): Δ = {s4(r['delta_acc_val']['media'])} ± {f4(r['delta_acc_val']['se'])} (umbral {f4(r['umbral'])}; "
                 f"amplitud de las medias entre tipos {f4(r['amplitud_entre_tipos'])}) → {r['lectura']}.")
    if c["avisos"]:
        L += ["", "## Avisos", ""] + [f"- ⚠ {a}" for a in c["avisos"]]
    esperados = [ruido.escenario(t, ruido.NIVELES[t][ruido.INDICE_MEDIO]) for t in ruido.TIPOS] + [BASE, ruido.escenario("oblicua", 0.6, 2)]
    faltan = [e for e in esperados if e not in por or len(por[e]) < 3]
    if faltan:
        L += ["", f"⚠ **Fase 1 incompleta** (falta o le faltan semillas): {', '.join(f'`{e}`' for e in faltan)}. Lo de arriba es parcial."]
    if figs:
        L += ["", "## Figuras", ""] + [f"![{f}]({f})" for f in figs] + ["", "![muestras-ruido.png](muestras-ruido.png)"]
    L += ["", "## Por dígito (exactitud val media entre semillas; Δ respecto de limpio)", ""]
    base_cl = {}
    for s in por.get(BASE, {}).values():
        for k, v in s.get("acc_val_por_clase", {}).items():
            base_cl.setdefault(k, []).append(v["acc"])
    for esc in [BASE] + orden:
        if esc not in por:
            continue
        cl = {}
        for s in por[esc].values():
            for k, v in s.get("acc_val_por_clase", {}).items():
                cl.setdefault(k, []).append(v["acc"])
        if esc == BASE or not base_cl:
            L.append(f"- `{esc}`: " + " · ".join(f"{k} {np.mean(v):.3f}" for k, v in sorted(cl.items())))
        else:
            L.append(f"- `{esc}`: " + " · ".join(f"{k} {np.mean(v) - np.mean(base_cl[k]):+.3f}" for k, v in sorted(cl.items()) if k in base_cl))
    return "\n".join(L) + "\n"


def main() -> int:
    por = leer()
    if not por:
        print("no hay ningún summary.json en nn/pesos/: nada que informar"); return 1
    dl = deltas(por)
    c = criterio(por, dl)
    figs = figura(dl)
    RES.mkdir(parents=True, exist_ok=True)
    texto = md(por, dl, c, figs)
    (RES / "RESULTADOS.md").write_text(texto, encoding="utf-8")
    (RES / "criterio-aplicado.json").write_text(json.dumps({"criterio": c, "deltas": dl}, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(texto); print(f"→ {RES / 'RESULTADOS.md'}" + (f", figura: {', '.join(figs)}" if figs else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
