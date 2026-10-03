#!/usr/bin/env python3
"""El informe de `ruido-comb`, del DISCO: Δ pareado por semilla contra `limpio` y contra el mejor
simple (`gaussiano@0.2-linea`), según `instrucciones/02-criterio.md`. Escribe resultados/RESULTADOS.md,
criterio-aplicado.json y delta.png.

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
import datos                                   # noqa: E402

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
PESOS = AQUI / "pesos"
RES = EXP / "resultados"
BASE = "limpio"
MEJOR_SIMPLE = "gaussiano@0.2-linea"
SIMPLES = ("gaussiano@0.2-linea", "recorte@0.6-linea")
COMBOS = ("recorte@0.6+gaussiano@0.2-linea", "recorte@0.6~gaussiano@0.2-linea")
ORDEN = (BASE,) + SIMPLES + COMBOS
DELTA = 0.01
FIGURA = "delta.png"


def leer() -> dict[str, dict[int, dict]]:
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


def _umbral(st):
    se = 0.0 if math.isnan(st["se"]) else st["se"]
    return max(2 * se, DELTA)


def pareado(por, a, b, clave="acc_val"):
    """Δ = a − b por semilla común."""
    comunes = sorted(set(por.get(a, {})) & set(por.get(b, {})))
    if not comunes:
        return None
    return _stats([por[a][s][clave] - por[b][s][clave] for s in comunes])


def criterio(por) -> dict:
    c = {"avisos": [], "abs": {}, "vs_limpio": {}, "vs_mejor": {}, "vs_recorte": {}}
    for esc in ORDEN:
        if esc not in por:
            c["avisos"].append(f"falta `{esc}`"); continue
        sems = por[esc]
        if len(sems) < 3:
            c["avisos"].append(f"`{esc}`: {len(sems)} semilla(s), no 3")
        copias = {s["huella_copia"] for s in sems.values()}
        if len(copias) > 1:
            c["avisos"].append(f"`{esc}`: sus semillas NO comparten la copia ({', '.join(sorted(copias))})")
        vals = {s["huella_x_val"] for s in sems.values()}
        if vals != {datos.HUELLA_X_VAL}:
            c["avisos"].append(f"`{esc}`: la val no es la congelada")
        c["abs"][esc] = {k: _stats([s[k] for s in sems.values()]) for k in ("acc_val", "ce_val", "acc_train", "ce_train")}
        c["abs"][esc]["huella_copia"] = sorted(copias)[0]
    for esc in SIMPLES + COMBOS:
        d = pareado(por, esc, BASE)
        if d:
            u = _umbral(d)
            c["vs_limpio"][esc] = {"delta": d, "ce": pareado(por, esc, BASE, "ce_val"), "umbral": u,
                                   "veredicto": "ayuda" if d["media"] > u else ("perjudica" if d["media"] < -u else "indistinguible")}
    for esc in COMBOS + ("recorte@0.6-linea",):
        d = pareado(por, esc, MEJOR_SIMPLE)
        if d:
            u = _umbral(d)
            c["vs_mejor"][esc] = {"delta": d, "ce": pareado(por, esc, MEJOR_SIMPLE, "ce_val"), "umbral": u,
                                  "veredicto": "SUMAN" if d["media"] > u else ("RESTAN" if d["media"] < -u else "indistinguible")}
    for esc in COMBOS:
        d = pareado(por, esc, "recorte@0.6-linea")
        if d:
            c["vs_recorte"][esc] = {"delta": d, "umbral": _umbral(d)}
    cands = {e: v for e, v in c["vs_mejor"].items() if e in COMBOS}
    if cands:
        mejor = max(cands, key=lambda e: cands[e]["delta"]["media"])
        c["mas_cerca_de_sumar"] = {"escenario": mejor, "delta": cands[mejor]["delta"]["media"], "veredicto": cands[mejor]["veredicto"]}
    return c


def figura(c) -> list[str]:
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt   # noqa: PLC0415
    except ImportError:
        return []
    filas = [(e, v) for e, v in c["vs_limpio"].items()]
    if not filas:
        return []
    RES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(filas) + 1.8))
    ys = np.arange(len(filas))
    ax.axvline(0, color="#6b7280", lw=1)
    ax.axvspan(-DELTA, DELTA, color="#e5e7eb", alpha=0.6, lw=0, label=f"±δ = {DELTA}")
    med = [v["delta"]["media"] for _, v in filas]; err = [0 if math.isnan(v["delta"]["se"]) else 2 * v["delta"]["se"] for _, v in filas]
    ax.errorbar(med, ys, xerr=err, fmt="o", ms=6, color="#2563eb", capsize=3, label="media ± 2·SE (3 semillas, pareado)")
    for y, (_, v) in zip(ys, filas):
        ax.scatter(v["delta"]["valores"], [y] * len(v["delta"]["valores"]), s=12, color="#2563eb", alpha=0.35, zorder=3)
    if MEJOR_SIMPLE in c["vs_limpio"]:
        ax.axvline(c["vs_limpio"][MEJOR_SIMPLE]["delta"]["media"], color="#dc2626", lw=1, ls="--", label=f"el mejor simple ({MEJOR_SIMPLE})")
    ax.set_yticks(ys); ax.set_yticklabels([e for e, _ in filas], fontsize=8); ax.invert_yaxis()
    ax.set_xlabel("Δ exactitud de val contra `limpio`"); ax.grid(axis="x", alpha=0.3)
    ax.set_title("ruido-comb: ¿suman los dos ruidos?", fontsize=10)
    fig.legend(fontsize=7, loc="lower center", ncol=3, frameon=False); fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(RES / FIGURA, dpi=130); plt.close(fig)
    return [FIGURA]


def md(c, figs) -> str:
    f4 = lambda x: "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"
    s4 = lambda x: "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.4f}"
    L = [f"# Resultados de `ruido-comb` — generado por `nn/informe.py` el {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", "",
         "**Generado del disco; no se edita a mano.** Criterio: `instrucciones/02-criterio.md`, escrito antes. Δ pareado por semilla; umbral = max(2·SE, 0,01).", "",
         "| escenario | n | acc val (media ± sd) | CE val | acc train | **Δ vs `limpio`** ± SE | umbral | veredicto | **Δ vs mejor simple** ± SE | umbral | ¿suman? |",
         "|---|---:|---|---|---|---|---|---|---|---|---|"]
    for esc in ORDEN:
        if esc not in c["abs"]:
            L.append(f"| `{esc}` | 0 | **falta** | | | | | | | | |"); continue
        a = c["abs"][esc]; vl = c["vs_limpio"].get(esc); vm = c["vs_mejor"].get(esc)
        L.append(f"| `{esc}` | {a['acc_val']['n']} | {f4(a['acc_val']['media'])} ± {f4(a['acc_val']['sd'])} | {f4(a['ce_val']['media'])} | {f4(a['acc_train']['media'])} "
                 f"| {('**' + s4(vl['delta']['media']) + '** ± ' + f4(vl['delta']['se'])) if vl else '—'} | {f4(vl['umbral']) if vl else '—'} | {vl['veredicto'] if vl else ('base' if esc == BASE else '—')} "
                 f"| {('**' + s4(vm['delta']['media']) + '** ± ' + f4(vm['delta']['se'])) if vm else ('—' if esc != MEJOR_SIMPLE else 'referencia')} | {f4(vm['umbral']) if vm else '—'} | {vm['veredicto'] if vm else '—'} |")
    L += ["", "## Lo que dice el criterio", ""]
    for esc in COMBOS:
        vm = c["vs_mejor"].get(esc)
        if vm:
            forma = "secuencial" if "+" in esc else "mezcla"
            L.append(f"- **`{esc}`** ({forma}) contra `{MEJOR_SIMPLE}`: Δ = {s4(vm['delta']['media'])} ± {f4(vm['delta']['se'])} (umbral {f4(vm['umbral'])}; CE val {s4(vm['ce']['media'])}) → **{vm['veredicto']}**.")
    if "mas_cerca_de_sumar" in c:
        m = c["mas_cerca_de_sumar"]
        L.append(f"- **La forma más cerca de sumar**: `{m['escenario']}` ({s4(m['delta'])}); veredicto {m['veredicto']}.")
    for esc, v in c["vs_recorte"].items():
        L.append(f"- `{esc}` contra `recorte@0.6-linea`: Δ = {s4(v['delta']['media'])} ± {f4(v['delta']['se'])} (umbral {f4(v['umbral'])}).")
    if c["avisos"]:
        L += ["", "## Avisos", ""] + [f"- ⚠ {a}" for a in c["avisos"]]
    if figs:
        L += ["", "## Figuras", ""] + [f"![{f}]({f})" for f in figs] + ["", "![muestras-ruido.png](muestras-ruido.png)"]
    return "\n".join(L) + "\n"


def main() -> int:
    por = leer()
    if not por:
        print("no hay ningún summary.json en nn/pesos/"); return 1
    c = criterio(por); figs = figura(c)
    RES.mkdir(parents=True, exist_ok=True)
    texto = md(c, figs)
    (RES / "RESULTADOS.md").write_text(texto, encoding="utf-8")
    (RES / "criterio-aplicado.json").write_text(json.dumps(c, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(texto)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
