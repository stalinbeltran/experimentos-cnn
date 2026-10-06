#!/usr/bin/env python3
"""La ANATOMÍA de las confusiones: para los pares dirigidos que más fallan (a → b: un `a` leído como `b`), qué detectores
empujan al compositor hacia `b`. Sin entrenar nada: el compositor de 180 (semilla 1) y sus pesos.

Para cada dígito, la contribución del detector j a la decisión «b antes que a» es Σ_celdas (W_b − W_a)·mapa_j: lo que ese
detector suma al logit de b menos lo que suma al de a. Se compara la media en los `a` MAL leídos con la de los `a` BIEN
leídos: el detector con más diferencia es el que se equivoca. Al lado, en qué fracción de cada grupo se enciende (máximo
del mapa ≥ su umbral) y el grosor de su tinta.

    python nn/pares.py lineas-nada [lineas-nada+norm3 ...]   → resultados/pares.json, resultados/pares-<combo>.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent)); sys.path.insert(0, str(AQUI))
import errores as E                               # noqa: E402
import evaluar as V                               # noqa: E402

RES = AQUI.parent / "resultados"
N_PARES = 4


def r3(v) -> float:
    return round(float(v), 3)


def anatomia(combo: str, dg: dict) -> dict:
    nombre, prepro = combo.rsplit("-", 1)
    bancos = [V.banco(n) for n in nombre.split("+")]
    w = dg["w"]; x = dg["x"][w]; y = dg["y"][w]; tr = dg["train"][w]
    s = np.concatenate([V.mapas(b, V.preparar(x, b["rep"], vi)) for b in bancos for vi in prepro.split("+")], 1)
    nombres = [f"{f}" + (f"·{b['nombre']}" if len(bancos) > 1 else "") + (f"·{vi}" if "+" in prepro else "")
               for b in bancos for vi in prepro.split("+") for f in b["familias"]]
    umb = np.concatenate([b["umbrales"] for b in bancos for _ in prepro.split("+")])
    W = V.ajustar(s[tr].reshape(tr.sum(), -1), y[tr], 1)
    Wt = W.weight.detach().numpy().reshape(10, s.shape[1], 8, 8)
    sv, yv, xv = s[~tr], y[~tr], x[~tr]
    pred = W(torch.from_numpy(sv.reshape(len(sv), -1))).argmax(1).numpy()
    grosor = E.propiedades(xv)["grosor"]
    pres = sv.reshape(len(sv), s.shape[1], -1).max(2) >= umb[None]
    conf = [((yv == a) & (pred == b)).sum() for a in range(10) for b in range(10)]
    orden = sorted(((n, i // 10, i % 10) for i, n in enumerate(conf) if i // 10 != i % 10), reverse=True)[:N_PARES]
    out = {"combo": combo, "errores": int((pred != yv).sum()), "pares": []}
    for n, a, b in orden:
        mal, bien, otro = (yv == a) & (pred == b), (yv == a) & (pred == a), (yv == b) & (pred == b)
        c = ((Wt[b] - Wt[a])[None] * sv).sum((2, 3))                     # (N, detectores): empuje hacia b
        dif = c[mal].mean(0) - c[bien].mean(0)
        top = np.argsort(-dif)[:4]
        out["pares"].append({
            "par": f"{a}→{b}", "n": int(n), "frac_de_errores": r3(n / max(1, (pred != yv).sum())),
            "grosor": {"mal": r3(grosor[mal].mean()), "bien": r3(grosor[bien].mean()), f"los {b}": r3(grosor[otro].mean())},
            "detectores_que_empujan": [{"detector": nombres[j], "empuje_extra": r3(dif[j]),
                                        "encendido": {"mal": r3(pres[mal, j].mean()), "bien": r3(pres[bien, j].mean()),
                                                      f"los {b}": r3(pres[otro, j].mean())}} for j in top],
            "_idx": {"mal": np.flatnonzero(mal)[:8].tolist(), "bien": np.flatnonzero(bien)[:8].tolist(),
                     "otro": np.flatnonzero(otro)[:8].tolist()}})
        p = out["pares"][-1]
        print(f"  {combo:<20} {p['par']} ×{p['n']} ({p['frac_de_errores']:.0%} de los errores) · grosor mal {p['grosor']['mal']} / bien "
              f"{p['grosor']['bien']} / los {b} {p['grosor'][f'los {b}']}", flush=True)
        for d in p["detectores_que_empujan"]:
            e = d["encendido"]
            print(f"      {d['detector']:<26} empuje +{d['empuje_extra']:.2f} · encendido mal {e['mal']:.2f} / bien {e['bien']:.2f} "
                  f"/ los {b} {e[f'los {b}']:.2f}", flush=True)
    figura(out, xv)
    for p in out["pares"]:
        p.pop("_idx")
    return out


def figura(out: dict, xv: np.ndarray) -> None:
    import matplotlib                                                # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                                  # noqa: PLC0415
    SUP, T1, T2 = "#fcfcfb", "#0b0b0b", "#52514e"
    filas = out["pares"]
    fig, axs = plt.subplots(len(filas) * 3, 8, figsize=(8, 1.05 * 3 * len(filas) + 0.4), dpi=100, facecolor=SUP)
    for i, p in enumerate(filas):
        a, b = p["par"].split("→")
        for k, (grupo, rot) in enumerate((("mal", f"{a} leído como {b}"), ("bien", f"{a} bien leído"), ("otro", f"un {b}"))):
            for j in range(8):
                ax = axs[3 * i + k, j]; ax.axis("off")
                idx = p["_idx"][grupo]
                if j < len(idx):
                    ax.imshow(xv[idx[j], 0], cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
            axs[3 * i + k, 0].text(-4, 16, rot, ha="right", va="center", fontsize=8, color=T1 if k == 0 else T2)
    fig.suptitle(f"Las confusiones más frecuentes — {out['combo']}", color=T1, fontsize=11, x=0.02, ha="left")
    fig.subplots_adjust(left=0.2, right=0.99, top=0.95, bottom=0.01, wspace=0.05, hspace=0.12)
    fig.savefig(RES / f"pares-{out['combo']}.png", facecolor=SUP); plt.close(fig)


def main(combos: list[str]) -> int:
    torch.set_num_threads(2)
    dg = V._digitos()
    ruta = RES / "pares.json"
    todo = json.loads(ruta.read_text(encoding="utf-8")) if ruta.is_file() else {}
    for combo in combos:
        todo[combo] = anatomia(combo, dg)
        RES.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(todo, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
