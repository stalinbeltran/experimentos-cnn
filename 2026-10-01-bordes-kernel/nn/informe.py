#!/usr/bin/env python3
"""El informe de `bor-k`: aplica el criterio de `instrucciones/02-criterio.md` TAL CUAL.

    python nn/informe.py          lee nn/pesos/*/best.pt, escribe resultados/informe.json,
                                  resultados/INFORME.md y resultados/kernels.png

QUE DECIDE, y es lo unico que decide (el criterio se escribio antes de entrenar):
  - por BRAZO y por BORDE, sobre `eval`: el borde esta APRENDIDO si
        media(e_suelo - e_brazo) > 2 * SE
    en las ventanas donde ese borde existe, con `e` el error absoluto de cada ventana y
    SE el error estandar de esa diferencia PAREADA. El suelo es el predictor trivial:
    la coordenada mediana de `train` de ese borde.
  - por `k`: aprendido si lo aprenden las TRES semillas; «inestable» si dos; si no, no.
NUNCA el promedio de bordes: el riesgo escrito antes es que el kernel se quede con un eje.

Lo que NO decide: si el kernel es util. Eso lo dice el banco.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

import datos
from modelo import BORDES, BRAZOS, construir

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
PESOS = AQUI / "pesos"
RESULTADOS = EXP / "resultados"


def _brazo(nombre: str, X: torch.Tensor):
    k, s = BRAZOS[nombre]
    ck = torch.load(PESOS / nombre / "best.pt", map_location="cpu", weights_only=False)
    red = construir(k, s)
    red.load_state_dict(ck["modelo"])
    red.eval()
    with torch.no_grad():
        logits, coords = red(X)
    return red, logits.numpy(), coords.numpy(), int(ck.get("epoca") or 0)


def main() -> int:
    _, Etr, Ctr, _ = datos.ventanas("train")
    Xev, Eev, Cev, _ = datos.ventanas("eval")
    X = torch.from_numpy(Xev)
    hay = Eev > 0.5
    mediana = np.array([np.median(Ctr[Etr[:, i] > 0.5, i]) for i in range(4)], np.float32)
    e_suelo = np.abs(mediana[None, :] - Cev)

    brazos = sorted(b for b in BRAZOS if (PESOS / b / "best.pt").is_file())
    if not brazos:
        print("✗ no hay ningun brazo entrenado en nn/pesos/")
        return 1
    filas, kernels = [], {}
    for b in brazos:
        red, logits, coords, epoca = _brazo(b, X)
        kernels[b] = red.kernel().detach().numpy().copy()
        e_brazo = np.abs(coords - Cev)
        pred = 1 / (1 + np.exp(-logits)) > 0.5
        fila = {"brazo": b, "k": BRAZOS[b][0], "semilla": BRAZOS[b][1], "epoca_best": epoca}
        for i, t in enumerate(BORDES):
            m = hay[:, i]
            dif = e_suelo[m, i] - e_brazo[m, i]
            se = float(dif.std(ddof=1) / math.sqrt(len(dif)))
            tp = int((pred[:, i] & m).sum())
            den = int(pred[:, i].sum() + m.sum())
            fila[t] = {"n": int(m.sum()), "err_px": round(float(e_brazo[m, i].mean()), 3),
                       "err_suelo_px": round(float(e_suelo[m, i].mean()), 3),
                       "a2": round(float((e_brazo[m, i] <= 2).mean()), 4),
                       "f1": round(2 * tp / den, 4) if den else 0.0,
                       "dif_media": round(float(dif.mean()), 3), "se": round(se, 4),
                       "aprendio": bool(dif.mean() > 2 * se)}
        filas.append(fila)

    por_k = {}
    for k in sorted({f["k"] for f in filas}):
        mias = [f for f in filas if f["k"] == k]
        por_k[k] = {}
        for t in BORDES:
            n = sum(f[t]["aprendio"] for f in mias)
            por_k[k][t] = ("aprendido" if n == 3 else "inestable" if n == 2 else "no") \
                if len(mias) == 3 else f"{n}/{len(mias)} semillas"

    RESULTADOS.mkdir(exist_ok=True)
    (RESULTADOS / "informe.json").write_text(json.dumps(
        {"criterio": "instrucciones/02-criterio.md", "particion": "eval",
         "suelo": "coordenada mediana de train por borde",
         "brazos": filas, "por_k": {str(k): v for k, v in por_k.items()}},
        indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    L = ["# `bor-k` — informe (generado por `nn/informe.py`; no se transcribe nada a mano)", "",
         "Por borde, sobre **`eval`**, contra el **suelo trivial** (la coordenada mediana de "
         "`train`). Criterio escrito antes: `instrucciones/02-criterio.md`.", "",
         "## Por `k` (3 semillas): ¿aprendió cada borde?", "",
         "| `k` | izq | der | sup | inf |", "|---:|---|---|---|---|"]
    marca = {"aprendido": "✅ aprendido", "inestable": "⚠ inestable (2/3)", "no": "✗ no"}
    for k, v in por_k.items():
        L.append(f"| {k} | " + " | ".join(marca.get(v[t], v[t]) for t in BORDES) + " |")
    L += ["", "## Por brazo: error medio en px (suelo entre paréntesis) y acierto ≤ 2 px", "",
          "| brazo | época | " + " | ".join(BORDES) + " |", "|---|---:|" + "---|" * 4]
    for f in filas:
        celdas = []
        for t in BORDES:
            d = f[t]
            celdas.append(f"{'**' if d['aprendio'] else ''}{d['err_px']:.2f}"
                          f"{'**' if d['aprendio'] else ''} ({d['err_suelo_px']:.2f}) · "
                          f"{100 * d['a2']:.0f} %")
        L.append(f"| {f['brazo']} | {f['epoca_best']} | " + " | ".join(celdas) + " |")
    L += ["", "En **negrita**, el borde que ese brazo aprendió (dif. pareada > 2·SE).", "",
          "![los 27 kernels](kernels.png)", ""]
    (RESULTADOS / "INFORME.md").write_text("\n".join(L), encoding="utf-8")
    _figura(kernels)
    print("\n".join(L[:6 + len(por_k) + 2]))
    print(f"\nescrito {RESULTADOS / 'INFORME.md'} · {len(filas)} brazo(s)")
    return 0


def _figura(kernels: dict) -> None:
    """Una fila por semilla, una columna por k; cada kernel estirado a su rango
    simetrico (gris = 0, blanco = +, negro = -)."""
    from PIL import Image                                   # noqa: PLC0415
    ks = sorted({BRAZOS[b][0] for b in kernels})
    ss = sorted({BRAZOS[b][1] for b in kernels})
    celda, pad = 19 * 6, 6
    hoja = Image.new("L", (len(ks) * (celda + pad) + pad, len(ss) * (celda + pad) + pad), 255)
    for b, w in kernels.items():
        k, s = BRAZOS[b]
        g = 127.5 + 127.5 * w / max(1e-9, float(np.abs(w).max()))
        im = Image.fromarray(g.round().astype(np.uint8)).resize((celda * k // 19,) * 2, Image.NEAREST)
        x = pad + ks.index(k) * (celda + pad) + (celda - im.width) // 2
        y = pad + ss.index(s) * (celda + pad) + (celda - im.height) // 2
        hoja.paste(im, (x, y))
    hoja.save(RESULTADOS / "kernels.png")


if __name__ == "__main__":
    raise SystemExit(main())
