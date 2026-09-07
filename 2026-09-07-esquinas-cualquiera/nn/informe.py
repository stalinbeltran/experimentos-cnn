#!/usr/bin/env python3
"""La tabla de resultados de los cinco brazos, en la epoca que guarda su `best.pt`.

    python nn/informe.py            la tabla, en texto
    python nn/informe.py --md       la misma, en markdown para el README

NO DECLARA UN GANADOR, y eso es una decision del dueno (2026-09-07): «No importa
quien gana, quiero ver todos los ganadores. Es un experimento, no un concurso».
Lo que si marca es el UMBRAL del criterio --12 % de acierto a <=2 px en LAS DOS
esquinas-- porque eso no es elegir a uno: es separar "aprendio" de "se quedo en
el suelo", que es la unica pregunta que un suelo medido puede contestar.

⚠ LA EPOCA QUE SE REPORTA ES LA DE `best.pt`, elegida por `val_loss` y con la
MISMA regla en los cinco. Reportar la mejor epoca de cada metrica por separado
seria elegir el mejor momento de cada brazo para cada numero, que es como se
fabrica una tabla que nadie puede reproducir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from modelo import BRAZOS, ESQUINAS, construir

AQUI = Path(__file__).resolve().parent
PESOS = AQUI / "pesos"

# Del criterio congelado (instrucciones/02-criterio.md), y por eso estan aqui
# como constantes con su nombre: si el criterio cambia, esto tiene que cambiar
# con el y verse en el diff.
# ⚠ TODOS MEDIDOS el 2026-09-07 con `entrenar_local.py --suelos` sobre el `val`
# publicado, ANTES de la primera epoca. Ninguno se hereda de `esq-2d`: alli las
# positivas eran 78 por esquina y aqui son 156 en una sola metrica.
SUELO = 0.058                  # acierto <=2 px de la red SIN entrenar (5,1 tl / 6,4 br)
SUELO_F1 = 0.572               # el "siempre si" con 40,1 % de positivas
SUELO_ERROR = 6.08             # px, el predictor constante en el centro
# suelo + 2 SE, con SE = sqrt(p(1-p)/156) = 1,9 % sobre las 156 positivas.
UMBRAL_ACIERTO = 0.096         # a <=2 px, en la metrica UNICA
UMBRAL_ERROR = 5.70            # px, metrica secundaria (suelo - 2 SE)


def _fila(brazo: str) -> dict | None:
    d = PESOS / brazo
    mejor, reg = d / "best.pt", d / "metrics.jsonl"
    if not (mejor.exists() and reg.exists()):
        return None
    ep = torch.load(mejor, map_location="cpu", weights_only=False)["epoca"]
    for linea in reg.open(encoding="utf-8"):
        f = json.loads(linea)
        if f["epoca"] == ep:
            f["brazo"], f["parametros"] = brazo, construir(brazo).n_parametros()
            # ⚠ UNA sola condicion: ya no hay dos esquinas que tengan que pasar
            # las dos. El desglose se REPORTA (es el ancla de comparabilidad con
            # `esq-2d`), pero no decide.
            f["pasa"] = f.get("acierto_2px", 0) > UMBRAL_ACIERTO
            return f
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--md", action="store_true")
    a = p.parse_args()

    filas = [f for f in (_fila(b) for b in BRAZOS) if f]
    if not filas:
        print("✗ no hay resultados todavia: entrena primero (nn/entrenar_local.py --brazo kNN)")
        return 1

    if a.md:
        print("| brazo | params | época | **acierto ≤2 px** | *(tl* | *br)* | ≤1 px | "
              "error px | f1 `existe` | fp otra diag. | kernel simétrico | ¿pasa? |")
        print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|")
        for f in filas:
            print(f"| `{f['brazo']}` | {f['parametros']} | {f['epoca']} | "
                  f"**{100*f['acierto_2px']:.1f} %** | "
                  f"*{100*f.get('acierto_2px_tl', 0):.1f} %* | "
                  f"*{100*f.get('acierto_2px_br', 0):.1f} %* | "
                  f"{100*f['acierto_1px']:.1f} % | {f['err_px']:.2f} px | "
                  f"{f['f1_existe']:.3f} | {100*f['fp_otra_diagonal']:.0f} % | "
                  f"{100*f['simetrico']:.0f} % | {'✅' if f['pasa'] else '❌'} |")
        print(f"| **suelo** (sin entrenar) | — | 0 | **{100*SUELO:.1f} %** | *5,1 %* | "
              f"*6,4 %* | 2,1 % | {SUELO_ERROR:.2f} px | {SUELO_F1:.3f} | 100 % | 62 % | — |")
        return 0

    print(f"{'brazo':>6} {'par':>5} {'ep':>4} | {'<=2px':>7} {'(tl':>7} {'br)':>7} | "
          f"{'<=1px':>7} {'err':>7} | {'f1':>6} {'fp-otra':>8} | {'S%':>4} | pasa")
    for f in filas:
        print(f"{f['brazo']:>6} {f['parametros']:>5} {f['epoca']:>4} | "
              f"{100*f['acierto_2px']:>6.1f}% {100*f.get('acierto_2px_tl', 0):>6.1f}% "
              f"{100*f.get('acierto_2px_br', 0):>6.1f}% | "
              f"{100*f['acierto_1px']:>6.1f}% {f['err_px']:>7.2f} | "
              f"{f['f1_existe']:>6.3f} {100*f['fp_otra_diagonal']:>7.0f}% | "
              f"{100*f['simetrico']:>3.0f}% | {'SI' if f['pasa'] else 'no'}")
    print(f"{'suelo':>6} {'—':>5} {0:>4} | {100*SUELO:>6.1f}% {5.1:>6.1f}% {6.4:>6.1f}% | "
          f"{2.1:>6.1f}% {SUELO_ERROR:>7.2f} | {SUELO_F1:>6.3f} {100:>7.0f}% |  62% | —")

    pasan = [f["brazo"] for f in filas if f["pasa"]]
    print(f"\numbral del criterio: acierto <=2 px > {100*UMBRAL_ACIERTO:.1f} % "
          f"(suelo {100*SUELO:.1f} % + 2 SE)")
    print(f"  pasan {len(pasan)} de {len(filas)}: {', '.join(pasan) if pasan else '(ninguno)'}")
    print(f"  (secundaria: error medio < {UMBRAL_ERROR} px · f1 de `existe` > {SUELO_F1})")

    # ⚠⚠ LA LECTURA QUE ESTE EXPERIMENTO PUEDE FALSEAR SI NO SE DICE.
    peor = min((f for f in filas), key=lambda f: f.get("acierto_2px_br", 0))
    print("\n⚠ EL DESGLOSE NO ES ADORNO. Las positivas son mitad `tl` y mitad `br`,")
    print("  asi que un ~50 % en la metrica unica es compatible con 'tl entero, br")
    print("  nada'. Medido el 2026-09-07: la tinta mas cercana a `br` esta a 8,1 px")
    print("  (mediana) y ningun brazo de k=5..13 tiene tanto radio (2..6 px).")
    print(f"  El brazo con peor `br` de los medidos: {peor['brazo']} con "
          f"{100*peor.get('acierto_2px_br', 0):.1f} %")
    print("\nNo se declara ganador a proposito: se reportan todos los que pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
