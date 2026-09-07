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
UMBRAL_ACIERTO = 0.12          # a <=2 px, en LAS DOS esquinas
UMBRAL_ERROR = 5.45            # px, metrica secundaria
SUELO_F1 = 0.334               # el "siempre si" por esquina
SUELO = {"tl": 0.051, "br": 0.064}


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
            f["pasa"] = all(f.get(f"acierto_2px_{c}", 0) > UMBRAL_ACIERTO for c in ESQUINAS)
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
        print("| brazo | params | época | acierto ≤2 px `tl` | `br` | ≤1 px `tl` | `br` | "
              "error `tl` | `br` | f1 `tl` | `br` | kernel antisim. | ¿pasa? |")
        print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|")
        for f in filas:
            print(f"| `{f['brazo']}` | {f['parametros']} | {f['epoca']} | "
                  f"{100*f['acierto_2px_tl']:.1f} % | {100*f['acierto_2px_br']:.1f} % | "
                  f"{100*f['acierto_1px_tl']:.1f} % | {100*f['acierto_1px_br']:.1f} % | "
                  f"{f['err_px_tl']:.2f} px | {f['err_px_br']:.2f} px | "
                  f"{f['f1_existe_tl']:.3f} | {f['f1_existe_br']:.3f} | "
                  f"{100*f['antisimetrico']:.0f} % | {'✅' if f['pasa'] else '❌'} |")
        print(f"| **suelo** (sin entrenar) | — | 0 | {100*SUELO['tl']:.1f} % | "
              f"{100*SUELO['br']:.1f} % | 1,3 % | 3,8 % | 5,96 px | 6,13 px | "
              f"{SUELO_F1:.3f} | {SUELO_F1:.3f} | 38 % | — |")
        return 0

    print(f"{'brazo':>6} {'par':>5} {'ep':>4} | {'<=2px tl':>9} {'br':>7} | "
          f"{'<=1px tl':>9} {'br':>7} | {'err tl':>7} {'br':>7} | {'f1 tl':>6} {'br':>6} | "
          f"{'A%':>4} | pasa")
    for f in filas:
        print(f"{f['brazo']:>6} {f['parametros']:>5} {f['epoca']:>4} | "
              f"{100*f['acierto_2px_tl']:>8.1f}% {100*f['acierto_2px_br']:>6.1f}% | "
              f"{100*f['acierto_1px_tl']:>8.1f}% {100*f['acierto_1px_br']:>6.1f}% | "
              f"{f['err_px_tl']:>7.2f} {f['err_px_br']:>7.2f} | "
              f"{f['f1_existe_tl']:>6.3f} {f['f1_existe_br']:>6.3f} | "
              f"{100*f['antisimetrico']:>3.0f}% | {'SI' if f['pasa'] else 'no'}")
    print(f"{'suelo':>6} {'—':>5} {0:>4} | {100*SUELO['tl']:>8.1f}% {100*SUELO['br']:>6.1f}% | "
          f"{1.3:>8.1f}% {3.8:>6.1f}% | {5.96:>7.2f} {6.13:>7.2f} | "
          f"{SUELO_F1:>6.3f} {SUELO_F1:>6.3f} |  38% | —")

    pasan = [f["brazo"] for f in filas if f["pasa"]]
    print(f"\numbral del criterio: acierto <=2 px > {100*UMBRAL_ACIERTO:.0f} % en LAS DOS esquinas")
    print(f"  pasan {len(pasan)} de {len(filas)}: {', '.join(pasan) if pasan else '(ninguno)'}")
    print(f"  (secundaria: error medio < {UMBRAL_ERROR} px · f1 de `existe` > {SUELO_F1})")
    print("\nNo se declara ganador a proposito: se reportan todos los que pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
