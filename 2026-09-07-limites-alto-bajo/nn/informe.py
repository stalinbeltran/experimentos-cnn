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

from modelo import BRAZOS, construir

AQUI = Path(__file__).resolve().parent
PESOS = AQUI / "pesos"

# Del criterio congelado (instrucciones/02-criterio.md), y por eso estan aqui
# como constantes con su nombre: si el criterio cambia, esto tiene que cambiar
# con el y verse en el diff.
# ✅ MEDIDOS Y CONGELADOS el 2026-09-07 con
# `entrenar_local.py --suelos` sobre el dataset de ESTE experimento, ANTES de la
# primera epoca (R13).
#
# NO SE PUDIERON HEREDAR DE `esq-cq`, y este es el fallo mas caro de todo el
# experimento: alli la distancia era euclidea en 2-D (suelo 5,8 %) y aqui es
# |dy| en 1-D (suelo ~28,6 %, medido el 2026-09-07 con las redes sin entrenar).
# El umbral de alli, 9,6 %, esta POR DEBAJO del suelo de aqui: con el heredado,
# los siete brazos "pasarian" en la epoca 0 sin haber aprendido nada, y el
# informe lo diria con un ✅. Medido: el suelo de aqui es 21,2 % y el umbral de
# alli 9,6 %, o sea que TODOS pasarian sin entrenar.
SUELO = 0.212                  # acierto |dy|<=2 px de la red SIN entrenar (media de los 7)
SUELO_F1 = 0.571               # el "siempre si" con 40,0 % de positivas
SUELO_ERROR = 4.28             # px de |dy|, el predictor constante en el centro
# suelo + 2 SE, con SE = sqrt(p(1-p)/312) = 2,3 % sobre las 312 positivas.
UMBRAL_ACIERTO = 0.258
UMBRAL_ERROR = 4.00            # px, metrica secundaria


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
            # ⚠ Si el umbral no se ha congelado todavia, NO se decide. `None` es
            # deliberado: un umbral heredado de otra metrica es peor que ninguno.
            f["pasa"] = (None if UMBRAL_ACIERTO is None
                         else f.get("acierto_2px", 0) > UMBRAL_ACIERTO)
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

    sin_umbral = UMBRAL_ACIERTO is None
    if sin_umbral:
        print("⚠⚠ EL CRITERIO NO ESTA CONGELADO: SUELO/UMBRAL siguen a None.")
        print("   Corre `nn/entrenar_local.py --suelos` y pon SUS numeros aqui")
        print("   ANTES de leer nada. Heredar los de `esq-cq` (suelo 5,8 %, umbral")
        print("   9,6 %) es invalido: alli la metrica era 2-D y aqui es 1-D, con")
        print("   suelo ~28,6 %. Los siete brazos 'pasarian' sin aprender nada.\n")

    cab = (f"{'brazo':>6} {'par':>5} {'ep':>4} | {'<=2px':>7} {'(alto':>7} {'bajo)':>7} | "
           f"{'tl':>6} {'tr':>6} {'bl':>6} {'br':>6} | {'PURO s':>7} {'PURO i':>7} | "
           f"{'f1':>6} {'fpI':>5} {'fpV':>5} | {'Sv%':>4} {'S180':>5} | pasa")
    if a.md:
        print("| brazo | params | época | **≤2 px** | *alto* | *bajo* | `tl` | `tr` | `bl` | "
              "`br` | **puro sup** | **puro inf** | f1 | fp int. | fp vert. | sim. vert. | ¿pasa? |")
        print("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|")
    else:
        print(cab)
    for f in filas:
        g = lambda k: 100 * f.get(k, 0)
        marca = "—" if f["pasa"] is None else ("✅" if f["pasa"] else "❌")
        if a.md:
            print(f"| `{f['brazo']}` | {f['parametros']} | {f['epoca']} | "
                  f"**{g('acierto_2px'):.1f} %** | *{g('acierto_2px_alto'):.1f} %* | "
                  f"*{g('acierto_2px_bajo'):.1f} %* | {g('acierto_2px_tl'):.1f} % | "
                  f"{g('acierto_2px_tr'):.1f} % | {g('acierto_2px_bl'):.1f} % | "
                  f"{g('acierto_2px_br'):.1f} % | **{g('acierto_2px_puro_sup'):.1f} %** | "
                  f"**{g('acierto_2px_puro_inf'):.1f} %** | {f['f1_existe']:.3f} | "
                  f"{g('fp_interior'):.0f} % | {g('fp_borde_vertical'):.0f} % | "
                  f"{g('simetrico_vertical'):.0f} % | {marca} |")
        else:
            print(f"{f['brazo']:>6} {f['parametros']:>5} {f['epoca']:>4} | "
                  f"{g('acierto_2px'):>6.1f}% {g('acierto_2px_alto'):>6.1f}% "
                  f"{g('acierto_2px_bajo'):>6.1f}% | {g('acierto_2px_tl'):>5.1f}% "
                  f"{g('acierto_2px_tr'):>5.1f}% {g('acierto_2px_bl'):>5.1f}% "
                  f"{g('acierto_2px_br'):>5.1f}% | {g('acierto_2px_puro_sup'):>6.1f}% "
                  f"{g('acierto_2px_puro_inf'):>6.1f}% | {f['f1_existe']:>6.3f} "
                  f"{g('fp_interior'):>4.0f}% {g('fp_borde_vertical'):>4.0f}% | "
                  f"{g('simetrico_vertical'):>3.0f}% {g('simetrico_rot180'):>4.0f}% | "
                  f"{'SI' if f['pasa'] else ('?' if f['pasa'] is None else 'no')}")

    if sin_umbral:
        print("\n⚠ Sin criterio congelado no se declara nada. Los numeros estan; el")
        print("  veredicto no, y es a proposito.")
        return 1

    pasan = [f["brazo"] for f in filas if f["pasa"]]
    print(f"\numbral: acierto <=2 px > {100*UMBRAL_ACIERTO:.1f} % "
          f"(suelo {100*SUELO:.1f} % + 2 SE)")
    print(f"  pasan {len(pasan)} de {len(filas)}: {', '.join(pasan) if pasan else '(ninguno)'}")
    print("\n⚠ LO QUE HAY QUE LEER ES EL DESGLOSE, no el numero global:")
    print("  · `tl`/`tr`/`bl`/`br` son las MISMAS 78/39/39/78 ventanas de `esq-cq`:")
    print("    es lo unico comparable columna a columna con aquel experimento.")
    print("  · `PURO s`/`PURO i` son el limite SIN esquina -- lo que `esq-cq` NO")
    print("    podia medir, y la razon de haber regenerado el dataset.")
    print("  · Prediccion registrada: `alto` satura pronto; `bajo` se atasca en su")
    print("    celda `br`, porque la ultima linea del parrafo es corta (7,00 px de")
    print("    tinta contra 1,00-2,00 en las otras tres celdas, medido 2026-09-07).")
    print("\nNo se declara ganador a proposito: se reportan todos los que pasan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
