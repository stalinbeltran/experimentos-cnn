#!/usr/bin/env python3
"""Las metricas del §9: IoU (titular), MAE por borde (diagnostico) y brecha. AUTONOMO.

    python nn/evaluar.py      comprueba las metricas sobre casos con respuesta conocida

CONVENIO DE CAJA, y es el mismo en todo el experimento:

    caja = (izq, der, sup, inf)      en PIXELES del marco de 128

o sea el orden de los cuatro canales de la cabeza (§7.3), no (x, y, w, h). Se eligio
asi porque es como la especificacion nombra la etiqueta en §3.9, y porque el
soft-argmax produce los cuatro bordes de forma independiente: no hay ningun sitio del
banco donde exista un "ancho" que no sea `der - izq`.

⚠ UNA CAJA PREDICHA PUEDE SER INVALIDA. Nada obliga a la red a poner `der > izq`, y
al principio del entrenamiento no lo hace. Un area negativa metida en la formula del
IoU da numeros sin sentido (IoU > 1, o negativos) que luego no hay quien interprete.
Aqui las areas se acotan a >= 0, con lo que una caja invertida da IoU 0, que es la
lectura correcta: no se solapa con nada.
"""

from __future__ import annotations

import numpy as np


def iou(pred: np.ndarray, real: np.ndarray) -> np.ndarray:
    """(N,4) x (N,4) -> (N,) IoU por muestra. Cajas (izq, der, sup, inf)."""
    p = np.asarray(pred, dtype=np.float64)
    r = np.asarray(real, dtype=np.float64)
    ancho_i = np.maximum(0.0, np.minimum(p[:, 1], r[:, 1]) - np.maximum(p[:, 0], r[:, 0]))
    alto_i = np.maximum(0.0, np.minimum(p[:, 3], r[:, 3]) - np.maximum(p[:, 2], r[:, 2]))
    inter = ancho_i * alto_i
    ap = np.maximum(0.0, p[:, 1] - p[:, 0]) * np.maximum(0.0, p[:, 3] - p[:, 2])
    ar = np.maximum(0.0, r[:, 1] - r[:, 0]) * np.maximum(0.0, r[:, 3] - r[:, 2])
    union = ap + ar - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def mae_por_borde(pred: np.ndarray, real: np.ndarray) -> dict[str, float]:
    """§9.1: error absoluto medio en PIXELES, desglosado en los cuatro bordes.

    En pixeles del marco de 128 y no normalizado, porque el diagnostico del §7.5 se
    lee en pixeles: «si el MAE se estanca cerca de 8 px» -- que es exactamente el
    tamanyo de una celda de la rejilla (128/16)."""
    e = np.abs(np.asarray(pred, np.float64) - np.asarray(real, np.float64))
    return {n: float(e[:, i].mean()) for i, n in enumerate(("izq", "der", "sup", "inf"))}


def resumen(pred: np.ndarray, real: np.ndarray) -> dict:
    v = iou(pred, real)
    return {"iou": float(v.mean()), "iou_mediana": float(np.median(v)),
            "mae": mae_por_borde(pred, real),
            "mae_medio": float(np.abs(np.asarray(pred, np.float64)
                                      - np.asarray(real, np.float64)).mean())}


def caja_media(train_real: np.ndarray) -> np.ndarray:
    """§10.1: el predictor constante. Ignora la imagen: es la media de `train`.

    Es el PISO del banco. Si saca un IoU alto, el generador coloca los parrafos con
    poca variabilidad, todas las condiciones se comprimen y el banco no discrimina
    nada -- y entonces se corrige el generador y NO se continua."""
    return np.asarray(train_real, dtype=np.float64).mean(axis=0)


def agregado(por_semilla: list[float]) -> dict:
    """§9.3: media +- desviacion ENTRE SEMILLAS. Una sola semilla no es un resultado.

    La desviacion es la MUESTRAL (ddof=1): con ddof=0 se subestima, y aqui la
    desviacion es el DENOMINADOR de los dos criterios de exito -- subestimarla haria
    que un kernel pareciera declarar cuando no declara."""
    a = np.asarray(por_semilla, dtype=np.float64)
    return {"media": float(a.mean()),
            "desv": float(a.std(ddof=1)) if len(a) > 1 else float("nan"),
            "n": int(len(a))}


def criterio_utilidad(kernel: dict, aleatorio: dict) -> dict:
    """§2.1: util si supera al aleatorio por MAS que la SUMA de las dos desviaciones.

    El margen es la SUMA, no la mayor ni la del kernel: es el mas exigente de los tres
    y es el que esta escrito."""
    margen = kernel["desv"] + aleatorio["desv"]
    dif = kernel["media"] - aleatorio["media"]
    return {"diferencia": dif, "margen": margen, "cumple": bool(dif > margen)}


def criterio_generalizacion(brecha_k: dict, brecha_id: dict, cumple_2_1: bool) -> dict:
    """§2.2: la brecha del kernel menor que la de la identidad, MISMO margen.

    ⚠ Es CONDICIONAL al §2.1 -- «si, ademas de cumplir 2.1» --, asi que un kernel que
    reduce la brecha pero no supera al aleatorio NO cumple este criterio."""
    margen = brecha_k["desv"] + brecha_id["desv"]
    dif = brecha_id["media"] - brecha_k["media"]
    return {"diferencia": dif, "margen": margen,
            "cumple": bool(cumple_2_1 and dif > margen),
            "cumple_ignorando_2_1": bool(dif > margen)}


def _main() -> int:
    fallos = []

    def ok(que, obtenido, esperado, tol=1e-9):
        bien = abs(obtenido - esperado) < tol if isinstance(esperado, float) \
            else obtenido == esperado
        print(f"  {'ok ' if bien else 'FALLA'}  {que:46} {obtenido}"
              f"{'' if bien else f'  (esperado {esperado})'}")
        if not bien:
            fallos.append(que)

    print("\nMetricas del §9\n")
    caja = np.array([[10.0, 30.0, 10.0, 30.0]])                 # 20x20 = 400
    ok("caja identica → IoU 1", float(iou(caja, caja)[0]), 1.0)
    ok("caja disjunta → IoU 0",
       float(iou(np.array([[50.0, 70.0, 50.0, 70.0]]), caja)[0]), 0.0)
    # solape de la mitad en x, todo en y: inter=200, union=400+400-200=600
    ok("solape mitad en x → IoU 1/3",
       float(iou(np.array([[20.0, 40.0, 10.0, 30.0]]), caja)[0]), 1 / 3)
    # una caja que contiene a la otra: inter=400, union=1600 -> 0.25
    ok("contenida → IoU 0,25",
       float(iou(np.array([[0.0, 40.0, 0.0, 40.0]]), caja)[0]), 0.25)
    ok("caja INVERTIDA (der<izq) → IoU 0",
       float(iou(np.array([[30.0, 10.0, 30.0, 10.0]]), caja)[0]), 0.0)

    m = mae_por_borde(np.array([[12.0, 30.0, 10.0, 34.0]]), caja)
    ok("MAE por borde: izq", m["izq"], 2.0)
    ok("MAE por borde: der", m["der"], 0.0)
    ok("MAE por borde: inf", m["inf"], 4.0)

    ok("caja media = media de train",
       tuple(caja_media(np.array([[0.0, 10.0, 0.0, 10.0], [10.0, 20.0, 10.0, 20.0]]))),
       (5.0, 15.0, 5.0, 15.0))

    a = agregado([0.90, 0.92, 0.94])
    ok("agregado: media", round(a["media"], 6), 0.92)
    ok("agregado: desv muestral (ddof=1)", round(a["desv"], 6), round(float(np.std([0.90, 0.92, 0.94], ddof=1)), 6))

    # §2.1: 0,90 vs 0,85 con desviaciones 0,01 y 0,02 -> dif 0,05 > margen 0,03 -> cumple
    c = criterio_utilidad({"media": 0.90, "desv": 0.01}, {"media": 0.85, "desv": 0.02})
    ok("§2.1 cumple cuando dif > suma de desviaciones", c["cumple"], True)
    c = criterio_utilidad({"media": 0.90, "desv": 0.03}, {"media": 0.85, "desv": 0.03})
    ok("§2.1 NO cumple cuando dif < suma (0,05 < 0,06)", c["cumple"], False)
    # §2.2 es condicional al §2.1
    g = criterio_generalizacion({"media": 0.05, "desv": 0.01},
                                {"media": 0.20, "desv": 0.01}, cumple_2_1=False)
    ok("§2.2 NO cumple si el §2.1 no cumple", g["cumple"], False)
    ok("§2.2 pero se registra que lo haria", g["cumple_ignorando_2_1"], True)

    print()
    if fallos:
        print(f"✗ {len(fallos)} fallo(s): " + ", ".join(fallos) + "\n")
        return 1
    print("Las metricas del §9 cumplen su contrato.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
