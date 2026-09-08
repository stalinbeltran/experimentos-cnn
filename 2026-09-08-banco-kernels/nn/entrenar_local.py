#!/usr/bin/env python3
"""La ENTRADA declarada de `banco-k`. Hoy se NIEGA a entrenar, y dice por que.

    python nn/entrenar_local.py              se niega (codigo 2) y lista que falta
    python nn/entrenar_local.py --comprobar  comprueba la arquitectura (codigo 0)

POR QUE ESTE FICHERO EXISTE ANTES DE PODER ENTRENAR
---------------------------------------------------
Dos razones, y ninguna es «dejar un hueco»:

1. EL CONTRATO DE NOMBRE CON EL FRENO. `experimento.json` declara
   `gasta: "entrena-local"`, y el freno del coordinador (`cerrable.mjs`, lista
   declarada `TRABAJOS`) casa EL NOMBRE `entrenar_local.py` para decidir si se puede
   apagar este server. Con otro nombre, el veredicto que el duenyo lee desde el movil
   diria «nada corriendo» con un barrido vivo. Es la regla 4 de escritura del
   proyecto -- «el freno nunca llega despues del acelerador» -- cumplida por
   adelantado: el nombre visible para el freno esta puesto ANTES de que haya algo que
   gastar tiempo, en vez de anyadirse en el commit que lo estrena.

2. SE NIEGA ANTES DE EMPEZAR, NO A MITAD (R2). El dataset que pide la especificacion
   §3 no esta publicado. La alternativa -- re-derivarlo al vuelo -- daria numeros que
   no se pueden comparar con los de nadie y NO fallaria por ningun lado.

⚠ Que este fichero exista NO significa que el banco este listo. Lo que falta esta en
`REGLAS.md` § Scripts y en `instrucciones/01-encargo.md`; esto solo garantiza que el
dia que entrene, el freno lo vea.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
MANIFIESTO = EXP / "experimento.json"


def _falta_para_entrenar() -> list[str]:
    """Que impide entrenar HOY. Estado utilizable, no presencia (regla 5)."""
    falta = []
    datos = json.loads(MANIFIESTO.read_text(encoding="utf-8"))

    nombre = datos.get("dataset")
    if not nombre:
        previsto = datos.get("dataset_previsto", "(sin nombre previsto)")
        falta.append(
            f"el dataset: `experimento.json` declara \"dataset\": null.\n"
            f"     Nombre previsto: {previsto}\n"
            f"     Se genera y se publica en foveal-vision-data/experimentos-cnn/ con el\n"
            f"     paso 2 de REGLAS.md § Procesos, y NO se re-deriva al vuelo: el dato de\n"
            f"     entrada tiene que ser EL MISMO fichero entre corridas, no uno equivalente."
        )
    else:
        # La puerta al dato es expcnn, que se niega si no esta publicado. Se importa
        # aqui dentro para que `--comprobar` funcione en un clon sin instalar nada.
        sys.path.insert(0, str(EXP.parent))
        from expcnn import ruta_dataset          # noqa: PLC0415
        if ruta_dataset(nombre) is None:
            falta.append(f"el dataset '{nombre}' no esta publicado en el repo de datos")

    # `bloqueado_por` = decisiones del duenyo que faltan (v1.2 cerro las tres).
    # `pendiente` = trabajo por hacer, que no necesita a nadie. Se distinguen porque
    # no se desbloquean igual: una se pregunta, la otra se escribe.
    for q in datos.get("bloqueado_por", []):
        falta.append(f"una DECISION del duenyo -- {q}")
    for q in datos.get("pendiente", []):
        falta.append(f"TRABAJO por hacer -- {q}")

    for script, para_que in (("pipeline.py", "kernel + recorte + estandarizacion (§6)"),
                             ("evaluar.py", "IoU, MAE por borde y brecha (§9.1)")):
        if not (AQUI / script).is_file():
            falta.append(f"nn/{script}: {para_que}")
    return falta


def _comprobar() -> int:
    """Lo unico que YA se puede comprobar sin dataset: la arquitectura del §7."""
    sys.path.insert(0, str(AQUI))
    from modelo import main as comprobar_modelo   # noqa: PLC0415
    return comprobar_modelo()


def main(argv: list[str]) -> int:
    if "--comprobar" in argv:
        return _comprobar()

    falta = _falta_para_entrenar()
    print("\n✗ `banco-k` NO puede entrenar todavia. Se niega ANTES de empezar, no a la")
    print("  epoca 30, porque entrenar sobre un dato re-derivado al vuelo daria numeros")
    print("  incomparables sin fallar por ningun lado (R2).\n")
    print(f"  Falta{'n' if len(falta) != 1 else ''} {len(falta)} cosa(s):\n")
    for i, f in enumerate(falta, 1):
        print(f"  {i}. {f}")
    print("\n  El orden en que se resuelven esta en REGLAS.md § Procesos.")
    print("  Lo que si se puede comprobar hoy:  python nn/entrenar_local.py --comprobar\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
