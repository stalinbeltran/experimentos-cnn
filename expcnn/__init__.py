"""`expcnn` — la poca fontanería que comparten TODOS los experimentos.

Superficie pública declarada (R5). Es a propósito diminuta: aquí no vive nada
del dominio. Lo que compartan dos experimentos para **medir igual** no va aquí,
va en un `comun/` que todavía no existe y que se crea cuando haya un segundo
caso, nunca con el primero.
"""

from .entorno import exigir_datos, exigir_fv, hay_fv, ruta_datos, ruta_fv
from .registro import ESTADOS, GASTOS, Experimento, experimentos, por_id, raiz

__all__ = [
    "ESTADOS",
    "GASTOS",
    "Experimento",
    "exigir_datos",
    "exigir_fv",
    "experimentos",
    "hay_fv",
    "por_id",
    "ruta_datos",
    "raiz",
    "ruta_fv",
]
