#!/usr/bin/env python3
"""Lo que el banco `banco-k` le hace a una imagen, COPIADO aqui. Solo numpy.

    python nn/banco.py        comprueba que esta copia coincide con el original

POR QUE UNA COPIA Y NO UN `import`
===================================
La regla del repo es que ningun experimento importa de otro (`REGLAS.ejemplo.md`
§ Scripts): se copia. El motivo es que un experimento no puede quedar atado a que
otro no cambie.

⚠ PERO AQUI LA COPIA TIENE UN RIESGO QUE NO TIENE EN OTRO SITIO, y por eso no se
deja sola. Este experimento existe para ENSENYAR lo que el banco vera: si esta
copia se desvia del §6 de `banco-k`, la vista previa miente, y miente de la peor
forma posible -- se ve bien y decide un parametro equivocado.

La salida no es compartir codigo (lo prohibe la regla) ni confiar en un
comentario (R17: una comprobacion que no corre sola no existe). Es MEDIR EL
ACUERDO: `probar.py` importa el `pipeline.py` del banco cuando esta en el disco y
exige que las dos implementaciones den el MISMO array bit a bit. Si `banco-k`
cambia su §6, la prueba falla y lo dice. Si `banco-k` no esta, la prueba lo dice
tambien, en vez de callarse.

QUE SE COPIO, Y DE DONDE
========================
Del experimento `banco-k`, `nn/pipeline.py` (§6 de su ESPECIFICACION.md):
`normalizar_kernel`, `comprobar_contrato`, `recorte_de`, `aplicar`,
`estadisticos`, `estandarizar`, y las constantes MARCO/DESCARTE/FINAL/K_MAX.
y de su `nn/kernels.py`: `gauss`.

⚠ La acumulacion por desplazamiento de `aplicar` NO es un detalle de estilo: la
version con `sliding_window_view` + `einsum` pedia 4,6 GiB con k=9 y 800 imagenes
y reventaba un server de 3,8 GB (medido por `banco-k` el 2026-09-08). Se copia
tal cual, con su motivo.
"""

from __future__ import annotations

import numpy as np

MARCO = 146          # el marco del dataset tras reducir /4
DESCARTE = 9         # §6.2: descarte fijo por lado, sea cual sea `k`
FINAL = 128          # MARCO - 2*DESCARTE
K_MAX = 19           # §5.2, CONGELADO en el banco
K_MIN = 3


def gauss(k: int, sigma: float) -> np.ndarray:
    """Gaussiana isotropa k x k. Suma POSITIVA.

    ⚠ `sigma` es OBLIGATORIO aqui, y en el banco tiene defecto `k/6`. Es a
    proposito: este experimento existe para elegir `sigma`, asi que dejarlo caer a
    un defecto seria contestar la pregunta antes de hacerla. El `k/6` del banco es
    una CONVENCION de su codigo, no algo que nadie midiera; `SIGMA_BANCO` de abajo
    lo deja disponible para comparar contra el.
    """
    if sigma <= 0:
        raise ValueError(f"sigma tiene que ser > 0 y es {sigma}. En el limite "
                         f"sigma->0 la gaussiana ES la identidad (§6.3 del banco), "
                         f"que ya es una condicion evaluada: no hace falta pedirla aqui")
    r = np.arange(k, dtype=np.float32) - (k - 1) / 2.0
    g = np.exp(-(r ** 2) / (2 * sigma ** 2))
    return np.outer(g, g).astype(np.float32)


def sigma_banco(k: int) -> float:
    """El `sigma` que `banco-k` usa por defecto: `k/6`. El punto ya evaluado."""
    return k / 6.0


def normalizar_kernel(k: np.ndarray) -> np.ndarray:
    """§5.4: norma L2 = 1. La unica diferencia entre kernels pasa a ser la FORMA."""
    k = np.asarray(k, dtype=np.float32)
    n = float(np.sqrt((k ** 2).sum()))
    if n == 0:
        raise ValueError("kernel de norma cero: no es un detector de nada")
    return (k / n).astype(np.float32)


def comprobar_contrato(k: np.ndarray) -> None:
    """§5.1: la puerta del banco. Se niega ANTES de convolucionar nada."""
    if k.ndim != 2 or k.shape[0] != k.shape[1]:
        raise ValueError(f"§5.1: el kernel tiene que ser (k, k), y es {k.shape}")
    n = k.shape[0]
    if n % 2 == 0:
        raise ValueError(
            f"§5.3: `k` tiene que ser IMPAR y es {n}. Con `k` par el centro cae entre "
            f"pixeles y el recorte queda asimetrico por medio pixel, lo que mete un "
            f"desplazamiento sistematico que DIFIERE entre condiciones")
    if not (K_MIN <= n <= K_MAX):
        raise ValueError(f"§5.2: {K_MIN} <= k <= {K_MAX}, y es {n}. K_MAX esta "
                         f"CONGELADO: si hace falta mas, se construye un banco nuevo")


def recorte_de(k: int | None) -> int:
    """Cuanto se recorta por lado para que el descarte total sea 9 (§6.2)."""
    if k is None:
        return DESCARTE                      # identidad: recorte puro
    return DESCARTE - (k - 1) // 2


def aplicar(imgs: np.ndarray, kernel: np.ndarray | None) -> np.ndarray:
    """(N,146,146) -> (N,128,128) float32. `kernel=None` es la identidad (§6.3)."""
    x = np.asarray(imgs, dtype=np.float32)
    if kernel is None:
        r = DESCARTE
        return np.ascontiguousarray(x[:, r:r + FINAL, r:r + FINAL])

    comprobar_contrato(kernel)
    k = normalizar_kernel(kernel)
    n = k.shape[0]
    kf = k[::-1, ::-1]                       # correlacion cruzada con el kernel volteado
    lado = MARCO - n + 1
    # Se acumula por DESPLAZAMIENTO: el unico array grande es el acumulador.
    # Ver el aviso del docstring de este modulo.
    y = np.zeros((x.shape[0], lado, lado), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            c = float(kf[i, j])
            if c != 0.0:
                y += c * x[:, i:i + lado, j:j + lado]
    r = recorte_de(n)
    out = y[:, r:r + FINAL, r:r + FINAL]
    assert out.shape[1:] == (FINAL, FINAL), f"tras recortar: {out.shape}"
    return np.ascontiguousarray(out)


def estadisticos(x_train: np.ndarray) -> tuple[float, float]:
    """§6.5: mu y sigma salen SOLO de `train` filtrado, y se aplican a todo.

    ⚠ AQUI `train` ES EL SUBCONJUNTO LIBRE DE LA RESERVA, no las 100 del banco.
    La diferencia esta medida y documentada en `muestras.py`: es un desplazamiento
    comun a TODAS las condiciones, asi que no altera la comparacion entre
    parametros, que es para lo que sirve este experimento.
    """
    return float(x_train.mean()), float(x_train.std())


def estandarizar(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """§6.5. Es un CONTROL, no una variable: sin ella un kernel de suma positiva
    hereda la media alta del fondo blanco y eso se contaria como calidad."""
    if sigma <= 0:
        raise ValueError("sigma <= 0: el filtrado dejo una imagen constante")
    return ((x - mu) / sigma).astype(np.float32)


def ks_validos() -> list[int]:
    """Los `k` que el contrato del banco admite: impares de 3 a 19."""
    return list(range(K_MIN, K_MAX + 1, 2))


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from probar import main                                  # noqa: PLC0415
    raise SystemExit(main())
