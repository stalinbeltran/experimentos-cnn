#!/usr/bin/env python3
"""El pipeline de aplicacion del §6: kernel -> recorte -> estandarizacion. AUTONOMO.

    python nn/pipeline.py     comprueba el §6 entero sobre datos sinteticos

    146x146  ->  conv 'valid' con k  ->  recorte central  ->  128x128  ->  estandarizar

LAS CUATRO COSAS QUE HAY QUE RESPETAR SI SE TOCA
================================================

1. EL DESCARTE ES SIEMPRE 9 px POR LADO, sea cual sea `k` (§6.2). Un kernel pequenyo
   convoluciona menos y recorta mas, y la suma da 9 exacto:

       k=3  -> valid 144, recorte 8   |  k=19 -> valid 128, recorte 0
       k=9  -> valid 138, recorte 5   |  identidad -> 146, recorte 9

   El motivo NO es estetico: si cada `k` fijara su propio tamanyo de salida, las
   condiciones verian DISTINTA CANTIDAD DE IMAGEN y la diferencia de desempenyo
   mezclaria calidad del kernel con campo de vision. Con recorte fijo, todas las
   condiciones ven EXACTAMENTE los mismos pixeles del original.

2. LA IDENTIDAD ES RECORTE PURO, no una convolucion con delta de Dirac (§6.3).
   ⚠ Medido el 2026-09-08 con ESTA implementacion: la delta da un resultado IDENTICO
   bit a bit (diferencia maxima 0,00e+00), porque los 80 terminos que sobran son
   exactamente 0*x y el que queda es 1,0*x. O sea que aqui el motivo del §6.3 NO es
   un ruido numerico observado. Sigue valiendo igualmente, y por una razon mejor: asi
   la linea base no depende de COMO se implemente la convolucion. Con einsum sale
   exacto; por FFT no saldria, y la identidad es aquello contra lo que se mide todo
   lo demas.

3. LA NORMA SE NORMALIZA AL RECIBIR EL KERNEL (§5.4). Un kernel multiplicado por un
   escalar es el MISMO detector pero con salida de otra magnitud, lo que cambia la
   escala de las activaciones, la de los gradientes y la velocidad efectiva de
   aprendizaje. Con epocas fijas, un kernel de salida grande PARECERIA mejor sin
   serlo. Normalizando, la unica diferencia entre kernels es su FORMA.

4. LA ESTANDARIZACION ES UN CONTROL, NO UNA VARIABLE (§6.5). mu y sigma salen SOLO de
   `train` filtrado y se aplican a las tres particiones. Sin ella, un kernel de suma
   positiva (tipo Gauss) hereda la media alta del fondo blanco y uno de suma cero
   (tipo Sobel) no, y esa diferencia de DISTRIBUCION se contaria como calidad.
   No se corren condiciones con y sin estandarizacion.
"""

from __future__ import annotations

import numpy as np

MARCO = 146          # tras reducir /4
DESCARTE = 9         # §6.2, fijo
FINAL = 128          # MARCO - 2*DESCARTE
K_MAX = 19           # §5.2, congelado
# El span de centros de la rejilla en el marco de 128 (§7.1/§7.5): 0, 8, ..., 120.
# Es el denominador de la normalizacion de coordenadas: con el, la salida [0,1] del
# soft-argmax cubre EXACTAMENTE lo que la rejilla puede senyalar. Normalizar por 127
# dejaria una franja de salida inalcanzable y desperdiciaria rango de medida.
SPAN = 120.0


def normalizar_kernel(k: np.ndarray) -> np.ndarray:
    """§5.4: norma L2 = 1. La unica diferencia entre kernels pasa a ser la forma."""
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
    if not (3 <= n <= K_MAX):
        raise ValueError(f"§5.2: 3 <= k <= {K_MAX}, y es {n}. K_MAX esta CONGELADO: si "
                         f"hace falta mas, se construye un banco nuevo")


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
    # Convolucion 'valid' por correlacion cruzada con el kernel volteado. Se hace con
    # stride tricks porque scipy no esta en el venv y una convolucion 146x146 con
    # k<=19 sobre 1000 imagenes cuesta poco; lo importante es que sea EXACTA.
    kf = k[::-1, ::-1]
    lado = MARCO - n + 1
    ventanas = np.lib.stride_tricks.sliding_window_view(x, (n, n), axis=(1, 2))
    y = np.einsum("nijkl,kl->nij", ventanas, kf, optimize=True).astype(np.float32)
    assert y.shape[1] == lado, f"conv valid dio {y.shape[1]}, se esperaba {lado}"
    r = recorte_de(n)
    out = y[:, r:r + FINAL, r:r + FINAL]
    assert out.shape[1:] == (FINAL, FINAL), f"tras recortar: {out.shape}"
    return np.ascontiguousarray(out)


def estadisticos(x_train: np.ndarray) -> tuple[float, float]:
    """§6.5: mu y sigma SOLO de `train` filtrado."""
    return float(x_train.mean()), float(x_train.std())


def estandarizar(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma <= 0: el filtrado dejo una imagen constante")
    return ((x - mu) / sigma).astype(np.float32)


def etiquetas_128(e584: np.ndarray) -> np.ndarray:
    """§6.4: (coord/4) - 9. Devuelve (N,4) float32 en pixeles del marco de 128."""
    return (np.asarray(e584, dtype=np.float32) / 4.0 - DESCARTE).astype(np.float32)


def normalizar_coords(e128: np.ndarray) -> np.ndarray:
    """Pixeles del marco de 128 -> [0,1] sobre el span de la rejilla (§7.3)."""
    return (np.asarray(e128, dtype=np.float32) / SPAN).astype(np.float32)


def desnormalizar_coords(c: np.ndarray) -> np.ndarray:
    return (np.asarray(c, dtype=np.float32) * SPAN).astype(np.float32)


# --------------------------------------------------------------- comprobacion
def _main() -> int:
    fallos = []

    def ok(que, obtenido, esperado):
        bien = obtenido == esperado
        print(f"  {'ok ' if bien else 'FALLA'}  {que:44} {obtenido}"
              f"{'' if bien else f'  (esperado {esperado})'}")
        if not bien:
            fallos.append(que)

    print("\nPipeline del §6\n")
    # §6.2: la tabla entera, que es el invariante que hace comparables las condiciones
    for k in range(3, K_MAX + 1, 2):
        conv = MARCO - k + 1
        r = recorte_de(k)
        ok(f"k={k:<2} valid={conv} recorte={r} → descarte total", (k - 1) // 2 + r, DESCARTE)
    ok("identidad → descarte total", recorte_de(None), DESCARTE)

    rng = np.random.default_rng(0)
    x = rng.random((3, MARCO, MARCO)).astype(np.float32) * 4080

    # todas las condiciones salen con la MISMA forma
    for k in (None, 3, 9, 19):
        kk = None if k is None else rng.random((k, k)).astype(np.float32)
        ok(f"forma de salida con k={k}", aplicar(x, kk).shape[1:], (FINAL, FINAL))

    # §5.4: la norma se va, y por tanto la ESCALA del kernel no cambia nada
    base = rng.standard_normal((9, 9)).astype(np.float32)
    a, b = aplicar(x, base), aplicar(x, base * 137.0)
    # El criterio es RELATIVO, no absoluto: la salida vale ~9e3, asi que un `atol` de
    # 1e-4 pide mas precision de la que float32 tiene y falla siempre. Medido el
    # 2026-09-08: el error relativo es 2,7e-7, o sea 2,3 x el eps de float32 (1,19e-7)
    # -- redondeo normal, no un fallo. El umbral 1e-5 deja sitio a eso y sigue cazando
    # cualquier error de verdad, que seria de orden 1.
    rel = float(np.abs(a - b).max() / np.abs(a).max())
    ok("§5.4 escalar el kernel no cambia la salida (rel < 1e-5)", rel < 1e-5, True)
    print(f"         (error relativo medido: {rel:.2e}; eps float32 = "
          f"{np.finfo(np.float32).eps:.2e})")
    ok("§5.4 norma L2 tras normalizar", round(float(np.linalg.norm(normalizar_kernel(base))), 6), 1.0)

    # §6.3: la identidad es recorte PURO -- coincide exactamente con el recorte manual
    ok("§6.3 identidad = recorte central exacto",
       bool(np.array_equal(aplicar(x, None), x[:, 9:137, 9:137])), True)

    # una delta de Dirac se PARECE a la identidad pero no es identica: por eso no se usa
    delta = np.zeros((9, 9), dtype=np.float32); delta[4, 4] = 1.0
    dif = float(np.abs(aplicar(x, delta) - aplicar(x, None)).max())
    print(f"         (delta de Dirac contra recorte puro: diferencia maxima {dif:.2e} "
          f"→ aqui la delta NO mete ruido; §6.3 vale igual, ver el docstring)")

    # §5.1/§5.3: el contrato se niega antes de convolucionar
    for mala, motivo in ((np.zeros((4, 4), np.float32), "k par"),
                         (np.zeros((21, 21), np.float32), "k > K_MAX"),
                         (np.zeros((5, 7), np.float32), "no cuadrado")):
        try:
            aplicar(x, mala)
            ok(f"§5 rechaza {motivo}", False, True)
        except ValueError:
            ok(f"§5 rechaza {motivo}", True, True)

    # §6.5
    xs = aplicar(x, None)
    mu, sd = estadisticos(xs)
    z = estandarizar(xs, mu, sd)
    ok("§6.5 media ~0 tras estandarizar", round(float(z.mean()), 5), 0.0)
    ok("§6.5 desviacion ~1 tras estandarizar", round(float(z.std()), 4), 1.0)

    # §6.4 sobre muestras conocidas
    e = np.array([[68, 512, 68, 512]], dtype=np.float32)
    ok("§6.4 (68,512) → (8,119)", tuple(etiquetas_128(e)[0][:2]), (8.0, 119.0))
    ok("§7.3 normalizar 120 px → 1.0", float(normalizar_coords(np.array([120.0]))[0]), 1.0)

    print()
    if fallos:
        print(f"✗ {len(fallos)} fallo(s): " + ", ".join(fallos) + "\n")
        return 1
    print("El pipeline del §6 cumple su contrato.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
