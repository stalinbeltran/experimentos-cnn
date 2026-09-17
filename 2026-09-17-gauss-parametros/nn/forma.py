#!/usr/bin/env python3
"""La FORMA de una gaussiana: lo unico que el banco llega a ver. Solo numpy.

    python nn/forma.py                tabla de sigma x k, con el truncamiento
    python nn/forma.py --k 9          solo ese ancho

POR QUE ESTE FICHERO EXISTE: `k` CASI NO ES UN PARAMETRO
=======================================================
Parece que hay dos mandos, `k` y `sigma`, y hay UNO Y MEDIO. El motivo es del
banco, no de la gaussiana: su §6.2 descarta **9 px por lado sea cual sea `k`**, asi
que `k` no cambia cuanta imagen se ve. Lo unico que hace `k` es TRUNCAR la
gaussiana, y una gaussiana truncada donde ya no queda nada es la misma gaussiana.

Medido el 2026-09-17 (`python nn/forma.py`), a igual `sigma`, comparando el kernel
de ancho `k` contra el de ancho 9 rellenado de ceros, los dos normalizados en L2:

    sigma = 1,0    k=11, 15, 19 contra k=9    L2 dif = 5,6e-06   -> el MISMO filtro
    sigma = 1,5    k=11, 15, 19 contra k=9    L2 dif = 4,8e-03   -> practicamente
    sigma = 3,0    k=11 contra k=9            L2 dif = 2,2e-01   -> ya NO

O sea que el par util no es «(k, sigma) en una rejilla»: es **`sigma`**, con `k`
elegido lo bastante ancho para no cortarla. Por eso la app pone `sigma` en el eje
grande y de `k` solo ensenya UNA cosa: cuanto esta cortando.

⚠ Y por eso hay un tope real que no esta en la formula: `k <= 19` esta CONGELADO
(§5.2 del banco). Con `sigma` = 5 y `k` = 19 se corta el 11 % de la gaussiana, asi
que por encima de `sigma` ~ 3 lo que se evalua ya no es «una gaussiana de ese
sigma»: es una gaussiana recortada, que se va pareciendo a una caja plana. No es un
fallo -- es el limite del banco, y hay que verlo al elegir en vez de descubrirlo
despues.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import banco                                          # noqa: E402

TRUNCAMIENTO_AVISO = 0.01      # 1 %: por encima, la app lo dice


def huella(k: np.ndarray) -> str:
    """El `sha256_16` TAL COMO LO CALCULA EL BANCO: sobre el kernel SIN normalizar.

    Copiado de `banco-k/nn/evaluar_kernel.py:82`. Es el apreton de manos: la app
    ensenya esta huella y `kernels.py --gauss --esperado <huella>` se niega si el
    kernel que el banco genera no es bit a bit el que se miro. Sin esto, la copia
    del pipeline podria desviarse y el dueno elegiria mirando una cosa mientras el
    banco mide otra, en silencio.
    """
    return hashlib.sha256(np.ascontiguousarray(
        np.asarray(k, dtype=np.float32)).tobytes()).hexdigest()[:16]


def truncamiento(k: int, sigma: float) -> float:
    """Que fraccion de la gaussiana se queda FUERA de la ventana de k x k.

    Se compara contra la misma gaussiana en una ventana lo bastante grande para
    que lo que falte sea despreciable (k + 12*sigma, siempre impar).
    """
    ancho = int(k + 12 * sigma) | 1
    if ancho <= k:
        return 0.0
    g = banco.gauss(k, sigma).sum()
    G = banco.gauss(ancho, sigma).sum()
    return float(max(0.0, 1.0 - g / G))


def k_minimo(sigma: float, tope: float = TRUNCAMIENTO_AVISO) -> int | None:
    """El `k` mas pequenyo del contrato que no corta mas de `tope`. None si ninguno."""
    for k in banco.ks_validos():
        if truncamiento(k, sigma) <= tope:
            return k
    return None


def describir(k: int, sigma: float) -> dict:
    """Los numeros que ayudan a DECIDIR, y ninguno que puntue.

    ⚠ Aqui no hay ninguna cifra de «calidad» a proposito. El criterio que declara
    es el del banco (§2.1/§2.2), escrito antes de mirar; una puntuacion inventada
    en esta pantalla competiria con el y se habria escrito DESPUES de mirar (R13).
    Todo lo de abajo describe la FORMA del filtro, no su merito.
    """
    g = banco.gauss(k, sigma)
    gn = banco.normalizar_kernel(g)
    c = k // 2
    delta = np.zeros((k, k), np.float32)
    delta[c, c] = 1.0
    caja = banco.normalizar_kernel(np.ones((k, k), np.float32))
    mitad = max(1, k // 4)
    trunc = truncamiento(k, sigma)
    kmin = k_minimo(sigma)
    return {
        "k": int(k),
        "sigma": round(float(sigma), 4),
        "sigma_banco": round(banco.sigma_banco(k), 4),
        "es_el_del_banco": bool(abs(sigma - banco.sigma_banco(k)) < 1e-6),
        "huella": huella(g),
        "centro_sobre_esquina": float(gn[c, c] / max(abs(gn[0, 0]), 1e-30)),
        "masa_central": float(gn[c - mitad:c + mitad + 1, c - mitad:c + mitad + 1].sum()
                              / gn.sum()),
        "dist_identidad": float(np.linalg.norm(gn - banco.normalizar_kernel(delta))),
        "dist_caja": float(np.linalg.norm(gn - caja)),
        "truncamiento": trunc,
        "trunca_mucho": bool(trunc > TRUNCAMIENTO_AVISO),
        "k_minimo": kmin,
        "k_de_sobra": bool(kmin is not None and k > kmin),
        "recorte": banco.recorte_de(k),
    }


def tabla(k: int = 0) -> int:
    """La tabla, SIN tocar `sys.argv`.

    ⚠ Separada de `_main` a proposito: el modo `forma` de la app la llama, y con
    el parseo dentro volvia a leer los argumentos de la app entera y fallaba con
    «unrecognized arguments: forma». Una funcion que parsea `sys.argv` no se puede
    llamar desde otro programa; una que recibe sus datos, si.
    """
    ks = [k] if k else banco.ks_validos()
    sigmas = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    print("\nQue fraccion de la gaussiana CORTA la ventana de k x k")
    print("(el banco descarta 9 px por lado sea cual sea `k`, asi que esto es")
    print(" lo UNICO que `k` cambia)\n")
    print(f"  {'sigma':>6} " + " ".join(f"k={k:<9d}" for k in ks))
    for s in sigmas:
        fila = " ".join(f"{truncamiento(k, s):<11.2e}" for k in ks)
        print(f"  {s:6.2f} {fila}")

    print(f"\n  El `k` mas pequenyo que corta <= {TRUNCAMIENTO_AVISO:.0%}:\n")
    print(f"  {'sigma':>6}  {'k minimo':>9}  {'sigma del banco con ese k':>26}")
    for s in sigmas:
        km = k_minimo(s)
        sb = f"k/6 = {banco.sigma_banco(km):.2f}" if km else "—"
        print(f"  {s:6.2f}  {str(km) if km else 'NINGUNO':>9}  {sb:>26}")
    print("\n  ⚠ 'NINGUNO' significa que ni k=19 (§5.2, congelado) contiene esa")
    print("    gaussiana: lo que se evaluaria seria una gaussiana RECORTADA.\n")
    return 0


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=0)
    return tabla(ap.parse_args().k)


if __name__ == "__main__":
    raise SystemExit(_main())
