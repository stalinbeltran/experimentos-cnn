#!/usr/bin/env python3
"""Que muestras se miran, y sobre todo CUALES NO. Solo numpy + expcnn.

    python nn/muestras.py            que hay, cuantas quedan libres y cuales se eligen
    python nn/muestras.py --n 10     las N que la app va a ensenyar

LA RESERVA DEL §3.7 ES LO PRIMERO DE ESTE FICHERO, NO UN DETALLE
===============================================================
`banco-k` reserva una familia tipografica (`LiberationMono`) y un rango de
interlineado ([1.45, 1.60]) para uso EXCLUSIVO suyo. Su regla, literal:

    «los procedimientos que producen kernels no pueden usar esta familia ni este
     rango de interlineado»

ELEGIR `sigma` MIRANDO MUESTRAS ES UN PROCEDIMIENTO QUE PRODUCE KERNELS. Da igual
que la eleccion la haga un ojo humano y no un optimizador: el §3.7 no habla de
como se elige, habla de QUE DATOS se miran. Un `sigma` afinado sobre las muestras
reservadas saldria optimista al evaluarlo en el banco, y el banco no tendria
forma de notarlo.

Asi que aqui la reserva:

1. **se comprueba en codigo, no se recuerda.** `filtrar_reserva()` es el unico
   camino por el que la app obtiene imagenes, y `comprobar_reserva()` vuelve a
   mirar el resultado. Es el mismo patron que `bor-p` (`comprobar_reserva()` en
   su `generar_paginas.py`), aplicado al lado del CONSUMO en vez del de la
   generacion;
2. **se lee del manifiesto del dataset**, no de una constante tecleada aqui. Si
   el banco cambia su reserva, esto la sigue. Si el manifiesto no la declara, se
   NIEGA en vez de suponer que no hay reserva -- suponer «no hay» es justo el
   fallo silencioso que la reserva existe para evitar.

DE QUE PARTICION, Y POR QUE `train`
===================================
De `train`. Dos motivos distintos, y los dos importan:

- `eval` son las 800 con las que el banco DECLARA. Mirarlas para elegir un
  hiperparametro es fuga de conjunto de prueba, que es un fallo mas viejo y mas
  conocido que el §3.7.
- `train` es ademas de donde el §6.5 saca mu y sigma, o sea que la red ya la ve.
  Mirar lo que la red ya ve no anyade fuga ninguna.

⚠ MU Y SIGMA SALEN DE LAS LIBRES, NO DE LAS 100
================================================
El §6.5 del banco los calcula sobre `train` entero. Aqui no se puede: 45 de esas
100 estan reservadas. Se calculan sobre las 55 libres, asi que la vista previa NO
es bit a bit lo que el banco vera.

Medido el 2026-09-17 (`python nn/muestras.py --desvio`), el desvio maximo en z:

    identidad      0,4248        gauss k=9 s=1,5    0,3889
    gauss k=9 s=0,6  0,4073      gauss k=19 s=5,0   0,4092

Lo que decide es que el desvio es PRACTICAMENTE EL MISMO en las cuatro: es un
desplazamiento comun a todas las condiciones, no algo que afecte a una mas que a
otra. La comparacion ENTRE parametros -- que es para lo que sirve este
experimento -- no se mueve. La cifra absoluta si, y por eso se dice aqui en vez
de presentar la vista previa como identica al banco.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent))          # para `expcnn`
sys.path.insert(0, str(AQUI))                        # para `banco`

from expcnn import exigir_dataset                    # noqa: E402
import banco                                         # noqa: E402

DATASET = "parrafos1000-584px-r4-r20260908b"         # el de `banco-k`
PARTICION = "train"
SEMILLA = 17                                         # la de la eleccion de muestras
N_POR_DEFECTO = 10


# ------------------------------------------------------------------ la reserva
def reserva_del_manifiesto(man: dict) -> tuple[str, tuple[float, float]]:
    """La reserva §3.7 tal como la DECLARA el dataset. Se niega si no esta.

    No se teclea aqui una constante: si el banco cambiara su reserva, una copia
    local quedaria desfasada y este experimento creeria estar cumpliendola.
    """
    r = man.get("reserva_§3.7") or man.get("reserva_3.7")
    if not isinstance(r, dict) or "fuente" not in r or "interlineado" not in r:
        raise RuntimeError(
            f"El manifiesto de '{man.get('nombre', '?')}' no declara la reserva §3.7.\n"
            f"  No se puede seguir: sin saber que esta reservado, cualquier muestra que\n"
            f"  se ensenye puede serlo, y un sigma elegido sobre ella sale optimista en\n"
            f"  el banco sin que nada lo avise.\n"
            f"  → comprueba que el dataset es el de `banco-k` y que su manifiesto no se\n"
            f"    ha reescrito.")
    lo, hi = r["interlineado"]
    return str(r["fuente"]), (float(lo), float(hi))


def es_reservada(m: dict, fuente_res: str, dens_res: tuple[float, float]) -> bool:
    """¿Esta muestra cae en la reserva? Basta con UNA de las dos condiciones."""
    if m["fuente"] == fuente_res:
        return True
    return dens_res[0] <= m["interlineado"] <= dens_res[1]


def comprobar_reserva(metas: list[dict], fuente_res: str,
                      dens_res: tuple[float, float]) -> list[str]:
    """La segunda mirada: ¿se ha colado alguna reservada? Devuelve los motivos."""
    malos = []
    for m in metas:
        if m["fuente"] == fuente_res:
            malos.append(f"muestra i={m['i']} usa {fuente_res}, reservada por §3.7")
        if dens_res[0] <= m["interlineado"] <= dens_res[1]:
            malos.append(f"muestra i={m['i']} tiene interlineado {m['interlineado']}, "
                         f"dentro de la reserva {list(dens_res)} del §3.7")
    return malos


# ------------------------------------------------------------------ la carga
class Muestras:
    """Las imagenes libres de reserva de `train`, y de donde salieron."""

    def __init__(self, dataset: str = DATASET, particion: str = PARTICION):
        raiz = exigir_dataset(dataset)
        self.dataset = dataset
        self.particion = particion
        self.manifiesto = json.loads((raiz / "manifiesto.json").read_text(encoding="utf-8"))
        self.fuente_res, self.dens_res = reserva_del_manifiesto(self.manifiesto)

        z = np.load(raiz / f"{particion}.npz")
        todas = z["imagenes"]
        etiquetas = z["etiquetas"]
        indices = z["indices"]
        meta = json.loads(str(np.load(raiz / "meta.npz", allow_pickle=True)["meta"]))

        metas = [meta[i] for i in indices]
        libre = np.array([not es_reservada(m, self.fuente_res, self.dens_res)
                          for m in metas])
        self.n_particion = len(metas)
        self.n_reservadas = int((~libre).sum())

        self.imagenes = todas[libre].astype(np.float32)
        self.etiquetas = etiquetas[libre]
        self.meta = [m for m, ok in zip(metas, libre) if ok]

        malos = comprobar_reserva(self.meta, self.fuente_res, self.dens_res)
        if malos:                                    # no deberia pasar nunca
            raise RuntimeError("§3.7: se colo dato reservado:\n  " + "\n  ".join(malos))
        if len(self.imagenes) == 0:
            raise RuntimeError(f"§3.7 no deja ninguna muestra de '{particion}'.")

    # --- lo que la app usa
    def elegir(self, n: int = N_POR_DEFECTO, semilla: int = SEMILLA) -> np.ndarray:
        """Los indices (dentro de las libres) de las N que se ensenyan.

        Semilla FIJA: la app tiene que ensenyar las MISMAS muestras cada vez que
        se abre, o comparar dos parametros mirados en dos momentos distintos no
        compara nada. Se cambia con `--semilla` a proposito, no por azar.
        """
        n = min(n, len(self.imagenes))
        return np.sort(np.random.default_rng(semilla).choice(
            len(self.imagenes), size=n, replace=False))

    def estadisticos(self, kernel) -> tuple[float, float]:
        """§6.5 sobre las LIBRES (ver el aviso de la cabecera)."""
        return banco.estadisticos(banco.aplicar(self.imagenes, kernel))

    def resumen(self) -> dict:
        return {
            "dataset": self.dataset,
            "particion": self.particion,
            "n_particion": self.n_particion,
            "n_reservadas": self.n_reservadas,
            "n_libres": len(self.imagenes),
            "reserva": {"fuente": self.fuente_res, "interlineado": list(self.dens_res)},
        }


# ------------------------------------------------------------------ informe
def _desvio() -> int:
    """Mide el desvio de usar solo las libres para mu/sigma (el aviso de arriba)."""
    m = Muestras()
    print(f"\nDesvio en z por calcular mu/sigma sobre las {len(m.imagenes)} libres\n"
          f"en vez de las {m.n_particion} de `train` (§6.5 del banco)\n")
    print(f"  {'condicion':<18} {'mu libres':>12} {'sd libres':>11} {'desvio z max':>13}")
    raiz = exigir_dataset(m.dataset)
    todas = np.load(raiz / f"{m.particion}.npz")["imagenes"].astype(np.float32)
    for nom, ker in (("identidad", None),
                     ("gauss k=9 s=0,6", banco.gauss(9, 0.6)),
                     ("gauss k=9 s=1,5", banco.gauss(9, 1.5)),
                     ("gauss k=19 s=5,0", banco.gauss(19, 5.0))):
        xs = banco.aplicar(todas, ker)
        mu1, sd1 = banco.estadisticos(xs)
        xl = banco.aplicar(m.imagenes, ker)
        mu2, sd2 = banco.estadisticos(xl)
        d = float(np.abs((xs - mu1) / sd1 - (xs - mu2) / sd2).max())
        print(f"  {nom:<18} {mu2:12.1f} {sd2:11.2f} {d:13.4f}")
    print("\n  El desvio es practicamente el mismo en las cuatro: es un desplazamiento")
    print("  COMUN, no algo que afecte a una condicion mas que a otra. La comparacion")
    print("  entre parametros no se mueve.\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_POR_DEFECTO)
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    ap.add_argument("--desvio", action="store_true", help="mide el desvio de mu/sigma")
    a = ap.parse_args()
    if a.desvio:
        return _desvio()

    m = Muestras()
    r = m.resumen()
    print(f"\ndataset   {r['dataset']}  ({r['particion']})")
    print(f"reserva   §3.7: fuente {r['reserva']['fuente']} · interlineado "
          f"{r['reserva']['interlineado']}  (del manifiesto, no tecleada aqui)")
    print(f"muestras  {r['n_particion']} en la particion · {r['n_reservadas']} "
          f"RESERVADAS y descartadas · {r['n_libres']} libres\n")
    elegidas = m.elegir(a.n, a.semilla)
    print(f"Las {len(elegidas)} que se ensenyan (semilla {a.semilla}):\n")
    print(f"  {'#':>3} {'i':>4} {'fuente':<16} {'cuerpo':>7} {'interlin':>9} {'gris':>6}")
    for pos, j in enumerate(elegidas):
        md = m.meta[j]
        print(f"  {pos:>3} {md['i']:>4} {md['fuente']:<16} {md['cuerpo']:>7.1f} "
              f"{md['interlineado']:>9.3f} {md['gris']:>6}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
