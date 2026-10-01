#!/usr/bin/env python3
"""`bor-pca`: el kernel de cada k es la COMPONENTE PRINCIPAL de los parches de borde.

    python nn/pca.py              los 9 k: escribe nn/pesos/kNN/best.pt, resultados/pca.json
                                  y la figura resultados/componentes.png
    python nn/pca.py --comprobar  la PCA sobre parches sinteticos con una direccion conocida

Sin entrenar, sin semilla de entrenamiento, segundos en el dev. Es la referencia barata
entre los clasicos del banco (`gauss`, `sobel`) y los kernels aprendidos.

QUE SE CALCULA, por k
    Los parches de `train` (k*k valores cada uno, en TINTA), MENOS LA MEDIA de todos los
    parches (PCA de libro: se centra el conjunto, no cada parche). Se descompone la
    covarianza por SVD y se guardan las DOS primeras direcciones:

      PC1  -> `nn/pesos/kNN/best.pt` como `conv.weight` (1, 1, k, k): es lo que va al banco
      PC2  -> en el mismo checkpoint (`pc2`), y NO va al banco. Decidido ANTES de mirar
              (`instrucciones/02-criterio.md`): el PC1 de parches de borde suele ser el
              escalon, pero no se da por hecho, y por eso se guarda el PC2 para poder verlo.

EL SIGNO de un vector propio es arbitrario. Se fija: la componente de mayor |valor| sale
POSITIVA. El banco normaliza la norma y estandariza la salida, asi que el signo no cambia
nada alli; fijarlo solo hace que dos corridas den el mismo fichero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import datos

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
PESOS = AQUI / "pesos"
RESULTADOS = EXP / "resultados"
K_BARRIDO = (3, 5, 7, 9, 11, 13, 15, 17, 19)


def _signo(v: np.ndarray) -> np.ndarray:
    return v if v[np.argmax(np.abs(v))] >= 0 else -v


def componentes(x: np.ndarray, n: int = 2):
    """(vectores (n, d), fraccion de varianza explicada (n,)) de los datos (P, d)."""
    xc = x - x.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = s ** 2
    return np.stack([_signo(vt[i]) for i in range(n)]), var[:n] / var.sum()


def calcular() -> int:
    import torch                                            # noqa: PLC0415
    RESULTADOS.mkdir(exist_ok=True)
    filas = []
    for k in K_BARRIDO:
        x, _ = datos.parches(k, "train")
        v, frac = componentes(x)
        pc1, pc2 = (v[i].reshape(k, k).astype(np.float32) for i in range(2))
        dir_b = PESOS / f"k{k:02d}"
        dir_b.mkdir(parents=True, exist_ok=True)
        torch.save({"modelo": {"conv.weight": torch.from_numpy(pc1).view(1, 1, k, k)},
                    "pc2": torch.from_numpy(pc2).view(1, 1, k, k),
                    "varianza_explicada": [float(f) for f in frac],
                    "epoca": None,
                    "config": {"k": k, "parches": int(len(x)), "dataset": datos.DATASET,
                               "puntos_por_borde": datos.PUNTOS_POR_BORDE,
                               "semilla_puntos": datos.SEMILLA_PUNTOS}},
                   dir_b / "best.pt")
        fila = {"k": k, "parches": int(len(x)), "var_pc1": round(float(frac[0]), 4),
                "var_pc2": round(float(frac[1]), 4), "suma_pc1": round(float(pc1.sum()), 4),
                "sha256_16_pc1": hashlib.sha256(np.ascontiguousarray(pc1).tobytes()).hexdigest()[:16]}
        filas.append(fila)
        print(f"  k={k:>2}: {fila['parches']} parches · PC1 explica {100 * frac[0]:.1f} % · "
              f"PC2 {100 * frac[1]:.1f} % · suma PC1 {fila['suma_pc1']:+.3f}")
    (RESULTADOS / "pca.json").write_text(json.dumps(
        {"fecha": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "dataset": datos.DATASET, "filas": filas}, indent=1) + "\n", encoding="utf-8")
    figura()
    print(f"\n{len(filas)} kernels en {PESOS}/kNN/best.pt · resumen en {RESULTADOS / 'pca.json'}")
    return 0


def figura() -> None:
    """PC1 (arriba) y PC2 (abajo) de cada k, cada uno estirado a su propio rango: gris =
    0, blanco = positivo, negro = negativo."""
    import torch                                            # noqa: PLC0415
    from PIL import Image                                   # noqa: PLC0415
    celda, pad = 19 * 6, 6
    hoja = Image.new("L", (len(K_BARRIDO) * (celda + pad) + pad, 2 * (celda + pad) + pad), 255)
    for i, k in enumerate(K_BARRIDO):
        ck = torch.load(PESOS / f"k{k:02d}" / "best.pt", map_location="cpu", weights_only=False)
        for fila, w in enumerate((ck["modelo"]["conv.weight"], ck["pc2"])):
            a = w[0, 0].numpy()
            g = 127.5 + 127.5 * a / max(1e-9, float(np.abs(a).max()))
            im = Image.fromarray(g.round().astype(np.uint8)).resize((celda * k // 19, celda * k // 19),
                                                                     Image.NEAREST)
            hoja.paste(im, (pad + i * (celda + pad) + (celda - im.width) // 2,
                            pad + fila * (celda + pad) + (celda - im.height) // 2))
    hoja.save(RESULTADOS / "componentes.png")


def comprobar() -> int:
    """Parches sinteticos con UNA direccion dominante conocida: la PCA tiene que darla."""
    rng = np.random.default_rng(0)
    k = 9
    escalon = np.zeros((k, k), np.float32); escalon[:, k // 2:] = 1.0
    d = (escalon - escalon.mean()).ravel(); d /= np.linalg.norm(d)
    x = rng.normal(size=(2000, 1))[:, :] * 3.0 * d[None, :] + rng.normal(size=(2000, k * k)) * 0.1
    v, frac = componentes(x.astype(np.float32))
    coseno = abs(float(v[0] @ d))
    bien = coseno > 0.99 and frac[0] > 0.5
    print(f"  coseno PC1 con la direccion sembrada: {coseno:.4f} · varianza PC1 {100 * frac[0]:.1f} % "
          f"{'ok' if bien else 'FALLA'}")
    return 0 if bien else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comprobar", action="store_true")
    a = ap.parse_args()
    return comprobar() if a.comprobar else calcular()


if __name__ == "__main__":
    raise SystemExit(main())
