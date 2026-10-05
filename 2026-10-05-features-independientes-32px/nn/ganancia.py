#!/usr/bin/env python3
"""Ganancia G = acierto en val / fracción de train, pedida por el dueño el 2026-10-05, y la prueba de si es independiente del tamaño
del dataset. Lo mismo que `nn/ganancia.py` de `feat-ind` (allí, a 8×8), con los 13 detectores de 32×32 y los píxeles de
32×32. `particion()` está COPIADA de allí sin cambios: los dos evalúan los mismos dígitos (la figura lo comprueba).

Dataset BALANCEADO de T dígitos (T/10 por clase) del pool de 5620; N = p·T a train, los T − N restantes se evalúan:

    G = ((val − errores) / val) / (train / (train + val))  =  acierto en val / p      (train = N, val = T − N, p = N / T)

Definición del dueño, CORREGIDA el 2026-10-05 (la primera, «aciertos ÷ T entre N ÷ T» = aciertos / N, era un error suyo;
queda en el json como `aciertos_por_muestra` para poder comparar). Techo 1/p (acierto 100 %); azar 1/(10·p).

T ∈ {500, 1000, 2000, 4000}, p ∈ {2, 4, 10, 20, 50} % (p·T/10 entero en todas), 3 semillas.

    python nn/ganancia.py      → resultados/ganancia.json      (necesita resultados/mapas-digitos.npz: nn/aplicar.py)
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import aplicar                                  # noqa: E402
import componer as K                            # noqa: E402
import datos                                    # noqa: E402

RES = AQUI.parent / "resultados"
T_TAMANOS = (500, 1000, 2000, 4000)
FRACCIONES = (0.02, 0.04, 0.10, 0.20, 0.50)
SEMILLAS = (1, 2, 3)


def particion(y: np.ndarray, T: int, p: float, sem: int) -> tuple[np.ndarray, np.ndarray]:
    """(train, test) de un dataset BALANCEADO de T dígitos del pool (T/10 por clase); p·T/10 por clase a train. El dataset
    depende sólo de (T, sem): los train de distintos p están anidados. ⚠ COPIADA tal cual en `feat-ind32`."""
    por = T // 10; n_tr = int(round(p * por))
    if T % 10 or abs(n_tr - p * por) > 1e-9 or n_tr < 1:
        raise ValueError(f"T={T}, p={p}: p·T/10 tiene que ser un entero ≥ 1")
    rng = np.random.default_rng(1_000_000 * sem + T)
    tr, te = [], []
    for c in range(10):
        idx = rng.permutation(np.flatnonzero(y == c))[:por]
        tr.append(idx[:n_tr]); te.append(idx[n_tr:])
    return np.concatenate(tr), np.concatenate(te)


def huella(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def main() -> int:
    if not aplicar.MAPAS.is_file():
        raise SystemExit("✗ no está resultados/mapas-digitos.npz: primero nn/aplicar.py")
    m = dict(np.load(aplicar.MAPAS))
    y = m["y"].astype(np.int64)
    reps = {"detectores 32×32 (13)": m["sigma"].reshape(len(y), -1).astype(np.float32),
            "píxeles 32×32": datos.digitos(con_extra=True)["x"].reshape(len(y), -1).astype(np.float32)}
    out = {"pool": int(len(y)), "huella_y": huella(y), "T": T_TAMANOS, "p": FRACCIONES, "semillas": SEMILLAS, "casos": {}}
    hp = hashlib.sha256(); t0 = time.time()
    for nombre, x in reps.items():
        filas = []
        for T in T_TAMANOS:
            for p in FRACCIONES:
                for sem in SEMILLAS:
                    tr, te = particion(y, T, p, sem)
                    if nombre == next(iter(reps)):
                        hp.update(tr.tobytes()); hp.update(te.tobytes())
                    acc = float((K.predecir(K.ajustar(x[tr], y[tr], sem), x[te]) == y[te]).mean())
                    Tr, N = len(tr) + len(te), len(tr)
                    filas.append({"T": T, "p": p, "sem": sem, "T_real": Tr, "N": N, "n_test": len(te), "acc": round(acc, 4),
                                  "pct_train": round(N / Tr, 5), "G": round(acc / (N / Tr), 3),
                                  "aciertos_por_muestra": round(acc * len(te) / N, 3)})
            print(f"{nombre:<22} T={T:>4}: " + " · ".join(
                f"p={q:.0%} G {np.mean([f['G'] for f in filas if f['T'] == T and f['p'] == q]):.1f}" for q in FRACCIONES)
                + f"  [{time.time() - t0:.0f} s]", flush=True)
        out["casos"][nombre] = filas
    out["huella_particiones"] = hp.hexdigest()[:16]
    (RES / "ganancia.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"→ resultados/ganancia.json · huella de las particiones {out['huella_particiones']} · y {out['huella_y']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
