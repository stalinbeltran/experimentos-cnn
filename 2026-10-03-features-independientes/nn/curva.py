#!/usr/bin/env python3
"""Corrida 11: curva de acierto según el nº de dígitos de train del compositor. Test FIJO de 717 (de las 1617 de
val); reserva de 900 para ampliar train. Criterio en instrucciones/02-criterio.md § «Corrida 11».

    python nn/curva.py        → resultados/curva.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compositor as C                          # noqa: E402
import compositor_comb as CC                    # noqa: E402

RES = Path(__file__).resolve().parent.parent / "resultados"
SEMILLA_REPARTO = 2026
RESERVA_POR_CLASE = 90
TAMANOS = (36, 90, 180, 360, 540, 900, 1080)
BANCOS = {"fino+grueso": ["fino", "grueso"], "todos": ["fino", "grueso", "dig", "cae5", "cae3"]}


def reparto(y, tr):
    """(orden de train ampliable, máscara de test). El orden pone primero los 180 originales —barajados
    estratificadamente para que cualquier prefijo de 36 o 90 sea equilibrado— y después la reserva, también
    intercalada por clase: un prefijo de longitud N es siempre estratificado y los tamaños quedan anidados."""
    rng = np.random.default_rng(SEMILLA_REPARTO)
    val = np.flatnonzero(~tr); reserva = []
    for c in range(10):
        reserva += list(rng.choice(val[y[val] == c], RESERVA_POR_CLASE, replace=False))
    reserva = np.array(reserva)
    test = ~tr.copy(); test[reserva] = False

    def intercalar(idx):
        por = [list(rng.permutation(idx[y[idx] == c])) for c in range(10)]
        out = []
        while any(por):
            for p in por:
                if p: out.append(p.pop())
        return np.array(out)

    return np.concatenate([intercalar(np.flatnonzero(tr)), intercalar(reserva)]), test


def lineal(x, y, idx, test, sem):
    return C.logistica(x[idx], y[idx], x[test], y[test], sem)["acc_val"]


def combinante(s, y, idx, test, sem):
    torch.manual_seed(sem)
    red = CC.Combinante(s.shape[1]); opt = torch.optim.Adam(red.parameters(), lr=CC.LR, weight_decay=CC.L2)
    xt, yt = torch.from_numpy(s[idx]), torch.from_numpy(y[idx])
    for _ in range(CC.EPOCAS):
        opt.zero_grad(); Fn.cross_entropy(red(xt), yt).backward(); opt.step()
    with torch.no_grad():
        return float((red(torch.from_numpy(s[test])).argmax(1).numpy() == y[test]).mean())


def main() -> int:
    partes = {g: CC.mapas(g) for g in ("fino", "grueso", "dig", "cae5", "cae3")}
    y, tr = partes["fino"][1], partes["fino"][2]
    orden, test = reparto(y, tr)
    assert np.bincount(y[orden[:180]]).tolist() == [18] * 10 and int(test.sum()) == 717
    salida = {"test": int(test.sum()), "tamanos": TAMANOS, "semillas": C.SEMILLAS, "curva": {}}
    t0 = time.time()
    for banco, grupos in BANCOS.items():
        s = np.concatenate([partes[g][0] for g in grupos], 1); x = s.reshape(len(s), -1)
        for n in TAMANOS:
            idx = orden[:n]
            lin = [lineal(x, y, idx, test, sem) for sem in C.SEMILLAS]
            com = [combinante(s, y, idx, test, sem) for sem in C.SEMILLAS]
            fila = {"lineal": round(float(np.mean(lin)), 4), "lineal_sd": round(float(np.std(lin, ddof=1)), 4),
                    "combinante": round(float(np.mean(com)), 4), "combinante_sd": round(float(np.std(com, ddof=1)), 4)}
            salida["curva"].setdefault(banco, {})[n] = fila
            print(f"{banco:<12} N={n:>4} · lineal {fila['lineal']:.4f} ± {fila['lineal_sd']:.4f} · combinante {fila['combinante']:.4f} ± {fila['combinante_sd']:.4f}"
                  f"  [{time.time() - t0:.0f} s]", flush=True)
            (RES / "curva.json").write_text(json.dumps(salida, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
