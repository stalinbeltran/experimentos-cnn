#!/usr/bin/env python3
"""COMPOSITORES POR LADO, pedidos por el dueño el 2026-10-08 (corrige una mala lectura: el brazo `compartido` —un detector
que mira cada lado y se queda con el MÁXIMO— no era lo pedido). Lo pedido es el compositor de dígitos entrenado:

  1. con CADA LADO por separado (8 compositores, uno por dirección);
  2. con los 8 a la vez (el completo);
  3. GRADUAL: con 1 lado, con 2, con 3 … con 8, para ver si añadir lados mejora o empeora.

Dos fuentes de «lado», las dos sin entrenar ningún detector (0 $, minutos en el dev):

  A  bordes   el mapa del borde de ese lado tal cual (la capa fija), reducido a 8×8 por medias: 64 números por lado
  B  detectores   los 13 detectores YA ENTRENADOS del brazo `compartido`, aplicados a ESE lado solo (`Detector.por_canal`,
              sin el máximo): 13 mapas 8×8 = 832 números por lado. Es el mismo detector para los 8 lados; lo que cambia
              es qué lados ve el compositor

El compositor es el de feat-ind32 (Linear, Adam, 300 épocas, lr 1e-2, L2 1e-3, 3 semillas), en 180 y en 3823 de train,
probado en los 1617 de val y en el cuartil de gruesos reales. Sin desplazar. El orden gradual importa, así que hay dos:
por ángulo (→ ↘ ↓ ↙ ← ↖ ↑ ↗) y por parejas opuestas (→ ← ↓ ↑ ↘ ↖ ↙ ↗).

    python nn/lados.py   → resultados/lados.json y lados.txt
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import componer as C                            # noqa: E402
import datos                                    # noqa: E402
import modelo                                   # noqa: E402

RES = AQUI.parent / "resultados"
ORDENES = {"por ángulo": [0, 1, 2, 3, 4, 5, 6, 7], "por parejas opuestas": [0, 4, 2, 6, 1, 5, 3, 7]}


@torch.no_grad()
def mapas_por_lado(x: np.ndarray, reds: list, lote: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """→ A (N, 8, 64) bordes reducidos a 8×8 · B (N, 8, 13·64) σ de los 13 detectores aplicados a cada lado."""
    A = np.zeros((len(x), 8, 64), np.float32); B = np.zeros((len(x), 8, len(reds) * 64), np.float32)
    for i in range(0, len(x), lote):
        xt = torch.from_numpy(np.ascontiguousarray(x[i:i + lote], dtype=np.float32))
        bor = reds[0].canales_borde(xt)                                         # (n, 8, 32, 32)
        A[i:i + lote] = bor.reshape(len(xt), 8, 8, 4, 8, 4).mean((3, 5)).reshape(len(xt), 8, 64).numpy()
        for j, r in enumerate(reds):
            B[i:i + lote, :, j * 64:(j + 1) * 64] = torch.sigmoid(r.por_canal(xt)).reshape(len(xt), 8, 64).numpy()
    return A, B


def medir(F: np.ndarray, lados: list[int], y, tr, va, gr) -> dict:
    x = F[:, lados].reshape(len(F), -1)
    Ws = [C.ajustar(x[tr], y[tr], s) for s in C.SEMILLAS]
    return {"val": C.r4(np.mean([C.acierto(W, x[va], y[va]) for W in Ws])),
            "gruesos": C.r4(np.mean([C.acierto(W, x[va][gr], y[va][gr]) for W in Ws]))}


def main() -> int:
    t0 = time.time(); torch.set_num_threads(2)
    reds = [modelo.cargar(AQUI / "pesos-compartido" / f / "best.pt")[0] for f in datos.F.CON_TRAZO]
    dg = datos.digitos(); y = dg["y"]
    va = dg["val"]; tinta = dg["x"].reshape(len(y), -1).sum(1); yv = y[va]
    gr = np.zeros(int(va.sum()), bool)
    for c in range(10):
        m = yv == c; gr[m] = tinta[va][m] >= np.quantile(tinta[va][m], .75)
    A, B = mapas_por_lado(dg["x"], reds)
    print(f"mapas: {time.time() - t0:.0f} s", flush=True)
    out = {"lados": list(modelo.FLECHAS), "ordenes": {k: [modelo.FLECHAS[i] for i in v] for k, v in ORDENES.items()},
           "cuando": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for fuente, F in (("A bordes", A), ("B detectores", B)):
        for reg, tr in (("180", dg["train"]), ("3823", dg["origen"] != "windep")):
            r = out.setdefault(fuente, {}).setdefault(reg, {})
            r["uno"] = {modelo.FLECHAS[k]: medir(F, [k], y, tr, va, gr) for k in range(8)}
            for nombre, orden in ORDENES.items():
                r[nombre] = [medir(F, orden[:k], y, tr, va, gr) for k in range(1, 9)]
            print(f"  {fuente} · {reg}: uno " + " ".join(f"{a}{v['val']:.3f}" for a, v in r["uno"].items()) +
                  " · gradual por ángulo " + " ".join(f"{v['val']:.3f}" for v in r["por ángulo"]) +
                  f"  [{time.time() - t0:.0f} s]", flush=True)
    RES.mkdir(exist_ok=True)
    (RES / "lados.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    L = []
    for fuente in ("A bordes", "B detectores"):
        for reg in ("180", "3823"):
            r = out[fuente][reg]
            L += [f"== {fuente} · compositor entrenado con {reg} · acierto en val (1617) / gruesos reales ==",
                  "un solo lado:  " + "  ".join(f"{a} {v['val']:.3f}/{v['gruesos']:.3f}" for a, v in r["uno"].items())]
            for nombre in ORDENES:
                L.append(f"gradual {nombre} ({' '.join(out['ordenes'][nombre])}): " +
                         "  ".join(f"{k + 1}:{v['val']:.3f}" for k, v in enumerate(r[nombre])))
            L.append("")
    (RES / "lados.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L)); print(f"→ resultados/lados.json en {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
