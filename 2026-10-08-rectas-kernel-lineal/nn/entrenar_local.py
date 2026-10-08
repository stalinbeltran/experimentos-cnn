#!/usr/bin/env python3
"""La rejilla de `rect-lin`: k × escalas × entrenamiento × N × semilla. Reanudable: salta lo que ya está en
resultados/rejilla.jsonl. Cada línea lleva las métricas del banco y los 2 kernels aprendidos.

⚠ Se llama `entrenar_local.py` a propósito: es el nombre que casa el freno (`cerrable.mjs`).

    python nn/entrenar_local.py                       la rejilla entera (432)
    python nn/entrenar_local.py --solo 7,3,continua,32,0    un brazo (para medir el tiempo)
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import datos as D  # noqa: E402
import evaluar as E  # noqa: E402
import modelo as M  # noqa: E402

KS = (5, 7, 9)
ESCALAS = (1, 2, 3)
MODOS = ("continua", "punteada")
NS = (4, 8, 16, 32, 64, 128, 256, 1000)
SEMILLAS = (0, 1, 2)
EPOCAS, LR = 400, 0.03
SALIDA = E.RES / "rejilla.jsonl"


def entrenar(x: np.ndarray, y: np.ndarray, k: int, nv: int, sem: int) -> tuple[M.Lineal, float]:
    torch.manual_seed(sem)
    m = M.Lineal(k, nv, sem)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    xt = torch.from_numpy(x)
    # 5 clases: las 4 orientaciones y «nada», con el logit de «nada» fijo en 0 → «detecta» sigue siendo logit > 0.
    # (Con BCE sobre 4 salidas, 7 de cada 8 objetivos son 0 y el modelo colapsaba a «nunca»: medido 2026-10-08.)
    cls = torch.from_numpy(np.where(y.sum(1) > 0, y.argmax(1), 4))
    cero = torch.zeros(len(xt), 1)
    for _ in range(EPOCAS):
        opt.zero_grad(); perdida = F.cross_entropy(torch.cat([m(xt), cero], 1), cls); perdida.backward(); opt.step()
    m.eval()
    return m, float(perdida.detach())


def hechos() -> set:
    if not SALIDA.exists():
        return set()
    return {(r["k"], r["escalas"], r["entreno"], r["n"], r["semilla"]) for r in map(json.loads, SALIDA.read_text().splitlines())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--solo", help="k,escalas,entreno,N,semilla")
    p.add_argument("--no-guardar", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(2)
    e = D.cargar(D.ENTRENO)
    xb, b = E.banco()
    if a.solo:
        k, nv, modo, n, s = a.solo.split(","); brazos = [(int(k), int(nv), modo, int(n), int(s))]
    else:
        brazos = list(itertools.product(KS, ESCALAS, MODOS, NS, SEMILLAS))
    ya = hechos()
    pendientes = [br for br in brazos if br not in ya]
    print(f"{len(brazos)} brazos, {len(brazos) - len(pendientes)} ya hechos, {len(pendientes)} por hacer", flush=True)
    E.RES.mkdir(exist_ok=True)
    for i, (k, nv, modo, n, s) in enumerate(pendientes, 1):
        t0 = time.time()
        x, y = D.entreno(e, modo, n)
        m, perdida = entrenar(x, y, k, nv, s)
        met = E.metricas(E.logits(m, xb), b)
        fila = {"k": k, "escalas": nv, "entreno": modo, "n": n, "semilla": s, "epocas": EPOCAS, "lr": LR,
                "perdida_final": E.r4(perdida), "segundos": round(time.time() - t0, 1), "parametros": m.n_parametros(),
                **met, "a": E.r4(m.a), "c": E.r4(m.c),
                "K0": m.K[0, 0].detach().numpy().round(4).tolist(), "K45": m.K[1, 0].detach().numpy().round(4).tolist()}
        if not a.no_guardar:
            with SALIDA.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(fila, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(pendientes)}] k={k} esc={nv} {modo} N={n} s={s}: fino {met['recall_fino']} grueso "
              f"{met['recall_grueso']} fp {met['fp_total']}  ({fila['segundos']} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
