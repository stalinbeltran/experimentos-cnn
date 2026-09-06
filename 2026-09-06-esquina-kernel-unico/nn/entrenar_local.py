#!/usr/bin/env python3
"""El entrenamiento de un brazo. REANUDABLE: se corta y se sigue donde iba.

    python nn/entrenar_local.py --brazo k03 --epocas 60
    python nn/entrenar_local.py --brazo k03 --epocas 120     # sigue desde donde iba
    python nn/entrenar_local.py --brazo k03 --desde-cero     # empieza de nuevo
    python nn/entrenar_local.py --comprobar                  # prueba el mecanismo SIN entrenar

⚠ EL NOMBRE DEL FICHERO ES UN CONTRATO, no una preferencia.
    `telegram-coordinator/scripts/cerrable.mjs:137` casa `entrenar_local\\.py` en su
    lista `TRABAJOS`. Es asi como este entrenamiento aparece en el veredicto
    "¿se puede apagar este server?" que el dueno lee desde el movil. Con otro
    nombre, el freno diria "nada corriendo" con 60 epocas vivas.

QUE SIGNIFICA REANUDABLE AQUI
    El estado de una reanudacion no son solo los pesos: son los pesos, el
    optimizador Y las tres semillas (torch, numpy, python). Sin las semillas, la
    misma corrida partida en dos no da lo mismo que entera, y entonces "reanudar"
    seria "empezar otra cosa parecida". Se guarda cada epoca, porque perder una
    epoca es barato y perder sesenta no.

    `metrics.jsonl` es de SOLO ANADIR: al reanudar no se reescribe, se sigue. Un
    historial que se reescribe no es un historial.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from modelo import BRAZOS, construir

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
DATOS = EXP / "datos"
PESOS = AQUI / "pesos"

LOTE = 128
LR = 0.05
LAMBDA_COORD = 0.01   # equilibra MSE en px (~64 al empezar) contra BCE (~0,7)
SEMILLA = 1


def _cargar(parte: str):
    z = np.load(DATOS / f"{parte}.npz", allow_pickle=True)
    v = torch.from_numpy(z["ventanas"].astype(np.float32) / 255.0).unsqueeze(1)
    return v, (torch.from_numpy(z["existe"]).float(),
               torch.from_numpy(z["x"]), torch.from_numpy(z["y"]))


def _perdida(salida, objetivo):
    logit, px, py, _ = salida
    existe, x, y = objetivo
    bce = F.binary_cross_entropy_with_logits(logit, existe)
    hay = existe > 0.5
    if hay.any():   # las coordenadas solo tienen sentido donde la esquina EXISTE
        coord = ((px[hay] - x[hay]) ** 2 + (py[hay] - y[hay]) ** 2).mean()
    else:
        coord = torch.zeros((), device=logit.device)
    return bce + LAMBDA_COORD * coord, bce.item(), coord.item()


def _estado_rng() -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate()}


def _poner_rng(e: dict) -> None:
    torch.set_rng_state(e["torch"])
    np.random.set_state(e["numpy"])
    random.setstate(e["python"])


def _guardar(destino: Path, red, opt, epoca: int, mejor: float) -> None:
    """Escribe primero a un temporal y renombra: un corte a media escritura
    dejaria un checkpoint truncado, que es peor que no tenerlo."""
    tmp = destino.with_suffix(".tmp")
    torch.save({"modelo": red.state_dict(), "optimizador": opt.state_dict(),
                "epoca": epoca, "mejor_val": mejor, "rng": _estado_rng()}, tmp)
    tmp.replace(destino)


def entrenar(brazo: str, epocas: int, desde_cero: bool) -> int:
    dir_b = PESOS / brazo
    dir_b.mkdir(parents=True, exist_ok=True)
    ultimo, mejor_f = dir_b / "last.pt", dir_b / "best.pt"

    torch.manual_seed(SEMILLA); np.random.seed(SEMILLA); random.seed(SEMILLA)
    red = construir(brazo, SEMILLA)
    opt = torch.optim.Adam(red.parameters(), lr=LR)
    ep0, mejor = 0, float("inf")

    if ultimo.exists() and not desde_cero:
        est = torch.load(ultimo, map_location="cpu", weights_only=False)
        red.load_state_dict(est["modelo"]); opt.load_state_dict(est["optimizador"])
        _poner_rng(est["rng"])
        ep0, mejor = est["epoca"], est["mejor_val"]
        print(f"reanudo {brazo} en la epoca {ep0} (mejor val {mejor:.4f})")
    else:
        print(f"empiezo {brazo} desde cero")
    if ep0 >= epocas:
        print(f"ya esta en la epoca {ep0} >= {epocas}: nada que hacer")
        return 0

    Vtr, Ytr = _cargar("train")
    Vva, Yva = _cargar("val")
    reg = dir_b / "metrics.jsonl"

    for ep in range(ep0 + 1, epocas + 1):
        t0 = time.time()
        red.train()
        orden = torch.randperm(len(Vtr))
        suma = n = 0
        for i in range(0, len(orden), LOTE):
            idx = orden[i:i + LOTE]
            opt.zero_grad()
            p, *_ = _perdida(red(Vtr[idx]), tuple(t[idx] for t in Ytr))
            p.backward(); opt.step()
            suma += p.item() * len(idx); n += len(idx)
        red.eval()
        with torch.no_grad():
            pv, bce, coord = _perdida(red(Vva), Yva)
        fila = {"epoca": ep, "train": suma / n, "val": pv.item(), "bce": bce,
                "coord_mse": coord, "beta": float(red.log_beta.exp()),
                "segundos": round(time.time() - t0, 2)}
        with reg.open("a", encoding="utf-8") as f:      # SOLO ANADIR
            f.write(json.dumps(fila) + "\n")
        if pv.item() < mejor:
            mejor = pv.item()
            _guardar(mejor_f, red, opt, ep, mejor)
        _guardar(ultimo, red, opt, ep, mejor)           # cada epoca: reanudable siempre
        print(f"  ep {ep:>3} train {fila['train']:.4f} val {fila['val']:.4f} "
              f"(bce {bce:.3f} coord {coord:.1f} px2) beta {fila['beta']:.2f} "
              f"{fila['segundos']}s")
    return 0


def comprobar() -> int:
    """Prueba el MECANISMO de reanudacion sin entrenar ni una epoca."""
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as d:
        for brazo in BRAZOS:
            red = construir(brazo, SEMILLA)
            opt = torch.optim.Adam(red.parameters(), lr=LR)
            f = Path(d) / f"{brazo}.pt"
            _guardar(f, red, opt, 7, 0.123)
            est = torch.load(f, map_location="cpu", weights_only=False)
            otra = construir(brazo, SEMILLA + 99)          # otra inicializacion a proposito
            distinta = not torch.equal(otra.conv.weight, red.conv.weight)
            otra.load_state_dict(est["modelo"])
            igual = torch.equal(otra.conv.weight, red.conv.weight)
            bien = distinta and igual and est["epoca"] == 7 and abs(est["mejor_val"] - 0.123) < 1e-9
            ok &= bien
            print(f"  {brazo}: guarda y recupera pesos, epoca y mejor_val ... {'ok' if bien else 'FALLA'}")
    print("el mecanismo de reanudacion funciona." if ok else "✗ la reanudacion NO funciona")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--brazo", choices=sorted(BRAZOS))
    p.add_argument("--epocas", type=int, default=60)
    p.add_argument("--desde-cero", action="store_true")
    p.add_argument("--comprobar", action="store_true")
    a = p.parse_args()
    if a.comprobar:
        return comprobar()
    if not a.brazo:
        p.error("hace falta --brazo (o --comprobar)")
    return entrenar(a.brazo, a.epocas, a.desde_cero)


if __name__ == "__main__":
    raise SystemExit(main())
