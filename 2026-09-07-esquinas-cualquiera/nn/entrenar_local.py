#!/usr/bin/env python3
"""El entrenamiento de un brazo. REANUDABLE: se corta y se sigue donde iba.

    python nn/entrenar_local.py --brazo k07 --epocas 300
    python nn/entrenar_local.py --comprobar      prueba la reanudacion SIN entrenar
    python nn/entrenar_local.py --suelos         mide los SUELOS sin entrenar nada

⚠ EL NOMBRE DEL FICHERO ES UN CONTRATO, no una preferencia.
    `telegram-coordinator/scripts/cerrable.mjs` casa `entrenar_local\\.py` en su
    lista declarada `TRABAJOS`. Es asi como este entrenamiento aparece en el
    veredicto "¿se puede apagar este server?" que el dueno lee desde el movil.

QUE SIGNIFICA REANUDABLE AQUI
    El estado no son solo los pesos: son los pesos, el optimizador Y las tres
    semillas (torch, numpy, python). Sin las semillas, la misma corrida partida
    en dos no da lo mismo que entera. Se guarda cada epoca; `metrics.jsonl` es de
    SOLO ANADIR.
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

from datos import cargar
from modelo import BRAZOS, construir, simetria

AQUI = Path(__file__).resolve().parent
PESOS = AQUI / "pesos"

LOTE = 128
LR = 0.05
# ⚠ MEDIDO, no elegido a ojo. La regla es la de `esq-k`: "los dos terminos parten
# IGUALES". Se re-mide con `--suelos` porque la etiqueta cambio (una esquina de
# dos clases, 40,1 % de positivas) aunque las ventanas sean las mismas.
LAMBDA_COORD = 0.03
SEMILLA = 1
EPOCAS_DEF = 300


def _tensores(parte: str):
    d = cargar(parte)
    v = torch.from_numpy(d["ventanas"].astype(np.float32) / 255.0).unsqueeze(1)
    return v, (torch.from_numpy(d["existe"]).float(),
               torch.from_numpy(d["x"]), torch.from_numpy(d["y"]))


def _perdida(salida, objetivo):
    logit, px, py, _ = salida
    existe, x, y = objetivo
    bce = F.binary_cross_entropy_with_logits(logit, existe)
    hay = existe > 0.5
    if hay.any():    # las coordenadas solo tienen sentido donde la esquina EXISTE
        coord = ((px[hay] - x[hay]) ** 2 + (py[hay] - y[hay]) ** 2).mean()
    else:
        coord = torch.zeros((), device=logit.device)
    return bce + LAMBDA_COORD * coord, bce, coord


def _metricas(salida, objetivo, clase=None) -> dict:
    """Lo que se mira, sobre las ventanas donde la esquina EXISTE.

    ⚠ Y ADEMAS PARTIDO POR CLASE (`tl` / `br`), aunque la tarea ya no distinga:
    es la unica forma de ver si "detecta cualquiera" se ha resuelto de verdad o
    se ha resuelto detectando SIEMPRE la misma. Un 50 % global con 100 % en `tl`
    y 0 % en `br` no es medio exito: es el fallo de `esq-2d` con otro nombre."""
    logit, px, py, _ = salida
    existe, x, y = objetivo
    hay = existe > 0.5
    m = {"n_positivas": int(hay.sum())}
    if hay.any():
        d = ((px[hay] - x[hay]) ** 2 + (py[hay] - y[hay]) ** 2).sqrt()
        m["err_px"] = d.mean().item()
        m["acierto_1px"] = (d <= 1).float().mean().item()
        m["acierto_2px"] = (d <= 2).float().mean().item()
    pred = torch.sigmoid(logit) > 0.5
    tp = (pred & hay).sum().item()
    den = pred.sum().item() + hay.sum().item()
    m["f1_existe"] = (2 * tp / den) if den > 0 else 0.0
    if clase is not None:
        for c in ("tl", "br"):
            sel = hay & torch.from_numpy(np.array([str(k) == f"esquina-{c}" for k in clase]))
            if sel.any():
                d = ((px[sel] - x[sel]) ** 2 + (py[sel] - y[sel]) ** 2).sqrt()
                m[f"acierto_2px_{c}"] = (d <= 2).float().mean().item()
                m[f"err_px_{c}"] = d.mean().item()
    return m


def _estado_rng() -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate()}


def _poner_rng(e: dict) -> None:
    torch.set_rng_state(e["torch"]); np.random.set_state(e["numpy"]); random.setstate(e["python"])


def _redondear(fila: dict) -> dict:
    """Seis cifras significativas: el `repr` de un float ocupa ~17 caracteres y
    ninguna metrica de aqui se lee mas alla de la cuarta."""
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in fila.items()}


def _guardar(destino: Path, red, opt, epoca: int, mejor: float) -> None:
    """Escribe a un temporal y renombra: un corte a media escritura dejaria un
    checkpoint truncado, que es peor que no tenerlo."""
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
        print(f"empiezo {brazo} desde cero ({red.n_parametros()} parametros)")
    if ep0 >= epocas:
        print(f"ya esta en la epoca {ep0} >= {epocas}: nada que hacer")
        return 0

    Vtr, Ytr = _tensores("train")
    Vva, Yva = _tensores("val")
    clase_va = cargar("val")["clase"]
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
            salida = red(Vva)
            pv, bce, coord = _perdida(salida, Yva)
            met = _metricas(salida, Yva, clase_va)
        sim, anti = simetria(red.kernel())
        fila = {"epoca": ep, "train": suma / n, "val": pv.item(), "bce": bce.item(),
                "coord_mse": coord.item(), "beta": float(red.log_beta.exp()),
                "simetrico": sim, "antisimetrico": anti,
                **met, "segundos": round(time.time() - t0, 2)}
        with reg.open("a", encoding="utf-8") as f:      # SOLO ANADIR
            f.write(json.dumps(_redondear(fila)) + "\n")
        if pv.item() < mejor:
            mejor = pv.item()
            _guardar(mejor_f, red, opt, ep, mejor)
        _guardar(ultimo, red, opt, ep, mejor)
        print(f"  ep {ep:>3} val {fila['val']:.4f} · acierto<=2px "
              f"{100*fila.get('acierto_2px', 0):>5.1f}% (tl {100*fila.get('acierto_2px_tl',0):.0f}"
              f" br {100*fila.get('acierto_2px_br',0):.0f}) · err {fila.get('err_px',0):.2f} px "
              f"· f1 {fila['f1_existe']:.3f} · S {100*sim:.0f}% · beta {fila['beta']:.2f} "
              f"· {fila['segundos']}s")
    return 0


def suelos() -> int:
    """Los SUELOS: que da cada estructura SIN entrenar, y con que lambda parten
    iguales los dos terminos de la perdida. Es lo que el criterio necesita ANTES
    de mirar, y por eso vive aqui y no en una libreta."""
    Vva, Yva = _tensores("val")
    clase_va = cargar("val")["clase"]
    print(f"val: {len(Vva)} ventanas · {int(Yva[0].sum())} con esquina "
          f"(tl o br, sin distinguir)\n")
    print(f"{'brazo':>6} {'acierto<=2px':>13} {'tl':>6} {'br':>6} {'<=1px':>7} {'err px':>8} "
          f"{'f1':>7} {'bce':>7} {'mse':>8} {'lambda':>8} {'S%':>5}")
    for brazo in BRAZOS:
        red = construir(brazo, SEMILLA); red.eval()
        with torch.no_grad():
            salida = red(Vva)
            _, bce, coord = _perdida(salida, Yva)
            met = _metricas(salida, Yva, clase_va)
        sim, _ = simetria(red.kernel())
        print(f"{brazo:>6} {100*met['acierto_2px']:>12.1f}% "
              f"{100*met.get('acierto_2px_tl',0):>5.1f}% {100*met.get('acierto_2px_br',0):>5.1f}% "
              f"{100*met['acierto_1px']:>6.1f}% {met['err_px']:>8.2f} {met['f1_existe']:>7.3f} "
              f"{bce.item():>7.3f} {coord.item():>8.2f} "
              f"{bce.item()/coord.item():>8.4f} {100*sim:>4.0f}%")
    print(f"\n  lambda congelada en {LAMBDA_COORD} (regla: los dos terminos parten iguales)")
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
            otra = construir(brazo, SEMILLA + 99)         # otra inicializacion a proposito
            distinta = not torch.equal(otra.conv.weight, red.conv.weight)
            otra.load_state_dict(est["modelo"])
            igual = torch.equal(otra.conv.weight, red.conv.weight)
            bien = distinta and igual and est["epoca"] == 7 and abs(est["mejor_val"] - 0.123) < 1e-9
            ok &= bien
            print(f"  {brazo:>6}: guarda y recupera pesos, epoca y mejor_val ... "
                  f"{'ok' if bien else 'FALLA'}")
    print("el mecanismo de reanudacion funciona." if ok else "✗ la reanudacion NO funciona")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--brazo", choices=sorted(BRAZOS))
    p.add_argument("--epocas", type=int, default=EPOCAS_DEF)
    p.add_argument("--desde-cero", action="store_true")
    p.add_argument("--comprobar", action="store_true")
    p.add_argument("--suelos", action="store_true")
    a = p.parse_args()
    if a.comprobar:
        return comprobar()
    if a.suelos:
        return suelos()
    if not a.brazo:
        p.error("hace falta --brazo (o --comprobar / --suelos)")
    return entrenar(a.brazo, a.epocas, a.desde_cero)


if __name__ == "__main__":
    raise SystemExit(main())
