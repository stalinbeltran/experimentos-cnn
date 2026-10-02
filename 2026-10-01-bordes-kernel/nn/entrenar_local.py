#!/usr/bin/env python3
"""El entrenamiento de un brazo de `bor-k`. REANUDABLE: se corta y se sigue donde iba.

    python nn/entrenar_local.py --brazo k09-s1                 300 epocas (el defecto)
    python nn/entrenar_local.py --brazo k09-s1 --hilos 2       para correr varios a la vez
    python nn/entrenar_local.py --suelos       los SUELOS y el lambda que iguala la perdida
    python nn/entrenar_local.py --comprobar    prueba la reanudacion SIN entrenar

⚠ EL NOMBRE DEL FICHERO ES UN CONTRATO, no una preferencia: `telegram-coordinator/
scripts/cerrable.mjs` casa `entrenar_local\\.py` en su lista TRABAJOS. Con otro nombre,
el freno diria «nada corriendo» con 300 epocas vivas.

REANUDABLE QUIERE DECIR: pesos + optimizador + las tres semillas (torch, numpy, python),
guardados cada epoca a un temporal que se renombra. `metrics.jsonl` es de SOLO ANADIR.

LA PERDIDA
    bce (los 4 `existe`, media) + LAMBDA_COORD * |coord - verdad| (px, media sobre los
    bordes que EXISTEN). Los dos terminos parten IGUALES: LAMBDA_COORD se mide con
    `--suelos` sobre la red sin entrenar y se CONGELA antes de la primera epoca.
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

import datos
from modelo import BORDES, BRAZOS, construir

AQUI = Path(__file__).resolve().parent
PESOS = AQUI / "pesos"

LOTE = 128
# Elegido por ESTABILIDAD, no por resultado: con 0,02 la perdida de train baja sin
# oscilar en el ensayo de mecanismo (k=9, 20 epocas, 2026-10-01). Ver `REGLAS.md`.
LR = 0.02
# ✅ CONGELADO el 2026-10-02 con `--suelos`, ANTES de la primera epoca: 0,1240 es la mediana
# de los 9 k del cociente bce/l1 en la red sin entrenar (de 0,1217 a 0,1257). La regla que
# se fija es «los dos terminos parten iguales», no el numero (ver `REGLAS.md`).
LAMBDA_COORD = 0.1240
EPOCAS_DEF = 300


def cargar(parte: str):
    x, e, c, info = datos.ventanas(parte)
    return (torch.from_numpy(x), torch.from_numpy(e), torch.from_numpy(c),
            torch.from_numpy(info))


def perdida(logits, coords, existe, verdad, lam: float):
    """(perdida, bce, coord_l1). Las coordenadas solo cuentan donde HAY borde."""
    bce = F.binary_cross_entropy_with_logits(logits, existe)
    hay = existe > 0.5
    l1 = (coords - verdad).abs()[hay].mean() if hay.any() else torch.zeros(())
    return bce + lam * l1, bce, l1


def metricas(logits, coords, existe, verdad) -> dict:
    """POR BORDE, nunca el promedio: el riesgo escrito antes es que el kernel se quede
    con un eje. f1 del `existe`, error medio en px y acierto <= 2 px donde hay borde."""
    m = {}
    pred = torch.sigmoid(logits) > 0.5
    hay = existe > 0.5
    err = (coords - verdad).abs()
    for i, b in enumerate(BORDES):
        tp = (pred[:, i] & hay[:, i]).sum().item()
        den = pred[:, i].sum().item() + hay[:, i].sum().item()
        m[f"f1_{b}"] = (2 * tp / den) if den else 0.0
        if hay[:, i].any():
            ei = err[hay[:, i], i]
            m[f"err_{b}"] = ei.mean().item()
            m[f"a2_{b}"] = (ei <= 2).float().mean().item()
    return m


def _redondear(fila: dict) -> dict:
    return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in fila.items()}


def _estado_rng() -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate()}


def _poner_rng(e: dict) -> None:
    torch.set_rng_state(e["torch"]); np.random.set_state(e["numpy"]); random.setstate(e["python"])


def _guardar(destino: Path, red, opt, epoca: int, mejor: float, config: dict) -> None:
    tmp = destino.with_suffix(".tmp")
    torch.save({"modelo": red.state_dict(), "optimizador": opt.state_dict(),
                "epoca": epoca, "mejor_val": mejor, "rng": _estado_rng(),
                "config": config}, tmp)
    tmp.replace(destino)


def _exigir_lambda() -> float:
    if LAMBDA_COORD is None:
        raise SystemExit("✗ LAMBDA_COORD no esta congelado. Corre `--suelos`, escribe el "
                         "numero en el codigo y en REGLAS.md, y despues entrena.")
    return LAMBDA_COORD


def entrenar(brazo: str, epocas: int, desde_cero: bool) -> int:
    lam = _exigir_lambda()
    k, semilla = BRAZOS[brazo]
    dir_b = PESOS / brazo
    dir_b.mkdir(parents=True, exist_ok=True)
    ultimo, mejor_f = dir_b / "last.pt", dir_b / "best.pt"
    config = {"brazo": brazo, "k": k, "semilla": semilla, "lote": LOTE, "lr": LR,
              "lambda_coord": lam, "dataset": datos.DATASET, "lado": datos.LADO,
              "vaiven": datos.VAIVEN, "semilla_ventanas": datos.SEMILLA_VENTANAS}

    torch.manual_seed(semilla); np.random.seed(semilla); random.seed(semilla)
    red = construir(k, semilla)
    opt = torch.optim.Adam(red.parameters(), lr=LR)
    ep0, mejor = 0, float("inf")
    if ultimo.exists() and not desde_cero:
        est = torch.load(ultimo, map_location="cpu", weights_only=False)
        red.load_state_dict(est["modelo"]); opt.load_state_dict(est["optimizador"])
        _poner_rng(est["rng"])
        ep0, mejor = est["epoca"], est["mejor_val"]
        print(f"reanudo {brazo} en la epoca {ep0} (mejor val {mejor:.4f})", flush=True)
    else:
        print(f"empiezo {brazo} desde cero ({red.n_parametros()} parametros)", flush=True)
    if ep0 >= epocas:
        print(f"ya esta en la epoca {ep0} >= {epocas}: nada que hacer")
        return 0

    Xtr, Etr, Ctr, _ = cargar("train")
    Xva, Eva, Cva, _ = cargar("val")
    reg = dir_b / "metrics.jsonl"
    for ep in range(ep0 + 1, epocas + 1):
        t0 = time.time()
        red.train()
        orden = torch.randperm(len(Xtr))
        suma = n = 0
        for i in range(0, len(orden), LOTE):
            idx = orden[i:i + LOTE]
            opt.zero_grad()
            lo, co = red(Xtr[idx])
            p, _, _ = perdida(lo, co, Etr[idx], Ctr[idx], lam)
            p.backward(); opt.step()
            suma += p.item() * len(idx); n += len(idx)
        red.eval()
        with torch.no_grad():
            lo, co = red(Xva)
            pv, bce, l1 = perdida(lo, co, Eva, Cva, lam)
            met = metricas(lo, co, Eva, Cva)
        fila = {"epoca": ep, "train": suma / n, "val": pv.item(), "bce": bce.item(),
                "l1": l1.item(), "beta": float(red.log_beta.exp()), **met,
                "s": round(time.time() - t0, 2)}
        with reg.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_redondear(fila)) + "\n")
        if pv.item() < mejor:
            mejor = pv.item()
            _guardar(mejor_f, red, opt, ep, mejor, config)
        _guardar(ultimo, red, opt, ep, mejor, config)
        if ep % 10 == 0 or ep == ep0 + 1:
            det = " · ".join(f"{b} {fila.get(f'err_{b}', float('nan')):.2f}px/"
                             f"{100 * fila.get(f'a2_{b}', 0):.0f}%" for b in BORDES)
            print(f"  {brazo} ep {ep:>3} val {fila['val']:.4f} · {det} · "
                  f"beta {fila['beta']:.1f} · {fila['s']}s", flush=True)
    return 0


def suelos() -> int:
    """Lo que el criterio necesita ANTES de mirar: el suelo TRIVIAL (predecir siempre la
    prevalencia de cada borde y su coordenada mediana en train), el de la red SIN
    entrenar, y el lambda que iguala los dos terminos de la perdida."""
    _, Etr, Ctr, _ = cargar("train")
    _, Eva, Cva, _ = cargar("val")
    Xva = cargar("val")[0]
    print(f"val: {len(Eva)} ventanas · existe por borde "
          + " · ".join(f"{b} {int(Eva[:, i].sum())}" for i, b in enumerate(BORDES)))

    prev = Etr.mean(0).clamp(1e-4, 1 - 1e-4)
    med = torch.stack([Ctr[Etr[:, i] > 0.5, i].median() for i in range(4)])
    lo_t = torch.log(prev / (1 - prev)).expand_as(Eva)
    co_t = med.expand_as(Cva)
    mt = metricas(lo_t, co_t, Eva, Cva)
    err = (co_t - Cva).abs()
    print("\nsuelo TRIVIAL (constante de train):")
    for i, b in enumerate(BORDES):
        ei = err[Eva[:, i] > 0.5, i]
        se = ei.std().item() / max(1, len(ei)) ** 0.5
        print(f"  {b}: err {mt[f'err_{b}']:.2f} px (SE {se:.2f}) · <=2px {100 * mt[f'a2_{b}']:.1f}% · "
              f"f1 {mt[f'f1_{b}']:.3f}")

    print("\nla red SIN entrenar, y el lambda que iguala bce y l1:")
    lams = []
    for brazo, (k, s) in BRAZOS.items():
        if s != 1:
            continue
        red = construir(k, s).eval()
        with torch.no_grad():
            lo, co = red(Xva)
            _, bce, l1 = perdida(lo, co, Eva, Cva, 0.0)
        lams.append(bce.item() / l1.item())
        print(f"  {brazo}: bce {bce.item():.3f} · l1 {l1.item():.2f} px · lambda {lams[-1]:.4f}")
    print(f"\n  lambda que iguala los dos terminos: {float(np.median(lams)):.4f} "
          f"(mediana de los {len(lams)} k). LAMBDA_COORD hoy: {LAMBDA_COORD}")
    return 0


def comprobar() -> int:
    import tempfile
    ok = True
    with tempfile.TemporaryDirectory() as d:
        for k in (3, 19):
            red = construir(k, 1)
            opt = torch.optim.Adam(red.parameters(), lr=LR)
            f = Path(d) / "x.pt"
            _guardar(f, red, opt, 7, 0.123, {"k": k})
            est = torch.load(f, map_location="cpu", weights_only=False)
            otra = construir(k, 99)
            distinta = not torch.equal(otra.conv.weight, red.conv.weight)
            otra.load_state_dict(est["modelo"])
            bien = (distinta and torch.equal(otra.conv.weight, red.conv.weight)
                    and est["epoca"] == 7 and tuple(est["modelo"]["conv.weight"].shape) == (1, 1, k, k))
            ok &= bien
            print(f"  k={k}: guarda y recupera pesos y epoca; conv.weight (1,1,{k},{k}) ... "
                  f"{'ok' if bien else 'FALLA'}")
    print("el mecanismo de reanudacion funciona." if ok else "✗ la reanudacion NO funciona")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--brazo", choices=sorted(BRAZOS))
    p.add_argument("--epocas", type=int, default=EPOCAS_DEF)
    p.add_argument("--desde-cero", action="store_true")
    p.add_argument("--hilos", type=int, help="torch.set_num_threads: para varios brazos a la vez")
    p.add_argument("--comprobar", action="store_true")
    p.add_argument("--suelos", action="store_true")
    a = p.parse_args()
    if a.hilos:
        torch.set_num_threads(a.hilos)
    if a.comprobar:
        return comprobar()
    if a.suelos:
        return suelos()
    if not a.brazo:
        p.error("hace falta --brazo (o --comprobar / --suelos)")
    return entrenar(a.brazo, a.epocas, a.desde_cero)


if __name__ == "__main__":
    raise SystemExit(main())
