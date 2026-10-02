#!/usr/bin/env python3
"""El entrenamiento de un brazo de `bor-ae`. REANUDABLE: se corta y se sigue donde iba.

    python nn/entrenar_local.py --brazo k09-s1                 300 epocas (el defecto)
    python nn/entrenar_local.py --brazo k09-s1 --hilos 2       para correr varios a la vez
    python nn/entrenar_local.py --comprobar    prueba la reanudacion SIN entrenar
    python nn/entrenar_local.py --tanteo-lambda   el tanteo que FIJA lambda (k=9, 60 epocas)

⚠ EL NOMBRE DEL FICHERO ES UN CONTRATO con el freno (`cerrable.mjs` casa
`entrenar_local\\.py`). Con otro nombre diria «nada corriendo» con 300 epocas vivas.

SIN ETIQUETA: se reconstruyen las MISMAS ventanas que usa `bor-k` (todas: las de borde y
las de al azar). LAMBDA no se elige aqui: sale del tanteo de `nn/tanteo_lambda.py` con la
regla de `instrucciones/02-criterio.md`, escrita antes de correrlo.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import datos
from modelo import BRAZOS, R, construir, delta

AQUI = Path(__file__).resolve().parent
PESOS = AQUI / "pesos"

LOTE = 128
LR = 0.01
# ✅ CONGELADO el 2026-10-02 por el tanteo (resultados/tanteo-lambda.json) y la regla de
# 02-criterio.md: el menor candidato fue 0 (delta 0,25, R2 0,988). Con dispersion, identidad.
LAMBDA = 0.0
EPOCAS_DEF = 300


def cargar(parte: str) -> torch.Tensor:
    return torch.from_numpy(datos.ventanas(parte)[0])


def perdida(red, x: torch.Tensor, lam: float):
    """(perdida, mse en R, media de z en O)."""
    xh, z = red(x)
    a, b = R
    mse = ((xh - x)[..., a:b, a:b] ** 2).mean()
    l1 = red.z_en_O(z).mean()
    return mse + lam * l1, mse, l1, z


def medir(red, x: torch.Tensor, lam: float, var_r: float) -> dict:
    with torch.no_grad():
        p, mse, l1, z = perdida(red, x, lam)
        activa = (red.z_en_O(z) > 0).float().mean().item()
    return {"val": p.item(), "mse": mse.item(), "r2": 1 - mse.item() / var_r,
            "l1": l1.item(), "activa": activa, "delta": delta(red.kernel())}


def varianza_r(x: torch.Tensor) -> float:
    a, b = R
    return float(x[..., a:b, a:b].var())


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


def bucle(red, opt, Xtr, Xva, lam: float, ep0: int, epocas: int, al_acabar_epoca=None):
    """El bucle de entrenamiento, compartido por `entrenar` y por el tanteo de lambda:
    el tanteo tiene que entrenar EXACTAMENTE como se entrenara despues."""
    var_r = varianza_r(Xva)
    for ep in range(ep0 + 1, epocas + 1):
        t0 = time.time()
        red.train()
        orden = torch.randperm(len(Xtr))
        suma = n = 0
        for i in range(0, len(orden), LOTE):
            idx = orden[i:i + LOTE]
            opt.zero_grad()
            p, *_ = perdida(red, Xtr[idx], lam)
            p.backward(); opt.step()
            suma += p.item() * len(idx); n += len(idx)
        red.eval()
        fila = {"epoca": ep, "train": suma / n, **medir(red, Xva, lam, var_r),
                "s": round(time.time() - t0, 2)}
        if al_acabar_epoca:
            al_acabar_epoca(fila)
    return red


def entrenar(brazo: str, epocas: int, desde_cero: bool) -> int:
    if LAMBDA is None:
        raise SystemExit("✗ LAMBDA no esta congelado. Corre `nn/tanteo_lambda.py`, aplica la "
                         "regla de instrucciones/02-criterio.md y escribe el numero aqui.")
    k, semilla = BRAZOS[brazo]
    dir_b = PESOS / brazo
    dir_b.mkdir(parents=True, exist_ok=True)
    ultimo, mejor_f = dir_b / "last.pt", dir_b / "best.pt"
    config = {"brazo": brazo, "k": k, "semilla": semilla, "lote": LOTE, "lr": LR,
              "lambda": LAMBDA, "dataset": datos.DATASET, "lado": datos.LADO,
              "semilla_ventanas": datos.SEMILLA_VENTANAS}
    torch.manual_seed(semilla); np.random.seed(semilla); random.seed(semilla)
    red = construir(k, semilla)
    opt = torch.optim.Adam(red.parameters(), lr=LR)
    ep0, estado = 0, {"mejor": float("inf")}
    if ultimo.exists() and not desde_cero:
        est = torch.load(ultimo, map_location="cpu", weights_only=False)
        red.load_state_dict(est["modelo"]); opt.load_state_dict(est["optimizador"])
        _poner_rng(est["rng"])
        ep0, estado["mejor"] = est["epoca"], est["mejor_val"]
        print(f"reanudo {brazo} en la epoca {ep0}", flush=True)
    else:
        print(f"empiezo {brazo} desde cero ({red.n_parametros()} parametros)", flush=True)
    if ep0 >= epocas:
        print(f"ya esta en la epoca {ep0} >= {epocas}: nada que hacer")
        return 0
    Xtr, Xva = cargar("train"), cargar("val")
    reg = dir_b / "metrics.jsonl"

    def al_acabar(fila):
        with reg.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_redondear(fila)) + "\n")
        if fila["val"] < estado["mejor"]:
            estado["mejor"] = fila["val"]
            _guardar(mejor_f, red, opt, fila["epoca"], estado["mejor"], config)
        _guardar(ultimo, red, opt, fila["epoca"], estado["mejor"], config)
        if fila["epoca"] % 10 == 0 or fila["epoca"] == ep0 + 1:
            print(f"  {brazo} ep {fila['epoca']:>3} val {fila['val']:.5f} · R2 {fila['r2']:.3f} · "
                  f"activa {100 * fila['activa']:.0f}% · delta {fila['delta']:.2f} · "
                  f"{fila['s']}s", flush=True)

    bucle(red, opt, Xtr, Xva, LAMBDA, ep0, epocas, al_acabar)
    return 0


# EL TANTEO DE LAMBDA. La regla esta escrita en `instrucciones/02-criterio.md` ANTES de
# correrlo, y aqui se codifica TAL CUAL para que el veredicto no dependa de quien lo lea.
TANTEO_K = 9
TANTEO_SEMILLA = 1
TANTEO_EPOCAS = 60
TANTEO_LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)
MAX_DELTA = 0.5
MIN_R2 = 0.5
MIN_ACTIVA = 0.01


def regla_lambda(filas: list[dict]) -> tuple[float | None, str]:
    """La regla del 02-criterio: el MENOR lambda que ya no da una delta, que todavia
    reconstruye y cuyo codigo no esta muerto. Si ninguno, None (y bor-ae no va al banco)."""
    buenos = [f for f in filas if f["delta"] <= MAX_DELTA and f["r2"] >= MIN_R2
              and f["activa"] >= MIN_ACTIVA]
    if not buenos:
        return None, ("NINGUN lambda cumple delta <= 0,5, R2 >= 0,5 y codigo vivo >= 1 %: "
                      "con un solo filtro, este autoencoder da la identidad o nada. "
                      "Sus kernels NO van al banco, y se le dice al dueno con esta tabla")
    elegido = min(buenos, key=lambda f: f["lambda"])
    return elegido["lambda"], (f"lambda = {elegido['lambda']}: el menor que cumple "
                               f"(delta {elegido['delta']:.2f}, R2 {elegido['r2']:.3f}, "
                               f"activa {100 * elegido['activa']:.1f} %)")


def tanteo_lambda() -> int:
    salida = AQUI.parent / "resultados" / "tanteo-lambda.json"
    if salida.exists():
        print(f"✗ {salida.name} ya existe: el tanteo se corre UNA vez y su resultado manda. "
              f"Borralo a mano si de verdad hay que repetirlo, y di por que.")
        return 1
    Xtr, Xva = cargar("train"), cargar("val")
    filas = []
    for lam in TANTEO_LAMBDAS:
        torch.manual_seed(TANTEO_SEMILLA); np.random.seed(TANTEO_SEMILLA); random.seed(TANTEO_SEMILLA)
        red = construir(TANTEO_K, TANTEO_SEMILLA)
        opt = torch.optim.Adam(red.parameters(), lr=LR)
        ultima = {}
        t0 = time.time()
        bucle(red, opt, Xtr, Xva, lam, 0, TANTEO_EPOCAS, ultima.update)
        fila = {"lambda": lam, **{k: ultima[k] for k in ("mse", "r2", "l1", "activa", "delta")},
                "segundos": round(time.time() - t0, 1),
                "kernel": [[round(v, 4) for v in fila_k] for fila_k in red.kernel().tolist()]}
        filas.append(fila)
        print(f"  lambda {lam:<5} · R2 {fila['r2']:.3f} · delta {fila['delta']:.2f} · "
              f"activa {100 * fila['activa']:.1f} % · l1 {fila['l1']:.4f} · {fila['segundos']} s",
              flush=True)
    elegido, motivo = regla_lambda(filas)
    salida.parent.mkdir(exist_ok=True)
    salida.write_text(json.dumps({
        "k": TANTEO_K, "semilla": TANTEO_SEMILLA, "epocas": TANTEO_EPOCAS, "lr": LR,
        "regla": {"max_delta": MAX_DELTA, "min_r2": MIN_R2, "min_activa": MIN_ACTIVA,
                  "donde": "instrucciones/02-criterio.md"},
        "filas": filas, "elegido": elegido, "motivo": motivo,
        "fecha": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n",
        encoding="utf-8")
    print(f"\n{motivo}\n(escrito en {salida})")
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
                    and tuple(est["modelo"]["conv.weight"].shape) == (1, 1, k, k))
            ok &= bien
            print(f"  k={k}: guarda y recupera; conv.weight (1,1,{k},{k}) ... {'ok' if bien else 'FALLA'}")
    print("el mecanismo de reanudacion funciona." if ok else "✗ la reanudacion NO funciona")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--brazo", choices=sorted(BRAZOS))
    p.add_argument("--epocas", type=int, default=EPOCAS_DEF)
    p.add_argument("--desde-cero", action="store_true")
    p.add_argument("--hilos", type=int)
    p.add_argument("--comprobar", action="store_true")
    p.add_argument("--tanteo-lambda", action="store_true")
    a = p.parse_args()
    if a.hilos:
        torch.set_num_threads(a.hilos)
    if a.comprobar:
        return comprobar()
    if a.tanteo_lambda:
        return tanteo_lambda()
    if not a.brazo:
        p.error("hace falta --brazo (o --comprobar)")
    return entrenar(a.brazo, a.epocas, a.desde_cero)


if __name__ == "__main__":
    raise SystemExit(main())
