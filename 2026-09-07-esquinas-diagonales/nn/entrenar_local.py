#!/usr/bin/env python3
"""El entrenamiento de un brazo. REANUDABLE: se corta y se sigue donde iba.

    python nn/entrenar_local.py --brazo rot --epocas 300
    python nn/entrenar_local.py --brazo rot --epocas 300 --desde-cero
    python nn/entrenar_local.py --comprobar      prueba la reanudacion SIN entrenar
    python nn/entrenar_local.py --suelos         mide los SUELOS sin entrenar nada

⚠ EL NOMBRE DEL FICHERO ES UN CONTRATO, no una preferencia.
    `telegram-coordinator/scripts/cerrable.mjs` casa `entrenar_local\\.py` en su
    lista declarada `TRABAJOS`. Es asi como este entrenamiento aparece en el
    veredicto "¿se puede apagar este server?" que el dueno lee desde el movil.
    Con otro nombre, el freno diria "nada corriendo" con 300 epocas vivas.

QUE SIGNIFICA REANUDABLE AQUI
    El estado de una reanudacion no son solo los pesos: son los pesos, el
    optimizador Y las tres semillas (torch, numpy, python). Sin las semillas, la
    misma corrida partida en dos no da lo mismo que entera. Se guarda cada epoca.
    `metrics.jsonl` es de SOLO ANADIR: un historial que se reescribe no es un
    historial.

LA PERDIDA, Y POR QUE ESTA FORMA
    Las dos esquinas pesan IGUAL (media de las dos BCE, media de los dos MSE):
    un experimento sobre compartir un kernel no puede favorecer a una de las dos
    mitades. Los brazos `ind-*` solo tienen una esquina, asi que su media es esa.
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

from datos import DATASET
from expcnn import exigir_dataset
from modelo import BRAZOS, ESQUINAS, construir, simetria

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
PESOS = AQUI / "pesos"

LOTE = 128
LR = 0.05
# ⚠ MEDIDO, no elegido a ojo. La regla es la misma de `esq-k` -- "los dos
# terminos parten IGUALES" -- y el numero se re-mide aqui porque la perdida
# cambio (dos BCE y dos MSE). Lo imprime `--suelos`, que es lo que hay que correr
# si se toca el dataset o la cabeza.
LAMBDA_COORD = 0.038
SEMILLA = 1
EPOCAS_DEF = 300


def _cargar(parte: str):
    """Del dataset PUBLICADO en el repo de datos, nunca de una copia local.

    Se niega antes de empezar si no esta (R2). Entrenar sobre un dato
    re-derivado al vuelo daria numeros incomparables con los de cualquier otro
    brazo, y no fallaria por ningun lado."""
    z = np.load(exigir_dataset(DATASET) / f"{parte}.npz", allow_pickle=True)
    v = torch.from_numpy(z["ventanas"].astype(np.float32) / 255.0).unsqueeze(1)
    obj = {c: (torch.from_numpy(z[f"existe_{c}"]).float(),
               torch.from_numpy(z[f"x_{c}"]), torch.from_numpy(z[f"y_{c}"]))
           for c in ESQUINAS}
    return v, obj


def _perdida(salida, objetivo, esquinas):
    """(perdida, bce_media, coord_mse_media). Las dos esquinas pesan igual."""
    bces, coords = [], []
    for c in esquinas:
        logit, px, py = salida[c]
        existe, x, y = objetivo[c]
        bces.append(F.binary_cross_entropy_with_logits(logit, existe))
        hay = existe > 0.5
        if hay.any():     # las coordenadas solo tienen sentido donde la esquina EXISTE
            coords.append(((px[hay] - x[hay]) ** 2 + (py[hay] - y[hay]) ** 2).mean())
    bce = torch.stack(bces).mean()
    coord = torch.stack(coords).mean() if coords else torch.zeros((), device=bce.device)
    return bce + LAMBDA_COORD * coord, bce, coord


def _metricas(salida, objetivo, esquinas) -> dict:
    """Por esquina, sobre las ventanas donde ESA esquina existe de verdad.

    ⚠ La metrica principal es la TASA DE ACIERTO a <=2 px, y el titular es la
    PEOR de las dos esquinas: una estructura que resuelve tl y no br no ha
    resuelto la tarea. El promedio la escondería."""
    m, aciertos = {}, []
    for c in esquinas:
        logit, px, py = salida[c]
        existe, x, y = objetivo[c]
        hay = existe > 0.5
        m[f"n_pos_{c}"] = int(hay.sum())
        if hay.any():
            d = ((px[hay] - x[hay]) ** 2 + (py[hay] - y[hay]) ** 2).sqrt()
            m[f"err_px_{c}"] = d.mean().item()
            m[f"acierto_1px_{c}"] = (d <= 1).float().mean().item()
            m[f"acierto_2px_{c}"] = (d <= 2).float().mean().item()
            aciertos.append(m[f"acierto_2px_{c}"])
        pred = torch.sigmoid(logit) > 0.5
        tp = (pred & hay).sum().item()
        den = pred.sum().item() + hay.sum().item()
        m[f"f1_existe_{c}"] = (2 * tp / den) if den > 0 else 0.0
    m["acierto_2px_peor"] = min(aciertos) if aciertos else 0.0
    return m


def _redondear(fila: dict) -> dict:
    """Seis cifras significativas en el registro, no diecisiete.

    ⚠ Es una decision de TAMANO, y con 25 brazos deja de ser cosmetica: el
    `repr` de un float de Python ocupa ~17 caracteres, y 300 epocas x 20 campos
    x 25 brazos son ~3 MB de historial en un repo cuyo tope declarado es ~5 MB
    (CLAUDE.md). Redondeado baja a la mitad, y ninguna metrica de aqui se lee mas
    alla de la cuarta cifra: el umbral del criterio es 12 % y el SE, 2,8 %."""
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in fila.items()}


def _estado_rng() -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate()}


def _poner_rng(e: dict) -> None:
    torch.set_rng_state(e["torch"]); np.random.set_state(e["numpy"]); random.setstate(e["python"])


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
            obj = {c: tuple(t[idx] for t in Ytr[c]) for c in red.esquinas}
            p, *_ = _perdida(red(Vtr[idx])[0], obj, red.esquinas)
            p.backward(); opt.step()
            suma += p.item() * len(idx); n += len(idx)
        red.eval()
        with torch.no_grad():
            salida, _ = red(Vva)
            pv, bce, coord = _perdida(salida, Yva, red.esquinas)
            met = _metricas(salida, Yva, red.esquinas)
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
        _guardar(ultimo, red, opt, ep, mejor)           # cada epoca: reanudable siempre
        det = " · ".join(f"{c} {100*fila.get(f'acierto_2px_{c}', 0):.0f}%/"
                         f"{fila.get(f'err_px_{c}', 0):.2f}px/f1 {fila[f'f1_existe_{c}']:.2f}"
                         for c in red.esquinas)
        print(f"  ep {ep:>3} val {fila['val']:.4f} · peor<=2px "
              f"{100*fila['acierto_2px_peor']:>5.1f}% · {det} · A {100*anti:.0f}% "
              f"· beta {fila['beta']:.2f} · {fila['segundos']}s")
    return 0


def suelos() -> int:
    """Los SUELOS: que da cada estructura SIN entrenar, y con que lambda parten
    iguales los dos terminos de la perdida.

    Es lo que el criterio necesita ANTES de mirar, y por eso vive aqui y no en
    una libreta: un suelo escrito de memoria no se distingue de uno medido."""
    Vva, Yva = _cargar("val")
    n = {c: int(Yva[c][0].sum()) for c in ESQUINAS}
    print(f"val: {len(Vva)} ventanas · {n['tl']} con esquina tl · {n['br']} con br\n")

    # El predictor CONSTANTE en el centro del mapa, que es el modo degenerado de
    # esta cabeza: si el mapa sale plano, la esperanza cae justo ahi.
    print(f"{'brazo':>7} {'esquina':>8} {'acierto<=2px':>13} {'<=1px':>7} {'err px':>8} "
          f"{'f1 existe':>10} {'bce':>7} {'mse':>8} {'lambda':>8}")
    for brazo in BRAZOS:
        red = construir(brazo, SEMILLA)
        red.eval()
        with torch.no_grad():
            salida, _ = red(Vva)
            _, bce, coord = _perdida(salida, Yva, red.esquinas)
            met = _metricas(salida, Yva, red.esquinas)
        lam = bce.item() / coord.item() if coord.item() else float("nan")
        for i, c in enumerate(red.esquinas):
            print(f"{brazo if i == 0 else '':>7} {c:>8} "
                  f"{100*met[f'acierto_2px_{c}']:>12.1f}% {100*met[f'acierto_1px_{c}']:>6.1f}% "
                  f"{met[f'err_px_{c}']:>8.2f} {met[f'f1_existe_{c}']:>10.3f} "
                  f"{bce.item() if i == 0 else 0:>7.3f} {coord.item() if i == 0 else 0:>8.2f} "
                  f"{lam if i == 0 else 0:>8.4f}")
    print(f"\n  lambda congelada en {LAMBDA_COORD} (regla: los dos terminos parten iguales)")
    print("  ⚠ un lambda por brazo haria que cada uno optimizase una funcion distinta")
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
            print(f"  {brazo:>7}: guarda y recupera pesos, epoca y mejor_val ... "
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
