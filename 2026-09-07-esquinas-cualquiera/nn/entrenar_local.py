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

import datos
from datos import DATASET
from expcnn import exigir_dataset
from modelo import BRAZOS, ESQUINAS, construir, simetria

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
PESOS = AQUI / "pesos"

LOTE = 128
LR = 0.05
# ⚠ MEDIDO, no elegido a ojo. La regla que se hereda es "los dos terminos parten
# IGUALES", NO el numero. Y el numero HAY QUE RE-MEDIRLO: el 0,038 sale de una
# perdida con DOS bce y DOS mse y un 20 % de positivas por esquina; aqui hay UNA
# de cada y las positivas son el 40,1 %. Lo imprime `--suelos`, y se congela ahi
# antes de la primera epoca.
# ✅ RE-MEDIDO Y CONGELADO el 2026-09-07 con `--suelos` sobre el val publicado,
# ANTES de la primera epoca: bce/mse = 0,0293 de media en los cinco brazos
# (0,0285 a 0,0303). El 0,038 de `esq-2d` era de otra perdida.
LAMBDA_COORD = 0.0293
SEMILLA = 1
EPOCAS_DEF = 300


def _cargar(parte: str):
    """Del dataset PUBLICADO en el repo de datos, nunca de una copia local.

    Se niega antes de empezar si no esta (R2). Entrenar sobre un dato
    re-derivado al vuelo daria numeros incomparables con los de cualquier otro
    brazo, y no fallaria por ningun lado."""
    z = np.load(exigir_dataset(DATASET) / f"{parte}.npz", allow_pickle=True)
    v = torch.from_numpy(z["ventanas"].astype(np.float32) / 255.0).unsqueeze(1)
    # LA PUERTA UNICA: la union tl|br se deriva en `datos.objetivo` y se importa.
    # No se repite aqui -- la usan tambien `muestras.py` e `informe.py`, y tres
    # copias de una derivacion divergen sin que nadie se entere.
    e, x, y = datos.objetivo(z)
    obj = (torch.from_numpy(e).float(), torch.from_numpy(x), torch.from_numpy(y))
    # Mascaras SOLO para el desglose al medir: la red no las ve nunca. Son el
    # ancla de comparabilidad con `esq-2d`, que midio sobre estas mismas 78+78.
    desglose = {c: torch.from_numpy(z[f"existe_{c}"] == 1) for c in ("tl", "br")}
    desglose["otra_diagonal"] = torch.from_numpy(
        (z["existe_tr"] == 1) | (z["existe_bl"] == 1))
    return v, obj, desglose


def _perdida(salida, objetivo):
    """(perdida, bce, coord_mse). UNA bce y UN mse: hay una sola salida.

    En `esq-2d` habia que promediar dos de cada y declarar que las dos esquinas
    pesaban igual. Aqui esa decision desaparece: no hay dos cosas que pesar."""
    logit, px, py = salida
    existe, x, y = objetivo
    bce = F.binary_cross_entropy_with_logits(logit, existe)
    hay = existe > 0.5
    if hay.any():         # las coordenadas solo tienen sentido donde HAY esquina
        coord = ((px[hay] - x[hay]) ** 2 + (py[hay] - y[hay]) ** 2).mean()
    else:
        coord = torch.zeros((), device=bce.device)
    return bce + LAMBDA_COORD * coord, bce, coord


def _metricas(salida, objetivo, desglose) -> dict:
    """La metrica UNICA, mas el desglose por esquina verdadera.

    ⚠⚠ EL DESGLOSE NO ES DIAGNOSTICO OPCIONAL: ES EL ANCLA DE COMPARABILIDAD.
    La metrica titular de `esq-cq` (una tasa sobre 156 ventanas) y la de `esq-2d`
    (la PEOR de dos tasas sobre 78) NO son el mismo numero: un 50 % aqui no es un
    50 % alli. Lo unico que se puede poner columna a columna son las tasas por
    esquina verdadera, sobre las MISMAS 78 + 78 ventanas.

    ⚠ Y hace falta por un motivo mas fuerte: con `br` sin senal local a menos de
    8 px (medido 2026-09-07) el techo esperable ronda el 50 %, porque las
    positivas son mitad y mitad. **Un 50 % se lee como "resuelve la mitad" y
    significa "tl entero, br nada".** Sin desglose, este experimento se
    autoengana.

    `fp_otra_diagonal` es el modo de fallo propio de un kernel casi simetrico:
    responder alto en `tr`/`bl`, que son el negativo duro."""
    logit, px, py = salida
    existe, x, y = objetivo
    hay = existe > 0.5
    m = {"n_pos": int(hay.sum())}

    d_todas = ((px - x) ** 2 + (py - y) ** 2).sqrt()
    if hay.any():
        d = d_todas[hay]
        m["err_px"] = d.mean().item()
        m["acierto_1px"] = (d <= 1).float().mean().item()
        m["acierto_2px"] = (d <= 2).float().mean().item()

    # desglose por esquina VERDADERA (la red no sabe cual es; nosotros si)
    for c in ("tl", "br"):
        sel = desglose[c]
        m[f"n_pos_{c}"] = int(sel.sum())
        if sel.any():
            dc = d_todas[sel]
            m[f"acierto_2px_{c}"] = (dc <= 2).float().mean().item()
            m[f"err_px_{c}"] = dc.mean().item()

    pred = torch.sigmoid(logit) > 0.5
    tp = (pred & hay).sum().item()
    den = pred.sum().item() + hay.sum().item()
    m["f1_existe"] = (2 * tp / den) if den > 0 else 0.0
    otra = desglose["otra_diagonal"]
    m["fp_otra_diagonal"] = pred[otra].float().mean().item() if otra.any() else 0.0
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

    Vtr, Ytr, _ = _cargar("train")          # el desglose solo hace falta al medir
    Vva, Yva, Dva = _cargar("val")
    reg = dir_b / "metrics.jsonl"

    for ep in range(ep0 + 1, epocas + 1):
        t0 = time.time()
        red.train()
        orden = torch.randperm(len(Vtr))
        suma = n = 0
        for i in range(0, len(orden), LOTE):
            idx = orden[i:i + LOTE]
            opt.zero_grad()
            obj = tuple(t[idx] for t in Ytr)
            p, *_ = _perdida(red(Vtr[idx])[0], obj)
            p.backward(); opt.step()
            suma += p.item() * len(idx); n += len(idx)
        red.eval()
        with torch.no_grad():
            salida, _ = red(Vva)
            pv, bce, coord = _perdida(salida, Yva)
            met = _metricas(salida, Yva, Dva)
        sim, anti = simetria(red.kernel())
        fila = {"epoca": ep, "train": suma / n, "val": pv.item(), "bce": bce.item(),
                "coord_mse": coord.item(), "beta": float(red.log_beta.exp().detach()),
                "simetrico": sim, "antisimetrico": anti,
                **met, "segundos": round(time.time() - t0, 2)}
        with reg.open("a", encoding="utf-8") as f:      # SOLO ANADIR
            f.write(json.dumps(_redondear(fila)) + "\n")
        if pv.item() < mejor:
            mejor = pv.item()
            _guardar(mejor_f, red, opt, ep, mejor)
        _guardar(ultimo, red, opt, ep, mejor)           # cada epoca: reanudable siempre
        # ⚠ El desglose va en la linea de cada epoca, no solo en el informe: es
        # donde se ve si el 40 % que sale es "medio tl y medio br" o "tl entero".
        det = " · ".join(f"{c} {100*fila.get(f'acierto_2px_{c}', 0):.0f}%"
                         for c in ("tl", "br"))
        print(f"  ep {ep:>3} val {fila['val']:.4f} · <=2px "
              f"{100*fila['acierto_2px']:>5.1f}% ({det}) · f1 {fila['f1_existe']:.2f} "
              f"· fp-otra {100*fila['fp_otra_diagonal']:.0f}% · S {100*sim:.0f}% "
              f"· beta {fila['beta']:.2f} · {fila['segundos']}s")
    return 0


def suelos() -> int:
    """Los SUELOS: que da la red SIN entrenar, y con que lambda parten iguales
    los dos terminos de la perdida.

    Es lo que el criterio necesita ANTES de mirar, y por eso vive aqui y no en
    una libreta: un suelo escrito de memoria no se distingue de uno medido.

    ⚠ Y aqui es donde se CONGELA `LAMBDA_COORD` para `esq-cq`. El 0,038 que se
    hereda de `esq-2d` sale de una perdida distinta (dos bce y dos mse, 20 % de
    positivas por esquina); esta tiene una de cada y 40,1 % de positivas. La
    regla que se hereda es "los dos terminos parten iguales", no el numero."""
    Vva, Yva, Dva = _cargar("val")
    existe = Yva[0]
    print(f"val: {len(Vva)} ventanas · {int(existe.sum())} positivas "
          f"({existe.mean():.1%}) · {int(Dva['tl'].sum())} son tl · "
          f"{int(Dva['br'].sum())} son br · {int(Dva['otra_diagonal'].sum())} "
          f"de la otra diagonal (negativo duro)\n")

    # El predictor CONSTANTE en el centro del mapa, que es el modo degenerado de
    # esta cabeza: si el mapa sale plano, la esperanza cae justo ahi.
    print(f"{'brazo':>7} {'<=2px':>7} {'(tl':>7} {'br)':>7} {'<=1px':>7} {'err px':>8} "
          f"{'f1':>7} {'fp-otra':>8} {'bce':>7} {'mse':>8} {'lambda':>8}")
    lams = []
    for brazo in BRAZOS:
        red = construir(brazo, SEMILLA)
        red.eval()
        with torch.no_grad():
            salida, _ = red(Vva)
            _, bce, coord = _perdida(salida, Yva)
            met = _metricas(salida, Yva, Dva)
        lam = bce.item() / coord.item() if coord.item() else float("nan")
        lams.append(lam)
        print(f"{brazo:>7} {100*met['acierto_2px']:>6.1f}% "
              f"{100*met.get('acierto_2px_tl', 0):>6.1f}% "
              f"{100*met.get('acierto_2px_br', 0):>6.1f}% "
              f"{100*met['acierto_1px']:>6.1f}% {met['err_px']:>8.2f} "
              f"{met['f1_existe']:>7.3f} {100*met['fp_otra_diagonal']:>7.1f}% "
              f"{bce.item():>7.3f} {coord.item():>8.2f} {lam:>8.4f}")

    medio = sum(lams) / len(lams)
    print(f"\n  lambda que iguala los dos terminos: {medio:.4f} (media de los {len(lams)} brazos)")
    print(f"  LAMBDA_COORD hoy en el codigo: {LAMBDA_COORD}")
    if abs(medio - LAMBDA_COORD) > 0.2 * max(medio, LAMBDA_COORD):
        print(f"  ⚠⚠ NO CUADRAN. Congela LAMBDA_COORD = {medio:.4f} ANTES de la primera")
        print("     epoca: es el numero heredado de `esq-2d`, con otra perdida detras.")
    print("  ⚠ un lambda por brazo haria que cada uno optimizase una funcion distinta")

    # El suelo del "siempre si", que es contra lo que se lee el f1.
    p_si = float(existe.mean())
    print(f"\n  suelo f1 del 'siempre si': {2 * p_si / (1 + p_si):.3f}")
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
