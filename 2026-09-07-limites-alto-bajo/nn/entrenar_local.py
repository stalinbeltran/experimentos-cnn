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
from modelo import BRAZOS, CELDAS, construir, simetria

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
# ✅ RE-MEDIDO Y CONGELADO el 2026-09-07 con `--suelos` sobre el `val` de ESTE
# dataset, ANTES de la primera epoca: bce/mse = 0,0522 de media en los 7 brazos
# (0,0491 a 0,0546). NO se heredo el 0,0293 de `esq-cq`: alli el termino de
# coordenada era (dx^2 + dy^2) y aqui es solo dy^2, asi que el mse cae ~a la
# mitad y la lambda sube ~al doble. Se cumplio: 0,0293 -> 0,0522, x1,78.
LAMBDA_COORD = 0.0522
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
    e, y = datos.objetivo(z)
    obj = (torch.from_numpy(e).float(), torch.from_numpy(y))
    # Mascaras SOLO para el desglose al medir: la red no las ve nunca.
    # ⚠ Las CUATRO celdas de esquina son el ancla de comparabilidad con `esq-cq`,
    # que midio sobre exactamente estas mismas 78/39/39/78 ventanas.
    desglose = {c: torch.from_numpy(z[f"existe_{c}"] == 1) for c in CELDAS}
    # Y el CASO PURO, que es lo que `esq-cq` no podia medir.
    cl = z["clase"]
    desglose["puro_sup"] = torch.from_numpy(cl == "borde-superior")
    desglose["puro_inf"] = torch.from_numpy(cl == "borde-inferior")
    # Los dos negativos duros de ESTE experimento.
    desglose["interior"] = torch.from_numpy(cl == "interior")
    desglose["borde_vertical"] = torch.from_numpy(
        np.isin(cl, ["borde-izquierdo", "borde-derecho"]))
    return v, obj, desglose


def _perdida(salida, objetivo):
    """(perdida, bce, alto_mse). UNA bce y UN mse, y el mse es SOLO en `y`.

    ⚠ El termino en `x` NO esta: un limite horizontal es una linea, asi que la
    `x` de la etiqueta seria el punto de anclaje sorteado al recortar. Meterla
    aqui seria entrenar contra ruido de muestreo."""
    logit, py = salida
    existe, y = objetivo
    bce = F.binary_cross_entropy_with_logits(logit, existe)
    hay = existe > 0.5
    if hay.any():         # la altura solo tiene sentido donde HAY limite
        coord = ((py[hay] - y[hay]) ** 2).mean()
    else:
        coord = torch.zeros((), device=bce.device)
    return bce + LAMBDA_COORD * coord, bce, coord


def _metricas(salida, objetivo, desglose) -> dict:
    """La metrica UNICA (|dy| <= 2 px), mas el desglose por celda.

    ⚠⚠ EL SUELO DE ESTA METRICA NO ES EL DE `esq-cq`, Y ESA ES LA TRAMPA GRANDE.
    Alli la distancia era euclidea en 2-D y el suelo salia 5,8 %; aqui es |dy| en
    1-D y el suelo sale ~28,6 % (medido 2026-09-07, red sin entrenar). Heredar el
    umbral de 9,6 % haria que los siete brazos "pasaran" desde la epoca 0 sin
    haber aprendido NADA. Los umbrales viven en `informe.py` y se re-miden con
    `--suelos`.

    El desglose tiene seis celdas y dos grupos:
      tl · tr · bl · br   las cuatro esquinas -- LO UNICO comparable columna a
                          columna con `esq-cq`, sobre las MISMAS 78/39/39/78
      puro_sup · puro_inf el limite SIN esquina, que `esq-cq` no podia medir

    Y dos falsos positivos, que son los modos de fallo de AQUI:
      fp_interior         una linea de texto es un borde horizontal falso, y hay
                          decenas por parrafo. Es el fallo mas probable
      fp_borde_vertical   tinta alineada con canto claro, pero VERTICAL"""
    logit, py = salida
    existe, y = objetivo
    hay = existe > 0.5
    m = {"n_pos": int(hay.sum())}

    d_todas = (py - y).abs()
    if hay.any():
        d = d_todas[hay]
        m["err_px"] = d.mean().item()
        m["acierto_1px"] = (d <= 1).float().mean().item()
        m["acierto_2px"] = (d <= 2).float().mean().item()

    for c in ("tl", "tr", "bl", "br", "puro_sup", "puro_inf"):
        sel = desglose[c]
        m[f"n_{c}"] = int(sel.sum())
        if sel.any():
            m[f"acierto_2px_{c}"] = (d_todas[sel] <= 2).float().mean().item()
            m[f"err_px_{c}"] = d_todas[sel].mean().item()

    # agregados por lado, que es como se lee la prediccion del experimento
    for lado, cs in (("alto", ("tl", "tr", "puro_sup")), ("bajo", ("bl", "br", "puro_inf"))):
        sel = desglose[cs[0]] | desglose[cs[1]] | desglose[cs[2]]
        if sel.any():
            m[f"acierto_2px_{lado}"] = (d_todas[sel] <= 2).float().mean().item()

    pred = torch.sigmoid(logit) > 0.5
    tp = (pred & hay).sum().item()
    den = pred.sum().item() + hay.sum().item()
    m["f1_existe"] = (2 * tp / den) if den > 0 else 0.0
    for nom, clave in (("fp_interior", "interior"), ("fp_borde_vertical", "borde_vertical")):
        sel = desglose[clave]
        m[nom] = pred[sel].float().mean().item() if sel.any() else 0.0
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
        s_vert, s_180 = simetria(red.kernel())
        fila = {"epoca": ep, "train": suma / n, "val": pv.item(), "bce": bce.item(),
                "coord_mse": coord.item(), "beta": float(red.log_beta.exp().detach()),
                # DOS simetrias: la vertical es la que predice el algebra de aqui;
                # la de 180 es la unica columna comparable con `esq-cq`.
                "simetrico_vertical": s_vert, "simetrico_rot180": s_180,
                **met, "segundos": round(time.time() - t0, 2)}
        with reg.open("a", encoding="utf-8") as f:      # SOLO ANADIR
            f.write(json.dumps(_redondear(fila)) + "\n")
        if pv.item() < mejor:
            mejor = pv.item()
            _guardar(mejor_f, red, opt, ep, mejor)
        _guardar(ultimo, red, opt, ep, mejor)           # cada epoca: reanudable siempre
        # ⚠ El desglose va en la linea de cada epoca, no solo en el informe.
        det = " · ".join(f"{c} {100*fila.get(f'acierto_2px_{c}', 0):.0f}%"
                         for c in ("alto", "bajo"))
        print(f"  ep {ep:>3} val {fila['val']:.4f} · <=2px "
              f"{100*fila['acierto_2px']:>5.1f}% ({det}) · f1 {fila['f1_existe']:.2f} "
              f"· fp int {100*fila['fp_interior']:.0f}% vert {100*fila['fp_borde_vertical']:.0f}% "
              f"· Sv {100*s_vert:.0f}% · beta {fila['beta']:.2f} · {fila['segundos']}s")
    return 0


def init(brazo: str | None = None) -> int:
    """Guarda los pesos de la EPOCA 0 -- la red construida y sin entrenar.

    Por que existe, y no es un capricho: `nn/transformacion.py` lee el kernel de
    `pesos/<brazo>/best.pt`, asi que sin esto NO SE PUEDE dibujar la figura de
    pagina entera del punto de partida. El suelo de la ventana si se puede medir
    sin pesos (`--suelos` construye la red al vuelo), pero el de la pagina no.

    Y de paso deja la fila 0 en `metrics.jsonl`, que es el suelo DE ESE BRAZO
    medido, no el de la tabla general: cuando el entrenamiento escriba la epoca 1
    el registro ya tiene contra que compararla.

    ⚠ NO PISA lo que ya exista. Un `--init` distraido sobre un brazo entrenado
    borraria sus pesos y su registro, que es justo el fallo que no se puede
    permitir un comando que se corre "para preparar"."""
    Vva, Yva, Dva = _cargar("val")
    brazos = [brazo] if brazo else list(BRAZOS)
    hechos = saltados = 0
    for b in brazos:
        dir_b = PESOS / b
        ultimo, mejor_f, reg = dir_b / "last.pt", dir_b / "best.pt", dir_b / "metrics.jsonl"
        if ultimo.exists() or mejor_f.exists() or reg.exists():
            print(f"  {b}: YA existe (epoca "
                  f"{torch.load(ultimo, map_location='cpu', weights_only=False)['epoca'] if ultimo.exists() else '?'})"
                  f" — no lo toco")
            saltados += 1
            continue
        dir_b.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(SEMILLA); np.random.seed(SEMILLA); random.seed(SEMILLA)
        red = construir(b, SEMILLA)
        opt = torch.optim.Adam(red.parameters(), lr=LR)
        red.eval()
        with torch.no_grad():
            salida, _ = red(Vva)
            pv, bce, coord = _perdida(salida, Yva)
            met = _metricas(salida, Yva, Dva)
        s_vert, s_180 = simetria(red.kernel())
        fila = {"epoca": 0, "train": None, "val": pv.item(), "bce": bce.item(),
                "coord_mse": coord.item(), "beta": float(red.log_beta.exp().detach()),
                "simetrico_vertical": s_vert, "simetrico_rot180": s_180,
                **met, "segundos": 0.0}
        with reg.open("w", encoding="utf-8") as f:
            f.write(json.dumps(_redondear(fila)) + "\n")
        _guardar(mejor_f, red, opt, 0, pv.item())
        _guardar(ultimo, red, opt, 0, pv.item())
        print(f"  {b}: {red.n_parametros():>3} par · val {pv.item():.4f} · "
              f"<=2px {100*met['acierto_2px']:.1f}% "
              f"(alto {100*met.get('acierto_2px_alto',0):.1f} "
              f"bajo {100*met.get('acierto_2px_bajo',0):.1f}) · Sv {100*s_vert:.0f}%")
        hechos += 1
    print(f"\n{hechos} brazo(s) inicializados en la epoca 0, {saltados} ya existian.")
    print("⚠ Son pesos SIN ENTRENAR: sirven de suelo y para las figuras de partida.")
    return 0


def suelos() -> int:
    """Los SUELOS de `lim-h`, y donde se CONGELAN sus dos numeros de criterio.

    ⚠⚠ NINGUNO SE HEREDA DE `esq-cq`, y no es prudencia: heredarlos lo invalida.
    La metrica pasa de una distancia euclidea 2-D a |dy| en 1-D, asi que el suelo
    sube de ~5,8 % a ~28,6 % -- por encima del umbral de 9,6 % de alli. Con el
    umbral heredado, los siete brazos "pasarian" desde la epoca 0."""
    Vva, Yva, Dva = _cargar("val")
    existe = Yva[0]
    print(f"val: {len(Vva)} ventanas · {int(existe.sum())} positivas "
          f"({existe.mean():.1%})")
    print(f"     celdas: tl {int(Dva['tl'].sum())} · tr {int(Dva['tr'].sum())} · "
          f"bl {int(Dva['bl'].sum())} · br {int(Dva['br'].sum())} · "
          f"PUROS {int(Dva['puro_sup'].sum())}+{int(Dva['puro_inf'].sum())}")
    print(f"     negativos duros: interior {int(Dva['interior'].sum())} · "
          f"borde vertical {int(Dva['borde_vertical'].sum())}\n")

    print(f"{'brazo':>7} {'<=2px':>7} {'(alto':>7} {'bajo)':>7} {'<=1px':>7} {'err|dy|':>8} "
          f"{'f1':>7} {'fp-int':>7} {'fp-ver':>7} {'bce':>7} {'mse':>8} {'lambda':>8}")
    lams, sue = [], []
    for brazo in BRAZOS:
        red = construir(brazo, SEMILLA); red.eval()
        with torch.no_grad():
            salida, _ = red(Vva)
            _, bce, coord = _perdida(salida, Yva)
            met = _metricas(salida, Yva, Dva)
        lam = bce.item() / coord.item() if coord.item() else float("nan")
        lams.append(lam); sue.append(met["acierto_2px"])
        print(f"{brazo:>7} {100*met['acierto_2px']:>6.1f}% "
              f"{100*met.get('acierto_2px_alto', 0):>6.1f}% "
              f"{100*met.get('acierto_2px_bajo', 0):>6.1f}% "
              f"{100*met['acierto_1px']:>6.1f}% {met['err_px']:>8.2f} "
              f"{met['f1_existe']:>7.3f} {100*met['fp_interior']:>6.1f}% "
              f"{100*met['fp_borde_vertical']:>6.1f}% "
              f"{bce.item():>7.3f} {coord.item():>8.2f} {lam:>8.4f}")

    medio = sum(lams) / len(lams)
    suelo = sum(sue) / len(sue)
    n = int(existe.sum())
    se = (suelo * (1 - suelo) / n) ** 0.5
    print(f"\n  LOS DOS NUMEROS QUE HAY QUE CONGELAR EN EL CRITERIO (R13):")
    print(f"    suelo <=2px = {100*suelo:.1f} %  ->  UMBRAL = suelo + 2 SE = "
          f"{100*(suelo + 2*se):.1f} %   (SE = {100*se:.1f} % sobre {n} positivas)")
    print(f"    LAMBDA_COORD = {medio:.4f}   (hoy en el codigo: {LAMBDA_COORD})")
    if abs(medio - LAMBDA_COORD) > 0.2 * max(medio, LAMBDA_COORD):
        print(f"    ⚠⚠ LAMBDA NO CUADRA. Congelala en {medio:.4f} antes de la epoca 1.")
    print(f"\n  ⚠ compara con `esq-cq`: alli el suelo era 5,8 % y el umbral 9,6 %.")
    print(f"    Heredarlos aqui haria que los 7 brazos pasaran sin aprender nada.")
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
    p.add_argument("--init", action="store_true",
                   help="guarda los pesos de la epoca 0 (sin entrenar) y su fila de registro")
    a = p.parse_args()
    if a.comprobar:
        return comprobar()
    if a.suelos:
        return suelos()
    if a.init:
        return init(a.brazo)
    if not a.brazo:
        p.error("hace falta --brazo (o --comprobar / --suelos)")
    return entrenar(a.brazo, a.epocas, a.desde_cero)


if __name__ == "__main__":
    raise SystemExit(main())
