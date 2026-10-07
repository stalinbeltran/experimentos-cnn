#!/usr/bin/env python3
"""Todo lo que va DESPUÉS de los detectores, en el dev: el COMPOSITOR de dígitos y su CURVA DE DESPLAZAMIENTO.
Se lanza como `nn/entrenar_local.py --componer [--bancos …]` (contrato con el freno: ver entrenar_local.py).

Tres bancos de 13 detectores, los tres con el MISMO compositor:
    lineas       los de `feat-ind32` (la tinta, sin bordes): la referencia, leída por su id y comprobada por su huella
    control      nn/pesos-control/<f>/best.pt      (los 8 canales de borde a la vez)
    compartido   nn/pesos-compartido/<f>/best.pt   (el mismo detector en cada canal, y máximo)

Entrada del compositor: los 13 mapas 8×8 (σ de los logits) = 832 números. Compositor = Linear(832→10), Adam, 300 épocas
a lote completo, lr 1e-2, L2 1e-3, 3 semillas: COPIA del de `feat-ind32` (nn/componer.py:44), que es lo que hace
comparables los números con su 0,949.

El DESPLAZAMIENTO es siempre de la IMAGEN, en píxeles, en las 8 direcciones, con relleno de ceros (lo que sale del lienzo
se pierde), y la imagen desplazada pasa otra vez por los detectores. NO se desplazan los mapas 8×8: una celda son 4 px
y en el boceto eso salió peor (docs/bocetos/2026-10-07-borde-de-un-lado, «Cuarta parte»).

  C1  180 de train / 1617 de val (windep), sin desplazar — el 0,949 de feat-ind32 y el 0,865 de feat-bor signo
  C2  el compositor ELEGIDO: entrenado con la entrada desplazada 0–2 px (≤2), en 180 y en los 3823 de otros escritores
  C3  la CURVA, pedida por el dueño el 2026-10-07: entrenado con 180 dígitos desplazados SÓLO s px (8 copias, sin el
      original) y con 0..s px (≤s, gradual), s ∈ {1, 2, 3, 4, 6, 8}; probado con val desplazado d ∈ {0, 1, 2, 3, 4, 6, 8, 12, 16} px,
      media y peor de las 8 direcciones
  y en C1/C2, los «gruesos reales»: el cuartil de val con más tinta de cada clase

    python nn/entrenar_local.py --componer                         → resultados/componer.json y componer.txt
    python nn/entrenar_local.py --componer --bancos lineas         la referencia sola (no necesita nada entrenado)
Los mapas de cada banco se guardan en nn/mapas/<banco>.npz (fuera de git) y se reutilizan si sus huellas casan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
sys.path.insert(0, str(EXP.parent))
sys.path.insert(0, str(AQUI))
from expcnn import por_id                       # noqa: E402
import datos                                    # noqa: E402
import features as F                            # noqa: E402
import modelo                                   # noqa: E402

RES = EXP / "resultados"
MAPAS = AQUI / "mapas"
BANCOS = ("lineas", "control", "compartido")
EPOCAS, LR, L2, SEMILLAS = 300, 1e-2, 1e-3, (1, 2, 3)
S = (1, 2, 3, 4, 6, 8)                                  # desplazamientos de ENTRENAMIENTO de la curva, px
D = (0, 1, 2, 3, 4, 6, 8, 12, 16)                      # desplazamientos de PRUEBA, px
ELEGIDO = 2                                             # el compositor de C2: ≤2 px
DIRS8 = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]


def r4(v) -> float:
    return round(float(v), 4)


def anillo(s: int) -> list[tuple[int, int]]:
    return [(s * a, s * b) for a, b in DIRS8]


def hasta(s: int) -> list[tuple[int, int]]:
    return [(0, 0)] + [m for k in range(1, s + 1) for m in anillo(k)]


def mover(x: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """(N, 1, 32, 32) desplazada dy, dx píxeles, relleno 0. COPIA de `mover` de prueba_compositores.py del boceto."""
    n = x.shape[-1]; r = np.zeros_like(x)
    ys, yd = (slice(0, n - dy), slice(dy, n)) if dy >= 0 else (slice(-dy, n), slice(0, n + dy))
    xs, xd = (slice(0, n - dx), slice(dx, n)) if dx >= 0 else (slice(-dx, n), slice(0, n + dx))
    r[..., yd, xd] = x[..., ys, xs]
    return r


def banco(nombre: str) -> tuple[list, dict]:
    """Los 13 detectores de un banco y sus huellas. `lineas` se comprueba contra la firma de feat-ind32."""
    reds, huellas = [], {}
    if nombre == "lineas":
        org = por_id("feat-ind32").carpeta
        firma = json.loads((org / "resultados" / "firma-por-clase.json").read_text(encoding="utf-8"))["huellas_best"]
    for f in F.CON_TRAZO:
        if nombre == "lineas":
            red, _ = modelo.cargar_tinta(org / "nn" / "pesos" / f / "best.pt")
            h = modelo.huella_pesos(red)
            if h != firma[f]:
                raise SystemExit(f"✗ feat-ind32/{f}: huella {h} ≠ {firma[f]} de su firma: sus pesos cambiaron. Me niego.")
        else:
            ruta = AQUI / f"pesos-{nombre}" / f / "best.pt"
            if not ruta.is_file():
                raise SystemExit(f"✗ falta {ruta}: el brazo '{nombre}' no está entrenado (nn/vast.sh {nombre})")
            red, _ = modelo.cargar(ruta); h = modelo.huella_pesos(red)
        reds.append(red); huellas[f] = h
    return reds, huellas


@torch.no_grad()
def mapas(reds: list, x: np.ndarray, lote: int = 512) -> np.ndarray:
    """(N, 1, 32, 32) → (N, 832) float16: σ de los 13 mapas 8×8, aplanados."""
    out = np.zeros((len(x), len(reds), 8, 8), np.float16)
    for i in range(0, len(x), lote):
        xt = torch.from_numpy(np.ascontiguousarray(x[i:i + lote], dtype=np.float32))
        for j, red in enumerate(reds):
            out[i:i + lote, j] = torch.sigmoid(red(xt))[:, 0].numpy()
    return out.reshape(len(x), -1)


def ajustar(x: np.ndarray, y: np.ndarray, sem: int) -> torch.nn.Linear:
    """COPIA de `feat-ind32` (nn/componer.py:44): Linear, Adam, 300 épocas a lote completo, lr 1e-2, L2 1e-3."""
    torch.manual_seed(sem)
    W = torch.nn.Linear(x.shape[1], 10); opt = torch.optim.Adam(W.parameters(), lr=LR, weight_decay=L2)
    xt, yt = torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y)
    for _ in range(EPOCAS):
        opt.zero_grad(); Fn.cross_entropy(W(xt), yt).backward(); opt.step()
    return W


def acierto(W, x: np.ndarray, y: np.ndarray) -> float:
    with torch.no_grad():
        return float((W(torch.from_numpy(x.astype(np.float32))).argmax(1).numpy() == y).mean())


def calcular_mapas(nombre: str, dg: dict) -> dict:
    """Todos los mapas que hacen falta para un banco, cacheados en nn/mapas/<banco>.npz con la huella de los pesos."""
    reds, huellas = banco(nombre)
    clave = hashlib.sha256(json.dumps(huellas, sort_keys=True).encode()).hexdigest()[:16]
    cache = MAPAS / f"{nombre}.npz"
    if cache.is_file():
        z = np.load(cache)
        if str(z["clave"]) == clave:
            print(f"  {nombre}: mapas de la caché ({clave})", flush=True)
            return {k: z[k] for k in z.files if k != "clave"} | {"huellas": huellas}
    t0 = time.time(); x = dg["x"]; out = {}
    out["val_0"] = mapas(reds, x[dg["val"]])
    for d in D[1:]:
        out[f"val_{d}"] = np.stack([mapas(reds, mover(x[dg["val"]], dy, dx)) for dy, dx in anillo(d)])
        print(f"  {nombre}: val desplazado {d} px  [{time.time() - t0:.0f} s]", flush=True)
    out["tr180"] = np.stack([mapas(reds, mover(x[dg["train"]], dy, dx)) for dy, dx in hasta(max(S))])     # 65 copias
    out["extra"] = np.stack([mapas(reds, mover(x[dg["extra"]], dy, dx)) for dy, dx in hasta(ELEGIDO)])    # 17 copias
    print(f"  {nombre}: todos los mapas en {time.time() - t0:.0f} s", flush=True)
    MAPAS.mkdir(exist_ok=True)
    np.savez(cache, clave=np.array(clave), **out)
    return out | {"huellas": huellas}


def componer_banco(nombre: str, dg: dict) -> dict:
    M = calcular_mapas(nombre, dg)
    y = dg["y"]; ytr, yva, yex = y[dg["train"]], y[dg["val"]], y[dg["extra"]]
    gr = dg["gruesos_en_val"]
    idx = {m: i for i, m in enumerate(hasta(max(S)))}
    out = {"huellas": M["huellas"]}

    def evaluar(Ws) -> dict:
        r = {"normal": r4(np.mean([acierto(W, M["val_0"], yva) for W in Ws])),
             "gruesos_reales": r4(np.mean([acierto(W, M["val_0"][gr], yva[gr]) for W in Ws])),
             "media": {}, "peor": {}}
        for d in D:
            accs = np.array([[acierto(W, M["val_0"], yva)] if d == 0 else [acierto(W, m, yva) for m in M[f"val_{d}"]]
                             for W in Ws])                                   # (semillas, direcciones)
            r["media"][str(d)] = r4(accs.mean()); r["peor"][str(d)] = r4(accs.mean(0).min())
        return r

    def plan(movs, base, ys):
        xs = np.concatenate([base[idx[m]] for m in movs]); return xs, np.tile(ys, len(movs))

    t0 = time.time()
    out["C1"] = evaluar([ajustar(M["tr180"][0], ytr, s) for s in SEMILLAS])
    out["C1"]["por_semilla"] = [r4(acierto(ajustar(M["tr180"][0], ytr, s), M["val_0"], yva)) for s in SEMILLAS]
    xs, ys = plan(hasta(ELEGIDO), M["tr180"], ytr)
    out["C2_180"] = evaluar([ajustar(xs, ys, s) for s in SEMILLAS])
    out["C2_3823_sin"] = evaluar([ajustar(M["extra"][0], yex, s) for s in SEMILLAS])
    xs = np.concatenate(list(M["extra"])); ys = np.tile(yex, len(M["extra"]))
    out["C2_3823"] = evaluar([ajustar(xs, ys, s) for s in SEMILLAS])
    print(f"  {nombre}: C1 {out['C1']['normal']:.4f} · C2 180 ≤{ELEGIDO} {out['C2_180']['normal']:.4f} · "
          f"3823 {out['C2_3823_sin']['normal']:.4f} → ≤{ELEGIDO} {out['C2_3823']['normal']:.4f}  [{time.time() - t0:.0f} s]", flush=True)
    out["C3"] = {}
    for s in S:
        for etiqueta, movs in ((f"solo {s}", anillo(s)), (f"≤{s}", hasta(s))):
            xs, ys = plan(movs, M["tr180"], ytr)
            out["C3"][etiqueta] = evaluar([ajustar(xs, ys, sem) for sem in SEMILLAS])
            print(f"  {nombre}: C3 {etiqueta:<7} " + " ".join(f"{out['C3'][etiqueta]['media'][str(d)]:.3f}" for d in D), flush=True)
    out["C3"]["base"] = out["C1"]
    return out


def tabla(res: dict) -> str:
    L = []
    for b, r in res["bancos"].items():
        L += [f"== {b} ==",
              f"C1 180/1617 sin desplazar: {r['C1']['normal']:.4f} (semillas {r['C1']['por_semilla']}) · gruesos reales {r['C1']['gruesos_reales']:.4f}",
              f"C2 180 ≤{ELEGIDO} px: {r['C2_180']['normal']:.4f} · gruesos {r['C2_180']['gruesos_reales']:.4f}",
              f"C2 3823 sin desplazar: {r['C2_3823_sin']['normal']:.4f} · ≤{ELEGIDO} px: {r['C2_3823']['normal']:.4f} · gruesos "
              f"{r['C2_3823']['gruesos_reales']:.4f}",
              f"C3 (180 de train), acierto MEDIO de las 8 direcciones según el desplazamiento de prueba d (px):",
              f"{'entrenado con':14s}" + "".join(f"{f'd={d}':>8s}" for d in D)]
        for e in ["base"] + [f"solo {s}" for s in S] + [f"≤{s}" for s in S]:
            L.append(f"{e:14s}" + "".join(f"{r['C3'][e]['media'][str(d)]:>8.3f}" for d in D))
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="entrenar_local.py --componer")
    p.add_argument("--bancos", default=",".join(BANCOS)); a = p.parse_args(argv)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    dg = datos.digitos()
    dg["extra"] = dg["origen"] != "windep"
    tinta = dg["x"].reshape(len(dg["y"]), -1).sum(1); va = np.flatnonzero(dg["val"]); yva = dg["y"][va]
    gr = np.zeros(len(va), bool)
    for c in range(10):
        m = yva == c; gr[m] = tinta[va][m] >= np.quantile(tinta[va][m], .75)
    dg["gruesos_en_val"] = gr
    destino = RES / "componer.json"
    res = json.loads(destino.read_text(encoding="utf-8")) if destino.is_file() else {"bancos": {}}
    res |= {"D": list(D), "S": list(S), "elegido": ELEGIDO, "n": {"train": int(dg["train"].sum()), "val": len(va),
            "extra": int(dg["extra"].sum()), "gruesos_reales": int(gr.sum())}}
    for b in a.bancos.split(","):
        t0 = time.time()
        res["bancos"][b] = componer_banco(b, dg) | {"segundos": round(time.time() - t0, 1),
                                                   "cuando": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        RES.mkdir(exist_ok=True)
        destino.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        (RES / "componer.txt").write_text(tabla(res) + "\n", encoding="utf-8")
    print(tabla(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
