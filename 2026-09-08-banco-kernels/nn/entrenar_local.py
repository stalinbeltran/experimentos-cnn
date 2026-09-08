#!/usr/bin/env python3
"""Entrena UNA condicion del banco `banco-k`. Protocolo del §8.

    python nn/entrenar_local.py --condicion identidad --semilla 0
    python nn/entrenar_local.py --condicion gauss --kernel kernels/gauss.npy --semilla 0
    python nn/entrenar_local.py --comprobar        los invariantes, sin dataset

EL NOMBRE DE ESTE FICHERO ES UN CONTRATO, NO ESTETICA
=====================================================
`experimento.json` declara `gasta: "entrena-local"`, y el freno del coordinador
(`cerrable.mjs`, lista declarada `TRABAJOS`) casa EL NOMBRE `entrenar_local.py` para
decidir si se puede apagar este server. Con otro nombre, el veredicto que el duenyo
lee desde el movil diria «nada corriendo» con un barrido vivo. `comprobar.py` del
repo lo verifica.

LO QUE ES IDENTICO ENTRE CONDICIONES, Y POR QUE IMPORTA (§8.2)
==============================================================
La UNICA variable del banco es el kernel aplicado a las entradas. Todo lo demas se
deriva de `--semilla` y por tanto es bit a bit identico en todas las condiciones:

  · los pesos iniciales      (torch.manual_seed(semilla) antes de construir la red)
  · el ORDEN DE LOS LOTES    (un Generator propio, sembrado con la semilla, que no
                              depende de la condicion ni de cuantas epocas van)
  · las particiones          (vienen del dataset publicado, son un dato)

Si esto se rompe, la diferencia entre dos condiciones deja de ser atribuible al
kernel y el banco no mide lo que dice medir. No es una precaucion teorica: es la
razon por la que el orden de los lotes se saca de un Generator aparte en vez del RNG
global de torch, que la construccion de la red ya ha consumido.

SIN PARADA TEMPRANA Y SIN AUMENTO DE DATOS, A PROPOSITO (§8.3, §8.4)
====================================================================
Con 100 muestras la red MEMORIZA `train`. Eso es lo buscado: la brecha
`IoU_train - IoU_eval` en regimen de sobreajuste ES la evidencia del criterio §2.2.
Evitar el sobreajuste destruiria la medida. Y el aumento de datos es en si mismo un
mecanismo de generalizacion: competiria con el kernel por el mismo efecto.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(EXP.parent))

# §8.1, y son INVARIANTES (§12): no se tocan sin construir un banco nuevo
LR = 3e-4
LOTE = 20
EPOCAS = 200
PARADAS = (25, 50, 100, 200)
NOMBRE_DATASET = "parrafos1000-584px-r4-r20260908b"


def cargar(nombre: str = NOMBRE_DATASET):
    """El dataset publicado, o la etapa local si aun no se ha publicado."""
    from expcnn import ruta_dataset                        # noqa: PLC0415
    p = ruta_dataset(nombre)
    origen = "publicado"
    if p is None:
        p = EXP / "datos"
        origen = "etapa local (SIN publicar)"
        if not (p / "train.npz").is_file():
            raise RuntimeError(
                f"no hay dataset. Ni '{nombre}' publicado ni etapa local en {p}.\n"
                f"  → python nn/datos.py --imagenes 1000   y luego --publicar")
    d = {}
    for k in ("train", "monitor", "eval"):
        z = np.load(p / f"{k}.npz")
        d[k] = (z["imagenes"], z["etiquetas"])
    return d, origen, p


def preparar(d, kernel):
    """§6 entero: kernel -> recorte -> estandarizacion con mu,sigma de `train`."""
    from pipeline import aplicar, estadisticos, estandarizar   # noqa: PLC0415
    from pipeline import etiquetas_128, normalizar_coords      # noqa: PLC0415
    x = {k: aplicar(v[0], kernel) for k, v in d.items()}
    mu, sd = estadisticos(x["train"])                      # §6.5: SOLO de train
    x = {k: estandarizar(v, mu, sd) for k, v in x.items()}
    y_px = {k: etiquetas_128(v[1]) for k, v in d.items()}  # §6.4
    y = {k: normalizar_coords(v) for k, v in y_px.items()}
    return x, y, y_px, (mu, sd)


def entrenar(x, y, y_px, semilla: int, epocas: int = EPOCAS, paradas=PARADAS,
             al_evaluar=None):
    import torch                                            # noqa: PLC0415
    from modelo import BancoCNN                             # noqa: PLC0415
    from evaluar import resumen                             # noqa: PLC0415
    from pipeline import desnormalizar_coords               # noqa: PLC0415

    torch.manual_seed(semilla)          # §8.2: mismos pesos iniciales
    modelo = BancoCNN()
    opt = torch.optim.Adam(modelo.parameters(), lr=LR)

    tx = {k: torch.from_numpy(v).unsqueeze(1) for k, v in x.items()}
    ty = {k: torch.from_numpy(v) for k, v in y.items()}
    n = tx["train"].shape[0]
    # §8.2: el orden de los lotes sale de SU PROPIO generador, sembrado con la semilla.
    # Con el RNG global de torch dependeria de cuanta aleatoriedad haya consumido la
    # construccion de la red, que es identica hoy pero no tiene por que serlo.
    g = torch.Generator().manual_seed(semilla + 10_000)

    filas, hist = [], {}
    for epoca in range(1, epocas + 1):
        modelo.train()
        orden = torch.randperm(n, generator=g)
        for i in range(0, n, LOTE):
            idx = orden[i:i + LOTE]
            opt.zero_grad()
            pred = modelo(tx["train"][idx])
            # §8.1: L1 sobre coordenadas normalizadas. L2 castigaria cuadraticamente
            # los atipicos y con n=100 unas pocas muestras dominarian el gradiente.
            perdida = (pred - ty["train"][idx]).abs().mean()
            perdida.backward()
            opt.step()

        if epoca in paradas:
            modelo.eval()
            fila = {"epoca": epoca, "semilla": semilla,
                    "train_loss": float(perdida.detach())}
            with torch.no_grad():
                for part in ("train", "monitor", "eval"):
                    p_px = desnormalizar_coords(modelo(tx[part]).numpy())
                    r = resumen(p_px, y_px[part])
                    fila[f"iou_{part}"] = r["iou"]
                    if part == "eval":
                        for b, v in r["mae"].items():
                            fila[f"mae_{b}"] = v
                        fila["mae_medio"] = r["mae_medio"]
            # §9.2: la brecha. El IoU de `train` NO es opcional -- sin el, facilitacion
            # y transferencia son indistinguibles, que es la distincion central (§2.3).
            fila["brecha"] = fila["iou_train"] - fila["iou_eval"]
            filas.append(fila)
            hist[epoca] = fila
            if al_evaluar:
                al_evaluar(fila)
    return filas, modelo


def _predictor_constante(d, y_px):
    """§10.1: la caja media. No entrena: ignora la imagen."""
    from evaluar import caja_media, resumen                 # noqa: PLC0415
    c = caja_media(y_px["train"])
    filas = []
    fila = {"epoca": 0, "semilla": 0}
    for part in ("train", "monitor", "eval"):
        p = np.repeat(c[None, :], len(y_px[part]), axis=0)
        r = resumen(p, y_px[part])
        fila[f"iou_{part}"] = r["iou"]
        if part == "eval":
            for b, v in r["mae"].items():
                fila[f"mae_{b}"] = v
            fila["mae_medio"] = r["mae_medio"]
    fila["brecha"] = fila["iou_train"] - fila["iou_eval"]
    fila["caja"] = [round(float(v), 3) for v in c]
    filas.append(fila)
    return filas


def correr(condicion: str, kernel_path: str | None, semilla: int, salida: Path) -> dict:
    d, origen, ruta = cargar()
    kernel = np.load(kernel_path).astype(np.float32) if kernel_path else None
    x, y, y_px, (mu, sd) = preparar(d, kernel)
    t0 = time.time()
    if condicion == "caja-media":
        filas = _predictor_constante(d, y_px)
    else:
        filas, _ = entrenar(x, y, y_px, semilla)
    seg = time.time() - t0

    salida.mkdir(parents=True, exist_ok=True)
    cab = list(filas[-1].keys())
    with (salida / "metricas.csv").open("w", encoding="utf-8") as f:
        f.write(",".join(cab) + "\n")
        for fila in filas:
            f.write(",".join(str(fila.get(c, "")) for c in cab) + "\n")
    cfg = {
        "condicion": condicion, "semilla": semilla,
        "kernel": kernel_path,
        "kernel_k": int(kernel.shape[0]) if kernel is not None else None,
        # §13.3: la norma ORIGINAL y el hash ANTES de normalizar. Despues de
        # normalizar, dos kernels que solo se diferencian en escala tienen el mismo
        # hash: la trazabilidad se perderia justo donde hace falta.
        "kernel_norma_original": float(np.linalg.norm(kernel)) if kernel is not None else None,
        "kernel_sha256_16": (__import__("hashlib").sha256(
            np.ascontiguousarray(kernel).tobytes()).hexdigest()[:16]
            if kernel is not None else None),
        "dataset": NOMBRE_DATASET, "dataset_origen": origen, "dataset_ruta": str(ruta),
        "especificacion": "v1.2",
        "epocas": EPOCAS, "lr": LR, "lote": LOTE, "paradas": list(PARADAS),
        "estandarizacion": {"mu": mu, "sigma": sd},
        "segundos": round(seg, 1),
    }
    (salida / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"filas": filas, "config": cfg}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condicion", default=None)
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--comprobar", action="store_true")
    a = ap.parse_args(argv)

    if a.comprobar or a.condicion is None:
        from modelo import main as m1                       # noqa: PLC0415
        from pipeline import _main as m2                    # noqa: PLC0415
        from evaluar import _main as m3                     # noqa: PLC0415
        rc = m1() + m2() + m3()
        if a.condicion is None and not a.comprobar:
            print("Para entrenar:  --condicion <nombre> [--kernel k.npy] [--semilla N]")
            print("Para calibrar:  python nn/calibrar.py --todo\n")
        return rc

    salida = EXP / "resultados" / (a.condicion if a.condicion == "caja-media"
                                   else f"{a.condicion}-s{a.semilla}")
    r = correr(a.condicion, a.kernel, a.semilla, salida)
    u = r["filas"][-1]
    print(f"{a.condicion} s{a.semilla}: iou_eval={u['iou_eval']:.4f} "
          f"iou_train={u['iou_train']:.4f} brecha={u['brecha']:+.4f} "
          f"mae={u['mae_medio']:.2f} px  ({r['config']['segundos']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
