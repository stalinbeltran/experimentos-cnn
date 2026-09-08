#!/usr/bin/env python3
"""LA PUERTA DEL BANCO: mete un kernel, saca un veredicto. `banco-k`.

    python nn/evaluar_kernel.py --contrato kernels/mio.npy    solo valida (§5), no entrena
    python nn/evaluar_kernel.py --kernel   kernels/mio.npy    lo evalua entero
    python nn/evaluar_kernel.py --kernel kernels/mio.npy --nombre mio-v2

Es lo unico que hay que saber para probar un kernel. Todo lo demas -- el pipeline, el
protocolo, las semillas, los controles -- ya esta fijado y CONGELADO (§12).

EL CONTRATO DE ENTRADA (§5.1), y se comprueba ANTES de entrenar nada
====================================================================
    fichero   .npy
    forma     (k, k)          cuadrado
    tipo      float32         (se convierte si hace falta)
    k         IMPAR, 3 <= k <= 19
    canales   1 -> 1

No hace falta normalizar nada: el banco normaliza la norma L2 al recibirlo (§5.4), y
guarda la norma ORIGINAL y el hash de ANTES de normalizar para la trazabilidad
(§13.3). Dos kernels que solo se diferencian en escala son EL MISMO detector y aqui
dan exactamente el mismo resultado.

⚠⚠ EL CONTROL ALEATORIO TIENE QUE SER DEL MISMO `k` (§2.1: «igual norma y mismo k»).
La calibracion corrio el suyo con k=9. Si tu kernel tiene otro `k`, este script
CORRE PRIMERO un control aleatorio nuevo con ese `k` y sus 10 semillas, porque
compararte contra un aleatorio de otro tamanyo no contesta la pregunta del §2.1 --
mezclaria la forma del kernel con su campo receptivo. Cuesta 10 corridas mas
(~9 min aqui) y se hace una sola vez por `k`.

⚠ DE DONDE SALE EL KERNEL NO ES ASUNTO DE ESTE BANCO (§1, §15). Lo unico que se pide
es el contrato. Pero si lo has obtenido con el MISMO generador de parrafos, respeta
la RESERVA del §3.7 -- `LiberationMono` y el interlineado [1,45 · 1,60] son de uso
exclusivo del banco -- o tendras fuga de distribucion aunque las muestras sean otras.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(EXP.parent))

RESULTADOS = EXP / "resultados"
SEMILLAS = list(range(10))          # §8.5: 10, fijas. Es invariante (§12)
K_CALIBRADO = 9                     # el `k` con el que corrio la calibracion


def _dir_aleatorio(k: int) -> str:
    """El control aleatorio de este `k`. El de la calibracion (k=9) se reutiliza.

    Se reutiliza a proposito y no se recalcula: son EXACTAMENTE las mismas corridas,
    con las mismas semillas y el mismo dataset, asi que recalcularlas daria los mismos
    numeros y gastaria 9 minutos en confirmarlo."""
    return "aleatorio" if k == K_CALIBRADO else f"aleatorio-k{k}"


def contrato(ruta: Path) -> dict:
    """§5.1: valida y describe. Se niega ANTES de tocar el dataset (R2)."""
    from pipeline import comprobar_contrato, normalizar_kernel   # noqa: PLC0415
    if not ruta.is_file():
        raise SystemExit(f"✗ no existe: {ruta}")
    if ruta.suffix != ".npy":
        raise SystemExit(f"✗ §5.1: el kernel va en un .npy, y esto es '{ruta.suffix}'")
    k = np.load(ruta)
    if k.dtype != np.float32:
        print(f"  aviso: dtype {k.dtype} -> se convierte a float32 (§5.1)")
        k = k.astype(np.float32)
    comprobar_contrato(k)                       # forma, k impar, 3 <= k <= 19
    kn = normalizar_kernel(k)
    return {"k": int(k.shape[0]), "norma_original": float(np.linalg.norm(k)),
            "suma": float(k.sum()), "suma_normalizada": float(kn.sum()),
            "sha256_16": hashlib.sha256(np.ascontiguousarray(k).tobytes()).hexdigest()[:16],
            "min": float(kn.min()), "max": float(kn.max())}


def _correr_grupo(condicion: str, kernel_de, semillas=SEMILLAS) -> dict:
    from calibrar import _grupo                 # noqa: PLC0415
    return _grupo(condicion, kernel_de, semillas)


def evaluar(ruta: Path, nombre: str | None) -> int:
    from evaluar import criterio_generalizacion, criterio_utilidad   # noqa: PLC0415

    nombre = nombre or ruta.stem
    c = contrato(ruta)
    k = c["k"]
    print(f"\n§5 contrato OK — k={k}, norma original {c['norma_original']:.4f}, "
          f"suma {c['suma']:+.4f}, sha {c['sha256_16']}\n")

    # El control aleatorio del MISMO k. Si no existe, se corre ahora (§2.1).
    d_al = _dir_aleatorio(k)
    f_al = RESULTADOS / d_al / "resumen.json"
    if not f_al.is_file():
        print(f"[control] no hay aleatorio con k={k}: se corre ahora, 10 semillas (§2.1)")
        from kernels import aleatorio            # noqa: PLC0415
        kdir = EXP / "kernels"
        kdir.mkdir(exist_ok=True)
        for s in SEMILLAS:
            f = kdir / f"{d_al}-r{s}.npy"
            if not f.is_file():
                np.save(f, aleatorio(k, 1000 + s))
        _correr_grupo(d_al, lambda s: str(kdir / f"{d_al}-r{s}.npy"))
    al = json.loads((RESULTADOS / d_al / "resumen.json").read_text(encoding="utf-8"))
    idn = json.loads((RESULTADOS / "identidad" / "resumen.json").read_text(encoding="utf-8"))

    print(f"[kernel] {nombre}: 10 semillas x 200 epocas")
    r = _correr_grupo(nombre, lambda s: str(ruta))

    u = criterio_utilidad(r["iou_eval"], al["iou_eval"])
    g = criterio_generalizacion(r["brecha"], idn["brecha"], u["cumple"])
    # §2.3: facilitacion o transferencia. Sin el IoU de `train` no se distinguen.
    d_tr = r["iou_train"]["media"] - idn["iou_train"]["media"]
    d_ev = r["iou_eval"]["media"] - idn["iou_eval"]["media"]
    mecanismo = ("transferencia" if d_ev > 0 and d_tr < d_ev / 2 else
                 "facilitacion" if d_ev > 0 else "sin efecto o perjudica")

    crit = {
        "kernel": str(ruta), "nombre": nombre, "contrato": c,
        "dataset": json.loads((RESULTADOS / nombre / "resumen.json")
                              .read_text(encoding="utf-8")).get("condicion", nombre),
        "comparado_contra": {"aleatorio": d_al, "identidad": "identidad"},
        "iou_eval": r["iou_eval"], "iou_train": r["iou_train"], "brecha": r["brecha"],
        "§2.1_utilidad": u, "§2.2_generalizacion": g,
        "§2.3_mecanismo": mecanismo,
        "veredicto": ("UTIL Y GENERALIZA" if g["cumple"] else
                      "UTIL (sin efecto de generalizacion)" if u["cumple"] else
                      "NO DECLARA"),
    }
    (RESULTADOS / nombre / "criterios.json").write_text(
        json.dumps(crit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n{'='*66}\n  {nombre}   k={k}\n{'='*66}")
    print(f"  IoU eval    {r['iou_eval']['media']:.4f} ± {r['iou_eval']['desv']:.4f}"
          f"   (aleatorio k={k}: {al['iou_eval']['media']:.4f} ± {al['iou_eval']['desv']:.4f})")
    print(f"  IoU train   {r['iou_train']['media']:.4f} ± {r['iou_train']['desv']:.4f}")
    print(f"  brecha      {r['brecha']['media']:+.4f} ± {r['brecha']['desv']:.4f}"
          f"   (identidad: {idn['brecha']['media']:+.4f} ± {idn['brecha']['desv']:.4f})")
    print(f"\n  §2.1 utilidad         dif {u['diferencia']:+.4f}  margen {u['margen']:.4f}"
          f"   -> {'CUMPLE' if u['cumple'] else 'no cumple'}")
    print(f"  §2.2 generalizacion   dif {g['diferencia']:+.4f}  margen {g['margen']:.4f}"
          f"   -> {'CUMPLE' if g['cumple'] else 'no cumple'}"
          + ("" if g["cumple"] or not g["cumple_ignorando_2_1"]
             else "  (pasaria, pero el §2.2 es CONDICIONAL al §2.1)"))
    print(f"  §2.3 mecanismo        {mecanismo}   (train {d_tr:+.4f}, eval {d_ev:+.4f})")
    print(f"\n  VEREDICTO: {crit['veredicto']}")
    print(f"  {RESULTADOS / nombre / 'criterios.json'}\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel")
    ap.add_argument("--contrato", help="solo valida el §5, no entrena")
    ap.add_argument("--nombre")
    a = ap.parse_args()
    # Un contrato incumplido es un ERROR DEL USUARIO, no un fallo del banco: se dice
    # en una linea y se sale con 2, en vez de escupir un traceback que hay que leer
    # entero para encontrar el motivo al final.
    try:
        if a.contrato:
            c = contrato(Path(a.contrato))
        elif a.kernel:
            return evaluar(Path(a.kernel), a.nombre)
    except ValueError as err:
        print(f"\n✗ el kernel no cumple el contrato del §5:\n  {err}\n")
        return 2
    if a.contrato:
        print("\n§5 contrato OK\n" + "\n".join(f"  {k:20} {v}" for k, v in c.items()) + "\n")
        return 0
    ap.error("dime que kernel: --kernel k.npy  ·  --contrato k.npy para solo validar")


if __name__ == "__main__":
    raise SystemExit(main())
