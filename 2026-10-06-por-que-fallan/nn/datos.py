#!/usr/bin/env python3
"""Los datos que ENTRENA `feat-fallos` (lo que sólo evalúa lo lee nn/evaluar.py por su nombre).

  GRUESO_ENTRENAR  `feat-fallos-sinteticas-grueso-32px-r20261006`: el vocabulario y el sorteo GRUESO de feat-ind (sus
                   semillas, 2000 + 400 por familia) con el perfil `grueso` de features.py: grosores 2, 3, 4, 6, 8, 10 y
                   12 px, el rango de los dígitos. Reducido 4×4 tiene que dar EXACTAMENTE `feat-ind-sinteticas-grueso-8px-
                   r20261003d` (la corrida 4 de feat-ind, a 8×8): se comprueba antes de publicar.
  FINO             `feat-ind32-sinteticas-32px-r20261005` (el de siempre, 2–4 px), para los bancos que se entrenen finos.

    python nn/datos.py --generar-grueso [--publicar]
    python nn/datos.py --comprobar
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent)); sys.path.insert(0, str(AQUI))
from expcnn import exigir_dataset, exigir_datos, SUBDIR_DATASETS  # noqa: E402
import bordes as B                                                 # noqa: E402
import features as F                                               # noqa: E402

GRUESO_ENTRENAR = "feat-fallos-sinteticas-grueso-32px-r20261006"
DATASET = GRUESO_ENTRENAR                                         # el nombre que espera entrenar_local.py (copiado)
FINO = "feat-ind32-sinteticas-32px-r20261005"
GRUESO_8PX = "feat-ind-sinteticas-grueso-8px-r20261003d"
N_TRAIN, N_VAL, N_TRAIN_VACIO, N_VAL_VACIO = 2000, 400, 1000, 200
SEMILLA_BASE = 20261003 + 1000                                      # la de feat-ind para su perfil grueso (SALTO_SEMILLA)
CAMPOS = ("imagenes", "principal", "secundaria", "ancla", "ancla32", "grosor", "radio", "angulo", "apertura", "largo")


def huella(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def generar_grueso() -> dict:
    F.usar_perfil("grueso")
    filas = {k: [] for k in CAMPOS + ("particion",)}
    for fam in F.FAMILIAS:
        for part, n in (("train", N_TRAIN_VACIO if fam == F.VACIO else N_TRAIN), ("val", N_VAL_VACIO if fam == F.VACIO else N_VAL)):
            rng = np.random.default_rng(SEMILLA_BASE + 10 * F.FAMILIAS.index(fam) + (0 if part == "train" else 1))
            for _ in range(n):
                s = F.muestra(rng, fam)
                filas["imagenes"].append(s["imagen"])
                for k in CAMPOS[1:]:
                    filas[k].append(s[k])
                filas["particion"].append(part)
    F.usar_perfil("fino")
    return {"imagenes": np.stack(filas["imagenes"]).astype(np.uint8), "principal": np.array(filas["principal"], np.int8),
            "secundaria": np.array(filas["secundaria"], np.int8), "ancla": np.stack(filas["ancla"]).astype(np.int8),
            "ancla32": np.stack(filas["ancla32"]).astype(np.float32), "grosor": np.array(filas["grosor"], np.int8),
            "radio": np.array(filas["radio"], np.float32), "angulo": np.array(filas["angulo"], np.float32),
            "apertura": np.array(filas["apertura"], np.float32), "largo": np.array(filas["largo"], np.float32),
            "particion": np.array(filas["particion"])}


def igual_que_8px(d: dict) -> bool:
    man8 = json.loads((exigir_dataset(GRUESO_8PX) / "manifiesto.json").read_text(encoding="utf-8"))
    r8 = d["imagenes"].reshape(-1, 8, 4, 8, 4).sum((2, 4)).astype(np.uint8)
    return huella(r8) == man8["huellas"]["imagenes"]


def publicar_grueso(d: dict) -> Path:
    destino = exigir_datos() / SUBDIR_DATASETS / GRUESO_ENTRENAR
    if (destino / "manifiesto.json").exists():
        raise SystemExit(f"✗ {GRUESO_ENTRENAR} ya está publicado: un dataset no se reescribe nunca.")
    destino.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destino / "datos.npz", **d)
    man = {"nombre": GRUESO_ENTRENAR, "experimento": "feat-fallos", "generado": time.strftime("%Y-%m-%d"),
           "origen": f"SINTÉTICO: el sorteo de feat-ind32 (semillas {SEMILLA_BASE}+…) con el perfil `grueso`, SIN reducir; "
                     f"reducido 4×4 es bit a bit {GRUESO_8PX}",
           "familias": list(F.FAMILIAS), "lado": 32, "valor_max": 1, "n": int(len(d["imagenes"])),
           "huellas": {k: huella(v) for k, v in d.items()}}
    (destino / "manifiesto.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (destino / "README.md").write_text(
        f"# `{GRUESO_ENTRENAR}`\n\nLas 13 features de feat-ind32 con grosores 2–12 px (el perfil `grueso`), a 32×32 sin reducir, "
        f"para ENTRENAR. Reducido 4×4 da bit a bit `{GRUESO_8PX}` (la corrida 4 de feat-ind).\n\n**Quién lo usa:** `feat-fallos` "
        f"de `experimentos-cnn`.\n", encoding="utf-8")
    return destino


def cargar(nombre: str = GRUESO_ENTRENAR) -> dict:
    raiz = exigir_dataset(nombre)
    man = json.loads((raiz / "manifiesto.json").read_text(encoding="utf-8"))
    d = dict(np.load(raiz / "datos.npz"))
    for k, h in man["huellas"].items():
        if huella(d[k]) != h:
            raise RuntimeError(f"{nombre}: '{k}' no casa con su manifiesto. Me niego.")
    d["manifiesto"] = man; d["nombre"] = nombre
    return d


def conjunto(d: dict, familia: str, particion: str, modo: str = "lineas") -> dict:
    """La de feat-bor (con CONTIENE), en uint8; se pasa a float por lote."""
    i = F.FAMILIAS.index(familia)
    part = d["particion"] == particion
    pos = part & (d["principal"] == i)
    neg = part & (d["principal"] != i) & (d["secundaria"] != i)
    cont = [F.FAMILIAS.index(c) for c in F.contenedoras(familia)]
    if cont:
        neg &= ~np.isin(d["principal"], cont) & ~np.isin(d["secundaria"], cont)
    return {"x_pos": B.bordes(d["imagenes"][pos], modo), "ancla_pos": d["ancla"][pos].astype(np.int64),
            "radio_pos": d["radio"][pos], "grosor_pos": d["grosor"][pos], "secundaria_pos": d["secundaria"][pos],
            "x_neg": B.bordes(d["imagenes"][neg], modo), "familia_neg": d["principal"][neg]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generar-grueso", action="store_true"); p.add_argument("--publicar", action="store_true")
    p.add_argument("--comprobar", action="store_true")
    a = p.parse_args()
    if a.generar_grueso:
        t0 = time.time(); d = generar_grueso(); igual = igual_que_8px(d)
        print(f"generadas {len(d['imagenes'])} imágenes en {time.time() - t0:.0f} s; reducidas 4×4 = {GRUESO_8PX}: {igual}")
        if a.publicar:
            if not igual:
                raise SystemExit("✗ no reproduce el sorteo de la corrida 4 de feat-ind: no se publica.")
            print(f"→ publicado en {publicar_grueso(d)}")
        return 0
    if a.comprobar:
        d = cargar(); c = conjunto(d, "recta-V", "train")
        print(f"  [   ok] {GRUESO_ENTRENAR}: {len(d['imagenes'])} imágenes; recta-V train {c['x_pos'].shape}; grosores "
              + ", ".join(f"{w}: {(d['grosor'] == w).sum()}" for w in sorted(set(d['grosor'].tolist())) if w > 0))
        return 0
    p.print_help(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
