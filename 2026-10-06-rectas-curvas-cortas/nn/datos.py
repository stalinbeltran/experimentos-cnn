#!/usr/bin/env python3
"""Los datos de `feat-cortas`, generados UNA vez aquí y publicados en el repo de datos; después se leen por su nombre.

  SINTETICAS  `feat-cortas-sinteticas-32px-r20261006`: las 8 primitivas cortas (4 rectas de 6–12 px, 4 arcos de radio 5–10 px
              y 60–100° de apertura) + `vacio`, a 32×32 binario, 2–4 px de grosor, con el ruido y la segunda feature de
              feat-ind32. 2000 train + 400 val por familia; `vacio`, 1000 + 200.
  DIGITOS     `uci-optdigits-orig-32px-r20261005`: los 5620 dígitos de NIST (windep con su reparto 180/1617).

    python nn/datos.py --generar [--publicar]
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
import features as F                                               # noqa: E402

DATASET = "feat-cortas-sinteticas-32px-r20261006"
DIGITOS = "uci-optdigits-orig-32px-r20261005"
N_TRAIN, N_VAL, N_TRAIN_VACIO, N_VAL_VACIO = 2000, 400, 1000, 200
SEMILLA_BASE = 20261006
CAMPOS = ("imagenes", "principal", "secundaria", "ancla", "ancla32", "grosor", "radio", "angulo", "apertura", "largo")


def huella(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def semilla(familia: str, particion: str) -> int:
    return SEMILLA_BASE + 10 * F.FAMILIAS.index(familia) + (0 if particion == "train" else 1)


def generar() -> dict:
    filas = {k: [] for k in CAMPOS + ("particion",)}
    for fam in F.FAMILIAS:
        for part, n in (("train", N_TRAIN_VACIO if fam == F.VACIO else N_TRAIN), ("val", N_VAL_VACIO if fam == F.VACIO else N_VAL)):
            rng = np.random.default_rng(semilla(fam, part))
            for _ in range(n):
                s = F.muestra(rng, fam)
                filas["imagenes"].append(s["imagen"])
                for k in CAMPOS[1:]:
                    filas[k].append(s[k])
                filas["particion"].append(part)
    return {"imagenes": np.stack(filas["imagenes"]).astype(np.uint8), "principal": np.array(filas["principal"], np.int8),
            "secundaria": np.array(filas["secundaria"], np.int8), "ancla": np.stack(filas["ancla"]).astype(np.int8),
            "ancla32": np.stack(filas["ancla32"]).astype(np.float32), "grosor": np.array(filas["grosor"], np.int8),
            "radio": np.array(filas["radio"], np.float32), "angulo": np.array(filas["angulo"], np.float32),
            "apertura": np.array(filas["apertura"], np.float32), "largo": np.array(filas["largo"], np.float32),
            "particion": np.array(filas["particion"])}


def publicar(d: dict) -> Path:
    destino = exigir_datos() / SUBDIR_DATASETS / DATASET
    if (destino / "manifiesto.json").exists():
        raise SystemExit(f"✗ {DATASET} ya está publicado: un dataset no se reescribe nunca.")
    destino.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destino / "datos.npz", **d)
    man = {"nombre": DATASET, "experimento": "feat-cortas", "generado": time.strftime("%Y-%m-%d"),
           "origen": f"SINTÉTICO: features.py de feat-cortas (rectas y curvas cortas), semillas {SEMILLA_BASE}+10·familia(+1 en val)",
           "familias": list(F.FAMILIAS), "lado": 32, "valor_max": 1, "n": int(len(d["imagenes"])),
           "huellas": {k: huella(v) for k, v in d.items()}}
    (destino / "manifiesto.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (destino / "README.md").write_text(
        f"# `{DATASET}`\n\nRectas cortas (6–12 px: |, —, /, \\) y curvas cortas (arcos de radio 5–10 px y 60–100° de apertura, con "
        f"el centro a E/W/N/S) a 32×32, grosor 2–4 px, con ruido y una segunda feature en la mitad. 2000 + 400 por familia, "
        f"`vacio` 1000 + 200.\n\n**Quién lo usa:** `feat-cortas` de `experimentos-cnn`.\n", encoding="utf-8")
    return destino


def cargar() -> dict:
    raiz = exigir_dataset(DATASET)
    man = json.loads((raiz / "manifiesto.json").read_text(encoding="utf-8"))
    d = dict(np.load(raiz / "datos.npz"))
    for k, h in man["huellas"].items():
        if huella(d[k]) != h:
            raise RuntimeError(f"{DATASET}: '{k}' no casa con su manifiesto. Me niego.")
    if list(man["familias"]) != list(F.FAMILIAS):
        raise RuntimeError("las familias del dataset y del código no casan. Me niego.")
    d["manifiesto"] = man; d["nombre"] = DATASET
    return d


def conjunto(d: dict, familia: str, particion: str) -> dict:
    """Positivos: principal == f. Negativos: ni la principal ni la secundaria son f (aquí ninguna familia contiene a otra)."""
    i = F.FAMILIAS.index(familia)
    part = d["particion"] == particion
    pos = part & (d["principal"] == i)
    neg = part & (d["principal"] != i) & (d["secundaria"] != i)
    x = d["imagenes"].astype(np.float32)[:, None]                    # float32, como en feat-ind32 (su entrenar_local lo espera)
    return {"x_pos": x[pos], "ancla_pos": d["ancla"][pos].astype(np.int64), "radio_pos": d["radio"][pos], "grosor_pos": d["grosor"][pos],
            "x_neg": x[neg], "familia_neg": d["principal"][neg], "secundaria_pos": d["secundaria"][pos]}


def comprobar() -> int:
    d = cargar(); c = conjunto(d, "curva-E", "train")
    ok = len(c["x_pos"]) == N_TRAIN and c["x_pos"].shape[1:] == (1, 32, 32)
    print(f"  [{'ok' if ok else 'FALLA':>5}] {DATASET}: huellas · {len(d['imagenes'])} imágenes · curva-E train {c['x_pos'].shape}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generar", action="store_true"); p.add_argument("--publicar", action="store_true")
    p.add_argument("--comprobar", action="store_true")
    a = p.parse_args()
    if a.generar:
        t0 = time.time(); d = generar()
        print(f"generadas {len(d['imagenes'])} imágenes en {time.time() - t0:.0f} s")
        if a.publicar:
            print(f"→ publicado en {publicar(d)}")
        return 0
    if a.comprobar:
        return comprobar()
    p.print_help(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
