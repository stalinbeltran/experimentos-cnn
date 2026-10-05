#!/usr/bin/env python3
"""Los datos de `feat-ind32`, generados UNA vez aquí y publicados en el repo de datos; después se leen por su
nombre y no se regeneran.

  SINTETICAS  `feat-ind32-sinteticas-32px-r20261005`: las 32.400 features de `feat-ind`, SIN reducir (bitmap
              binario 32×32). Mismo sorteo que allí: su reducción 4×4 es bit a bit `feat-ind-sinteticas-8px-r20261003`.
  DIGITOS     `uci-optdigits-orig-32px-r20261005`: los 5620 dígitos de UCI *optdigits-orig* (NIST, 32×32 binarios).
              `windep` (1797, 13 escritores) son EXACTAMENTE los de `uci-optdigits-8px-r20261002`, en el mismo orden
              (medido 2026-10-05: 1797/1797 iguales al reducir); se les pone su mismo reparto 180/1617. Los otros
              3823 (`tra`, `cv`, `wdep`: 30 escritores distintos) van como `extra`.

    python nn/datos.py --generar [--publicar]     las sintéticas
    python nn/datos.py --digitos <dir> [--publicar]   los dígitos, desde el zip de UCI descomprimido (los .Z)
    python nn/datos.py --comprobar                huellas + las dos igualdades con los publicados de 8 px
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
sys.path.insert(0, str(EXP.parent))                       # el repo: donde vive `expcnn`
sys.path.insert(0, str(AQUI))
from expcnn import exigir_dataset, exigir_datos, SUBDIR_DATASETS  # noqa: E402
import features as F                                               # noqa: E402

DATASET = "feat-ind32-sinteticas-32px-r20261005"
DIGITOS = "uci-optdigits-orig-32px-r20261005"
SINTETICAS_8PX = "feat-ind-sinteticas-8px-r20261003"      # sólo para COMPROBAR la igualdad; no se entrena con él
DIGITOS_8PX = "uci-optdigits-8px-r20261002"               # ídem, y de él sale el reparto 180/1617
URL_UCI = "https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip"
FICHEROS_UCI = ("windep", "tra", "cv", "wdep")             # windep PRIMERO: sus índices 0..1796 = los del 8 px
N_TRAIN, N_VAL = 2000, 400
N_TRAIN_VACIO, N_VAL_VACIO = 1000, 200
SEMILLA_BASE = 20261003                                    # LA DE feat-ind: es lo que hace iguales los dos datasets
LADO = 32
CAMPOS = ("imagenes", "principal", "secundaria", "ancla", "ancla32", "grosor", "radio", "angulo", "apertura",
          "largo", "particion")


def huella(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def semilla(familia: str, particion: str) -> int:
    return SEMILLA_BASE + 10 * F.FAMILIAS.index(familia) + (0 if particion == "train" else 1)


def generar() -> dict:
    F.usar_perfil("fino")
    filas = {k: [] for k in CAMPOS + ("imagen8",)}
    for fam in F.FAMILIAS:
        for part, n in (("train", N_TRAIN_VACIO if fam == F.VACIO else N_TRAIN),
                        ("val", N_VAL_VACIO if fam == F.VACIO else N_VAL)):
            rng = np.random.default_rng(semilla(fam, part))
            for _ in range(n):
                s = F.muestra(rng, fam)
                filas["imagenes"].append(s["imagen"]); filas["imagen8"].append(s["imagen8"])
                for k in ("principal", "secundaria", "ancla", "ancla32", "grosor", "radio", "angulo", "apertura", "largo"):
                    filas[k].append(s[k])
                filas["particion"].append(part)
    return {"imagenes": np.stack(filas["imagenes"]).astype(np.uint8), "imagen8": np.stack(filas["imagen8"]).astype(np.uint8),
            "principal": np.array(filas["principal"], np.int8), "secundaria": np.array(filas["secundaria"], np.int8),
            "ancla": np.stack(filas["ancla"]).astype(np.int8), "ancla32": np.stack(filas["ancla32"]).astype(np.float32),
            "grosor": np.array(filas["grosor"], np.int8), "radio": np.array(filas["radio"], np.float32),
            "angulo": np.array(filas["angulo"], np.float32), "apertura": np.array(filas["apertura"], np.float32),
            "largo": np.array(filas["largo"], np.float32), "particion": np.array(filas["particion"])}


def leer_uci(ruta: Path) -> tuple[np.ndarray, np.ndarray]:
    """Un fichero optdigits-orig (.Z o descomprimido): 21 líneas de cabecera, y por dígito 32 líneas de 32
    caracteres 0/1 seguidas de la etiqueta."""
    if ruta.suffix == ".Z":
        crudo = subprocess.run(["gzip", "-dc", str(ruta)], check=True, capture_output=True).stdout.decode()
    else:
        crudo = ruta.read_text()
    imgs, ys, buf = [], [], []
    for linea in crudo.split("\n")[21:]:
        s = linea.strip()
        if len(s) == 32 and set(s) <= {"0", "1"}:
            buf.append([c == "1" for c in s])
        elif s.isdigit() and len(buf) == 32:
            imgs.append(buf); ys.append(int(s)); buf = []
    return np.array(imgs, np.uint8), np.array(ys, np.int64)


def digitos_desde_uci(carpeta: Path) -> dict:
    xs, ys, origen = [], [], []
    for nombre in FICHEROS_UCI:
        cand = [carpeta / f"optdigits-orig.{nombre}.Z", carpeta / f"optdigits-orig.{nombre}"]
        x, y = leer_uci(next(c for c in cand if c.is_file()))
        xs.append(x); ys.append(y); origen += [nombre] * len(y)
    x, y, origen = np.concatenate(xs), np.concatenate(ys), np.array(origen)
    # el reparto de los 1797 windep es EL MISMO que el publicado a 8 px (por índice, que es el mismo dígito)
    z8 = np.load(exigir_dataset(DIGITOS_8PX) / "datos.npz")
    w = origen == "windep"
    if not (x[w].reshape(-1, 8, 4, 8, 4).sum((2, 4)) == z8["imagenes"]).all() or not (y[w] == z8["etiquetas"]).all():
        raise SystemExit(f"✗ los windep reducidos NO son {DIGITOS_8PX}: el reparto por índice no valdría. Me niego.")
    particion = np.full(len(y), "extra", dtype="<U5")
    particion[w] = z8["particion"]
    return {"imagenes": x, "etiquetas": y, "particion": particion, "origen": origen}


def _publicar(nombre: str, d: dict, man: dict, readme: str) -> Path:
    destino = exigir_datos() / SUBDIR_DATASETS / nombre
    if (destino / "manifiesto.json").exists():
        raise SystemExit(f"✗ {nombre} ya está publicado: un dataset no se reescribe nunca. Dato nuevo = nombre nuevo.")
    destino.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destino / "datos.npz", **d)
    man["huellas"] = {k: huella(v) for k, v in d.items()}
    (destino / "manifiesto.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (destino / "README.md").write_text(readme, encoding="utf-8")
    return destino


def publicar_sinteticas(d: dict) -> Path:
    d = {k: v for k, v in d.items() if k != "imagen8"}
    man = {"nombre": DATASET, "experimento": "feat-ind32", "generado": time.strftime("%Y-%m-%d"),
           "origen": f"SINTÉTICO: el mismo sorteo que {SINTETICAS_8PX} (semillas {SEMILLA_BASE}+…), guardado SIN reducir",
           "familias": list(F.FAMILIAS), "lado": LADO, "valor_max": 1, "n": int(len(d["imagenes"])),
           "campos": {"imagenes": "(N,32,32) uint8 0/1, CON ruido y con la secundaria", "ancla": "(fila, col) de la celda 8x8 (4 px)",
                      "ancla32": "(fila, col) en px de 32"}}
    readme = (f"# `{DATASET}`\n\nLas features sintéticas de `feat-ind` (13 familias + `vacio`), **sin reducir**: bitmap binario "
              f"32×32. Mismo sorteo que `{SINTETICAS_8PX}`: reducir cada imagen contando bloques 4×4 da exactamente "
              f"aquel dataset (lo comprueba `nn/datos.py --comprobar` de `feat-ind32`).\n\n**Quién lo usa:** `feat-ind32` de "
              f"`experimentos-cnn`.\n")
    return _publicar(DATASET, d, man, readme)


def publicar_digitos(d: dict) -> Path:
    man = {"nombre": DIGITOS, "experimento": "feat-ind32", "generado": time.strftime("%Y-%m-%d"), "url": URL_UCI,
           "origen": "UCI Optical Recognition of Handwritten Digits, versión ORIGINAL (optdigits-orig): bitmaps 32x32 de NIST",
           "n": int(len(d["etiquetas"])), "por_origen": {o: int((d["origen"] == o).sum()) for o in FICHEROS_UCI},
           "particion": {p: int((d["particion"] == p).sum()) for p in ("train", "val", "extra")},
           "campos": {"imagenes": "(N,32,32) uint8 0/1", "etiquetas": "0..9", "origen": "windep|tra|cv|wdep",
                      "particion": f"windep: la de {DIGITOS_8PX} (train/val); el resto: extra"}}
    readme = (f"# `{DIGITOS}`\n\nLos 5620 dígitos de UCI *optdigits-orig* (bitmaps binarios 32×32 de NIST, sin reducir), de "
              f"<{URL_UCI}>.\n\n- `windep` (1797, 13 escritores) son **los mismos dígitos y en el mismo orden** que "
              f"`{DIGITOS_8PX}`: reducidos por bloques 4×4 coinciden 1797/1797 (medido 2026-10-05). Llevan su reparto "
              f"180 train / 1617 val.\n- `tra`, `cv`, `wdep` (3823, otros 30 escritores): `particion = extra`.\n\n"
              f"**Quién lo usa:** `feat-ind32` de `experimentos-cnn`.\n")
    return _publicar(DIGITOS, d, man, readme)


def _cargar(nombre: str) -> dict:
    raiz = exigir_dataset(nombre)
    man = json.loads((raiz / "manifiesto.json").read_text(encoding="utf-8"))
    d = dict(np.load(raiz / "datos.npz"))
    for k, h in man["huellas"].items():
        if huella(d[k]) != h:
            raise RuntimeError(f"{nombre}/datos.npz: '{k}' no casa con su manifiesto. Me niego.")
    d["manifiesto"] = man; d["nombre"] = nombre
    return d


def cargar() -> dict:
    d = _cargar(DATASET)
    if list(d["manifiesto"]["familias"]) != list(F.FAMILIAS):
        raise RuntimeError("las familias del dataset y del código no casan. Me niego.")
    return d


def conjunto(d: dict, familia: str, particion: str) -> dict:
    """Positivos: principal == f. Negativos: principal != f y secundaria != f, y ni la principal ni la secundaria una
    familia que CONTIENE a f (la enmienda de la corrida 2 de feat-ind, adoptada aquí desde el principio).
    x (N, 1, 32, 32) float32 0/1."""
    i = F.FAMILIAS.index(familia)
    part = d["particion"] == particion
    pos = part & (d["principal"] == i)
    neg = part & (d["principal"] != i) & (d["secundaria"] != i)
    cont = [F.FAMILIAS.index(c) for c in F.contenedoras(familia)]
    if cont:
        neg &= ~np.isin(d["principal"], cont) & ~np.isin(d["secundaria"], cont)
    x = d["imagenes"].astype(np.float32)[:, None]
    return {"x_pos": x[pos], "ancla_pos": d["ancla"][pos].astype(np.int64), "radio_pos": d["radio"][pos], "grosor_pos": d["grosor"][pos],
            "x_neg": x[neg], "familia_neg": d["principal"][neg], "secundaria_pos": d["secundaria"][pos]}


def digitos(con_extra: bool = False) -> dict:
    """Los dígitos 32×32. Por defecto sólo los 1797 windep (comparables con feat-ind), con su reparto 180/1617;
    con_extra añade los 3823 de otros escritores (particion 'extra')."""
    d = _cargar(DIGITOS)
    keep = np.ones(len(d["etiquetas"]), bool) if con_extra else d["origen"] == "windep"
    return {"x": d["imagenes"][keep].astype(np.float32)[:, None], "y": d["etiquetas"][keep].astype(np.int64),
            "train": d["particion"][keep] == "train", "extra": d["particion"][keep] == "extra", "origen": d["origen"][keep]}


def comprobar() -> int:
    ok = True
    d = cargar()
    r8 = d["imagenes"].reshape(-1, 8, 4, 8, 4).sum((2, 4)).astype(np.uint8)
    man8 = json.loads((exigir_dataset(SINTETICAS_8PX) / "manifiesto.json").read_text(encoding="utf-8"))
    igual = huella(r8) == man8["huellas"]["imagenes"]
    print(f"  [{'ok' if igual else 'FALLA':>5}] sintéticas: reducir 4×4 da {SINTETICAS_8PX} bit a bit"); ok &= igual
    dg = _cargar(DIGITOS); w = dg["origen"] == "windep"
    z8 = np.load(exigir_dataset(DIGITOS_8PX) / "datos.npz")
    igual = bool((dg["imagenes"][w].reshape(-1, 8, 4, 8, 4).sum((2, 4)) == z8["imagenes"]).all() and (dg["etiquetas"][w] == z8["etiquetas"]).all()
                 and (dg["particion"][w] == z8["particion"]).all())
    print(f"  [{'ok' if igual else 'FALLA':>5}] dígitos: windep reducido = {DIGITOS_8PX}, mismas etiquetas y reparto"); ok &= igual
    print(f"  {DATASET}: {len(d['imagenes'])} · {DIGITOS}: {dg['manifiesto']['por_origen']}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generar", action="store_true"); p.add_argument("--publicar", action="store_true")
    p.add_argument("--digitos", type=Path, help="carpeta con los optdigits-orig.*.Z del zip de UCI")
    p.add_argument("--comprobar", action="store_true")
    a = p.parse_args()
    if a.generar:
        t0 = time.time(); d = generar()
        man8 = json.loads((exigir_dataset(SINTETICAS_8PX) / "manifiesto.json").read_text(encoding="utf-8"))
        igual = huella(d["imagen8"]) == man8["huellas"]["imagenes"]
        print(f"generadas {len(d['imagenes'])} imágenes en {time.time() - t0:.0f} s; reducidas = {SINTETICAS_8PX}: {igual}")
        if not igual:
            raise SystemExit("✗ el sorteo ya no da el dataset de feat-ind: no se publica.")
        if a.publicar:
            print(f"→ publicado en {publicar_sinteticas(d)}")
        return 0
    if a.digitos:
        d = digitos_desde_uci(a.digitos)
        print(f"{len(d['etiquetas'])} dígitos; windep = {DIGITOS_8PX} ✓")
        if a.publicar:
            print(f"→ publicado en {publicar_digitos(d)}")
        return 0
    if a.comprobar:
        return comprobar()
    p.print_help(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
