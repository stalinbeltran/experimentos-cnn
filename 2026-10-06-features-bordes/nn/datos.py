#!/usr/bin/env python3
"""Los datos de `feat-bor`. Se leen por su nombre (no se regeneran) y se pasan a BORDES al cargar (nn/bordes.py):

  ENTRENAR  `feat-ind32-sinteticas-32px-r20261005`: las MISMAS 32.400 features con las que se entrenaron los detectores de
            feat-ind32 (grosores 2–4 px). Lo único distinto es la transformación al cargar.
  GRUESO    `feat-bor-sinteticas-grueso-32px-r20261006`: la PRUEBA de grosor, generada y publicada aquí UNA vez. El mismo
            vocabulario dibujado con el perfil `grueso` de features.py (grosores 2, 3, 4, 6, 8, 10 y 12 px), 400 por
            familia + 200 vacías. Los detectores NUNCA ven más de 4 px al entrenar: 6–12 px es lo no visto.
  DIGITOS   `uci-optdigits-orig-32px-r20261005`: los 5620 dígitos de NIST (sus 1797 windep con el reparto 180/1617).

    python nn/datos.py --generar-grueso [--publicar]     la prueba de grosor (una vez)
    python nn/datos.py --comprobar                       huellas, conjuntos y que la prueba existe
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
EXP = AQUI.parent
sys.path.insert(0, str(EXP.parent))                       # el repo: donde vive `expcnn`
sys.path.insert(0, str(AQUI))
from expcnn import exigir_dataset, exigir_datos, SUBDIR_DATASETS  # noqa: E402
import bordes as B                                                 # noqa: E402
import features as F                                               # noqa: E402

DATASET = "feat-ind32-sinteticas-32px-r20261005"
GRUESO = "feat-bor-sinteticas-grueso-32px-r20261006"
DIGITOS = "uci-optdigits-orig-32px-r20261005"
N_GRUESO, N_GRUESO_VACIO = 400, 200
SEMILLA_GRUESO = 20261006                                  # otra que la de entrenar (20261003): otras instancias
N_TRAIN = 2000
CAMPOS = ("imagenes", "principal", "secundaria", "ancla", "ancla32", "grosor", "radio", "angulo", "apertura", "largo")


def huella(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _cargar(nombre: str) -> dict:
    raiz = exigir_dataset(nombre)
    man = json.loads((raiz / "manifiesto.json").read_text(encoding="utf-8"))
    d = dict(np.load(raiz / "datos.npz"))
    for k, h in man["huellas"].items():
        if huella(d[k]) != h:
            raise RuntimeError(f"{nombre}/datos.npz: '{k}' no casa con su manifiesto. Me niego.")
    d["manifiesto"] = man; d["nombre"] = nombre
    return d


def cargar(nombre: str = DATASET) -> dict:
    d = _cargar(nombre)
    if list(d["manifiesto"]["familias"]) != list(F.FAMILIAS):
        raise RuntimeError("las familias del dataset y del código no casan. Me niego.")
    return d


def conjunto(d: dict, familia: str, particion: str | None, modo: str) -> dict:
    """COPIA de `conjunto` de feat-ind32 con la representación al final. Positivos: principal == f. Negativos: principal
    != f y secundaria != f, y ni la principal ni la secundaria una familia que CONTIENE a f. `particion` None = todas (la
    prueba de grosor no tiene reparto). x (N, C, 32, 32) uint8 0/1: se pasa a float en cada lote (mismos valores)."""
    i = F.FAMILIAS.index(familia)
    part = np.ones(len(d["principal"]), bool) if particion is None else d["particion"] == particion
    pos = part & (d["principal"] == i)
    neg = part & (d["principal"] != i) & (d["secundaria"] != i)
    cont = [F.FAMILIAS.index(c) for c in F.contenedoras(familia)]
    if cont:
        neg &= ~np.isin(d["principal"], cont) & ~np.isin(d["secundaria"], cont)
    return {"x_pos": B.bordes(d["imagenes"][pos], modo), "ancla_pos": d["ancla"][pos].astype(np.int64),
            "radio_pos": d["radio"][pos], "grosor_pos": d["grosor"][pos], "secundaria_pos": d["secundaria"][pos],
            "x_neg": B.bordes(d["imagenes"][neg], modo), "familia_neg": d["principal"][neg], "grosor_neg": d["grosor"][neg]}


def generar_grueso() -> dict:
    """La prueba de grosor: el sorteo de features.py con el perfil `grueso`, otra semilla por familia."""
    F.usar_perfil("grueso")
    filas = {k: [] for k in CAMPOS}
    for fam in F.FAMILIAS:
        rng = np.random.default_rng(SEMILLA_GRUESO + 10 * F.FAMILIAS.index(fam))
        for _ in range(N_GRUESO_VACIO if fam == F.VACIO else N_GRUESO):
            s = F.muestra(rng, fam)
            filas["imagenes"].append(s["imagen"])
            for k in CAMPOS[1:]:
                filas[k].append(s[k])
    F.usar_perfil("fino")
    return {"imagenes": np.stack(filas["imagenes"]).astype(np.uint8), "principal": np.array(filas["principal"], np.int8),
            "secundaria": np.array(filas["secundaria"], np.int8), "ancla": np.stack(filas["ancla"]).astype(np.int8),
            "ancla32": np.stack(filas["ancla32"]).astype(np.float32), "grosor": np.array(filas["grosor"], np.int8),
            "radio": np.array(filas["radio"], np.float32), "angulo": np.array(filas["angulo"], np.float32),
            "apertura": np.array(filas["apertura"], np.float32), "largo": np.array(filas["largo"], np.float32)}


def publicar_grueso(d: dict) -> Path:
    destino = exigir_datos() / SUBDIR_DATASETS / GRUESO
    if (destino / "manifiesto.json").exists():
        raise SystemExit(f"✗ {GRUESO} ya está publicado: un dataset no se reescribe nunca. Dato nuevo = nombre nuevo.")
    destino.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destino / "datos.npz", **d)
    g = d["grosor"][d["principal"] != F.FAMILIAS.index(F.VACIO)]
    man = {"nombre": GRUESO, "experimento": "feat-bor", "generado": time.strftime("%Y-%m-%d"),
           "origen": f"SINTÉTICO: features.py de feat-bor (copia de feat-ind32) con el perfil `grueso`, semillas {SEMILLA_GRUESO}+10·familia",
           "familias": list(F.FAMILIAS), "lado": 32, "valor_max": 1, "n": int(len(d["imagenes"])),
           "por_grosor_de_la_principal": {str(int(w)): int((g == w).sum()) for w in sorted(set(g.tolist()))},
           "campos": {"imagenes": "(N,32,32) uint8 0/1, CON ruido y con la secundaria (también gruesa)",
                      "ancla": "(fila, col) de la celda 8x8", "grosor": "px de la principal"},
           "huellas": {k: huella(v) for k, v in d.items()}}
    (destino / "manifiesto.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (destino / "README.md").write_text(
        f"# `{GRUESO}`\n\nLa PRUEBA DE GROSOR de `feat-bor` (`experimentos-cnn`): las 13 features (+ `vacio`) de feat-ind32 "
        f"dibujadas con el perfil `grueso` (grosores 2, 3, 4, 6, 8, 10 y 12 px), {N_GRUESO} por familia y {N_GRUESO_VACIO} "
        f"vacías, sin reparto: sólo se evalúa. Los detectores se entrenan con 2–4 px; 6–12 px es lo no visto.\n\n"
        f"**Quién lo usa:** `feat-bor` de `experimentos-cnn`.\n", encoding="utf-8")
    return destino


def digitos() -> dict:
    """Los 5620 dígitos 32×32 (windep con su reparto 180/1617; el resto `extra`). x (N,1,32,32) uint8 0/1."""
    d = _cargar(DIGITOS)
    return {"x": d["imagenes"][:, None].astype(np.uint8), "y": d["etiquetas"].astype(np.int64),
            "train": d["particion"] == "train", "val": d["particion"] == "val", "origen": d["origen"].astype(str)}


def comprobar() -> int:
    ok = True

    def mira(que, cond, det=""):
        nonlocal ok
        print(f"  [{'ok' if cond else 'FALLA':>5}] {que}" + (f"  {det}" if det else "")); ok &= bool(cond)

    d = cargar()
    mira(f"{DATASET}: huellas del manifiesto", True, f"{len(d['imagenes'])} imágenes")
    for modo in B.REPRESENTACIONES:
        c = conjunto(d, "arco-E", "train", modo)
        mira(f"{modo}: arco-E train = 2000 positivas de ({B.CANALES[modo]}, 32, 32) uint8",
             len(c["x_pos"]) == N_TRAIN and c["x_pos"].shape[1:] == (B.CANALES[modo], 32, 32) and c["x_pos"].dtype == np.uint8)
    c = conjunto(d, "arco-E", "train", "lineas")
    mira("negativos de arco-E: sin lazo (lo contiene)", F.FAMILIAS.index("lazo") not in set(np.unique(c["familia_neg"])))
    try:
        g = cargar(GRUESO)
        gr = g["grosor"][g["principal"] != F.FAMILIAS.index(F.VACIO)]
        mira(f"{GRUESO}: huellas, {len(g['imagenes'])} imágenes", True,
             "grosores " + ", ".join(f"{w}: {(gr == w).sum()}" for w in sorted(set(gr.tolist()))))
    except (SystemExit, RuntimeError) as e:
        mira(f"{GRUESO} publicado", False, str(e))
    dg = digitos()
    mira(f"{DIGITOS}: 5620 dígitos, 180/1617 en windep", len(dg["y"]) == 5620 and dg["train"].sum() == 180 and dg["val"].sum() == 1617)
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generar-grueso", action="store_true"); p.add_argument("--publicar", action="store_true")
    p.add_argument("--comprobar", action="store_true")
    a = p.parse_args()
    if a.generar_grueso:
        t0 = time.time(); d = generar_grueso()
        g = d["grosor"][d["principal"] != F.FAMILIAS.index(F.VACIO)]
        print(f"generadas {len(d['imagenes'])} imágenes en {time.time() - t0:.0f} s; grosor de la principal: " +
              ", ".join(f"{w} px {int((g == w).sum())}" for w in sorted(set(g.tolist()))))
        if a.publicar:
            print(f"→ publicado en {publicar_grueso(d)}")
        return 0
    if a.comprobar:
        return comprobar()
    p.print_help(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
