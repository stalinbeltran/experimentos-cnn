#!/usr/bin/env python3
"""El dato de `feat-ind`: el dataset sintético `feat-ind-sinteticas-8px-r20261003`, generado UNA vez
aquí y publicado en el repo de datos; después se lee por su nombre y no se regenera.

    python nn/datos.py --generar [--publicar]   genera (determinista) y, con --publicar, lo publica.
                                               Se NIEGA a pisar uno publicado.
    python nn/datos.py --comprobar              las huellas del publicado casan con su manifiesto
    python nn/datos.py --rederivar              la receta lo vuelve a dar, bit a bit (prueba, no sustituto)

Training set de un detector: ver `conjunto(f, particion, contra)`.
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
import features as F                                               # noqa: E402

DATASET = "feat-ind-sinteticas-8px-r20261003"
# ⚠ El primero publicado (sin la «b») NO se usa: rectas y esquinas cortas con 12 px salían manchas
# cuadradas, o sea ejemplos mal etiquetados. Se conserva publicado con su aviso, que es la regla; su receta
# ya no se puede re-derivar desde este código (cambiaron los topes de grosor).
DATASET_GRUESO_DESCARTADO = "feat-ind-sinteticas-grueso-8px-r20261003"
# ⚠ El «b» tampoco se usa: acotaba el grosor por el largo y casi ninguna recta llegaba a engordar
# (recta-V 1,20 celdas de ancho medio, 0 % ≥ 2,5; un «1» manuscrito mide 2,9). El «c» sortea primero el
# grosor y alarga el trazo para que siga siendo un trazo.
# ⚠ El «c» tampoco: alargaba la recta hasta 26 px, que con 12 px de grosor NO CABE en el lienzo; el sorteo
# se repetía hasta salir una fina (recta-V 1,27 celdas). El «d» acota el largo para que quepa.
DATASET_GRUESO = "feat-ind-sinteticas-grueso-8px-r20261003d"
PERFIL_DE = {DATASET: "fino", DATASET_GRUESO: "grueso"}
# el grueso usa OTRAS semillas: si no, sus primeras imágenes repetirían las del fino con otro grosor
SALTO_SEMILLA = {DATASET: 0, DATASET_GRUESO: 1000}
DIGITOS = "uci-optdigits-8px-r20261002"
N_TRAIN, N_VAL = 2000, 400
N_TRAIN_VACIO, N_VAL_VACIO = 1000, 200
SEMILLA_BASE = 20261003
CAMPOS = ("imagenes", "mascara8", "principal", "secundaria", "ancla", "ancla32", "grosor", "radio",
          "angulo", "apertura", "largo", "particion")


def huella(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def semilla(familia: str, particion: str, nombre: str = DATASET) -> int:
    return SEMILLA_BASE + SALTO_SEMILLA[nombre] + 10 * F.FAMILIAS.index(familia) + (0 if particion == "train" else 1)


def generar(nombre: str = DATASET) -> dict:
    F.usar_perfil(PERFIL_DE[nombre])
    filas = {k: [] for k in CAMPOS}
    for fam in F.FAMILIAS:
        for part, n in (("train", N_TRAIN_VACIO if fam == F.VACIO else N_TRAIN),
                        ("val", N_VAL_VACIO if fam == F.VACIO else N_VAL)):
            rng = np.random.default_rng(semilla(fam, part, nombre))
            for _ in range(n):
                s = F.muestra(rng, fam)
                filas["imagenes"].append(s["imagen"]); filas["mascara8"].append(s["mascara8"])
                filas["principal"].append(s["principal"]); filas["secundaria"].append(s["secundaria"])
                filas["ancla"].append(s["ancla"]); filas["ancla32"].append(s["ancla32"])
                for k in ("grosor", "radio", "angulo", "apertura", "largo"):
                    filas[k].append(s[k])
                filas["particion"].append(part)
    return {"imagenes": np.stack(filas["imagenes"]).astype(np.uint8), "mascara8": np.stack(filas["mascara8"]).astype(np.uint8),
            "principal": np.array(filas["principal"], np.int8), "secundaria": np.array(filas["secundaria"], np.int8),
            "ancla": np.stack(filas["ancla"]).astype(np.int8), "ancla32": np.stack(filas["ancla32"]).astype(np.float32),
            "grosor": np.array(filas["grosor"], np.int8), "radio": np.array(filas["radio"], np.float32),
            "angulo": np.array(filas["angulo"], np.float32), "apertura": np.array(filas["apertura"], np.float32),
            "largo": np.array(filas["largo"], np.float32), "particion": np.array(filas["particion"])}


def manifiesto(d: dict, nombre: str = DATASET) -> dict:
    por = {f: {"train": int(((d["principal"] == i) & (d["particion"] == "train")).sum()),
               "val": int(((d["principal"] == i) & (d["particion"] == "val")).sum())} for i, f in enumerate(F.FAMILIAS)}
    return {"nombre": nombre, "experimento": "feat-ind", "perfil": PERFIL_DE[nombre],
            "grosores_px": list(F.PERFILES[PERFIL_DE[nombre]]), "generado": time.strftime("%Y-%m-%d"),
            "origen": "SINTÉTICO: cada feature dibujada como máscara de 32x32 (PIL) y reducida a 8x8 contando bits por "
                      "bloque 4x4 (0..16), como NIST. Especificación: ESPECIFICACION.md del experimento",
            "familias": list(F.FAMILIAS), "lado": F.LADO, "valor_max": F.BITS, "n": int(len(d["imagenes"])),
            "por_familia": por, "semillas": {f: {"train": semilla(f, "train", nombre), "val": semilla(f, "val", nombre)} for f in F.FAMILIAS},
            "secundaria_p": F.P_SECUNDARIA, "ruido": {"p_on": F.P_ON, "p_off": F.P_OFF},
            "campos": {"imagenes": "(N,8,8) uint8 0..16, CON ruido y con la secundaria", "mascara8": "(N,8,8) uint8: la principal sola y limpia",
                       "principal": "índice en familias", "secundaria": "índice o -1", "ancla": "(fila, col) del 8x8; -1 en vacio",
                       "ancla32": "(fila, col) en 32x32", "radio/grosor/angulo/apertura/largo": "-1 donde no aplica"},
            "huellas": {k: huella(d[k]) for k in CAMPOS}}


def publicar(d: dict, nombre: str = DATASET) -> Path:
    destino = exigir_datos() / SUBDIR_DATASETS / nombre
    if (destino / "manifiesto.json").exists():
        raise SystemExit(f"✗ {nombre} ya está publicado: un dataset no se reescribe nunca. Dato nuevo = nombre nuevo.")
    destino.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destino / "datos.npz", **d)
    man = manifiesto(d, nombre)
    (destino / "manifiesto.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    filas = "\n".join(f"| `{f}` | {v['train']} | {v['val']} |" for f, v in man["por_familia"].items())
    (destino / "README.md").write_text(
        f"# `{nombre}`\n\nPerfil de grosor `{PERFIL_DE[nombre]}` ({', '.join(map(str, F.PERFILES[PERFIL_DE[nombre]]))} px de 32). Features **sintéticas** de 8×8 para `feat-ind` de `experimentos-cnn`: 13 familias (arcos por "
        f"dirección del centro, rectas, lazo, esquinas) y `vacio`, dibujadas a 32×32 y reducidas contando bits por bloque "
        f"4×4 (0..16), como los dígitos de `{DIGITOS}`. La mitad lleva una segunda feature de otra familia; todas llevan "
        f"ruido leve. La especificación (parámetros, anclas, porqués) está en `ESPECIFICACION.md` del experimento.\n\n"
        f"| familia | train | val |\n|---|---|---|\n{filas}\n\n"
        f"`datos.npz`: {', '.join(CAMPOS)}. Huellas en `manifiesto.json`; `python nn/datos.py --comprobar` desde el experimento.\n\n"
        f"**Quién lo usa:** `feat-ind`. Compartir el dataset no es compartir condiciones.\n", encoding="utf-8")
    return destino


def cargar(nombre: str = DATASET) -> dict:
    raiz = exigir_dataset(nombre)
    man = json.loads((raiz / "manifiesto.json").read_text(encoding="utf-8"))
    d = dict(np.load(raiz / "datos.npz"))
    for k in CAMPOS:
        h = huella(d[k])
        if h != man["huellas"][k]:
            raise RuntimeError(f"{nombre}/datos.npz: '{k}' da {h} y el manifiesto dice {man['huellas'][k]}: no es el dato publicado. Me niego.")
    if list(man["familias"]) != list(F.FAMILIAS):
        raise RuntimeError(f"el dataset publicado tiene las familias {man['familias']} y el código {F.FAMILIAS}: no casan. Me niego.")
    d["manifiesto"] = man; d["nombre"] = nombre
    return d


def conjunto(d: dict, familia: str, particion: str, contra: list[str] | None = None) -> dict:
    """Positivos: principal == f. Negativos: principal != f y secundaria != f, y (enmienda de la corrida 2)
    que ni la principal ni la secundaria sea una familia que CONTIENE a f; restringidos a `contra` si se da
    (`contra` manda: si nombra una contenedora, entra). x en [0, 1] (cuentas/16), (N, 1, 8, 8) float32."""
    i = F.FAMILIAS.index(familia)
    part = d["particion"] == particion
    pos = part & (d["principal"] == i)
    neg = part & (d["principal"] != i) & (d["secundaria"] != i)
    if contra:
        idx = [F.FAMILIAS.index(c) for c in contra]
        neg &= np.isin(d["principal"], idx)
    else:
        cont = [F.FAMILIAS.index(c) for c in F.contenedoras(familia)]
        if cont:
            neg &= ~np.isin(d["principal"], cont) & ~np.isin(d["secundaria"], cont)
    x = (d["imagenes"].astype(np.float32) / F.BITS)[:, None]
    return {"x_pos": x[pos], "ancla_pos": d["ancla"][pos].astype(np.int64), "radio_pos": d["radio"][pos], "grosor_pos": d["grosor"][pos],
            "x_neg": x[neg], "familia_neg": d["principal"][neg], "secundaria_pos": d["secundaria"][pos]}


def digitos() -> dict:
    """Los 1797 dígitos publicados, en el MISMO espacio (cuentas/16), con su reparto 180/1617."""
    raiz = exigir_dataset(DIGITOS)
    man = json.loads((raiz / "manifiesto.json").read_text(encoding="utf-8"))
    z = np.load(raiz / "datos.npz")
    x, y, part = z["imagenes"], z["etiquetas"], z["particion"]
    for k, a in (("imagenes", x), ("etiquetas", y), ("particion", part)):
        if huella(a) != man["huellas"][k]:
            raise RuntimeError(f"{DIGITOS}: '{k}' no casa con su manifiesto. Me niego.")
    return {"x": (x.astype(np.float32) / F.BITS)[:, None], "y": y.astype(np.int64), "train": part == "train"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generar", action="store_true"); p.add_argument("--publicar", action="store_true")
    p.add_argument("--comprobar", action="store_true"); p.add_argument("--rederivar", action="store_true")
    p.add_argument("--dataset", default=DATASET, choices=list(PERFIL_DE), help="cuál (por defecto el fino original)")
    a = p.parse_args()
    if a.generar:
        t0 = time.time(); d = generar(a.dataset)
        print(f"generadas {len(d['imagenes'])} imágenes en {time.time() - t0:.0f} s; huella imagenes {huella(d['imagenes'])}")
        if a.publicar:
            print(f"→ publicado en {publicar(d, a.dataset)}")
        return 0
    if a.comprobar:
        d = cargar(a.dataset)
        print(f"{a.dataset}: {len(d['imagenes'])} imágenes, huellas ok; familias {F.FAMILIAS}")
        return 0
    if a.rederivar:
        d = cargar(a.dataset); g = generar(a.dataset)
        malos = [k for k in CAMPOS if huella(d[k]) != huella(g[k])]
        print("la receta vuelve a dar el publicado, bit a bit" if not malos else f"✗ difieren: {malos}")
        return 0 if not malos else 1
    p.print_help(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
