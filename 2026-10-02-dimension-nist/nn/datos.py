#!/usr/bin/env python3
"""El dato de `dim-nist`: los dígitos de 8×8 (NIST → UCI optdigits, la copia de scikit-learn),
publicados UNA vez en el repo de datos y reducidos a W por PROMEDIO DE ÁREA con pesos exactos.

    python nn/datos.py --publicar      construye el dataset desde scikit-learn y lo PUBLICA (una vez)
    python nn/datos.py --comprobar     huellas del manifiesto, huellas congeladas de cada W, piso

Qué es el dato: 1797 imágenes de 8×8 con valores enteros 0..16 (cada píxel es el número de bits
encendidos en un bloque 4×4 del bitmap original de 32×32 de NIST), 10 clases equilibradas.
Reparto DECLARADO en el manifiesto: 10 % train estratificado (18 por clase = 180) / 90 % val
(1617), semilla 0. El mismo train para todos los brazos.

Reducción 8 → W (W = 7, 6, 5, 4): cada píxel de salida es la media del área que cubre en la
imagen de 8. Los pesos son múltiplos EXACTOS de 1/8 (la rejilla de salida cae en múltiplos de
1/W), así que se calcula en enteros: S = A·x·Aᵀ con A entera (filas que suman 8) y
x = S / (8·8·16) ∈ [0, 1]. Exacto, con huella congelada, y 8→4 es la media de bloques 2×2.
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
sys.path.insert(0, str(EXP.parent))                       # el repo: donde vive `expcnn`
from expcnn import SUBDIR_DATASETS, exigir_dataset, exigir_datos  # noqa: E402

DATASET = "uci-optdigits-8px-r20261002"   # r<fecha> = fecha de EXTRACCION de scikit-learn, no de render
LADO = 8
VALOR_MAX = 16          # bits encendidos en un bloque 4x4
W_TODOS = (8, 7, 6, 5, 4)
POR_CLASE_TRAIN = 18    # 18 x 10 clases = 180 = 10 % de 1797
SEMILLA_REPARTO = 0
DIVISOR = LADO * LADO * VALOR_MAX   # 1024: S/DIVISOR es la fraccion de tinta en [0,1]

# Huellas CONGELADAS el 2026-10-02 (sha256 de la matriz int64 S = A·x·Aᵀ, (1797, W, W), orden
# del dataset publicado, bytes little-endian, 16 hex). `cargar` SE NIEGA si no casan.
HUELLAS = {
    8: "ee9218e7323cc76a",
    7: "6b9f3ef6b1ef852a",
    6: "91ee3970a6036d50",
    5: "3bdbbc60af71beec",
    4: "39483d316ebda8eb",
}
HUELLA_IMAGENES = "8f26b2bd9d135c25"     # uint8 (1797, 8, 8), tal como las da scikit-learn
HUELLA_ETIQUETAS = "8ba4f891220f5e4c"    # uint8 (1797,)


def _huella(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _huella_i8(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).astype("<i8").tobytes()).hexdigest()[:16]


def matriz(W: int) -> np.ndarray:
    """A (W, 8) entera: A[i, j]·(1/8) es el peso del pixel j de entrada en el pixel i de salida.
    Cada fila suma 8 (= promedio). Para W = 4 es la media de bloques 2x2."""
    if not 1 <= W <= LADO:
        raise ValueError(f"W={W} fuera de [1, {LADO}]")
    A = np.zeros((W, LADO), np.int64)
    for i in range(W):
        lo, hi = i * LADO / W, (i + 1) * LADO / W
        for j in range(LADO):
            ov = max(0.0, min(hi, j + 1) - max(lo, j))     # multiplo de 1/W
            A[i, j] = int(round(ov * W))
    assert (A.sum(1) == LADO).all(), A
    return A


def sumas(x: np.ndarray, W: int) -> np.ndarray:
    """(N, 8, 8) uint8 -> (N, W, W) int64 = A·x·Aᵀ. Exacto; max 1024."""
    A = matriz(W)
    return np.einsum("ij,njk,lk->nil", A, x.astype(np.int64), A)


def fraccion(S: np.ndarray) -> np.ndarray:
    """(N, W, W) int64 -> (N, 1, W, W) float32: fraccion de TINTA en [0, 1] (0 fondo, 1 tinta)."""
    return (S.astype(np.float64) / DIVISOR).astype(np.float32)[:, None]


def control_de4(x4: np.ndarray) -> np.ndarray:
    """(N, 1, 4, 4) -> (N, 1, 8, 8): cada pixel de 4 es un bloque uniforme de 2x2."""
    assert x4.shape[-2:] == (4, 4), x4.shape
    return np.repeat(np.repeat(x4, 2, axis=-2), 2, axis=-1)


def _reparto(y: np.ndarray) -> np.ndarray:
    """'train' para 18 de cada clase (semilla 0), 'val' para el resto."""
    rng = np.random.default_rng(SEMILLA_REPARTO)
    tr = np.zeros(len(y), bool)
    for c in range(10):
        idx = np.flatnonzero(y == c)
        tr[rng.choice(idx, POR_CLASE_TRAIN, replace=False)] = True
    return np.where(tr, "train", "val")


def publicar() -> int:
    """Construye el dataset desde scikit-learn y lo publica. SE NIEGA a pisar uno existente."""
    raiz = exigir_datos() / SUBDIR_DATASETS / DATASET
    if raiz.exists():
        print(f"✗ {raiz} ya existe: un dataset publicado no se reescribe (dato nuevo = nombre nuevo)")
        return 2
    import sklearn                                           # noqa: PLC0415
    from sklearn.datasets import load_digits                 # noqa: PLC0415
    d = load_digits()
    x = d.images.astype(np.uint8)
    y = d.target.astype(np.uint8)
    assert x.shape == (1797, LADO, LADO) and x.max() == VALOR_MAX, (x.shape, x.max())
    part = _reparto(y)
    raiz.mkdir(parents=True)
    np.savez_compressed(raiz / "datos.npz", imagenes=x, etiquetas=y, particion=part)
    man = {
        "nombre": DATASET,
        "experimento": "dim-nist",
        "generado": "2026-10-02",
        "origen": "scikit-learn `load_digits` (copia del TEST set de UCI Optical Recognition of "
                  "Handwritten Digits, E. Alpaydin, 1998): bitmaps de 32x32 extraidos con los "
                  "programas de preprocesado de NIST, divididos en bloques 4x4 y contados los bits "
                  "encendidos -> 8x8 con valores 0..16",
        "sklearn": sklearn.__version__,
        "licencia": "UCI ML Repository (la pagina de UCI lo publica como CC BY 4.0: NO comprobado desde esta maquina); atribucion: E. Alpaydin y C. Kaynak, 1998; viaja empaquetado con scikit-learn (BSD-3)",
        "escritores": "las 1797 imagenes son el TEST set de UCI, escrito por 13 personas: cualquier reparto deja a los mismos escritores en train y en val",
        "lado": LADO, "valor_max": VALOR_MAX, "clases": 10, "n": int(len(y)),
        "por_clase": {str(c): int((y == c).sum()) for c in range(10)},
        "reparto": {"regla": f"{POR_CLASE_TRAIN} por clase a train (10 %), estratificado, semilla "
                             f"{SEMILLA_REPARTO} (numpy default_rng); el resto a val",
                    "train": int((part == "train").sum()), "val": int((part == "val").sum())},
        "huellas": {"imagenes": _huella(x), "etiquetas": _huella(y), "particion": _huella(part)},
        "reducciones": {str(W): {"huella_S": _huella_i8(sumas(x, W)),
                                 "matriz_A_fila0": matriz(W)[0].tolist()} for W in W_TODOS},
        "nota": "8x8 YA es una reduccion /4 del bitmap de NIST (conteo de bits por bloque 4x4): "
                "el 'original' de este experimento es ese 8x8; el 32x32 no esta en scikit-learn",
    }
    (raiz / "manifiesto.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    (raiz / "README.md").write_text(f"""# `{DATASET}`

Los **dígitos manuscritos de 8×8** (UCI *Optical Recognition of Handwritten Digits*, la copia
que empaqueta scikit-learn como `load_digits`, 1797 imágenes), publicados aquí para `dim-nist` de `experimentos-cnn`. Cada píxel es el **número de bits encendidos en un bloque
4×4 del bitmap de 32×32** que extrajeron los programas de preprocesado de NIST (valores
0..16). O sea que **8×8 ya es una reducción /4**; el 32×32 no viene con scikit-learn.

| | |
|---|---|
| `datos.npz` | `imagenes` (1797, 8, 8) uint8 0..16 · `etiquetas` (1797,) uint8 0..9 · `particion` ('train'/'val') |
| reparto | **10 % train estratificado: 18 por clase = 180** · val 1617 · semilla {SEMILLA_REPARTO} |
| clases | equilibradas (174–183 por clase) |
| licencia | UCI ML Repository (CC BY 4.0 según su página, no comprobado desde aquí); atribución Alpaydin y Kaynak 1998; scikit-learn {sklearn.__version__} |
| escritores | **13 personas** (es el *test set* de UCI): train y val comparten escritores |
| `r20261002` | fecha de **extracción** desde scikit-learn, no de render |

Huellas (sha256, 16 hex): imágenes `{man['huellas']['imagenes']}` · etiquetas
`{man['huellas']['etiquetas']}` · partición `{man['huellas']['particion']}`. Se comprueban con
`python nn/datos.py --comprobar` desde la carpeta del experimento.

**Quién lo usa:** `dim-nist`. Compartir el dataset no es compartir condiciones.
""", encoding="utf-8")
    print(f"publicado en {raiz}: {len(y)} imagenes, train {man['reparto']['train']} / val {man['reparto']['val']}")
    return 0


def crudo() -> dict:
    raiz = exigir_dataset(DATASET)
    man = json.loads((raiz / "manifiesto.json").read_text(encoding="utf-8"))
    d = np.load(raiz / "datos.npz")
    x, y, part = d["imagenes"], d["etiquetas"], d["particion"]
    for nombre, a in (("imagenes", x), ("etiquetas", y), ("particion", part)):
        if _huella(a) != man["huellas"][nombre]:
            raise RuntimeError(f"{DATASET}/datos.npz: '{nombre}' da {_huella(a)} y el manifiesto "
                               f"dice {man['huellas'][nombre]}: no es el dato publicado")
    return {"imagenes": x, "etiquetas": y, "particion": part, "manifiesto": man}


def piso(y_val: np.ndarray) -> float:
    """El piso es ADIVINAR AL AZAR entre 10 clases: 1/10. «La clase mayoritaria de train» no existe
    aqui (18 por clase, empatan), y la de val seria 0,102: se fija 0,1 por honestidad."""
    return 0.1


def cargar(W: int, control: bool = False) -> dict:
    if W not in W_TODOS:
        raise ValueError(f"W={W} no esta en {W_TODOS}")
    if control and W != 8:
        raise ValueError("el control es w8-de4: W tiene que ser 8")
    d = crudo()
    w_dato = 4 if control else W
    S = sumas(d["imagenes"], w_dato)
    h = _huella_i8(S)
    if h != HUELLAS[w_dato]:
        raise RuntimeError(f"la reduccion a W={w_dato} da {h} y la huella congelada es "
                           f"{HUELLAS[w_dato]}: NO es el mismo dato reducido. Me niego.")
    x = fraccion(S)
    if control:
        x = control_de4(x)
    y = d["etiquetas"].astype(np.int64)
    tr = d["particion"] == "train"
    va = ~tr
    assert tr.sum() == 180 and va.sum() == 1617, (tr.sum(), va.sum())
    return {"W": W, "control": control, "dataset": DATASET,
            "x_train": x[tr], "y_train": y[tr], "x_val": x[va], "y_val": y[va],
            "huella_x": h, "huella_y": _huella(d["etiquetas"])}


def comprobar() -> int:
    ok = True

    def linea(que, bien, det=""):
        nonlocal ok
        ok &= bool(bien)
        print(f"  [{'ok' if bien else 'FALLA':>5}] {que}" + (f"  {det}" if det else ""))

    d = crudo()
    x, y = d["imagenes"], d["etiquetas"]
    print(f"dataset {DATASET}: {len(y)} imagenes, manifiesto casado")
    linea("imagenes uint8 0..16 y etiquetas 0..9", x.dtype == np.uint8 and x.max() == VALOR_MAX and y.max() == 9)
    linea("huella imagenes / etiquetas congeladas",
          _huella(x) == HUELLA_IMAGENES and _huella(y) == HUELLA_ETIQUETAS)
    for W in W_TODOS:
        S = sumas(x, W)
        linea(f"W={W}: huella congelada", _huella_i8(S) == HUELLAS[W], f"{_huella_i8(S)} (max {S.max()}, A fila0 {matriz(W)[0].tolist()})")
    linea("W=8 es exactamente la imagen (S/1024 == x/16)",
          np.allclose(fraccion(sumas(x, 8))[:, 0], x / 16.0))
    # el promedio de area conserva la MEDIA: suma(S) = W²·suma(x) (cada columna de A suma W)
    linea("media conservada en cada W (suma de S = W²·suma de x)",
          all(int(sumas(x, W).sum()) == W * W * int(x.astype(np.int64).sum()) for W in W_TODOS))
    x4 = fraccion(sumas(x, 4))
    ctl = control_de4(x4)
    linea("control w8-de4: (N,1,8,8) y bloques 2x2 uniformes",
          ctl.shape == (len(x), 1, 8, 8) and bool((ctl[:, :, 1::2, 1::2] == x4).all()))
    tr = d["particion"] == "train"
    linea("reparto 180 / 1617, 18 por clase en train", tr.sum() == 180 and all((y[tr] == c).sum() == 18 for c in range(10)))
    linea("piso = 1/10 (azar entre 10 clases; clase mayoritaria de val 0,102, de train no existe)",
          abs(piso(y[~tr]) - 0.1) < 1e-9 and abs(np.bincount(y[~tr]).max() / (~tr).sum() - 0.102) < 0.001)
    print("\nel dato esta en orden." if ok else "\n✗ ALGO NO CASA: no se entrena sobre esto.")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--publicar", action="store_true")
    p.add_argument("--comprobar", action="store_true")
    a = p.parse_args()
    if a.publicar:
        return publicar()
    if a.comprobar:
        return comprobar()
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
