#!/usr/bin/env python3
"""Los datos de `rect-bor`: los MISMOS de `rect-lin`, leídos por su nombre y su huella. COPIA de su `datos.py`; aquí no
se publica nada (`--generar` sólo existe para ver la receta: el dato vale lo publicado, no lo regenerado).

Lo de abajo es el docstring original de rect-lin:

  ENTRENO  `rect-lin-entreno-r20261008`: 1000 rectas continuas (grosor 2–4), 1000 punteadas (discos de 2 px cada 3–6 px)
           y 1000 negativos (ruido · puntos sueltos · mancha). Cada bloque barajado: la curva de N toma los N primeros.
  BANCO    `rect-lin-banco-r20261008`: el banco de prueba fijo — rectas (ángulo cada 3°, grosor 2–14, largo 10/16/22),
           punteadas (separación 2–12), curvas (radio 6–40) y negativos. Nunca se entrena con él.

Convención de ángulo: x a la derecha, y hacia ABAJO; 0° = —, 45° = \\, 90° = |, 135° = /.

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
sys.path.insert(0, str(AQUI.parent.parent))
from expcnn import SUBDIR_DATASETS, exigir_dataset, exigir_datos  # noqa: E402

ENTRENO = "rect-lin-entreno-r20261008"
BANCO = "rect-lin-banco-r20261008"
LADO = 32
SEMILLA = 20261008
N_ENTRENO = 1000

# tipos de muestra (columna `tipo`)
CONTINUA, PUNTEADA, CURVA, NEG_RUIDO, NEG_PUNTOS, NEG_MANCHA = range(6)
NOMBRE_TIPO = ("continua", "punteada", "curva", "ruido", "puntos-sueltos", "mancha")
CAMPOS = ("imagenes", "tipo", "angulo", "grosor", "largo", "separacion", "radio")

_YY, _XX = np.mgrid[0:LADO, 0:LADO] + 0.5


def huella(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _u(ang: float) -> np.ndarray:
    t = np.deg2rad(ang)
    return np.array([np.cos(t), np.sin(t)])          # y hacia abajo: 45° baja hacia la derecha = «\»


def recta(cx: float, cy: float, largo: float, ang: float, grosor: float) -> np.ndarray:
    u = _u(ang); px, py = _XX - cx, _YY - cy
    a = px * u[0] + py * u[1]; p = -px * u[1] + py * u[0]
    return ((np.abs(a) <= largo / 2) & (np.abs(p) <= grosor / 2)).astype(np.uint8)


def discos(centros: np.ndarray, diametro: float = 2.0) -> np.ndarray:
    img = np.zeros((LADO, LADO), bool)
    for x, y in centros:
        img |= (_XX - x) ** 2 + (_YY - y) ** 2 <= (diametro / 2) ** 2
    return img.astype(np.uint8)


def punteada(cx: float, cy: float, largo: float, ang: float, sep: float) -> np.ndarray:
    u = _u(ang); n = int(np.floor(largo / sep)) + 1
    t = (np.arange(n) - (n - 1) / 2) * sep
    return discos(np.stack([cx + t * u[0], cy + t * u[1]], 1))


def arco(mx: float, my: float, radio: float, ang: float, largo: float, grosor: float) -> np.ndarray:
    """Arco de radio `radio` y longitud `largo`, con su punto medio en (mx, my) y tangente `ang` ahí."""
    u = _u(ang); n = np.array([-u[1], u[0]])
    c = np.array([mx, my]) + radio * n
    dx, dy = _XX - c[0], _YY - c[1]
    r = np.hypot(dx, dy)
    a_mid = np.arctan2(-n[1], -n[0])
    da = np.angle(np.exp(1j * (np.arctan2(dy, dx) - a_mid)))
    medio = min(largo / radio, 2 * np.pi) / 2
    return ((np.abs(r - radio) <= grosor / 2) & (np.abs(da) <= medio)).astype(np.uint8)


def negativo(rng: np.random.Generator, tipo: int) -> np.ndarray:
    if tipo == NEG_RUIDO:
        return (rng.random((LADO, LADO)) < 0.01).astype(np.uint8)
    if tipo == NEG_PUNTOS:
        return discos(rng.uniform(3, LADO - 3, size=(int(rng.integers(4, 11)), 2)))
    r = rng.uniform(2, 5); c = rng.uniform(10, 22, 2)
    return (((_XX - c[0]) ** 2 + (_YY - c[1]) ** 2) <= r * r).astype(np.uint8)


def _fila(img, tipo, angulo=np.nan, grosor=np.nan, largo=np.nan, sep=np.nan, radio=np.nan):
    return {"imagenes": img, "tipo": tipo, "angulo": angulo, "grosor": grosor, "largo": largo, "separacion": sep, "radio": radio}


def _apilar(filas: list[dict]) -> dict:
    d = {"imagenes": np.stack([f["imagenes"] for f in filas]).astype(np.uint8),
         "tipo": np.array([f["tipo"] for f in filas], np.int8)}
    for k in CAMPOS[2:]:
        d[k] = np.array([f[k] for f in filas], np.float32)
    return d


def generar_entreno() -> dict:
    filas = []
    rng = np.random.default_rng(SEMILLA)
    for _ in range(N_ENTRENO):
        c = rng.uniform(10, 22, 2); ang = rng.uniform(0, 180); lg = rng.uniform(14, 26); g = int(rng.integers(2, 5))
        filas.append(_fila(recta(c[0], c[1], lg, ang, g), CONTINUA, ang, g, lg))
    rng = np.random.default_rng(SEMILLA + 1)
    for _ in range(N_ENTRENO):
        c = rng.uniform(10, 22, 2); ang = rng.uniform(0, 180); lg = rng.uniform(14, 26); s = rng.uniform(3, 6)
        filas.append(_fila(punteada(c[0], c[1], lg, ang, s), PUNTEADA, ang, 2, lg, s))
    rng = np.random.default_rng(SEMILLA + 2)
    for i in range(N_ENTRENO):
        t = (NEG_RUIDO, NEG_PUNTOS, NEG_MANCHA)[i % 3]
        filas.append(_fila(negativo(rng, t), t))
    return _apilar(filas)


def generar_banco() -> dict:
    rng = np.random.default_rng(SEMILLA + 100)
    filas = []
    for ang in range(0, 180, 3):
        for g in (2, 3, 4, 6, 8, 10, 12, 14):
            for lg in (10, 16, 22):
                for _ in range(2):
                    c = rng.uniform(12, 20, 2)
                    filas.append(_fila(recta(c[0], c[1], lg, ang, g), CONTINUA, ang, g, lg))
        for s in (2, 3, 4, 6, 8, 10, 12):
            for _ in range(2):
                c = rng.uniform(12, 20, 2)
                filas.append(_fila(punteada(c[0], c[1], 22, ang, s), PUNTEADA, ang, 2, 22, s))
    for radio in (6, 9, 12, 18, 27, 40):
        for g in (2, 4, 8):
            for ang in range(0, 360, 45):
                for _ in range(3):
                    c = rng.uniform(14, 18, 2)
                    filas.append(_fila(arco(c[0], c[1], radio, ang, 20, g), CURVA, ang % 180, g, 20, np.nan, radio))
    for t in (NEG_RUIDO, NEG_PUNTOS, NEG_MANCHA):
        for _ in range(300):
            filas.append(_fila(negativo(rng, t), t))
    return _apilar(filas)


def publicar(nombre: str, d: dict, descripcion: str) -> Path:
    destino = exigir_datos() / SUBDIR_DATASETS / nombre
    if (destino / "manifiesto.json").exists():
        raise SystemExit(f"✗ {nombre} ya está publicado: un dataset no se reescribe nunca.")
    destino.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destino / "datos.npz", **d)
    man = {"nombre": nombre, "experimento": "rect-lin", "generado": time.strftime("%Y-%m-%d"),
           "origen": f"SINTÉTICO: nn/datos.py de rect-lin, semilla base {SEMILLA}", "lado": LADO, "valor_max": 1,
           "tipos": list(NOMBRE_TIPO), "n": int(len(d["imagenes"])), "huellas": {k: huella(v) for k, v in d.items()}}
    (destino / "manifiesto.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (destino / "README.md").write_text(f"# `{nombre}`\n\n{descripcion}\n\n**Quién lo usa:** `rect-lin` de `experimentos-cnn`.\n",
                                       encoding="utf-8")
    return destino


def cargar(nombre: str) -> dict:
    raiz = exigir_dataset(nombre)
    man = json.loads((raiz / "manifiesto.json").read_text(encoding="utf-8"))
    d = dict(np.load(raiz / "datos.npz"))
    for k, h in man["huellas"].items():
        if huella(d[k]) != h:
            raise RuntimeError(f"{nombre}: '{k}' no casa con su manifiesto. Me niego.")
    return d


def entreno(d: dict, modo: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Los N primeros positivos del modo (continua|punteada) + los N primeros negativos → (x (2N,1,32,32), y (2N,4))."""
    pos = np.flatnonzero(d["tipo"] == (CONTINUA if modo == "continua" else PUNTEADA))[:n]
    neg = np.flatnonzero(d["tipo"] >= NEG_RUIDO)[:n]
    x = d["imagenes"][np.concatenate([pos, neg])].astype(np.float32)[:, None]
    y = np.zeros((len(x), 4), np.float32)
    y[np.arange(n), orientacion(d["angulo"][pos])] = 1
    return x, y


def orientacion(ang: np.ndarray) -> np.ndarray:
    """El centro más cercano (0/45/90/135) → índice 0..3."""
    return (np.round(np.asarray(ang) / 45).astype(int)) % 4


def comprobar() -> int:
    ok = True
    for nombre in (ENTRENO, BANCO):
        d = cargar(nombre)
        cuenta = {NOMBRE_TIPO[t]: int((d["tipo"] == t).sum()) for t in range(6) if (d["tipo"] == t).any()}
        print(f"  [   ok] {nombre}: huellas · {len(d['imagenes'])} imágenes · {cuenta}")
    x, y = entreno(cargar(ENTRENO), "continua", 8)
    ok &= x.shape == (16, 1, 32, 32) and y[:8].sum() == 8 and y[8:].sum() == 0
    print(f"  [{'ok' if ok else 'FALLA':>5}] entreno(continua, 8) → {x.shape}, etiquetas {y[:8].argmax(1).tolist()}")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generar", action="store_true"); p.add_argument("--publicar", action="store_true")
    p.add_argument("--comprobar", action="store_true")
    a = p.parse_args()
    if a.generar:
        t0 = time.time(); e = generar_entreno(); b = generar_banco()
        print(f"entreno {len(e['imagenes'])} · banco {len(b['imagenes'])} imágenes en {time.time() - t0:.0f} s")
        if a.publicar:
            raise SystemExit("✗ rect-bor no publica: sus datasets son los de rect-lin.")
            print("→", publicar(ENTRENO, e, "Rectas de ángulo continuo a 32×32 para entrenar un kernel lineal: 1000 continuas "
                                         "(grosor 2–4, largo 14–26), 1000 punteadas (discos de 2 px cada 3–6 px) y 1000 "
                                         "negativos (ruido 1 % · 4–10 puntos sueltos · mancha de radio 2–5). Bloques "
                                         "barajados: la curva de N toma los N primeros."))
            print("→", publicar(BANCO, b, "Banco de PRUEBA fijo de rect-lin: rectas (ángulo cada 3°, grosor 2–14, largo "
                                        "10/16/22), punteadas (largo 22, separación 2–12), arcos (radio 6–40, largo 20, "
                                        "grosor 2/4/8) y 300 negativos de cada tipo."))
        return 0
    if a.comprobar:
        return comprobar()
    p.print_help(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
