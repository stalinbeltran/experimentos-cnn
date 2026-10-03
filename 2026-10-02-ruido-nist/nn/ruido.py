#!/usr/bin/env python3
"""Los nueve tipos de ruido de `ruido-nist`, DETERMINISTAS por generador, sobre las CUENTAS de
tinta (0..16) de un dígito de 8×8.

    python nn/ruido.py                    la tabla: tipo, índice t, niveles, cuál es el medio
    python nn/ruido.py --muestras [png]   la rejilla de 10 ejemplos por tipo (para MIRAR el ruido)

⚠ El dígito NO existe a 32×32 en este dato (scikit-learn sólo trae el 8×8 de cuentas por bloque
4×4), así que «dibujar a 32×32 y reducir contando bloques» (S3) se hace sobre lo único que se
puede dibujar: la MÁSCARA del trazo. Se rasteriza a 32×32, se reduce a 8×8 contando los bits
del bloque (cobertura c ∈ {0, 1/16, …, 1}) y se COMPONE a 8×8 con la transparencia (S4):

    x' = x + α·c·(v − x)        v = 1 (tinta) para lo que añade, v = 0 (papel) para lo que borra

Para `borrado` y `externos` se simula el nivel de bit SIN posiciones: dentro de un bloque,
apagar cada bit de tinta con probabilidad p quita Binomial(cuenta, p) bits, y encender cada bit
de fondo con probabilidad p añade Binomial(16 − cuenta, p). Es exactamente «dibujado a 32×32 y
contado», porque contar bits no depende de dónde estén. `gaussiano` y `sal-pimienta` van a 8×8
directamente, como dice el plan.

Un nivel 0 (α, p o σ = 0) deja la imagen INTACTA en los nueve tipos: es el «ruido nulo» con el
que `probar.py` comprueba que la cadena entera da los mismos pesos que `limpio`, bit a bit.
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path

import numpy as np

LADO = 8
BLOQUE = 4                 # 32×32 → 8×8 contando bloques 4×4 (lo que hizo NIST)
LADO32 = LADO * BLOQUE
BITS = BLOQUE * BLOQUE     # 16 bits por bloque: cuenta ∈ 0..16

# EL ORDEN ES UN CONTRATO: t = índice del tipo en la semilla de ruido 1000 + 10·t + i.
TIPOS = ("borrado", "externos", "horizontal", "vertical", "oblicua", "curva", "recorte",
         "gaussiano", "sal-pimienta")
ALFAS = (0.2, 0.4, 0.6, 0.8, 1.0)
# Cinco niveles por tipo, de menos a más, para que i ∈ 0..4 valga en todos (la semilla de ruido
# los indexa así) y «la intensidad media» sea el índice 2 en todos. Los del plan original con 3 ó
# 2 valores se completaron por los extremos SIN mover su medio (ESPECIFICACION.md §2).
NIVELES = {
    "borrado":      (0.05, 0.1, 0.2, 0.3, 0.4),      # p: cada bit de tinta se apaga con prob p
    "externos":     (0.01, 0.02, 0.05, 0.1, 0.2),    # p: cada bit de fondo se enciende con prob p
    "horizontal":   ALFAS, "vertical": ALFAS, "oblicua": ALFAS, "curva": ALFAS,
    "recorte":      ALFAS,                           # α: opacidad del trazo / del borrado
    "gaussiano":    (0.02, 0.05, 0.1, 0.2, 0.3),     # σ sobre x ∈ [0,1], recortado
    "sal-pimienta": (0.02, 0.05, 0.1, 0.2, 0.3),     # p: fracción de los 64 píxeles a 0 ó 1
}
PARAMETRO = {"borrado": "p", "externos": "p", "gaussiano": "σ", "sal-pimienta": "p"}
INDICE_MEDIO = 2
LIMPIO = "limpio"
SEMILLA_BASE = 1000
SALTO_REALIZACION = 5000   # la segunda copia de un escenario: semilla + 5000·(r − 1)
# Fase 3: `<tipo>@<nivel>-linea` = la MISMA semilla de ruido, pero una realización NUEVA por época
# (la de la época 1 es exactamente la copia fija; las siguientes salen del mismo generador).
SUFIJO_LINEA = "-linea"
# Fase 4: variantes del DIBUJO de los trazos (sólo rectas y curvas). El plan fijaba 1–2 px de grosor y
# 1–2 trazos; `-grueso` sube el grosor a 3–4 px y `-doble` el número de trazos a 3–4. Misma semilla.
VARIANTES = {"grueso": {"grosor": (3, 4), "trazos": (1, 2)}, "doble": {"grosor": (1, 2), "trazos": (3, 4)}}
TIPOS_TRAZO = ("horizontal", "vertical", "oblicua", "curva")
# Tamaño del dibujo (a 32 px): rectas de 1–2 px, 1–2 trazos; recorte de 8–16 px de lado.
GROSOR = (1, 2)
TRAZOS = (1, 2)
LADO_RECORTE = (8, 16)
_ESC = re.compile(r"^(?P<tipo>[a-z]+(?:-[a-z]+)?)(?:@(?P<nivel>\d+(?:\.\d+)?))?(?:-r(?P<r>\d+))?$")


def nombre_nivel(nivel: float) -> str:
    return f"{nivel:g}"


def escenario(tipo: str, nivel: float, realizacion: int = 1, linea: bool = False, variante: str | None = None) -> str:
    """El nombre canónico: `horizontal@0.6`, `oblicua@0.6-r2`, `recorte@0.6-linea`, `vertical@1-grueso-linea`, `limpio`."""
    if tipo == LIMPIO:
        return LIMPIO
    return (f"{tipo}@{nombre_nivel(nivel)}" + (f"-r{realizacion}" if realizacion != 1 else "")
            + (f"-{variante}" if variante else "") + (SUFIJO_LINEA if linea else ""))


def es_linea(nombre: str) -> bool:
    return nombre.endswith(SUFIJO_LINEA)


def fijo_de(nombre: str) -> str:
    """El gemelo de copia fija de un escenario en línea: 'recorte@0.6-linea' -> 'recorte@0.6'."""
    return nombre[:-len(SUFIJO_LINEA)] if es_linea(nombre) else nombre


def variante_de(nombre: str) -> str | None:
    """'vertical@1-grueso-linea' -> 'grueso'; sin variante -> None."""
    base = fijo_de(nombre)
    for v in VARIANTES:
        if base.endswith("-" + v):
            return v
    return None


def base_de(nombre: str) -> str:
    """El escenario SIN variante de dibujo, conservando -linea: 'vertical@1-grueso-linea' -> 'vertical@1-linea'."""
    v = variante_de(nombre)
    if v is None:
        return nombre
    return fijo_de(nombre)[:-len("-" + v)] + (SUFIJO_LINEA if es_linea(nombre) else "")


def parsear(nombre: str) -> tuple[str, float | None, int]:
    """'oblicua@0.6-r2' -> ('oblicua', 0.6, 2). Acepta el sufijo -linea (se pregunta aparte con es_linea).
    Se niega con un tipo o nivel que no esté en la tabla."""
    if es_linea(nombre):
        if fijo_de(nombre) == LIMPIO:
            raise ValueError("'limpio' no tiene versión en línea")
        nombre = fijo_de(nombre)
    v = variante_de(nombre)
    if v is not None:
        nombre = nombre[:-len("-" + v)]
        if nombre.split("@")[0] not in TIPOS_TRAZO:
            raise ValueError(f"la variante '{v}' sólo vale para los trazos {TIPOS_TRAZO}")
    m = _ESC.match(nombre)
    if not m:
        raise ValueError(f"escenario '{nombre}': la forma es <tipo>@<nivel>[-r<n>] o 'limpio'")
    tipo, nivel, r = m["tipo"], m["nivel"], int(m["r"] or 1)
    if tipo == LIMPIO:
        if nivel is not None or r != 1:
            raise ValueError("'limpio' no lleva nivel ni realización")
        return LIMPIO, None, 1
    if tipo not in TIPOS:
        raise ValueError(f"tipo '{tipo}' desconocido; los tipos son {TIPOS} o '{LIMPIO}'")
    if nivel is None:
        raise ValueError(f"'{nombre}': falta el nivel (@<valor>); los de {tipo} son {NIVELES[tipo]}")
    v = float(nivel)
    if not any(math.isclose(v, n) for n in NIVELES[tipo]):
        raise ValueError(f"'{nombre}': el nivel {v:g} no está en la tabla de {tipo}: {NIVELES[tipo]}")
    if r < 1:
        raise ValueError(f"'{nombre}': la realización empieza en 1")
    return tipo, v, r


def indice_nivel(tipo: str, nivel: float) -> int:
    for i, n in enumerate(NIVELES[tipo]):
        if math.isclose(n, nivel):
            return i
    raise ValueError(f"{tipo}: nivel {nivel:g} no está en {NIVELES[tipo]}")


def semilla(tipo: str, nivel: float, realizacion: int = 1) -> int:
    """1000 + 10·t + i, y +5000 por cada realización extra (ESPECIFICACION.md §1 bis)."""
    return SEMILLA_BASE + 10 * TIPOS.index(tipo) + indice_nivel(tipo, nivel) + SALTO_REALIZACION * (realizacion - 1)


def huella(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------- el dibujo a 32×32 → cobertura

def _lienzo():
    from PIL import Image, ImageDraw                     # noqa: PLC0415 (sólo quien dibuja)
    im = Image.new("L", (LADO32, LADO32), 0)
    return im, ImageDraw.Draw(im)


def cobertura(mascara32: np.ndarray) -> np.ndarray:
    """(N, 32, 32) 0/1 -> (N, 8, 8) float32 = bits cubiertos / 16 (contar bloques 4×4)."""
    m = (mascara32 > 0).astype(np.float32)
    return m.reshape(-1, LADO, BLOQUE, LADO, BLOQUE).mean(axis=(2, 4))


def _recta_h(d, rng, grosor=GROSOR):
    y = int(rng.integers(2, LADO32 - 2)); w = int(rng.integers(grosor[0], grosor[1] + 1))
    d.line([(0, y), (LADO32 - 1, y)], fill=1, width=w)


def _recta_v(d, rng, grosor=GROSOR):
    x = int(rng.integers(2, LADO32 - 2)); w = int(rng.integers(grosor[0], grosor[1] + 1))
    d.line([(x, 0), (x, LADO32 - 1)], fill=1, width=w)


def _recta_oblicua(d, rng, grosor=GROSOR):
    # ángulo uniforme en (15°, 75°) ∪ (105°, 165°): ni casi horizontal ni casi vertical
    ang = math.radians(float(rng.uniform(15, 75)) + 90.0 * int(rng.integers(0, 2)))
    cx, cy = (float(v) for v in rng.uniform(6, LADO32 - 6, size=2))
    w = int(rng.integers(grosor[0], grosor[1] + 1))
    dx, dy = 40 * math.cos(ang), 40 * math.sin(ang)
    d.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=1, width=w)


def _curva(d, rng, grosor=GROSOR):
    p = rng.uniform(2, LADO32 - 2, size=(3, 2))          # Bézier cuadrático con 3 puntos
    w = int(rng.integers(grosor[0], grosor[1] + 1))
    t = np.linspace(0, 1, 48)[:, None]
    pts = (1 - t) ** 2 * p[0] + 2 * (1 - t) * t * p[1] + t ** 2 * p[2]
    d.line([(float(a), float(b)) for a, b in pts], fill=1, width=w, joint="curve")


def _recorte(d, rng, grosor=None):
    lado = int(rng.integers(LADO_RECORTE[0], LADO_RECORTE[1] + 1))
    x0, y0 = (int(v) for v in rng.integers(0, LADO32 - lado + 1, size=2))
    d.rectangle([x0, y0, x0 + lado - 1, y0 + lado - 1], fill=1)


_TRAZO = {"horizontal": (_recta_h, 1.0), "vertical": (_recta_v, 1.0), "oblicua": (_recta_oblicua, 1.0),
          "curva": (_curva, 1.0), "recorte": (_recorte, 0.0)}      # (cómo se dibuja, v)


def mascaras(tipo: str, n: int, rng: np.random.Generator, variante: str | None = None) -> np.ndarray:
    """(n, 32, 32) uint8 0/1: la máscara de cada imagen, con sus parámetros sacados de rng en orden."""
    dibujar, _ = _TRAZO[tipo]
    if variante is not None and tipo not in TIPOS_TRAZO:
        raise ValueError(f"la variante '{variante}' sólo vale para los trazos {TIPOS_TRAZO}")
    grosor = VARIANTES[variante]["grosor"] if variante else GROSOR
    n_trazos = VARIANTES[variante]["trazos"] if variante else TRAZOS
    out = np.zeros((n, LADO32, LADO32), np.uint8)
    for k in range(n):
        im, d = _lienzo()
        trazos = 1 if tipo == "recorte" else int(rng.integers(n_trazos[0], n_trazos[1] + 1))
        for _ in range(trazos):
            dibujar(d, rng, grosor)
        out[k] = np.asarray(im, dtype=np.uint8)
    return out


# ---------------------------------------------------------------- aplicar un tipo a las cuentas

def componer(x: np.ndarray, c: np.ndarray, alfa: float, v: float) -> np.ndarray:
    """x' = x + α·c·(v − x): mezcla hacia v con opacidad α, ponderada por la cobertura c del píxel."""
    return (x + alfa * c * (v - x)).astype(np.float32)


def aplicar(tipo: str, nivel: float, cuentas: np.ndarray, rng: np.random.Generator, variante: str | None = None) -> np.ndarray:
    """(N, 8, 8) uint8 cuentas 0..16 -> (N, 8, 8) float32 en [0, 1], CON el ruido del tipo al nivel.
    Determinista dado rng. nivel = 0 devuelve exactamente cuentas/16 en los nueve tipos."""
    if tipo not in TIPOS:
        raise ValueError(f"tipo '{tipo}' desconocido: {TIPOS}")
    if variante is not None and (variante not in VARIANTES or tipo not in TIPOS_TRAZO):
        raise ValueError(f"variante '{variante}' no vale para {tipo}: {list(VARIANTES)} sólo en {TIPOS_TRAZO}")
    if not 0.0 <= nivel <= 1.0:
        raise ValueError(f"{tipo}: nivel {nivel} fuera de [0, 1]")
    if cuentas.dtype != np.uint8 or cuentas.ndim != 3 or cuentas.max() > BITS:
        raise ValueError("las cuentas son (N, 8, 8) uint8 en 0..16")
    x = cuentas.astype(np.float32) / BITS
    if tipo == "borrado":
        quitados = rng.binomial(cuentas.astype(np.int64), nivel)
        return ((cuentas.astype(np.int64) - quitados) / BITS).astype(np.float32)
    if tipo == "externos":
        anadidos = rng.binomial(BITS - cuentas.astype(np.int64), nivel)
        return ((cuentas.astype(np.int64) + anadidos) / BITS).astype(np.float32)
    if tipo == "gaussiano":
        return np.clip(x + rng.normal(0.0, 1.0, x.shape).astype(np.float32) * np.float32(nivel), 0, 1).astype(np.float32)
    if tipo == "sal-pimienta":
        k = int(round(nivel * LADO * LADO))
        out = x.copy()
        for i in range(len(out)):
            idx = rng.choice(LADO * LADO, size=k, replace=False)
            val = rng.integers(0, 2, size=k).astype(np.float32)
            out[i].reshape(-1)[idx] = val
        return out
    dibujar, v = _TRAZO[tipo]
    c = cobertura(mascaras(tipo, len(cuentas), rng, variante))
    return componer(x, c, nivel, v)


def copia(tipo: str, nivel: float, cuentas: np.ndarray, realizacion: int = 1, variante: str | None = None) -> tuple[np.ndarray, int]:
    """La copia ruidosa FIJA de un escenario: generador propio con su semilla. Devuelve (x', semilla)."""
    s = semilla(tipo, nivel, realizacion)
    return aplicar(tipo, nivel, cuentas, np.random.default_rng(s), variante), s


def tabla() -> list[dict]:
    return [{"t": t, "tipo": tipo, "parametro": PARAMETRO.get(tipo, "α"), "niveles": NIVELES[tipo],
             "medio": NIVELES[tipo][INDICE_MEDIO], "semilla_medio": semilla(tipo, NIVELES[tipo][INDICE_MEDIO]),
             "v": "papel (0)" if tipo in ("borrado", "recorte") else ("tinta (1)" if tipo in _TRAZO or tipo == "externos" else "—")}
            for t, tipo in enumerate(TIPOS)]


def muestras(cuentas: np.ndarray, destino: Path, n: int = 10) -> Path:
    """Rejilla PNG: fila = tipo (limpio + 9 al nivel medio), columna = las primeras n imágenes."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt   # noqa: PLC0415
    filas = [(LIMPIO, cuentas[:n].astype(np.float32) / BITS)]
    for tipo in TIPOS:
        nivel = NIVELES[tipo][INDICE_MEDIO]
        filas.append((escenario(tipo, nivel), copia(tipo, nivel, cuentas[:n])[0]))
    fig, axs = plt.subplots(len(filas), n, figsize=(n * 0.9, len(filas) * 0.95))
    for r, (nombre, xs) in enumerate(filas):
        for k in range(n):
            ax = axs[r, k]; ax.imshow(1 - xs[k], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                ax.set_ylabel(nombre, rotation=0, ha="right", va="center", fontsize=8)
    fig.suptitle("ruido-nist: las copias de train al nivel medio (tinta en negro)", fontsize=9)
    fig.tight_layout(); destino.parent.mkdir(parents=True, exist_ok=True); fig.savefig(destino, dpi=130); plt.close(fig)
    return destino


def main() -> int:
    if "--muestras" in sys.argv:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import datos                                                   # noqa: PLC0415
        d = datos.limpio()
        destino = Path(sys.argv[sys.argv.index("--muestras") + 1]) if len(sys.argv) > sys.argv.index("--muestras") + 1 \
            else Path(__file__).resolve().parent.parent / "resultados" / "muestras-ruido.png"
        print(f"→ {muestras(d['cuentas_train'], destino)}")
        return 0
    print(f"{'t':>2} {'tipo':<13} {'par':<3} {'niveles (i = 0..4)':<30} medio  semilla(medio)  v")
    for f in tabla():
        print(f"{f['t']:>2} {f['tipo']:<13} {f['parametro']:<3} {str(f['niveles']):<30} {f['medio']:<6g} {f['semilla_medio']:<14} {f['v']}")
    print(f"realización r: semilla + {SALTO_REALIZACION}·(r − 1) · '{LIMPIO}' no tiene semilla de ruido")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
