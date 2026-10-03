#!/usr/bin/env python3
"""El VOCABULARIO de `feat-ind` y su rasterizador: cada feature se dibuja como máscara de 32×32 y se
reduce a 8×8 contando los bits de cada bloque 4×4 (cuentas 0..16), que es lo que hizo NIST con los
dígitos. La especificación entera, con el porqué de cada número, está en ../ESPECIFICACION.md.

    python nn/features.py              la tabla del vocabulario
    python nn/features.py --muestras   nn/../resultados/muestras-features.png (una fila por familia)

Idea del dibujo copiada de `ruido-nist` (dibujar a 32 y contar bloques); el código es de aquí.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

LADO = 8
BLOQUE = 4
LADO32 = LADO * BLOQUE
BITS = BLOQUE * BLOQUE
MARGEN = 1.5                     # ningún punto del trazo a menos de esto del borde (px de 32)
GROSORES = (2, 3, 4)
P_SECUNDARIA = 0.5
P_ON, P_OFF = (0.0, 0.02), (0.0, 0.10)

# EL ORDEN ES UN CONTRATO: el índice es lo que guarda el dataset en `principal`/`secundaria`.
FAMILIAS = ("arco-E", "arco-W", "arco-N", "arco-S",
            "recta-V", "recta-H", "recta-S", "recta-B",
            "lazo",
            "esquina-NE", "esquina-NW", "esquina-SE", "esquina-SW",
            "vacio")
GRUPOS = {"arcos": FAMILIAS[0:4], "rectas": FAMILIAS[4:8], "lazos": FAMILIAS[8:9], "esquinas": FAMILIAS[9:13]}
VACIO = "vacio"
CON_TRAZO = tuple(f for f in FAMILIAS if f != VACIO)

# direcciones en coordenadas de imagen (x a la derecha, y hacia ABAJO): E=0°, S=90°, W=180°, N=270°
DIR = {"E": 0.0, "S": 90.0, "W": 180.0, "N": 270.0}
ARCO_RADIO = (4.0, 14.0)
ARCO_APERTURA = (100.0, 200.0)
ARCO_JITTER = 40.0               # ±: deja 10° de hueco entre clases vecinas
RADIO_TRAMOS = (("chico", 4.0, 7.0), ("medio", 7.0, 10.0), ("grande", 10.0, 14.01))
RECTA_LARGO = (10.0, 28.0)
RECTA_ANGULO = {"recta-V": 90.0, "recta-H": 0.0, "recta-S": -45.0, "recta-B": 45.0}
RECTA_JITTER = 12.0
LAZO_A = (4.0, 11.0)
LAZO_RAZON = (0.7, 1.4)
ESQ_BRAZO = (8.0, 16.0)
ESQ_ANGULO = (65.0, 115.0)       # 90 ± 25 entre brazos
ESQ_JITTER = 20.0                # ± rotación del conjunto
ESQ_BRAZOS = {"esquina-NE": ("N", "E"), "esquina-NW": ("N", "W"), "esquina-SE": ("S", "E"), "esquina-SW": ("S", "W")}


# ENMIENDA 2026-10-03 (corrida 2): qué familia CONTIENE a otra como parte. Un negativo de `f` no puede
# ser una imagen que lleve `f` dentro: un lazo lleva los cuatro arcos, y los dos brazos de una esquina
# son una recta vertical y una horizontal. En la corrida 1 eran negativos y el detector «fallaba» al
# reconocerlos — o sea se le pedía DISCRIMINAR, que es justo lo que la premisa prohíbe.
CONTIENE = {"lazo": ("arco-E", "arco-W", "arco-N", "arco-S"),
            "esquina-NE": ("recta-V", "recta-H"), "esquina-NW": ("recta-V", "recta-H"),
            "esquina-SE": ("recta-V", "recta-H"), "esquina-SW": ("recta-V", "recta-H")}


def contenedoras(familia: str) -> tuple[str, ...]:
    """Las familias que llevan `familia` como parte (y por tanto no valen de negativo)."""
    return tuple(g for g, partes in CONTIENE.items() if familia in partes)


def tramo_radio(r: float) -> str:
    for nombre, lo, hi in RADIO_TRAMOS:
        if lo <= r < hi:
            return nombre
    return "fuera"


def _u(grados: float) -> np.ndarray:
    a = math.radians(grados)
    return np.array([math.cos(a), math.sin(a)])


def _dentro(pts: np.ndarray, w: int) -> bool:
    m = MARGEN + w / 2
    return bool((pts >= m).all() and (pts <= LADO32 - 1 - m).all())


# ---------------------------------------------------------------- las curvas, como polilíneas (x, y)

def _arco(rng, familia: str) -> dict | None:
    """Arco de circunferencia cuyo CENTRO queda en la dirección de la familia (±jitter) visto desde el
    punto medio del arco. Ancla = punto medio (sobre la tinta)."""
    r = float(rng.uniform(*ARCO_RADIO))
    ap = float(rng.uniform(*ARCO_APERTURA))
    theta = DIR[familia[-1]] + float(rng.uniform(-ARCO_JITTER, ARCO_JITTER))
    w = int(rng.choice(GROSORES))
    for _ in range(60):
        c = rng.uniform(0, LADO32, size=2)
        a = np.radians(theta + 180.0 + np.linspace(-ap / 2, ap / 2, 64))
        pts = c + r * np.stack([np.cos(a), np.sin(a)], 1)
        if _dentro(pts, w):
            return {"pts": [pts], "cerrado": False, "ancla": c - r * _u(theta), "grosor": w,
                    "radio": r, "angulo": theta, "apertura": ap, "largo": -1.0}
    return None


def _recta(rng, familia: str) -> dict | None:
    L = float(rng.uniform(*RECTA_LARGO))
    ang = RECTA_ANGULO[familia] + float(rng.uniform(-RECTA_JITTER, RECTA_JITTER))
    w = int(rng.choice(GROSORES))
    for _ in range(60):
        c = rng.uniform(0, LADO32, size=2)
        pts = np.stack([c - L / 2 * _u(ang), c + L / 2 * _u(ang)])
        if _dentro(pts, w):
            return {"pts": [pts], "cerrado": False, "ancla": c, "grosor": w,
                    "radio": -1.0, "angulo": ang, "apertura": -1.0, "largo": L}
    return None


def _lazo(rng, familia: str) -> dict | None:
    a_ = float(rng.uniform(*LAZO_A))
    razon = float(rng.uniform(*LAZO_RAZON))
    b_ = float(np.clip(a_ * razon, 3.5, 12.0))
    rot = float(rng.uniform(0, 180))
    w = int(rng.choice(GROSORES))
    t = np.linspace(0, 2 * math.pi, 72, endpoint=False)
    base = np.stack([a_ * np.cos(t), b_ * np.sin(t)], 1)
    R = np.array([[math.cos(math.radians(rot)), -math.sin(math.radians(rot))],
                  [math.sin(math.radians(rot)), math.cos(math.radians(rot))]])
    for _ in range(60):
        c = rng.uniform(0, LADO32, size=2)
        pts = c + base @ R.T
        if _dentro(pts, w):
            return {"pts": [pts], "cerrado": True, "ancla": c, "grosor": w,
                    "radio": a_, "angulo": rot, "apertura": razon, "largo": -1.0}
    return None


def _esquina(rng, familia: str) -> dict | None:
    d1, d2 = ESQ_BRAZOS[familia]
    entre = float(rng.uniform(*ESQ_ANGULO))
    rot = float(rng.uniform(-ESQ_JITTER, ESQ_JITTER))
    # bisectriz nominal de los dos brazos; cada brazo a ±entre/2 de ella
    bis = math.degrees(math.atan2(*(_u(DIR[d1]) + _u(DIR[d2]))[::-1])) + rot
    L1, L2 = (float(rng.uniform(*ESQ_BRAZO)) for _ in range(2))
    w = int(rng.choice(GROSORES))
    for _ in range(60):
        v = rng.uniform(0, LADO32, size=2)
        p1 = v + L1 * _u(bis - entre / 2)
        p2 = v + L2 * _u(bis + entre / 2)
        pts = np.stack([p1, v, p2])
        if _dentro(pts, w):
            return {"pts": [pts], "cerrado": False, "ancla": v, "grosor": w,
                    "radio": -1.0, "angulo": bis, "apertura": entre, "largo": (L1 + L2) / 2}
    return None


def instancia(rng: np.random.Generator, familia: str) -> dict | None:
    """Una instancia de la familia con sus parámetros sorteados de rng, o None para `vacio`."""
    if familia == VACIO:
        return None
    if familia.startswith("arco"):
        return _arco(rng, familia)
    if familia.startswith("recta"):
        return _recta(rng, familia)
    if familia == "lazo":
        return _lazo(rng, familia)
    if familia.startswith("esquina"):
        return _esquina(rng, familia)
    raise ValueError(f"familia '{familia}' desconocida: {FAMILIAS}")


# ---------------------------------------------------------------- rasterizar y reducir

def rasterizar(inst: dict | None) -> np.ndarray:
    """(32, 32) uint8 0/1: la máscara del trazo."""
    from PIL import Image, ImageDraw                     # noqa: PLC0415
    im = Image.new("L", (LADO32, LADO32), 0)
    if inst is not None:
        d = ImageDraw.Draw(im)
        for pts in inst["pts"]:
            xy = [(float(x), float(y)) for x, y in pts]
            if inst["cerrado"]:
                xy.append(xy[0])
            d.line(xy, fill=1, width=inst["grosor"], joint="curve")
    return np.asarray(im, dtype=np.uint8)


def cuentas(mascara32: np.ndarray) -> np.ndarray:
    """(32, 32) 0/1 -> (8, 8) uint8: bits encendidos por bloque 4×4 (0..16)."""
    m = (mascara32 > 0).astype(np.uint8)
    return m.reshape(LADO, BLOQUE, LADO, BLOQUE).sum(axis=(1, 3)).astype(np.uint8)


def ruido(mascara32: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p_on, p_off = float(rng.uniform(*P_ON)), float(rng.uniform(*P_OFF))
    u = rng.random(mascara32.shape)
    m = mascara32.copy()
    m[(mascara32 == 0) & (u < p_on)] = 1
    m[(mascara32 == 1) & (u < p_off)] = 0
    return m


def muestra(rng: np.random.Generator, familia: str) -> dict:
    """Una imagen completa: principal + (quizá) secundaria + ruido. Determinista dado rng."""
    inst = instancia(rng, familia)
    while inst is None and familia != VACIO:          # no cupo: otro sorteo (consume rng, determinista)
        inst = instancia(rng, familia)
    m_p = rasterizar(inst)
    sec, m_s = -1, np.zeros_like(m_p)
    # `vacio` NUNCA lleva secundaria: es el negativo «sólo ruido», y con un trazo dejaría de serlo
    if familia != VACIO and rng.random() < P_SECUNDARIA:
        otras = [f for f in CON_TRAZO if f != familia]
        fam_s = str(rng.choice(otras))
        inst_s = instancia(rng, fam_s)
        while inst_s is None:
            inst_s = instancia(rng, fam_s)
        sec, m_s = FAMILIAS.index(fam_s), rasterizar(inst_s)
    m = ruido(np.maximum(m_p, m_s), rng)
    if inst is None:
        ancla32, ancla = np.array([-1.0, -1.0]), np.array([-1, -1])
    else:
        x, y = inst["ancla"]
        ancla32 = np.array([y, x])                                      # (fila, col)
        ancla = np.array([int(y // BLOQUE), int(x // BLOQUE)])
    p = inst or {"grosor": -1, "radio": -1.0, "angulo": -1.0, "apertura": -1.0, "largo": -1.0}
    return {"imagen": cuentas(m), "mascara8": cuentas(m_p), "principal": FAMILIAS.index(familia), "secundaria": sec,
            "ancla": ancla, "ancla32": ancla32, "grosor": int(p["grosor"]), "radio": float(p["radio"]),
            "angulo": float(p["angulo"]), "apertura": float(p["apertura"]), "largo": float(p["largo"])}


def tabla() -> list[str]:
    out = []
    for g, fams in GRUPOS.items():
        out.append(f"{g:<9} {', '.join(fams)}")
    out.append(f"vacío     {VACIO}  (sólo ruido)")
    out.append(f"arcos: r {ARCO_RADIO} px, apertura {ARCO_APERTURA}°, centro a ±{ARCO_JITTER}° · rectas: largo {RECTA_LARGO}, ±{RECTA_JITTER}°")
    out.append(f"lazo: a {LAZO_A}, b/a {LAZO_RAZON} · esquinas: brazos {ESQ_BRAZO}, entre {ESQ_ANGULO}°, ±{ESQ_JITTER}° · grosor {GROSORES} px")
    out.append(f"ruido: p_on {P_ON}, p_off {P_OFF} · segunda feature con p = {P_SECUNDARIA}")
    return out


def muestras_png(destino: Path, n: int = 12, semilla: int = 7) -> Path:
    """Rejilla: una fila por familia, n imágenes (tinta en negro). Sin matplotlib: PIL a secas."""
    from PIL import Image                                # noqa: PLC0415
    rng = np.random.default_rng(semilla)
    esc, sep = 6, 2
    W, H = n * (LADO * esc + sep) + sep, len(FAMILIAS) * (LADO * esc + sep) + sep
    im = Image.new("L", (W, H), 128)
    for r, fam in enumerate(FAMILIAS):
        for k in range(n):
            s = muestra(rng, fam)
            a = (255 - s["imagen"].astype(np.float32) / BITS * 255).astype(np.uint8)
            tile = Image.fromarray(a).resize((LADO * esc, LADO * esc), Image.NEAREST)
            # marca el ancla con un píxel gris en el centro de su celda
            if s["ancla"][0] >= 0:
                px = tile.load()
                cy, cx = (int(v) * esc + esc // 2 for v in s["ancla"])
                px[cx, cy] = 90
            im.paste(tile, (sep + k * (LADO * esc + sep), sep + r * (LADO * esc + sep)))
    destino.parent.mkdir(parents=True, exist_ok=True)
    im.save(destino)
    return destino


def main() -> int:
    if "--muestras" in sys.argv:
        destino = Path(__file__).resolve().parent.parent / "resultados" / "muestras-features.png"
        print(f"→ {muestras_png(destino)}  (filas en el orden de FAMILIAS: {', '.join(FAMILIAS)})")
        return 0
    print("\n".join(tabla()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
