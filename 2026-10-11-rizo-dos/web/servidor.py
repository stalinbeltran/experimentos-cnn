#!/usr/bin/env python3
"""rizo-2 — REVISOR DE DÍGITOS: app web móvil que CALCULA EN VIVO lo que ve el detector y lo que decide el compositor
(pedida por el dueño el 2026-10-15: «revisar los dígitos (los fallos, con más frecuencia), las respuestas de los detectores y
los valores asignados por el compositor; que corra en el dev y se calcule según lo pido; mantén varios compositores»).

    ../.venv/bin/python web/servidor.py            la levanta en el 8050 (o la unidad `rizo2-revisor`)
    ../.venv/bin/python web/servidor.py url        la URL con su token, ya comprobada contra el socket y ufw
    ../.venv/bin/python web/servidor.py estado     ¿se llega desde el móvil? (unidad · socket · ufw · token)
    ../.venv/bin/python web/servidor.py instalar   unidad de systemd + ufw, idempotente (en una máquina recién hecha)

QUÉ ES. Un servidor de stdlib (sin framework) que carga el dataset y el mapa delgado, calcula UNA VEZ las 394 columnas de
`cresta_curvas.caract` para todos los dígitos (train × 13 desplazamientos + val; 8 ms por dígito medido el 2026-10-15, ~30 s
en total, y se cachea en `web/cache/*.npz`, fuera de git) y, a partir de ahí, un COMPOSITOR es sólo «qué bloques entran»:
se entrena en segundos (MLP 256, 400 pasos, 3 semillas) y se queda en memoria. Las recetas de los compositores se guardan en
`web/cache/recetas.json` y se reentrenan al arrancar, así que sobreviven al reinicio de la unidad (no a rehacer el dev: esto
corre en el dev a propósito, y muere con él, como `foveal-vision-web`).

LO QUE CONTESTA, pantalla a pantalla (sale de las figuras 75–83 de rizo-2):
  LISTA      qué dígitos falla (o acierta) cada compositor, por clase y por confusión, con el dígito en miniatura
  DÍGITO     tinta, delgado, cresta y los objetos con nombre y posición (recta, arco, lazo, entrante, golpe, resto, cruz)
  COMPOSITOR las rejillas que recibe, los logits de las 10 clases y la atribución de cada entrada (predicho − real)
  COMPARAR   el mismo dígito leído por varios compositores: sólo se despliegan los bloques que difieren

LAS CINCO DECISIONES
1. **El dato es PRIVADO y este repo es PÚBLICO**, así que todo lleva TOKEN, que crea el servidor al arrancar y vive en
   `~/.cache/rizo2-revisor.token` (nunca se configura: la lección de la web móvil del 2026-09-15). `/salud` es lo único sin token.
2. **Los números son los de las corridas**: mismo `caract`, mismo aumento (13 copias ±1–3 px), mismo MLP que
   `todo_objetos_entrada.py` / `rastrear.entrenar` (copiado aquí, con el test que lo compara contra el de verdad). Un
   compositor de aquí con la receta «aprobada» tiene que caer en el `val_rango` del JSON de su corrida; lo comprueba el test.
3. **Nada se precalcula fuera**: cambiar de receta, de semillas o de rejilla es un clic y segundos. Lo caro (las 394 columnas)
   se calcula una vez por versión de `cresta_curvas.py` (la caché lleva el hash de su texto) y 4×4 cuando se pide.
4. **`estado` pregunta al socket y al cortafuegos**, nunca al disco ni a `is-active` (las cuatro puertas del 2026-09-15); el
   puerto se abre en `ufw` en el MISMO paso que la unidad.
5. **Un modo desconocido se NIEGA** (2026-09-08): el modo se fija una vez y no se relee.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import torch.nn.functional as F

AQUI = Path(__file__).resolve().parents[1]                        # 2026-10-11-rizo-dos
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import cresta_curvas as CC                                        # noqa: E402
import vertice_libre as VLB                                       # noqa: E402  (el vértice LIBRE, sobre el delgado: § 105)
from expcnn import exigir_dataset, ruta_datos                     # noqa: E402

PUERTO = 8050
UNIDAD = "rizo2-revisor"
DATASET = "uci-optdigits-orig-32px-r20261005"                    # el de rizo-2 (nn/evaluar.py)
DELGADOS = AQUI.parent / "2026-10-15-delgado-cnn" / "resultados" / "delgados-k3x3.npz"
CACHE = AQUI / "web" / "cache"
AUM = (-3, -2, -1, 1, 2, 3)                                       # resistencia.AUM: 13 copias por dígito de train
SEMILLAS = (0, 1, 2)
OCULTA, PASOS = 256, 400                                          # el MLP de `entrenar`; van en la huella de los pesos guardados


def almacen_revisor() -> Path | None:
    """Dónde SOBREVIVEN a rehacer el dev los PESOS de cada compositor y las recetas: el repo de DATOS (el almacén), o None si
    no está clonado. Pedido por el dueño el 2026-10-16 («guarda los pesos de los compositores, para que la próxima vez entre en
    operación inmediatamente»): hasta entonces un dev nuevo reentrenaba las 5 recetas con extra (~1,5 h) antes de servir.
    Un `.pt` por compositor (~1,3 MB: 3 redes de 104 k parámetros) con su HUELLA (dataset, hash de cresta_curvas.py, hash de
    las etiquetas a mano, receta, MLP): si al arrancar la huella coincide, se restaura sin entrenar; si no, se reentrena y se
    vuelve a guardar. Se commitea y empuja SÓLO esa carpeta tras cada entrenamiento; si el push falla (mini apagado) queda
    commiteado y se reintenta en el siguiente. El `.gitignore` del repo de datos ya deja entrar
    `experimentos-cnn-resultados/**/*.pt`."""
    if os.environ.get("REVISOR_SIN_ALMACEN"):                     # las PRUEBAS: un test que entrena no puede empujar al almacén
        return None                                               # (pasó el 2026-10-16: «prueba-golpes» acabó commiteado allí)
    d = ruta_datos()
    return (d / "experimentos-cnn-resultados" / "rizo-2" / "revisor") if d else None


def cache_del_almacen(f: Path) -> Path | None:
    """La copia en el almacén de un fichero de la caché de features (`web/cache/*.npz`), o None sin repo de datos.
    Pedido por el dueño el 2026-10-16 («todo lo que se deba guardar debe guardarse en el volumen»): las features de extra tardan
    ~8 min (471 s medidos el 2026-10-16) y las de base ~30 s, y sin esto un dev nuevo las recalculaba aunque los pesos ya
    estuvieran. Viven en `revisor/cache/`, con el hash de cresta_curvas.py en el nombre, así que una versión nueva del detector
    no puede leer la caché de la vieja."""
    alm = almacen_revisor()
    return (alm / "cache" / f.name) if alm else None


def traer_cache(f: Path, que: str = "") -> bool:
    """Si `f` no está en local pero sí en el almacén, se copia (True). Y si está en local y FALTA en el almacén, se sube:
    medido el 2026-10-16, la caché de extra la calculó la suite de tests (almacén apagado) y el entrenamiento real la encontró
    en local y nunca la subió. Nunca lanza: una copia rota se recalcula como siempre."""
    if f.is_file():
        a = cache_del_almacen(f)
        if a is not None and not a.is_file():
            guardar_cache(f, que or "de features (ya estaba en local)")
        return True
    a = cache_del_almacen(f)
    if a is None or not a.is_file():
        return False
    try:
        f.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(a, f); return True
    except OSError as e:
        print(f"  ⚠ no pude traer {a.name} del almacén: {e}", flush=True); return False


def guardar_cache(f: Path, que: str):
    """Copia `f` al almacén y lo empuja. Nunca lanza (el aviso es una comodidad; lo local sigue sirviendo)."""
    a = cache_del_almacen(f)
    if a is None:
        print("  ⚠ sin repo de datos: la caché NO sobrevive a rehacer el dev", flush=True); return
    try:
        a.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(f, a); empujar_almacen(f"rizo-2 revisor: caché {que} ({f.name}, {f.stat().st_size // 1024} KB)")
    except OSError as e:
        print(f"  ⚠ no pude guardar {f.name} en el almacén: {e}", flush=True)


def empujar_almacen(mensaje: str):
    """`git add` de la carpeta del revisor en el repo de datos, commit si hay algo y push. Nunca lanza: si el push falla (mini
    apagado) queda commiteado y lo reintenta el siguiente. Lo usan los pesos, las recetas y la caché."""
    alm = almacen_revisor()
    if alm is None:
        return
    try:
        repo = alm.parents[2]; rel = alm.relative_to(repo)
        g = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, timeout=120)
        g("add", "--", str(rel))
        if g("diff", "--cached", "--quiet", "--", str(rel)).returncode != 0:
            g("commit", "-q", "-m", mensaje)
        r = g("push", "-q")
        for intento in range(3):                                  # 2026-10-16: con el entrenamiento en PARALELO (un server por
            if r.returncode == 0:                                 # receta) varios empujan a la vez y el segundo es non-fast-forward.
                break                                             # Cada uno toca su propio .pt, así que el rebase es limpio; si no
            g("pull", "-q", "--rebase"); time.sleep(2 + 3 * intento); r = g("push", "-q")   # lo es, queda commiteado como antes
        print(f"  almacén ← {rel}" + ("" if r.returncode == 0 else f"  ⚠ push falló (queda commiteado): {(r.stderr or r.stdout).strip()[-200:]}"), flush=True)
    except Exception as e:                                        # noqa: BLE001
        print(f"  ⚠ no pude empujar al almacén: {type(e).__name__}: {e}", flush=True)

# Cómo se llama cada bloque de `caract` para una persona, y cómo se nombran sus columnas (bloque · orientación/cono · celda).
ORI = ["0° —", "45° /", "90° |", "135° \\"]; SENT = ["k→", "k←", "k↓", "k↑"]; DIR4 = ["→ der", "← izq", "↓ abajo", "↑ arriba"]
BLOQUES = {   # clave de caract → (etiqueta, familia, qué lista de objetos pinta)
    "rectas": ("rectas", "4 orient × (rejilla + cuenta + largo/32)", "rectas"),
    "rectas_todo": ("rectas ∪ restos (§ 74)", "4 orient × (rejilla + cuenta + largo/32)", "rectas+restos"),
    "restos": ("restos aparte", "4 orient × (rejilla + cuenta + largo/32)", "restos"),
    "arcos": ("arcos", "4 sentidos × (rejilla + cuenta + largo/32)", "arcos"),
    "golpes": ("golpes", "rejilla + cuenta", "golpes"),
    "lazos": ("lazos (presencia)", "rejilla + cuenta", "lazos"),
    "lazos_region": ("lazos REGIÓN (§ 79, aprobada)", "rejilla (fracción de celda cubierta) + cuenta", "lazos"),
    "lazos_area": ("lazos ÁREA (§ 79)", "rejilla (altura área/300) + cuenta", "lazos"),
    "concavidades": ("concavidades de arco (§ 54)", "4 direcciones × (rejilla + cuenta + abertura/32)", "concavidades"),
    "entrantes": ("entrantes a 45° (§ 58)", "4 direcciones × (rejilla + cuenta + abertura/32)", "entrantes"),
    "entrantes4": ("entrantes sin posición: fuerza (§ 59)", "4 números", "entrantes"),
    "entrantes4n": ("entrantes sin posición: cuenta (§ 59)", "4 números", "entrantes"),
    "entrantes60": ("entrantes conos 60°, presencia (§ 66, aprobada)", "4 conos × rejilla 3×3", "entrantes"),
    "entrantes60f": ("entrantes conos 60°, fuerza (§ 66)", "4 conos × rejilla 3×3", "entrantes"),
    "cruces": ("cruces (§ 69)", "rejilla 3×3 + cuenta", "cruces"),
    "vertice": ("vértice del 4 (bordes de la tinta, § 95)", "rejilla + cuenta", "vertice"),
    "vertice_1": ("vértice del 4 en UNA celda (prueba, 2026-10-16)", "rejilla one-hot: la celda que contiene el centro, sin mancha ni solape + cuenta", "vertice"),
    # El vértice LIBRE (§ 105, el dueño, 2026-10-17: «métELOS en el app… yo debo poder elegir cuál entreno»): en cualquier sitio de
    # la cresta del DELGADO, 4 direcciones N E S O, UNA celda, y por celda presencia + amplitud + las tres redondeces de § 104.
    # Cada una es una PIEZA: se elige al entrenar. Calculado en nn/vertice_libre.py, fuera de cresta_curvas.py (su hash es aparte).
    "vertice_libre": ("vértice LIBRE en el delgado (§ 105)", "N E S O × (rejilla + amplitud + despegue + cerca + radio + cuenta)", "vertices_libres"),
}
SIN_POSICION = {"entrantes4", "entrantes4n"}   # regla del dueño (2026-10-15): se DICE cuál no lleva posición
# RETIRADOS (el dueño, 2026-10-17: «elimina los features de tinta… aunque por el compositor existente deban seguir existiendo en
# el código, ya no deben aparecer como opciones para elegir»). Regla del repo: TODO se calcula sobre el DELGADO, y estos dos miran
# la TINTA. Siguen en BLOQUES para que los compositores ya entrenados con ellos carguen y se lean; no se ofrecen en «entrenar
# otro» (`bloques_ofrecidos`) y un compositor NUEVO que los pida se niega (`comprobar_nuevo`).
RETIRADOS = {"vertice": "usa la TINTA (regla: todo sobre el delgado)", "vertice_1": "usa la TINTA (regla: todo sobre el delgado)"}


def bloques_ofrecidos() -> list[str]:
    """Los bloques que se pueden ELEGIR para entrenar: todos menos los retirados."""
    return [k for k in BLOQUES if k not in RETIRADOS]


def comprobar_nuevo(bloques: list[str]):
    """Un compositor NUEVO no puede llevar un bloque retirado (los ya guardados sí cargan: ver App.nuevo(existente=True))."""
    malos = [b for b in bloques if pieza(b)[0] in RETIRADOS]
    if malos:
        raise ValueError("bloque retirado: " + "; ".join(f"«{pieza(b)[0]}» {RETIRADOS[pieza(b)[0]]}" for b in malos))

RECETAS_BASE = {   # las corridas de 144 del README, por nombre: lo que un compositor «es» son sus bloques
    # ⚠ Regla del dueño (2026-10-16): los GOLPES NO van por defecto en un compositor NUEVO (CLAUDE.md del repo). Estas
    # recetas los llevan porque ya están entrenadas y medidas (§ 85i, § 95, § 96); no se tocan sin que él lo ordene.
    "aprobada: lazo región (§ 79)": ["rectas_todo", "arcos", "golpes", "lazos_region", "entrantes60"],
    "restos fundido, lazo presencia (§ 74)": ["rectas_todo", "arcos", "golpes", "lazos", "entrantes60"],
    "lazo área (§ 79)": ["rectas_todo", "arcos", "golpes", "lazos_area", "entrantes60"],
    "conos 60° sin restos (§ 66)": ["rectas", "arcos", "golpes", "lazos", "entrantes60"],
}
BLOQUES_DEFECTO = ["rectas_todo", "arcos", "lazos_region", "entrantes60"]   # aprobadas, SIN golpes (el dueño, 2026-10-16) y SIN el vértice
RECETA_DEFECTO = "aprobadas, sin golpes · 4 en dos · +extra"                  # de tinta (retirado el 2026-10-17, ver RETIRADOS)
CORRIDA_DE = {   # receta → JSON de su corrida, para que el test compruebe que esto reproduce los números del README
    "aprobada: lazo región (§ 79)": "40-entrada-cresta-lazos-region.json",
    "restos fundido, lazo presencia (§ 74)": "40-entrada-cresta-restos-fundido.json",
    "lazo área (§ 79)": "40-entrada-cresta-lazos-area.json",
    "conos 60° sin restos (§ 66)": "40-entrada-cresta-entrantes60-presencia.json",
}


def celdas_nombres(c: int) -> list[str]:
    if c == 3:
        return ["arriba-izq", "arriba-centro", "arriba-der", "centro-izq", "centro", "centro-der", "abajo-izq", "abajo-centro", "abajo-der"]
    return [f"f{i}c{j}" for i in range(c) for j in range(c)]


def columnas(bloque: str, celdas: int) -> list[dict]:
    """Una entrada por columna del bloque: {nombre, sub (orientación/cono), celda (índice o None), tipo (rejilla|cuenta|largo|abertura|fuerza)}."""
    c = celdas * celdas; cn = celdas_nombres(celdas); out = []
    if bloque in ("rectas", "rectas_todo", "restos", "arcos", "concavidades", "entrantes"):
        et = SENT if bloque == "arcos" else DIR4 if bloque in ("concavidades", "entrantes") else ORI
        extra = "abertura/32" if bloque in ("concavidades", "entrantes") else "largo/32"
        for o in range(4):
            out += [{"sub": et[o], "celda": k, "tipo": "rejilla", "nombre": f"{cn[k]}"} for k in range(c)]
            out += [{"sub": et[o], "celda": None, "tipo": "cuenta", "nombre": "cuenta"}, {"sub": et[o], "celda": None, "tipo": "largo", "nombre": extra}]
    elif bloque in ("golpes", "lazos", "lazos_region", "lazos_area", "vertice", "vertice_1"):
        out += [{"sub": "", "celda": k, "tipo": "rejilla", "nombre": cn[k]} for k in range(c)] + [{"sub": "", "celda": None, "tipo": "cuenta", "nombre": "cuenta"}]
    elif bloque in ("entrantes4", "entrantes4n"):
        out += [{"sub": DIR4[o], "celda": None, "tipo": "fuerza" if bloque == "entrantes4" else "cuenta", "nombre": "fuerza" if bloque == "entrantes4" else "cuenta"} for o in range(4)]
    elif bloque in ("entrantes60", "entrantes60f"):
        c3 = celdas_nombres(3)
        for o in range(4):
            out += [{"sub": DIR4[o], "celda": k, "tipo": "rejilla", "nombre": c3[k]} for k in range(9)]
    elif bloque == "vertice_libre":                               # mismo orden que VLB.bloque: por dirección, sus piezas
        for d in VLB.ORDEN:
            out += [{"sub": d, "celda": k, "tipo": "rejilla", "nombre": cn[k]} for k in range(c)]
            for q in VLB.PIEZAS[1:-1]:                                # cada medida, su propia rejilla («N · radio»…)
                out += [{"sub": f"{d} · {q}", "celda": k, "tipo": q, "nombre": cn[k]} for k in range(c)]
            out += [{"sub": d, "celda": None, "tipo": "cuenta", "nombre": "cuenta"}]
    elif bloque == "cruces":
        c3 = celdas_nombres(3)
        out += [{"sub": "", "celda": k, "tipo": "rejilla", "nombre": c3[k]} for k in range(9)] + [{"sub": "", "celda": None, "tipo": "cuenta", "nombre": "cuenta"}]
    return out


def bloque_vertice_1(tinta, celdas: int = 3) -> list[float]:
    """El vértice puesto en UNA sola celda: la que contiene su centro, partiendo los 32 px en `celdas` tramos SIN solape
    (int(cx·celdas/32)). El dueño, 2026-10-16: «al detectar el vértice lo posicionamos en 2 celdas; ¿puede estar sólo en
    una?». En el bloque `vertice` la mancha gaussiana (σ 2 px) y las celdas solapadas de adaptive_max_pool ([0,11) [10,22)
    [21,32)) encendían DOS celdas en 159 de 475 vértices (medido ese día). Se calcula aquí y no en cresta_curvas.py a
    propósito: tocar ese fichero cambia su hash, y con él la huella de todo compositor guardado y las cachés de features."""
    v = CC.vertices(tinta); g = [0.0] * (celdas * celdas)
    for o in v:
        c = min(celdas - 1, int(o["cx"] * celdas / 32)); r = min(celdas - 1, int(o["cy"] * celdas / 32)); g[r * celdas + c] = 1.0
    return g + [float(len(v))]


def rangos(celdas: int) -> dict:
    """Dónde está cada bloque en la matriz que usa la app: los de caract (CC.bloques) y, al final, los que se calculan aquí."""
    B = dict(CC.bloques(celdas)); n = max(z for _, z in B.values())
    B["vertice_1"] = (n, n + celdas * celdas + 1); n = B["vertice_1"][1]
    B["vertice_libre"] = (n, n + len(columnas("vertice_libre", celdas)))
    return B


def partes(bloque: str) -> list[str]:
    """Las PIEZAS de un bloque, en su orden de columnas: «rejilla», «cuenta», «largo», «abertura», «fuerza» (las que tenga)."""
    return list(dict.fromkeys("largo" if c["tipo"] == "largo" and c["nombre"] == "largo/32" else
                              "abertura" if c["tipo"] == "largo" else c["tipo"] for c in columnas(bloque, 3)))


def pieza(entrada: str) -> tuple[str, list[str] | None]:
    """«vertice» → ("vertice", None) = el bloque entero; «vertice:rejilla+cuenta» → ("vertice", ["rejilla", "cuenta"]).
    El dueño (2026-10-16): «separa las rejillas 3×3 de todo lo demás; quiero entrenar sin la cuenta y no se puede»."""
    base, sep, resto = entrada.partition(":")
    return base, ([q for q in resto.split("+") if q] if sep else None)   # «vertice:» = ninguna pieza, no «todas»


def normalizar(entrada: str) -> str:
    """Una entrada con TODAS las piezas se guarda como el bloque a secas: así una receta vieja conserva su nombre, su huella
    y sus pesos del almacén. Se niega ante un bloque o una pieza desconocidos."""
    base, ps = pieza(entrada)
    if base not in BLOQUES:
        raise ValueError(f"bloque desconocido: {base}")
    if ps is None:
        return base
    todas = partes(base); malas = [q for q in ps if q not in todas]
    if malas or not ps:
        raise ValueError(f"«{base}» no tiene la pieza {malas or '(ninguna)'}; tiene {todas}")
    ps = [q for q in todas if q in ps]                             # el orden de las columnas, no el del clic
    return base if ps == todas else f"{base}:{'+'.join(ps)}"


def columnas_de(entrada: str, celdas: int) -> tuple[str, list[int], list[dict]]:
    """(bloque, índices DENTRO del bloque, columnas) de una entrada: todas, o sólo las de sus piezas."""
    base, ps = pieza(entrada); cols = columnas(base, celdas)
    if ps is None:
        return base, list(range(len(cols))), cols
    tipo = lambda c: "abertura" if c["tipo"] == "largo" and c["nombre"] != "largo/32" else c["tipo"]
    idx = [k for k, c in enumerate(cols) if tipo(c) in ps]
    return base, idx, [cols[k] for k in idx]


def mover(a, dx=0, dy=0):
    """= evaluar.mover (relleno de fondo, nunca np.roll)."""
    o = np.zeros_like(a); H, W = a.shape
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    o[yd, xd] = a[ys, xs]; return o


def entrenar(xtr, ytr, sem, oculta=256, salidas=10):
    """= rastrear.entrenar, copiado para no arrastrar sus imports (matplotlib, los bocetos de Gabor…). `probar_revisor.py`
    comprueba que da los MISMOS logits que el original con la misma semilla."""
    torch.manual_seed(sem); mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    red = torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], oculta), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(oculta, salidas))
    opt = torch.optim.Adam(red.parameters(), lr=3e-3, weight_decay=1e-3)
    xt, yt = torch.from_numpy((xtr - mu) / sd).float(), torch.from_numpy(ytr).long()
    for _ in range(400):
        red.train(); opt.zero_grad(); F.cross_entropy(red(xt), yt).backward(); opt.step()
    red.eval(); return red, mu, sd


def _limpio(o):
    """numpy → json."""
    if isinstance(o, dict):
        return {k: _limpio(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_limpio(v) for v in o]
    if isinstance(o, np.ndarray):
        return np.round(o.astype(float), 2).tolist()
    if isinstance(o, (np.floating, float)):
        return round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


class Datos:
    """El dataset, el delgado y las 394 columnas de todos los dígitos (y del train aumentado), por rejilla."""

    def __init__(self):
        d = dict(np.load(exigir_dataset(DATASET) / "datos.npz"))
        self.tinta = (d["imagenes"] > 0).astype(np.uint8); self.y = d["etiquetas"].astype(int); part = d["particion"]
        if not DELGADOS.is_file():
            print(f"  no está {DELGADOS.name}: lo regenero con exportar_delgados.py (segundos)…", flush=True)
            subprocess.run([sys.executable, str(DELGADOS.parents[1] / "nn" / "exportar_delgados.py")], cwd=DELGADOS.parents[1], check=True)
        self.M = np.load(DELGADOS)["M"].astype(np.float32)
        self.tr, self.va, self.ex = np.flatnonzero(part == "train"), np.flatnonzero(part == "val"), np.flatnonzero(part == "extra")
        self.aum = [(0, 0)] + [(k, 0) for k in AUM] + [(0, k) for k in AUM]
        self.hash = hashlib.sha1((AQUI / "nn" / "cresta_curvas.py").read_bytes()).hexdigest()[:10]
        self.X = {}                                               # celdas → (X_todos (5620,394), X_tr_aum (180·13,394), Y_tr_aum)
        self._ob = {}; self._lock = threading.Lock(); self._finas = None

    def subgrupo4(self, i: int) -> bool:
        """El 4 CERRADO por arriba = «4l» (el dueño, 2026-10-15, tercera versión: «198, 225, 239 y 250 son un 4l… el lazo no está
        visible, pero debo separar bien las clases»). Dos señales, cualquiera vale: (a) la cresta tiene un LAZO; (b) el trazo
        grueso rellena el triángulo y no deja lazo, pero sólo UN extremo libre de la cresta queda por encima del nudo más alto
        (el vértice); un 4 abierto deja DOS (la punta izquierda y la del palo). Medido en val (fig. 87): 27 de 163 (21 con lazo +
        6 rellenos); con la primera versión (cruz + lazo) eran 13 y con la segunda (sólo lazo) 21."""
        ob = self.objetos(i)
        if len(ob["lazos"]) > 0:
            return True
        return self.extremos_sobre_el_nudo(i) <= 1

    def extremos_sobre_el_nudo(self, i: int, margen: float = 2.0) -> int:
        """Cuántos extremos libres de la cresta (racimos con un solo tramo) quedan ≥ `margen` px por encima del nudo más alto
        (racimo con ≥ 3 tramos). Sin ningún nudo, cuenta todos los extremos."""
        tr, _ = CC.tramos(self.M[i]); cnt = {}; pos = {}
        for t in tr:
            p = np.asarray(t["pts"], float)
            for k, q in ((t["a"], p[0]), (t["b"], p[-1])):
                cnt[k] = cnt.get(k, 0) + 1; pos.setdefault(k, []).append(q)
        ext = [np.mean(pos[k], 0) for k in cnt if cnt[k] == 1]; nud = [np.mean(pos[k], 0) for k in cnt if cnt[k] >= 3]
        if not nud:
            return len(ext)
        yn = min(q[1] for q in nud); return int(sum(q[1] < yn - margen for q in ext))

    def etiquetas_finas(self) -> np.ndarray:
        """y con el 4 partido: 4 = «4a» (abierto), 10 = «4l» (cerrado por arriba: lazo, o un solo extremo sobre el nudo). Se calcula una vez (sólo los 4: ~570 dígitos, segundos)."""
        if self._finas is None:
            f = self.y.copy()
            for i in np.flatnonzero(self.y == 4):
                if self.subgrupo4(i):
                    f[i] = 10
            # ⚠ Las correcciones A MANO del dueño mandan sobre la regla (2026-10-15: «son varios, prefiero marcarlos a mano»).
            # Viven en web/etiquetas-4.json, EN GIT (no en la caché): son un dato que no se re-deriva (R9).
            for i, cl in self.manuales().items():
                if self.y[i] == 4:
                    f[i] = 10 if cl == "4l" else 4
            self._finas = f
        return self._finas

    def hash_manuales(self) -> str:
        """Huella de las etiquetas a mano: cambiarlas cambia las clases finas, así que unos pesos guardados con otras no valen."""
        f = AQUI / "web" / "etiquetas-4.json"
        return hashlib.sha1(f.read_bytes()).hexdigest()[:10] if f.is_file() else "sin"

    def manuales(self) -> dict:
        """{idx: '4l'|'4a'} decididos a mano. Fichero: {"4l": [idx…], "4a": [idx…], "quien": …, "cuando": …}."""
        f = AQUI / "web" / "etiquetas-4.json"
        if not f.is_file():
            return {}
        try:
            j = json.loads(f.read_text()); return {**{int(i): "4l" for i in j.get("4l", [])}, **{int(i): "4a" for i in j.get("4a", [])}}
        except (json.JSONDecodeError, ValueError):
            return {}

    def marcar(self, i: int, clase: str):
        """Guarda una decisión a mano ('4l', '4a' o 'auto' = quitarla) y olvida las etiquetas calculadas (NO reentrena nada)."""
        if self.y[i] != 4:
            raise ValueError(f"idx {i} no es un 4")
        f = AQUI / "web" / "etiquetas-4.json"; m = self.manuales(); m.pop(i, None)
        if clase in ("4l", "4a"):
            m[i] = clase
        elif clase != "auto":
            raise ValueError("clase: 4l · 4a · auto")
        f.write_text(json.dumps({"4l": sorted(k for k, v in m.items() if v == "4l"), "4a": sorted(k for k, v in m.items() if v == "4a"),
                                 "quien": "el dueño, desde /etiquetar", "cuando": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                                 "regla_base": "Datos.subgrupo4 (§ 85f): lazo, o un solo extremo libre sobre el nudo"}, indent=1) + "\n")
        self._finas = None

    def cuatros(self) -> list[dict]:
        """Todos los 4 del dataset con su parte, la clase por la REGLA y la decisión a mano si la hay."""
        m = self.manuales(); parte = {int(i): "train" for i in self.tr}; parte.update({int(i): "val" for i in self.va}); parte.update({int(i): "extra" for i in self.ex})
        return [{"idx": int(i), "parte": parte.get(int(i), "?"), "regla": "4l" if self.subgrupo4(int(i)) else "4a", "manual": m.get(int(i)), "lazo": len(self.objetos(int(i))["lazos"]) > 0}
                for i in np.flatnonzero(self.y == 4)]

    def _v1(self, tintas, celdas):
        return np.asarray([bloque_vertice_1(t, celdas) for t in tintas], np.float32)

    def vert1(self, celdas: int, extra: bool = False):
        """Las columnas de `vertice_1` para (todos, train aumentado) o para los extra aumentados, en el MISMO orden que caract.
        ~30 s todo (0,6 ms por dígito, medido 2026-10-16); caché propia `vert1-…`, también en el almacén."""
        f = CACHE / f"vert1-{self.hash}-{celdas}x{celdas}{'-extra' if extra else ''}.npz"
        if traer_cache(f, f"vertice_1 {celdas}×{celdas}"):
            z = np.load(f); return (z["ex"],) if extra else (z["todos"], z["tr"])
        idx = self.ex if extra else self.tr
        aum = np.concatenate([self._v1(np.stack([mover(x, *m) for x in self.tinta[idx]]), celdas) for m in self.aum])   # aum[0] = sin mover
        if extra:
            np.savez_compressed(f, ex=aum); guardar_cache(f, f"vertice_1 {celdas}×{celdas} extra"); return (aum,)
        todos = self._v1(self.tinta, celdas); np.savez_compressed(f, todos=todos, tr=aum); guardar_cache(f, f"vertice_1 {celdas}×{celdas}")
        return todos, aum

    def vlibre(self, celdas: int, extra: bool = False):
        """Las columnas de `vertice_libre` (sobre el DELGADO) para (todos, train aumentado) o para los extra aumentados, en el MISMO
        orden que caract. ~8 ms por dígito (medido 2026-10-17): ~1 min todos + train, ~7 min los extra. Caché propia
        `vlibre-<hash de vertice_libre.py + cresta_curvas.py>-…`, también en el almacén."""
        f = CACHE / f"vlibre-{VLB.huella_modulo()}-{celdas}x{celdas}{'-extra' if extra else ''}.npz"
        if traer_cache(f, f"vértice libre {celdas}×{celdas}"):
            z = np.load(f); return (z["ex"],) if extra else (z["todos"], z["tr"])
        uno = lambda Ms: np.asarray([VLB.bloque(m, celdas) for m in Ms], np.float32)
        t = time.time(); idx = self.ex if extra else self.tr
        print(f"  calculando el vértice libre {celdas}×{celdas} ({'extra' if extra else 'todos + train'})…", flush=True)
        aum = np.concatenate([uno(np.stack([mover(x, *m) for x in self.M[idx]])) for m in self.aum])   # aum[0] = sin mover
        if extra:
            np.savez_compressed(f, ex=aum); guardar_cache(f, f"vértice libre {celdas}×{celdas} extra"); print(f"  listo en {time.time() - t:.0f} s", flush=True); return (aum,)
        todos = uno(self.M); np.savez_compressed(f, todos=todos, tr=aum); guardar_cache(f, f"vértice libre {celdas}×{celdas}")
        print(f"  listo en {time.time() - t:.0f} s", flush=True); return todos, aum

    def features(self, celdas: int):
        if celdas in self.X:
            return self.X[celdas]
        todos, tr, ytr = self._features_caract(celdas); vt, vtr = self.vert1(celdas); lt, ltr = self.vlibre(celdas)
        self.X[celdas] = (np.concatenate([todos, vt, lt], 1), np.concatenate([tr, vtr, ltr], 1), ytr); return self.X[celdas]

    def _features_caract(self, celdas: int):
        CACHE.mkdir(parents=True, exist_ok=True); f = CACHE / f"caract-{self.hash}-{celdas}x{celdas}.npz"
        if traer_cache(f, f"de features {celdas}×{celdas}"):      # local, o la copia del almacén (del dev anterior)
            z = np.load(f); return z["todos"], z["tr"], z["ytr"]
        t = time.time(); print(f"  calculando caract {celdas}×{celdas} de {len(self.M)} + {len(self.tr) * 12} dígitos…", flush=True)
        todos = CC.caract(self.M, celdas, self.tinta)                     # la tinta va con el delgado: el bloque «vertice» la necesita (2026-10-16)
        tr = np.concatenate([todos[self.tr]] + [CC.caract(np.stack([mover(x, *m) for x in self.M[self.tr]]), celdas, np.stack([mover(x, *m) for x in self.tinta[self.tr]])) for m in self.aum[1:]])
        ytr = np.concatenate([self.y[self.tr]] * len(self.aum))
        np.savez_compressed(f, todos=todos, tr=tr, ytr=ytr)
        print(f"  listo en {time.time() - t:.0f} s → {f.name}", flush=True); guardar_cache(f, f"de features {celdas}×{celdas}"); return todos, tr, ytr

    def features_extra(self, celdas: int):
        """Las 394 columnas de los 3823 «extra» × 13 desplazamientos, para entrenar con MÁS dígitos (§ 819: 180 + extra = 4003).
        ⚠ Con extra en train no queda conjunto a ciegas: val es la única medida. Se calcula una vez (~7 min estimados: 50 k dígitos
        a 8 ms) y se cachea como las demás."""
        f = CACHE / f"caract-{self.hash}-{celdas}x{celdas}-extra.npz"
        if traer_cache(f, f"de features extra {celdas}×{celdas}"):
            z = np.load(f); return np.concatenate([z["ex"], self.vert1(celdas, extra=True)[0], self.vlibre(celdas, extra=True)[0]], 1), z["yex"]
        t = time.time(); print(f"  calculando caract {celdas}×{celdas} de los {len(self.ex)} extra × {len(self.aum)}…", flush=True)
        todos = self._features_caract(celdas)[0]
        ex = np.concatenate([todos[self.ex]] + [CC.caract(np.stack([mover(x, *m) for x in self.M[self.ex]]), celdas, np.stack([mover(x, *m) for x in self.tinta[self.ex]])) for m in self.aum[1:]])
        yex = np.concatenate([self.y[self.ex]] * len(self.aum)); np.savez_compressed(f, ex=ex, yex=yex)
        print(f"  listo en {time.time() - t:.0f} s → {f.name}", flush=True); guardar_cache(f, f"de features extra {celdas}×{celdas}")
        return np.concatenate([ex, self.vert1(celdas, extra=True)[0], self.vlibre(celdas, extra=True)[0]], 1), yex

    def objetos(self, i: int) -> dict:
        with self._lock:
            if i not in self._ob:
                self._ob[i] = CC.objetos(self.M[i])
            return self._ob[i]


class Compositor:
    """Una receta (bloques + celdas + semillas) entrenada: redes, predicciones de val, fallos, confusiones y cuota."""

    CLASES = [str(k) for k in range(10)] + ["4l"]              # la 11.ª sólo existe con sub4; entonces la «4» es la 4a (abierta)
    CLASES_FINAS = [str(k) if k != 4 else "4a" for k in range(10)] + ["4l"]

    def __init__(self, nombre: str, bloques: list[str], celdas: int = 3, semillas=SEMILLAS, sub4: bool = False, extra: bool = False):
        self.nombre, self.bloques, self.celdas, self.semillas, self.sub4, self.extra = nombre, list(bloques), int(celdas), tuple(semillas), bool(sub4), bool(extra)
        self.estado = "pendiente"; self.error = ""; self.redes = []; self.segundos = 0.0; self.resumen = {}

    def cortes(self, datos: Datos | None = None) -> dict:
        n = 0; out = {}
        for b in self.bloques:
            k = len(columnas_de(b, self.celdas)[1]); out[b] = (n, n + k); n += k
        return out

    def corta(self, X, datos=None):
        """Las columnas de la receta: cada entrada es un bloque entero o sólo algunas de sus piezas (2026-10-16)."""
        B = rangos(self.celdas); partes_x = []
        for b in self.bloques:
            base, idx, _ = columnas_de(b, self.celdas); a = B[base][0]
            partes_x.append(X[:, [a + k for k in idx]])
        return np.concatenate(partes_x, 1)

    def entrenar(self, datos: Datos):
        try:
            self.estado = "entrenando"; t = time.time()
            todos, Xtr, Ytr = datos.features(self.celdas); Xtr, Xva = self.corta(Xtr, datos), self.corta(todos[datos.va], datos)
            idx_tr = np.concatenate([datos.tr] * len(datos.aum))  # de qué dígito sale cada fila de train (para las etiquetas finas)
            if self.extra:                                        # § 819: 180 + 3823 extra; ⚠ sin conjunto a ciegas
                Xex, Yex = datos.features_extra(self.celdas); Xtr = np.concatenate([Xtr, self.corta(Xex, datos)]); Ytr = np.concatenate([Ytr, Yex])
                idx_tr = np.concatenate([idx_tr, np.concatenate([datos.ex] * len(datos.aum))])
            self.n = Xtr.shape[1]; self.redes = []; preds = []; finos = []; accs = []; accs_finos = []; cuotas = []
            if self.sub4:                                         # el 4 partido en dos: 11 salidas; el acierto se mide con 4l → 4 (comparable) y también con las dos separadas
                yf = datos.etiquetas_finas(); Ytr = yf[idx_tr]; assert len(Ytr) == len(Xtr)
            for s in self.semillas:
                red, mu, sd = entrenar(Xtr, Ytr, s, salidas=11 if self.sub4 else 10); self.redes.append((red, mu, sd))
                with torch.no_grad():
                    lg = red(torch.from_numpy((Xva - mu) / sd).float()).numpy()
                pf = lg.argmax(1); finos.append(pf); pj = np.where(pf == 10, 4, pf)
                # ⚠ Con el 4 en dos, la clase «4» NO EXISTE (el dueño, 2026-10-15): se evalúa en las 11 clases, y un 4l leído 4a es
                # un fallo. «Juntando» queda sólo como dato secundario (val_juntando) para comparar con las recetas sin partir.
                p = pf if self.sub4 else pj; preds.append(p); accs.append(float((p == (yf[datos.va] if self.sub4 else datos.y[datos.va])).mean()))
                if self.sub4:
                    accs_finos.append(float((pj == datos.y[datos.va]).mean()))   # = val_juntando
                W = red[0].weight.detach().numpy(); wt = (W ** 2).sum(); cuotas.append({b: float((W[:, a:z] ** 2).sum() / wt) for b, (a, z) in self.cortes(datos).items()})
                if s == self.semillas[0]:
                    self.logits_va = lg
            y = datos.etiquetas_finas()[datos.va] if self.sub4 else datos.y[datos.va]; self.preds = np.stack(preds); self.veces = (self.preds != y).sum(0); p0 = preds[0]; self.finos = np.stack(finos)
            et = lambda k: self.CLASES_FINAS[k] if self.sub4 else str(k)
            sub = None
            if self.sub4:                                         # cómo le va al 4 partido (semilla 0): cuántos hay y cómo se leen
                yf = datos.etiquetas_finas(); yv, pf = yf[datos.va], finos[0]; es4 = datos.y[datos.va] == 4
                cuenta = lambda real, lee: int(((yv == real) & (pf == lee)).sum())
                sub = {"val_juntando": round(float(np.mean(accs_finos)), 4), "val_juntando_rango": [round(min(accs_finos), 4), round(max(accs_finos), 4)], "fallos_juntando_s0": int((np.where(finos[0] == 10, 4, finos[0]) != datos.y[datos.va]).sum()), "train_4l": int((yf[idx_tr] == 10).sum() // len(datos.aum)), "train_4a": int((yf[idx_tr] == 4).sum() // len(datos.aum)), "val_4l": int((yv == 10).sum()), "val_4a": int((yv == 4).sum()),
                       "4l_leido_4l": cuenta(10, 10), "4l_leido_4a": cuenta(10, 4), "4l_leido_otro": int(((yv == 10) & (pf != 10) & (pf != 4)).sum()),
                       "4a_leido_4a": cuenta(4, 4), "4a_leido_4l": cuenta(4, 10), "4a_leido_otro": int(((yv == 4) & (pf != 10) & (pf != 4)).sum()),
                       "otros_leidos_4l": int((~es4 & (pf == 10)).sum()), "otros_leidos_4a": int((~es4 & (pf == 4)).sum())}
            conf = {}
            for j in np.flatnonzero(p0 != y):
                k = f"{et(y[j])}→{et(p0[j])}"; conf[k] = conf.get(k, 0) + 1
            self.segundos = time.time() - t
            self.resumen = {"nombre": self.nombre, "bloques": self.bloques, "celdas": self.celdas, "semillas": list(self.semillas), "n": int(self.n),
                            "val": round(float(np.mean(accs)), 4), "val_rango": [round(min(accs), 4), round(max(accs), 4)],
                            "fallos_s0": int((p0 != y).sum()), "fallan_en_todas": int((self.veces == len(self.semillas)).sum()),
                            "confusiones": dict(sorted(conf.items(), key=lambda z: -z[1])), "cuota": {b: round(float(np.mean([c[b] for c in cuotas])), 3) for b in self.bloques},
                            "segundos": round(self.segundos, 1), "sin_posicion": [b for b in self.bloques if pieza(b)[0] in SIN_POSICION], "sub4": self.sub4, "sub4_resumen": sub, "extra": self.extra, "n_train": int(len(idx_tr) // len(datos.aum))}
            self.estado = "listo"
        except Exception as e:                                    # noqa: BLE001
            self.estado = "error"; self.error = f"{type(e).__name__}: {e}"; traceback.print_exc()

    # --- pesos que sobreviven al dev (el dueño, 2026-10-16). `huella` dice CON QUÉ se entrenaron; guardar/restaurar no tocan Datos
    def huella(self, datos: Datos) -> dict:
        h = {"dataset": DATASET, "cresta": datos.hash, "etiquetas4": datos.hash_manuales(), "bloques": list(self.bloques), "celdas": self.celdas,
             "semillas": list(self.semillas), "sub4": self.sub4, "extra": self.extra, "oculta": OCULTA, "pasos": PASOS, "aum": list(AUM)}
        if any(pieza(b)[0] == "vertice_libre" for b in self.bloques):      # sólo quien lo usa: la huella de los demás no cambia
            h["vertice_libre"] = VLB.huella_modulo()
        return h

    def fichero(self, carpeta: Path) -> Path:
        limpio = "".join(ch if ch.isalnum() else "-" for ch in self.nombre.lower()).strip("-")[:40]
        return carpeta / f"{limpio}-{hashlib.sha1(self.nombre.encode()).hexdigest()[:8]}.pt"

    def guardar(self, carpeta: Path, huella: dict) -> Path:
        """Todo lo que `leer`/`lista` necesitan de un compositor LISTO: las 3 redes (state_dict + mu + sd), las predicciones de
        val, los logits de la semilla 0 y el resumen. Un `.pt` por compositor; se pisa (misma receta = mismo fichero)."""
        if self.estado != "listo":
            raise RuntimeError(f"«{self.nombre}» está {self.estado}: no hay nada que guardar")
        carpeta.mkdir(parents=True, exist_ok=True); f = self.fichero(carpeta)
        torch.save({"huella": huella, "nombre": self.nombre, "n": int(self.n), "segundos": self.segundos, "resumen": self.resumen,
                    "redes": [(red.state_dict(), np.asarray(mu), np.asarray(sd)) for red, mu, sd in self.redes],
                    "preds": self.preds, "veces": self.veces, "finos": self.finos, "logits_va": self.logits_va,
                    "guardado": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}, f)
        return f

    def restaurar(self, carpeta: Path, huella: dict) -> bool:
        """True si había pesos con EXACTAMENTE esta huella y quedan cargados (estado «listo», sin entrenar). Cualquier otra
        cosa —no hay fichero, huella distinta, fichero roto— es False y se entrena como siempre: nunca se sirve con pesos de
        otra receta, otro `cresta_curvas.py` u otras etiquetas."""
        f = self.fichero(carpeta)
        if not f.is_file():
            return False
        try:
            z = torch.load(f, weights_only=False, map_location="cpu")
        except Exception as e:                                    # noqa: BLE001
            print(f"  ⚠ {f.name} no se puede leer ({type(e).__name__}: {e}): se reentrena", flush=True); return False
        if z.get("huella") != huella:
            return False
        self.n = int(z["n"]); self.redes = []
        for sd_red, mu, sd in z["redes"]:
            red = torch.nn.Sequential(torch.nn.Linear(self.n, OCULTA), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(OCULTA, 11 if self.sub4 else 10))
            red.load_state_dict(sd_red); red.eval(); self.redes.append((red, mu, sd))
        self.preds, self.veces, self.finos, self.logits_va = z["preds"], z["veces"], z["finos"], z["logits_va"]
        self.segundos = float(z["segundos"]); self.resumen = {**z["resumen"], "restaurado": z.get("guardado", "?")}; self.estado = "listo"
        if self.resumen.get("emulado"):                           # web/emular.py: redes SIN entrenar para probar la restauración
            print(f"  ⚠ «{self.nombre}»: pesos EMULADOS ({self.resumen['emulado']}); no sirven para revisar. Para tener los de verdad: /api/reentrenar", flush=True)
        return True

    def leer(self, datos: Datos, i: int) -> dict:
        """Lo que recibe y decide el compositor para el dígito i (semilla 0): entradas por bloque, logits, atribución."""
        todos, _, _ = datos.features(self.celdas); x = self.corta(todos[i:i + 1], datos)[0]; red, mu, sd = self.redes[0]
        z = torch.from_numpy((x - mu) / sd).float()[None].requires_grad_(True); lg = red(z)[0]
        pf = int(lg.argmax()); pred = 4 if pf == 10 else pf; real = int(datos.y[i]); p = torch.softmax(lg, 0).detach().numpy()
        real_f = int(datos.etiquetas_finas()[i]) if self.sub4 else real   # con sub4 la explicación va en el espacio FINO (4l ≠ 4a)
        contra = real_f if real_f != pf else int(np.argsort(lg.detach().numpy())[-2])   # si acierta, se explica contra el segundo
        (lg[pf] - lg[contra]).backward(); atr = (z.grad[0] * z.detach()[0]).numpy()
        bloques = []
        for b, (a, zz) in self.cortes(datos).items():
            base, _, cols = columnas_de(b, self.celdas); vals = x[a:zz]; at = atr[a:zz]
            subs = []
            for sub in dict.fromkeys(c["sub"] for c in cols):
                idx = [k for k, c in enumerate(cols) if c["sub"] == sub]
                rej = [k for k in idx if cols[k]["celda"] is not None]; otros = [k for k in idx if cols[k]["celda"] is None]   # = tipo «rejilla» en los demás bloques
                lado = int(round(len(rej) ** 0.5)) if rej else 0
                subs.append({"sub": sub, "lado": lado, "rejilla": [float(vals[k]) for k in rej], "rejilla_atr": [float(at[k]) for k in rej],
                             "otros": [{"nombre": cols[k]["nombre"], "valor": float(vals[k]), "atr": float(at[k])} for k in otros]})
            et = BLOQUES[base][0] + ("" if pieza(b)[1] is None else f" — sólo {' + '.join(pieza(b)[1])}")
            bloques.append({"bloque": b, "etiqueta": et, "sin_posicion": base in SIN_POSICION, "subs": subs, "atr_total": float(at.sum())})
        nombres = [f"{BLOQUES[pieza(b)[0]][0].split(' (')[0]} · {c['sub']} · {c['nombre']}".replace(" ·  ·", " ·") for b, (a, zz) in self.cortes(datos).items() for c in columnas_de(b, self.celdas)[2]]
        # de qué bloque, sub y celda es cada columna: lo usa el popup de «qué pesa» para enseñar la rejilla con la celda
        # resaltada (el dueño, 2026-10-16: «al clickear una barra, la barra y el feature 3×3 con su posición resaltada»)
        meta = [{"bloque": b, "sub": c["sub"], "celda": c["celda"], "tipo": c["tipo"]} for b, (a, zz) in self.cortes(datos).items() for c in columnas_de(b, self.celdas)[2]]
        orden = np.argsort(-np.abs(atr))[:16]
        va_pos = int(np.flatnonzero(datos.va == i)[0]) if i in set(datos.va.tolist()) else None
        return {"idx": int(i), "real": real, "predicho": pred, "contra": contra, "p_pred": float(p[pf]), "p_real": float(p[real_f]),
                "predicho_fino": pf, "real_fino": real_f, "clases": self.CLASES_FINAS if self.sub4 else self.CLASES[:10], "sub4": self.sub4, "acierta": (pf == real_f) if self.sub4 else (pred == real),
                "logits": [float(v) for v in lg.detach().numpy()], "semillas_que_fallan": int(self.veces[va_pos]) if va_pos is not None else None,
                "semillas": len(self.semillas), "bloques": bloques, "pesan": [{"nombre": nombres[k], "valor": float(x[k]), "atr": float(atr[k]), **meta[k]} for k in orden],
                "suma_por_bloque": {b: float(atr[a:zz].sum()) for b, (a, zz) in self.cortes(datos).items()}}


def serializar_objetos(ob: dict) -> dict:
    """Los objetos de `cresta_curvas.objetos` con sus polilíneas, redondeados, y el cono de cada entrante."""
    out = {}
    for k, lista in ob.items():
        out[k] = []
        for o in lista:
            d = {kk: v for kk, v in o.items() if kk not in ("pts", "dir", "dirs")}
            if k == "cruces":                                     # una cruz lleva sus BRAZOS (polilíneas de distinto largo) y dos direcciones
                d["brazos_pts"] = [np.round(np.asarray(b, float), 2).tolist() for b in o.get("pts", [])]; d["pts"] = []
            else:
                d["pts"] = np.round(np.asarray(o["pts"], float), 2).tolist() if "pts" in o else []
            if "dir" in o:
                d["dir"] = np.round(np.asarray(o["dir"], float), 3).tolist(); d["cono"] = int(CC.cono(o))
            out[k].append(_limpio(d))
    return out


def archivados() -> list[str]:
    """Los features ARCHIVADOS en «entrenar otro» (el dueño, 2026-10-17): van a un contenedor plegado. Es sólo cómo se
    enseñan: no cambian ninguna receta. Local (esta unidad) > almacén (el dev anterior) > ninguno."""
    alm = almacen_revisor()
    for f in [CACHE / "archivados.json"] + ([alm / "archivados.json"] if alm else []):
        if f.is_file():
            try:
                return [b for b in json.loads(f.read_text()) if b in BLOQUES and b not in RETIRADOS]   # un retirado ya no se ofrece ni archivado
            except (json.JSONDecodeError, TypeError):
                pass
    return []


def archivar(bloque: str, si: bool) -> list[str]:
    """Archiva o desarchiva un feature; guarda la lista en local y en el almacén y la empuja. Se niega ante uno desconocido."""
    if bloque not in BLOQUES:
        raise ValueError(f"bloque desconocido: {bloque}")
    lst = [b for b in archivados() if b != bloque] + ([bloque] if si else [])
    texto = json.dumps(lst, ensure_ascii=False, indent=1); CACHE.mkdir(parents=True, exist_ok=True); (CACHE / "archivados.json").write_text(texto)
    alm = almacen_revisor()
    if alm:
        alm.mkdir(parents=True, exist_ok=True); (alm / "archivados.json").write_text(texto)
        empujar_almacen(f"rizo-2 revisor: {'archivado' if si else 'desarchivado'} el feature «{bloque}»")
    return lst


def borrar_pesos(comps) -> list[str]:
    """Quita del almacén el `.pt` de cada compositor dado (si lo hay). Devuelve los nombres quitados. NO empuja (lo hace quien
    llama, junto con las recetas). Nunca lanza."""
    alm = almacen_revisor(); out = []
    if alm is None:
        return out
    for c in comps:
        try:
            f = c.fichero(alm)
            if f.is_file():
                f.unlink(); out.append(f.name)
        except OSError as e:
            print(f"  ⚠ no pude quitar los pesos de «{c.nombre}»: {e}", flush=True)
    return out


def pesos_huerfanos(alm: Path, recetas: dict) -> list[Path]:
    """Los `.pt` del almacén que NO son de ninguna receta guardada: de compositores ya borrados. La identidad es el nombre de
    fichero que `Compositor.fichero` deriva del NOMBRE de la receta (sha1), no una lista escrita a mano (R16)."""
    vivos = {Compositor(n, ["rectas"]).fichero(alm).name for n in recetas}
    return sorted(f for f in alm.glob("*.pt") if f.name not in vivos)


def _json_ok(f: Path) -> bool:
    try:
        return f.is_file() and isinstance(json.loads(f.read_text()), dict)
    except (OSError, json.JSONDecodeError):
        return False


def limpiar_pesos(seco: bool) -> int:
    """`servidor.py limpiar-pesos [--seco]`: quita del almacén los pesos de compositores que ya no están en las recetas (los
    borrados antes del 2026-10-17, cuando borrar no tocaba los .pt) y empuja. Con --seco sólo los lista."""
    alm = almacen_revisor()
    if alm is None:
        print("✗ sin repo de datos (almacén): nada que limpiar"); return 2
    # Sólo con recetas LEÍDAS de un fichero: sin recetas.json (o roto) `recetas_guardadas` cae a la receta por defecto, y entonces
    # los pesos de las recetas de verdad parecerían huérfanos (lo señaló el agente revisor, 2026-10-17).
    if not any(_json_ok(f) for f in (CACHE / "recetas.json", alm / "recetas.json")):
        print("✗ no hay recetas.json legible (ni local ni en el almacén): no sé cuáles son las vivas, no quito nada"); return 2
    rec = recetas_guardadas(); h = pesos_huerfanos(alm, rec)
    print(f"recetas vivas: {', '.join(rec) or '(ninguna)'}\npesos huérfanos en {alm}: {len(h)}")
    for f in h:
        print(f"  {'(seco) ' if seco else ''}quito {f.name}")
    if seco or not h:
        return 0
    for f in h:
        f.unlink()
    empujar_almacen(f"rizo-2 revisor: quitados {len(h)} pesos de compositores ya borrados ({', '.join(f.name for f in h)})")
    return 0


def recetas_guardadas() -> dict:
    """Las recetas con las que arranca la app Y las que entrena la consola (`entrenar`): local (esta unidad) > almacén (el dev
    anterior) > las de partida."""
    # local (esta unidad) > almacén (el dev anterior) > las de partida. Así un dev nuevo arranca con las recetas que el dueño
    # dejó en el anterior, y con sus pesos si la huella coincide (2026-10-16).
    alm = almacen_revisor()
    for f in [CACHE / "recetas.json"] + ([alm / "recetas.json"] if alm else []):
        if f.is_file():
            try:
                return json.loads(f.read_text())
            except json.JSONDecodeError:
                pass
    # Desde el 2026-10-15 (el dueño: «elimina los compositores previos al 4a y 4l») el arranque de un dev nuevo trae SÓLO
    # recetas con el 4 en dos clases, con extra (sin extra, train tiene 0 cuatros con lazo). ⚠ La primera vez calcula las
    # features de extra (~7 min) y cada una tarda ~15 min.
    # Y desde el 2026-10-16 (tarde) el dueño quitó TODAS las que llevan `golpes` («no queremos usar golpes por default») y dejó
    # UNA: las aprobadas (rectas ∪ restos, arcos, lazo región, conos 60°) + el vértice (§ 95). Sin golpes.
    return {RECETA_DEFECTO: {"bloques": BLOQUES_DEFECTO, "celdas": 3, "semillas": list(SEMILLAS), "sub4": True, "extra": True}}


class App:
    def __init__(self):
        self.datos = Datos(); self.comps: dict[str, Compositor] = {}; self.cola = []; self._lock = threading.Lock()
        self.hilo = threading.Thread(target=self._bucle, daemon=True); self.hilo.start()
        for nombre, bl in self.recetas_guardadas().items():
            self.nuevo(nombre, bl["bloques"], bl.get("celdas", 3), bl.get("semillas", SEMILLAS), bl.get("sub4", False), bl.get("extra", False), existente=True)

    def recetas_guardadas(self) -> dict:
        return recetas_guardadas()

    def guardar_recetas(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        texto = json.dumps({c.nombre: {"bloques": c.bloques, "celdas": c.celdas, "semillas": list(c.semillas), "sub4": c.sub4, "extra": c.extra} for c in self.comps.values()}, ensure_ascii=False, indent=1)
        (CACHE / "recetas.json").write_text(texto)
        alm = almacen_revisor()
        if alm:                                                   # la copia que sobrevive al dev; se empuja con los pesos
            alm.mkdir(parents=True, exist_ok=True); (alm / "recetas.json").write_text(texto)

    def _persistir(self, c: Compositor):
        """Guarda los pesos de un compositor recién entrenado en el almacén y los empuja. NUNCA lanza: el aviso es una
        comodidad y la red sigue en memoria; lo que falle se dice en el log y se reintenta en el siguiente entrenamiento."""
        alm = almacen_revisor()
        if alm is None:
            print("  ⚠ sin repo de datos: los pesos NO sobreviven a rehacer el dev", flush=True); return
        try:
            f = c.guardar(alm, c.huella(self.datos)); self.guardar_recetas()
            print(f"  pesos → {f.name}", flush=True); empujar_almacen(f"rizo-2 revisor: pesos de «{c.nombre}» (val {c.resumen.get('val')})")
        except Exception as e:                                    # noqa: BLE001
            print(f"  ⚠ no pude guardar los pesos de «{c.nombre}»: {type(e).__name__}: {e}", flush=True)

    def nuevo(self, nombre: str, bloques: list[str], celdas=3, semillas=SEMILLAS, sub4=False, extra=False, existente=False) -> Compositor:
        if not bloques:
            raise ValueError("sin bloques")
        bloques = [normalizar(b) for b in bloques]                 # se niega ante un bloque o pieza desconocidos
        if not existente:                                         # uno guardado (al arrancar) carga aunque lleve un retirado
            comprobar_nuevo(bloques)
        if int(celdas) not in (3, 4):
            raise ValueError("celdas: 3 o 4")
        nombre = nombre.strip() or " + ".join(bloques) + (f" · {celdas}×{celdas}" if int(celdas) != 3 else "") + (" · 4 en dos" if sub4 else "") + (" · +extra" if extra else "")
        c = Compositor(nombre, bloques, celdas, semillas, sub4, extra); alm = almacen_revisor()
        restaurado = bool(alm) and c.restaurar(alm, c.huella(self.datos))   # del dev anterior, si la huella coincide: ni cola ni espera
        with self._lock:
            self.comps[nombre] = c
            if not restaurado:
                self.cola.append(c)
        self.guardar_recetas(); return c

    def borrar(self, nombre: str):
        with self._lock:
            c = self.comps.pop(nombre, None)
        self.guardar_recetas()
        # 2026-10-17: borrar escribía recetas.json en el almacén y NO lo empujaba: el freno lo veía como «sin empujar» y un
        # dev nuevo habría resucitado lo borrado. Y desde ese mismo día (el dueño: «si los borro es porque ya no son necesarios»)
        # se quitan también sus PESOS del almacén: cada dev nuevo los clonaba sin que nadie los fuera a usar. Quitarlos es un
        # commit, no reescribir historia: siguen recuperables con git (y el volumen no encoge por esto).
        quitados = borrar_pesos([c] if c is not None else [])
        empujar_almacen(f"rizo-2 revisor: borrado «{nombre}» (recetas" + (f" y sus pesos {', '.join(quitados)})" if quitados else ")"))

    def _bucle(self):
        while True:
            with self._lock:
                c = self.cola.pop(0) if self.cola else None
            if c is None:
                time.sleep(0.3); continue
            if self.comps.get(c.nombre) is not c:                 # borrado mientras esperaba en la cola: no se entrena (13–20 min con extra)
                continue
            if c.estado == "listo" and not getattr(c, "reentrenar", False):   # ya entrenado desde que se encoló («reentrenar» dos veces): no se repite
                continue
            c.reentrenar = False; c.entrenar(self.datos)
            if c.estado == "listo":
                self._persistir(c)

    def reentrenar(self, nombre: str | None = None) -> int:
        """Vuelve a la cola los compositores con el 4 en dos (todos), o SÓLO `nombre` (sea sub4 o no; 2026-10-16: para sustituir
        unos pesos emulados sin pagar los demás). Devuelve cuántos encoló. El que ya espera no se duplica; el que está entrenando
        se vuelve a encolar (sus etiquetas son las viejas) y se marca: aunque termine «listo» antes de que le toque, se repite."""
        if nombre is not None and nombre not in self.comps:
            raise KeyError(f"no hay compositor «{nombre}»")
        n = 0
        with self._lock:
            for c in list(self.comps.values()):
                if (c.nombre == nombre if nombre is not None else c.sub4) and c not in self.cola:
                    c.reentrenar = True
                    if c.estado != "entrenando":
                        c.estado = "pendiente"
                    self.cola.append(c); n += 1
        return n

    def listo(self, nombre: str) -> Compositor:
        c = self.comps.get(nombre)
        if c is None:
            raise KeyError(f"no hay compositor «{nombre}»")
        if c.estado != "listo":
            raise RuntimeError(f"«{nombre}» está {c.estado}" + (f": {c.error}" if c.error else ""))
        return c

    # --- lo que contesta
    def estado(self) -> dict:
        return {"compositores": [{"nombre": c.nombre, "estado": c.estado, "error": c.error, "bloques": c.bloques, "celdas": c.celdas, "sub4": c.sub4, "extra": c.extra, **c.resumen} for c in self.comps.values()],
                "bloques": [{"clave": k, "etiqueta": BLOQUES[k][0], "forma": BLOQUES[k][1], "sin_posicion": k in SIN_POSICION, "partes": partes(k)} for k in bloques_ofrecidos()],
                "bloques_defecto": BLOQUES_DEFECTO, "archivados": archivados(),                # lo que viene marcado en «entrenar otro» (el dueño, 2026-10-16: sólo las aprobadas + vértice, sin golpes)
                "n_val": int(len(self.datos.va)), "hash_cresta": self.datos.hash, "pendientes": len(self.cola)}

    def lista(self, nombre: str, modo="fallos", clase=None, conf=None, hasta=200, sub=None) -> dict:
        c = self.listo(nombre); d = self.datos; yf = d.etiquetas_finas()[d.va]; y = yf if c.sub4 else d.y[d.va]; p0 = c.preds[0]
        # el subgrupo del 4 es del DÍGITO (detector), exista o no sub4 en el compositor; con sub4, y y p0 están en las 11 clases
        fino = lambda e: 10 if e == "4l" else 4 if e == "4a" else int(e)
        if sub in ("l", "a"):
            clase = 4
        sel = np.ones(len(d.va), bool)
        if modo == "fallos":
            sel &= p0 != y
        elif modo == "todas":                                      # fallan en todas las semillas
            sel &= c.veces == len(c.semillas)
        elif modo == "aciertos":
            sel &= p0 == y
        if clase is not None:
            sel &= (d.y[d.va] == int(clase)) if not c.sub4 or int(clase) != 4 else ((y == 4) | (y == 10))
        if sub == "l":
            sel &= yf == 10
        elif sub == "a":
            sel &= (y == 4) & (yf == 4)
        if conf:
            a, b = conf.replace("->", "→").split("→"); sel &= (y == fino(a)) & (p0 == fino(b))
        ids = np.flatnonzero(sel); p = torch.softmax(torch.from_numpy(c.logits_va[ids]), 1).numpy(); pf = c.finos[0]
        filas = [{"idx": int(d.va[j]), "real": int(d.y[d.va][j]), "predicho": int(np.where(p0[j] == 10, 4, p0[j])), "fallo": bool(p0[j] != y[j]), "veces": int(c.veces[j]), "p": float(p[k, pf[j]]),
                  "sub": ("l" if yf[j] == 10 else "a") if y[j] == 4 else None, "predicho_fino": int(pf[j])} for k, j in enumerate(ids)]
        yc = d.y[d.va]; mal = p0 != y
        por_clase = {str(k): {"n": int((yc == k).sum()), "fallos": int((mal & (yc == k)).sum())} for k in range(10)}
        por_clase["4l"] = {"n": int((yf == 10).sum()), "fallos": int((mal & (yf == 10)).sum())}; por_clase["4a"] = {"n": int(((yc == 4) & (yf == 4)).sum()), "fallos": int((mal & (yc == 4) & (yf == 4)).sum())}
        return {"total": len(filas), "filas": filas[:hasta], "por_clase": por_clase, "confusiones": c.resumen.get("confusiones", {})}

    def digito(self, i: int) -> dict:
        d = self.datos; ob = dict(d.objetos(i)); ob["vertices"] = CC.vertices(d.tinta[i])   # el vértice sale de la TINTA (§ 95)
        return {"idx": int(i), "real": int(d.y[i]), "tinta": d.tinta[i].ravel().tolist(), "M": np.round(d.M[i], 2).ravel().tolist(),
                "objetos": serializar_objetos(ob), "en_val": bool(i in set(d.va.tolist())), "sub4": (d.subgrupo4(i) if d.y[i] == 4 else None)}

    def miniaturas(self, ids: list[int]) -> list[dict]:
        d = self.datos; out = []
        for i in ids[:60]:
            ob = dict(d.objetos(i)); ob["vertices"] = CC.vertices(d.tinta[i]); s = serializar_objetos(ob)
            out.append({"idx": int(i), "M": np.round(d.M[i], 2).ravel().tolist(), "objetos": {k: [{kk: v for kk, v in o.items() if kk in ("pts", "s", "o", "cx", "cy", "cono", "fuerza")} for o in s[k]] for k in ("rectas", "restos", "arcos", "lazos", "entrantes", "golpes", "vertices")}})
        return out

    def comparar(self, i: int, nombres: list[str]) -> dict:
        lecturas = {n: self.listo(n).leer(self.datos, i) for n in nombres}
        # qué bloques difieren entre compositores (misma clave y mismos valores → «idéntico»)
        firmas = {}
        for n, L in lecturas.items():
            for b in L["bloques"]:
                firmas.setdefault(b["bloque"], set()).add(json.dumps([s["rejilla"] + [o["valor"] for o in s["otros"]] for s in b["subs"]]))
        return {"idx": int(i), "real": int(self.datos.y[i]), "lecturas": lecturas, "bloques_que_difieren": [b for b, f in firmas.items() if len(f) > 1 or len([n for n, L in lecturas.items() if any(x["bloque"] == b for x in L["bloques"])]) < len(lecturas)]}


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    app: App = None; token: str = ""

    def log_message(self, fmt, *a):
        sys.stderr.write(f"{time.strftime('%H:%M:%S')} {fmt % a}\n")

    def _envia(self, code, tipo, cuerpo: bytes):
        self.send_response(code); self.send_header("Content-Type", tipo); self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(cuerpo)

    def _json(self, o, code=200):
        self._envia(code, "application/json; charset=utf-8", json.dumps(_limpio(o), ensure_ascii=False).encode())

    def _ruta(self):
        u = urlparse(self.path); q = parse_qs(u.query); pide = lambda k, d=None: q.get(k, [d])[0]
        return u, pide

    def do_GET(self):                                             # noqa: N802
        u, pide = self._ruta()
        if u.path == "/salud":                                    # sin token, a propósito
            return self._json({"ok": True, "compositores": len(self.app.comps)})
        if not secrets.compare_digest(str(pide("t", "")), self.token):
            return self._json({"error": "falta el token. Pídelo con `web/servidor.py url`"}, 403)
        try:
            if u.path == "/etiquetar":
                return self._envia(200, "text/html; charset=utf-8", (AQUI / "web" / "etiquetar.html").read_bytes())
            if u.path == "/api/cuatros":
                return self._json({"cuatros": self.app.datos.cuatros(), "M": {int(i): np.round(self.app.datos.M[i], 2).ravel().tolist() for i in np.flatnonzero(self.app.datos.y == 4)}})
            if u.path == "/api/marcar":
                self.app.datos.marcar(int(pide("idx")), pide("clase", "auto")); return self._json({"ok": True, "manuales": self.app.datos.manuales()})
            if u.path == "/api/reentrenar":                        # tras corregir etiquetas: vuelve a la cola todo compositor con el 4 en dos (~15 min cada uno con extra)
                return self._json({"ok": True, "en_cola": self.app.reentrenar(pide("nombre", "") or None)})   # `nombre`: uno solo (p. ej. para sustituir pesos emulados)
            if u.path == "/":
                return self._envia(200, "text/html; charset=utf-8", (AQUI / "web" / "index.html").read_bytes())
            if u.path == "/api/estado":
                return self._json(self.app.estado())
            if u.path == "/api/lista":
                return self._json(self.app.lista(pide("c"), pide("modo", "fallos"), pide("clase"), pide("conf"), int(pide("hasta", 200)), pide("sub")))
            if u.path == "/api/digito":
                return self._json(self.app.digito(int(pide("idx"))))
            if u.path == "/api/miniaturas":
                return self._json(self.app.miniaturas([int(x) for x in pide("ids", "").split(",") if x]))
            if u.path == "/api/leer":
                return self._json(self.app.listo(pide("c")).leer(self.app.datos, int(pide("idx"))))
            if u.path == "/api/comparar":
                return self._json(self.app.comparar(int(pide("idx")), [x for x in pide("cs", "").split("|") if x]))
            if u.path == "/api/nuevo":
                c = self.app.nuevo(pide("nombre", ""), [x for x in pide("bloques", "").split(",") if x], int(pide("celdas", 3)), tuple(range(int(pide("semillas", 3)))), pide("sub4", "0") in ("1", "true"), pide("extra", "0") in ("1", "true"))
                return self._json({"ok": True, "nombre": c.nombre, "estado": c.estado})
            if u.path == "/api/archivar":                          # ?bloque=…&si=1|0 (2026-10-17)
                return self._json({"ok": True, "archivados": archivar(pide("bloque", ""), pide("si", "1") in ("1", "true"))})
            if u.path == "/api/borrar":
                self.app.borrar(pide("nombre", "")); return self._json({"ok": True})
        except (KeyError, ValueError, RuntimeError) as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:                                    # noqa: BLE001
            traceback.print_exc(); return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        return self._json({"error": "no existe"}, 404)


# ---------------------------------------------------------------- máquina: token, url, estado, unidad, ufw (patrón de gauss-p)
def _ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("1.1.1.1", 80)); ip = s.getsockname()[0]; s.close(); return ip
    except OSError:
        return "127.0.0.1"


def _token_de_disco(crear: bool) -> str:
    """El token lo crea el servidor al arrancar y vive FUERA de los repos; `url` lo lee y si no existe lo dice."""
    p = Path(os.environ.get("REVISOR_TOKEN_FILE", Path.home() / ".cache" / f"{UNIDAD}.token"))
    if p.is_file():
        return p.read_text().strip()
    if not crear:
        return ""
    p.parent.mkdir(parents=True, exist_ok=True); t = secrets.token_urlsafe(18); p.write_text(t); p.chmod(0o600); return t


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:                                             # noqa: BLE001
        return ""


def estado(puerto: int) -> int:
    """¿Se llega desde el móvil? Se pregunta al SOCKET y al CORTAFUEGOS, no al disco (las cuatro puertas del 2026-09-15)."""
    print(f"\nrizo2-revisor · puerto {puerto}\n"); problemas = []
    activa = _sh(["systemctl", "is-active", UNIDAD]) or "(no instalada)"
    print(f"  unidad     {UNIDAD}: {activa}")
    for l in _sh(["systemctl", "show", UNIDAD, "-p", "Result", "-p", "NRestarts"]).splitlines():
        print(f"             {l}")
    if activa != "active":
        problemas.append(f"la unidad no está activa: sudo systemctl start {UNIDAD}")
    ss = _sh(["ss", "-ltn"]); linea = [l for l in ss.splitlines() if f":{puerto} " in l or l.endswith(f":{puerto}")]
    print(f"  socket     {linea[0].strip() if linea else 'NADA escuchando en ' + str(puerto)}")
    if not linea:
        problemas.append(f"nada escucha en el {puerto}")
    elif "127.0.0.1" in linea[0]:
        problemas.append("escucha SOLO en loopback: desde el móvil no se llega")
    ufw = _sh(["sudo", "-n", "ufw", "status"]); abierto = any(str(puerto) in l for l in ufw.splitlines()) if ufw else None
    print(f"  cortafuegos {'abierto' if abierto else 'CERRADO' if ufw else 'no se pudo mirar'} para el {puerto}")
    if ufw and not abierto:
        problemas.append(f"ufw no deja pasar el {puerto}: sudo ufw allow {puerto}/tcp   (o `web/servidor.py abrir`)")
    if not ufw:
        problemas.append("no pude leer `ufw status`: NO sé si se llega desde fuera")
    t = _token_de_disco(crear=False); print(f"  token      {'sí' if t else 'NO'}")
    if not t:
        problemas.append("no hay token: la app no ha arrancado nunca aquí")
    salud = _sh(["curl", "-s", "-m", "5", f"http://127.0.0.1:{puerto}/salud"])
    print(f"  salud      {salud or 'no contesta'}")
    if linea and not salud:
        problemas.append("el puerto está atado pero /salud no contesta: ¿sigue cargando? mira `journalctl -u " + UNIDAD + "`")
    print()
    if problemas:
        print("✗ desde el móvil NO se llega:"); [print(f"    {x}") for x in problemas]; print(); return 1
    print(f"ok  http://{_ip()}:{puerto}/?t={t}\n\n  ⚠ Lo único que esto NO puede comprobar es la red de en medio: la prueba de verdad es abrir esa URL en el móvil.\n")
    return 0


def cortafuegos(puerto: int, abrir: bool) -> int:
    cmd = ["sudo", "-n", "ufw", *("allow" if abrir else "delete allow").split(), f"{puerto}/tcp"]; print("  " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60); print((r.stdout + r.stderr).strip()); return r.returncode


UNIT = """[Unit]
Description=rizo2-revisor (revisor de dígitos de rizo-2, experimentos-cnn)
After=network-online.target
Wants=network-online.target
# ⚠ EN [Unit], NO EN [Service] (ahí systemd las ignora). Ventana alcanzable: con RestartSec=5 caben 5 arranques en 300 s.
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
User={usuario}
WorkingDirectory={repo}
ExecStart=/bin/bash -lc 'exec .venv/bin/python {rel} --puerto {puerto}'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""


def instalar(puerto: int) -> int:
    """Unidad + ufw, idempotente. Hace falta como comando porque el dev es efímero y `experimentos-cnn` no está en los repos
    que clona un dev nuevo: al rehacer la máquina esto no existe."""
    import getpass
    repo = AQUI.parent; unit = UNIT.format(usuario=getpass.getuser(), repo=repo, rel=Path(__file__).resolve().relative_to(repo), puerto=puerto)
    destino = f"/etc/systemd/system/{UNIDAD}.service"; print(f"\n  escribiendo {destino}")
    r = subprocess.run(["sudo", "-n", "tee", destino], input=unit, text=True, capture_output=True, timeout=60)
    if r.returncode != 0:
        print(f"✗ no pude escribir la unidad: {r.stderr.strip()}\n  → hace falta sudo. Contenido:\n\n{unit}"); return 1
    for cmd in (["sudo", "-n", "systemctl", "daemon-reload"], ["sudo", "-n", "systemctl", "enable", "--now", UNIDAD], ["sudo", "-n", "systemctl", "restart", UNIDAD]):
        print("  " + " ".join(cmd[2:])); subprocess.run(cmd, capture_output=True, timeout=120)
    cortafuegos(puerto, True)                                     # en el MISMO paso (cuarta puerta del 2026-09-15)
    for _ in range(180):                                          # se espera al SOCKET, no a is-active: la carga tarda (caract ~30 s la primera vez)
        if any(f":{puerto} " in l or l.endswith(f":{puerto}") for l in _sh(["ss", "-ltn"]).splitlines()):
            break
        time.sleep(1)
    print(); return estado(puerto)


def servir(puerto: int, host: str = "0.0.0.0") -> int:
    torch.set_num_threads(2); t = time.time(); print("rizo2-revisor · cargando dataset y delgado…", flush=True)
    Handler.app = App(); Handler.token = _token_de_disco(crear=True)
    rest = sum(c.estado == "listo" for c in Handler.app.comps.values())
    print(f"  {len(Handler.app.datos.va)} dígitos de val · {len(Handler.app.comps)} compositor(es): {rest} restaurado(s) del almacén, {len(Handler.app.comps) - rest} en cola ({time.time() - t:.0f} s)", flush=True)
    print(f"\n  http://{_ip()}:{puerto}/?t={Handler.token}\n", flush=True)
    srv = ThreadingHTTPServer((host, puerto), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nadiós")
    return 0


def entrenar_consola(que: list[str], seco: bool) -> int:
    """`servidor.py entrenar <qué…> [--seco]`: entrena UNA o varias recetas por consola —sin unidad ni puerto— y deja cada `.pt`
    en el almacén, commiteado y empujado, igual que la app. Es lo que corre cada server del entrenamiento EN PARALELO
    (web/entrenar_paralelo.sh, el dueño, 2026-10-16: un server por receta en vez de ~15 min cada una en serie en el dev).
    `que` es una lista de: `emulados` (las que tienen pesos de web/emular.py), `todas`, el ID de 8 hex del nombre del `.pt`,
    o el nombre exacto. Se NIEGA ante un nombre que no conozca (no adivina). `--seco`: dice qué entrenaría y sale."""
    rec = recetas_guardadas(); alm = almacen_revisor()
    ids = {hashlib.sha1(n.encode()).hexdigest()[:8]: n for n in rec}
    elegidas = []
    for q in que:
        if q == "todas":
            elegidas += list(rec)
        elif q == "emulados":
            if alm is None:
                print("✗ `emulados` necesita el repo de datos (almacén), y no está clonado"); return 2
            for n, bl in rec.items():
                c = Compositor(n, bl["bloques"], bl.get("celdas", 3), bl.get("semillas", SEMILLAS), bl.get("sub4", False), bl.get("extra", False))
                f = c.fichero(alm)
                try:
                    z = torch.load(f, weights_only=False, map_location="cpu") if f.is_file() else None
                except Exception:                                 # noqa: BLE001
                    z = None
                if z is None or z.get("resumen", {}).get("emulado"):   # sin pesos, rotos o emulados: hay que entrenarla
                    elegidas.append(n)
        elif q in ids:
            elegidas.append(ids[q])
        elif q in rec:
            elegidas.append(q)
        else:
            print(f"\n✗ no conozco la receta «{q}». Las que hay (ID · nombre):"); [print(f"  {i} · {n}") for i, n in ids.items()]; return 2
    elegidas = list(dict.fromkeys(elegidas))
    if not elegidas:
        print("nada que entrenar (ninguna receta elegida)"); return 0
    print(("🧪 SECO — no entreno nada. " if seco else "") + f"{len(elegidas)} receta(s)" + ("" if alm else " ⚠ SIN almacén: los pesos no sobrevivirán"))
    for n in elegidas:
        print(f"  {hashlib.sha1(n.encode()).hexdigest()[:8]} · {n}")
    if seco:
        return 0
    datos = Datos(); mal = 0
    for n in elegidas:
        bl = rec[n]; c = Compositor(n, bl["bloques"], bl.get("celdas", 3), bl.get("semillas", SEMILLAS), bl.get("sub4", False), bl.get("extra", False))
        print(f"\n==== {time.strftime('%H:%M:%S', time.gmtime())} UTC · entrenando «{n}»", flush=True); c.entrenar(datos)
        if c.estado != "listo":
            print(f"  ✗ «{n}»: {c.error}", flush=True); mal += 1; continue
        print(f"  val {c.resumen['val']} · {c.resumen['segundos']} s", flush=True)
        if alm is not None:
            f = c.guardar(alm, c.huella(datos)); (alm / "recetas.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
            print(f"  pesos → {f.name}", flush=True); empujar_almacen(f"rizo-2 revisor: pesos de «{n}» (val {c.resumen.get('val')})")
    return 1 if mal else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--puerto", type=int, default=int(os.environ.get("REVISOR_PUERTO", PUERTO))); ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--seco", action="store_true", help="con `entrenar`: dice qué entrenaría y sale")
    ap.add_argument("modo", nargs="*", default=[], help="url · estado · instalar · abrir · cerrar · entrenar <emulados|todas|id|nombre…> [--seco] · limpiar-pesos [--seco] · (vacío: servir)")
    a = ap.parse_args()
    modo = " ".join(a.modo).strip().lower()                       # se fija AQUÍ y no se relee (2026-09-08)
    print(f"rizo2-revisor: modo `{modo or 'servir'}`", file=sys.stderr)
    if modo == "estado":
        return estado(a.puerto)
    if modo == "instalar":
        return instalar(a.puerto)
    if modo in ("abrir", "cerrar"):
        return cortafuegos(a.puerto, modo == "abrir")
    if a.modo and a.modo[0].lower() == "entrenar":
        return entrenar_consola(a.modo[1:], a.seco)
    if modo == "limpiar-pesos":
        return limpiar_pesos(a.seco)
    if modo == "url":
        t = _token_de_disco(crear=False)
        if not t:
            print(f"✗ no hay token: la app no se ha levantado nunca en esta máquina.\n  → `web/servidor.py instalar`  o  sudo systemctl start {UNIDAD}"); return 1
        if estado(a.puerto) != 0:
            return 1
        print(f"http://{_ip()}:{a.puerto}/?t={t}"); print(f"etiquetar los 4: http://{_ip()}:{a.puerto}/etiquetar?t={t}"); return 0
    if modo not in ("", "servir"):                                # se NIEGA, no adivina
        print(f"\n✗ no sé qué es «{modo}».\n  url · estado · instalar · abrir · cerrar · entrenar … · limpiar-pesos · (nada: levantarla)\n"); return 2
    return servir(a.puerto, a.host)


if __name__ == "__main__":
    sys.exit(main())
