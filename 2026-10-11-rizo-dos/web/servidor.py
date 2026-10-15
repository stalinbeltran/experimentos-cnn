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
from expcnn import exigir_dataset                                 # noqa: E402

PUERTO = 8050
UNIDAD = "rizo2-revisor"
DATASET = "uci-optdigits-orig-32px-r20261005"                    # el de rizo-2 (nn/evaluar.py)
DELGADOS = AQUI.parent / "2026-10-15-delgado-cnn" / "resultados" / "delgados-k3x3.npz"
CACHE = AQUI / "web" / "cache"
AUM = (-3, -2, -1, 1, 2, 3)                                       # resistencia.AUM: 13 copias por dígito de train
SEMILLAS = (0, 1, 2)

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
}
SIN_POSICION = {"entrantes4", "entrantes4n"}   # regla del dueño (2026-10-15): se DICE cuál no lleva posición

RECETAS_BASE = {   # las corridas de 144 del README, por nombre: lo que un compositor «es» son sus bloques
    "aprobada: lazo región (§ 79)": ["rectas_todo", "arcos", "golpes", "lazos_region", "entrantes60"],
    "restos fundido, lazo presencia (§ 74)": ["rectas_todo", "arcos", "golpes", "lazos", "entrantes60"],
    "lazo área (§ 79)": ["rectas_todo", "arcos", "golpes", "lazos_area", "entrantes60"],
    "conos 60° sin restos (§ 66)": ["rectas", "arcos", "golpes", "lazos", "entrantes60"],
}
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
    elif bloque in ("golpes", "lazos", "lazos_region", "lazos_area"):
        out += [{"sub": "", "celda": k, "tipo": "rejilla", "nombre": cn[k]} for k in range(c)] + [{"sub": "", "celda": None, "tipo": "cuenta", "nombre": "cuenta"}]
    elif bloque in ("entrantes4", "entrantes4n"):
        out += [{"sub": DIR4[o], "celda": None, "tipo": "fuerza" if bloque == "entrantes4" else "cuenta", "nombre": "fuerza" if bloque == "entrantes4" else "cuenta"} for o in range(4)]
    elif bloque in ("entrantes60", "entrantes60f"):
        c3 = celdas_nombres(3)
        for o in range(4):
            out += [{"sub": DIR4[o], "celda": k, "tipo": "rejilla", "nombre": c3[k]} for k in range(9)]
    elif bloque == "cruces":
        c3 = celdas_nombres(3)
        out += [{"sub": "", "celda": k, "tipo": "rejilla", "nombre": c3[k]} for k in range(9)] + [{"sub": "", "celda": None, "tipo": "cuenta", "nombre": "cuenta"}]
    return out


def mover(a, dx=0, dy=0):
    """= evaluar.mover (relleno de fondo, nunca np.roll)."""
    o = np.zeros_like(a); H, W = a.shape
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    o[yd, xd] = a[ys, xs]; return o


def entrenar(xtr, ytr, sem, oculta=256):
    """= rastrear.entrenar, copiado para no arrastrar sus imports (matplotlib, los bocetos de Gabor…). `probar_revisor.py`
    comprueba que da los MISMOS logits que el original con la misma semilla."""
    torch.manual_seed(sem); mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    red = torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], oculta), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(oculta, 10))
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
        self.tr, self.va = np.flatnonzero(part == "train"), np.flatnonzero(part == "val")
        self.aum = [(0, 0)] + [(k, 0) for k in AUM] + [(0, k) for k in AUM]
        self.hash = hashlib.sha1((AQUI / "nn" / "cresta_curvas.py").read_bytes()).hexdigest()[:10]
        self.X = {}                                               # celdas → (X_todos (5620,394), X_tr_aum (180·13,394), Y_tr_aum)
        self._ob = {}; self._lock = threading.Lock()

    def features(self, celdas: int):
        if celdas in self.X:
            return self.X[celdas]
        CACHE.mkdir(parents=True, exist_ok=True); f = CACHE / f"caract-{self.hash}-{celdas}x{celdas}.npz"
        if f.is_file():
            z = np.load(f); self.X[celdas] = (z["todos"], z["tr"], z["ytr"]); return self.X[celdas]
        t = time.time(); print(f"  calculando caract {celdas}×{celdas} de {len(self.M)} + {len(self.tr) * 12} dígitos…", flush=True)
        todos = CC.caract(self.M, celdas)
        tr = np.concatenate([todos[self.tr]] + [CC.caract(np.stack([mover(x, *m) for x in self.M[self.tr]]), celdas) for m in self.aum[1:]])
        ytr = np.concatenate([self.y[self.tr]] * len(self.aum))
        np.savez_compressed(f, todos=todos, tr=tr, ytr=ytr); self.X[celdas] = (todos, tr, ytr)
        print(f"  listo en {time.time() - t:.0f} s → {f.name}", flush=True); return self.X[celdas]

    def objetos(self, i: int) -> dict:
        with self._lock:
            if i not in self._ob:
                self._ob[i] = CC.objetos(self.M[i])
            return self._ob[i]


class Compositor:
    """Una receta (bloques + celdas + semillas) entrenada: redes, predicciones de val, fallos, confusiones y cuota."""

    def __init__(self, nombre: str, bloques: list[str], celdas: int = 3, semillas=SEMILLAS):
        self.nombre, self.bloques, self.celdas, self.semillas = nombre, list(bloques), int(celdas), tuple(semillas)
        self.estado = "pendiente"; self.error = ""; self.redes = []; self.segundos = 0.0; self.resumen = {}

    def cortes(self, datos: Datos) -> dict:
        B = CC.bloques(self.celdas); n = 0; out = {}
        for b in self.bloques:
            k = B[b][1] - B[b][0]; out[b] = (n, n + k); n += k
        return out

    def corta(self, X, datos):
        B = CC.bloques(self.celdas)
        return np.concatenate([X[:, B[b][0]:B[b][1]] for b in self.bloques], 1)

    def entrenar(self, datos: Datos):
        try:
            self.estado = "entrenando"; t = time.time()
            todos, Xtr, Ytr = datos.features(self.celdas); Xtr, Xva = self.corta(Xtr, datos), self.corta(todos[datos.va], datos)
            self.n = Xtr.shape[1]; self.redes = []; preds = []; accs = []; cuotas = []
            for s in self.semillas:
                red, mu, sd = entrenar(Xtr, Ytr, s); self.redes.append((red, mu, sd))
                with torch.no_grad():
                    lg = red(torch.from_numpy((Xva - mu) / sd).float()).numpy()
                p = lg.argmax(1); preds.append(p); accs.append(float((p == datos.y[datos.va]).mean()))
                W = red[0].weight.detach().numpy(); wt = (W ** 2).sum(); cuotas.append({b: float((W[:, a:z] ** 2).sum() / wt) for b, (a, z) in self.cortes(datos).items()})
                if s == self.semillas[0]:
                    self.logits_va = lg
            y = datos.y[datos.va]; self.preds = np.stack(preds); self.veces = (self.preds != y).sum(0); p0 = preds[0]
            conf = {}
            for j in np.flatnonzero(p0 != y):
                k = f"{y[j]}→{p0[j]}"; conf[k] = conf.get(k, 0) + 1
            self.segundos = time.time() - t
            self.resumen = {"nombre": self.nombre, "bloques": self.bloques, "celdas": self.celdas, "semillas": list(self.semillas), "n": int(self.n),
                            "val": round(float(np.mean(accs)), 4), "val_rango": [round(min(accs), 4), round(max(accs), 4)],
                            "fallos_s0": int((p0 != y).sum()), "fallan_en_todas": int((self.veces == len(self.semillas)).sum()),
                            "confusiones": dict(sorted(conf.items(), key=lambda z: -z[1])), "cuota": {b: round(float(np.mean([c[b] for c in cuotas])), 3) for b in self.bloques},
                            "segundos": round(self.segundos, 1), "sin_posicion": [b for b in self.bloques if b in SIN_POSICION]}
            self.estado = "listo"
        except Exception as e:                                    # noqa: BLE001
            self.estado = "error"; self.error = f"{type(e).__name__}: {e}"; traceback.print_exc()

    def leer(self, datos: Datos, i: int) -> dict:
        """Lo que recibe y decide el compositor para el dígito i (semilla 0): entradas por bloque, logits, atribución."""
        todos, _, _ = datos.features(self.celdas); x = self.corta(todos[i:i + 1], datos)[0]; red, mu, sd = self.redes[0]
        z = torch.from_numpy((x - mu) / sd).float()[None].requires_grad_(True); lg = red(z)[0]
        pred = int(lg.argmax()); real = int(datos.y[i]); p = torch.softmax(lg, 0).detach().numpy()
        contra = real if real != pred else int(np.argsort(lg.detach().numpy())[-2])   # si acierta, se explica contra el segundo
        (lg[pred] - lg[contra]).backward(); atr = (z.grad[0] * z.detach()[0]).numpy()
        bloques = []
        for b, (a, zz) in self.cortes(datos).items():
            cols = columnas(b, self.celdas); vals = x[a:zz]; at = atr[a:zz]
            subs = []
            for sub in dict.fromkeys(c["sub"] for c in cols):
                idx = [k for k, c in enumerate(cols) if c["sub"] == sub]
                rej = [k for k in idx if cols[k]["tipo"] == "rejilla"]; otros = [k for k in idx if cols[k]["tipo"] != "rejilla"]
                lado = int(round(len(rej) ** 0.5)) if rej else 0
                subs.append({"sub": sub, "lado": lado, "rejilla": [float(vals[k]) for k in rej], "rejilla_atr": [float(at[k]) for k in rej],
                             "otros": [{"nombre": cols[k]["nombre"], "valor": float(vals[k]), "atr": float(at[k])} for k in otros]})
            bloques.append({"bloque": b, "etiqueta": BLOQUES[b][0], "sin_posicion": b in SIN_POSICION, "subs": subs, "atr_total": float(at.sum())})
        nombres = [f"{BLOQUES[b][0].split(' (')[0]} · {c['sub']} · {c['nombre']}".replace(" ·  ·", " ·") for b, (a, zz) in self.cortes(datos).items() for c in columnas(b, self.celdas)]
        orden = np.argsort(-np.abs(atr))[:16]
        va_pos = int(np.flatnonzero(datos.va == i)[0]) if i in set(datos.va.tolist()) else None
        return {"idx": int(i), "real": real, "predicho": pred, "contra": contra, "p_pred": float(p[pred]), "p_real": float(p[real]),
                "logits": [float(v) for v in lg.detach().numpy()], "semillas_que_fallan": int(self.veces[va_pos]) if va_pos is not None else None,
                "semillas": len(self.semillas), "bloques": bloques, "pesan": [{"nombre": nombres[k], "valor": float(x[k]), "atr": float(atr[k])} for k in orden],
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


class App:
    def __init__(self):
        self.datos = Datos(); self.comps: dict[str, Compositor] = {}; self.cola = []; self._lock = threading.Lock()
        self.hilo = threading.Thread(target=self._bucle, daemon=True); self.hilo.start()
        for nombre, bl in self.recetas_guardadas().items():
            self.nuevo(nombre, bl["bloques"], bl.get("celdas", 3), bl.get("semillas", SEMILLAS))

    # --- recetas: se guardan para sobrevivir al reinicio de la unidad (se reentrenan al arrancar, segundos)
    def recetas_guardadas(self) -> dict:
        f = CACHE / "recetas.json"
        if f.is_file():
            try:
                return json.loads(f.read_text())
            except json.JSONDecodeError:
                pass
        return {k: {"bloques": v, "celdas": 3, "semillas": list(SEMILLAS)} for k, v in RECETAS_BASE.items()}

    def guardar_recetas(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        (CACHE / "recetas.json").write_text(json.dumps({c.nombre: {"bloques": c.bloques, "celdas": c.celdas, "semillas": list(c.semillas)} for c in self.comps.values()}, ensure_ascii=False, indent=1))

    def nuevo(self, nombre: str, bloques: list[str], celdas=3, semillas=SEMILLAS) -> Compositor:
        malos = [b for b in bloques if b not in BLOQUES]
        if malos or not bloques:
            raise ValueError(f"bloques desconocidos: {malos}" if malos else "sin bloques")
        if int(celdas) not in (3, 4):
            raise ValueError("celdas: 3 o 4")
        nombre = nombre.strip() or " + ".join(bloques) + (f" · {celdas}×{celdas}" if int(celdas) != 3 else "")
        c = Compositor(nombre, bloques, celdas, semillas)
        with self._lock:
            self.comps[nombre] = c; self.cola.append(c)
        self.guardar_recetas(); return c

    def borrar(self, nombre: str):
        with self._lock:
            self.comps.pop(nombre, None)
        self.guardar_recetas()

    def _bucle(self):
        while True:
            with self._lock:
                c = self.cola.pop(0) if self.cola else None
            if c is None:
                time.sleep(0.3); continue
            c.entrenar(self.datos)

    def listo(self, nombre: str) -> Compositor:
        c = self.comps.get(nombre)
        if c is None:
            raise KeyError(f"no hay compositor «{nombre}»")
        if c.estado != "listo":
            raise RuntimeError(f"«{nombre}» está {c.estado}" + (f": {c.error}" if c.error else ""))
        return c

    # --- lo que contesta
    def estado(self) -> dict:
        return {"compositores": [{"nombre": c.nombre, "estado": c.estado, "error": c.error, "bloques": c.bloques, "celdas": c.celdas, **c.resumen} for c in self.comps.values()],
                "bloques": [{"clave": k, "etiqueta": v[0], "forma": v[1], "sin_posicion": k in SIN_POSICION} for k, v in BLOQUES.items()],
                "n_val": int(len(self.datos.va)), "hash_cresta": self.datos.hash, "pendientes": len(self.cola)}

    def lista(self, nombre: str, modo="fallos", clase=None, conf=None, hasta=200) -> dict:
        c = self.listo(nombre); d = self.datos; y = d.y[d.va]; p0 = c.preds[0]
        sel = np.ones(len(d.va), bool)
        if modo == "fallos":
            sel &= p0 != y
        elif modo == "todas":                                      # fallan en todas las semillas
            sel &= c.veces == len(c.semillas)
        elif modo == "aciertos":
            sel &= p0 == y
        if clase is not None:
            sel &= y == int(clase)
        if conf:
            a, b = conf.replace("->", "→").split("→"); sel &= (y == int(a)) & (p0 == int(b))
        ids = np.flatnonzero(sel); p = torch.softmax(torch.from_numpy(c.logits_va[ids]), 1).numpy()
        filas = [{"idx": int(d.va[j]), "real": int(y[j]), "predicho": int(p0[j]), "veces": int(c.veces[j]), "p": float(p[k, p0[j]])} for k, j in enumerate(ids)]
        por_clase = {str(k): {"n": int((y == k).sum()), "fallos": int(((p0 != y) & (y == k)).sum())} for k in range(10)}
        return {"total": len(filas), "filas": filas[:hasta], "por_clase": por_clase, "confusiones": c.resumen.get("confusiones", {})}

    def digito(self, i: int) -> dict:
        d = self.datos; ob = d.objetos(i)
        return {"idx": int(i), "real": int(d.y[i]), "tinta": d.tinta[i].ravel().tolist(), "M": np.round(d.M[i], 2).ravel().tolist(),
                "objetos": serializar_objetos(ob), "en_val": bool(i in set(d.va.tolist()))}

    def miniaturas(self, ids: list[int]) -> list[dict]:
        d = self.datos; out = []
        for i in ids[:60]:
            ob = d.objetos(i); s = serializar_objetos(ob)
            out.append({"idx": int(i), "M": np.round(d.M[i], 2).ravel().tolist(), "objetos": {k: [{kk: v for kk, v in o.items() if kk in ("pts", "s", "o", "cx", "cy", "cono", "fuerza")} for o in s[k]] for k in ("rectas", "restos", "arcos", "lazos", "entrantes", "golpes")}})
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
            if u.path == "/":
                return self._envia(200, "text/html; charset=utf-8", (AQUI / "web" / "index.html").read_bytes())
            if u.path == "/api/estado":
                return self._json(self.app.estado())
            if u.path == "/api/lista":
                return self._json(self.app.lista(pide("c"), pide("modo", "fallos"), pide("clase"), pide("conf"), int(pide("hasta", 200))))
            if u.path == "/api/digito":
                return self._json(self.app.digito(int(pide("idx"))))
            if u.path == "/api/miniaturas":
                return self._json(self.app.miniaturas([int(x) for x in pide("ids", "").split(",") if x]))
            if u.path == "/api/leer":
                return self._json(self.app.listo(pide("c")).leer(self.app.datos, int(pide("idx"))))
            if u.path == "/api/comparar":
                return self._json(self.app.comparar(int(pide("idx")), [x for x in pide("cs", "").split("|") if x]))
            if u.path == "/api/nuevo":
                c = self.app.nuevo(pide("nombre", ""), [x for x in pide("bloques", "").split(",") if x], int(pide("celdas", 3)), tuple(range(int(pide("semillas", 3)))))
                return self._json({"ok": True, "nombre": c.nombre, "estado": c.estado})
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
    print(f"  {len(Handler.app.datos.va)} dígitos de val · {len(Handler.app.comps)} compositor(es) en cola ({time.time() - t:.0f} s)", flush=True)
    print(f"\n  http://{_ip()}:{puerto}/?t={Handler.token}\n", flush=True)
    srv = ThreadingHTTPServer((host, puerto), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nadiós")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--puerto", type=int, default=int(os.environ.get("REVISOR_PUERTO", PUERTO))); ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("modo", nargs="*", default=[], help="url · estado · instalar · abrir · cerrar · (vacío: servir)")
    a = ap.parse_args()
    modo = " ".join(a.modo).strip().lower()                       # se fija AQUÍ y no se relee (2026-09-08)
    print(f"rizo2-revisor: modo `{modo or 'servir'}`", file=sys.stderr)
    if modo == "estado":
        return estado(a.puerto)
    if modo == "instalar":
        return instalar(a.puerto)
    if modo in ("abrir", "cerrar"):
        return cortafuegos(a.puerto, modo == "abrir")
    if modo == "url":
        t = _token_de_disco(crear=False)
        if not t:
            print(f"✗ no hay token: la app no se ha levantado nunca en esta máquina.\n  → `web/servidor.py instalar`  o  sudo systemctl start {UNIDAD}"); return 1
        if estado(a.puerto) != 0:
            return 1
        print(f"http://{_ip()}:{a.puerto}/?t={t}"); return 0
    if modo not in ("", "servir"):                                # se NIEGA, no adivina
        print(f"\n✗ no sé qué es «{modo}».\n  url · estado · instalar · abrir · cerrar · (nada: levantarla)\n"); return 2
    return servir(a.puerto, a.host)


if __name__ == "__main__":
    sys.exit(main())
