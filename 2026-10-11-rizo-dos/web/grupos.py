"""rizo-2 · revisor — SEPARAR un dígito en dos grupos, a mano sobre una regla (el dueño, 2026-10-17: «separemos los 1 en 2 grupos:
los que son sólo una vertical única y los otros… 1a y 1l, donde el 1l es sólo la recta vertical (pues también parece una l
minúscula). Para los 4 creamos una pantalla separadora donde yo las reviso. Crea una para el 1»).

Los grupos son DATO (`GRUPOS`): qué dígito, sus dos clases, el fichero de las decisiones a mano y la regla automática de partida.
La pantalla es `/separar?d=1` (`separar.html`); las decisiones van a `web/etiquetas-1.json` (en git, como las del 4: un dato que
no se re-deriva, R9) y se copian al almacén en cada marca, para que una revisión larga no muera con el dev.

⚠ Lo que esto NO hace, a propósito (agente arquitecto, 2026-10-17): el 4 sigue en `/etiquetar` y `etiquetas-4.json`, que entra en
la HUELLA de los compositores (`Datos.hash_manuales`). Este módulo no toca ese camino: separar los 1 no cambia ninguna huella ni
ninguna etiqueta fina, y no alimenta ningún entrenamiento. El paso «1l como clase del compositor» es otro, y obliga a reentrenar.
⚠ La regla va AQUÍ y no en `nn/cresta_curvas.py`: el hash de ese fichero entra en la huella de todos los compositores.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parents[1]                        # 2026-10-11-rizo-dos

# La regla del 1l, escrita ANTES de mirar a mano (R13), con sus umbrales aquí (R14). Sobre la CRESTA del DELGADO, nunca la tinta.
ESPOLON = 4.0        # px: un tramo más corto es un espolón del esqueleto, no un trazo
RECTA_MAX = 2.0      # px: lo más que se aparta la cresta de su recta (cuerda de ajuste) para ser «una recta»
INCL_MAX = 25.0      # grados desde la vertical. ⚠ El dueño dijo «la recta VERTICAL»: si un palo recto inclinado no es 1l, se baja
# Medido el 2026-10-17 con estos umbrales sobre los 571 unos (18 train · 164 val · 389 extra): 76 salen 1l. Con RECTA_MAX 1,5 →
# 57, 2,5 → 90, 3,0 → 107. ⚠ Un 1 grueso que el delgado deja con DOBLE cresta (el fallo de entrada de § 49–50) sale 1a: la mano.
# ⚠ Y una BANDERA CORTA se pierde: el esqueleto se come la esquina y deja de la bandera un tramo < ESPOLON, así que el 1 sale 1l.
# Medido con trazos sintéticos el 2026-10-17: bandera de 8,5 px dibujados → tramo de 2,8 px → 1l; de 10 px → 4,2 px → 1a. La mano.


def regla_1(M) -> str:
    """«1l» si la cresta del delgado M, sin espolones, es UN solo trazo, sin lazos, recto (≤ RECTA_MAX px de su recta) y casi
    vertical (≤ INCL_MAX°); si no, «1a»."""
    import cresta_curvas as CC                                    # perezoso: el módulo se puede importar sin el resto
    tr, puros = CC.tramos(np.asarray(M, np.float32))
    largos = [t for t in tr if float(np.sum(np.hypot(*np.diff(t["pts"], axis=0).T))) >= ESPOLON]
    if len(largos) != 1 or puros:
        return "1a"
    p = np.asarray(largos[0]["pts"], float); mu = p.mean(0)
    _, vec = np.linalg.eigh(np.cov((p - mu).T)); e, nrm = vec[:, 1], vec[:, 0]
    incl = abs(np.degrees(np.arctan2(e[0], e[1]))); incl = min(incl, 180 - incl)
    return "1l" if np.abs((p - mu) @ nrm).max() <= RECTA_MAX and incl <= INCL_MAX else "1a"


GRUPOS = {
    "1": {"digito": 1, "clases": ["1l", "1a"], "fichero": "etiquetas-1.json", "regla": regla_1,
          "texto": {"1l": "sólo la recta vertical (parece una l)", "1a": "los demás (bandera, base, curvos…)"}},
}


def carpeta() -> Path:
    """Dónde se escriben las decisiones: `web/` del experimento, o REVISOR_ETIQUETAS_DIR (las pruebas: nunca el fichero real)."""
    return Path(os.environ.get("REVISOR_ETIQUETAS_DIR", AQUI / "web"))


def grupo(d: str) -> dict:
    """El grupo de un dígito, o se NIEGA (R2): el 4 vive en /etiquetar, y un dígito sin grupos declarados no tiene pantalla."""
    d = str(d)
    if d == "4":
        raise ValueError("el 4 se separa en /etiquetar (sus etiquetas entran en la huella de los compositores)")
    if d not in GRUPOS:
        raise ValueError(f"el dígito {d} no tiene grupos declarados (hay: {', '.join(GRUPOS)})")
    return GRUPOS[d]


def manuales(d: str, alm: Path | None = None) -> dict:
    """{idx: clase} decididos a mano. Local (en git) > copia del almacén (si el dev se rehízo antes del commit) > ninguno."""
    g = grupo(d)
    for f in [carpeta() / g["fichero"]] + ([alm / g["fichero"]] if alm else []):
        if f.is_file():
            try:
                j = json.loads(f.read_text())
                return {int(i): c for c in g["clases"] for i in j.get(c, [])}
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    return {}


_REGLA: dict = {}                                                 # (d, idx) → clase por la regla: no cambia mientras corre


def elementos(datos, d: str, alm: Path | None = None) -> list[dict]:
    """Todos los dígitos d del dataset con su parte, la clase por la REGLA y la decisión a mano si la hay."""
    g = grupo(d); m = manuales(d, alm)
    parte = {int(i): "train" for i in datos.tr}; parte.update({int(i): "val" for i in datos.va}); parte.update({int(i): "extra" for i in datos.ex})
    out = []
    for i in np.flatnonzero(datos.y == g["digito"]):
        i = int(i)
        if (d, i) not in _REGLA:
            _REGLA[(d, i)] = g["regla"](datos.M[i])
        out.append({"idx": i, "parte": parte.get(i, "?"), "regla": _REGLA[(d, i)], "manual": m.get(i)})
    return out


def marcar(datos, d: str, i: int, clase: str, alm: Path | None = None) -> Path:
    """Guarda una decisión a mano (una de las clases del grupo, o «auto» = quitarla). Se niega con un idx que no es de ese
    dígito o una clase desconocida. Devuelve el fichero escrito (quien llama lo copia al almacén)."""
    g = grupo(d); i = int(i)
    if not 0 <= i < len(datos.y) or int(datos.y[i]) != g["digito"]:
        raise ValueError(f"idx {i} no es un {g['digito']}")
    if clase not in g["clases"] + ["auto"]:
        raise ValueError(f"clase: {' · '.join(g['clases'])} · auto")
    m = manuales(d, alm); m.pop(i, None)
    if clase != "auto":
        m[i] = clase
    f = carpeta() / g["fichero"]; f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({**{c: sorted(k for k, v in m.items() if v == c) for c in g["clases"]},
                             "quien": "el dueño, desde /separar", "cuando": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                             "regla_base": f"grupos.regla_{g['digito']} (ESPOLON {ESPOLON}, RECTA_MAX {RECTA_MAX}, INCL_MAX {INCL_MAX})"},
                            indent=1, ensure_ascii=False) + "\n")
    return f


def respaldar(f: Path, alm: Path | None, empujar) -> None:
    """Copia el fichero de decisiones al almacén y lo empuja. Nunca lanza: lo local sigue valiendo."""
    if alm is None:
        return
    try:
        alm.mkdir(parents=True, exist_ok=True); shutil.copyfile(f, alm / f.name); empujar(f"rizo-2 revisor: {f.name} (separar a mano)")
    except OSError as e:
        print(f"  ⚠ no pude copiar {f.name} al almacén: {e}", flush=True)
