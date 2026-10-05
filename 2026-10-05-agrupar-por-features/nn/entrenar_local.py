#!/usr/bin/env python3
"""Los k-means de `feat-agr`. El NOMBRE es el contrato con el freno: `cerrable.mjs` casa `entrenar_local.py`, y así el
veredicto «¿se puede apagar este server?» ve esto mientras corre. Aquí NO entra la etiqueta: se lee
resultados/representaciones.npz (nn/representar.py), que tampoco la tiene.

    python nn/entrenar_local.py              todo lo que falte (REANUDABLE: cada corrida se guarda y se salta si ya está)
    python nn/entrenar_local.py --solo a     sólo un ajuste

Ajustes (REGLAS.md §4):
  a  los 5620                     Z32 Z8 P32 M32 X8, K 10 20 30 50, semillas 1–5   (+ Z32-a / Z8-a si L7 lo pide)
  b  los 3823 de 30 escritores    Z32 Z8 X8, K 10 20 30 50 100, semillas 1–5      (escritores nuevos y L6)
  c  los 1797 de windep           Z32 Z8 X8, K 30, semillas 1–5                   (los mismos, agrupados aparte)

La partición PRINCIPAL de cada (ajuste, brazo, K) es la de menor inercia de las 5 semillas: se elige sin etiquetas.

La regla de Z−a, escrita antes de mirar (REGLAS.md §5): si en un banco los 4 arcos aportan MÁS DE LA MITAD de la
separación entre grupos de la principal (a, Z, K = 30), se agrupa también ese banco sin los arcos.

Al final, por cada brazo de zonas que corrió: resultados/grupos-<brazo>-K30.json con todo lo que NO depende de la
etiqueta: a qué grupo va cada dígito, tamaños, medoides, árbol de Ward, qué se enciende en cada grupo y qué manda (L7).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import detectores as D                          # noqa: E402
import metricas as M                            # noqa: E402
import representar as R                         # noqa: E402

RES = AQUI.parent / "resultados"
CACHE = RES / "grupos"
SEMILLAS = (1, 2, 3, 4, 5)
AJUSTES = {"a": {"brazos": ("Z32", "Z8", "P32", "M32", "X8"), "K": (10, 20, 30, 50)},
           "b": {"brazos": ("Z32", "Z8", "X8"), "K": (10, 20, 30, 50, 100)},
           "c": {"brazos": ("Z32", "Z8", "X8"), "K": (30,)}}
K_PRINCIPAL = 30
UMBRAL_ARCOS = 0.5                               # «más de la mitad de la separación»: dispara Z−a


def mascara(ajuste: str, origen: np.ndarray) -> np.ndarray:
    return {"a": np.ones(len(origen), bool), "b": origen != "windep", "c": origen == "windep"}[ajuste]


def ruta(ajuste: str, brazo: str, K: int, sem: int) -> Path:
    return CACHE / f"{ajuste}-{brazo}-K{K}-s{sem}.npz"


def correr(ajuste: str, brazo: str, K: int, X: np.ndarray, t0: float) -> None:
    for sem in SEMILLAS:
        destino = ruta(ajuste, brazo, K, sem)
        if destino.is_file():
            continue
        t = time.time(); km = M.kmeans(X, K, sem)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(destino, grupos=km["grupos"], centroides=km["centroides"], inercia=km["inercia"],
                 iteraciones=km["iteraciones"], convergio=km["convergio"])
        print(f"  {ajuste} {brazo:<6} K={K:<3} s{sem}: {km['iteraciones']:>3} it{'' if km['convergio'] else ' (SIN converger)'}"
              f" · inercia {km['inercia']:.1f} · {time.time() - t:.1f} s  [{time.time() - t0:.0f} s]", flush=True)


def cargar(ajuste: str, brazo: str, K: int, sem: int) -> dict:
    z = np.load(ruta(ajuste, brazo, K, sem))
    return {"grupos": z["grupos"].astype(np.int64), "centroides": z["centroides"], "inercia": float(z["inercia"]),
            "iteraciones": int(z["iteraciones"]), "convergio": bool(z["convergio"]), "semilla": sem}


def todas(ajuste: str, brazo: str, K: int) -> list:
    return [cargar(ajuste, brazo, K, s) for s in SEMILLAS]


def principal(ajuste: str, brazo: str, K: int) -> dict:
    """La de menor inercia de las 5 semillas (sin etiquetas)."""
    return min(todas(ajuste, brazo, K), key=lambda p: p["inercia"])


def nombres_features(brazo: str) -> tuple:
    return D.FAMILIAS[len(D.ARCOS):] if brazo.endswith("-a") else D.FAMILIAS


def que_manda(X: np.ndarray, lab: np.ndarray, K: int, brazo: str) -> dict:
    """L7: la separación entre grupos (suma de cuadrados entre grupos) repartida por feature y por zona."""
    fam = nombres_features(brazo)
    sep = M.separacion(X, lab, K).reshape(len(fam), 9)
    tot = float(sep.sum())
    return {"total": tot, "por_feature": {f: round(float(sep[i].sum() / tot), 4) for i, f in enumerate(fam)},
            "por_zona": {z: round(float(sep[:, j].sum() / tot), 4) for j, z in enumerate(R.ZONAS)},
            "arcos": round(float(sum(sep[i].sum() for i, f in enumerate(fam) if f in D.ARCOS) / tot), 4)}


def resumen(brazo: str, X: np.ndarray, umbrales: np.ndarray) -> dict:
    """Lo que se commitea de una partición principal (a, brazo, K = 30): nada de esto depende de la etiqueta."""
    tod = todas("a", brazo, K_PRINCIPAL)
    p = min(tod, key=lambda q: q["inercia"])
    lab, C = p["grupos"], p["centroides"]
    tam = np.bincount(lab, minlength=K_PRINCIPAL)
    Z = M.ward(C, tam)
    fam = nombres_features(brazo)
    u = umbrales[len(D.ARCOS):] if brazo.endswith("-a") else umbrales
    enc = (X.reshape(len(X), len(fam), 9) >= u[None, :, None])
    enciende = [[[round(float(v), 3) for v in enc[lab == g, i].mean(0)] for i in range(len(fam))] for g in range(K_PRINCIPAL)]
    return {"brazo": brazo, "ajuste": "a (los 5620)", "K": K_PRINCIPAL, "semilla": p["semilla"], "inercia": round(p["inercia"], 3),
            "inercias_por_semilla": {q["semilla"]: round(q["inercia"], 3) for q in tod},
            "iteraciones": p["iteraciones"], "convergio": p["convergio"],
            "features": list(fam), "zonas": list(R.ZONAS), "umbrales": [round(float(v), 3) for v in u],
            "tamanos": tam.tolist(), "medoides": M.medoides(X, C, lab).tolist(),
            "arbol_ward": [[int(a), int(b), round(h, 4), int(n)] for a, b, h, n in Z],
            "orden_del_arbol": M.orden_hojas(Z, K_PRINCIPAL),
            "enciende": enciende,
            "que_manda": que_manda(X, lab, K_PRINCIPAL, brazo),
            "centroides": [[round(float(v), 4) for v in c] for c in C],
            "grupo_de_cada_digito": lab.tolist()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo", choices=list(AJUSTES))
    a = ap.parse_args()
    t0 = time.time()
    r = R.cargar(); origen = r["origen"].astype(str)
    for ajuste, cfg in AJUSTES.items():
        if a.solo and ajuste != a.solo:
            continue
        m = mascara(ajuste, origen)
        for brazo in cfg["brazos"]:
            X = R.rep(r, brazo)[m]
            for K in cfg["K"]:
                correr(ajuste, brazo, K, X, t0)
    if a.solo and a.solo != "a":
        return 0
    zonales, decision = ["Z32", "Z8"], {}
    for banco_brazo in ("Z32", "Z8"):
        p = principal("a", banco_brazo, K_PRINCIPAL)
        qm = que_manda(R.rep(r, banco_brazo), p["grupos"], K_PRINCIPAL, banco_brazo)
        dispara = qm["arcos"] > UMBRAL_ARCOS
        decision[banco_brazo] = {"arcos": qm["arcos"], "umbral": UMBRAL_ARCOS, "corre_sin_arcos": dispara}
        print(f"L7 {banco_brazo}: los arcos aportan {qm['arcos']:.1%} de la separación entre grupos → "
              f"{'SÍ' if dispara else 'no'} se agrupa {banco_brazo}-a", flush=True)
        if dispara:
            X = R.rep(r, f"{banco_brazo}-a")
            for K in AJUSTES["a"]["K"]:
                correr("a", f"{banco_brazo}-a", K, X, t0)
            zonales.append(f"{banco_brazo}-a")
    (RES / "regla-sin-arcos.json").write_text(json.dumps(decision, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for brazo in zonales:
        um = r["umbrales32"] if R.BANCO[brazo] == "32" else r["umbrales8"]
        out = resumen(brazo, R.rep(r, brazo), um)
        (RES / f"grupos-{brazo}-K30.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n",
                                                      encoding="utf-8")
        print(f"→ resultados/grupos-{brazo}-K30.json (semilla {out['semilla']}, {out['iteraciones']} it)", flush=True)
    print(f"listo en {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
