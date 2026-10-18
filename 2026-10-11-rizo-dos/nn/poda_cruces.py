#!/usr/bin/env python3
"""rizo-2 § 121 — ¿se puede podar la cresta de NIST (sus ramitas falsas) sin quitar la BARRA del 7? Por longitud, y respetando los
cruces (un brazo corto se queda si al otro lado del nudo hay otro casi alineado). Imprime 1l con ramas, 7 y 4 con nudo. Medido 2026-10-18."""
import sys, numpy as np
R = str(__import__("pathlib").Path(__file__).resolve().parents[1])
sys.path.insert(0, R + "/nn"); sys.path.insert(0, R + "/web")
import cresta_curvas as CC, servidor as S, grupos as GR
d = S.Datos(); els = GR.elementos(d, "1", S.almacen_revisor()); l1 = [e["idx"] for e in els if (e["manual"] or e["regla"]) == "1l"]; y = d.y
def largo(p): p = np.asarray(p, float); return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0
def podar(tr, L):
    """quita los tramos < L px que acaban en una punta libre (racimo de un solo tramo), una pasada."""
    cnt = {}
    for t in tr:
        for r in (t["a"], t["b"]): cnt[r] = cnt.get(r, 0) + 1
    libre = lambda r: r < 0 or cnt.get(r, 0) == 1
    return [t for t in tr if not ((libre(t["a"]) or libre(t["b"])) and not (libre(t["a"]) and libre(t["b"])) and largo(t["pts"]) < L)]
def nudos(tr):
    c = {}
    for t in tr:
        for r in (t["a"], t["b"]):
            if r > 0: c[r] = c.get(r, 0) + 1
    return sum(v >= 3 for v in c.values())
T = {i: CC.tramos(d.M[i])[0] for i in set(l1) | set(np.flatnonzero((y == 7) | (y == 4)).tolist())}
for L in (0, 3, 4, 5, 6, 8):
    f = lambda i: podar(T[i], L) if L else T[i]
    r1 = np.mean([len(f(i)) >= 3 for i in l1]); r7 = np.mean([nudos(f(i)) > 0 for i in np.flatnonzero(y == 7)]); r4 = np.mean([nudos(f(i)) > 0 for i in np.flatnonzero(y == 4)])
    print(f"poda < {L} px: 1l con ramas {100*r1:.0f} % · 7 con nudo {100*r7:.0f} % · 4 con nudo {100*r4:.0f} %")
print("--- poda que respeta los CRUCES (un brazo corto se queda si al otro lado del nudo hay otro casi alineado: ≥ ANG°)")
def dir_salida(t, r, n=4):
    p = np.asarray(t["pts"], float); p = p if t["a"] == r else p[::-1]; q = p[min(n, len(p) - 1)] - p[0]
    return q / (np.hypot(*q) + 1e-9)
def podar2(tr, L, ANG):
    cnt = {}
    for t in tr:
        for r in (t["a"], t["b"]): cnt[r] = cnt.get(r, 0) + 1
    libre = lambda r: r < 0 or cnt.get(r, 0) == 1
    out = []
    for t in tr:
        corto = (libre(t["a"]) != libre(t["b"])) and largo(t["pts"]) < L
        if corto:
            r = t["b"] if libre(t["a"]) else t["a"]; u = dir_salida(t, r)
            otros = [o for o in tr if o is not t and r in (o["a"], o["b"])]
            if any(np.degrees(np.arccos(np.clip(np.dot(u, dir_salida(o, r)), -1, 1))) >= ANG for o in otros if not (libre(o["a"]) and libre(o["b"]))) and \
               any(libre(o["a"]) != libre(o["b"]) and largo(o["pts"]) < L for o in otros):     # el opuesto también es corto: una barra que cruza
                out.append(t); continue
            continue
        out.append(t)
    return out
for L in (5, 6, 8):
    for ANG in (120, 140):
        f = lambda i: podar2(T[i], L, ANG)
        r1 = np.mean([len(f(i)) >= 3 for i in l1]); r7 = np.mean([nudos(f(i)) > 0 for i in np.flatnonzero(y == 7)]); r4 = np.mean([nudos(f(i)) > 0 for i in np.flatnonzero(y == 4)])
        print(f"poda < {L} px, cruce ≥ {ANG}°: 1l con ramas {100*r1:.0f} % · 7 con nudo {100*r7:.0f} % · 4 con nudo {100*r4:.0f} %")
