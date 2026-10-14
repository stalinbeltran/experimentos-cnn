#!/usr/bin/env python3
"""rizo-2 — DESGLOSE de un fallo con las features APROBADAS (§ 74): para cada idx, los objetos que ve el detector, las entradas
no nulas que recibe el compositor (con nombre: bloque · orientación/cono · celda), y la ATRIBUCIÓN de cada entrada a
«logit de la clase predicha − logit de la clase real» (entrada estandarizada × gradiente), por bloque y las mayores.
Entrena el compositor aprobado con la misma receta (180 × 13, MLP 256, semilla 0).

    ../.venv/bin/python nn/desglose.py 101 902 1516 1716   → resultados/76-desglose-<idx>.txt (y lo imprime)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent)); sys.path.insert(0, str(AQUI.parent / "docs" / "bocetos" / "2026-10-09-curvas-gabor"))
import cresta_curvas as CC                                          # noqa: E402
import evaluar as EV                                                # noqa: E402
import rastrear as RA                                               # noqa: E402
import resistencia as R                                             # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

BLOQUES = ["rectas_todo", "arcos", "golpes", "lazos", "entrantes60"]          # las features aprobadas, 144
ORI = ["0° —", "45° /", "90° |", "135° \\"]; SENT = ["k→", "k←", "k↓", "k↑"]; CONO = ["→ der", "← izq", "↓ abajo", "↑ arriba"]
CELDA = ["arriba-izq", "arriba-centro", "arriba-der", "centro-izq", "centro", "centro-der", "abajo-izq", "abajo-centro", "abajo-der"]


def nombres():
    n = []
    for b in BLOQUES:
        if b in ("rectas_todo", "arcos"):
            et = ORI if b == "rectas_todo" else SENT; lab = "recta" if b == "rectas_todo" else "arco"
            for o in range(4):
                n += [f"{lab} {et[o]} · {c}" for c in CELDA] + [f"{lab} {et[o]} · cuenta", f"{lab} {et[o]} · largo/32"]
        elif b in ("golpes", "lazos"):
            n += [f"{b[:-1]} · {c}" for c in CELDA] + [f"{b[:-1]} · cuenta"]
        else:
            for o in range(4):
                n += [f"entrante {CONO[o]} · {c}" for c in CELDA]
    return n


def main() -> int:
    ids = [int(a) for a in sys.argv[1:]] or [101, 902, 1516, 1716]
    torch.set_num_threads(2)
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz")); y, part = d["etiquetas"].astype(int), d["particion"]
    M = np.load(AQUI.parent / "2026-10-15-delgado-cnn" / "resultados" / "delgados-k3x3.npz")["M"].astype(np.float32)
    tr = np.flatnonzero(part == "train"); aum = [(0, 0)] + [(k, 0) for k in R.AUM] + [(0, k) for k in R.AUM]; B = CC.bloques()
    corta = lambda F: np.concatenate([F[:, B[b][0]:B[b][1]] for b in BLOQUES], 1)
    X = corta(np.concatenate([CC.caract(R.mover(M[tr], *m)) for m in aum])); Y = np.concatenate([y[tr]] * len(aum))
    red, mu, sd = RA.entrenar(X, Y, 0); nom = nombres(); assert len(nom) == X.shape[1] == 144
    salida = []
    for i in ids:
        ob = CC.objetos(M[i]); f = corta(CC.caract(M[i:i + 1]))[0]
        z = torch.from_numpy(((f - mu) / sd).astype(np.float32))[None].requires_grad_(True)
        lg = red(z)[0]; pred = int(lg.argmax()); real = int(y[i]); p = torch.softmax(lg, 0).detach().numpy()
        (lg[pred] - lg[real]).backward(); atr = (z.grad[0] * z.detach()[0]).numpy()          # entrada estandarizada × gradiente
        L = [f"\n=== idx {i} · real {real} · predicho {pred} · logits: " + " ".join(f"{c}:{float(lg[c]):+.1f}" for c in range(10)) + f" · p(pred) {p[pred]:.2f} p(real) {p[real]:.2f}"]
        L.append("OBJETOS:")
        for r_ in ob["rectas"]: L.append(f"  recta  {ORI[r_['o']]:6s} centro ({r_['cx']:4.1f},{r_['cy']:4.1f}) largo {r_['largo']:4.1f} px")
        for r_ in ob["restos"]: L.append(f"  resto  {ORI[r_['o']]:6s} centro ({r_['cx']:4.1f},{r_['cy']:4.1f}) largo {r_['largo']:4.1f} px")
        for a in ob["arcos"]: L.append(f"  arco   {SENT[a['s']]:6s} centro ({a['cx']:4.1f},{a['cy']:4.1f}) largo {a['largo']:4.1f} px · barrido {a['barrido']:.0f}°")
        for g in ob["golpes"]: L.append(f"  golpe         centro ({g['cx']:4.1f},{g['cy']:4.1f})")
        for z_ in ob["lazos"]: L.append(f"  lazo          centro ({z_['cx']:4.1f},{z_['cy']:4.1f}) área {z_['area']:.0f} px²")
        for e in ob["entrantes"]: L.append(f"  entrante {CONO[CC.cono(e)]:8s} centro ({e['cx']:4.1f},{e['cy']:4.1f}) área {e['area']:.0f} px² · {'esquina' if e['s'] == -1 else 'arco'} {e['barrido']:.0f}°")
        L.append(f"ENTRADAS NO NULAS ({int((f != 0).sum())} de 144):")
        for k in np.flatnonzero(f != 0): L.append(f"  {nom[k]:38s} = {f[k]:.2f}")
        L.append(f"ATRIBUCIÓN a (logit {pred} − logit {real}) = {float(lg[pred] - lg[real]):+.2f}; por bloque (+ empuja a {pred}, − empuja a {real}):")
        k0 = 0
        for b in BLOQUES:
            w = B[b][1] - B[b][0]; L.append(f"  {b:12s} {atr[k0:k0 + w].sum():+.2f}"); k0 += w
        L.append("  las 10 entradas que más empujan a la clase predicha:")
        for k in np.argsort(-atr)[:10]:
            if atr[k] > 0: L.append(f"    {atr[k]:+.2f}  {nom[k]:38s} (valor {f[k]:.2f}, media train {mu[k]:.2f})")
        L.append(f"  las 5 que más empujan a {real}:")
        for k in np.argsort(atr)[:5]:
            if atr[k] < 0: L.append(f"    {atr[k]:+.2f}  {nom[k]:38s} (valor {f[k]:.2f}, media train {mu[k]:.2f})")
        txt = "\n".join(L); print(txt); (AQUI / "resultados" / f"76-desglose-{i}.txt").write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
