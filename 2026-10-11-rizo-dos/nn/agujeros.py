#!/usr/bin/env python3
"""rizo-2 § 122 — cuántos dígitos de cada clase tienen un AGUJERO cerrado (fondo encerrado ≥ 3 px; imagen > 0,5) en NIST y en los
dos datasets de 32 px de alto (§ 120 y § 122). Medido 2026-10-18."""
import numpy as np
from scipy import ndimage
B = "/home/deploy/src/foveal-vision-data/experimentos-cnn/"
D = {"NIST": "uci-optdigits-orig-32px-r20261005", "alto32 (§ 120, gris 32 px)": "uci-optdigits-orig-32px-trazo4-alto32-r20261018",
     "alto32-x64 (§ 122, gris 64 px)": "uci-optdigits-orig-32px-trazo4-alto32-x64-r20261018"}
def agujeros(t):
    fondo, n = ndimage.label(~t); borde = set(np.unique(np.r_[fondo[0], fondo[-1], fondo[:, 0], fondo[:, -1]]))
    return sum(1 for k in range(1, n + 1) if k not in borde and (fondo == k).sum() >= 3)
for nom, ds in D.items():
    z = np.load(B + ds + "/datos.npz"); X = z["imagenes"].astype(np.float32) > 0.5; y = z["etiquetas"]
    h = np.array([agujeros(x) for x in X])
    print(f"{nom:32s} con agujero: " + " · ".join(f"{c}: {100*np.mean(h[y==c]>=1):.0f} %" for c in range(10)) + f" · 8 con 2: {100*np.mean(h[y==8]>=2):.0f} %")
