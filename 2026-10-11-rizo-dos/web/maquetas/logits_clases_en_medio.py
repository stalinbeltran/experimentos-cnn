"""Maqueta (2026-10-19, pedida por el dueño): variante de logits_hacia_abajo.py con las CLASES EN MEDIO, donde iba la
línea del 0: las positivas suben desde la franja de las clases y las negativas bajan. Mismos datos reales
(LazosEntrantes, idx 68: es un 4a, lo lee 1).
    ../.venv/bin/python web/maquetas/logits_clases_en_medio.py   ->  web/maquetas/logits_clases_en_medio.png
"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

K = ['0', '1', '2', '3', '4a', '5', '6', '7', '8', '9', '4l']
V = [-2.77, 5.54, -0.21, -5.96, 4.76, -4.73, 0.93, -1.87, 2.68, -5.19, -1.7]
LEIDO, REAL = 1, 4
ROJO, VERDE, GRIS = '#b22222', '#1f8a4c', '#bbbbbb'
col = [ROJO if k == LEIDO else VERDE if k == REAL else GRIS for k in range(len(K))]
mx = max(abs(v) for v in V)
G = 0.11          # media altura de la franja de las clases

def barra(ax, k, v, base):
    h = v / mx
    if abs(h) < 2 / 60: h = 2 / 60 * (1 if v >= 0 else -1)
    y0 = base if v >= 0 else -base
    ax.bar(k, h, bottom=y0, width=0.85, color=col[k])
    ax.text(k, y0 + h + (0.04 if v >= 0 else -0.04), f'{v:.1f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=7)

fig, (a, b) = plt.subplots(2, 1, figsize=(5.2, 7.6), dpi=200)
for ax in (a, b):
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_yticks([]); ax.tick_params(length=0)

# la de antes: línea del 0 en medio, clases abajo
for k, v in enumerate(V): barra(a, k, v, 0)
a.axhline(0, color='#444', lw=0.8)
a.set_xticks(range(len(K))); a.set_xticklabels(K, fontsize=9)
a.set_ylim(-1.25, 1.25)
a.set_title('ANTES: línea del 0 en medio, clases abajo', fontsize=10, loc='left')

# NUEVA: las clases ocupan la franja del 0
for k, v in enumerate(V): barra(b, k, v, G)
b.axhspan(-G, G, color='#f2f2f2', lw=0)
for k in range(len(K)):
    b.text(k, 0, K[k], ha='center', va='center', fontsize=9,
           color=col[k] if col[k] != GRIS else '#222', fontweight='bold' if col[k] != GRIS else 'normal')
b.set_xticks([])
b.set_ylim(-1.25 - G, 1.25 + G)
b.set_title('NUEVA: clases en medio (donde iba el 0)', fontsize=10, loc='left')

fig.suptitle('LazosEntrantes · idx 68 · es un 4a, lo lee 1\nrojo = 1 (leído) · verde = 4a (real)', fontsize=9, color='#555', y=0.995)
fig.tight_layout()
out = Path(__file__).with_suffix('.png'); fig.savefig(out, facecolor='white'); print(out)
