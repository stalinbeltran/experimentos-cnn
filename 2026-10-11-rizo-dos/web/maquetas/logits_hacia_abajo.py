"""Maqueta (2026-10-19, pedida por el dueño): las barras de logits de la pantalla Compositor, hoy y con los negativos
hacia abajo. Datos REALES: compositor LazosEntrantes, idx 68 (es un 4a, lo lee 1), sacados de /api/leer ese día.
    ../.venv/bin/python web/maquetas/logits_hacia_abajo.py   ->  web/maquetas/logits_hacia_abajo.png
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

fig, (a, b) = plt.subplots(2, 1, figsize=(5.2, 6.4), dpi=200, gridspec_kw={'height_ratios': [1, 1.6]})
for ax in (a, b):
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_yticks([]); ax.set_xticks(range(len(K))); ax.set_xticklabels(K, fontsize=9); ax.tick_params(length=0)

# HOY: altura = |v|, todas hacia arriba; el negativo sólo se distingue por la transparencia
for k, v in enumerate(V):
    h = max(2 / 60, abs(v) / mx)
    a.bar(k, h, width=0.85, color=col[k], alpha=0.35 if v < 0 else 1)
    a.text(k, h + 0.04, f'{v:.1f}', ha='center', va='bottom', fontsize=7)
a.set_ylim(0, 1.25)
a.set_title('HOY: todas hacia arriba (negativas = más claras)', fontsize=10, loc='left')

# PROPUESTA: la línea del 0 en medio; positivas arriba, negativas abajo, misma escala
for k, v in enumerate(V):
    h = v / mx
    if abs(h) < 2 / 60: h = 2 / 60 * (1 if v >= 0 else -1)
    b.bar(k, h, width=0.85, color=col[k])
    b.text(k, h + (0.04 if v >= 0 else -0.04), f'{v:.1f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=7)
b.axhline(0, color='#444', lw=0.8)
b.text(len(K) - 0.45, 0.02, '0', fontsize=7, color='#444', va='bottom', ha='right')
b.set_ylim(-1.25, 1.25)
b.xaxis.set_ticks_position('bottom')
b.set_title('PROPUESTA: negativas hacia abajo', fontsize=10, loc='left')

fig.suptitle('LazosEntrantes · idx 68 · es un 4a, lo lee 1\nrojo = 1 (leído) · verde = 4a (real)', fontsize=9, color='#555', y=0.995)
fig.tight_layout()
out = Path(__file__).with_suffix('.png'); fig.savefig(out, facecolor='white'); print(out)
