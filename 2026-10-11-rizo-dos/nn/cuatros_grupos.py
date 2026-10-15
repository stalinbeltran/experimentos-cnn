"""Una lámina por grupo del 4 partido (4a abierto · 4l cerrado por arriba), cada dígito con su id y su partición.
La regla es la de web/servidor.py (Datos.subgrupo4), importada, no copiada. Pedido por el dueño el 2026-10-15."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "web"))
from servidor import Datos  # noqa: E402

d = Datos(); f = d.etiquetas_finas()
part = np.empty(len(d.y), object); part[d.tr] = "tr"; part[d.va] = "val"; part[d.ex] = "ex"
for cod, nom, fich in ((4, "4a (abierto)", "88-grupo-4a.png"), (10, "4l (cerrado por arriba)", "89-grupo-4l.png")):
    ids = np.flatnonzero(f == cod); n = len(ids); cols = 20 if n > 60 else 10; filas = -(-n // cols)
    fig, ax = plt.subplots(filas, cols, figsize=(cols * 1.0, filas * 1.15 + 0.6))
    for k, a in enumerate(np.ravel(ax)):
        a.axis("off")
        if k < n:
            i = ids[k]; a.imshow(d.tinta[i], cmap="gray_r", interpolation="nearest")
            a.set_title(f"{i} {part[i]}", fontsize=7, pad=1)
    fig.suptitle(f"Grupo {nom}: {n} dígitos — id y partición (tr/ex/val)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99)); fig.savefig(AQUI / "resultados" / fich, dpi=90); plt.close(fig)
    print(nom, n, {p: int((part[ids] == p).sum()) for p in ("tr", "ex", "val")})
