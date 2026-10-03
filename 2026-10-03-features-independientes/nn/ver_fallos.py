#!/usr/bin/env python3
"""TODOS los dígitos de val mal clasificados por el compositor posicional, grandes y etiquetados
«real→pred», ordenados por par. Lee resultados/errores{sufijo}.json (de nn/errores.py).

    python nn/ver_fallos.py [--sufijo -c3]   → resultados/fallos{sufijo}.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos                                    # noqa: E402

RES = Path(__file__).resolve().parent.parent / "resultados"
ESC, COLS, ETQ = 14, 11, 18


def main() -> int:
    from PIL import Image, ImageDraw                              # noqa: PLC0415
    suf = sys.argv[sys.argv.index("--sufijo") + 1] if "--sufijo" in sys.argv else ""
    e = json.loads((RES / f"errores{suf}.json").read_text(encoding="utf-8"))
    x = datos.digitos()["x"][:, 0]
    fs = sorted(e["fallos"], key=lambda f: (f"{f['real']}{f['pred']}" not in ("18", "81"), f["real"], f["pred"]))
    lado = 8 * ESC; sep = 6
    filas = (len(fs) + COLS - 1) // COLS
    im = Image.new("L", (COLS * (lado + sep) + sep, filas * (lado + ETQ + sep) + sep), 255)
    d = ImageDraw.Draw(im)
    for k, f in enumerate(fs):
        r, c = divmod(k, COLS)
        x0, y0 = sep + c * (lado + sep), sep + r * (lado + ETQ + sep)
        tile = Image.fromarray(((1 - x[f["i"]]) * 255).astype(np.uint8)).resize((lado, lado), Image.NEAREST)
        im.paste(tile, (x0, y0))
        d.rectangle([x0 - 1, y0 - 1, x0 + lado, y0 + lado], outline=160)
        marca = "*" if f["dificil_en_si"] else ""
        d.text((x0 + 2, y0 + lado + 3), f"{f['real']}->{f['pred']}{marca}  {f['culpable']}", fill=0)
    destino = RES / f"fallos{suf}.png"
    im.save(destino)
    print(f"{len(fs)} fallos → {destino}  (* = también falla con píxeles crudos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
