#!/usr/bin/env python3
"""Corrida 5: junta los mapas de dos juegos de detectores en un solo npz para el MISMO compositor.

    python nn/combinar.py        resultados/mapas-digitos.npz (fino) + mapas-digitos-c4.npz (grueso)
                                 → resultados/mapas-digitos-c24.npz, 26 mapas con sus nombres
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F                            # noqa: E402

RES = Path(__file__).resolve().parent.parent / "resultados"


def main() -> int:
    a, b = dict(np.load(RES / "mapas-digitos.npz")), dict(np.load(RES / "mapas-digitos-c4.npz"))
    if not (np.array_equal(a["y"], b["y"]) and np.array_equal(a["train"], b["train"])):
        raise SystemExit("✗ los dos npz no son de los mismos dígitos en el mismo orden: me niego a juntarlos")
    nombres = [f"fino:{f}" for f in F.CON_TRAZO] + [f"grueso:{f}" for f in F.CON_TRAZO]
    np.savez_compressed(RES / "mapas-digitos-c24.npz", sigma=np.concatenate([a["sigma"], b["sigma"]], 1), y=a["y"],
                        train=a["train"], umbrales=np.concatenate([a["umbrales"], b["umbrales"]]), nombres=np.array(nombres))
    print(f"{len(nombres)} mapas por dígito → resultados/mapas-digitos-c24.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
