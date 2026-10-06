#!/usr/bin/env python3
"""Las huellas de los 8 detectores (`modelo.huella_pesos` de cada best.pt), para que otro experimento los lea POR SU ID y
compruebe que son éstos (Regla 0).

    python nn/huellas.py      → resultados/huellas.json
"""
from __future__ import annotations

import json

import evaluar as V

out = {f: V.modelo.huella_pesos(V.modelo.cargar(V.PESOS / f / "best.pt")[0]) for f in V.F.CON_TRAZO}
(V.RES / "huellas.json").write_text(json.dumps({"huellas_best": out}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(out)
