#!/usr/bin/env python3
"""¿POR QUÉ predicen peor las cortas? En qué fracción de los dígitos de cada clase se enciende cada detector corto (el máximo
de su mapa ≥ su umbral), crudo y a 3 px; y, como resumen, cuánto DISCRIMINA cada uno: la desviación típica entre clases de
esa fracción (0 = se enciende igual en todas las clases y no dice nada del dígito). Al lado, lo mismo para las 13 largas.

    python nn/firma.py      → resultados/firma.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

import evaluar as V

sys.path.insert(0, str(V.EXP.parent))
from expcnn import exigir_dataset, por_id       # noqa: E402


def firma(reds, umbrales, nombres, x, y) -> dict:
    s = V.mapas(reds, x).reshape(len(x), len(reds), -1).max(2) >= np.array(umbrales)[None]
    por = {str(c): {f: round(float(s[y == c, j].mean()), 3) for j, f in enumerate(nombres)} for c in range(10)}
    disc = {f: round(float(np.std([por[str(c)][f] for c in range(10)])), 3) for f in nombres}
    media = {f: round(float(s[:, j].mean()), 3) for j, f in enumerate(nombres)}
    return {"por_clase": por, "se_enciende_en_todos": media, "discrimina_std_entre_clases": disc,
            "discrimina_medio": round(float(np.mean(list(disc.values()))), 3)}


def main() -> int:
    V.torch.set_num_threads(2)
    d = np.load(exigir_dataset(V.DIGITOS) / "datos.npz")
    w = d["origen"] == "windep"
    x = d["imagenes"][w][:, None].astype(np.float32); y = d["etiquetas"][w].astype(np.int64)
    vistas = {"nada": x, "norm3": V.N.normalizar(x, 3).astype(np.float32)}
    fams = list(V.F.CON_TRAZO)
    cortas = V.cargar_banco(V.PESOS, fams)
    uc = [json.loads((V.PESOS / f / "summary.json").read_text(encoding="utf-8"))["best"]["umbral"] for f in fams]
    org = por_id("feat-ind32").carpeta
    largas = V.cargar_banco(org / "nn" / "pesos", V.LARGAS)
    ul = [json.loads((org / "nn" / "pesos" / f / "summary.json").read_text(encoding="utf-8"))["best"]["umbral"] for f in V.LARGAS]
    out = {}
    for v, xv in vistas.items():
        out[f"cortas {v}"] = firma(cortas, uc, fams, xv, y)
        out[f"largas {v}"] = firma(largas, ul, list(V.LARGAS), xv, y)
        for k in (f"cortas {v}", f"largas {v}"):
            o = out[k]
            print(f"  {k:<13} discrimina (std entre clases, media) {o['discrimina_medio']:.3f} · se enciende en "
                  + " ".join(f"{f}:{p:.2f}" for f, p in o["se_enciende_en_todos"].items()), flush=True)
    (V.RES / "firma.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
