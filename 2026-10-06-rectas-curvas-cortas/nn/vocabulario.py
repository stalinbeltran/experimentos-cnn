#!/usr/bin/env python3
"""¿Pierden las cortas por el LARGO o por el VOCABULARIO? (instrucciones/02-criterio.md, el diagnóstico de las 02:47). El
compositor de 180 (3 semillas) sobre los dígitos crudos con cinco bancos:

  8 cortas · 8 largas-trazo (4 arcos + 4 rectas de feat-ind32) · 13 largas · 8 cortas + lazo + 4 esquinas (de las largas)

    python nn/vocabulario.py      → resultados/vocabulario.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

import evaluar as V

sys.path.insert(0, str(V.EXP.parent))
from expcnn import exigir_dataset, por_id       # noqa: E402

TRAZO = ("arco-E", "arco-W", "arco-N", "arco-S", "recta-V", "recta-H", "recta-S", "recta-B")
NO_TRAZO = ("lazo", "esquina-NE", "esquina-NW", "esquina-SE", "esquina-SW")


def main() -> int:
    V.torch.set_num_threads(2)
    d = np.load(exigir_dataset(V.DIGITOS) / "datos.npz")
    w = d["origen"] == "windep"
    x = d["imagenes"][w][:, None].astype(np.float32); y = d["etiquetas"][w].astype(np.int64); tr = d["particion"][w] == "train"
    org = por_id("feat-ind32").carpeta
    huellas = json.loads((org / "resultados" / "firma-por-clase.json").read_text(encoding="utf-8"))["huellas_best"]
    largas = dict(zip(V.LARGAS, V.cargar_banco(org / "nn" / "pesos", V.LARGAS, huellas)))
    cortas = V.cargar_banco(V.PESOS, list(V.F.CON_TRAZO))
    mc = V.mapas(cortas, x)
    ml = {f: V.mapas([r], x) for f, r in largas.items()}
    bancos = {"8 cortas": mc,
              "8 largas-trazo": np.concatenate([ml[f] for f in TRAZO], 1),
              "13 largas": np.concatenate([ml[f] for f in V.LARGAS], 1),
              "8 cortas + lazo + 4 esquinas": np.concatenate([mc] + [ml[f] for f in NO_TRAZO], 1),
              "5 sin trazo (lazo + esquinas) solas": np.concatenate([ml[f] for f in NO_TRAZO], 1)}
    out = {}
    for k, s in bancos.items():
        xm = s.reshape(len(x), -1)
        c = V.acierto(xm[tr], y[tr], xm[~tr], y[~tr])
        out[k] = {"compositor_180": c, "mapas": int(s.shape[1])}
        print(f"  {k:<36} {s.shape[1]:>2} mapas · compositor {c:.4f}", flush=True)
    c = {k: v["compositor_180"] for k, v in out.items()}
    if c["8 cortas + lazo + 4 esquinas"] >= c["13 largas"] - 0.01:
        ver = "vocabulario"
    elif c["8 largas-trazo"] > c["8 cortas"] + 0.02:
        ver = "largo"
    else:
        ver = "no lo separa"
    out["veredicto"] = ver
    print(f"  veredicto: {ver}")
    (V.RES / "vocabulario.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
