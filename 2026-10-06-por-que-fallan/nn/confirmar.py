#!/usr/bin/env python3
"""La confirmación CIEGA de la iteración 7: el mismo compositor (los 180 de train de `windep`, 3 semillas), medido en los
3823 dígitos que NO son de `windep` (origen tra · cv · wdep), cuyo acierto ningún paso anterior del estudio ha mirado.

    python nn/confirmar.py lineas-nada lineas+lineas-grueso+cortas-nada+norm3 ...   → resultados/confirmacion.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent)); sys.path.insert(0, str(AQUI))
from expcnn import exigir_dataset                 # noqa: E402
import evaluar as V                               # noqa: E402

RES = AQUI.parent / "resultados"


def main(combos: list[str]) -> int:
    torch.set_num_threads(2)
    d = np.load(exigir_dataset(V.DIGITOS) / "datos.npz")
    x = d["imagenes"][:, None].astype(np.float32); y = d["etiquetas"].astype(np.int64); org = d["origen"]
    tr = (org == "windep") & (d["particion"] == "train"); ext = org != "windep"
    ruta = RES / "confirmacion.json"
    out = json.loads(ruta.read_text(encoding="utf-8")) if ruta.is_file() else {}
    for combo in combos:
        nombre, prepro = combo.rsplit("-", 1)
        bancos = [V.banco(n) for n in nombre.split("+")]
        idx = np.flatnonzero(tr | ext)
        s = np.concatenate([V.mapas(b, V.preparar(x[idx], b["rep"], vi)) for b in bancos for vi in prepro.split("+")], 1)
        s = s.reshape(len(idx), -1)
        mtr, mext = tr[idx], ext[idx]
        preds = [V.ajustar(s[mtr], y[idx][mtr], sem)(torch.from_numpy(s[mext])).argmax(1).numpy() for sem in (1, 2, 3)]
        yv, ov = y[idx][mext], org[idx][mext]
        acc = [V.r4((p == yv).mean()) for p in preds]
        out[combo] = {"acierto_3823": V.r4(np.mean(acc)), "por_semilla": acc,
                      "por_origen": {o: V.r4(np.mean([(p[ov == o] == yv[ov == o]).mean() for p in preds])) for o in ("tra", "cv", "wdep")}}
        print(f"  {combo:<42} 3823: {out[combo]['acierto_3823']:.4f} · por origen {out[combo]['por_origen']}", flush=True)
        ruta.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if "lineas-nada" in out and "lineas+lineas-grueso+cortas-nada+norm3" in out:
        r, b = out["lineas-nada"]["acierto_3823"], out["lineas+lineas-grueso+cortas-nada+norm3"]["acierto_3823"]
        out["criterio"] = {"C1": {"veredicto": "confirmada" if b >= r + 0.01 else "refutada", "S6b": b, "referencia": r},
                           "C2": {"veredicto": "confirmada" if (1 - b) <= 0.75 * (1 - r) else "refutada",
                                  "errores_quitados": V.r4(1 - (1 - b) / (1 - r))}}
        ruta.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  C1 {out['criterio']['C1']['veredicto']} · C2 {out['criterio']['C2']['veredicto']} "
              f"({out['criterio']['C2']['errores_quitados']:.0%} de los errores)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
