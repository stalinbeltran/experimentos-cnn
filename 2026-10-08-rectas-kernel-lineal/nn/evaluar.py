#!/usr/bin/env python3
"""El banco de prueba de `rect-lin` y las métricas del criterio (instrucciones/02-criterio.md).

    python nn/evaluar.py --referencias    Gabor (k × escalas, umbral calibrado en negativos de entreno) y la CNN
    python nn/evaluar.py --tablas         tablas del README desde resultados/rejilla.jsonl y referencias.json
    python nn/evaluar.py --figuras        resultados/*.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import datos as D  # noqa: E402

RES = AQUI.parent / "resultados"
GRUPOS = {"fino": (2, 3, 4), "medio": (6, 8), "grueso": (10, 12, 14)}


def r4(v) -> float:
    return round(float(v), 4)


def _desvio(ang: np.ndarray) -> np.ndarray:
    """Distancia del ángulo al centro de orientación más cercano (0–22,5°)."""
    d = np.mod(ang, 45.0)
    return np.minimum(d, 45.0 - d)


@torch.no_grad()
def logits(modelo, x: np.ndarray, lote: int = 1024) -> np.ndarray:
    return np.concatenate([modelo(torch.from_numpy(x[i:i + lote])).numpy() for i in range(0, len(x), lote)])


def metricas(L: np.ndarray, b: dict) -> dict:
    """L (N, 4) logits sobre el banco b → todas las métricas del criterio."""
    det = L.max(1) > 0
    ori_ok = L.argmax(1) == D.orientacion(np.nan_to_num(b["angulo"]))
    acierto = det & ori_ok
    t, g, lg = b["tipo"], b["grosor"], b["largo"]
    cont = t == D.CONTINUA
    m = {}
    for nom, gs in GRUPOS.items():
        m[f"recall_{nom}"] = r4(acierto[cont & np.isin(g, gs)].mean())
    m["recall_total"] = r4(acierto[cont].mean())
    m["recall_por_grosor"] = {int(v): r4(acierto[cont & (g == v)].mean()) for v in np.unique(g[cont])}
    fino = cont & np.isin(g, GRUPOS["fino"])
    m["recall_fino_por_largo"] = {int(v): r4(acierto[fino & (lg == v)].mean()) for v in np.unique(lg[cont])}
    dv = _desvio(b["angulo"])
    m["recall_fino_por_desvio"] = {f"{a}-{z}": r4(acierto[fino & (dv >= a) & (dv < z + 1e-6)].mean())
                                   for a, z in ((0, 7.5), (7.5, 15), (15, 22.5))}
    m["deteccion_fino_sin_orientacion"] = r4(det[fino].mean())
    pun = t == D.PUNTEADA
    m["punteada_por_separacion"] = {int(s): r4(acierto[pun & (b["separacion"] == s)].mean()) for s in np.unique(b["separacion"][pun])}
    cur = t == D.CURVA
    m["curva_deteccion_por_radio"] = {int(r): r4(det[cur & (b["radio"] == r)].mean()) for r in np.unique(b["radio"][cur])}
    m["curva_deteccion_por_radio_y_grosor"] = {f"{int(r)}/{int(gg)}": r4(det[cur & (b["radio"] == r) & (g == gg)].mean())
                                               for r in np.unique(b["radio"][cur]) for gg in (2, 4, 8)}
    # fuerza: mediana del mayor logit de las curvas MENOS la de las rectas de largo 22 y el mismo grosor (en logits)
    fz = {}
    for r in np.unique(b["radio"][cur]):
        dif = [np.median(L.max(1)[cur & (b["radio"] == r) & (g == gg)]) - np.median(L.max(1)[cont & (g == gg) & (lg == 22)])
               for gg in (2, 4, 8)]
        fz[int(r)] = r4(np.mean(dif))
    m["curva_fuerza_menos_recta"] = fz
    for tt in (D.NEG_RUIDO, D.NEG_PUNTOS, D.NEG_MANCHA):
        m[f"fp_{D.NOMBRE_TIPO[tt]}"] = r4(det[t == tt].mean())
    m["fp_total"] = r4(det[t >= D.NEG_RUIDO].mean())
    return m


def banco() -> tuple[np.ndarray, dict]:
    b = D.cargar(D.BANCO)
    return b["imagenes"].astype(np.float32)[:, None], b


def referencias() -> dict:
    import modelo as M  # noqa: PLC0415
    from referencia_cnn import CNN  # noqa: PLC0415
    xb, b = banco()
    e = D.cargar(D.ENTRENO)
    xneg = torch.from_numpy(e["imagenes"][e["tipo"] >= D.NEG_RUIDO].astype(np.float32)[:, None])
    out = {"gabor": []}
    for k in (5, 7, 9):
        for nv in (1, 2, 3):
            g = M.Gabor(k, nv); u = g.calibrar(xneg, 0.05)
            out["gabor"].append({"k": k, "escalas": nv, "umbral": r4(u), **metricas(logits(g, xb), b)})
            print(f"  gabor k={k} esc={nv}: recall fino {out['gabor'][-1]['recall_fino']}  grueso {out['gabor'][-1]['recall_grueso']}")
    cnn = CNN()
    out["cnn"] = {"id": "feat-ind32", "huellas": cnn.huellas, "parametros": cnn.n_parametros(), **metricas(logits(cnn, xb), b)}
    print(f"  cnn: recall fino {out['cnn']['recall_fino']}  grueso {out['cnn']['recall_grueso']}  fp {out['cnn']['fp_total']}")
    RES.mkdir(exist_ok=True)
    (RES / "referencias.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--referencias", action="store_true")
    a = p.parse_args()
    if a.referencias:
        referencias(); return 0
    p.print_help(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
