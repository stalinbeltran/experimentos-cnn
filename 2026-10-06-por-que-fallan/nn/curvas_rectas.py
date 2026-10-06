#!/usr/bin/env python3
"""¿Se distingue una CURVA de una RECTA? (la pregunta del dueño del 2026-10-06). Para un banco, sobre imágenes con UNA sola
feature (sin segunda):

  arco→recta   en qué fracción de las imágenes de RECTA se enciende ALGÚN detector de arco (y cuál)
  recta→arco   en qué fracción de las imágenes de ARCO se enciende ALGÚN detector de recta
  y, como control, cuánto acierta el detector correcto (su recall)

en el val sintético fino (2–4 px), por tramo de radio del arco, y en la prueba gruesa de feat-bor por tramo de grosor; y en
los dígitos, el mismo cruce donde se sabe la respuesta: arcos en los 1 (una recta) y rectas en los 0 (una curva).

    python nn/curvas_rectas.py lineas nada      → resultados/curvas-rectas/<banco>-<preprocesado>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent.parent)); sys.path.insert(0, str(AQUI))
from expcnn import exigir_dataset                 # noqa: E402
import evaluar as V                               # noqa: E402
import features as F                              # noqa: E402

ARCOS, RECTAS = ("arco-E", "arco-W", "arco-N", "arco-S"), ("recta-V", "recta-H", "recta-S", "recta-B")
RES = AQUI.parent / "resultados" / "curvas-rectas"


def r4(v) -> float:
    return round(float(v), 4)


def cruce(b: dict, x: np.ndarray, prin: np.ndarray, sec: np.ndarray, prepro: str) -> tuple:
    m = sec < 0
    s = V.mapas(b, V.preparar(x[m], b["rep"], prepro)).reshape(m.sum(), len(b["reds"]), -1).max(2)
    h = s >= b["umbrales"][None]
    idx = {f: b["familias"].index(f) for f in ARCOS + RECTAS}
    return h, prin[m], idx, m


def resumen(h, prin, idx, filtro=None) -> dict:
    es_recta = np.isin(prin, [F.FAMILIAS.index(r) for r in RECTAS]); es_arco = np.isin(prin, [F.FAMILIAS.index(a) for a in ARCOS])
    if filtro is not None:
        es_recta &= filtro; es_arco &= filtro
    algun_arco = h[:, [idx[a] for a in ARCOS]].any(1); alguna_recta = h[:, [idx[r] for r in RECTAS]].any(1)
    correcto = np.array([h[i, idx[F.FAMILIAS[p]]] if F.FAMILIAS[p] in idx else False for i, p in enumerate(prin)])
    return {"n_rectas": int(es_recta.sum()), "n_arcos": int(es_arco.sum()),
            "arco_se_enciende_en_rectas": r4(algun_arco[es_recta].mean()) if es_recta.any() else None,
            "recta_se_enciende_en_arcos": r4(alguna_recta[es_arco].mean()) if es_arco.any() else None,
            "recall_rectas": r4(correcto[es_recta].mean()) if es_recta.any() else None,
            "recall_arcos": r4(correcto[es_arco].mean()) if es_arco.any() else None,
            "por_detector_de_arco_en_rectas": {a: r4(h[es_recta, idx[a]].mean()) for a in ARCOS} if es_recta.any() else None}


def main(nombre: str, prepro: str) -> int:
    b = V.banco(nombre); out = {"banco": nombre, "preprocesado": prepro}
    f = np.load(exigir_dataset("feat-ind32-sinteticas-32px-r20261005") / "datos.npz")
    val = f["particion"] == "val"
    h, prin, idx, m = cruce(b, f["imagenes"][val][:, None].astype(np.float32), f["principal"][val], f["secundaria"][val], prepro)
    radio = f["radio"][val][m]
    out["fino_val"] = {"todo": resumen(h, prin, idx)}
    for nombre_t, lo, hi in F.RADIO_TRAMOS:
        out["fino_val"][f"arcos de radio {nombre_t}"] = resumen(h, prin, idx, (radio >= lo) & (radio < hi) | (radio < 0))
    g = np.load(exigir_dataset(V.GRUESO) / "datos.npz")
    h, prin, idx, m = cruce(b, g["imagenes"][:, None].astype(np.float32), g["principal"], g["secundaria"], prepro)
    gr = g["grosor"][m]
    out["grueso"] = {"2–4 px": resumen(h, prin, idx, np.isin(gr, V.VISTOS)), "6–12 px": resumen(h, prin, idx, np.isin(gr, V.NO_VISTOS))}
    dg = V._digitos(); w = dg["w"]
    s = V.mapas(b, V.preparar(dg["x"][w], b["rep"], prepro)).reshape(w.sum(), len(b["reds"]), -1).max(2) >= b["umbrales"][None]
    y = dg["y"][w]
    out["digitos"] = {"algun_arco_en_los_1": r4(s[y == 1][:, [idx[a] for a in ARCOS]].any(1).mean()),
                      "alguna_recta_en_los_0": r4(s[y == 0][:, [idx[r] for r in RECTAS]].any(1).mean())}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{nombre}-{prepro}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    t = out["fino_val"]["todo"]; gg = out["grueso"]
    print(f"{nombre}-{prepro}: FINO arco→recta {t['arco_se_enciende_en_rectas']:.3f} · recta→arco {t['recta_se_enciende_en_arcos']:.3f}"
          f" (recall rectas {t['recall_rectas']:.3f}, arcos {t['recall_arcos']:.3f}) · por radio arco→recta / recta→arco: "
          + " ".join(f"{k.split()[-1]} {v['recta_se_enciende_en_arcos']:.2f}" for k, v in out["fino_val"].items() if k != "todo")
          + f" · GRUESO 6–12 arco→recta {gg['6–12 px']['arco_se_enciende_en_rectas']:.3f} recta→arco {gg['6–12 px']['recta_se_enciende_en_arcos']:.3f}"
          + f" · DÍGITOS arco en los 1 {out['digitos']['algun_arco_en_los_1']:.2f}, recta en los 0 {out['digitos']['alguna_recta_en_los_0']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
