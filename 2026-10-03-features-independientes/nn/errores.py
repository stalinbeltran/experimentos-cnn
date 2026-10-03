#!/usr/bin/env python3
"""Atribución del error de los dígitos mal clasificados por el compositor POSICIONAL (02-criterio.md §
«Atribución del error»). Lineal ⇒ la diferencia de logits se reparte EXACTA entre los 13 detectores.

    python nn/errores.py [--sufijo ""]     → resultados/errores{sufijo}.json y errores{sufijo}.png
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compositor as C                          # noqa: E402
import features as F                            # noqa: E402

RES = Path(__file__).resolve().parent.parent / "resultados"
J = len(F.CON_TRAZO)
PCT = 90


def ajustar(xtr, ytr, semilla: int = 1) -> torch.nn.Linear:
    """El mismo lineal que compositor.logistica (épocas, lr, L2), devolviendo los pesos."""
    torch.manual_seed(semilla)
    W = torch.nn.Linear(xtr.shape[1], 10)
    opt = torch.optim.Adam(W.parameters(), lr=C.LR, weight_decay=C.L2)
    x, y = torch.from_numpy(xtr), torch.from_numpy(ytr)
    for _ in range(C.EPOCAS):
        opt.zero_grad(); Fn.cross_entropy(W(x), y).backward(); opt.step()
    return W


def main() -> int:
    suf = sys.argv[sys.argv.index("--sufijo") + 1] if "--sufijo" in sys.argv else ""
    m = dict(np.load(RES / f"mapas-digitos{suf}.npz")); tr, y = m["train"], m["y"]
    mapas = m["sigma"].astype(np.float32)                       # (N, 13, 8, 8)
    X = mapas.reshape(len(y), -1)
    W = ajustar(X[tr], y[tr])
    with torch.no_grad():
        pred = W(torch.from_numpy(X)).argmax(1).numpy()
    Wn, bn = W.weight.detach().numpy().reshape(10, J, 64), W.bias.detach().numpy()
    va = ~tr
    acc = float((pred[va] == y[va]).mean())
    ref = json.loads((RES / f"compositores{suf}.json").read_text())["posicional"]["semilla_1"]["acc_val"]
    print(f"posicional semilla 1: acc_val {acc:.4f} (compositores{suf}.json dice {ref})")

    # control: lineal sobre los 64 píxeles crudos
    import datos                                                 # noqa: PLC0415
    dig = datos.digitos(); P = dig["x"].reshape(len(y), -1)
    Wp = ajustar(P[tr], y[tr])
    with torch.no_grad():
        pred_px = Wp(torch.from_numpy(P)).argmax(1).numpy()
    acc_px = float((pred_px[va] == y[va]).mean())

    # mapa medio por (clase, detector) en TRAIN, y distancias de referencia en val bien clasificados
    medio = np.stack([mapas[tr & (y == c)].mean(0) for c in range(10)])          # (10, 13, 8, 8)
    dist = np.sqrt(((mapas - medio[y]) ** 2).sum((2, 3)))                        # (N, 13)
    ok = va & (pred == y)
    corte = np.stack([np.percentile(dist[ok & (y == c)], PCT, axis=0) for c in range(10)])   # (10, 13)

    fallos = []
    for i in np.flatnonzero(va & (pred != y)):
        t, p = int(y[i]), int(pred[i])
        empuje = ((Wn[p] - Wn[t]) * mapas[i].reshape(J, 64)).sum(1)            # (13,)
        sesgo = float(bn[p] - bn[t])
        orden = np.argsort(-empuje)
        j = int(orden[0])
        raro = bool(dist[i, j] > corte[t, j])
        fallos.append({"i": int(i), "real": t, "pred": p, "margen": round(float(empuje.sum() + sesgo), 3),
                       "culpable": F.CON_TRAZO[j], "empuje_culpable": round(float(empuje[j]), 3),
                       "segundo": F.CON_TRAZO[int(orden[1])], "empuje_segundo": round(float(empuje[orden[1]]), 3),
                       "fraccion_culpable": round(float(empuje[j] / max(1e-9, empuje[empuje > 0].sum())), 3),
                       "tipo": "reconocimiento" if raro else "composicion",
                       "distancia": round(float(dist[i, j]), 3), "corte_p90": round(float(corte[t, j]), 3),
                       "dificil_en_si": bool(pred_px[i] != t), "pred_pixeles": int(pred_px[i]),
                       "detectores_raros": [F.CON_TRAZO[k] for k in range(J) if dist[i, k] > corte[t, k]]})

    n = len(fallos)
    pares = Counter(f"{f['real']}→{f['pred']}" for f in fallos).most_common()
    culp = Counter(f["culpable"] for f in fallos).most_common()
    tipos = Counter(f["tipo"] for f in fallos)
    dif = sum(f["dificil_en_si"] for f in fallos)
    rec_dif = sum(f["dificil_en_si"] for f in fallos if f["tipo"] == "reconocimiento")
    # ¿cuántos detectores raros, por fallo?, y la línea de base: lo mismo en los aciertos
    n_raros_fallo = np.mean([len(f["detectores_raros"]) for f in fallos]) if n else 0
    n_raros_ok = float((dist[ok] > corte[y[ok]]).sum(1).mean())
    out = {"sufijo": suf, "acc_val": round(acc, 4), "n_val": int(va.sum()), "n_fallos": n,
           "control_pixeles": {"acc_val": round(acc_px, 4), "fallos": int((pred_px[va] != y[va]).sum())},
           "pares": pares, "culpables": culp, "tipos": dict(tipos),
           "dificiles_en_si": dif, "dificiles_entre_reconocimiento": rec_dif,
           "detectores_raros_media_fallos": round(float(n_raros_fallo), 2),
           "detectores_raros_media_aciertos": round(n_raros_ok, 2),
           "fallos": fallos}
    (RES / f"errores{suf}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"fallos {n}/{int(va.sum())} · control píxeles crudos: acc {acc_px:.4f} ({out['control_pixeles']['fallos']} fallos)")
    print(f"tipos: {dict(tipos)} · difíciles en sí (fallan también con píxeles): {dif} de {n}")
    print(f"detectores 'raros' por dígito: {n_raros_fallo:.2f} en los fallos vs {n_raros_ok:.2f} en los aciertos")
    print("pares:", ", ".join(f"{k} ×{v}" for k, v in pares))
    print("culpables:", ", ".join(f"{k} ×{v}" for k, v in culp))
    rejilla(dig["x"], mapas, fallos, RES / f"errores{suf}.png")
    return 0


def rejilla(x, mapas, fallos, destino: Path) -> None:
    """Por fallo: el dígito, el mapa del culpable, y el mapa medio de ese detector en la clase REAL (train)
    y en la PREDICHA. Ordenados por par real→pred."""
    from PIL import Image                                         # noqa: PLC0415
    esc, sep = 5, 2; t = 8 * esc
    fs = sorted(fallos, key=lambda f: (f["real"], f["pred"]))
    # mapa medio de train por clase (de nuevo, sin depender de main)
    m = dict(np.load(RES / "mapas-digitos.npz")); tr, y = m["train"], m["y"]
    medio = np.stack([m["sigma"][tr & (y == c)].mean(0) for c in range(10)])
    cols = 4; filas = len(fs); por_fila = 3
    bloques = [fs[k::por_fila] for k in range(por_fila)]
    alto = max(len(b) for b in bloques)
    W = por_fila * (cols * (t + sep) + 3 * sep) + sep; H = alto * (t + sep) + sep
    im = Image.new("L", (W, H), 128)
    for b, bloque in enumerate(bloques):
        x0 = sep + b * (cols * (t + sep) + 3 * sep)
        for r, f in enumerate(bloque):
            j = F.CON_TRAZO.index(f["culpable"])
            tiles = [1 - x[f["i"], 0], 1 - mapas[f["i"], j], 1 - medio[f["real"], j], 1 - medio[f["pred"], j]]
            for k, tt in enumerate(tiles):
                im.paste(Image.fromarray((np.clip(tt, 0, 1) * 255).astype(np.uint8)).resize((t, t), Image.NEAREST),
                         (x0 + k * (t + sep), sep + r * (t + sep)))
    im.save(destino)


if __name__ == "__main__":
    raise SystemExit(main())
