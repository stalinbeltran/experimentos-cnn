#!/usr/bin/env python3
"""Todo lo que va DESPUÉS de los detectores, en un solo paso (minutos, en el dev). Lee resultados/mapas-digitos.npz
(nn/aplicar.py) y los pesos para los mapas de los dígitos transformados.

  C4  compositores presencia (13) y posicional (832), 180 train / 1617 val, 3 semillas — corrida 2 de feat-ind
  C5  curva por N de train sobre el test fijo de 717 (el MISMO reparto que la corrida 11 de feat-ind) y, aparte,
      con los 3823 dígitos de otros escritores añadidos al train
  C6  generalización G1/G2/G3 con 5 transformaciones y los compositores A (posicional) y B (máx 3×3) —
      corridas 12 y 13 —, más los píxeles crudos de 32×32 como referencia

    python nn/componer.py       → resultados/compositores.json, curva.json, generalizacion.json, transformaciones.png

El compositor lineal (Adam, 300 épocas, lr 1e-2, L2 1e-3) y el reparto son los de feat-ind, COPIADOS: es lo que
hace comparables los números (REGLAS.md §1). Las transformaciones NO son las mismas: aquí se aplican al bitmap de
32×32 (ver `transformar`).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import aplicar                                  # noqa: E402
import datos                                    # noqa: E402

RES = AQUI.parent / "resultados"
EPOCAS, LR, L2, SEMILLAS = 300, 1e-2, 1e-3, (1, 2, 3)
SEMILLA_REPARTO = 2026
RESERVA_POR_CLASE = 90
TAMANOS = (36, 90, 180, 360, 540, 900, 1080)
SEMILLA_T = 2027
TRANSFORMACIONES = ("desplazar", "ruido", "engrosar", "adelgazar", "ocluir")
PARES = ((6, 9), (9, 6), (2, 5), (5, 2), (1, 8), (8, 1), (4, 1))


def ajustar(x, y, sem):
    torch.manual_seed(sem)
    W = torch.nn.Linear(x.shape[1], 10); opt = torch.optim.Adam(W.parameters(), lr=LR, weight_decay=L2)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(EPOCAS):
        opt.zero_grad(); Fn.cross_entropy(W(xt), yt).backward(); opt.step()
    return W


def predecir(W, x):
    with torch.no_grad():
        return W(torch.from_numpy(x)).argmax(1).numpy()


def confusion(y, p):
    c = np.zeros((10, 10), int)
    for a, b in zip(y, p):
        c[a, b] += 1
    return c


def reparto(y, tr):
    """COPIA de feat-ind/nn/curva.py: (orden de train ampliable, máscara de test de 717)."""
    rng = np.random.default_rng(SEMILLA_REPARTO)
    val = np.flatnonzero(~tr); reserva = []
    for c in range(10):
        reserva += list(rng.choice(val[y[val] == c], RESERVA_POR_CLASE, replace=False))
    reserva = np.array(reserva)
    test = ~tr.copy(); test[reserva] = False

    def intercalar(idx):
        por = [list(rng.permutation(idx[y[idx] == c])) for c in range(10)]
        out = []
        while any(por):
            for p in por:
                if p: out.append(p.pop())
        return np.array(out)

    return np.concatenate([intercalar(np.flatnonzero(tr)), intercalar(reserva)]), test


def transformar(x: np.ndarray, que: str) -> np.ndarray:
    """x (N,1,32,32) 0/1 → transformada, determinista. ANÁLOGAS a las de feat-ind, no iguales (allí, sobre 8×8):
    desplazar 1 CELDA = 4 px en una de 8 direcciones · ruido gaussiano σ 0,15 · engrosar = dilatar 2 px · adelgazar =
    erosionar 1 px · ocluir un bloque de 12×12 px (3×3 celdas)."""
    rng = np.random.default_rng(SEMILLA_T + TRANSFORMACIONES.index(que))
    t = torch.from_numpy(x)
    if que == "desplazar":
        out = np.zeros_like(x)
        dirs = [(dy, dx) for dy in (-4, 0, 4) for dx in (-4, 0, 4) if (dy, dx) != (0, 0)]
        for i, k in enumerate(rng.integers(0, 8, len(x))):
            dy, dx = dirs[k]
            src = x[i, 0, max(0, -dy):32 - max(0, dy), max(0, -dx):32 - max(0, dx)]
            out[i, 0, max(0, dy):max(0, dy) + src.shape[0], max(0, dx):max(0, dx) + src.shape[1]] = src
        return out
    if que == "ruido":
        return np.clip(x + rng.normal(0, 0.15, x.shape).astype(np.float32), 0, 1)
    if que == "engrosar":
        return Fn.max_pool2d(t, 5, 1, 2).numpy()
    if que == "adelgazar":
        return (-Fn.max_pool2d(-t, 3, 1, 1)).numpy()
    if que == "ocluir":
        out = x.copy()
        for i, (r, c) in enumerate(rng.integers(0, 21, (len(x), 2))):
            out[i, 0, r:r + 12, c:c + 12] = 0
        return out
    raise ValueError(que)


def entradas(s: np.ndarray, variante: str) -> np.ndarray:
    if variante == "presencia":
        return s.reshape(len(s), s.shape[1], -1).max(2)
    t = torch.from_numpy(s)
    if variante == "B_max3":
        t = Fn.max_pool2d(t, 3, 1, 1)
    return t.flatten(1).numpy()


def media(v):
    return round(float(np.mean(v)), 4), round(float(np.std(v, ddof=1)), 4)


def c4(s, y, tr) -> dict:
    out = {"n_train": int(tr.sum()), "n_val": int((~tr).sum()), "azar": 0.1}
    for nombre in ("presencia", "posicional"):
        x = entradas(s, nombre).astype(np.float32)
        accs, conf = [], np.zeros((10, 10), int)
        for sem in SEMILLAS:
            W = ajustar(x[tr], y[tr], sem); p = predecir(W, x[~tr])
            accs.append(float((p == y[~tr]).mean()))
            if sem == 1:
                conf = confusion(y[~tr], p); acc_tr = float((predecir(W, x[tr]) == y[tr]).mean())
        m, sd = media(accs)
        out[nombre] = {"entradas": int(x.shape[1]), "acc_val_media": m, "acc_val_sd": sd, "acc_val_por_semilla": accs,
                       "acc_train_semilla1": round(acc_tr, 4), "confusion_semilla1": conf.tolist(),
                       "pares_semilla1": {f"{a}_como_{b}": int(conf[a, b]) for a, b in PARES}}
        print(f"C4 {nombre:<11} ({x.shape[1]:>3}): acc_val {m:.4f} ± {sd:.4f} · " +
              " ".join(f"{a}→{b} {conf[a, b]}" for a, b in PARES), flush=True)
    out["delta_posicional_menos_presencia"] = round(out["posicional"]["acc_val_media"] - out["presencia"]["acc_val_media"], 4)
    return out


def c5(s, y, tr, extra) -> dict:
    w = ~extra
    orden, test = reparto(y[w], tr[w])
    assert np.bincount(y[w][orden[:180]]).tolist() == [18] * 10 and int(test.sum()) == 717
    xw = entradas(s[w], "posicional").astype(np.float32); yw = y[w]
    out = {"test": 717, "curva": {}, "con_extra": {}}
    for n in TAMANOS:
        accs = [float((predecir(ajustar(xw[orden[:n]], yw[orden[:n]], sem), xw[test]) == yw[test]).mean()) for sem in SEMILLAS]
        out["curva"][n] = dict(zip(("lineal", "lineal_sd"), media(accs)))
        print(f"C5 N={n:>4}: {out['curva'][n]['lineal']:.4f} ± {out['curva'][n]['lineal_sd']:.4f}", flush=True)
    # otros escritores: train = 1080 (los mismos de N=1080) + los 3823 extra; mismo test de 717
    xe = entradas(s[extra], "posicional").astype(np.float32); ye = y[extra]
    for nombre, (xa, ya) in {"extra_solo": (xe, ye),
                             "1080+extra": (np.concatenate([xw[orden[:1080]], xe]), np.concatenate([yw[orden[:1080]], ye]))}.items():
        accs = [float((predecir(ajustar(xa, ya, sem), xw[test]) == yw[test]).mean()) for sem in SEMILLAS]
        out["con_extra"][nombre] = {"n_train": int(len(ya)), **dict(zip(("lineal", "lineal_sd"), media(accs)))}
        print(f"C5 {nombre:<11} (N={len(ya)}): {out['con_extra'][nombre]['lineal']:.4f}", flush=True)
    return out


def c6(s, y, tr, extra, x32) -> dict:
    w = ~extra
    s, y, tr, x32 = s[w], y[w], tr[w], x32[w]
    orden, test = reparto(y, tr)
    yt = y[test]
    reds, _, _ = aplicar.detectores()
    xt = {t: transformar(x32[test], t) for t in TRANSFORMACIONES}
    st = {t: aplicar.mapas(reds, xt[t]) for t in TRANSFORMACIONES}
    out = {"test": int(test.sum()), "transformaciones": TRANSFORMACIONES, "casos": {}}
    casos = {"detectores A_posicional": ("A", None), "detectores B_max3": ("B_max3", None),
             "píxeles crudos 32×32": ("crudo", None)}
    for caso, (v, _) in casos.items():
        if v == "crudo":
            X = x32.reshape(len(y), -1); XT = {t: xt[t].reshape(len(yt), -1) for t in TRANSFORMACIONES}
        else:
            X = entradas(s, v); XT = {t: entradas(st[t], v) for t in TRANSFORMACIONES}
        X = X.astype(np.float32); XT = {t: a.astype(np.float32) for t, a in XT.items()}
        r = {"train": [], "limpio": [], "n36": [], "n1080": []} | {t: [] for t in TRANSFORMACIONES}
        for sem in SEMILLAS:
            W = ajustar(X[tr], y[tr], sem)
            r["train"].append(float((predecir(W, X[tr]) == y[tr]).mean()))
            r["limpio"].append(float((predecir(W, X[test]) == yt).mean()))
            for t in TRANSFORMACIONES:
                r[t].append(float((predecir(W, XT[t]) == yt).mean()))
            for n in (36, 1080):
                r[f"n{n}"].append(float((predecir(ajustar(X[orden[:n]], y[orden[:n]], sem), X[test]) == yt).mean()))
        m = {k: float(np.mean(v_)) for k, v_ in r.items()}
        fila = {"acc_train": round(m["train"], 4), "acc_test": round(m["limpio"], 4), "G1_brecha": round(m["train"] - m["limpio"], 4),
                "acc_n36": round(m["n36"], 4), "acc_n1080": round(m["n1080"], 4), "G2": round(m["n36"] / m["n1080"], 4),
                "acc_transformado": {t: round(m[t], 4) for t in TRANSFORMACIONES},
                "G3": {t: round(m[t] / m["limpio"], 4) for t in TRANSFORMACIONES}}
        fila["G3_medio"] = round(float(np.mean(list(fila["G3"].values()))), 4)
        out["casos"][caso] = fila
        print(f"C6 {caso:<24} test {fila['acc_test']:.3f} · G1 {fila['G1_brecha']:.3f} · G2 {fila['G2']:.3f} · " +
              " ".join(f"{t[:4]} {fila['acc_transformado'][t]:.3f}" for t in TRANSFORMACIONES) + f" · G3 medio {fila['G3_medio']:.3f}", flush=True)
    dibujar(x32[test], xt)
    return out


def dibujar(x0, xt):
    from PIL import Image, ImageDraw                    # noqa: PLC0415
    lado, sep, etq = 64, 4, 70; filas = {"limpio": x0} | xt
    im = Image.new("L", (etq + 8 * (lado + sep), len(filas) * (lado + sep) + sep), 255); d = ImageDraw.Draw(im)
    for r, (f, x) in enumerate(filas.items()):
        d.text((4, r * (lado + sep) + lado // 2), f, fill=0)
        for c in range(8):
            im.paste(Image.fromarray(((1 - x[c * 37, 0]) * 255).astype(np.uint8)).resize((lado, lado), Image.NEAREST),
                     (etq + c * (lado + sep), sep + r * (lado + sep)))
    im.save(RES / "transformaciones.png")


def main() -> int:
    if not aplicar.MAPAS.is_file():
        raise SystemExit("✗ no está resultados/mapas-digitos.npz: primero nn/aplicar.py")
    m = dict(np.load(aplicar.MAPAS)); s, y, tr, extra = m["sigma"], m["y"], m["train"], m["extra"]
    x32 = datos.digitos(con_extra=True)["x"]
    t0 = time.time()
    w = ~extra
    for nombre, f in (("compositores", lambda: c4(s[w], y[w], tr[w])), ("curva", lambda: c5(s, y, tr, extra)),
                      ("generalizacion", lambda: c6(s, y, tr, extra, x32))):
        r = f(); r["segundos"] = round(time.time() - t0, 1)
        (RES / f"{nombre}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"listo en {time.time() - t0:.0f} s → resultados/compositores.json, curva.json, generalizacion.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
