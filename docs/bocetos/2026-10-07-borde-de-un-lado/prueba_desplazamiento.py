#!/usr/bin/env python3
"""Prueba rápida (0 $, en el dev), 2026-10-07: la CURVA de desplazamiento del compositor. Pedida por el dueño:
«prueba los entrenamientos del compositor con varios desplazamientos de su entrada, cada uno por separado y luego
añadiéndolos gradualmente; en teoría debe ser más resistente a un desplazamiento ligero, y debe perder su capacidad
cuando el dígito quede fuera de su campo de visión».

Mismos kernels FIJOS, misma regresión logística y mismo mapa 8×8 por vista (posición tal cual, sin máximo) que
prueba_compositores.py. El desplazamiento es de la IMAGEN, en píxeles (= desplazar los mapas de las vistas antes de
reducir), en las 8 direcciones, con relleno de ceros: lo que sale del lienzo se pierde.

  Entrenamiento (3823 dígitos de otros escritores):
    base            sin desplazar
    solo s          SÓLO las 8 copias desplazadas exactamente s px (sin el original); también s = 12 y 16, para ver
                    cuánta información queda cuando el dígito ya ha salido del lienzo 12 o 16 px
    ≤s              el original y las copias de 1, 2, …, s px (1 + 8s copias): el compositor «gradual»
  Prueba: los 1617 de `val` desplazados d px, para cada d, en cada una de las 8 direcciones; se da la media de las 8
  (y la peor). Más el cuartil de gruesos reales sin desplazar.

    /tmp/vizenv/bin/python prueba_desplazamiento.py --uci <dir con datos.npz> [--planes "≤6,≤8"]
    → resultados-desplazamiento.{json,txt}; con --planes sólo corre esos y los AÑADE al json que ya haya

⚠ Memoria: «≤8» son 65 copias de 3823 dígitos (248.000 filas); con todo en float64 no cabía en los 3,9 GB del dev junto a
otro trabajo y lo mató el OOM killer (2026-10-07). Los mapas van en float32 y cada plan se libera antes del siguiente.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from prueba_compositores import mapas, mover, juntar

AQUI = Path(__file__).resolve().parent
BRAZOS = {"tinta": ["tinta"], "8 vistas": [0, 45, 90, 135, 180, 225, 270, 315]}
S = (1, 2, 3, 4, 6, 8)                                  # desplazamientos de entrenamiento, px
D = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16)                # desplazamientos de prueba, px
DIRS8 = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]


def anillo(s):
    return [(s * a, s * b) for a, b in DIRS8]


def main():
    a = argparse.ArgumentParser(); a.add_argument("--uci", required=True); a.add_argument("--planes", default="")
    a = a.parse_args()
    d = np.load(Path(a.uci) / "datos.npz"); X, y, part = d["imagenes"], d["etiquetas"], d["particion"]
    te, tr = part == "val", part == "extra"
    tinta = X.reshape(len(X), -1).sum(1)
    gruesos = np.zeros(len(X), bool)
    for c in range(10):
        m = te & (y == c); gruesos[m] = tinta[m] >= np.quantile(tinta[m], .75)
    Xtr, ytr = X[tr], y[tr]

    t = time.time()                                     # mapas de prueba: cada d, cada dirección
    m32 = lambda Z: {k: v.astype(np.float32) for k, v in mapas(Z, "pos").items()}
    Fte = {0: [m32(X[te])]}
    for dd in D[1:]:
        Fte[dd] = [m32(mover(X[te], dy, dx)) for dy, dx in anillo(dd)]
    Fgr = m32(X[gruesos])
    print(f"mapas de prueba: {time.time() - t:.0f} s", flush=True)

    planes = {"base": [(0, 0)]}
    planes |= {f"solo {s}": anillo(s) for s in S}
    planes |= {f"≤{s}": [(0, 0)] + [m for k in range(1, s + 1) for m in anillo(k)] for s in S}
    planes |= {f"solo {s}": anillo(s) for s in (12, 16)}     # ¿cuánto queda cuando sale del lienzo 12 o 16 px?
    destino = AQUI / "resultados-desplazamiento.json"
    salida = {"cuando": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "D": list(D), "acierto": {}}
    if a.planes:
        planes = {k: v for k, v in planes.items() if k in a.planes.split(",")}
        if destino.is_file():
            salida = json.loads(destino.read_text(encoding="utf-8"))
            salida["acierto"] = {n: {**r, "media": {int(k): v for k, v in r["media"].items()},
                                     "peor": {int(k): v for k, v in r["peor"].items()}} for n, r in salida["acierto"].items()}
    for nombre, movs in planes.items():
        Ftr = {}
        for dy, dx in movs:                                  # por copia y en float32: «≤8» no cabía de golpe
            for k, v in m32(mover(Xtr, dy, dx)).items():
                Ftr.setdefault(k, []).append(v)
        Ftr = {k: np.concatenate(v) for k, v in Ftr.items()}
        b = np.tile(ytr, len(movs))
        for brazo, vs in BRAZOS.items():
            t = time.time()
            m = LogisticRegression(max_iter=3000, C=1.0).fit(juntar(Ftr, vs).astype(np.float32), b)
            r = {"media": {}, "peor": {}}
            for dd in D:
                acc = [float((m.predict(juntar(F, vs)) == y[te]).mean()) for F in Fte[dd]]
                r["media"][dd] = round(float(np.mean(acc)), 4); r["peor"][dd] = round(min(acc), 4)
            r["gruesos-reales"] = round(float((m.predict(juntar(Fgr, vs)) == y[gruesos]).mean()), 4)
            salida["acierto"][f"{brazo} · {nombre}"] = r
            print(f"  {brazo:9s} {nombre:7s} {len(b):>7d} ej. {time.time() - t:6.1f} s  "
                  + " ".join(f"{r['media'][dd]:.3f}" for dd in D) + f"  | gruesos {r['gruesos-reales']:.3f}", flush=True)
        destino.write_text(json.dumps(salida, ensure_ascii=False, indent=1))
        del Ftr

    lineas = [f"== acierto en val (1617), MEDIA de las 8 direcciones, según el desplazamiento de prueba d (px) ==",
              f"{'compositor':22s}" + "".join(f"{f'd={dd}':>8s}" for dd in D) + f"{'gruesos':>9s}"]
    for n, r in salida["acierto"].items():
        lineas.append(f"{n:22s}" + "".join(f"{r['media'][dd]:>8.3f}" for dd in D) + f"{r['gruesos-reales']:>9.3f}")
    lineas += ["", "== la PEOR de las 8 direcciones =="]
    for n, r in salida["acierto"].items():
        lineas.append(f"{n:22s}" + "".join(f"{r['peor'][dd]:>8.3f}" for dd in D))
    (AQUI / "resultados-desplazamiento.txt").write_text("\n".join(lineas) + "\n")
    print("\n".join(lineas))


if __name__ == "__main__":
    main()
