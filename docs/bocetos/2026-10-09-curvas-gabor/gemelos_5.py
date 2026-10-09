#!/usr/bin/env python3
"""Los «5» de ENTRENAMIENTO que se parecen al 5 que falla (dígito 1018 de val), y por qué ellos salen 5 y éste 9
(pedido del dueño, 2026-10-09: «algunos son idénticos, según un humano, al 5 que falló»).

    8-gemelos-1-parecidos.png   los 18 «5» de entrenamiento ordenados por cuánta tinta comparten con este (IoU)
    8-gemelos-2-diferencia.png  este contra sus dos gemelos: qué píxeles cambian y cuánto voto mueve cada celda
    8-gemelos-3-desplazar.png   el mismo dígito movido ±2 px: qué lee el compositor en cada posición

El compositor es el COMBINADO de la semilla 0, reproducido con `por_que_5.py`.

    python gemelos_5.py   → imagenes/8-gemelos-*.png · resultados-gemelos.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.colors import ListedColormap, TwoSlopeNorm          # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parents[2]))
import digitos as D                                                 # noqa: E402
import por_que_5 as P                                               # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

IMG = AQUI / "imagenes"
SUP, T1, T2, NAR, AZU = P.SUP, P.T1, P.T2, P.NAR, P.AZU
I_FALLO, A, B = 1018, 5, 9
MOV = 3              # el IoU «con desplazamiento» prueba ±3 px


def iou(a, c):
    return float((a & c).sum() / (a | c).sum())


def mover(a, dy, dx):
    """Desplaza sin dar la vuelta (lo que sale por un borde se pierde, entra blanco)."""
    out = np.zeros_like(a)
    ys, yd = (slice(0, 32 - dy), slice(dy, 32)) if dy >= 0 else (slice(-dy, 32), slice(0, 32 + dy))
    xs, xd = (slice(0, 32 - dx), slice(dx, 32)) if dx >= 0 else (slice(-dx, 32), slice(0, 32 + dx))
    out[yd, xd] = a[ys, xs]
    return out


def main() -> int:
    torch.set_num_threads(2); IMG.mkdir(exist_ok=True); P.estilo()
    d = dict(np.load(exigir_dataset(D.DIGITOS) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va = np.flatnonzero(part == "train"), np.flatnonzero(part == "val")
    X = P.caract(img); W, b, mu, sd = P.compositor(X[tr], y[tr], 0)

    def logits(XX):
        return ((XX - mu) / sd) @ W.T + b
    pv = logits(X[va]).argmax(1); assert (pv != y[va]).sum() == 68, "no reproduce el compositor"
    x = img[I_FALLO]; lx = logits(X[[I_FALLO]])[0]
    out = {"fallo": I_FALLO, "margen_9_menos_5": round(float(lx[B] - lx[A]), 2),
           "acierto_entreno": float((logits(X[tr]).argmax(1) == y[tr]).mean())}

    # ─── 1 · los 18 «5» por parecido ──────────────────────────────────────────────────────────────────────────────
    t5 = tr[y[tr] == A]; Lt = logits(X[t5])
    filas = []
    for k, i in enumerate(t5):
        mejor = max((iou(x, mover(img[i], dy, dx)), dy, dx) for dy in range(-MOV, MOV + 1) for dx in range(-MOV, MOV + 1))
        filas.append({"indice": int(i), "iou": round(iou(x, img[i]), 3), "iou_movido": round(mejor[0], 3),
                      "mejor_dy_dx": [mejor[1], mejor[2]], "margen_9_menos_5": round(float(Lt[k, B] - Lt[k, A]), 2),
                      "pred": int(Lt[k].argmax())})
    filas.sort(key=lambda f: -f["iou"]); out["cincos_de_entreno"] = filas
    gemelos = [f["indice"] for f in filas[:2]]
    fig = plt.figure(figsize=(15, 6.6))
    gs = fig.add_gridspec(2, 11, hspace=0.55, wspace=0.12)
    a = fig.add_subplot(gs[0, 0]); a.imshow(x, cmap="gray_r", vmin=0, vmax=1); P.limpio(a)
    a.set_title(f"EL QUE FALLA\nval {I_FALLO} → 9\n9−5 = {lx[B] - lx[A]:+.1f}", color=NAR, fontsize=8)
    for k, f in enumerate(filas):
        a = fig.add_subplot(gs[k // 10, 1 + k % 10] if k < 10 else gs[1, 1 + (k - 10)])
        a.imshow(img[f["indice"]], cmap="gray_r", vmin=0, vmax=1); P.limpio(a)
        a.contour(x.astype(float), levels=[0.5], colors=[NAR], linewidths=0.9)
        a.set_title(f"entreno {f['indice']}\nIoU {f['iou']:.2f} · 9−5 {f['margen_9_menos_5']:+.1f}", fontsize=7,
                    color=AZU if k < 2 else T1, fontweight="bold" if k < 2 else "normal")
        if k < 2:
            for s in a.spines.values():
                s.set_edgecolor(AZU); s.set_linewidth(2.5)
    fig.suptitle("1 · Los 18 «5» de entrenamiento, ordenados por cuánta tinta comparten con el que falla (IoU = intersección / unión).\n"
                 "Naranja: el contorno del que falla, encima. Recuadro azul: los dos gemelos. LOS 18 salen «5» en entrenamiento "
                 "(9−5 negativo = vota 5).", color=T1, fontsize=10)
    fig.savefig(IMG / "8-gemelos-1-parecidos.png", dpi=105, bbox_inches="tight"); plt.close(fig)
    print("→", IMG / "8-gemelos-1-parecidos.png")

    # ─── 2 · este contra cada gemelo: Δ(9−5) = Σ (W9 − W5)·(z_este − z_gemelo) ─────────────────────────────────────
    dW = W[B] - W[A]; zx = (X[I_FALLO] - mu) / sd
    fig = plt.figure(figsize=(15, 7.4))
    gs = fig.add_gridspec(2, 5, hspace=0.45, wspace=0.3, width_ratios=[1, 1, 1, 1, 1.9])
    out["diferencia"] = []
    cmap_dif = ListedColormap(["#ffffff", NAR, AZU, "#3a3a38"])
    for r, g in enumerate(gemelos):
        zg = (X[g] - mu) / sd; lg = logits(X[[g]])[0]
        dv = dW * (zx - zg)                                    # suma = (9−5 de este) − (9−5 del gemelo)
        celda = dv.reshape(11, 8, 8).sum(0); canal = dv.reshape(11, 64).sum(1)
        a = fig.add_subplot(gs[r, 0]); a.imshow(img[g], cmap="gray_r", vmin=0, vmax=1); P.limpio(a)
        a.set_title(f"GEMELO: entreno {g}\n→ {lg.argmax()} · 9−5 = {lg[B] - lg[A]:+.1f}", color=AZU)
        a = fig.add_subplot(gs[r, 1]); a.imshow(x, cmap="gray_r", vmin=0, vmax=1); P.limpio(a)
        a.set_title(f"EL QUE FALLA: val {I_FALLO}\n→ 9 · 9−5 = {lx[B] - lx[A]:+.1f}", color=NAR)
        a = fig.add_subplot(gs[r, 2])
        cod = (x > 0).astype(int) * 1 + (img[g] > 0).astype(int) * 2   # 1 sólo éste · 2 sólo gemelo · 3 los dos
        a.imshow(cod, cmap=cmap_dif, vmin=0, vmax=3); P.limpio(a)
        a.set_title(f"píxeles: naranja sólo en el que falla\nazul sólo en el gemelo · IoU {iou(x, img[g]):.2f}", fontsize=8)
        a = fig.add_subplot(gs[r, 3]); mx = np.abs(celda).max()
        P.sobre_digito(a, x, celda, TwoSlopeNorm(0, -mx, mx))
        a.set_title(f"cuánto voto 9 GANA cada celda\nal pasar del gemelo a éste\n(suma {dv.sum():+.1f})", fontsize=8)
        a = fig.add_subplot(gs[r, 4])
        nombres = [n.replace("rectas ", "").replace(" · ", " ") for n in P.CANALES]
        a.barh(range(11), canal, color=[NAR if v > 0 else AZU for v in canal]); a.set_yticks(range(11))
        a.set_yticklabels(nombres, fontsize=7); a.invert_yaxis(); a.axvline(0, color=T2, lw=0.8)
        a.spines[["top", "right"]].set_visible(False)
        a.set_title(f"por canal: {lg[B] - lg[A]:+.1f} (gemelo) + {dv.sum():+.1f} = {lx[B] - lx[A]:+.1f} (el que falla)", fontsize=8, loc="left")
        out["diferencia"].append({"gemelo": int(g), "margen_gemelo": round(float(lg[B] - lg[A]), 2), "delta": round(float(dv.sum()), 2),
                                  "por_canal": {n: round(float(v), 2) for n, v in zip(nombres, canal)}})
    fig.suptitle("2 · ¿Por qué el gemelo sale 5 y éste 9?  El compositor es lineal, así que la diferencia de (logit 9 − logit 5) entre los "
                 "dos\nes exactamente la suma, celda a celda, de (W₉ − W₅)·(z_este − z_gemelo). Naranja: empuja a 9 · azul: empuja a 5.",
                 color=T1, fontsize=10)
    fig.savefig(IMG / "8-gemelos-2-diferencia.png", dpi=105, bbox_inches="tight"); plt.close(fig)
    print("→", IMG / "8-gemelos-2-diferencia.png")

    # ─── 3 · el mismo dígito, movido ──────────────────────────────────────────────────────────────────────────────
    R = 2
    movidos = np.stack([mover(x, dy, dx) for dy in range(-R, R + 1) for dx in range(-R, R + 1)])
    Lm = logits(P.caract(movidos))
    fig, axs = plt.subplots(2 * R + 1, 2 * R + 1, figsize=(9, 10))
    rej = {}
    for n, (dy, dx) in enumerate([(dy, dx) for dy in range(-R, R + 1) for dx in range(-R, R + 1)]):
        a = axs[dy + R, dx + R]; pr = int(Lm[n].argmax()); mg = Lm[n, B] - Lm[n, A]
        a.imshow(movidos[n], cmap="gray_r", vmin=0, vmax=1); P.limpio(a)
        for c in range(4, 32, 4):                                        # la rejilla de celdas 4×4 del max-pool
            a.axhline(c - 0.5, color="#bdbcb8", lw=0.4); a.axvline(c - 0.5, color="#bdbcb8", lw=0.4)
        col = AZU if pr == A else (NAR if pr == B else T1)
        a.set_title(f"{'ORIGINAL  ' if dy == dx == 0 else ''}dy {dy:+d} · dx {dx:+d}\n→ {pr}  (9−5 {mg:+.1f})", color=col, fontsize=8,
                    fontweight="bold" if dy == dx == 0 else "normal")
        for s in a.spines.values():
            s.set_edgecolor(col); s.set_linewidth(2.2 if dy == dx == 0 else 1.2)
        rej[f"{dy:+d},{dx:+d}"] = {"pred": pr, "margen": round(float(mg), 2)}
    out["desplazado"] = rej
    # ¿es este dígito una rareza? cuántos de val cambian de lectura al moverse 1 px EN HORIZONTAL.
    # Sólo horizontal: los dígitos ocupan los 32 px de alto (1555 de 1617 en val), así que un desplazamiento vertical
    # CORTA tinta por un borde y ya no es el mismo dígito movido. En horizontal hay margen (este ocupa columnas 6–25).
    n_val = len(va); p0 = logits(X[va]).argmax(1)
    ph = np.stack([logits(P.caract(np.stack([mover(img[i], 0, dx) for i in va]))).argmax(1) for dx in (-1, 1)])
    cambia = (ph != p0).any(0); fallo = p0 != y[va]
    out["val_1px_horizontal"] = {"cambian_de_lectura": int(cambia.sum()), "de": n_val,
                                 "acierto_original": round(float((~fallo).mean()), 4),
                                 "acierto_movido_izq_der": [round(float((q == y[va]).mean()), 4) for q in ph],
                                 "fallos_que_un_px_arregla": int(((ph == y[va]).any(0) & fallo).sum()), "fallos": int(fallo.sum()),
                                 "aciertos_que_un_px_estropea": int(((ph != y[va]).any(0) & ~fallo).sum())}
    v = out["val_1px_horizontal"]
    fig.suptitle(f"3 · El MISMO dígito movido de 1 en 1 px (gris: la rejilla de celdas 4×4 donde se toma el máximo). "
                 f"Azul: lo lee 5 · naranja: 9.\n⚠ Las filas de arriba y abajo CORTAN tinta (el dígito ocupa los 32 px de alto): "
                 f"la prueba limpia es la FILA DEL MEDIO.\nEn todo val, {v['cambian_de_lectura']} de {n_val} dígitos "
                 f"({100 * v['cambian_de_lectura'] / n_val:.1f} %) cambian de lectura al moverlos 1 px a la izquierda o a la derecha.",
                 color=T1, fontsize=10)
    fig.savefig(IMG / "8-gemelos-3-desplazar.png", dpi=105, bbox_inches="tight"); plt.close(fig)
    print("→", IMG / "8-gemelos-3-desplazar.png")
    (AQUI / "resultados-gemelos.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out["val_1px_horizontal"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
