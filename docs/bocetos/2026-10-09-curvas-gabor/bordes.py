#!/usr/bin/env python3
"""El detector de curvas sobre los BORDES de la imagen, con el filtro de rectas más estrecho que funciona (2026-10-10).

La idea del dueño, ese día: *«que se usen sólo los bordes de la imagen real»*. El detector no ve el trazo, ve su
contorno: un trazo grueso son dos curvas paralelas (la de fuera, de radio R + g/2, y la de dentro, R − g/2), y un trazo
fino es casi una sola. Encaja con lo que el Gabor estrecho ya hacía por su cuenta (§ «Por qué no las detecta» del README).

    0  BORDES    b = x AND NOT erosión₃ₓ₃(x): el contorno interior, de 1 px. El marco de la imagen NO es borde
                 (erosión con border_value = 1): un trazo cortado por el marco no gana un borde que no tiene
    1  BANCO     el mismo Gabor par de curvas.py, pero λ = 3 (franja central 1,5 px), 12 orientaciones, 9×9
    2  CAMPO     igual, salvo que la energía max_i r_i se SUAVIZA (σ = 1 px) antes de decidir dónde hay trazo:
                 con franja estrecha, un borde oblicuo de 1 px es una escalera y su energía sube y baja píxel a
                 píxel; sin suavizar, la máscara se rompe en trozos y no queda dónde medir el giro a ±4 px
    3  GIRO      igual que curvas.py
    4  TROZOS    el veredicto es por TROZO, no por componente: el contorno de un trazo es UNA componente cerrada
                 con sus dos lados y sus dos extremos, y lo que interesa son las curvas que lleva dentro. Un trozo
                 curvo es una racha de ≥ 4 px con |κ| ≥ 2 °/px, y su radio es 57,3 / mediana(|κ|)

Los cuatro números que cambian respecto de curvas.py, y de dónde salen (⚠ ELEGIDOS, no calibrados):
    λ = 3        el más estrecho que funciona. Barrido 2026-10-10 sobre los 32 trazos de la galería en bordes, con
                 λ ∈ {2; 2,5; 3; 4; 6}, K ∈ {5, 7, 9}, 12/24 orientaciones, σ ∈ {0,7; 1; 1,5}, TAU ∈ {0,3…1,2}:
                 λ = 2 no ve NINGÚN arco en ninguna combinación (0/22); λ = 3 ve 20/22
    σE = 1, TAU = 0,5    los mejores de ese barrido para λ = 3; con TAU 0,3–0,6 apenas cambia
    KAPPA_MIN = 2 °/px   (radio máx. ≈ 29 px) elegido mirando el banco de rect-lin: con 1 °/px la escalera de un
                 borde recto oblicuo se leía como curva de R ≈ 24–45 en el 19 % de las rectas finas. ⚠ Por eso el
                 banco YA NO es una prueba a ciegas para este número: el umbral se eligió viéndolo

    python bordes.py           → imagenes/11-bordes-*.png y resultados-bordes.json (los mismos 32 trazos de galeria.py)
    python bordes.py --banco   → además el banco de rect-lin entero, el detector original contra éste, en
                                 resultados-bordes-banco.txt (~1 min)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_erosion, gaussian_filter, label

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import curvas as cv                                                 # noqa: E402
import galeria                                                      # noqa: E402
from curvas import detectar, plt                                    # noqa: E402

LAM_B, SIGMA_E, TAU_B, KAPPA_MIN_B = 3.0, 1.0, 0.5, 2.0
OK, MAL = galeria.OK, galeria.MAL
_campo_original = cv.campo


def bordes(x: np.ndarray) -> np.ndarray:
    x = x.astype(bool)
    return (x & ~binary_erosion(x, np.ones((3, 3)), border_value=1)).astype(np.uint8)


def configurar() -> None:
    """Pasa curvas.py al modo bordes. El detector lee estos globales en cada llamada: basta con sustituirlos."""
    cv.LAM, cv.TAU, cv.KAPPA_MIN = LAM_B, TAU_B, KAPPA_MIN_B
    cv.GABOR = torch.from_numpy(np.stack([cv.kernel_gabor(cv.K, t, LAM_B) for t in cv.THETAS])[:, None])

    def campo(x):
        c = _campo_original(x)
        c["Emax"] = gaussian_filter(c["Emax"], SIGMA_E)
        return c
    cv.campo = campo


def trozos(kappa: np.ndarray, vs: list[dict]) -> tuple[list[dict], int]:
    """Trozos curvos (con su radio y su máscara) y número de trozos rectos, dentro del lomo de cada componente."""
    curvos, rectos = [], 0
    for v in vs:
        med = v["nucleo"] & np.isfinite(kappa)
        for curvo in (True, False):
            sel = med & ((np.abs(kappa) >= cv.KAPPA_MIN) if curvo else (np.abs(kappa) < cv.KAPPA_MIN))
            lab, n = label(sel, structure=np.ones((3, 3)))
            for z in range(1, n + 1):
                m = lab == z
                if m.sum() < cv.MIN_TROZO:
                    continue
                if curvo:
                    curvos.append({"radio": round(float(57.3 / np.median(np.abs(kappa[m]))), 1), "px": int(m.sum()), "m": m})
                else:
                    rectos += 1
    return sorted(curvos, key=lambda t: -t["px"]), rectos


def radios_de_borde(r: float, g: int) -> list[float]:
    """Los radios que tiene de verdad el contorno de un arco de radio r y grosor g (los < 3 px no se cuentan)."""
    return [o for o in (r + g / 2, r, r - g / 2) if o >= 3]


def acierta(esperado: str, r_real, g, curvos, hay) -> bool:
    if esperado == "recta":          # el contorno de una recta tiene lados rectos y extremos: ninguna curva
        return bool(hay) and not curvos
    return any(abs(np.log(t["radio"] / o)) < np.log(1.25) for t in curvos for o in radios_de_borde(r_real, g))


def figura(nombre_png: str, titulo: str, items) -> list[dict]:
    cv._estilo()
    n = len(items)
    fig, axs = plt.subplots(4, n, figsize=(1.55 * n + 1.2, 8.3))
    res = []
    for j, (etq, x, esperado, r_real, g) in enumerate(items):
        b = bordes(x)
        c, kappa, mask, vs = detectar(b)
        curvos, rectos = trozos(kappa, vs)
        bien = acierta(esperado, r_real, g, curvos, vs)
        a0, a1, a2, a3 = axs[:, j]
        a0.imshow(x, cmap="gray_r", vmin=0, vmax=1); a0.set_title(etq, fontsize=8.5)
        a1.imshow(b, cmap="gray_r", vmin=0, vmax=1)
        a2.imshow(c["Emax"], cmap="Greys", vmin=0, vmax=max(1.5, c["Emax"].max())); cv._ticks(a2, c, mask)
        a3.imshow(b, cmap="gray_r", vmin=0, vmax=1, alpha=0.12)
        a3.imshow(np.where(mask & ~np.isfinite(kappa), 1.0, np.nan), cmap="Greys", vmin=0, vmax=2.5)
        a3.imshow(np.where(np.isfinite(kappa), np.abs(kappa), np.nan), cmap="Oranges", vmin=0, vmax=12)
        for t in curvos:
            a3.contour(t["m"].astype(float), levels=[0.5], colors=[cv.AZU], linewidths=0.8)
        if curvos:
            txt = "CURVA R≈" + " · ".join(f"{t['radio']:.0f}" for t in curvos[:3])
        else:
            txt = f"SIN CURVAS · {rectos} recto(s)" if vs else "NADA"
        if r_real:
            txt += "\nbordes R " + " · ".join(f"{o:g}" for o in radios_de_borde(r_real, g))
        a3.set_xlabel(("✓ " if bien else "✗ ") + txt, fontsize=8, color=OK if bien else MAL, linespacing=1.1)
        for a in axs[:, j]:
            a.set_xticks([]); a.set_yticks([]); a.set_xlim(-0.5, x.shape[1] - 0.5); a.set_ylim(x.shape[0] - 0.5, -0.5)
        res.append({"trazo": etq, "esperado": esperado, "radio_real": r_real, "grosor": g, "acierta": bien,
                    "curvas": [{k: v for k, v in t.items() if k != "m"} for t in curvos], "trozos_rectos": rectos})
    for i, rot in enumerate(["1 · IMAGEN", "2 · BORDES\n(lo que entra)", "3 · FILTRO DE RECTAS\nλ = 3 (energía + orientación)",
                             "4 · DETECTOR\n|κ| °/px · curvas"]):
        axs[i, 0].set_ylabel(rot, fontsize=8.5, color=cv.T1)
    acier = sum(r["acierta"] for r in res)
    fig.suptitle(f"{titulo}  ·  SOBRE LOS BORDES   —   {acier}/{n} como se esperaba", color=cv.T1, fontsize=11)
    fig.text(0.5, 0.005, "fila 4: naranja = giro medido · gris = borde donde no se puede medir · contorno azul = cada trozo "
             "curvo, con su radio abajo · ✓ arco: algún trozo a ±25 % de un radio real del contorno (R ± g/2) · "
             "✓ recta: ningún trozo curvo", ha="center", fontsize=7.5, color=cv.T2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(cv.IMG / nombre_png, dpi=110); plt.close(fig)
    print("→", cv.IMG / nombre_png, f"{acier}/{n}")
    return res


def banco() -> str:
    """El banco de rect-lin con el MISMO criterio por trozos: el detector original sobre la imagen, y éste sobre bordes."""
    raiz = cv.exigir_dataset("rect-lin-banco-r20261008")
    b = dict(np.load(raiz / "datos.npz"))
    nombres = ("continua", "punteada", "curva", "ruido", "puntos-sueltos", "mancha")

    def pasar(usar_bordes: bool) -> dict:
        t = {}
        for i in range(len(b["imagenes"])):
            x = b["imagenes"][i]
            _, kappa, _, vs = detectar(bordes(x) if usar_bordes else x)
            curvos, _ = trozos(kappa, vs)
            tipo = nombres[int(b["tipo"][i])]
            if tipo == "curva":
                R, g = int(b["radio"][i]), int(b["grosor"][i])
                k, ok = f"arcos R {'≤ 27' if R <= 27 else '40'}", acierta("curva", R, g, curvos, vs)
            elif tipo == "continua":
                g = int(b["grosor"][i])
                k, ok = f"rectas grosor {'2–4' if g <= 4 else '6–8' if g <= 8 else '10–14'}", not curvos
            else:
                k, ok = tipo, not curvos
            s = t.setdefault(k, [0, 0]); s[0] += ok; s[1] += 1
        return t

    a = pasar(False)                 # antes de configurar: curvas.py tal cual (λ = 6, sobre la imagen)
    configurar()
    c = pasar(True)
    filas = ["| banco de rect-lin | n | qué cuenta como acierto | original (λ 6, imagen) | bordes (λ 3) |",
             "|---|---:|---|---:|---:|"]
    for k in sorted(a):
        que = "algún trozo a ±25 % de R, R ± g/2" if k.startswith("arcos") else "ningún trozo curvo"
        filas.append(f"| {k} | {a[k][1]} | {que} | {100 * a[k][0] / a[k][1]:.1f} % | {100 * c[k][0] / c[k][1]:.1f} % |")
    return "\n".join(filas)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--banco", action="store_true")
    args = ap.parse_args()
    if args.banco:
        txt = banco()
        (AQUI / "resultados-bordes-banco.txt").write_text(txt + "\n")
        print(txt)
    configurar()
    out = {k: figura(f"11-bordes-{png}.png", t, items) for k, (png, t, items) in galeria.trazos().items()}
    (AQUI / "resultados-bordes.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
