#!/usr/bin/env python3
"""Las figuras de `feat-agr` (REGLAS.md § Salidas). Leen los grupos (nn/entrenar_local.py) y la lectura (nn/leer.py); la
etiqueta sólo se usa para ESCRIBIR la mezcla de cada grupo, nunca para ordenarlo.

    python nn/figuras.py      → resultados/*.png  (presupuesto: ≤ 150 KB cada una)

  grupos-<brazo>-K30.png     Z32, Z8, X8: una fila por grupo en el orden del árbol de Ward; tamaño, mezcla de etiquetas,
                             el medoide (doble, con marco) y 15 miembros al azar
  enciende-<brazo>-K30.png   Z32, Z8: por grupo, qué fracción de sus miembros enciende cada detector en cada zona (3×3)
  arbol-<brazo>-K30.png      Z32, Z8: el árbol de Ward de los 30 grupos
  los-1.png                  todos los «grupos del 1» de Z32, Z8 y X8, con sus 1 ordenados por bandera; y los 24 unos con
                             más bandera, con el grupo en que cae cada uno
  consistencia.png           κ por brazo y transformación (L4)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import datos                                    # noqa: E402
import detectores as D                          # noqa: E402
import entrenar_local as E                      # noqa: E402
import metricas as M                            # noqa: E402
import representar as R                         # noqa: E402

RES = AQUI.parent / "resultados"
K0 = 30
# La paleta de referencia (skill dataviz): superficie, tinta y los tres primeros huecos categóricos (validados: pasan los
# seis controles; el aqua queda < 3:1 de contraste, así que los valores van además en la tabla del README).
SUP, T1, T2, REJ = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
GLIFOS = ("(", ")", "∪", "∩", "|", "—", "/", "\\", "O", "└", "┘", "┌", "┐")   # el dibujo de cada feature, en el orden de FAMILIAS


def fuente(t: int):
    for nombre in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(nombre, t)
        except OSError:
            continue
    return ImageFont.load_default()


def particion(r: dict, brazo: str) -> dict:
    p = E.principal("a", brazo, K0); lab = p["grupos"]; X = R.rep(r, brazo)
    tam = np.bincount(lab, minlength=K0)
    Z = M.ward(p["centroides"], tam)
    return {"grupos": lab, "tam": tam, "arbol": Z, "orden": M.orden_hojas(Z, K0),
            "medoides": M.medoides(X, p["centroides"], lab), "semilla": p["semilla"]}


def mezcla(lab: np.ndarray, y: np.ndarray, g: int, n: int = 3) -> str:
    c = np.bincount(y[lab == g], minlength=10); t = c.sum()
    return "  ".join(f"{d}: {100 * c[d] / t:.0f}%" for d in np.argsort(-c, kind="stable")[:n] if c[d] / t >= 0.03)


def baldosa(x: np.ndarray, lado: int) -> Image.Image:
    return Image.fromarray(((1 - x) * 255).astype(np.uint8)).resize((lado, lado), Image.NEAREST)


def guardar(im: Image.Image, nombre: str) -> None:
    ruta = RES / nombre
    im.save(ruta, optimize=True)
    print(f"  {nombre}: {ruta.stat().st_size / 1024:.0f} KB")


def rejilla(r, brazo, x32, y) -> None:
    pt = particion(r, brazo); f1, f2 = fuente(15), fuente(13)
    n_mi, lado, sep, txt = 15, 32, 3, 210
    fila = 2 * lado + 2 * sep
    W = txt + 2 * lado + 2 * sep + n_mi * (lado + sep) + sep
    H = 34 + K0 * fila
    im = Image.new("L", (W, H), 255); d = ImageDraw.Draw(im)
    d.text((sep, 8), f"{brazo} · K = 30 · semilla {pt['semilla']} (la de menor inercia) · filas en el orden del árbol de Ward",
           fill=0, font=f1)
    for k, g in enumerate(pt["orden"]):
        y0 = 34 + k * fila
        d.text((sep, y0 + 6), f"g{g:02d}  ·  {pt['tam'][g]} dígitos", fill=0, font=f1)
        d.text((sep, y0 + 28), mezcla(pt["grupos"], y, g), fill=80, font=f2)
        m = pt["medoides"][g]
        im.paste(baldosa(x32[m, 0], 2 * lado), (txt, y0))
        d.rectangle([txt - 1, y0 - 1, txt + 2 * lado, y0 + 2 * lado], outline=120)
        miembros = np.flatnonzero(pt["grupos"] == g)
        elegidos = np.random.default_rng(g).choice(miembros, min(n_mi, len(miembros)), replace=False)
        for j, i in enumerate(elegidos):
            im.paste(baldosa(x32[i, 0], lado), (txt + 2 * lado + 2 * sep + j * (lado + sep), y0 + lado // 2))
    guardar(im, f"grupos-{brazo}-K30.png")


def enciende(brazo, x32, y) -> None:
    js = json.loads((RES / f"grupos-{brazo}-K30.json").read_text(encoding="utf-8"))
    lab = np.array(js["grupo_de_cada_digito"]); enc = np.array(js["enciende"])        # (30, 13, 9)
    f1, f2 = fuente(15), fuente(13)
    cel, sep, txt = 8, 6, 250
    t3 = 3 * cel
    W = txt + 32 + sep + len(D.FAMILIAS) * (t3 + sep) + sep
    fila = max(32, t3) + 8
    H = 60 + K0 * fila
    im = Image.new("L", (W, H), 255); d = ImageDraw.Draw(im)
    d.text((sep, 6), f"{brazo}: fracción de cada grupo que enciende cada detector, por zona (negro = todos)", fill=0, font=f1)
    for j, gl in enumerate(GLIFOS):
        d.text((txt + 32 + sep + j * (t3 + sep) + 6, 32), gl, fill=0, font=f1)
    for k, g in enumerate(js["orden_del_arbol"]):
        y0 = 60 + k * fila
        d.text((sep, y0 + 2), f"g{g:02d} · {js['tamanos'][g]}", fill=0, font=f2)
        d.text((sep + 80, y0 + 2), mezcla(lab, y, g, 2), fill=80, font=f2)
        im.paste(baldosa(x32[js["medoides"][g], 0], 32), (txt, y0))
        for j in range(len(D.FAMILIAS)):
            t = Image.fromarray(((1 - enc[g, j].reshape(3, 3)) * 255).astype(np.uint8)).resize((t3, t3), Image.NEAREST)
            x0 = txt + 32 + sep + j * (t3 + sep)
            im.paste(t, (x0, y0 + 4)); d.rectangle([x0 - 1, y0 + 3, x0 + t3, y0 + 4 + t3], outline=190)
    guardar(im, f"enciende-{brazo}-K30.png")


def arbol(r, brazo, y) -> None:
    import matplotlib                                                # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                                  # noqa: PLC0415
    pt = particion(r, brazo); Z = pt["arbol"]; orden = pt["orden"]
    pos = {g: i for i, g in enumerate(orden)}; alt = {g: 0.0 for g in range(K0)}
    fig, ax = plt.subplots(figsize=(7.2, 8.6), dpi=100, facecolor=SUP)
    ax.set_facecolor(SUP)
    for i, (a, b, h, _) in enumerate(Z):
        n = K0 + i
        for hijo in (a, b):
            ax.plot([alt[hijo], h], [pos[hijo], pos[hijo]], color=T2, lw=1.4, solid_capstyle="round")
        ax.plot([h, h], [pos[a], pos[b]], color=T2, lw=1.4, solid_capstyle="round")
        pos[n], alt[n] = (pos[a] + pos[b]) / 2, h
    ax.set_yticks(range(K0))
    ax.set_yticklabels([f"g{g:02d} · {pt['tam'][g]:>3} · {mezcla(pt['grupos'], y, g, 2)}" for g in orden], fontsize=8.5, color=T1)
    ax.invert_yaxis(); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("distancia de Ward (√(2·coste) de la unión)", color=T2, fontsize=9)
    ax.tick_params(axis="x", colors=T2, labelsize=8)
    ax.grid(axis="x", color=REJ, lw=1); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(REJ)
    ax.set_title(f"Árbol de Ward de los 30 grupos — {brazo} (K = 30)", color=T1, fontsize=11, loc="left")
    fig.tight_layout(); fig.savefig(RES / f"arbol-{brazo}-K30.png", facecolor=SUP); plt.close(fig)
    print(f"  arbol-{brazo}-K30.png: {(RES / f'arbol-{brazo}-K30.png').stat().st_size / 1024:.0f} KB")


def los_unos(r, x32, y, lect) -> None:
    import leer                                                      # noqa: PLC0415
    m1 = np.flatnonzero(y == 1)
    b = np.zeros(len(y)); b[m1] = leer.bandera_corregida(x32[m1])   # ⚠ el índice A POSTERIORI (ver leer.py)
    f1, f2 = fuente(15), fuente(12)
    lado, sep, txt, n_mi = 32, 3, 360, 20
    filas = []
    for brazo in ("Z32", "Z8", "X8"):
        lab = E.principal("a", brazo, K0)["grupos"]
        for g in lect["L2"][brazo]["1"]["grupos_de_c"]:
            idx = np.flatnonzero((y == 1) & (lab == g["grupo"]))
            idx = idx[np.argsort(-b[idx], kind="stable")]
            filas.append((f"{brazo} g{g['grupo']:02d} · {g['n_c']} unos · mayoría {g['mayoria']} ({100 * g['frac_mayoria']:.0f}%)",
                          f"b′ {b[idx].mean():+.1f} (b {g['bandera_medio']:+.1f}) · grosor {g['grosor_medio']:.1f} · "
                          f"incl. {g['inclinacion_medio']:+.1f}°", idx[:n_mi], None))
    top = m1[np.argsort(-b[m1], kind="stable")[:n_mi]]
    labz = E.principal("a", "Z32", K0)["grupos"]
    filas.append(("Los 20 unos con MÁS bandera corregida b′", "debajo: su grupo en Z32", top, labz))
    alto = lado + 18
    W = txt + n_mi * (lado + sep) + sep; H = 34 + len(filas) * (alto + 8)
    im = Image.new("L", (W, H), 255); d = ImageDraw.Draw(im)
    d.text((sep, 8), "Los «grupos del 1» (≥ 10 % de los 1) de cada brazo, con sus 1 ordenados por la bandera corregida b′ "
                     "(a posteriori; debajo, b′ en px)", fill=0, font=f1)
    for k, (t1, t2, idx, labs) in enumerate(filas):
        y0 = 34 + k * (alto + 8)
        d.text((sep, y0 + 2), t1, fill=0, font=f2); d.text((sep, y0 + 20), t2, fill=80, font=f2)
        for j, i in enumerate(idx):
            x0 = txt + j * (lado + sep)
            im.paste(baldosa(x32[i, 0], lado), (x0, y0))
            d.text((x0 + 2, y0 + lado + 2), f"g{labs[i]:02d}" if labs is not None else f"{b[i]:+.0f}", fill=60, font=f2)
    guardar(im, "los-1.png")


def auditoria_unos(x32, y) -> None:
    """⚠ A POSTERIORI: 40 unos al azar (semilla 0) de cada grupo de Z32 con ≥ 30 unos, numerados, para CONTAR A OJO cuántos
    tienen bandera. Se hizo porque los dos índices de píxeles (b y b′) resultaron no medir la bandera (README)."""
    lab = E.principal("a", "Z32", K0)["grupos"]; f1, f2 = fuente(15), fuente(11)
    gs = [g for g in range(K0) if ((lab == g) & (y == 1)).sum() >= 30]
    lado, sep, txt, n = 36, 4, 150, 40
    por_fila = 20
    filas = [(g, np.random.default_rng(0).choice(np.flatnonzero((lab == g) & (y == 1)),
                                                  min(n, int(((lab == g) & (y == 1)).sum())), replace=False)) for g in gs]
    H = 34 + sum(2 * (lado + 16) + 10 for _ in filas); W = txt + por_fila * (lado + sep) + sep
    im = Image.new("L", (W, H), 255); d = ImageDraw.Draw(im)
    d.text((4, 8), "Z32: 40 unos al azar de cada grupo con ≥ 30 unos (para contar a ojo cuántos tienen bandera)", fill=0, font=f1)
    y0 = 34
    for g, idx in filas:
        c = np.bincount(y[lab == g], minlength=10)
        d.text((4, y0 + 4), f"g{g:02d}", fill=0, font=f1)
        d.text((4, y0 + 24), f"{int(c[1])} unos de {int(c.sum())}", fill=80, font=f2)
        d.text((4, y0 + 38), f"mayoría {int(c.argmax())}", fill=80, font=f2)
        for j, i in enumerate(idx):
            x0 = txt + (j % por_fila) * (lado + sep); yy = y0 + (j // por_fila) * (lado + 16)
            im.paste(baldosa(x32[i, 0], lado), (x0, yy)); d.text((x0 + 10, yy + lado + 1), str(j + 1), fill=110, font=f2)
        y0 += 2 * (lado + 16) + 10
    guardar(im, "auditoria-1-Z32.png")


def consistencia(lect) -> None:
    import matplotlib                                                # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                                  # noqa: PLC0415
    brazos = ("Z32", "Z8", "P32", "M32", "X8"); trans = ("desplazar", "engrosar", "adelgazar")
    nombres = {"desplazar": "desplazar 4 px", "engrosar": "engrosar 2 px", "adelgazar": "adelgazar 1 px"}
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=100, facecolor=SUP); ax.set_facecolor(SUP)
    ancho = 0.24
    for j, t in enumerate(trans):
        v = [lect["L4"][b]["principal"][t]["kappa"] for b in brazos]
        ax.bar(np.arange(len(brazos)) + (j - 1) * (ancho + 0.02), v, ancho, color=SERIES[j], label=nombres[t], zorder=3)
    ax.set_xticks(range(len(brazos))); ax.set_xticklabels(brazos, color=T1, fontsize=10)
    ax.set_ylim(0, 1); ax.set_ylabel("κ (consistencia corregida por azar)", color=T2, fontsize=9)
    ax.tick_params(axis="y", colors=T2, labelsize=8); ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=REJ, lw=1, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(REJ)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.13), labelcolor=T1)
    ax.set_title("¿El mismo dígito, transformado, cae en su grupo? (K = 30)", color=T1, fontsize=11, loc="left", pad=28)
    fig.tight_layout(); fig.savefig(RES / "consistencia.png", facecolor=SUP); plt.close(fig)
    print(f"  consistencia.png: {(RES / 'consistencia.png').stat().st_size / 1024:.0f} KB")


def main() -> int:
    r = R.cargar(); y = datos.etiquetas(); x32 = datos.cargar()["x32"]
    lect = json.loads((RES / "lecturas.json").read_text(encoding="utf-8"))
    for brazo in ("Z32", "Z8", "X8"):
        rejilla(r, brazo, x32, y)
    for brazo in ("Z32", "Z8"):
        enciende(brazo, x32, y)
        arbol(r, brazo, y)
    los_unos(r, x32, y, lect)
    auditoria_unos(x32, y)
    consistencia(lect)
    total = sum(p.stat().st_size for p in RES.glob("*.png"))
    print(f"total de figuras: {total / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
