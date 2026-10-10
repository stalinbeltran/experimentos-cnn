#!/usr/bin/env python3
"""El BORDE se saca con un filtro GABOR, no con morfología (regla del dueño, 2026-10-10).

*«No hagas más la obtención de bordes que haces. Reemplázalo con filtro Gabor, prueba varios si es necesario. Prueba
sin suavizar y luego suavizado, aunque sospecho que con Gabor podría no hacer falta el suavizado.»*

Hasta hoy el borde era `bordes.bordes`: x AND NOT erosión₃ₓ₃(x), un contorno binario de 1 px. Desde hoy NO se usa.
El Gabor que mira LÍNEAS es el PAR (coseno); el que mira BORDES es el IMPAR (seno): responde a un escalón de claro a
oscuro, con el signo según el lado. Aquí:

    dígito ─► banco de Gabor IMPAR (N orientaciones, λ_b, K_b×K_b) ─► borde = max_i |o_i| / (respuesta a un escalón
           ideal) ─► [binarizado en 0,5, opcional] ─► el detector de curvas de `bordes.py` sin tocar (Gabor par λ = 3)

El borde sale GRIS y en la FRONTERA entre píxeles (2 px de ancho), no binario y de 1 px: es un cambio de entrada para el
detector, que se calibró sobre el contorno morfológico. Por eso se prueban varios λ_b y las dos formas (gris y
binarizado). El suavizado se aplica al dígito ANTES del Gabor, como gaussiana σ y SIN re-binarizar (el Gabor ya acepta
grises; re-binarizar sólo hacía falta para la erosión).

PREDICCIÓN (escrita antes de correrlo, 2026-10-10):
    · sin suavizar, el mejor Gabor queda cerca del morfológico (0,88–0,90 en val): en la galería ve las mismas curvas
      pero sobrestima el radio en trazos finos, porque sus dos bordes se juntan en una banda;
    · el suavizado ayuda MENOS que con la erosión (+0,036 allí): ≤ +0,02. El Gabor ya integra sobre su ventana, que es
      justo lo que el suavizado le daba a la erosión — la sospecha del dueño.

    python bordes_gabor.py   → imagenes/17-gabor-*.png · resultados-bordes-gabor.json  (~10 min)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI)); sys.path.insert(0, str(AQUI.parents[2]))
import curvas as C                                                  # noqa: E402
import bordes as B                                                  # noqa: E402
import digitos as D                                                 # noqa: E402
import digitos_bordes as DB                                         # noqa: E402
import galeria                                                      # noqa: E402
import suavizado as SV                                              # noqa: E402
from curvas import plt                                              # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

K_B, N_B = 5, 8                     # kernel 5×5 y 8 orientaciones: los mejores del barrido de la galería (abajo)
LAMS = (3.0, 4.0, 6.0)
SIGMAS = (0.0, 1.0, 1.5)
VERDE, NAR, AZU = "#1b7f3b", "#c2410c", "#2a78d6"


def kernel_impar(k: int, ang: float, lam: float) -> np.ndarray:
    """El Gabor IMPAR: la misma envolvente que `curvas.kernel_gabor`, con seno en vez de coseno (media cero ya)."""
    r = (k - 1) / 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    t = np.deg2rad(ang)
    u = xx * np.cos(t) + yy * np.sin(t)
    v = -xx * np.sin(t) + yy * np.cos(t)
    g = np.exp(-(u ** 2) / (2 * (k / 3) ** 2) - (v ** 2) / (2 * (lam / 2) ** 2)) * np.sin(2 * np.pi * v / lam)
    return (g / np.linalg.norm(g)).astype(np.float32)


class BordeGabor:
    """x (32×32, 0–1) → borde en [0, 1]. Normalizado por la respuesta a un escalón ideal: un borde limpio vale ≈ 1."""

    def __init__(self, lam: float, k: int = K_B, n: int = N_B, binarizar: bool = False):
        self.lam, self.k, self.n, self.binarizar = lam, k, n, binarizar
        self.banco = torch.from_numpy(np.stack([kernel_impar(k, a, lam) for a in np.arange(n) * 180.0 / n])[:, None])
        paso = np.zeros((64, 64), np.float32); paso[:, 32:] = 1
        self.escala = float(F.conv2d(torch.from_numpy(paso)[None, None], self.banco, padding=k // 2)[0].abs().max())

    @torch.no_grad()
    def __call__(self, x: np.ndarray) -> np.ndarray:
        o = F.conv2d(torch.from_numpy(x.astype(np.float32))[None, None], self.banco, padding=self.k // 2)[0]
        e = np.clip(o.abs().max(0).values.numpy() / self.escala, 0, 1)
        return (e > 0.5).astype(np.float32) if self.binarizar else e

    def nombre(self) -> str:
        return f"Gabor impar λ {self.lam:g}" + (" · binarizado" if self.binarizar else " · gris")


def suave(x: np.ndarray, s: float) -> np.ndarray:
    """Gaussiana σ sobre el dígito, SIN re-binarizar (el Gabor acepta grises)."""
    if s == 0:
        return x.astype(np.float32)
    return gaussian_filter(x.astype(np.float32), (0, s, s) if x.ndim == 3 else s)


def galeria_puntos(borde) -> tuple[int, int, int, int]:
    a = r = na = nr = 0
    for grupo in galeria.trazos().values():
        for etq, x, esp, R, g, *_ in grupo[2]:
            _, kappa, _, vs = C.detectar(borde(x)); cur, _ = B.trozos(kappa, vs)
            if esp == "curva":
                na += 1; a += B.acierta("curva", R, g, cur, vs)
            else:
                nr += 1; r += bool(vs) and not cur
    return a, na, r, nr


def dibujar_detector(ax, e: np.ndarray, x: np.ndarray, titulo: str, color=None):
    c = C.campo(e); kappa, mask, emin = C.giro(c)
    vs = C.veredicto(c, kappa, mask, emin); curvos, _ = B.trozos(kappa, vs, c["theta"])
    med = mask & np.isfinite(kappa)
    ax.imshow(np.where(x > 0.5, 1.0, np.nan), cmap=ListedColormap(["#e3edf8"]), vmin=0, vmax=1)
    ax.imshow(np.where(mask & ~med, 1.0, np.nan), cmap=ListedColormap(["#a9a8a4"]), vmin=0, vmax=1)
    ax.imshow(np.where(med & (np.abs(kappa) < C.KAPPA_MIN), 1.0, np.nan), cmap="Greens", vmin=0, vmax=1.4)
    ax.imshow(np.where(med & (np.abs(kappa) >= C.KAPPA_MIN), np.abs(kappa), np.nan), cmap="Oranges", vmin=0, vmax=10)
    for t in curvos:
        (cx, cy), dd = t["centroide"], np.deg2rad(t["direccion"])
        ax.annotate("", xy=(cx - 0.5 + 4 * np.cos(dd), cy - 0.5 + 4 * np.sin(dd)), xytext=(cx - 0.5, cy - 0.5),
                    arrowprops=dict(arrowstyle="-|>", color=AZU, lw=1.1, mutation_scale=7))
    ax.set_title(titulo, fontsize=8.5, pad=2, color=color or C.T1); ax.set_xticks([]); ax.set_yticks([])


def main() -> int:
    torch.set_num_threads(2); t0 = time.time()
    B.configurar()                                     # el detector (Gabor PAR λ 3 + giro) sin tocar
    out: dict = {"protocolo": "el de digitos_bordes.py; sólo cambia cómo se saca el borde", "k_b": K_B, "n_b": N_B}

    # ── 1. la galería (32 trazos), para ver que el detector sigue viendo curvas y rectas ──
    gal = {"morfológico (YA NO SE USA)": galeria_puntos(B.bordes)}
    bordes_ = [BordeGabor(l, binarizar=b) for l in LAMS for b in (False, True)]
    for bg in bordes_:
        gal[bg.nombre()] = galeria_puntos(bg)
    for n, (a, na, r, nr) in gal.items():
        print(f"galería · {n:36s} arcos {a}/{na} · rectas {r}/{nr}", flush=True)
    out["galeria"] = {n: {"arcos": f"{a}/{na}", "rectas": f"{r}/{nr}"} for n, (a, na, r, nr) in gal.items()}

    # ── 2. los dígitos ──
    d = dict(np.load(exigir_dataset(D.DIGITOS) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va, ex = np.flatnonzero(part == "train"), np.flatnonzero(part == "val"), np.flatnonzero(part == "extra")

    def evaluar(X):
        accs, ciega, p0 = [], [], None
        for sem in D.SEMILLAS:
            pr = DB.entrenar(X[tr], y[tr], sem)
            pv = pr(X[va]); accs.append(float((pv == y[va]).mean())); ciega.append(float((pr(X[ex]) == y[ex]).mean()))
            if sem == 0:
                p0 = pv
        return {"val": round(float(np.mean(accs)), 4), "val_rango": [round(min(accs), 4), round(max(accs), 4)],
                "ciega": round(float(np.mean(ciega)), 4)}, p0

    res, preds = {}, {}
    # referencias: el borde morfológico, sin suavizar y con σ 1,5 re-binarizado (lo medido en suavizado.py)
    for etq, xs in (("morfológico · σ 0", img), ("morfológico · σ 1.5 (re-binarizado)", SV.suavizar(img, 1.5))):
        res[etq], preds[etq] = evaluar(DB.caract(xs)["bordes"])
        print(f"dígitos · {etq:44s} val {res[etq]['val']:.4f} · a ciegas {res[etq]['ciega']:.4f}  ({time.time() - t0:.0f} s)",
              flush=True)
    for bg in bordes_:
        for s in SIGMAS:
            etq = f"{bg.nombre()} · σ {s:g}"
            res[etq], preds[etq] = evaluar(DB.caract(suave(img, s), borde=bg)["bordes"])
            print(f"dígitos · {etq:44s} val {res[etq]['val']:.4f} {res[etq]['val_rango']} · a ciegas {res[etq]['ciega']:.4f}"
                  f"  ({time.time() - t0:.0f} s)", flush=True)
    base = preds["morfológico · σ 0"]
    fallos, aciertos = np.flatnonzero(base != y[va]), np.flatnonzero(base == y[va])
    for etq, p in preds.items():
        res[etq]["de_los_181_arreglados"] = int((p[fallos] == y[va][fallos]).sum())
        res[etq]["aciertos_que_se_rompen"] = int((p[aciertos] != y[va][aciertos]).sum())
    out["digitos"] = res
    gabor = [k for k in res if k.startswith("Gabor")]
    mejor = max(gabor, key=lambda k: res[k]["val"])
    mejor_s0 = max([k for k in gabor if k.endswith("σ 0")], key=lambda k: res[k]["val"])
    out["mejor_gabor"], out["mejor_gabor_sin_suavizar"] = mejor, mejor_s0
    print(f"mejor Gabor: {mejor} · sin suavizar: {mejor_s0}")
    (AQUI / "resultados-bordes-gabor.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # ── figura 1: tabla de resultados ──
    C._estilo()
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})
    a = axs[0]
    filas = [f"{BordeGabor(l, binarizar=b).nombre()}" for l in LAMS for b in (False, True)]
    M = np.array([[res[f"{f} · σ {s:g}"]["val"] for s in SIGMAS] for f in filas])
    a.imshow(M, cmap="Blues", vmin=0.84, vmax=0.94, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            etq = f"{filas[i]} · σ {SIGMAS[j]:g}"
            a.text(j, i, f"{M[i, j]:.3f}\na ciegas {res[etq]['ciega']:.3f}", ha="center", va="center", fontsize=8.5,
                   color="white" if M[i, j] > 0.905 else C.T1, fontweight="bold" if etq == mejor else "normal")
    a.set_xticks(range(len(SIGMAS))); a.set_xticklabels([f"σ {s:g}" + (" (sin suavizar)" if s == 0 else "") for s in SIGMAS])
    a.set_yticks(range(len(filas))); a.set_yticklabels(filas, fontsize=8.5)
    a.set_title("acierto en val (1617, 3 semillas) con el borde de Gabor impar · kernel 5×5, 8 orientaciones",
                loc="left", fontsize=9.5)
    a = axs[1]; a.axis("off")
    ref0, ref15 = res["morfológico · σ 0"], res["morfológico · σ 1.5 (re-binarizado)"]
    lineas = ["Referencias (borde morfológico, YA NO SE USA):",
              f"   sin suavizar        val {ref0['val']:.3f} · a ciegas {ref0['ciega']:.3f}",
              f"   σ 1,5 re-binarizado  val {ref15['val']:.3f} · a ciegas {ref15['ciega']:.3f}", "",
              f"Mejor Gabor: {mejor}", f"   val {res[mejor]['val']:.3f} · a ciegas {res[mejor]['ciega']:.3f}",
              f"   de los 181 fallos de antes: {res[mejor]['de_los_181_arreglados']} arreglados",
              f"   aciertos de antes rotos: {res[mejor]['aciertos_que_se_rompen']}", "",
              f"Mejor Gabor SIN suavizar: {mejor_s0}", f"   val {res[mejor_s0]['val']:.3f} · a ciegas {res[mejor_s0]['ciega']:.3f}", "",
              "Galería (32 trazos), arcos · rectas:"]
    lineas += [f"   {n}: {v['arcos']} · {v['rectas']}" for n, v in out["galeria"].items()]
    a.text(0, 1, "\n".join(lineas), va="top", fontsize=8.5, family="monospace", color=C.T1)
    fig.suptitle("17 · El borde con Gabor IMPAR en vez de morfología: dígitos y galería", fontsize=11.5, color=C.T1)
    fig.tight_layout(); fig.savefig(DB.IMG / "17-gabor-1-resultados.png", dpi=105); plt.close(fig)
    print("→ 17-gabor-1-resultados.png")

    # ── figura 2: qué ve el detector con cada borde, en los fallos frecuentes ──
    ejemplos = json.loads((AQUI / "resultados-suavizado.json").read_text())["ejemplos_val"]
    lam_m = float(mejor.split("λ ")[1].split(" ")[0]); bin_m = "binarizado" in mejor
    s_m = float(mejor.split("σ ")[-1])
    bg = BordeGabor(lam_m, binarizar=bin_m)
    pos = {int(i): k for k, i in enumerate(va)}
    cols = ["dígito", "borde morfológico\n(YA NO SE USA)", f"borde Gabor σ 0\n({bg.nombre()})", "detector · Gabor σ 0",
            f"borde Gabor σ {s_m:g}", f"detector · Gabor σ {s_m:g}"]
    fig, axs = plt.subplots(len(ejemplos), len(cols), figsize=(len(cols) * 1.9, len(ejemplos) * 1.95 + 0.7))
    for r_, i in enumerate(ejemplos):
        x = img[i]; j = pos[i]
        e0, es = bg(suave(x, 0)), bg(suave(x, s_m))
        p0, ps = preds[f"{bg.nombre()} · σ 0"][j], preds[mejor][j]
        axs[r_, 0].imshow(x, cmap="gray_r", vmin=0, vmax=1); axs[r_, 0].set_ylabel(f"real {y[i]}", fontsize=9)
        axs[r_, 0].set_title(f"morf. σ 0 lee {base[j]}", fontsize=8.5, pad=2, color=VERDE if base[j] == y[i] else NAR)
        axs[r_, 1].imshow(B.bordes(x), cmap="gray_r", vmin=0, vmax=1)
        axs[r_, 2].imshow(e0, cmap="gray_r", vmin=0, vmax=1)
        dibujar_detector(axs[r_, 3], e0, x, f"lee {p0}" + (" ✓" if p0 == y[i] else ""), VERDE if p0 == y[i] else NAR)
        axs[r_, 4].imshow(es, cmap="gray_r", vmin=0, vmax=1)
        dibujar_detector(axs[r_, 5], es, suave(x, s_m), f"lee {ps}" + (" ✓" if ps == y[i] else ""), VERDE if ps == y[i] else NAR)
        for c_ in range(len(cols)):
            axs[r_, c_].set_xticks([]); axs[r_, c_].set_yticks([])
            if r_ == 0:
                axs[r_, c_].text(0.5, 1.28, cols[c_], transform=axs[r_, c_].transAxes, ha="center", fontsize=8.5)
    fig.suptitle("17 · Un fallo de cada confusión frecuente: el borde morfológico (antes) contra el de Gabor impar (ahora)",
                 fontsize=11, color=C.T1, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(DB.IMG / "17-gabor-2-bordes.png", dpi=100); plt.close(fig)
    print("→ 17-gabor-2-bordes.png")
    print(f"total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
