#!/usr/bin/env python3
"""rizo-2 — que las RECTAS cuenten MENOS (pedido por el dueño el 2026-10-12, tras `peso_grupos.py`: rectas y gris pesan
igual por feature, pero son 512 contra 384 y una recta larga enciende varias celdas seguidas). Tres opciones anotadas en el
README; ésta es la **3**: rectas en celdas 4×4 en vez de 8×8 (512 → 128), con el gris a 8×8 y el rizo igual. De paso, una
sub-variante: 4×4 y una sola escala (máximo de ×1 y ×2; 64).

Protocolo de `detectores.py`: 180 train (con aumento ±1–3 px, 13 copias) / 1617 val / 3823 a ciegas, MLP 256, 3 semillas.
Por variante: acierto val y ciega, cuota de peso y de sensibilidad por grupo, qué pasa con los 23 fallos de la figura 19
(cuántos se arreglan, cuántos nuevos) y qué lee del idx 784.

Opción **4** (`--opcion 4`): UN VOTO POR RECTA. Sobre el mapa integrado de cada orientación, un píxel sobrevive sólo si es
el máximo en una ventana alargada en la dirección de su recta (3 px de ancho, L px de largo): una recta larga deja un pico
cada L px en vez de encender todas las celdas por las que pasa. Mismo tamaño (512), más ralo. L = 15 (el integrador) y 31.

    python nn/menos_rectas.py              → resultados/23-menos-rectas.json · 23-menos-rectas.png  (~5 min)
    python nn/menos_rectas.py --opcion 4   → resultados/24-un-voto.json · 24-un-voto.png
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent)); sys.path.insert(0, str(AQUI.parent / "docs" / "bocetos" / "2026-10-09-curvas-gabor"))
import compositores as CO                                           # noqa: E402
import digitos_variantes as DV                                      # noqa: E402
import evaluar as EV                                                # noqa: E402
import rastrear as RA                                               # noqa: E402
import resistencia as R                                             # noqa: E402
import digitos as D                                                 # noqa: E402
import gris_recalibrado as GR                                       # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

IDX = 784
DIRS = {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1)}          # dirección de la cresta de cada Gabor (leída del kernel en rectas_dibujo.py)


@torch.no_grad()
def mapas_rectas(img):
    """Los mapas integrados (n, 4, H, W) a las dos escalas, antes de leerlos en celdas: lo que hace R.rectas por dentro."""
    xt = torch.from_numpy(img.astype(np.float32)[:, None]); out = []
    for e in range(2):
        xe = xt if e == 0 else torch.nn.functional.avg_pool2d(xt, 2 ** e)
        m = torch.nn.functional.relu(torch.nn.functional.conv2d(xe, D.GABOR4, padding=4)); out.append(torch.nn.functional.conv2d(m, D.INTEG, padding=D.L_INT // 2, groups=4))
    return out


@torch.no_grad()
def un_voto(m, L, ancho=1):
    """Supresión del no máximo con ventana alargada: para la orientación o, el píxel queda si es el máximo entre los
    desplazados t·dir (t = −L/2…L/2) y s·normal (s = −ancho…ancho). Devuelve el mapa con sólo los picos."""
    n, _, H, W = m.shape; out = torch.zeros_like(m); P = L // 2 + ancho + 1
    for o, (dx, dy) in DIRS.items():
        mo = torch.nn.functional.pad(m[:, o], (P, P, P, P)); mx = torch.full_like(m[:, o], -1.0)
        nx, ny = -dy, dx
        for t in range(-(L // 2), L // 2 + 1):
            for s_ in range(-ancho, ancho + 1):
                ox, oy = round(t * dx / np.hypot(dx, dy) + s_ * nx / np.hypot(nx, ny)), round(t * dy / np.hypot(dx, dy) + s_ * ny / np.hypot(nx, ny))
                if (ox, oy) == (0, 0):
                    continue
                mx = torch.maximum(mx, mo[:, P + oy:P + oy + H, P + ox:P + ox + W])
        out[:, o] = torch.where((m[:, o] > mx) & (m[:, o] > 0), m[:, o], torch.zeros_like(m[:, o]))
    return out


def rectas_un_voto(img, L, celdas=8):
    return torch.cat([torch.nn.functional.adaptive_max_pool2d(un_voto(m, max(3, L // 2 ** e)), celdas).flatten(1) for e, m in enumerate(mapas_rectas(img))], 1).numpy()   # a ×2 la ventana se escala con la imagen


def rectas_variante(img, nombre):
    if nombre == "8×8 (referencia)":
        return R.rectas(img, 8)
    if nombre == "4×4":
        return R.rectas(img, 4)
    if nombre == "4×4, una escala (máx ×1, ×2)":
        r = R.rectas(img, 4); return np.maximum(r[:, :64], r[:, 64:])
    if nombre.startswith("un voto, L = "):
        return rectas_un_voto(img, int(nombre.split("= ")[1]))
    raise ValueError(nombre)


OPCIONES = {"3": (["8×8 (referencia)", "4×4", "4×4, una escala (máx ×1, ×2)"], "23-menos-rectas", "opción 3: las rectas en celdas 4×4 en vez de 8×8 (gris 8×8 y rizo iguales)"),
            "4": (["8×8 (referencia)", "un voto, L = 15", "un voto, L = 31"], "24-un-voto", "opción 4: UN VOTO por recta (supresión del no máximo a lo largo de la recta; celdas 8×8)")}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--opcion", default="3", choices=list(OPCIONES)); args = ap.parse_args()
    VARIANTES, SALIDA, TITULO = OPCIONES[args.opcion]
    torch.set_num_threads(2); t0 = time.time(); GR.configurar(**R.CAL_GRIS)
    d = dict(np.load(exigir_dataset(EV.DATASET) / "datos.npz"))
    img, y, part = (d["imagenes"] > 0).astype(np.uint8), d["etiquetas"].astype(int), d["particion"]
    tr, va, ex = [np.flatnonzero(part == p) for p in ("train", "val", "extra")]
    cache = pickle.loads(R.CACHE.read_bytes()); todos = pickle.loads(DV.CACHE.read_bytes())      # 18: train movido · 15: los 5620 sin mover
    aum = [(0, 0)] + [(k, 0) for k in R.AUM] + [(0, k) for k in R.AUM]
    Itr = [R.mover(img[tr], *m) for m in aum]
    gris_tr = np.concatenate([R.gris(I, 8) for I in Itr]); rizo_tr = np.concatenate([CO.compacta(cache[("tr", m)], "rizo (h0, 8px)") for m in aum]); Y = np.concatenate([y[tr]] * len(aum))
    resto = {"va": (R.gris(img[va], 8), CO.compacta([todos[i] for i in va], "rizo (h0, 8px)")), "ex": (R.gris(img[ex], 8), CO.compacta([todos[i] for i in ex], "rizo (h0, 8px)"))}
    print(f"gris y rizo listos ({time.time() - t0:.0f} s)", flush=True)
    fallos19 = {f["idx"]: f for f in json.loads((AQUI / "resultados" / "19-fallos-mejor.json").read_text())["fallos"]}
    j784 = int(np.flatnonzero(va == IDX)[0]); out = {"variantes": {}}
    for nombre in VARIANTES:
        rtr = np.concatenate([rectas_variante(I, nombre) for I in Itr]); X = np.concatenate([rtr, gris_tr, rizo_tr], 1); n = rtr.shape[1]
        G = {"rectas": (0, n), "gris": (n, n + 384), "rizo": (n + 384, n + 389)}
        Xva = np.concatenate([rectas_variante(img[va], nombre), *resto["va"]], 1); Xex = np.concatenate([rectas_variante(img[ex], nombre), *resto["ex"]], 1)
        r = {"n_rectas": int(n), "val": [], "ciega": [], "peso": {g: [] for g in G}, "sens": {g: [] for g in G}, "fallos19_arreglados": [], "fallos_nuevos": [], "idx784": [], "fallos_val_s0": None}
        for sem in D.SEMILLAS:
            red, mu, sd = RA.entrenar(X, Y, sem); W = red[0].weight.detach().numpy(); wt = (W ** 2).sum()
            pv, pe = RA.logits(red, mu, sd, Xva).argmax(1), RA.logits(red, mu, sd, Xex).argmax(1)
            r["val"].append(float((pv == y[va]).mean())); r["ciega"].append(float((pe == y[ex]).mean()))
            z = torch.from_numpy((Xva - mu) / sd).float().requires_grad_(True); L = red(z); t2 = L.topk(2, 1).values; (t2[:, 0] - t2[:, 1]).sum().backward(); A = np.abs((z.grad * z).detach().numpy()); at = A.sum()
            for g, (a, b) in G.items():
                r["peso"][g].append(float((W[:, a:b] ** 2).sum() / wt)); r["sens"][g].append(float(A[:, a:b].sum() / at))
            mal = set(int(va[k]) for k in np.flatnonzero(pv != y[va]))
            r["fallos19_arreglados"].append(len(set(fallos19) - mal)); r["fallos_nuevos"].append(len(mal - set(fallos19))); r["idx784"].append(int(pv[j784]))
            if sem == 0:
                r["fallos_val_s0"] = sorted(mal)
        res = {"n_rectas": n, "val": round(float(np.mean(r["val"])), 4), "val_rango": [round(min(r["val"]), 4), round(max(r["val"]), 4)], "ciega": round(float(np.mean(r["ciega"])), 4),
               "cuota_peso": {g: round(float(np.mean(v)), 3) for g, v in r["peso"].items()}, "cuota_sensibilidad": {g: round(float(np.mean(v)), 3) for g, v in r["sens"].items()},
               "fallos19_arreglados_por_semilla": r["fallos19_arreglados"], "fallos_nuevos_por_semilla": r["fallos_nuevos"], "idx784_por_semilla": r["idx784"], "fallos_val_s0": r["fallos_val_s0"]}
        out["variantes"][nombre] = res
        print(f"{nombre:32s} rectas {n:3d} · val {res['val']:.4f} [{res['val_rango'][0]:.4f}–{res['val_rango'][1]:.4f}] · ciega {res['ciega']:.4f} · cuota peso {res['cuota_peso']} · "
              f"arregla {r['fallos19_arreglados']} de 23 · nuevos {r['fallos_nuevos']} · 784→{r['idx784']}  ({time.time() - t0:.0f} s)", flush=True)
    (AQUI / "resultados" / f"{SALIDA}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6)); V = out["variantes"]; nombres = list(V); xs = np.arange(len(nombres))
    axs[0].bar(xs - 0.2, [V[n]["val"] for n in nombres], 0.4, color="#2a78d6", label="val (1617)"); axs[0].bar(xs + 0.2, [V[n]["ciega"] for n in nombres], 0.4, color="#52514e", label="a ciegas (3823)")
    for i, n in enumerate(nombres):
        axs[0].text(i - 0.2, V[n]["val"], f"{V[n]['val']:.4f}", ha="center", va="bottom", fontsize=7); axs[0].text(i + 0.2, V[n]["ciega"], f"{V[n]['ciega']:.4f}", ha="center", va="bottom", fontsize=7)
    axs[0].set_ylim(0.95, 1.0); axs[0].legend(fontsize=7, frameon=False); axs[0].set_title("acierto (3 semillas, con aumento)", fontsize=9)
    for k, (g, c) in enumerate(zip(("rectas", "gris", "rizo"), ("#52514e", "#2a78d6", "#c2410c"))):
        axs[1].bar(xs + (k - 1) * 0.27, [V[n]["cuota_peso"][g] for n in nombres], 0.27, color=c, label=g)
    axs[1].set_title("cuota de ‖W‖² de la 1.ª capa por grupo", fontsize=9); axs[1].legend(fontsize=7, frameon=False); axs[1].set_ylim(0, 1)
    axs[2].bar(xs - 0.2, [np.mean(V[n]["fallos19_arreglados_por_semilla"]) for n in nombres], 0.4, color="#1b7f3b", label="de los 23 fallos (fig. 19), arreglados")
    axs[2].bar(xs + 0.2, [np.mean(V[n]["fallos_nuevos_por_semilla"]) for n in nombres], 0.4, color="#c2410c", label="fallos nuevos"); axs[2].legend(fontsize=7, frameon=False); axs[2].set_title("fallos en val (media de 3 semillas)", fontsize=9)
    for ax in axs:
        ax.set_xticks(xs); ax.set_xticklabels([f"rectas {n}\n({V[n]['n_rectas']})" for n in nombres], fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"rizo-2 · {SALIDA.split(chr(45))[0]} · {TITULO} · MLP 256, con aumento", fontsize=10)
    fig.tight_layout(); fig.savefig(AQUI / "resultados" / f"{SALIDA}.png", dpi=100); print(f"→ resultados/{SALIDA}.json · {SALIDA}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
