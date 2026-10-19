#!/usr/bin/env python3
"""delg-cnn — AFINAR el adelgazador con el FEEDBACK DEL RE-TRAZO (rizo-2 § 125; el dueño, 2026-10-19: «procede»).

La propuesta de rizo-2 § 123: el re-trazo se puede hacer DIFERENCIABLE sobre el mapa M —dilatar M con un disco del grosor del dígito—
y su diferencia con la tinta es una pérdida que se calcula sobre los DÍGITOS REALES, que no tienen línea media de verdad. Se suma a la
pérdida supervisada de los sintéticos (la de siempre, que mantiene M fina). Castiga la barra perdida del 7 (tinta sin re-trazo) y las
ramas falsas (re-trazo sin tinta).

  re-trazo  R(p) = max_q  M(q) · K_d(p − q)       K_d = disco de radio = grosor del dígito (a 64 px), borde erf σ 1,4 (= 0,7 a 32 px)
  pérdida   = BCE_ponderada(sintéticos) + λ · (sin explicar + inventado)   con R calculado sobre la CRESTA SUAVE σ((M − 0,4)/0,05)
            sin explicar = Σ núcleo·(1−R) / Σ núcleo · inventado = Σ R·(1−cerca) / Σ R
            núcleo / cerca = la tinta (> 0,5; el dígito de mismo trazo ×2) erosionada / dilatada 1 px de 32: las PUNTAS no cuentan

· PARTE de unos pesos ya entrenados (`--desde`, por defecto k3x3-g0.7x-64: el mejor de § 122) y los mismos sintéticos con los que se
  entrenaron (600, uno y dos trazos, gris σ 0,7, ×2).
· La pérdida de re-trazo usa SÓLO train + extra (4003 dígitos). val (1617) queda FUERA para medir el criterio sin trampa.
· El grosor de cada dígito es el de rizo-2/nn/retrazo.py (2 × distancia media al borde sobre el esqueleto de tinta > 0,5), a 32 px.
· Se guardan los pesos del ÚLTIMO paso como best.pt (no se elige por ninguna medida: elegir por val sería mirar el examen).

    ../.venv/bin/python nn/afinar_retrazo.py [--lambda 1] [--inventado 0] [--pasos 1500] [--nombre k3x3-g0.7x-64-rt]
        → nn/pesos/<nombre>/{best.pt,config.json,metrics.jsonl}
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from skimage.morphology import skeletonize

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI.parents[0])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from expcnn import exigir_dataset  # noqa: E402
from entrenar_local import SoloKernels, ampliar, conjunto  # noqa: E402

T4 = "uci-optdigits-orig-32px-trazo4-r20261018"
BORDE = 0.7 * 2                                   # σ del borde de la pluma, a 64 px
CRESTA, SUAVE = 0.4, 0.05                         # el umbral de la cresta de rizo-2 (M > 0,4) y su suavizado
TOL = 2                                           # tolerancia (px de 64 = 1 px de 32) para sin explicar e inventado


def grosor(t: np.ndarray) -> float:
    """El de rizo-2/nn/retrazo.py: 2 × distancia media al borde sobre el esqueleto (px de 32)."""
    if not t.any():
        return float("nan")
    dt = ndimage.distance_transform_edt(t); sk = skeletonize(t)
    return 2 * float(dt[sk].mean()) if sk.any() else float("nan")


def retrazo(M: torch.Tensor, radio: torch.Tensor, r: int) -> torch.Tensor:
    """R(p) = max_q M(q)·K(p−q), con K un disco suave de `radio` px (uno por dígito). M: (B,1,H,W) → (B,1,H,W)."""
    B, _, H, W = M.shape
    oy, ox = torch.meshgrid(torch.arange(-r, r + 1.0), torch.arange(-r, r + 1.0), indexing="ij")
    dist = torch.sqrt(oy ** 2 + ox ** 2).reshape(1, -1, 1)                                   # (1, k², 1)
    K = 0.5 * (1 - torch.erf((dist - radio.reshape(B, 1, 1)) / (BORDE * math.sqrt(2))))      # (B, k², 1)
    U = F.unfold(M, 2 * r + 1, padding=r)                                                    # (B, k², H·W)
    return (U * K).amax(1).reshape(B, 1, H, W)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default="k3x3-g0.7x-64")
    ap.add_argument("--nombre", default="k3x3-g0.7x-64-rt")
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0, help="peso de la pérdida de re-trazo")
    ap.add_argument("--pasos", type=int, default=1500)
    ap.add_argument("--inventado", type=float, default=1.0, help="peso del término de inventado dentro de la pérdida de re-trazo")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lote-sint", type=int, default=64)
    ap.add_argument("--lote-real", type=int, default=32)
    ap.add_argument("--semilla", type=int, default=303)
    ap.add_argument("--limite", type=int, default=0, help="usar sólo N dígitos reales y N sintéticos (para cronometrar)")
    a = ap.parse_args()
    torch.manual_seed(a.semilla); rng = np.random.default_rng(a.semilla)
    base = AQUI / "nn" / "pesos" / a.desde; cfg = json.loads((base / "config.json").read_text()); k = cfg["ampliar"]
    t0 = time.time()

    # sintéticos: los MISMOS con los que se entrenó la base (misma semilla y parámetros) + 10 nunca vistos
    n_sint = a.limite or cfg["n"]
    Xs, Ts, _ = conjunto(n_sint, cfg["semilla"], cfg["sigma"], cfg["suavizar"], cfg["dos"], k)
    Xn, Tn, _ = conjunto(10, 202, cfg["sigma"], cfg["suavizar"], cfg["dos"], k)
    print(f"sintéticos: {len(Xs)} + 10 nuevos ({time.time() - t0:.0f} s)", flush=True)

    # reales: train + extra del dataset de mismo trazo, ampliados ×k con la MISMA función que exportar_delgados.py
    d = np.load(exigir_dataset(T4) / "datos.npz"); img = d["imagenes"].astype(np.float32); part = d["particion"]
    ids = np.flatnonzero(part != "val")
    g = np.array([grosor(img[i] > 0.5) for i in ids]); ok = ~np.isnan(g); ids, g = ids[ok], g[ok]
    if a.limite:
        ids, g = ids[:a.limite], g[:a.limite]
    Xr = np.stack([ampliar(img[i], k) for i in ids]).astype(np.float32)
    radio = (g * k / 2).astype(np.float32)            # pluma = grosor (px de 32) → radio = grosor/2, ×k a 64 px
    radio = np.minimum(radio, np.percentile(radio, 99.5))   # unos pocos dígitos raros no deben agrandar la ventana de todos
    r = int(math.ceil(radio.max() + 2 * BORDE))
    print(f"reales: {len(ids)} (train+extra), grosor 10·50·90 % = {np.percentile(g, [10, 50, 90]).round(2)} px de 32, "
          f"ventana {2 * r + 1}×{2 * r + 1} ({time.time() - t0:.0f} s)", flush=True)

    red = SoloKernels(cfg["capas"], cfg["canales"], cfg.get("kernel", 3)); red.load_state_dict(torch.load(base / "best.pt"))
    opt = torch.optim.Adam(red.parameters(), lr=a.lr); bce = nn.BCELoss(reduction="none")
    xs, ts = torch.from_numpy(Xs)[:, None], torch.from_numpy(Ts)[:, None]
    xn, tn = torch.from_numpy(Xn)[:, None], torch.from_numpy(Tn)[:, None]
    xr, rr = torch.from_numpy(Xr)[:, None], torch.from_numpy(radio)

    def sup(x, t):
        return (bce(red(x), t) * (1 + 4 * (t > 0.5).float())).mean()

    def ret(x, rad):
        # sobre la CRESTA SUAVE (M > 0,4 es la cresta que extrae rizo-2), no sobre M: bajar M entero no puede salir barato.
        # Con TOLERANCIA de 1 px (a 32) en los dos lados: la pluma redonda se pasa de la tinta en cada PUNTA, y un brazo corto
        # (la barra del 7) es casi todo puntas; sin tolerancia, el término de inventado premia BORRARLO (rizo-2 § 125, medido).
        C = torch.sigmoid((red(x) - CRESTA) / SUAVE); R = retrazo(C, rad, r)
        nucleo = -F.max_pool2d(-(x > 0.5).float(), 2 * TOL + 1, 1, TOL)         # la tinta erosionada: lo que SÍ hay que cubrir
        cerca = F.max_pool2d((x > 0.5).float(), 2 * TOL + 1, 1, TOL)            # la tinta dilatada: lo que NO es inventar
        sin_explicar = (nucleo * (1 - R)).sum((1, 2, 3)) / nucleo.sum((1, 2, 3)).clamp_min(1)
        inventado = (R * (1 - cerca)).sum((1, 2, 3)) / R.sum((1, 2, 3)).clamp_min(1e-6)
        return (sin_explicar + a.inventado * inventado).mean()

    out = AQUI / "nn" / "pesos" / a.nombre; out.mkdir(parents=True, exist_ok=True)
    fijos = torch.from_numpy(rng.choice(len(xr), min(64, len(xr)), replace=False))        # para seguir la curva, siempre los mismos
    t1 = time.time()
    with (out / "metrics.jsonl").open("w") as f:
        for paso in range(0, a.pasos + 1):
            if paso % 50 == 0:
                red.eval()
                with torch.no_grad():
                    m = {"paso": paso, "nuevos": sup(xn, tn).item(), "retrazo": ret(xr[fijos], rr[fijos]).item(), "s": round(time.time() - t1, 1)}
                f.write(json.dumps(m) + "\n"); f.flush()
                print(f"paso {paso:5d}  sint nuevos {m['nuevos']:.4f}  retrazo {m['retrazo']:.4f}  ({m['s']:.0f} s)", flush=True)
            if paso == a.pasos:
                break
            red.train(); opt.zero_grad()
            js = torch.from_numpy(rng.choice(len(xs), min(a.lote_sint, len(xs)), replace=False))
            jr = torch.from_numpy(rng.choice(len(xr), min(a.lote_real, len(xr)), replace=False))
            p = sup(xs[js], ts[js]) + a.lam * ret(xr[jr], rr[jr]); p.backward(); opt.step()
    torch.save(red.state_dict(), out / "best.pt")
    dur = time.time() - t0
    (out / "config.json").write_text(json.dumps({**cfg, "nombre": a.nombre, "desde": a.desde, "retrazo": vars(a),
                                                  "reales": int(len(ids)), "segundos": round(dur, 1)}, indent=1))
    print(f"→ nn/pesos/{a.nombre}/best.pt · {dur:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
