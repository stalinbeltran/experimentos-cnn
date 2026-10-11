"""lazos — el detector de curvas DELGADO (copiado) y el detector de LAZOS (nuevo).

Detector de curvas: COPIADO de docs/bocetos/2026-10-09-curvas-gabor (curvas.kernel_gabor, curvas.campo, curvas.giro;
la calibración de delgadas.py vía gris_recalibrado.configurar), con TAU 0,6 (tau_comparar.py). No se importa nada de
allí: si el boceto cambia, esto no.

Detector de lazos (ver instrucciones/02-criterio.md):
    1  cada píxel con giro medible y |κ| ≥ KAPPA_MIN vota en p + R·k̂ (R = 57,3/|κ|, k̂ = dirección del vector de
       curvatura κ·(−sin θ, cos θ), que apunta al centro de curvatura)
    2  los votos se suavizan (σ_v); cada máximo local con ≥ V_min votos y fuera de la tinta es un candidato
    3  radio = la corona (2–12 px) con más tinta; cobertura = fracción de 36 sectores de 10° con tinta en [0,6 R, 1,4 R]
    4  cierre = (cobertura − 0,5)/0,5 en [0, 1]; con cobertura < 0,5 no es un lazo
    5  orientación = centro del hueco (racha de sectores vacíos) más grande: apunta a la abertura
    6  dos candidatos a menos de R/2 son el mismo lazo: se queda el de más votos
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter

# ─── el detector de curvas delgado (COPIADO) ──────────────────────────────────────────────────────────────────────
K, N_ORI = 9, 12
THETAS = np.arange(N_ORI) * 180.0 / N_ORI
LAM, TAU, SIGMA_E, COH_MIN, KAPPA_MIN = 6.0, 0.6, 0.5, 0.25, 2.0
SIGMA_CAMPO, D_GIRO = 1.0, 4


def kernel_gabor(k: int, ang: float, lam: float = LAM) -> np.ndarray:
    r = (k - 1) / 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    t = np.deg2rad(ang)
    u = xx * np.cos(t) + yy * np.sin(t)
    v = -xx * np.sin(t) + yy * np.cos(t)
    g = np.exp(-(u ** 2) / (2 * (k / 3) ** 2) - (v ** 2) / (2 * (lam / 2) ** 2)) * np.cos(2 * np.pi * v / lam)
    g -= g.mean()
    return (g / np.linalg.norm(g)).astype(np.float32)


GABOR = torch.from_numpy(np.stack([kernel_gabor(K, t) for t in THETAS])[:, None])
COS2 = np.cos(np.deg2rad(2 * THETAS))[:, None, None]
SIN2 = np.sin(np.deg2rad(2 * THETAS))[:, None, None]


@torch.no_grad()
def campo(x: np.ndarray) -> dict:
    r = F.relu(F.conv2d(torch.from_numpy(x.astype(np.float32))[None, None], GABOR, padding=K // 2))[0].numpy()
    Emax = gaussian_filter(r.max(0), SIGMA_E)
    E = gaussian_filter(r.sum(0), SIGMA_CAMPO)
    Cx, Cy = gaussian_filter((r * COS2).sum(0), SIGMA_CAMPO), gaussian_filter((r * SIN2).sum(0), SIGMA_CAMPO)
    return {"Emax": Emax, "E": E, "Cx": Cx, "Cy": Cy, "theta": (0.5 * np.degrees(np.arctan2(Cy, Cx))) % 180.0,
            "coh": np.hypot(Cx, Cy) / (E + 1e-6)}


def _muestra(c, ys, xs):
    cx = map_coordinates(c["Cx"], [ys, xs], order=1, mode="nearest")
    cy = map_coordinates(c["Cy"], [ys, xs], order=1, mode="nearest")
    e = map_coordinates(c["E"], [ys, xs], order=1, mode="nearest")
    return (0.5 * np.degrees(np.arctan2(cy, cx))) % 180.0, np.hypot(cx, cy) / (e + 1e-6)


def giro(c: dict, d: int = D_GIRO) -> np.ndarray:
    """κ (°/px) por píxel; NaN donde no se puede medir."""
    mask = c["Emax"] > TAU
    kappa = np.full(mask.shape, np.nan)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return kappa
    t = np.deg2rad(c["theta"][ys, xs]); ux, uy = np.cos(t), np.sin(t)
    th_p, co_p = _muestra(c, ys + d * uy, xs + d * ux)
    th_m, co_m = _muestra(c, ys - d * uy, xs - d * ux)
    e_p = map_coordinates(c["Emax"], [ys + d * uy, xs + d * ux], order=1, mode="constant")
    e_m = map_coordinates(c["Emax"], [ys - d * uy, xs - d * ux], order=1, mode="constant")
    dth = (th_p - th_m + 90.0) % 180.0 - 90.0
    ok = (c["coh"][ys, xs] > COH_MIN) & (co_p > COH_MIN) & (co_m > COH_MIN) & (e_p > TAU) & (e_m > TAU)
    kappa[ys[ok], xs[ok]] = dth[ok] / (2 * d)
    return kappa


# ─── el detector de LAZOS ─────────────────────────────────────────────────────────────────────────────────────────
SECTORES = 36
RADIOS = np.arange(2, 13)
_YY, _XX = np.mgrid[0:32, 0:32] + 0.5


def votos(x: np.ndarray) -> np.ndarray:
    """Acumulador 32×32 de los votos (sin suavizar) de los píxeles curvos por su centro de curvatura."""
    c = campo(x); kappa = giro(c)
    ys, xs = np.nonzero(np.isfinite(kappa) & (np.abs(np.nan_to_num(kappa)) >= KAPPA_MIN))
    acc = np.zeros(x.shape)
    if len(ys) == 0:
        return acc
    k = kappa[ys, xs]; th = np.deg2rad(c["theta"][ys, xs])
    kx, ky = -k * np.sin(th), k * np.cos(th)
    nrm = np.hypot(kx, ky); R = 57.3 / np.abs(k)
    cx, cy = xs + 0.5 + R * kx / nrm, ys + 0.5 + R * ky / nrm
    ix, iy = np.floor(cx).astype(int), np.floor(cy).astype(int)
    ok = (ix >= 0) & (ix < x.shape[1]) & (iy >= 0) & (iy < x.shape[0])
    np.add.at(acc, (iy[ok], ix[ok]), 1.0)
    return acc


def anillo(x: np.ndarray, cx: float, cy: float) -> tuple[float, float, np.ndarray]:
    """Radio (corona con más tinta), cobertura (0–1) y sectores cubiertos alrededor de (cx, cy)."""
    tinta = x > 0
    dx, dy = _XX - cx, _YY - cy
    dist = np.hypot(dx, dy); ang = np.degrees(np.arctan2(dy, dx)) % 360
    cuenta = [int((tinta & (np.abs(dist - r) <= 0.75)).sum()) / r for r in RADIOS]      # por perímetro: no premia lo lejano
    R = float(RADIOS[int(np.argmax(cuenta))])
    en = tinta & (dist >= 0.6 * R) & (dist <= 1.4 * R)
    sec = np.zeros(SECTORES, bool)
    sec[(ang[en] // (360 / SECTORES)).astype(int) % SECTORES] = True
    return R, float(sec.mean()), sec


def _hueco(sec: np.ndarray) -> float | None:
    """Ángulo (grados, coordenadas de imagen: 0 = →, 90 = ↓) del centro de la racha de sectores vacíos más larga."""
    if sec.all():
        return None
    s = np.roll(sec, -int(np.argmax(sec)))               # empezar en un sector cubierto: las rachas no dan la vuelta
    mejor, ini, cur, cur_ini = 0, 0, 0, 0
    for i, v in enumerate(s):
        if not v:
            if cur == 0:
                cur_ini = i
            cur += 1
            if cur > mejor:
                mejor, ini = cur, cur_ini
        else:
            cur = 0
    centro = (ini + (mejor - 1) / 2 + int(np.argmax(sec))) % SECTORES
    return float((centro + 0.5) * 360 / SECTORES % 360)


def lazos(x: np.ndarray, sigma_v: float, v_min: float) -> list[dict]:
    """Los lazos de una imagen binaria 32×32: centro (x, y en px, centro del píxel = índice + 0,5), radio, cierre
    (0–1), orientación (grados hacia la abertura; None si está cerrado del todo) y votos."""
    acc = gaussian_filter(votos(x), sigma_v) * (2 * np.pi * sigma_v ** 2)   # en «votos»: un pico aislado conserva su masa
    pico = (acc == maximum_filter(acc, size=3)) & (acc >= v_min) & (x == 0)
    ys, xs = np.nonzero(pico)
    cand = []
    for i in np.argsort(-acc[ys, xs]):
        cx, cy = xs[i] + 0.5, ys[i] + 0.5
        R, cob, sec = anillo(x, cx, cy)
        if cob < 0.5:
            continue
        if any(np.hypot(cx - c["centro"][0], cy - c["centro"][1]) < c["radio"] / 2 for c in cand):
            continue
        cierre = float(np.clip((cob - 0.5) / 0.5, 0, 1))
        cand.append({"centro": (float(cx), float(cy)), "radio": R, "cobertura": cob, "cierre": cierre,
                     "orientacion": _hueco(sec), "votos": float(acc[ys[i], xs[i]]), "sectores": sec})
    return cand


# ─── los RIZOS CERRADOS por rayos (COPIADO del sondeo nn/rizos.py de `lazos`: RAYOS_MIN 0,97 · RMAX 5) ──────────
from scipy.ndimage import label as _label   # noqa: E402

RAYOS_MIN, RMAX = 0.97, 5.0
_ANG = np.deg2rad(np.arange(36) * 10 + 5); _CA, _SA = np.cos(_ANG), np.sin(_ANG)
_PS = np.arange(0.5, RMAX + 0.01, 0.25)


def _rayos(x: np.ndarray, ys: np.ndarray, xs: np.ndarray):
    cx, cy = xs + 0.5, ys + 0.5
    px = cx[:, None, None] + _CA[None, :, None] * _PS[None, None, :]
    py = cy[:, None, None] + _SA[None, :, None] * _PS[None, None, :]
    ix, iy = np.floor(px).astype(int), np.floor(py).astype(int)
    ok = (ix >= 0) & (ix < x.shape[1]) & (iy >= 0) & (iy < x.shape[0])
    hit = np.zeros(ix.shape, bool); hit[ok] = x[iy[ok], ix[ok]]
    choca = hit.any(2)
    dist = np.where(choca, _PS[np.argmax(hit, axis=2)], np.nan)
    return choca, dist


def rizos_cerrados(x: np.ndarray, rayos_min: float = RAYOS_MIN) -> list[dict]:
    x = np.asarray(x) > 0
    ys, xs = np.nonzero(~x)
    if len(ys) == 0:
        return []
    choca, _ = _rayos(x, ys, xs)
    m = np.zeros(x.shape, bool); m[ys, xs] = choca.mean(1) >= rayos_min
    lab, n = _label(m, structure=np.ones((3, 3)))
    out = []
    for z in range(1, n + 1):
        gy, gx = np.nonzero(lab == z)
        cx, cy = gx.mean() + 0.5, gy.mean() + 0.5
        ch, di = _rayos(x, np.array([cy - 0.5]), np.array([cx - 0.5]))
        frac = float(ch[0].mean()); libres = _ANG[~ch[0]]
        ori = float(np.degrees(np.arctan2(np.sin(libres).mean(), np.cos(libres).mean())) % 360) if len(libres) else None
        out.append({"centro": (float(cx), float(cy)), "radio": float(np.nanmedian(di[0])),
                    "cierre": float(np.clip((frac - 0.5) / 0.5, 0, 1)), "orientacion": ori, "puntuacion": frac})
    return out


def estrategia_v(x: np.ndarray, sigma_v: float = 1.0, v_min: float = 3.0) -> list[dict]:
    """V = la REFERENCIA: lo que `lazos` ya medía (lazos por votos + rizos cerrados por rayos), con los mismos parámetros."""
    out = [{"centro": l["centro"], "radio": l["radio"], "cierre": l["cierre"], "orientacion": l["orientacion"],
            "puntuacion": l["votos"]} for l in lazos(x, sigma_v, v_min)]
    for r in rizos_cerrados(x):
        if not any(np.hypot(r["centro"][0] - o["centro"][0], r["centro"][1] - o["centro"][1]) < 2.5 for o in out):
            out.append(r)
    return out
