#!/usr/bin/env python3
"""El OBTENEDOR de features del grupo `dig`: 13 kernels 5×5 sacados de los DÍGITOS sin mirar sus etiquetas
(k-means esférico de parches), y cada kernel convertido en un detector independiente con salida mapa 8×8.
Criterio y método escritos antes en instrucciones/02-criterio.md § «Corrida 6».

    python nn/obtenedor.py --aprender      20 dígitos al azar de train → nn/pesos-dig/kernels.pt + kernels.json
                                           + resultados/kernels-dig.png
    python nn/obtenedor.py --aplicar       los 13 detectores sobre los 1797 dígitos → resultados/mapas-digitos-dig.npz,
                                           y el combinado fino+grueso+dig → resultados/mapas-digitos-c24dig.npz

El detector es AUTOCONTENIDO (`DetectorKernel`): carga `kernels.pt` y no importa nada del repo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

AQUI = Path(__file__).resolve().parent
PESOS = AQUI / "pesos-dig"
RES = AQUI.parent / "resultados"
GRUPO = "dig"
K = 13
LADO_K = 5
N_DIGITOS = 20
SEMILLA_DIGITOS = 20261003
SEMILLA_KMEANS = 1
ITER = 200
NORMA_MIN = 1.0          # parches con menos tinta que esto no entran al k-means
SUAVE = 0.5              # |p| + SUAVE en el denominador: un parche casi vacío no da similitud alta
UMBRAL = 0.8             # sólo para la «firma» por clase; el compositor no lo usa


class DetectorKernel(torch.nn.Module):
    """Un kernel = un detector: mapa 8×8 = clip(p·k / (|k|·(|p| + SUAVE)), 0, 1) en cada celda."""

    def __init__(self, kernel: torch.Tensor):
        super().__init__()
        self.register_buffer("k", kernel.reshape(1, 1, LADO_K, LADO_K).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:             # x (N,1,8,8) en [0,1]
        pad = LADO_K // 2
        num = Fn.conv2d(x, self.k, padding=pad)
        norma_p = torch.sqrt(Fn.conv2d(x * x, torch.ones_like(self.k), padding=pad).clamp_min(0))
        return (num / (self.k.norm() * (norma_p + SUAVE))).clamp(0, 1)


def parches(x: np.ndarray) -> np.ndarray:
    """(N,8,8) → (N·64, 25): el parche 5×5 centrado en cada celda, con ceros fuera."""
    p = LADO_K // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    out = [xp[:, i:i + LADO_K, j:j + LADO_K].reshape(len(x), -1) for i in range(8) for j in range(8)]
    return np.concatenate(out, 0)


def kmeans_esferico(P: np.ndarray, k: int, semilla: int, iters: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(semilla)
    C = [P[rng.integers(len(P))]]
    for _ in range(1, k):                                            # k-means++ con distancia coseno
        d = np.clip(1 - np.max(P @ np.stack(C).T, 1), 0, None)   # redondeo: un parche idéntico da -1e-16
        C.append(P[rng.choice(len(P), p=d / d.sum())])
    C = np.stack(C)
    for _ in range(iters):
        asig = np.argmax(P @ C.T, 1)
        nuevo = np.stack([P[asig == c].sum(0) if (asig == c).any() else C[c] for c in range(k)])
        nuevo /= np.linalg.norm(nuevo, axis=1, keepdims=True)
        if np.allclose(nuevo, C):
            break
        C = nuevo
    return C, np.bincount(np.argmax(P @ C.T, 1), minlength=k)


def alias(kern: np.ndarray) -> str:
    """Orientación dominante (tensor de estructura) + dónde está la tinta dentro del kernel."""
    k = kern.reshape(LADO_K, LADO_K)
    gy, gx = np.gradient(k)
    jxx, jyy, jxy = (gx * gx).sum(), (gy * gy).sum(), (gx * gy).sum()
    coher = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / max(1e-9, jxx + jyy)
    if coher < 0.3:
        ori = "~"
    else:
        # dirección del GRADIENTE dominante; el trazo va perpendicular a ella
        ang = (np.degrees(0.5 * np.arctan2(2 * jxy, jxx - jyy)) + 90) % 180     # ángulo del trazo, y hacia abajo
        ori = "H" if ang < 22.5 or ang >= 157.5 else ("\\" if ang < 67.5 else ("V" if ang < 112.5 else "/"))
    m = np.clip(k, 0, None); tot = m.sum() or 1.0
    fy = (m.sum(1) * np.arange(LADO_K)).sum() / tot - 2; fx = (m.sum(0) * np.arange(LADO_K)).sum() / tot - 2
    v = "N" if fy < -0.5 else ("S" if fy > 0.5 else ""); h = "W" if fx < -0.5 else ("E" if fx > 0.5 else "")
    return f"{ori}-{(v + h) or 'c'}"


def aprender() -> int:
    sys.path.insert(0, str(AQUI)); import datos                     # noqa: E401,PLC0415
    dig = datos.digitos()
    idx_train = np.flatnonzero(dig["train"])
    elegidos = np.sort(np.random.default_rng(SEMILLA_DIGITOS).choice(idx_train, N_DIGITOS, replace=False))
    x = dig["x"][elegidos, 0]                                        # (20,8,8) — la etiqueta NO se lee
    P = parches(x)
    norma = np.linalg.norm(P, axis=1)
    P = P[norma >= NORMA_MIN] / norma[norma >= NORMA_MIN, None]
    C, tam = kmeans_esferico(P, K, SEMILLA_KMEANS, ITER)
    orden = np.argsort(-tam)                                         # dig:01 = el patrón más frecuente
    C, tam = C[orden], tam[orden]
    nombres = [f"{GRUPO}:{i + 1:02d}" for i in range(K)]
    alias_ = [alias(c) for c in C]
    PESOS.mkdir(parents=True, exist_ok=True)
    torch.save({"kernels": torch.from_numpy(C.reshape(K, LADO_K, LADO_K).astype(np.float32)), "nombres": nombres,
                "alias": alias_, "suave": SUAVE, "lado": LADO_K}, PESOS / "kernels.pt")
    meta = {"grupo": GRUPO, "metodo": "k-means esférico de parches 5x5, sin etiquetas", "k": K, "n_digitos": N_DIGITOS,
            "indices_digitos": elegidos.tolist(), "semilla_digitos": SEMILLA_DIGITOS, "semilla_kmeans": SEMILLA_KMEANS,
            "n_parches": int(len(P)), "norma_min": NORMA_MIN, "suave": SUAVE,
            "detectores": [{"nombre": n, "alias": a, "parches": int(t)} for n, a, t in zip(nombres, alias_, tam)]}
    (PESOS / "kernels.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    png = dibujar(C, nombres, alias_, x)
    print(f"{N_DIGITOS} dígitos (sin etiqueta) → {len(P)} parches → {K} kernels → {PESOS / 'kernels.pt'}, {png}")
    for n, a, t in zip(nombres, alias_, tam):
        print(f"  {n}  {a:<6} {t:>4} parches")
    return 0


def dibujar(C, nombres, alias_, x) -> Path:
    """Arriba: los 20 dígitos usados. Abajo: los 13 kernels (tinta en negro) con su nombre y alias."""
    from PIL import Image, ImageDraw                                 # noqa: PLC0415
    esc, sep, etq = 16, 8, 14
    ancho = max(len(x), K) * (8 * 5 // 2 + sep)
    t_d, t_k = 40, LADO_K * esc
    W = sep + K * (t_k + sep); H = sep + t_d + 2 * sep + t_k + etq * 2 + sep
    W = max(W, sep + len(x) * (t_d + 4))
    im = Image.new("L", (W, H), 255); d = ImageDraw.Draw(im)
    for i, xi in enumerate(x):
        im.paste(Image.fromarray(((1 - xi) * 255).astype(np.uint8)).resize((t_d, t_d), Image.NEAREST), (sep + i * (t_d + 4), sep))
    y0 = sep + t_d + 2 * sep
    for i, c in enumerate(C):
        k = c.reshape(LADO_K, LADO_K); k = np.clip(k / max(1e-9, k.max()), 0, 1)
        x0 = sep + i * (t_k + sep)
        im.paste(Image.fromarray(((1 - k) * 255).astype(np.uint8)).resize((t_k, t_k), Image.NEAREST), (x0, y0))
        d.rectangle([x0 - 1, y0 - 1, x0 + t_k, y0 + t_k], outline=150)
        d.text((x0, y0 + t_k + 2), nombres[i], fill=0); d.text((x0, y0 + t_k + 2 + etq), alias_[i], fill=0)
    destino = RES / "kernels-dig.png"; destino.parent.mkdir(parents=True, exist_ok=True); im.save(destino)
    return destino


@torch.no_grad()
def aplicar() -> int:
    sys.path.insert(0, str(AQUI)); import datos                     # noqa: E401,PLC0415
    est = torch.load(PESOS / "kernels.pt", weights_only=False)
    dets = [DetectorKernel(k) for k in est["kernels"]]
    dig = datos.digitos(); x = torch.from_numpy(dig["x"])
    sigma = np.stack([d(x)[:, 0].numpy() for d in dets], 1).astype(np.float32)          # (N,13,8,8)
    nombres = list(est["nombres"])
    np.savez_compressed(RES / "mapas-digitos-dig.npz", sigma=sigma, y=dig["y"], train=dig["train"],
                        umbrales=np.full(K, UMBRAL, np.float32), nombres=np.array(nombres))
    c24 = dict(np.load(RES / "mapas-digitos-c24.npz"))
    if not (np.array_equal(c24["y"], dig["y"]) and np.array_equal(c24["train"], dig["train"])):
        raise SystemExit("✗ mapas-digitos-c24.npz no es de los mismos dígitos: me niego a juntarlos")
    np.savez_compressed(RES / "mapas-digitos-c24dig.npz", sigma=np.concatenate([c24["sigma"], sigma], 1), y=dig["y"],
                        train=dig["train"], umbrales=np.concatenate([c24["umbrales"], np.full(K, UMBRAL, np.float32)]),
                        nombres=np.array([str(n) for n in c24["nombres"]] + nombres))
    print(f"{K} detectores {GRUPO} sobre {len(x)} dígitos → mapas-digitos-dig.npz y mapas-digitos-c24dig.npz (39 mapas)")
    return 0


if __name__ == "__main__":
    if "--aprender" in sys.argv:
        raise SystemExit(aprender())
    if "--aplicar" in sys.argv:
        raise SystemExit(aplicar())
    print(__doc__); raise SystemExit(0)
