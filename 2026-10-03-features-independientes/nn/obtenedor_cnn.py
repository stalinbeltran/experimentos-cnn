#!/usr/bin/env python3
"""Grupo `cae`: 13 detectores CNN aprendidos de 100 dígitos SIN etiquetas, con un autocodificador
convolucional disperso. Método y criterio escritos antes en instrucciones/02-criterio.md § «Corrida 7».

    python nn/obtenedor_cnn.py --aprender     elige λ (sólo con los 100 dígitos) y entrena → nn/pesos-cae/
                                              + resultados/detectores-cae.png
    python nn/obtenedor_cnn.py --aplicar      → resultados/mapas-digitos-cae.npz y mapas-digitos-c24cae.npz

`Codificador` es AUTOCONTENIDO: 13 CNN independientes (conv por grupos, ningún peso compartido).
`detector(j)` extrae la j-ésima como una red suelta, para usarla sin las otras.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

AQUI = Path(__file__).resolve().parent
PESOS = AQUI / "pesos-cae"
RES = AQUI.parent / "resultados"
GRUPO = "cae"
K = 13
CANALES = 8
LADO_DEC = 5
N_DIGITOS = 100
SEMILLA_DIGITOS = 20261004
SEMILLA = 1
LAMBDAS = (0.003, 0.01, 0.03)      # versión L1, DEGENERÓ en copiar píxeles (criterio, enmienda corrida 7)
VIDA = 0.2                         # WTA de vida: cada detector conserva su 20 % de imágenes más activas del lote
PASOS = 3000
LR = 3e-3
VIVO = 0.01
UMBRAL = 0.5


class Codificador(nn.Module):
    """13 CNN independientes: groups=K en todas las capas salvo la primera, cuya entrada es 1 canal
    (así cada grupo de CANALES filtros de la capa 1 es exclusivo de su detector)."""

    def __init__(self, k: int = K, c: int = CANALES):
        super().__init__()
        self.k, self.c = k, c
        self.c1 = nn.Conv2d(1, k * c, 3, padding=1)
        self.c2 = nn.Conv2d(k * c, k * c, 3, padding=1, groups=k)
        self.c3 = nn.Conv2d(k * c, k, 1, groups=k)

    def forward(self, x):
        return Fn.relu(self.c3(Fn.relu(self.c2(Fn.relu(self.c1(x))))))         # (N, K, 8, 8), ≥ 0 (enmienda 2: no sigmoide)


def wta(m: torch.Tensor, vida: float = VIDA) -> torch.Tensor:
    """Espacial: de cada mapa, sólo su celda máxima por imagen. De vida: de cada detector, sólo las
    `vida`·N imágenes del lote en que ese máximo es mayor. Todo lo demás a 0 (el gradiente sólo pasa por los
    ganadores)."""
    n, k = m.shape[:2]
    plano = m.flatten(2)
    mx, idx = plano.max(2)                                                  # (N, K)
    keep = torch.zeros_like(mx, dtype=torch.bool)
    top = max(1, int(round(vida * n)))
    keep.scatter_(0, mx.topk(top, dim=0).indices, True)                     # por detector, sus mejores imágenes
    out = torch.zeros_like(plano)
    out.scatter_(2, idx.unsqueeze(2), (mx * keep).unsqueeze(2))
    return out.view_as(m)


class Autocodificador(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Codificador()
        self.dec = nn.Conv2d(K, 1, LADO_DEC, padding=LADO_DEC // 2, bias=True)        # un kernel 5×5 por detector

    def forward(self, x, usar_wta: bool = False):
        m = self.enc(x)
        return self.dec(wta(m) if usar_wta else m), m              # enmienda 3: decodificador LINEAL


def detector(enc: Codificador, j: int) -> nn.Sequential:
    """La j-ésima CNN como red suelta (1 → 1 canal), copiando SÓLO sus pesos."""
    c = enc.c
    d1, d2, d3 = nn.Conv2d(1, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, 1, 1)
    with torch.no_grad():
        d1.weight.copy_(enc.c1.weight[j * c:(j + 1) * c]); d1.bias.copy_(enc.c1.bias[j * c:(j + 1) * c])
        d2.weight.copy_(enc.c2.weight[j * c:(j + 1) * c]); d2.bias.copy_(enc.c2.bias[j * c:(j + 1) * c])
        d3.weight.copy_(enc.c3.weight[j:j + 1]); d3.bias.copy_(enc.c3.bias[j:j + 1])
    return nn.Sequential(d1, nn.ReLU(), d2, nn.ReLU(), d3, nn.ReLU())


def desplazar(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Cada imagen desplazada ±1 celda al azar (con ceros), para no memorizar 100 dígitos."""
    out = torch.zeros_like(x)
    d = torch.randint(-1, 2, (len(x), 2), generator=g)
    for i, (dy, dx) in enumerate(d.tolist()):
        out[i] = torch.roll(x[i], (dy, dx), (1, 2))
        if dy == 1: out[i, :, 0] = 0
        if dy == -1: out[i, :, -1] = 0
        if dx == 1: out[i, :, :, 0] = 0
        if dx == -1: out[i, :, :, -1] = 0
    return out


LAMBDA_L1 = 0.0                     # enmienda 2 la puso a 0,01 y mató 11/13; enmienda 3: sin L1


class Escala(nn.Module):
    """Al aplicar: WTA espacial (sólo la celda máxima, con su valor) y escala a [0, 1] (enmienda 3)."""

    def __init__(self, e: float):
        super().__init__(); self.e = e

    def forward(self, m):
        return (wta(m, vida=1.0) / self.e).clamp(0, 1)


def entrenar(x: torch.Tensor, lam: float = LAMBDA_L1) -> tuple[Autocodificador, dict]:
    torch.manual_seed(SEMILLA); g = torch.Generator().manual_seed(SEMILLA)
    ae = Autocodificador(); opt = torch.optim.Adam(ae.parameters(), lr=LR)
    for _ in range(PASOS):
        xb = desplazar(x, g)
        rec, m = ae(xb, usar_wta=True)
        loss = Fn.mse_loss(rec, xb) + lam * m.mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        rec, m = ae(x, usar_wta=True)
        act = m.mean((0, 2, 3)).numpy()
        out = {"lambda": lam, "mse": round(float(Fn.mse_loss(rec, x)), 5), "activacion": [round(float(a), 4) for a in act],
               "vivos": int((act > VIVO).sum())}
    return ae, out


def aprender() -> int:
    sys.path.insert(0, str(AQUI)); import datos; import obtenedor            # noqa: E401,PLC0415
    dig = datos.digitos()
    elegidos = np.sort(np.random.default_rng(SEMILLA_DIGITOS).choice(np.flatnonzero(dig["train"]), N_DIGITOS, replace=False))
    x = torch.from_numpy(dig["x"][elegidos])                                    # la etiqueta NO se lee
    t0 = time.time()
    ae, r = entrenar(x)
    lam = LAMBDA_L1; ensayos = {"wta": (ae, r)}
    with torch.no_grad():                                                    # escala por detector: p99 de sus ganadores
        ganadores = ae.enc(x).flatten(2).max(2).values                       # (N, K)
        escalas = torch.quantile(ganadores, 0.99, dim=0).clamp_min(1e-6)
        r["activacion"] = [round(float(a), 4) for a in (wta(ae.enc(x), 1.0) / escalas[None, :, None, None]).clamp(0, 1).flatten(2).max(2).values.mean(0)]
        r["vivos"] = int(sum(a > VIVO for a in r["activacion"]))
    print(f"  WTA (vida {VIDA}): mse {r['mse']:.5f}  vivos {r['vivos']}/{K}  ({time.time() - t0:.0f} s)", flush=True)
    orden = np.argsort(-np.array(r["activacion"]))                              # cae:01 = el que más se enciende
    dec = ae.dec.weight.detach()[0].numpy()                                     # (K, 5, 5)
    nombres = [f"{GRUPO}:{i + 1:02d}" for i in range(K)]
    alias_ = [obtenedor.alias(dec[j]) for j in orden]
    PESOS.mkdir(parents=True, exist_ok=True)
    torch.save({"autocodificador": ae.state_dict(), "escalas": escalas.tolist(), "orden": orden.tolist(), "nombres": nombres, "alias": alias_,
                "k": K, "canales": CANALES}, PESOS / "autocodificador.pt")
    meta = {"grupo": GRUPO, "metodo": "autocodificador convolucional disperso (WTA); codificador = 13 CNN independientes; sin etiquetas",
            "n_digitos": N_DIGITOS, "indices_digitos": elegidos.tolist(), "semilla_digitos": SEMILLA_DIGITOS, "semilla": SEMILLA,
            "pasos": PASOS, "lr": LR, "dispersion": f"WTA espacial (máximo por mapa) + de vida ({VIDA}) + L1 {LAMBDA_L1} sobre el mapa entero; salida ReLU",
            "escalas_p99": [round(float(e), 4) for e in escalas],
            "nota": "la versión L1 (λ en {0,003, 0,01, 0,03}) degeneró en copiar píxeles; ver criterio, enmienda corrida 7",
            "ensayos": [r for _, r in ensayos.values()], "segundos": round(time.time() - t0, 1),
            "detectores": [{"nombre": n, "alias": a, "canal_original": int(j), "activacion_media": r["activacion"][j]}
                           for n, a, j in zip(nombres, alias_, orden)]}
    (PESOS / "autocodificador.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    png = dibujar(ae, orden, nombres, alias_, x)
    print(f"WTA: {N_DIGITOS} dígitos (sin etiqueta) → {K} detectores → {PESOS / 'autocodificador.pt'}, {png}")
    for d in meta["detectores"]:
        print(f"  {d['nombre']}  {d['alias']:<6} activación media {d['activacion_media']:.3f}")
    return 0


@torch.no_grad()
def dibujar(ae, orden, nombres, alias_, x) -> Path:
    """Por detector: su kernel de decodificador (lo que «pinta») y su mapa sobre 6 de los dígitos usados."""
    from PIL import Image, ImageDraw                                        # noqa: PLC0415
    _, m = ae(x[:6]); dec = ae.dec.weight[0].numpy()        # mapas SIN wta: lo que ve el detector al aplicarlo
    m = (wta(m, 1.0) / torch.quantile(ae.enc(x).flatten(2).max(2).values, 0.99, dim=0).clamp_min(1e-6)[None, :, None, None]).clamp(0, 1)
    esc, sep, etq = 6, 6, 12; t = 8 * esc; tk = LADO_DEC * 8
    W = sep + 120 + 7 * (t + sep); H = sep + 2 * etq + (t + sep) * (K + 1)
    im = Image.new("L", (W, H), 255); d = ImageDraw.Draw(im)
    for c in range(6):
        im.paste(Image.fromarray(((1 - x[c, 0].numpy()) * 255).astype(np.uint8)).resize((t, t), Image.NEAREST), (sep + 120 + (c + 1) * (t + sep), sep))
    for r, j in enumerate(orden):
        y0 = sep + (r + 1) * (t + sep)
        k = dec[j]; k = np.clip(k / max(1e-9, np.abs(k).max()), 0, 1)
        im.paste(Image.fromarray(((1 - k) * 255).astype(np.uint8)).resize((tk, tk), Image.NEAREST), (sep + 70, y0))
        d.text((sep, y0 + 4), nombres[r], fill=0); d.text((sep, y0 + 18), alias_[r], fill=0)
        for c in range(6):
            im.paste(Image.fromarray(((1 - m[c, j].numpy()) * 255).astype(np.uint8)).resize((t, t), Image.NEAREST),
                     (sep + 120 + (c + 1) * (t + sep), y0))
    destino = RES / "detectores-cae.png"; im.save(destino)
    return destino


@torch.no_grad()
def aplicar() -> int:
    sys.path.insert(0, str(AQUI)); import datos                            # noqa: E401,PLC0415
    est = torch.load(PESOS / "autocodificador.pt", weights_only=False)
    ae = Autocodificador(); ae.load_state_dict(est["autocodificador"]); ae.eval()
    esc = est["escalas"]
    dets = [nn.Sequential(detector(ae.enc, j), Escala(esc[j])) for j in est["orden"]]   # cada uno, suelto
    dig = datos.digitos(); x = torch.from_numpy(dig["x"])
    sigma = np.stack([dd(x)[:, 0].numpy() for dd in dets], 1).astype(np.float32)
    # comprobación: los detectores sueltos dan EXACTAMENTE lo mismo que el codificador entero
    juntos = (wta(ae.enc(x), 1.0) / torch.tensor(esc)[None, :, None, None]).clamp(0, 1)[:, est["orden"]].numpy()
    assert np.allclose(sigma, juntos, atol=1e-5), "los detectores sueltos no reproducen el codificador"
    nombres = list(est["nombres"])
    np.savez_compressed(RES / "mapas-digitos-cae.npz", sigma=sigma, y=dig["y"], train=dig["train"],
                        umbrales=np.full(K, UMBRAL, np.float32), nombres=np.array(nombres))
    c24 = dict(np.load(RES / "mapas-digitos-c24.npz"))
    if not (np.array_equal(c24["y"], dig["y"]) and np.array_equal(c24["train"], dig["train"])):
        raise SystemExit("✗ mapas-digitos-c24.npz no es de los mismos dígitos")
    np.savez_compressed(RES / "mapas-digitos-c24cae.npz", sigma=np.concatenate([c24["sigma"], sigma], 1), y=dig["y"],
                        train=dig["train"], umbrales=np.concatenate([c24["umbrales"], np.full(K, UMBRAL, np.float32)]),
                        nombres=np.array([str(n) for n in c24["nombres"]] + nombres))
    print(f"{K} detectores {GRUPO} (sueltos, verificados contra el codificador) sobre {len(x)} dígitos → mapas-digitos-cae.npz, -c24cae.npz")
    return 0


if __name__ == "__main__":
    if "--aprender" in sys.argv:
        raise SystemExit(aprender())
    if "--aplicar" in sys.argv:
        raise SystemExit(aplicar())
    print(__doc__); raise SystemExit(0)
