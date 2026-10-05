#!/usr/bin/env python3
"""Los DOS bancos de 13 detectores de hoy, COPIADOS aquí y congelados (Regla 0: se copia, no se importa):

  32  los de `feat-ind32` (C3): entrada 32×32 0/1 → mapa 8×8. La red es copia de su `nn/modelo.py`.
  8   los `fino` de `feat-ind` (corrida 2): entrada 8×8 = cuentas 4×4 ÷ 16 → mapa 8×8. Copia de su `nn/modelo.py`.

Ninguno vio un dígito: se entrenaron con dibujos sintéticos, y su umbral `u_f` se eligió sobre el train sintético.

    python nn/detectores.py --copiar      UNA vez: copia los best.pt desde los dos experimentos (pedidos al registro por su
                                          id) y escribe nn/detectores.json (origen, commit, huella, umbral). Se niega a pisar
                                          una copia que ya exista.
    python nn/detectores.py --comprobar   las huellas de las copias son las de detectores.json, y la firma por clase de los
                                          1797 `windep` recalculada con ellas es EXACTAMENTE la que guardó cada origen
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
sys.path.insert(0, str(EXP.parent))                       # el repo: donde vive `expcnn`
sys.path.insert(0, str(AQUI))

# EL ORDEN ES EL DE LOS DOS ORÍGENES (`features.CON_TRAZO` en los dos): las columnas de los mapas van en este orden.
FAMILIAS = ("arco-E", "arco-W", "arco-N", "arco-S", "recta-V", "recta-H", "recta-S", "recta-B", "lazo",
            "esquina-NE", "esquina-NW", "esquina-SE", "esquina-SW")
ARCOS = FAMILIAS[:4]
BANCOS = ("32", "8")
ORIGEN = {"32": "feat-ind32", "8": "feat-ind"}
CARPETA = {"32": AQUI / "detectores-32px", "8": AQUI / "detectores-8px"}
CLAVE_FIRMA = {"32": "firma_windep", "8": "firma"}      # cada origen la guardó con su nombre
REGISTRO = AQUI / "detectores.json"
LADO = 8
K = 3


class Detector32(nn.Module):
    """COPIA de la red de `feat-ind32` (nn/modelo.py), sin cambiar nombres: el state_dict tiene que casar.
    Conv3×3 16 → Conv3×3/2 32 → Conv3×3/2 32 → 3 × Conv3×3 32 → Conv1×1, con padding y ReLU: 32×32 → 1 canal 8×8."""

    CANALES = (16, 32, 32, 32, 32, 32)
    STRIDES = (1, 2, 2, 1, 1, 1)

    def __init__(self, canales=CANALES, strides=STRIDES):
        super().__init__()
        capas, cin = [], 1
        for c, s in zip(canales, strides):
            capas += [nn.Conv2d(cin, c, K, stride=s, padding=K // 2), nn.ReLU()]
            cin = c
        self.tronco = nn.Sequential(*capas)
        self.salida = nn.Conv2d(cin, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.salida(self.tronco(x))


class Detector8(nn.Module):
    """COPIA de la red de `feat-ind` (nn/modelo.py): 3 × Conv3×3 (16, 32, 32) + Conv1×1, con padding: 8×8 → 8×8."""

    CANALES = (16, 32, 32)

    def __init__(self, canales=CANALES):
        super().__init__()
        capas, cin = [], 1
        for c in canales:
            capas += [nn.Conv2d(cin, c, K, padding=K // 2), nn.ReLU()]
            cin = c
        self.tronco = nn.Sequential(*capas)
        self.salida = nn.Conv2d(cin, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.salida(self.tronco(x))


def huella_pesos(red: nn.Module) -> str:
    """La MISMA huella que usan los dos orígenes (sus `huellas_best`)."""
    h = hashlib.sha256()
    for p in red.parameters():
        h.update(p.detach().contiguous().to(torch.float32).numpy().astype("<f4").tobytes())
    return h.hexdigest()[:16]


def _sha_fichero(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()[:16]


def _cargar_pt(ruta: Path, banco: str) -> tuple[nn.Module, float]:
    est = torch.load(ruta, map_location="cpu", weights_only=False)
    cfg = est.get("config", {})
    if banco == "32":
        red = Detector32(cfg.get("canales", Detector32.CANALES), cfg.get("strides", Detector32.STRIDES))
    else:
        red = Detector8(cfg.get("canales", Detector8.CANALES))
    red.load_state_dict(est["modelo"]); red.eval()
    return red, float(est.get("umbral", 0.5))


def registro() -> dict:
    if not REGISTRO.is_file():
        raise SystemExit(f"✗ no está {REGISTRO.name}: primero `python nn/detectores.py --copiar`")
    return json.loads(REGISTRO.read_text(encoding="utf-8"))


def cargar(banco: str) -> tuple[list, np.ndarray]:
    """Los 13 detectores del banco, en el orden de FAMILIAS, y sus umbrales. Se NIEGA si una huella no casa."""
    reg = registro()[banco]["detectores"]
    reds, umbrales = [], np.zeros(len(FAMILIAS), np.float32)
    for j, f in enumerate(FAMILIAS):
        red, u = _cargar_pt(CARPETA[banco] / f / "best.pt", banco)
        if huella_pesos(red) != reg[f]["huella"]:
            raise SystemExit(f"✗ banco {banco}, {f}: la huella no es la registrada. Me niego.")
        reds.append(red); umbrales[j] = u
    return reds, umbrales


@torch.no_grad()
def mapas(banco: str, x: np.ndarray, lote: int = 1024, reds: list | None = None) -> np.ndarray:
    """x (N,1,32,32) para el banco 32 o (N,1,8,8) para el 8 → σ (N, 13, 8, 8). Lo mismo que el `aplicar.py` de cada origen."""
    reds = reds if reds is not None else cargar(banco)[0]
    out = np.zeros((len(x), len(reds), LADO, LADO), np.float32)
    for i in range(0, len(x), lote):
        xt = torch.from_numpy(np.ascontiguousarray(x[i:i + lote], dtype=np.float32))
        for j, red in enumerate(reds):
            out[i:i + lote, j] = torch.sigmoid(red(xt))[:, 0].numpy()
    return out


def firma(s: np.ndarray, y: np.ndarray, umbrales: np.ndarray) -> dict:
    """La firma por clase de los orígenes: fracción de cada clase en que cada detector se enciende (max σ ≥ u_f)."""
    pres = s.reshape(len(y), len(FAMILIAS), -1).max(2) >= umbrales[None]
    return {str(c): {f: round(float(pres[y == c, j].mean()), 3) for j, f in enumerate(FAMILIAS)} for c in range(10)}


def _firma_de_origen(banco: str) -> dict:
    from expcnn import por_id                                        # noqa: PLC0415
    ruta = por_id(ORIGEN[banco]).carpeta / "resultados" / "firma-por-clase.json"
    return json.loads(ruta.read_text(encoding="utf-8"))


def copiar() -> int:
    from expcnn import por_id                                        # noqa: PLC0415
    reg = {}
    for banco in BANCOS:
        org = por_id(ORIGEN[banco]).carpeta
        origen_firma = _firma_de_origen(banco)
        if list(origen_firma["columnas"]) != list(FAMILIAS):
            raise SystemExit(f"✗ {ORIGEN[banco]}: sus columnas no van en el orden de FAMILIAS. Me niego.")
        dets = {}
        for f in FAMILIAS:
            rel = Path("nn") / "pesos" / f / "best.pt"
            src, dst = org / rel, CARPETA[banco] / f / "best.pt"
            if dst.exists():
                raise SystemExit(f"✗ {dst.relative_to(EXP)} ya existe: una copia no se pisa.")
            red, u = _cargar_pt(src, banco)
            h = huella_pesos(red)
            if h != origen_firma["huellas_best"][f]:
                raise SystemExit(f"✗ {ORIGEN[banco]}/{rel}: su huella no es la de su firma-por-clase.json. Me niego.")
            commit = subprocess.run(["git", "-C", str(org), "log", "-1", "--format=%H", "--", str(rel)],
                                    capture_output=True, text=True, check=True).stdout.strip()
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            dets[f] = {"huella": h, "umbral": u, "sha256_fichero": _sha_fichero(dst), "ruta_en_origen": str(rel),
                       "commit_en_origen": commit}
        reg[banco] = {"origen": ORIGEN[banco], "entrada": "32×32 0/1" if banco == "32" else "8×8 = cuentas 4×4 ÷ 16",
                      "clave_firma_en_origen": CLAVE_FIRMA[banco], "detectores": dets}
        print(f"banco {banco}: 13 detectores copiados de {ORIGEN[banco]} → {CARPETA[banco].relative_to(EXP)}")
    REGISTRO.write_text(json.dumps(reg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"→ {REGISTRO.relative_to(EXP)}")
    return 0


def comprobar() -> int:
    import datos                                                     # noqa: PLC0415
    ok = True
    d = datos.cargar(); y = datos.etiquetas()
    w = d["origen"] == "windep"
    for banco in BANCOS:
        reg = registro()[banco]["detectores"]
        malos = [f for f in FAMILIAS if _sha_fichero(CARPETA[banco] / f / "best.pt") != reg[f]["sha256_fichero"]]
        reds, umbrales = cargar(banco)                               # se niega solo si una huella de pesos no casa
        print(f"  [{'ok' if not malos else 'FALLA':>5}] banco {banco}: 13 ficheros y 13 huellas de pesos = detectores.json"
              + (f" (distintos: {malos})" if malos else "")); ok &= not malos
        x = d["x32"] if banco == "32" else d["x8"]
        mia = firma(mapas(banco, x[w], reds=reds), y[w], umbrales)
        suya = _firma_de_origen(banco)[CLAVE_FIRMA[banco]]
        dif = max(abs(mia[c][f] - suya[c][f]) for c in mia for f in FAMILIAS)
        print(f"  [{'ok' if dif == 0 else 'FALLA':>5}] banco {banco}: firma de los 1797 windep = la de {ORIGEN[banco]} "
              f"(diferencia máxima {dif:.3f} en 130 valores)"); ok &= dif == 0
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--copiar", action="store_true"); p.add_argument("--comprobar", action="store_true")
    a = p.parse_args()
    if a.copiar:
        return copiar()
    if a.comprobar:
        return comprobar()
    p.print_help(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
