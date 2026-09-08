#!/usr/bin/env python3
"""Los kernels de CONTROL del §10. AUTONOMO: solo numpy.

    python nn/kernels.py [--k 9] [--guardar]   los imprime; --guardar los deja en kernels/

Los controles son lo que convierte una cifra en evidencia (§10). No son kernels
"evaluados": son la vara con la que se mide cualquier kernel que llegue.

  aleatorio  ¿el merito es de ESTE kernel, o de filtrar cualquier cosa? (§10.2)
  gauss      referencia clasica de suma POSITIVA (§10)
  sobel      referencia clasica de suma CERO (§10)

⚠ EL ALEATORIO NECESITA VARIAS SEMILLAS, NO UNA (§10.2). Su desempenyo tambien varia,
y es el COMPARADOR del criterio 2.1: su desviacion entra en el margen que un kernel
tiene que superar. Medirlo con una sola semilla haria el margen artificialmente
pequenyo y declararia utiles kernels que no lo son.

⚠ TODOS SALEN CON LA MISMA NORMA porque el banco los normaliza al recibirlos (§5.4),
asi que "igual norma" no hay que construirlo: esta garantizado. Lo que hay que
construir es "mismo `k`", y por eso todos toman `k` como argumento.

⚠ SOBEL A TAMANYO ARBITRARIO: el Sobel clasico es 3x3. Para comparar a igual `k` hace
falta una version de tamanyo `k`, y aqui se usa la DERIVADA DE UNA GAUSSIANA en x,
que a k=3 reproduce la estructura del Sobel (suavizado en y, diferencia en x) y
generaliza sin inventar nada. Se dice porque no es "el" Sobel: es su familia.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
KERNELS = AQUI.parent / "kernels"
K_CONTROL = 9        # el `k` de los controles en la calibracion: el medio de 3..19


def aleatorio(k: int, semilla: int) -> np.ndarray:
    """Ruido gaussiano. La norma la fija el banco (§5.4)."""
    return np.random.default_rng(semilla).standard_normal((k, k)).astype(np.float32)


def gauss(k: int, sigma: float | None = None) -> np.ndarray:
    """Gaussiana isotropa. Suma POSITIVA: hereda la media del fondo blanco."""
    sigma = sigma if sigma is not None else k / 6.0
    r = np.arange(k, dtype=np.float32) - (k - 1) / 2.0
    g = np.exp(-(r ** 2) / (2 * sigma ** 2))
    return np.outer(g, g).astype(np.float32)


def sobel(k: int, sigma: float | None = None) -> np.ndarray:
    """Derivada de gaussiana en x. Suma CERO: detecta borde vertical."""
    sigma = sigma if sigma is not None else max(0.8, k / 6.0)
    r = np.arange(k, dtype=np.float32) - (k - 1) / 2.0
    suave = np.exp(-(r ** 2) / (2 * sigma ** 2))
    deriv = -r / (sigma ** 2) * suave
    return np.outer(suave, deriv).astype(np.float32)


def laplaciano(k: int, sigma: float | None = None) -> np.ndarray:
    """Laplaciana de gaussiana. Suma CERO e ISOTROPA: ni pasa-bajos ni direccional.

    Se anyade aqui el 2026-09-08 porque ya se habia EVALUADO con un script suelto, y un
    kernel que no se puede regenerar desde codigo commiteado es un dato huerfano: su
    `.npy` esta en git, pero la receta que lo produjo vivia en la linea de comandos de
    una sesion que se va con la maquina. Los controles se regeneran; este tambien."""
    sigma = sigma if sigma is not None else k / 6.0
    r = np.arange(k, dtype=np.float32) - (k - 1) / 2.0
    X, Y = np.meshgrid(r, r)
    r2 = X ** 2 + Y ** 2
    g = np.exp(-r2 / (2 * sigma ** 2))
    lap = (r2 - 2 * sigma ** 2) / (sigma ** 4) * g
    return (lap - lap.mean()).astype(np.float32)      # suma cero exacta


def controles(k: int = K_CONTROL, semillas: int = 10) -> dict[str, np.ndarray]:
    """Todos los controles con kernel. `identidad` y `caja-media` NO llevan kernel."""
    out = {f"aleatorio-r{s}": aleatorio(k, 1000 + s) for s in range(semillas)}
    out["gauss"] = gauss(k)
    out["sobel"] = sobel(k)
    out["laplaciano"] = laplaciano(k)
    return out


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=K_CONTROL)
    ap.add_argument("--guardar", action="store_true")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(AQUI))
    from pipeline import comprobar_contrato, normalizar_kernel   # noqa: PLC0415

    print(f"\nControles con k={a.k}\n")
    ks = controles(a.k)
    for nombre, k in sorted(ks.items()):
        comprobar_contrato(k)
        kn = normalizar_kernel(k)
        print(f"  {nombre:14} suma={k.sum():+9.4f}  suma_norm={kn.sum():+8.4f}  "
              f"norma_L2={np.linalg.norm(kn):.4f}  min={kn.min():+.3f} max={kn.max():+.3f}")
    print("\n  gauss tiene suma POSITIVA y sobel suma ~CERO: es justo la diferencia de")
    print("  distribucion que el §6.5 estandariza para que no se cuente como calidad.\n")
    print(f"  suma de gauss  = {gauss(a.k).sum():.4f}")
    print(f"  suma de sobel  = {sobel(a.k).sum():.2e}  (cero hasta redondeo)")

    if a.guardar:
        KERNELS.mkdir(parents=True, exist_ok=True)
        for nombre, k in sorted(ks.items()):
            np.save(KERNELS / f"{nombre}.npy", k.astype(np.float32))
        (KERNELS / "README.md").write_text(
            f"""# Kernels de `banco-k`

Los **controles** del §10, generados por `nn/kernels.py` con `k={a.k}` (el medio del
rango 3..19 del contrato §5).

| kernel | qué contesta | suma |
|---|---|---|
| `aleatorio-r0..r9` | ¿el mérito es de *este* kernel, o de **filtrar cualquier cosa**? (§10.2) | ~0 |
| `gauss` | referencia clásica de **suma positiva** | > 0 |
| `sobel` | referencia clásica de **suma cero** (derivada de gaussiana en x) | ~0 |

⚠ **La identidad y la caja media no tienen `.npy`**: la identidad es un **recorte
puro** (§6.3) y la caja media es un **predictor constante** que ni siquiera mira la
imagen (§10.1).

⚠ **La norma no hace falta igualarla aquí**: el banco normaliza en L2 al recibir el
kernel (§5.4), así que «igual norma» está garantizado. Lo que sí hay que construir es
«mismo `k`».

⚠ **Aquí no hay ningún kernel *evaluado* todavía**, y es correcto: el banco es
agnóstico al origen del kernel (§1) y los métodos para obtenerlos están fuera de
alcance (§15). Con estos controles el banco se calibra y se valida entero.

## ⚠ Reserva contra fuga de distribución (§3.7)

Un kernel que se obtenga con el mismo generador tiene **fuga aunque las muestras sean
distintas**. Por eso el dataset reserva **`LiberationMono`** y el interlineado
`[1.45, 1.60]` para uso **exclusivo del banco**: un procedimiento que produzca
kernels **no puede usarlos**.
""", encoding="utf-8")
        (KERNELS / "controles.json").write_text(json.dumps(
            {"k": a.k, "generados_por": "nn/kernels.py",
             "aleatorio_semillas": list(range(1000, 1010)),
             "sobel": "derivada de gaussiana en x (el Sobel clasico es 3x3; esta es su "
                      "familia a tamanyo arbitrario)",
             "gauss_sigma": a.k / 6.0}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(f"\n  guardados {len(ks)} kernels en {KERNELS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
