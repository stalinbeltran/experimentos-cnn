# Resultados de `ruido-nist` — generado por `nn/informe.py` el 2026-10-03 01:18 UTC

**Generado del disco (`nn/pesos/*/summary.json`); no se edita a mano.** Criterio: `instrucciones/02-criterio.md`, escrito antes de entrenar.

Δ = medida(ruido, s) − medida(`limpio`, s), **pareado por semilla** (mismos pesos iniciales, mismo orden de lotes), media de las semillas y SE = sd/√n. Umbral = max(2·SE, δ) con δ = 0.01 en exactitud; mecanismo leído en la entropía cruzada de las 180 de train limpias con δ_ce = 0.05 nats.

**Base `limpio`** (semillas [1, 2, 3]): exactitud val **0.8703 ± 0.0317**, CE val 1.0372, exactitud train (180) 1.0000, CE train 0.0006.

| escenario | tipo | nivel | n | acc val | **Δ acc val** ± SE | umbral | Δ CE val | Δ acc train | Δ CE train | veredicto | mecanismo |
|---|---|---:|---:|---|---|---|---|---|---|---|---|
| `borrado@0.2` | borrado | 0.2 | 3 | 0.8788 ± 0.0269 | **+0.0084** ± 0.0057 | 0.0114 | +0.0160 | +0.0000 | -0.0003 | **indistinguible** | train limpio sin cambio distinguible |
| `externos@0.05` | externos | 0.05 | 3 | 0.8699 ± 0.0347 | **-0.0004** ± 0.0030 | 0.0100 | -0.0380 | +0.0000 | -0.0000 | **indistinguible** | train limpio sin cambio distinguible |
| `horizontal@0.6` | horizontal | 0.6 | 3 | 0.8600 ± 0.0436 | **-0.0103** ± 0.0071 | 0.0143 | +0.0031 | +0.0000 | +0.0005 | **indistinguible** | train limpio sin cambio distinguible |
| `vertical@0.6` | vertical | 0.6 | 3 | 0.8798 ± 0.0301 | **+0.0095** ± 0.0051 | 0.0101 | -0.1202 | +0.0000 | +0.0001 | **indistinguible** | train limpio sin cambio distinguible |
| `oblicua@0.6` | oblicua | 0.6 | 3 | 0.8765 ± 0.0272 | **+0.0062** ± 0.0043 | 0.0100 | -0.1093 | +0.0000 | +0.0001 | **indistinguible** | train limpio sin cambio distinguible |
| `oblicua@0.6-r2` | oblicua | 0.6 | 3 | 0.8654 ± 0.0630 | **-0.0049** ± 0.0182 | 0.0365 | -0.0601 | +0.0000 | +0.0005 | **indistinguible** | train limpio sin cambio distinguible |
| `curva@0.6` | curva | 0.6 | 3 | 0.8730 ± 0.0408 | **+0.0027** ± 0.0054 | 0.0107 | -0.0622 | +0.0000 | +0.0000 | **indistinguible** | train limpio sin cambio distinguible |
| `recorte@0.6` | recorte | 0.6 | 3 | 0.8992 ± 0.0250 | **+0.0289** ± 0.0106 | 0.0212 | -0.3865 | -0.0037 | +0.0103 | **ayuda** | train limpio sin cambio distinguible |
| `gaussiano@0.1` | gaussiano | 0.1 | 3 | 0.8736 ± 0.0470 | **+0.0033** ± 0.0102 | 0.0204 | -0.1359 | +0.0000 | +0.0020 | **indistinguible** | train limpio sin cambio distinguible |
| `sal-pimienta@0.1` | sal-pimienta | 0.1 | 3 | 0.8778 ± 0.0305 | **+0.0074** ± 0.0105 | 0.0210 | -0.1896 | +0.0000 | +0.0015 | **indistinguible** | train limpio sin cambio distinguible |

## Lo que dice el criterio

- **Ayudan** (media Δ > umbral): `recorte@0.6` (+0.0289).
- **Perjudican**: ninguno.
- **Pasan a la fase 2** (ayuda, o indistinguible con media > 0, al nivel medio): `borrado`, `vertical`, `oblicua`, `curva`, `recorte`, `gaussiano`, `sal-pimienta`.
- **La realización** (`oblicua@0.6-r2` contra `oblicua@0.6`): Δ = -0.0111 ± 0.0218 (umbral 0.0437; amplitud de las medias entre tipos 0.0392) → la realización no se distingue (|Δ| dentro del umbral): una copia fija sirve para esta fase.

## Figuras

![delta-por-tipo.png](delta-por-tipo.png)

![muestras-ruido.png](muestras-ruido.png)

## Por dígito (exactitud val media entre semillas; Δ respecto de limpio)

- `limpio`: 0 0.948 · 1 0.783 · 2 0.920 · 3 0.873 · 4 0.783 · 5 0.880 · 6 0.912 · 7 0.946 · 8 0.759 · 9 0.899
- `borrado@0.2`: 0 +0.029 · 1 +0.000 · 2 +0.006 · 3 -0.014 · 4 -0.014 · 5 +0.016 · 6 +0.022 · 7 -0.010 · 8 +0.038 · 9 +0.012
- `externos@0.05`: 0 +0.000 · 1 -0.006 · 2 -0.006 · 3 -0.008 · 4 -0.008 · 5 +0.016 · 6 +0.018 · 7 -0.010 · 8 -0.009 · 9 +0.008
- `horizontal@0.6`: 0 +0.029 · 1 +0.045 · 2 +0.021 · 3 -0.002 · 4 -0.055 · 5 +0.004 · 6 +0.008 · 7 -0.048 · 8 -0.062 · 9 -0.045
- `vertical@0.6`: 0 +0.006 · 1 +0.016 · 2 +0.004 · 3 -0.004 · 4 +0.029 · 5 +0.024 · 6 +0.022 · 7 -0.002 · 8 -0.002 · 9 -0.000
- `oblicua@0.6`: 0 +0.008 · 1 +0.012 · 2 +0.006 · 3 +0.012 · 4 +0.006 · 5 +0.035 · 6 +0.012 · 7 -0.002 · 8 -0.034 · 9 +0.004
- `oblicua@0.6-r2`: 0 +0.029 · 1 +0.037 · 2 +0.029 · 3 +0.002 · 4 -0.045 · 5 -0.016 · 6 +0.014 · 7 -0.006 · 8 -0.081 · 9 -0.014
- `curva@0.6`: 0 +0.046 · 1 +0.037 · 2 +0.031 · 3 +0.000 · 4 -0.039 · 5 +0.016 · 6 +0.039 · 7 -0.012 · 8 -0.047 · 9 -0.045
- `recorte@0.6`: 0 +0.025 · 1 +0.098 · 2 +0.017 · 3 -0.020 · 4 +0.063 · 5 -0.008 · 6 +0.055 · 7 -0.014 · 8 +0.068 · 9 +0.006
- `gaussiano@0.1`: 0 +0.021 · 1 +0.043 · 2 +0.010 · 3 -0.008 · 4 +0.029 · 5 +0.008 · 6 +0.020 · 7 -0.027 · 8 -0.030 · 9 -0.035
- `sal-pimienta@0.1`: 0 +0.029 · 1 +0.022 · 2 +0.008 · 3 -0.008 · 4 +0.006 · 5 +0.053 · 6 +0.018 · 7 -0.025 · 8 -0.011 · 9 -0.021
