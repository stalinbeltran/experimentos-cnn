# Resultados de `ruido-comb` — generado por `nn/informe.py` el 2026-10-03 07:04 UTC

**Generado del disco; no se edita a mano.** Criterio: `instrucciones/02-criterio.md`, escrito antes. Δ pareado por semilla; umbral = max(2·SE, 0,01).

| escenario | n | acc val (media ± sd) | CE val | acc train | **Δ vs `limpio`** ± SE | umbral | veredicto | **Δ vs mejor simple** ± SE | umbral | ¿suman? |
|---|---:|---|---|---|---|---|---|---|---|---|
| `limpio` | 3 | 0.8703 ± 0.0317 | 1.0372 | 1.0000 | — | — | base | — | — | — |
| `gaussiano@0.2-linea` | 3 | 0.9122 ± 0.0207 | 0.3592 | 1.0000 | **+0.0418** ± 0.0064 | 0.0127 | ayuda | referencia | — | — |
| `recorte@0.6-linea` | 3 | 0.8994 ± 0.0191 | 0.4904 | 1.0000 | **+0.0291** ± 0.0074 | 0.0148 | ayuda | **-0.0128** ± 0.0014 | 0.0100 | RESTAN |
| `recorte@0.6+gaussiano@0.2-linea` | 3 | 0.9192 ± 0.0143 | 0.2977 | 1.0000 | **+0.0489** ± 0.0140 | 0.0280 | ayuda | **+0.0070** ± 0.0088 | 0.0176 | indistinguible |
| `recorte@0.6~gaussiano@0.2-linea` | 3 | 0.9027 ± 0.0206 | 0.3965 | 1.0000 | **+0.0324** ± 0.0188 | 0.0375 | indistinguible | **-0.0095** ± 0.0141 | 0.0282 | indistinguible |

## Lo que dice el criterio

- **`recorte@0.6+gaussiano@0.2-linea`** (secuencial) contra `gaussiano@0.2-linea`: Δ = +0.0070 ± 0.0088 (umbral 0.0176; CE val -0.0616) → **indistinguible**.
- **`recorte@0.6~gaussiano@0.2-linea`** (mezcla) contra `gaussiano@0.2-linea`: Δ = -0.0095 ± 0.0141 (umbral 0.0282; CE val +0.0373) → **indistinguible**.
- **La forma más cerca de sumar**: `recorte@0.6+gaussiano@0.2-linea` (+0.0070); veredicto indistinguible.
- `recorte@0.6+gaussiano@0.2-linea` contra `recorte@0.6-linea`: Δ = +0.0198 ± 0.0090 (umbral 0.0180).
- `recorte@0.6~gaussiano@0.2-linea` contra `recorte@0.6-linea`: Δ = +0.0033 ± 0.0127 (umbral 0.0253).

## Figuras

![delta.png](delta.png)

![muestras-ruido.png](muestras-ruido.png)
