# `feat-ind` — qué salió

Especificación en `ESPECIFICACION.md`, criterio (con sus tres enmiendas fechadas) en
`instrucciones/02-criterio.md`, reglas en `REGLAS.md`.

## Corrida 2 (2026-10-03): 13 detectores, unidad `feat-ind-detectores`, `Result=success`, `NRestarts=0`

16/32/32 canales (14.081 parámetros), 80 épocas, semilla 1, ~80 s por detector en el dev. Val sintético:

| detector | F1 | P | R | posición ≤1 celda | veredicto (§A) |
|---|---:|---:|---:|---:|---|
| lazo | 0,906 | 0,89 | 0,92 | 1,00 | **aprendió** |
| recta-V / H / `/` / `\` | 0,88 / 0,86 / 0,87 / 0,85 | 0,81–0,86 | 0,89–0,91 | 0,95–0,99 | a medias |
| arco-N | 0,757 | 0,67 | 0,86 | 0,92 | a medias |
| arco-E / W / S | 0,70 / 0,70 / 0,74 | 0,62–0,65 | 0,78–0,85 | 0,85–0,92 | no aprendió |
| esquina-NE / NW / SE / SW | 0,73 / 0,74 / 0,72 / 0,70 | 0,61–0,65 | 0,81–0,90 | 0,98–1,00 | no aprendió |

La posición casi siempre es buena; lo que falla es la **precisión** de arcos y esquinas (se disparan
entre sí). Corrida 1 (antes de las enmiendas) en `resultados/corrida-1-2026-10-03.json`.

## Compositores sobre dígitos (`uci-optdigits-8px-r20261002`, 180 train / 1617 val, 3 semillas)

| compositor | entradas | acc val | acc train |
|---|---:|---:|---:|
| presencia (máximo de cada mapa) | 13 | **0,718 ± 0,005** | 0,870 |
| posicional (los 13 mapas enteros) | 832 | **0,959 ± 0,001** | 1,000 |

- Los dos superan holgadamente el 0,40 de §B: **los detectores, entrenados sólo con dibujos, ven
  algo en manuscrito**.
- **La posición aporta +0,24**, y 6↔9 pasa de 2 confusiones a 0. Es el primer apoyo empírico para
  `feat-pos`.
- Contexto, **no comparación** (el dueño apartó la CNN monolítica por ahora, y aquella entrena sobre
  los dígitos mientras aquí sólo se entrena un lineal): con estos mismos 180/1617, `dim-nist` daba
  0,851 y `ruido-nist` 0,870–0,912.

Firma por clase en `resultados/firma-por-clase.json`; mapas en `resultados/mapas-digitos.png`.

## Lo que queda pendiente

- Re-entrenar arcos y esquinas con sus contra-casos (`--desde … --contra …`): sus FP están en cada
  `summary.json` (`fp_por_familia`).
- Atribución del error y curva por tamaño de train (§7 de la especificación): no hechas.
- Una sola semilla por detector; un solo compositor lineal (sin MLP).
