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

## Corrida 3 (2026-10-03): re-entreno con contra-casos — NO se adopta

Los 4 arcos y las 4 esquinas, desde su `last.pt` de la corrida 2, 40 épocas a `lr` 1e-3, con la mitad
de cada lote de negativas sacada de sus 3 familias con más FP (`--enfatizar`; criterio escrito antes en
`02-criterio.md` § «Corrida 3»). Unidad `feat-ind-reentreno`, `Result=success`, `NRestarts=0`.
Pesos en `nn/pesos-c3/` (los de la corrida 2 siguen en `nn/pesos/`).

| detector | F1 c2 → c3 | P c2 → c3 |
|---|---|---|
| arco-E | 0,695 → 0,712 | 0,63 → 0,65 |
| arco-W | 0,704 → 0,736 | 0,62 → 0,67 |
| arco-N | 0,757 → 0,770 | 0,67 → 0,70 |
| arco-S | 0,737 → 0,740 | 0,65 → 0,67 |
| esquina-NE | 0,730 → 0,734 | 0,61 → 0,65 |
| esquina-NW | 0,735 → 0,751 | 0,65 → 0,70 |
| esquina-SE | 0,725 → 0,726 | 0,65 → 0,69 |
| esquina-SW | 0,695 → 0,723 | 0,61 → 0,66 |

| compositor (dígitos, 3 semillas) | corrida 2 | corrida 3 |
|---|---:|---:|
| presencia | 0,718 ± 0,005 | 0,710 ± 0,002 |
| posicional | 0,959 ± 0,001 | 0,946 ± 0,000 |

**Contra lo esperado** (escrito antes): la precisión sube 0,02–0,05 y **no llega a 0,75 en ninguno**;
el F1 de ninguno pasa de 0,80. Y sobre dígitos **empeora** un poco (−0,008 y −0,013). O sea: los
contra-casos enfatizados se aprenden en sintético y eso no transfiere, o transfiere al revés. Lo dice
el propio criterio: *el cuello de botella en dígitos no era la precisión sintética*. Se quedan como
oficiales los de la corrida 2; `pesos-c3` se conserva como registro.

Lectura probable (sin medir): a 8×8, un arco de radio 4–7 px (1–2 celdas) y una esquina **son casi el
mismo patrón de cuentas** — el recall del tramo `chico` era el más bajo de los arcos (0,64–0,79). Si
es así, ningún re-entreno lo arregla: el techo es del dato, no del detector. Se comprueba midiendo F1
por tramo de radio en los FP, o subiendo el radio mínimo.

## Lo que queda pendiente

- ~~Re-entrenar arcos y esquinas con sus contra-casos~~: hecho en la corrida 3, no mejora (arriba).
- Comprobar si el techo de ~0,70–0,77 de arcos y esquinas es **ambigüedad del dato a 8×8** (radio chico).
- Atribución del error y curva por tamaño de train (§7 de la especificación): no hechas.
- Una sola semilla por detector; un solo compositor lineal (sin MLP).
