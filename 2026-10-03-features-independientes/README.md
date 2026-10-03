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

## Atribución del error (2026-10-03): en qué fallan los dígitos mal clasificados

Detectores de la corrida 2 + compositor posicional (semilla 1, acc 0,9598 = la de `compositores.json`).
Método escrito antes de mirar en `02-criterio.md` § «Atribución del error»; script `nn/errores.py`;
detalle por fallo en `resultados/errores.json`, rejilla en `resultados/errores.png` (por fallo: dígito ·
mapa del detector culpable · su mapa medio en la clase real · en la predicha).

**65 fallos de 1617.**

| | |
|---|---|
| fallos de **reconocimiento** (el mapa culpable es raro para la clase real: > p90 de los aciertos) | **50** (77 %) |
| fallos de **composición** (mapas normales, el lineal los sumó mal) | 15 (23 %) |
| **difíciles en sí** (un lineal sobre los 64 píxeles crudos también falla) | 44 (68 %) |
| detectores «raros» por dígito | 3,18 en los fallos · 1,32 en los aciertos |

Control con píxeles crudos: 0,905 (154 fallos). No es una comparación de arquitecturas: sirve para
marcar qué dígitos son difíciles con cualquier lectura lineal.

**Un tercio de los fallos es un solo par: 1↔8** (1→8 ×12, 8→1 ×10). Lo siguiente, 2–3 por par.
Culpables: `arco-E` ×16, `recta-V` ×12, `arco-W` ×9, `esquina-NE` ×8, `lazo` ×7.

**Y la causa del 1↔8 es el GROSOR del trazo.** Medido como celdas con tinta (> 0,25) por fila:

| | ancho medio por fila |
|---|---:|
| recta-V **sintética** (todas / sólo grosor 4 px) | 1,12 / 1,29 |
| «1» bien clasificados | 2,87 |
| «1» clasificados como 8 | **3,41** (los más gruesos) |
| «8» clasificados como 1 | **3,05** (los más estrechos) |
| «8» bien clasificados | 3,36 |

Un «1» manuscrito, tras la reducción /4 de NIST, mide **casi 3 celdas de ancho**: 2–3 veces más que
cualquier trazo con el que se entrenaron los detectores. En los «1» más gruesos, `lazo` y `arco-E`
ven los bordes del bloque como curvas; en los «8» más estrechos, los lazos se aplastan en 2 columnas y
lo que queda parece una recta vertical con esquinas. A ese ancho, un «1» gordo y un «8» flaco son
**el mismo bloque de tinta**. Es el riesgo que el criterio señaló antes de correr nada (*«lo que más
probablemente salga mal: la transferencia de grosor»*), ahora medido.

Contra lo esperado: el 1↔8 no estaba en la lista (se esperaba 3↔8↔9 y 1↔7), y **los arcos y esquinas
sí son los culpables principales**, como se esperaba — pero no por su precisión sintética (la corrida 3
la subió y los dígitos empeoraron), sino porque **ven mal trazos gruesos**.

## Lo que queda pendiente

- ~~Re-entrenar arcos y esquinas con sus contra-casos~~: hecho en la corrida 3, no mejora (arriba).
- Comprobar si el techo de ~0,70–0,77 de arcos y esquinas es **ambigüedad del dato a 8×8** (radio chico).
- **Lo que sale del análisis del error:** un dataset sintético nuevo (nombre nuevo) con grosores que
  cubran el manuscrito —hasta 8–12 px de 32, o sea 2–3 celdas—, y re-entrenar desde los pesos actuales.
  La predicción: el 1↔8 cae a la mitad o menos, y los fallos de reconocimiento bajan más que los de
  composición.
- Curva por tamaño de train (§7 de la especificación): no hecha. La atribución del error, hecha (arriba).
- Una sola semilla por detector; un solo compositor lineal (sin MLP).
