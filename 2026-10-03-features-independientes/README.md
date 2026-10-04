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

## Corrida 4 (2026-10-03): trazos gruesos — arregla el 1↔8 y abre un 4→1; empate neto

Los 13 desde `nn/pesos/` (corrida 2), 40 épocas a `lr` 1e-3 sobre `feat-ind-sinteticas-grueso-8px-r20261003d`.
Unidad `feat-ind-grueso`, `Result=success`, `NRestarts=0`. Pesos en `nn/pesos-c4/`. Criterio escrito antes.

| sobre dígitos | corrida 2 (fino) | corrida 4 (grueso) |
|---|---:|---:|
| compositor de presencia | 0,718 ± 0,005 | **0,769 ± 0,001** |
| compositor posicional | 0,959 ± 0,001 | 0,957 ± 0,000 |
| fallos (posicional, semilla 1) | 65 | 70 |
| par 1↔8 | 22 | **3** |
| par 4→1 | 1 | **16** |
| de reconocimiento / composición | 50 / 15 | 58 / 12 |

De los 65 fallos de antes, **40 se arreglan** (21 de ellos 1↔8) y aparecen **45 nuevos** (14 son 4→1). Los
16 «4→1» tienen todos el mismo culpable, `recta-V`: los «4» de este dataset son compactos —un triángulo
relleno con un palo debajo— y el detector, ahora que sabe que una recta vertical puede medir 3 celdas,
**ve el cuerpo macizo del 4 como una barra gruesa**. Rejilla en `resultados/fallos-c4.png`.

Contra el criterio: el 1↔8 baja a ≤ 11 ✅ (a 3); el posicional **no** sube ❌ (−0,002, dentro del ruido de
un par de dígitos). La hipótesis del grosor queda **confirmada como causa del 1↔8**, pero arreglarlo así
desplaza el error en vez de quitarlo: un detector de recta gruesa no distingue «trazo grueso» de «zona
rellena». El de presencia sí gana +0,05, porque el «está / no está» de la recta vertical ahora se parece
más al dígito real.

**No se adopta** como oficial (el posicional, que es el bueno, no mejora). Se conserva.

## Corrida 5 (2026-10-03): fino + grueso en el MISMO compositor — el mejor resultado

Sin entrenar detectores: los 13 mapas de la corrida 2 y los 13 de la 4, juntos (26 mapas por dígito,
`nn/combinar.py`), a los mismos dos compositores lineales. Criterio escrito antes (`02-criterio.md`
§ «Corrida 5»).

| sobre dígitos (3 semillas) | fino (c2) | grueso (c4) | **fino + grueso (c24)** |
|---|---:|---:|---:|
| compositor posicional | 0,959 | 0,957 | **0,972 ± 0,000** |
| compositor de presencia | 0,718 | 0,769 | **0,829 ± 0,003** |
| fallos (posicional, semilla 1) | 65 | 70 | **45** |
| 1↔8 | 22 | 3 | **4** |
| 4→1 | 1 | 16 | **6** |
| reconocimiento / composición | 50 / 15 | 58 / 12 | 41 / 4 |
| difíciles en sí (fallan también con píxeles) | 44 | 37 | 32 (71 %) |

Contra el criterio: posicional ≥ 0,965 ✅ (0,972: 21 dígitos mejor que el mejor solo); 1↔8 < 5 ✅ (4);
4→1 < 5 ❌ (6). El riesgo de sobreajuste por doblar las entradas no se materializó.

Lo que dice: **las dos versiones de los detectores llevan información distinta y el compositor la
aprovecha**. Lo que una confunde, la otra lo separa: el lineal aprende a qué versión creer en cada caso.
Los fallos de composición casi desaparecen (15 → 4), y de los 45 que quedan, 32 los falla también un lineal
sobre los píxeles: son, sobre todo, dígitos difíciles en sí. Rejilla en `resultados/fallos-c24.png`.

## Corrida 6 (2026-10-03): el grupo `dig` — 13 detectores obtenidos de los dígitos SIN etiquetas

**Obtenedor** (`nn/obtenedor.py`): 20 dígitos al azar de train (la etiqueta no se lee) → 1.196 parches 5×5
con tinta → k-means esférico, K = 13 → 13 kernels. Cada kernel es un detector independiente: su mapa 8×8 es
la similitud coseno (suavizada) entre el kernel y el parche de cada celda. Kernels en `nn/pesos-dig/`
(`kernels.pt`, `kernels.json` con qué dígitos se usaron), imagen en `resultados/kernels-dig.png`.

**Nombres.** Grupo `dig`; los anteriores son `fino` y `grueso`. Cada detector `dig:NN` (01 = el patrón más
frecuente entre los parches) más un alias automático orientación-posición de la tinta (`V-E`, `/-SE`, `~-N`…).
⚠ El alias es aproximado: `dig:06` dice `V-c` y a la vista es más un lazo; la identidad es el número.

| sobre dígitos (3 semillas) | `dig` solo (13) | fino + grueso (26) | **fino + grueso + dig (39)** |
|---|---:|---:|---:|
| compositor posicional | 0,888 ± 0,002 | **0,972** | 0,964 ± 0,001 |
| compositor de presencia | 0,524 ± 0,037 | 0,829 | 0,842 ± 0,003 |
| fallos (posicional, semilla 1) | — | 45 | 59 |

Contra el criterio: `dig` solo entre 0,85 y 0,93 ✅ (0,888 — 13 kernels de 20 dígitos sin etiquetas dan casi
lo mismo que los 13 sintéticos finos, 0,959, menos 7 puntos); los 39 juntos ≥ 0,975 ❌ (**0,964: añadir `dig`
EMPEORA** al mejor, −0,008, 13 dígitos).

**Contribución** (dato curioso; quitar un detector y re-entrenar, semilla 1, `resultados/contribucion-*.json`):
- Con los 39, **sólo 6 detectores ayudan** al quitarlos: el que más, `grueso:recta-V` (+7 dígitos); luego
  `fino:arco-E` y `grueso:arco-W` (+3 cada uno). Quitar casi cualquier otro **mejora** el acierto (hasta 7
  dígitos al quitar `fino:lazo`), y **ningún `dig` ayuda** (−2 a −6 dígitos cada uno).
- Dentro de `dig` solo, igual: quitar cualquiera menos `dig:13` mejora (hasta 7 dígitos, `dig:08`).
- ⚠ Con una sola semilla el ruido es de ±2–3 dígitos, así que casi todas estas cifras son ruido; sólo los
  extremos (±7) dicen algo.

**Lectura** (sin medir aún): con 180 dígitos de train y 2.496 entradas, el compositor lineal ya está
**sobreajustado** (train 1,000) y cada mapa añadido le da más con qué memorizar. Los mapas `dig` son densos
—un kernel 5×5 normalizado se parece «algo» a casi cualquier parche— y por eso el análisis del error marca
15 detectores «raros» por fallo frente a 4 por acierto. O sea: los kernels obtenidos **sí llevan información**
(0,888 solos), pero lo que sobra ahora no son detectores, sino capacidad del compositor frente a 180 dígitos.

## Corrida 7 (2026-10-03): el grupo `cae` — 13 detectores CNN de 100 dígitos sin etiquetas — PARCIAL

**Obtenedor** (`nn/obtenedor_cnn.py`): autocodificador convolucional disperso. El codificador son **13 CNN
independientes** (convolución por grupos, ningún peso compartido; `detector(j)` saca cada una como red suelta, y
se comprueba que suelta da exactamente lo mismo); el decodificador, un kernel 5×5 por detector. 100 dígitos de
train al azar, sin leer la etiqueta. Pesos en `nn/pesos-cae/`, imagen en `resultados/detectores-cae.png`.

**Costó tres enmiendas, las tres escritas antes de mirar etiquetas o val** (`02-criterio.md`, corrida 7):
1. Con dispersión L1 el autocodificador **copió píxeles** (kernels = puntos sueltos; un detector = el dígito
   entero). Es la salida trivial de un autocodificador sobrecompleto.
2. Con WTA («el ganador se lo lleva todo») los kernels pasaron a ser **trazos**, pero los mapas saturaron a ≈1.
3. Con ReLU + L1, **11 de 13 murieron**. Se volvió a la receta completa (decodificador lineal, sin L1, WTA
   espacial y de vida, y WTA espacial también al aplicar): **7 de 13 vivos**, con kernels de trazo
   (`\`, `/`, verticales y uno en forma de lazo) y un pico donde encuentran su trazo. Los otros 6 nacieron con
   la ReLU apagada y nunca recibieron gradiente. **Ahí se paró**, como estaba escrito.

| sobre dígitos (3 semillas) | `cae` solo (13, 7 vivos) | `dig` solo | fino + grueso (26) | fino + grueso + `cae` (39) |
|---|---:|---:|---:|---:|
| posicional | 0,743 ± 0,000 | 0,888 | **0,972** | 0,970 ± 0,000 |
| presencia | 0,572 ± 0,011 | 0,524 | 0,829 | **0,875 ± 0,001** |

Contra el criterio: `cae` solo ≥ `dig` ❌ (0,743, 14 puntos por debajo); los 39 no pasan de 0,972 ✅ (0,970).
Lo único que mejora es el compositor de **presencia** con los 39 (0,829 → 0,875).

**Contribución dentro de `cae`** (quitar uno y re-entrenar): aquí sí es grande y limpia, porque los mapas son
ralos y no se solapan: `cae:02` vale **114 dígitos**, `cae:03` 74, `cae:01` 48, `cae:07`/`05`/`04` 24–28; los 6
muertos, 0 (−1, ruido). O sea: **6 detectores CNN hacen todo el trabajo del grupo**.

Lectura: el método funciona —kernels de trazo, independientes, sin etiquetas— pero (a) pierde la mitad de los
detectores por neuronas muertas y (b) el mapa de un pico es menos informativo que el mapa graduado de `dig` o de
los sintéticos. Arreglo propuesto, NO aplicado: LeakyReLU o sesgo inicial positivo en la última capa del
codificador (que todos empiecen encendidos), y probar el mapa entero en vez del pico al aplicarlo.

## Corrida 8 (2026-10-03/04): `cae` arreglado en dos tamaños — `cae5` (5×5) y `cae3` (3×3)

Los dos arreglos propuestos al cerrar la 7 (criterio escrito antes, § «Corrida 8»): LeakyReLU + sesgo inicial
+0,5 para que nadie nazca apagado, y el **mapa entero** (graduado) al aplicarlo en vez del pico. Mismos 100
dígitos sin etiqueta. `cae3`: kernel de decodificador 3×3 y campo receptivo 3×3. Pesos en `nn/pesos-cae5/` y
`nn/pesos-cae3/`, imágenes en `resultados/detectores-cae5.png` y `-cae3.png`.

- **13/13 vivos en los dos** ✅ (la 7 tenía 7). **Sin saturar** ✅: activación media del mapa ≤ 0,01 (`cae5`) y
  ≤ 0,18 (`cae3`), muy por debajo del 0,5 que habría parado la corrida.
- Reconstrucción: `cae5` mse 0,040 (la 7: 0,072); `cae3` 0,072.

| sobre dígitos (3 semillas) | `cae` (c7) | **`cae5`** | **`cae3`** | `dig` | fino + grueso | + `cae5` (39) | + `cae3` (39) |
|---|---:|---:|---:|---:|---:|---:|---:|
| posicional | 0,743 | 0,855 | **0,926** | 0,888 | **0,972** | 0,967 | 0,967 |
| presencia | 0,572 | **0,726** | 0,490 | 0,524 | 0,829 | **0,891** | 0,866 |

Contra el criterio: `cae5` > 0,743 ✅ (0,855) y cerca de `dig` (0,888) — casi; **`cae3` por debajo de `cae5` ❌:
salió al revés, 0,926**, el mejor de los grupos obtenidos sin etiquetas y a 3 puntos de los sintéticos finos
(0,959). Lectura posible (sin medir): el trazo 3×3 es un trozo genérico que aparece en muchos sitios, y lo que
informa es **dónde** aparece; el posicional lo aprovecha (0,926) y el de presencia, que no ve el dónde, se hunde
(0,490). Con el 5×5 es al revés: el trozo es más específico y ya «dice» más sólo con estar (presencia 0,726).

Añadidos a fino + grueso, ninguno pasa del 0,972 (0,967 los dos) — el mismo techo que `dig` y `cae`, coherente
con el sobreajuste del compositor (pendiente). El de presencia sí sube: 0,829 → **0,891** con `cae5`.

**Contribución dentro de cada grupo** (quitar uno y re-entrenar, semilla 1):
- `cae5`: todos menos uno ayudan; `cae5:01` 71 dígitos, `cae5:04` 63, `cae5:09` 37, `cae5:03`/`08` ~31.
- `cae3`: reparto mucho más plano (`cae3:01` 18, `cae3:03` 14, el resto ≤ 10): ningún trozo 3×3 es imprescindible.

## Lo que queda pendiente

- ~~Re-entrenar arcos y esquinas con sus contra-casos~~: hecho en la corrida 3, no mejora (arriba).
- Comprobar si el techo de ~0,70–0,77 de arcos y esquinas es **ambigüedad del dato a 8×8** (radio chico).
- ~~Dataset con trazos gruesos y re-entreno~~: hecho en la corrida 4 (arriba) — 1↔8 arreglado, 4→1 nuevo.
- ~~Fino y grueso a la vez~~: hecho en la corrida 5 (arriba) — 0,972, el mejor.
- ~~Obtener features de los dígitos sin etiquetas~~: hecho en la corrida 6 (grupo `dig`), no mejora al combinar.
- ⏳ **PENDIENTE PRIORITARIO, pedido por el dueño el 2026-10-03: el SOBREAJUSTE del compositor.** Con 180
  dígitos de train y 832–2.496 entradas, el compositor posicional llega a acierto 1,000 en train en todas las
  corridas, y en la 6 quitar casi cualquier detector *mejora* val. Antes de juzgar si un banco de detectores
  más grande ayuda, hay que frenar eso. Opciones, a elegir con criterio escrito antes: (a) L2 del compositor
  elegido por validación cruzada **dentro** de los 180 de train (val no se toca), (b) mapas reducidos a 4×4
  (÷4 entradas), (c) compositor de presencia + posición gruesa (máximo por cuadrante). Lo que decide: si con
  el compositor regularizado los 39 detectores superan a los 26.
- ~~`cae` con los 13 vivos y mapa graduado~~: hecho en la corrida 8 (`cae5`, `cae3`).
- Un `dig` con mapas más ralos (umbral, o competencia entre kernels) para que no se parezca «algo» a todo.
- Curva por tamaño de train (§7 de la especificación): no hecha. La atribución del error, hecha (arriba).
- Una sola semilla por detector; un solo compositor lineal (sin MLP).
