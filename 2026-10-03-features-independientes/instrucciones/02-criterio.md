# Criterio — escrito el 2026-10-03, ANTES de generar el dato y de entrenar

Lo que se decide aquí se decide sin haber visto ningún número de este experimento. Si al correr
hace falta cambiar algo, se añade como **enmienda fechada** al final, nunca editando lo de arriba.

## A. Por detector (13), sobre val sintético

Val de un detector `f`: sus 400 positivas (`principal == f`) y **todas** las negativas de val
(`principal != f` y `secundaria != f`), que es desbalanceado a propósito: así es el uso real (la
feature está en pocas imágenes).

Con la regla de lectura de `ESPECIFICACION.md` §5 (hallada si `max σ ≥ 0,5`; posición = celda del
máximo; buena si dista ≤ 1 celda del ancla), con `best.pt`:

| veredicto | condición |
|---|---|
| **aprendió** | F1 ≥ 0,90 **y** posición ≤ 1 celda en ≥ 0,90 de las positivas halladas |
| **a medias** | F1 ≥ 0,75, o F1 ≥ 0,90 con posición < 0,90 |
| **no aprendió** | lo demás |

Y **tres lecturas obligatorias aunque el veredicto sea «aprendió»**, porque son las que deciden el
siguiente paso y no el ranking:

1. **Recall por tramo de radio** (arcos, lazo): si algún tramo queda **≥ 0,15** por debajo del
   mejor, se anota como candidato a partir el detector por radio. No se parte en esta corrida.
2. **Falsos positivos por familia negativa**: la familia con más FP es el **contra-caso prioritario**
   del re-entrenamiento. Si una familia sola concentra ≥ 50 % de los FP, se dice.
3. **Grosor**: si el recall con grosor 2 px queda ≥ 0,15 bajo el de 4 px, el detector depende del
   grosor, y los dígitos reales (trazos de 2–3 celdas de ancho tras la reducción) lo van a sufrir.

## B. Compositores sobre dígitos (1797 de `uci-optdigits-8px-r20261002`, 180 train / 1617 val)

No se declara ganador contra nada. Se reporta:

- exactitud de val del **compositor de presencia** (13 entradas) y del **posicional** (832);
- la **matriz de confusión** del posicional y, en concreto, el par **6 ↔ 9** y el par **presencia
  vs posicional en 6/9**: es la medida directa de si la posición entra en el resultado;
- **azar = 0,10**; «los detectores ven algo en manuscrito» si el de presencia supera **0,40**
  (cuatro veces el azar con 13 números por imagen). Si no lo supera, la transferencia
  sintético → manuscrito falló y los compositores no tienen materia prima: eso es el resultado, y
  el siguiente paso es §A.2/§A.3, no el compositor.

## C. Qué esperaríamos ver (hoy, sin nada corrido)

- Rectas y lazo: **aprendió** con holgura. Arcos: aprendió, con el tramo `chico` (r < 7 px, menos
  de 2 celdas) como el flojo — a esa escala un arco y una esquina se parecen. Esquinas: a medias;
  su contra-caso principal serán los arcos de la misma orientación.
- Sobre dígitos, el de presencia entre 0,40 y 0,65; el posicional por encima del de presencia, y
  la diferencia concentrada en 6/9 y en 2/5. Si el posicional **no** gana al de presencia, el
  argumento de `feat-pos` pierde su primer apoyo empírico y hay que decirlo así.
- Lo que más probablemente salga mal: la transferencia de **grosor** (el trazo sintético de 2–4 px
  frente al manuscrito, más grueso y más irregular tras la reducción).

## Enmiendas del 2026-10-03, DESPUÉS de la corrida 1 (y antes de la 2)

La corrida 1 (13 detectores, criterio de arriba tal cual; resumen en
`resultados/corrida-1-2026-10-03.json`) dio **12 «no aprendió» y 1 «a medias»** (`lazo`), con un
patrón uniforme: recall 0,70–0,94 y posición ≤1 celda 0,91–1,00, pero **precisión 0,17–0,40**. Las
dos enmiendas salen de leer *dónde* estaban los falsos positivos, no de bajar el listón:

1. **Un negativo no puede CONTENER la feature.** Los FP de los arcos eran sobre todo `lazo`
   (tasa 0,57–0,86) y las esquinas de su misma orientación (0,5–0,67); los de las rectas, sobre las
   esquinas. Una elipse **lleva** un `(` y una `└` **lleva** una recta vertical: contarlos como
   negativos pedía al detector *discriminar* (que no es la premisa), y el «fallo» era un acierto.
   Desde la corrida 2, `features.CONTIENE` declara la relación (lazo ⊃ 4 arcos; esquina ⊃ recta-V,
   recta-H) y esas familias salen de los negativos del detector contenido, en train **y en val**.
   `--contra` sigue mandando si se nombra una explícitamente.
2. **El umbral de «hallada» se elige POR DETECTOR sobre TRAIN** (el que maximiza F1 sobre las 2000
   positivas y 6000 negativas fijas de train), y queda guardado en el checkpoint. El 0,5 fijo era
   una decisión arbitraria sobre un número cuya escala depende del peso de la pérdida: medido en
   `arco-E`, cambiar ese peso de 8 a 1 movía la lectura de (P 0,21, R 0,84) a (P 0,80, R 0,27) sin
   que el mapa fuera mejor ni peor. Val sigue sin intervenir en nada que se elija.

Lo que **no** cambia: la red (los 3.585 parámetros; se ensayó 16/32/32 en `arco-E` y dio +0,03 de
F1, no es ahí), las 40 épocas, la semilla, los umbrales del veredicto (§A) y los de §B.

⚠ Lo que sí se mueve con la enmienda 1 y hay que leer con eso delante: la **precisión de la corrida
2 no es comparable** con la de la 1, porque el conjunto de negativos es otro (más pequeño y más
difícil de confundir por definición). Es el precio de medir lo que se quería medir.

### Enmienda 3 (mismo día, antes de la corrida 2): la pérdida entrena también el MÁXIMO, y la red crece

Con las enmiendas 1 y 2 solas, `arco-E` daba F1 0,45 y `recta-V` 0,70 (ensayos en `/tmp`, 40 épocas):
mejor que la corrida 1, lejos del 0,90. La causa que quedaba estaba en la pérdida: se entrenaba
**cada celda** contra su gaussiana y se leía **el máximo** del mapa como presencia, y nadie entrenaba
el máximo. Se añade un término `BCE(max logits, ¿hay ancla?)` con peso 1 (estilo instancia múltiple:
una imagen positiva tiene que tener *algún* máximo alto; una negativa, ninguno). Medido en los mismos
dos detectores: `recta-V` 0,70 → **0,85**, `arco-E` 0,45 → **0,61** (best en la época 40, aún
subiendo). Con 80 épocas, 0,64; con 80 épocas **y canales 16/32/32** (14.081 parámetros), **0,71**
con posición 0,93. La corrida 2 va con eso: presencia ×1, 80 épocas, 16/32/32. Todo lo ensayado está
en `resultados/corrida-1-2026-10-03.json` y arriba; ninguna de estas elecciones miró el val de otro
detector que no fueran `arco-E` y `recta-V`, y los 11 restantes se ven por primera vez en la corrida 2.

## Corrida 3 — re-entrenar con contra-casos (escrito el 2026-10-03, ANTES de lanzarla)

**Qué se hace.** Los 8 detectores que no llegaron a «aprendió» o se quedaron cerca (los 4 arcos y las
4 esquinas) se re-entrenan **desde su `last.pt` de la corrida 2**, 40 épocas más, `lr` 1e-3 (la mitad:
es un ajuste, no un entrenamiento desde cero). Las rectas y el lazo no se tocan.

**Cuáles son sus contra-casos — regla mecánica, no elección:** las **3 familias con más falsos
positivos** (en número) en el `summary.json` de la corrida 2 (`entrenar_local.contra_casos`). Ninguna
concentraba ≥ 50 % de los FP (la mayor, 13 %), así que **no** se restringen los negativos a ellas
(`--contra`, que haría olvidar el resto): se **enfatizan** (`--enfatizar`), la mitad de las negativas
de cada lote sale de esas tres familias y la otra mitad de todas, como antes.

**Qué se reporta:** el mismo §A, sobre el mismo val, para los 8; y los compositores (§B) con la
mezcla «corrida 2 para rectas y lazo + corrida 3 para los 8» (`resultados/compositores-c3.json`),
al lado de los de la corrida 2. Los pesos de la corrida 2 **no se pisan** (`nn/pesos/`); los nuevos
van a `nn/pesos-c3/`.

**Qué se espera:** precisión de arcos y esquinas de ~0,62 a ≥ 0,75, recall casi igual, y el F1 de al
menos la mitad de los 8 por encima de 0,80. Sobre dígitos, un efecto **pequeño** en el compositor
posicional (ya está en 0,959) y uno mayor en el de presencia, que es el que depende de que el
«está / no está» sea limpio. Si los 8 mejoran en val y los compositores **no** se mueven, eso dice que
el cuello de botella en dígitos no era la precisión sintética.

**Lo que no se decide aquí:** adoptar `pesos-c3` como los oficiales. Se decide mirando §A y §B juntos,
y se escribe.
