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

## Atribución del error sobre dígitos (escrito el 2026-10-03, ANTES de mirar ningún fallo)

Sobre los detectores de la **corrida 2** (los oficiales) y el compositor **posicional** con semilla 1.
Como el compositor es **lineal**, su error se descompone exactamente: para un dígito de clase real `t`
predicho como `p`, `logit_p − logit_t = Σ_j (W_p,j − W_t,j) · mapa_j + (b_p − b_t)`. El término de cada
detector `j` dice **cuánto empujó hacia la clase equivocada**. Se llama **culpable** al detector de
mayor empuje.

Después se pregunta si el mapa culpable es **raro para su clase real**: su distancia euclídea al mapa
medio de ese detector en los dígitos de **train** de clase `t`, comparada con la misma distancia en los
dígitos de **val** de clase `t` bien clasificados. Si cae por encima del percentil 90 de esos aciertos:

- **fallo de reconocimiento** — el detector vio algo que no suele ver en esa clase (o no vio lo que suele);
- si no: **fallo de composición** — los mapas son los normales de la clase y aun así el lineal los sumó mal.

Y un control que **no** es una comparación de arquitecturas: una regresión logística sobre los **64 píxeles
crudos**, mismos 180 de train y mismos hiperparámetros. Un dígito que falla también ahí se marca como
**difícil en sí** (mal escrito o ambiguo a 8×8), sin decir nada de los detectores.

Lo que se espera: pocos fallos (≈66 de 1617), concentrados en 3↔8↔9 y 1↔7; **mayoría de reconocimiento**
con arcos o esquinas de culpables (son los detectores con peor precisión), y un tercio o más de los
fallos marcados como difíciles en sí.

## Corrida 4 — trazos GRUESOS (escrito el 2026-10-03, ANTES de lanzarla)

**Por qué.** La atribución del error mostró que un tercio de los fallos es 1↔8 y que la causa medible es el
ancho: un «1» manuscrito tiene ~2,9 celdas con tinta por fila y la recta sintética ~1,1.

**Qué se hace.** Dataset nuevo `feat-ind-sinteticas-grueso-8px-r20261003d` (perfil `grueso`, grosores
2–12 px de 32; otras semillas que el fino). Los **13** detectores se re-entrenan desde su `last.pt` de la
corrida 2, 40 épocas, `lr` 1e-3, **sólo sobre el grueso** (que también trae grosores 2–4, así que no deja
de ver trazos finos). Pesos a `nn/pesos-c4/`. Después: mapas, compositores, atribución y la rejilla de
fallos, todos con sufijo `-c4`.

⚠ **Lo que el dato nuevo cubre y lo que no** (medido sobre 400 muestras por familia antes de publicar):
recta-V 1,74 celdas de ancho medio por fila, 21 % ≥ 2,5; esquinas 58 % ≥ 2,5; **arcos sólo 8 %**, porque su
grosor está acotado por el radio (un arco más grueso que su radio deja de ser un arco). O sea que esto
ataca sobre todo el lado «1 grueso» del par, no el «8 estrecho».

⚠ **Tres versiones descartadas antes de entrenar nada**, publicadas con su aviso: la sin letra (manchas
cuadradas), la `b` (casi no engordaba) y la `c` (los gruesos no cabían en el lienzo). La medida que las
descartó es la misma que la de arriba.

**Qué se espera:** el par 1↔8 baja de 22 a ≤ 11 fallos; los fallos de reconocimiento bajan más que los de
composición; el compositor posicional sube de 0,959. **Qué lo refutaría:** 1↔8 no baja, o el posicional baja
(los detectores pierden precisión en trazos finos sin ganar en gruesos). En sintético se reporta §A sobre el
val **grueso** (otro val: no comparable con la corrida 2) — lo que decide es el dígito.

## Corrida 5 — fino Y grueso en el mismo compositor (escrito el 2026-10-03, ANTES de mirar)

**Qué se hace.** No se entrena ningún detector. Se juntan los 13 mapas de la corrida 2 (`nn/pesos/`, «fino»)
y los 13 de la corrida 4 (`nn/pesos-c4/`, «grueso») de cada dígito: **26 mapas**. Los mismos dos
compositores lineales, mismos hiperparámetros, mismas 3 semillas, mismos 180/1617: presencia con 26
entradas, posicional con 26 × 64 = 1.664. Y la misma atribución del error (`errores.py`, sufijo `-c24`),
con los detectores nombrados `fino:<f>` y `grueso:<f>`.

**Por qué puede servir.** Los dos sistemas empatan (0,959 y 0,957) pero fallan en dígitos distintos: sólo
25 fallos en común de 65 y 70. Un lineal que vea los dos puede aprender, p. ej., «`recta-V` grueso encendido
y fino apagado ⇒ cuerpo de un 4, no un 1».

**Riesgo escrito antes:** con 180 imágenes de train, doblar las entradas (832 → 1.664) puede sobreajustar
más; el posicional ya llega a 1,000 en train con 832.

**Qué se espera:** posicional ≥ 0,965 (al menos ~10 dígitos mejor que el mejor de los dos), con 1↔8 y 4→1
**los dos** por debajo de 5. **Qué lo refuta:** posicional ≤ 0,959 — entonces combinar no suma y el límite
está en otro sitio (el compositor lineal o los 180 dígitos de train).

## Corrida 6 — el grupo `dig`: 13 detectores obtenidos de los DÍGITOS sin etiquetas (escrito el 2026-10-03, ANTES de mirar)

**Encargo del dueño:** un «obtenedor» de features a partir de los dígitos mismos, sin usar las etiquetas,
con pocos dígitos, del orden de 13 detectores (como los grupos anteriores), con nombres propios por detector
y por grupo, y la contribución de cada uno como dato curioso.

**Método (k-means esférico de parches, sin etiquetas).** Se eligen **20 dígitos al azar** de la partición
de train (semilla fija; la etiqueta no se lee). De cada uno se sacan los parches **5×5** centrados en cada
celda (relleno con ceros), se descartan los casi vacíos (norma < 1,0) y se normalizan a norma 1. k-means
esférico con **K = 13**, inicialización k-means++ (semilla 1), 200 iteraciones. Cada centro es un **kernel**.

**Cada kernel es un detector independiente**: su mapa 8×8 es, en cada celda, la similitud coseno entre el
kernel y el parche 5×5 de esa celda, con la norma del parche suavizada (`p·k / (|k|·(|p| + 0,5))`, para que
un parche casi vacío no dé similitud alta) y recortada a [0, 1]. No compite con los otros kernels (no hay
«gana el más parecido»): cada mapa sale sólo de su kernel. Mismo formato que los detectores anteriores, así
que el compositor no cambia.

**Nombres.** Grupo **`dig`** (los otros dos grupos pasan a llamarse, como ya hace el npz combinado, `fino` y
`grueso`). Cada detector: `dig:NN`, numerado por **tamaño de su grupo de parches** (01 = el patrón más
frecuente), y un **alias descriptivo automático** sacado del propio kernel: orientación dominante del trazo
(tensor de estructura: `H`, `V`, `/`, `\`, o `~` si no tiene una) y hacia dónde está la tinta dentro del
kernel (`c`entro, `N`, `S`, `E`, `W`, `NE`…). P. ej. `dig:03 V-W` = trazo vertical en el lado izquierdo del
parche. El alias es sólo para leer; la identidad es `dig:NN`.

**Qué se mide (compositores, mismos de siempre, 3 semillas, 180/1617):**
1. `dig` solo (13 mapas) — ¿cuánto dan 13 kernels sacados de 20 dígitos?
2. `fino + grueso + dig` (39 mapas) contra `fino + grueso` (26, el 0,972 de la corrida 5).
3. **Contribución** (dato curioso, no decide nada): para cada uno de los 39, quitar su mapa y re-entrenar el
   compositor posicional (semilla 1); contribución = acierto con los 39 − acierto sin él. Y lo mismo dentro
   de `dig` solo.

**Riesgo escrito antes:** los 20 dígitos salen de los 180 de train del compositor, así que los kernels han
«visto» parte de ese train (sin etiquetas). No toca val. Y un kernel 5×5 en un dígito de 8×8 abarca más de la
mitad del dígito: es una feature compleja por construcción, como se pidió.

**Qué se espera:** `dig` solo entre 0,85 y 0,93 (kernels sacados del propio dato, pero sin etiquetas y con 20
dígitos); los 39 juntos ≥ 0,975; contribuciones individuales pequeñas (|Δ| < 0,005 casi todas), porque con
39 detectores hay mucha redundancia.

## Corrida 7 — el grupo `cae`: 13 detectores CNN aprendidos de 100 dígitos SIN etiquetas (escrito el 2026-10-03, ANTES de mirar)

**Encargo del dueño:** como `dig`, pero con CNN en vez de k-means, y con unos 100 dígitos aleatorios.

**Método: autocodificador convolucional disperso.**
- **100 dígitos al azar** de la partición de train (semilla fija; la etiqueta NO se lee).
- **Codificador = 13 CNN independientes** (convolución por grupos, sin ningún peso compartido): cada una
  Conv 3×3 (1→8) + ReLU, Conv 3×3 (8→8) + ReLU, Conv 1×1 (8→1) + sigmoide, con relleno → **un mapa 8×8 en
  [0,1]**. Campo receptivo 5×5, como `dig`. Al aplicarlo, cada una es un detector que no depende de las otras.
- **Decodificador**: un kernel 5×5 propio por detector; la reconstrucción es la suma de los 13 mapas convolu-
  cionados con su kernel. Pérdida: error cuadrático de reconstrucción + λ · media de los mapas (dispersión:
  obliga a que cada detector se encienda poco y se especialice).
- Para no memorizar 100 imágenes, en cada paso el dígito se desplaza al azar ±1 celda.
- **λ se elige sólo con los 100 dígitos y sin etiquetas**: entre {0,003, 0,01, 0,03}, el mayor con el que
  los 13 detectores siguen vivos (cada uno con activación media > 0,01) y la reconstrucción no se degrada más
  de un 50 % respecto del menor. Val no interviene.

**Nombres.** Grupo **`cae`** (*convolutional autoencoder*). Detectores `cae:NN`, numerados por activación
media sobre los 100 dígitos (01 = el que más se enciende), con el mismo alias automático que `dig`, calculado
sobre su **kernel de decodificador** (= lo que ese detector «pinta» cuando se enciende).

**Qué se mide:** lo mismo que la corrida 6 — `cae` solo, `fino + grueso + cae` (39) frente a `fino + grueso`
(26), y la contribución por eliminación. Con el compositor **sin cambios** (el sobreajuste queda pendiente,
así que la comparación de 39 contra 26 arrastra el mismo problema y se lee con eso delante).

**Qué se espera:** `cae` solo ≥ `dig` solo (0,888), porque ve 5× más dígitos y cada detector es no lineal;
entre 0,88 y 0,94. Los 39 juntos, igual que con `dig`: no más de 0,972 mientras el compositor sobreajuste.

### Enmienda a la corrida 7 (mismo día, ANTES de mirar ninguna etiqueta ni val)

Con la regla de λ de arriba se eligió λ = 0,003 (13/13 vivos) y el autocodificador **degeneró en copiar
píxeles**: los kernels de decodificador son puntos sueltos, un detector (`cae:09`) reproduce el dígito entero
y dos (`cae:01`, `cae:02`) son el negativo del fondo (`resultados/detectores-cae.png` de esa versión, en la
historia de git). Es la solución trivial de un autocodificador **sobrecompleto** —13 mapas 8×8 para
reconstruir un 8×8—, y la dispersión L1 sola no la impide: con λ mayor mueren detectores antes de que deje de
copiar (12/13 con 0,01; 11/13 con 0,03). Se juzgó sólo con los 100 dígitos y su reconstrucción, sin etiquetas.

**Cambio: dispersión «el ganador se lo lleva todo» (WTA), en vez de L1.** Al entrenar, de cada mapa se conserva
sólo su celda más activa por imagen (espacial), y de esa celda sólo el 20 % de imágenes del lote en que más se
activa (de vida, para que cada detector se especialice en unas imágenes y no en todas). La reconstrucción es
la suma de esos «sellos» 5×5. Así copiar un píxel no sirve: cada detector tiene que explicar un trozo de
trazo con su kernel. Es la técnica de Makhzani y Frey (2015) *(de memoria, no comprobado desde aquí)*. **Al
aplicarlo como detector** se usa el mapa entero (sin WTA): la WTA sólo da forma al aprendizaje. Sin λ que
elegir; mismos 100 dígitos, pasos, `lr` y semilla.

### Enmienda 2 a la corrida 7 (mismo día, todavía sin etiquetas ni val)

Con WTA los **kernels de decodificador salieron trazos** (lo buscado), pero los **mapas del codificador
saturaron a ≈ 1 en todas las celdas** (activación media 0,93–0,99): la WTA sólo entrena la celda ganadora y
nada empuja hacia abajo a las demás, así que la sigmoide se queda arriba. Un mapa constante no es un detector.
**Cambio:** salida ReLU (no sigmoide) y, además de la WTA, una penalización L1 pequeña sobre el mapa ENTERO
(λ = 0,01 sobre la media): los ganadores cargan la reconstrucción y el resto baja a 0. Al aplicarlo, cada
mapa se divide por su escala (percentil 99 de sus valores ganadores sobre los 100 dígitos, guardado con los
pesos) y se recorta a [0, 1], para que esté en el mismo rango que los demás grupos.

### Enmienda 3 a la corrida 7 (mismo día, todavía sin etiquetas ni val)

La enmienda 2 se fue al otro extremo: **11 de 13 detectores muertos** (ReLU a cero, rematados por la L1) y la
reconstrucción sin aprender (mse 0,12, lo de predecir una constante). Dos parches seguidos han fallado, así que
se vuelve a la receta **completa** del método en vez de seguir improvisando: codificador ReLU, **decodificador
lineal** (la sigmoide de salida impedía reconstruir con sellos), **sin L1**, WTA espacial + de vida al
entrenar. La de vida asegura que cada detector gane en el 20 % de cada lote, así que ninguno puede morir.
**Al aplicarlo como detector se usa la WTA espacial** (sólo la celda máxima del mapa, con su valor, escalado por
el p99 de sus ganadores y recortado a [0, 1]): es lo mismo que vio el decodificador, y da un pico en la
posición donde encontró su trazo —el mismo formato «¿está? ¿dónde?» que los detectores sintéticos—.
Si esto también degenera, se para y se dice, en vez de un cuarto parche.

## Corrida 8 — `cae` arreglado, en dos tamaños: `cae5` y `cae3` (escrito el 2026-10-03, ANTES de mirar)

**Encargo del dueño:** aplicar los dos arreglos propuestos al cerrar la corrida 7, y hacer lo mismo con kernels 3×3.

**Los dos arreglos** (sobre la receta de la enmienda 3: decodificador lineal, WTA espacial + de vida, sin L1):
1. **Que nadie nazca muerto:** LeakyReLU (pendiente 0,1) en las capas ocultas **y** en la salida del codificador
   durante el entrenamiento —un detector apagado sigue recibiendo gradiente—, y sesgo inicial +0,5 en la última
   capa de cada detector.
2. **Mapa graduado al aplicarlo:** el detector entrega el mapa ENTERO (`max(0, mapa)` / escala, recortado a
   [0, 1]), no sólo el pico. La escala sigue siendo el p99 de sus valores ganadores sobre los 100 dígitos.

**Dos tamaños, mismos 100 dígitos, misma semilla, mismos pasos:**

| grupo | kernel del decodificador | campo receptivo del codificador | capas de cada detector |
|---|---|---|---|
| `cae5` | 5×5 | 5×5 | Conv 3×3 → Conv 3×3 → Conv 1×1 (como la corrida 7) |
| `cae3` | 3×3 | 3×3 | Conv 3×3 → Conv 1×1 → Conv 1×1 |

Nombres `cae5:NN` y `cae3:NN`, numerados por activación media; el `cae` de la corrida 7 se conserva tal cual.

**Qué se mide:** lo de siempre — cada grupo solo, cada uno añadido a fino + grueso (39), y la contribución dentro
de cada grupo. Con el compositor **sin cambios** (el sobreajuste sigue pendiente).

**Qué se espera:** 13/13 vivos en los dos. `cae5` solo por encima del `cae` de la corrida 7 (0,743) y cerca de
`dig` (0,888): más detectores vivos y mapa graduado. `cae3` por debajo de `cae5` solo —un 3×3 en un 8×8 es un
trozo de trazo, menos específico—, pero con más detectores repartiendo el trabajo. **Qué lo refuta:** que siga
habiendo detectores muertos, o que el mapa graduado sature como en la enmienda 1 (activación media > 0,5); si
pasa, se dice y se para.

## Corrida 9 — el SOBREAJUSTE del compositor (escrito el 2026-10-04, ANTES de mirar)

**El problema** (anotado como pendiente prioritario): con 180 dígitos de train y 832–2.496 entradas, el
compositor posicional llega a 1,000 en train en todas las corridas, y añadir grupos de detectores (`dig`, `cae`,
`cae5`, `cae3`) nunca supera al de 26 (0,972): o los grupos nuevos no aportan, o el compositor no puede
aprovecharlos con 180 ejemplos. Esta corrida separa las dos cosas.

**Método.** Se mantiene el compositor (regresión logística, mismo optimizador, 300 épocas), y se eligen **dos
mandos por validación cruzada de 5 pliegues DENTRO de los 180 de train** (estratificada por clase; **val no se
toca** hasta el final):
- **L2** ∈ {0,001 (el actual) · 0,003 · 0,01 · 0,03 · 0,1 · 0,3 · 1};
- **resolución del mapa** ∈ {8×8 (la actual) · 4×4 (media de bloques 2×2: ÷4 entradas)}.

Se elige la pareja de mayor acierto medio en los 5 pliegues (empate → la más regularizada), se re-entrena con
los 180 y se mide en val con las 3 semillas de siempre. Sólo el compositor **posicional**.

**Bancos que se comparan** (todos con su propia elección de mandos):

| banco | detectores |
|---|---:|
| fino | 13 |
| fino + grueso | 26 |
| fino + grueso + dig | 39 |
| fino + grueso + cae5 | 39 |
| fino + grueso + cae3 | 39 |
| **todos** (fino + grueso + dig + cae5 + cae3) | **65** |

**Qué decide** (umbral = 0,003, unos 5 dígitos, ≈ 3× la sd entre semillas medida hasta ahora):
- Si con el compositor regularizado **algún banco grande supera a fino + grueso por ≥ 0,003**, el sobreajuste
  era lo que tapaba el aporte de los grupos nuevos.
- Si no, los grupos nuevos no aportan nada que fino + grueso no tenga, regularizado o no.
- Aparte: si la regularización sube a fino + grueso por ≥ 0,003 sobre su 0,972.

**Qué se espera:** la CV elige L2 mayor que el actual (0,03–0,3) en los bancos grandes; fino + grueso sube
poco (≤ 0,975); el banco de 65 supera a 26 por ~0,005. Y el acierto en train deja de ser 1,000.

## Corrida 10 — un compositor que COMBINA detectores (escrito el 2026-10-04, ANTES de mirar)

**Por qué.** El compositor lineal sólo **suma** evidencias: «lazo arriba» y «recta abajo» suman cada una por
su lado, pero no puede expresar «lazo arriba **Y** recta abajo» ni «recta-V gruesa **pero NO** fina». La corrida
9 descartó el sobreajuste como causa del techo; queda probar si lo que falta es **combinar**.

**El compositor combinante** (`nn/compositor_comb.py`): entrada = los J mapas 8×8 apilados como canales.
1. **Conv 3×3, J → 16 canales, ReLU, con relleno**: cada canal de salida es una combinación aprendida de
   detectores **en una celda y sus 8 vecinas** (p. ej. «arco-W fino aquí y recta-V gruesa justo debajo»). La
   ReLU es lo que permite el «Y» / «pero no».
2. **Lineal 16 × 64 → 10** (la parte posicional, como el compositor de siempre).

Fijo antes de mirar: 16 canales, Adam `lr` 3e-3, L2 0,001, 300 épocas a lote completo, 3 semillas, los mismos
180/1617. Parámetros: J·9·16 + 16 + 10.250 (de 12.138 con 13 detectores a 19.626 con 65).

**Bancos:** cae3 · fino · grueso · fino + grueso · cae3 + fino + grueso · todos (65). Junto a cada uno, su
compositor lineal (ya medido) para la diferencia.

**Qué decide** (umbral 0,003 ≈ 5 dígitos, como en la corrida 9):
1. ¿El combinante supera al lineal en fino + grueso? → combinar ayuda aunque no haya detectores nuevos.
2. ¿Con el combinante, un banco grande (cae3 + fino + grueso, o todos) supera a fino + grueso por ≥ 0,003? →
   los grupos nuevos sí llevaban información, pero sólo **combinada**.

**Qué se espera:** el combinante sube a todos los bancos (más en los pequeños: con cae3 solo, de 0,926 a ~0,95);
fino + grueso hacia 0,975–0,98; y **sí** gana el banco grande, por poco (~0,005). Riesgo: con 180 dígitos y 12–20
mil parámetros, sobreajuste dañino de verdad; si el combinante queda **por debajo** del lineal en todos los
bancos, es eso.

## Corrida 11 — curva de acierto según el nº de dígitos de train del COMPOSITOR (escrito el 2026-10-04, ANTES de mirar)

**Pregunta:** ¿el techo de ~0,972 es de datos? Si lo es, con más dígitos para entrenar el compositor (a) todos
los bancos suben, y (b) los bancos grandes (65 detectores) y el compositor combinante, que hoy no ganan, pasan
a ganar.

**Reparto nuevo, fijo antes de mirar** (semilla 2026, estratificado por clase; el dato publicado no cambia):
- de las 1617 de val salen **900 a una reserva de train** (90 por clase) y quedan **717 como TEST**;
- el **test es el mismo para todos los puntos de la curva**, y no se ha usado para elegir nada.
- Los kernels de `dig` y `cae*` se aprendieron con 20/100 dígitos de los **180 originales de train** (sin
  etiqueta), y los detectores sintéticos no vieron ningún dígito: nada del test entró en ningún detector.

**Tamaños de train del compositor:** N ∈ {36, 90, 180, 360, 540, 900, 1080}. Por debajo de 180, un subconjunto
estratificado de los 180 originales; por encima, los 180 + una parte estratificada de la reserva (anidados: el de
540 contiene al de 360).

**Qué se compara en cada N:** fino + grueso (26) contra todos (65), con el compositor **lineal** y el
**combinante** (los de las corridas 9 y 10, mismos hiperparámetros), 3 semillas.

⚠ Los números de esta corrida van sobre el **test de 717**, no sobre las 1617 de antes: comparables entre sí,
**no** con los de las corridas 1–10.

**Qué decide** (umbral 0,003, como siempre):
1. ¿Sube fino + grueso lineal de N = 180 a N = 1080 en ≥ 0,01? → el techo era de datos.
2. ¿En algún N ≥ 360, todos (65) supera a fino + grueso (26) en ≥ 0,003 con el mismo compositor? → los grupos
   nuevos aportan cuando hay datos para aprovecharlos.
3. ¿En algún N ≥ 360, el combinante supera al lineal en ≥ 0,003 con el mismo banco?

**Qué se espera:** (1) sí, hacia 0,985 en N = 1080; (2) sí desde N ≈ 540; (3) sí con 65 detectores desde N ≈ 900.

## Corrida 12 — la capacidad GENERALIZADORA de cada caso (escrito el 2026-10-04, ANTES de evaluar)

### Definición

**Generalizar** = acertar con dígitos que el sistema **no vio**, en tres sentidos distintos, y se miden los tres
porque no tienen por qué ir juntos:

- **G1 · brecha (memoria contra generalización):** `acierto en train − acierto en test`, con el compositor
  entrenado con los 180 originales. Pequeña = lo aprendido vale igual fuera. ⚠ Con train en 1,000 la brecha es
  sólo `1 − test`, así que G1 dice lo mismo que el acierto en test; se reporta igual, pero no es la medida fuerte.
- **G2 · eficiencia de datos (generalizar desde pocos ejemplos):** `acierto(N = 36) / acierto(N = 1080)` en el
  test: qué fracción de su mejor acierto alcanza viendo **3,6 dígitos por clase**. Más alto = generaliza antes.
- **G3 · robustez (generalizar a condiciones NO vistas):** el compositor se entrena con los 180 limpios y se
  evalúa sobre el test **transformado** (los detectores también ven la imagen transformada). Medida:
  `acierto transformado / acierto limpio`. Cinco transformaciones, fijas (semilla 2027), ninguna vista nunca al
  entrenar nada:
  1. **desplazar** cada dígito 1 celda en una de las 8 direcciones (relleno con 0);
  2. **ruido** gaussiano σ = 0,15, recortado a [0, 1];
  3. **engrosar**: `0,5·x + 0,5·máximo 3×3`;
  4. **adelgazar**: `0,5·x + 0,5·mínimo 3×3`;
  5. **ocluir** un bloque 3×3 al azar (a 0).
  Y su media, **G3 medio**.

### Casos

Los bancos de detectores fino · grueso · fino + grueso · dig · cae5 · cae3 · todos, con el compositor **lineal
posicional** (el de siempre), y como referencia el mismo lineal sobre los **64 píxeles crudos** (no es una
comparación de arquitecturas: dice cuánto aportan los detectores frente a no tener ninguno). Test = los 717 de la
corrida 11; 3 semillas; N = 36 y 1080 para G2 con el mismo reparto anidado.

### Qué se espera

- **G3, desplazar:** los detectores son convolucionales, pero el compositor posicional no lo es — un desplazamiento
  de 1 celda mueve todos los mapas, así que **todos caen mucho** (a ~0,6–0,7 del limpio), y los píxeles crudos
  también. Es la debilidad esperada del diseño posicional.
- **G3, engrosar/adelgazar:** fino + grueso es el más robusto (ya cubre dos grosores); `cae3` y `dig`, aprendidos
  de los propios dígitos, los más frágiles.
- **G3, ruido y ocluir:** los detectores sintéticos aguantan mejor que los píxeles crudos (se entrenaron con ruido
  y con una segunda feature encima).
- **G2:** fino + grueso por encima de los píxeles crudos (0,87 frente a algo menor), `todos` por debajo de fino +
  grueso (más entradas, peor con pocos datos, como ya se vio).

## Corrida 13 — robustez al DESPLAZAMIENTO: compositores que toleran ±1 celda (escrito el 2026-10-04, ANTES de mirar)

**Por qué.** En la corrida 12, desplazar 1 celda hundía a todos (G3 0,32–0,60). Los detectores son
convolucionales —su mapa se desplaza con el dígito—; lo que no tolera nada es el compositor posicional, que pega
cada evidencia a una celda. Se prueban tres formas de darle tolerancia, **sin tocar los detectores**:

| variante | qué se hace a cada mapa antes del lineal | entradas por detector |
|---|---|---:|
| **A · posicional** (la de siempre) | nada | 64 |
| **B · máx 3×3** | máximo en la vecindad 3×3 de cada celda (paso 1, relleno): una evidencia vale en su celda y en las 8 vecinas | 64 |
| **C · máx 2×2 → 4×4** | máximo por bloques 2×2: la posición se lee en una rejilla 4×4 | 16 |
| **D · aumento** | A, pero el compositor se entrena con cada dígito de train y sus 8 desplazamientos de 1 celda (180 × 9) | 64 |

⚠ **D no mide lo mismo que B y C.** En D el compositor **ha visto** desplazamientos al entrenar (en otros dígitos),
así que su G3-desplazar deja de ser «generalizar a una condición no vista» y pasa a ser «invariancia aprendida».
Se reporta igual, marcado.

**Casos:** fino + grueso, cae3, todos (65) y los píxeles crudos (a los que se aplica lo mismo: máximo 3×3, 2×2,
aumento). Test de 717, compositor de los 180 originales, 3 semillas, mismas 5 transformaciones de la corrida 12.

**Qué decide** (umbral 0,003 sobre el acierto absoluto):
1. ¿B o C suben el acierto **desplazado** de fino + grueso (0,583) en ≥ 0,10? → la tolerancia del compositor era
   lo que faltaba.
2. ¿A qué precio en el **limpio** (0,974)? Si baja más de 0,01, se dice.
3. ¿Cambia el ranking de casos bajo desplazamiento?

**Qué se espera:** B sube el desplazado a ~0,80 perdiendo ≤ 0,005 en limpio; C sube algo menos y pierde más en
limpio (6↔9 necesita posición fina); D es la que más sube el desplazado (~0,90), a costa de no ser comparable.
Los píxeles crudos con máximo 3×3 mejoran mucho menos que los detectores (un máximo de píxeles emborrona el dígito;
un máximo de detectores sólo mueve evidencias).
