# `feat-ind` — Especificación de las features (2026-10-03)

**Orden del dueño:** empezar por **curvas abiertas**, con distintos radios y distintas direcciones
(la dirección del centro), todo sobre **8×8** para que el aprendizaje sea corto; las curvas pueden
ir suavizadas; **guardar los pesos** de cada detector; que cada detector sea **re-entrenable** cuando
haya otras features (contra-casos); generar además **rectas, curvas cerradas y esquinas**; un
training set por grupo según mi criterio; y la salida de todo detector es *«¿está la feature?»* y
*«dónde, aproximadamente, en una matriz 8×8»*.

Todo lo de aquí abajo es una decisión escrita **antes** de entrenar nada. Las que son mías (no del
encargo) van marcadas con *(decido)* y su motivo.

## 1. Cómo se dibuja: a 32×32, y se reduce a 8×8 contando bits

Igual que hizo NIST con los dígitos del dataset `uci-optdigits-8px-r20261002`: el trazo se
rasteriza como **máscara binaria de 32×32**, se parte en bloques 4×4 y cada celda del 8×8 vale el
**número de bits encendidos (0..16)**. Así una feature sintética y un dígito real viven en el **mismo
espacio** —cuentas 0..16 por celda, la misma reducción /4— y los detectores entrenados sobre
dibujos se pueden aplicar a los dígitos tal cual.

**El «suavizado» sale de ahí** *(decido)*: el conteo de bits convierte un trazo binario en 16 niveles
de gris por celda, que es exactamente el antialias que traen los dígitos. No se añade ningún
desenfoque aparte: lo haría distinto del dato real.

Unidades: 1 celda del 8×8 = 4 px del lienzo de 32. Todas las medidas de abajo están **en píxeles
de 32**; entre paréntesis, en celdas.

## 2. El vocabulario: 13 features en 4 grupos (+ el vacío)

| grupo | feature | qué es | parámetros que varían (uniformes) | ancla (la «posición») |
|---|---|---|---|---|
| **arcos** (curvas abiertas) | `arco-E` | arco de circunferencia cuyo **centro queda al Este** del arco: `(` | radio `r` ∈ [4, 14] px (1–3,5 celdas) · apertura ∈ [100°, 200°] · dirección del centro = E ± 40° · grosor ∈ {2, 3, 4} px | el **punto medio del arco** (sobre la tinta) |
| | `arco-W` | centro al Oeste: `)` | ídem, W ± 40° | ídem |
| | `arco-N` | centro al Norte: `∪` | ídem, N ± 40° | ídem |
| | `arco-S` | centro al Sur: `∩` | ídem, S ± 40° | ídem |
| **rectas** | `recta-V` | vertical | largo ∈ [10, 28] px · ángulo 90° ± 12° · grosor {2,3,4} | el punto medio |
| | `recta-H` | horizontal | ídem, 0° ± 12° | ídem |
| | `recta-S` | diagonal `/` | ídem, −45° ± 12° | ídem |
| | `recta-B` | diagonal `\` | ídem, +45° ± 12° | ídem |
| **lazos** (curvas cerradas) | `lazo` | elipse cerrada | semieje `a` ∈ [4, 11] px · razón b/a ∈ [0,7, 1,4] · rotación libre · grosor {2,3,4} | el **centro** de la elipse (sobre el hueco, no sobre la tinta) |
| **esquinas** | `esquina-NE` | dos brazos desde un vértice, hacia el Norte y el Este: `└` | ángulo entre brazos 90° ± 25° · rotación del conjunto ± 20° · cada brazo ∈ [8, 16] px · grosor {2,3,4} | el **vértice** |
| | `esquina-NW` | `┘` | ídem | ídem |
| | `esquina-SE` | `┌` | ídem | ídem |
| | `esquina-SW` | `┐` | ídem | ídem |
| — | `vacio` | nada dibujado (sólo ruido) | — | no tiene |

**Convención de direcciones** (coordenadas de imagen, `y` hacia abajo): E = (+x), S = (+y), W = (−x),
N = (−y). «Dirección del centro» de un arco = hacia dónde queda su centro de curvatura visto desde el
punto medio del arco. Los ±40° de jitter dejan 10° de hueco entre clases vecinas a propósito: dos
clases que se tocan no se pueden evaluar.

**Por qué la dirección es la clase y el radio NO** *(decido)*: el dueño pide «distintos radios y
distintas direcciones». La dirección cambia la *forma* (un `(` y un `∪` no se parecen); el radio
cambia el *tamaño* de la misma forma, y lo que queremos de un detector es que sea **invariante al
tamaño** dentro del rango. Por eso un detector por dirección, con el radio como variación interna —
**y el resumen de cada detector da el acierto por tramo de radio** (`chico` [4,7), `medio` [7,10),
`grande` [10,14]): si un tramo falla, la decisión de partir el detector por radio se toma con ese
dato, no antes.

**Por qué el ancla es el punto medio del arco y no el centro de curvatura** *(decido)*: el centro de
curvatura de un arco de radio 14 cae fuera del lienzo la mitad de las veces; el punto medio siempre
está sobre tinta. Para el lazo sí es el centro: es lo que lo distingue (el hueco).

## 3. Lo que ve cada imagen, además de su feature

- **Ruido en todas** *(decido)*: antes de contar bits, cada bit de fondo se enciende con probabilidad
  `p_on` ∈ [0, 0,02] y cada bit de tinta se apaga con `p_off` ∈ [0, 0,10]. Son los `externos` y el
  `borrado` que `ruido-nist` midió como no dañinos; sin ruido un detector aprende el píxel exacto.
- **Una segunda feature en la mitad de las imágenes** *(decido)*: con probabilidad 0,5 se dibuja
  **otra** feature (de cualquier familia distinta) en el mismo lienzo, en posición libre, pudiendo
  tocar o cruzar a la principal. Es lo que convierte «reconocer» en «reconocer entre otras», y es el
  caso real: en un dígito nunca hay una feature sola. El ancla y la máscara guardadas son **sólo de
  la principal**.
- **Posición libre** con margen: ningún punto del trazo sale del lienzo (≥ 1,5 px del borde).

## 4. El dataset publicado: `feat-ind-sinteticas-8px-r20261003`

Se genera **una vez**, determinista (una semilla por familia y partición), y se publica en
`foveal-vision-data/experimentos-cnn/`. Por familia: **2000 train + 400 val**; `vacio`: 1000 + 200.
Total 32.400 imágenes de 8×8 (≈2 MB de cuentas + 2 MB de máscaras).

Cada muestra trae: `imagenes` (8×8 uint8 0..16, **con** ruido y con la secundaria si la hay) ·
`mascara8` (8×8 uint8 0..16: la cobertura de la principal **sola y limpia**) · `principal`,
`secundaria` (índice de familia; −1 = ninguna) · `ancla` (fila, col del 8×8; −1 en `vacio`) ·
`ancla32` (fila, col en el 32×32, float) · `radio`, `grosor`, `angulo`, `apertura`, `largo` (−1
donde no aplica) · `particion`.

**Training set de un detector `f`** (se arma al cargar, no se guarda aparte):

- positivos: `principal == f` (2000 train);
- negativos: `principal != f` **y** `secundaria != f` — o sea todas las demás familias y el vacío,
  menos las que lleven `f` de segunda. ~24.000.
- **`--contra`** restringe los negativos a una lista de familias: es el mando para **re-entrenar**
  un detector cuando aparezcan contra-casos nuevos (`--desde pesos/<f>/last.pt --contra <lista>`).
  Una familia nueva se añade al vocabulario, se **publica un dataset nuevo** (nombre nuevo, nunca
  se reescribe el de hoy) y se re-entrena desde los pesos guardados.

## 5. La salida de un detector: UN mapa 8×8 de logits

Red **totalmente convolucional** con padding (el mapa no se encoge): 3 convoluciones 3×3 (8, 16, 16
canales) + una 1×1 → **un canal 8×8**. Campo receptivo 7×7 celdas. 3.585 parámetros.

- **Objetivo** de una imagen positiva: una campana gaussiana de σ = 0,6 celdas centrada en el ancla
  (vale 1 en la celda del ancla, ≈0,25 en las 4 vecinas, ≈0,06 en las diagonales). De una negativa:
  todo cero.
- **Pérdida**: entropía cruzada binaria por celda, con las celdas del objetivo pesadas ×8.
- **Lectura** («¿está?» y «¿dónde?») — una sola regla, sin cabeza aparte:
  - **hallada** si `max σ(mapa) ≥ 0,5`;
  - **posición** = la celda del máximo. Se da por buena si está a **≤ 1 celda** (Chebyshev) del ancla.

**Por qué un solo mapa y no mapa + escalar** *(decido, y es la sugerencia que pedía)*: un escalar
«presencia» aparte se entrena con el mismo dato que el máximo del mapa y acaba redundante; y el mapa
solo generaliza gratis a **dos instancias** de la misma feature (dos máximos), que el escalar no
puede expresar. El precio es que el umbral 0,5 sobre el máximo es una decisión; el resumen de cada
detector guarda la curva precisión/recall por umbral para poder moverlo después.

**Dos alternativas que dejo anotadas y no elegidas**: (a) que el objetivo sea la **máscara entera**
del trazo (`mascara8` está guardada para poder probarlo sin regenerar nada): da «dónde» con más
detalle, pero «cuántas» y «cuál» se vuelven una segmentación, que es otro problema; (b) salida con
**dos canales** (presencia + desplazamiento sub-celda): más precisión de posición de la que 8×8
justifica hoy.

## 6. Qué se guarda de cada detector

`nn/pesos/<feature>/`: `last.pt` · `best.pt` (mejor F1 de val) · `config.json` · `metrics.jsonl`
(por época) · `summary.json` (precisión, recall, F1, posición ≤1 celda, acierto por tramo de radio y
por grosor, **falsos positivos por familia negativa**, curva P/R por umbral). 13 detectores ×
~15 KB cada `.pt`: muy por debajo del tope de 5 MB del repo.

Los falsos positivos por familia son el dato que dice **qué contra-caso falta**: si `arco-E` dispara
con `esquina-SE`, ésa es la familia que pesa más en el siguiente re-entrenamiento.

## 7. Evaluación con dígitos (los compositores) — ideas, y lo que se hace hoy

Una vez entrenados los 13, se pasan **todos** sobre los 1797 dígitos de
`uci-optdigits-8px-r20261002` (180 train / 1617 val, el reparto ya publicado): 13 mapas de 8×8 por
dígito. Sobre eso, de menos a más estructura:

1. **Mirar.** Una rejilla PNG: por dígito, su imagen y los 13 mapas. Es lo primero, porque dice si
   los detectores *ven algo* en trazos manuscritos antes de que ningún número lo resuma.
2. **Compositor de presencia** (13 números: el máximo de cada mapa) → regresión logística a 10
   clases, entrenada con las 180 de train. Contesta «¿basta con *qué* features hay?». Es ciego a
   la posición: el 6 y el 9 deberían confundirse — y si no se confunden, el vocabulario los separa
   por otra vía (p. ej. `arco-E` contra `arco-W`).
3. **Compositor posicional** (13 × 64 = 832 números) → regresión logística. Contesta cuánto añade
   el *dónde*. Un MLP pequeño encima, sólo si el lineal se queda corto.
4. **Atribución del error** (la ventaja propia de esta arquitectura): para cada val fallado por el
   compositor 3, ¿qué detectores dispararon? Se tabula por (clase real, clase predicha) qué features
   faltaron o sobraron respecto del patrón medio de la clase real. Separa «el detector no vio» de
   «se compuso mal» sin anotar nada a mano.
5. **Curva por tamaño de train** (18 → 180 por clase no se puede, el train publicado es 180; se
   submuestrea 36, 90, 180): si la ventaja de componer existe, aparece con pocos datos (§4 de
   `OBSERVACIONES.md`).

**Hoy se hacen la 1, la 2 y la 3** (`nn/aplicar.py` y `nn/compositor.py`). La 4 y la 5 son el
siguiente paso y dependen de que la 3 dé algo por encima del azar. **No se compara contra una CNN
monolítica**: el dueño lo apartó por ahora. Los 0,851/0,870 de `dim-nist`/`ruido-nist` son
contexto, no brazo.

El criterio numérico —qué se llama «aprendió» por detector y qué se espera de los compositores— va
en `instrucciones/02-criterio.md`, escrito antes de correr nada.

## 8. Enmienda del 2026-10-03 tras la corrida 1: la relación «contiene» y el umbral por detector

Dos cosas de §4 y §5 cambian a partir de la corrida 2 (el motivo, medido, en
`instrucciones/02-criterio.md` § «Enmiendas»):

- **§4, negativos:** además de `principal != f` y `secundaria != f`, se excluyen las familias que
  **contienen** `f` como parte (`features.CONTIENE`: `lazo` ⊃ los cuatro arcos; cada `esquina-*` ⊃
  `recta-V` y `recta-H`). Un detector de `(` que dispara sobre el lado izquierdo de un `0` está
  reconociendo, no fallando. Esto convierte la lista de features en un **vocabulario con relación de
  parte**, que es lo que el compositor necesita de todos modos.
- **§5, lectura:** «hallada si `max σ ≥ u_f`», con **`u_f` elegido por detector sobre train** y
  guardado en su checkpoint, en vez del 0,5 fijo. La posición sigue siendo la celda del máximo.
- **§5, pérdida y red:** a la BCE por celda se suma una **BCE sobre el máximo del mapa** («¿hay
  ancla en esta imagen?»), peso 1: es lo que se lee, así que se entrena. Y los canales pasan de
  8/16/16 (3.585 parámetros) a **16/32/32 (14.081)**, con **80 épocas**. Los tres números salen de
  ensayos sobre `arco-E` y `recta-V` solamente (criterio, enmienda 3).
