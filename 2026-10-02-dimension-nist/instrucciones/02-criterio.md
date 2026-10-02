# Criterio — escrito el 2026-10-02, ANTES de entrenar

Es el criterio de `dim-gen` con **exactitud** en vez de IoU, y los números de este dato. Lo que
diga aquí manda sobre lo que se vea después (R13); si hay que cambiarlo, antes de la primera
corrida y con fecha.

## Qué se entrena

- **30 corridas**: `W ∈ {8, 7, 6, 5, 4}` × semillas `{1..5}`, más el control `w8-de4` × 5 (la imagen
  de 4 px repetida 2×2 hasta 8, con la red de `w8`: mismos parámetros que `w8`, misma
  información que `w4`).
- La red: `L = 2` capas `valid` con `n = ⌈W/2⌉` (4, 4, 3, 3, 2), `C = 8`, promedio global y
  `Linear(8 → 10)`. Parámetros: 1.258 (W = 8 y 7), 754 (6 y 5), 394 (4): **×3,2** entre extremos.
- **4000 pasos** de lote 20 (9 pasos por época sobre las 180 de train → 444 épocas), Adam con **un
  solo `lr`** para todos los brazos (congelado en el ensayo de mecanismo mirando sólo la pérdida de
  train), entropía cruzada, sin aumento, sin parada temprana, **sin selección**: cuenta `last.pt`.
- La semilla fija la inicialización y el orden de los lotes (común a los brazos de la semilla).

## Qué se mide

Por corrida, con `last.pt`: `acc_val` (exactitud sobre las **1617**), `acc_train` (180),
`brecha = acc_train − acc_val`, **y la entropía cruzada** `ce_train`, `ce_val`,
`brecha_ce = ce_val − ce_train`. Por brazo: media ± sd de 5 semillas, `SE = sd/√5`. Curva cada
10 épocas en `metrics.jsonl`; `summary.json` lleva además la exactitud **por dígito**.

⚠ **Por qué la entropía cruzada**: el ensayo de mecanismo (2026-10-02) ya enseñó que a `W = 8` la
exactitud de train llega a **1,0**. Saturada, `brecha = 1 − acc_val` y toda caída se leería como
«peor generalización» sin que (a) pudiera salir nunca. La **exactitud** contesta las preguntas
del dueño (W\*, suficiente, estorba); la **entropía cruzada**, que no satura, es la que lleva la
descomposición (a)/(b) y «la brecha crece».

## El piso, escrito antes

**Adivinar al azar entre 10 clases: 1/10 = 0,1.** «La clase mayoritaria de train» no existe (18
por clase: empatan) y la de val sería 0,102. Un brazo **aprendió** si `acc_val − 0,1 > 2·SE`.

## La resolución del instrumento

Dos medias de exactitud son **distintas** si `|Δ| > umbral`, con `umbral = max(2·SE_dif, δ)`,
`SE_dif = √(SE_a² + SE_b²)` y **`δ = 0,01`**: un punto porcentual sobre 1617 imágenes (16
imágenes), y 1,3 veces el error estándar binomial de una sola corrida con `p ≈ 0,9` y `n = 1617`
(0,0075, calculado). Para la entropía cruzada, **`δ_ce = 0,05` nats** con la misma forma.

## Las dos preguntas, y cómo se contestan

1. **¿Hay un `W` mínimo suficiente?** `W*` = el `W` de mayor `acc_val`; suficiente = a menos del
   umbral de `W*`; el mínimo suficiente es el menor de ésos.
2. **¿La resolución estorba?** «Estorba» si `acc_val(8) < acc_val(W*) − umbral` con `W* < 8`.
   «La brecha crece con la resolución» si `brecha_ce(8) − brecha_ce(W mín. suficiente) > umbral_ce`.

## Cómo se separan las tres causas de una caída (la lección de `dim-gen`)

| causa | cómo se ve | qué la separa |
|---|---|---|
| **(a) menos información** | `ce_train` **sube** más de `umbral_ce` y `brecha_ce` no crece | la descomposición `ce_val = ce_train + brecha_ce` |
| **(b) peor generalización** | `ce_train` se mantiene y `brecha_ce` crece más de `umbral_ce` | la misma |
| **(c) más parámetros** | indistinguible de (b) con los 25 brazos | sólo el control `w8-de4` |

⚠ **Y una cuarta que `dim-gen` enseñó**: una caída de `acc_train` puede venir del **optimizador**
(semillas que no entrenan), no del dato. Se mira la pérdida final de train de **cada semilla**:
una semilla cuya entropía cruzada de train final queda en o por encima de **1,0** (la de
adivinar al azar entre 10 clases es ln 10 = 2,30; en el ensayo, una red que aprende baja de 0,5
en todos los `W`) se reporta como **atascada**, y se dice si el orden de la curva cambia al
quitarla. Lo que decide el veredicto siguen siendo las 5.

**Control**: `acc_val(w8-de4) ≈ acc_val(w4)` → los parámetros no explican `w8 − w4` (es
información); `≈ acc_val(w8)` → lo explican los parámetros; en medio → se reporta la fracción.

## Las formas posibles, escritas antes

(i) sube con `W` y ningún `W < 8` es suficiente: más píxeles siguen ayudando. (ii) meseta:
hay un `W` mínimo suficiente `< 8` y 8 no estorba. (iii) máximo interior: `W* < 8` y 8 estorba.
**Y «no hubo señal» es un resultado.** ⚠ Expectativa escrita antes: aquí 8 px ya es un dato
**muy reducido** (bloques 4×4 de un 32×32), y las redes son diminutas, así que la forma más
probable es (i) o una meseta corta — lo contrario de `dim-gen`, donde 128 px sobraban. Escribirlo
ahora es lo que permite leer (i) como resultado y no como «no salió».

## Lo que NO decide

- La tarea es **clasificar dígitos**: nada sobre párrafos, cajas, ni la red de producción.
- `W > 8` no existe en este dato (el 32×32 no viene con scikit-learn).
- Un ganador para `ESTADO.md`: no mueve ningún parámetro de `foveal-vision`.

## Riesgos, escritos antes

- **`W = 4`**: 16 píxeles para 10 clases; se espera la caída ahí, y es (a) salvo que la
  descomposición diga otra cosa.
- **Un `lr`**: si en el ensayo no hay uno con el que `W = 4` y `W = 8` bajen sin oscilar, se para.
- **Cabeza de 8 features** (GAP sobre `C = 8`): si en el ensayo `acc_train` a `W = 8` no se acerca
  a 1,0, es el cuello de botella de la cabeza, no la resolución; se sube `C` **antes** de la
  primera corrida y se escribe aquí.
- **4000 pasos**: cortos o largos, lo dice la curva de train; si se cambia, para todos.
- **Zigzag por paridad** (lo señaló el `revisor`): con `n = ⌈W/2⌉`, `f` vale 0,50 en los `W` pares
  y 0,57 / 0,60 en los impares, y el mapa final es 2×2 o 1×1. Si la curva sube y baja alternando
  entre pares e impares, **es la arquitectura, no la resolución**, y se dice; la lectura por
  pares {8, 6, 4} —que es exactamente la regla de `dim-gen`— se reporta aparte.
- **Los escritores**: las 1797 imágenes son el *test set* de UCI, de **13 personas**; train y val
  comparten escritores. Lo que se mide es generalizar a dígitos nuevos **de los mismos
  escritores**, no a escritores nuevos.

## Enmienda del 2026-10-02, antes de que ningún brazo entrenara

El primer lanzamiento se negó en los 30 brazos al arrancar: 4000 pasos no son épocas enteras de
9 pasos (180 imágenes / lote 20). Se fija **3996 pasos = 444 épocas**, exactamente lo que corrió
el ensayo de mecanismo. Ningún brazo había entrenado: nada se miró.
