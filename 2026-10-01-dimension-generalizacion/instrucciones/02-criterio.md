# Criterio — escrito el 2026-10-01, ANTES de escribir el código y ANTES de entrenar

Lo que diga aquí manda sobre lo que se vea después (R13). Si al implementar hace falta cambiar
algo, se cambia **antes de la primera corrida** de los brazos y se dice en el commit.

## Qué se entrena

- **25 corridas**: `W ∈ {128, 64, 32, 16, 8}` × semillas `{1, 2, 3, 4, 5}` (`w064-s3`, …). La
  misma red salvo `W` y lo que `W` arrastra —`n = W/4` y, con él, el número de parámetros—:
  `L = 4` capas `valid`, `f = 1/L = 0,25`, `C` canales por capa (decisión 2 del dueño,
  `ESPECIFICACION.md` §7), cabeza `Linear(4·4·C → 4)`.
- **+5 si entra el control `w128-de16`** (decisión 5): la imagen de `W = 16` **reescalada** a
  128 por repetición de píxeles (cada píxel de 16 → un bloque 8×8 del mismo valor), con la red
  **de `w128`**: mismos parámetros que `w128`, misma información que `w016`.
- **1000 pasos** de lote 20 (200 épocas sobre las 100 imágenes de `train`), Adam con **un solo
  `lr` para todos los brazos** (congelado en el ensayo de mecanismo, `REGLAS.md` § Procesos 3),
  L1 sobre las cuatro coordenadas normalizadas, sin aumento, sin parada temprana y **sin
  selección**: cuentan los pesos de la época final (`last.pt`).
- La semilla `s` fija la inicialización y el orden de los lotes. El **orden de los lotes es el
  mismo para todos los brazos de una semilla** (generador propio sembrado con `s`); la
  inicialización no puede serlo, porque las formas son otras.

## Qué se mide

Por corrida `(brazo, s)`, con `last.pt`:

| medida | qué es |
|---|---|
| `IoU_val` | IoU medio de la caja predicha contra la real sobre las **900** (`monitor ∪ eval`), en coordenadas normalizadas a `[0, 1]` —**iguales para todo `W`**—. La predicción se recorta a `[0, 1]`; si `der ≤ izq` o `inf ≤ sup`, IoU = 0 |
| `IoU_train` | lo mismo sobre las 100 de `train` |
| `brecha` | `IoU_train − IoU_val` |

Por brazo: **media ± desviación** entre las 5 semillas, y `SE = sd / √5`. Se guarda también la
curva (cada 10 épocas, `metrics.jsonl`) para poder decir si 1000 pasos bastaron: si a la época
200 la pérdida de `train` sigue bajando con pendiente clara, **se dice**; no se alarga a
posteriori (ver riesgos). Y en `summary.json` va el IoU de `val` **desglosado por factor** del
dato (`fuente`, `cuerpo`, `gris_nivel`, área de caja en cuartiles): el desglose no decide nada
de lo de abajo, pero evita tener que conservar pesos para leerlo después.

## El piso, medido antes de entrenar

**Caja media**: predecir siempre la caja media de `train`. IoU sobre las 900: **0,2464**
*(medido el 2026-10-01 con las etiquetas publicadas; sobre `train`, 0,2699)*. Un brazo
**aprendió** si `IoU_val − 0,2464 > 2·SE`. Un brazo que no aprende se reporta, pero **no entra**
en las comparaciones de abajo: compararse entre pisos no dice nada de resolución.

## La resolución del instrumento: qué diferencia cuenta

Dos medias se consideran **distintas** si `|Δ| > umbral`, con

    umbral = max(2·SE_dif, δ)      SE_dif = √(SE_a² + SE_b²)      δ = 0,01

`δ` es la diferencia por debajo de la cual **no importa aunque sea estadísticamente visible**:
vale 1,3–1,8 veces la desviación entre semillas que este mismo dato dio con otra red
(0,0057–0,0077; `banco-k`, 2026-09-08) y 1/55 del rango útil del dato (0,55). Se fija ahora
para que no se elija después con los números delante. Vale igual para `IoU_val`, `IoU_train`
y `brecha`.

## Las dos preguntas del encargo, y cómo se contestan

1. **¿Hay un `W` mínimo suficiente?** `W*` = el `W` de mayor `IoU_val` medio. Un `W` es
   *suficiente* si `IoU_val(W*) − IoU_val(W) ≤ umbral`. El **`W` mínimo suficiente** es el
   menor `W` suficiente (`W*` mismo si no hay otro). Se reporta con lo que ahorra respecto de
   `W = 128`: parámetros y ms por paso.
2. **¿La resolución estorba?** Se declara **«estorba»** si `IoU_val(128) < IoU_val(W*) − umbral`
   con `W* < 128`: una imagen más pequeña generaliza **mejor**, distinguiblemente. Y sobre la
   brecha: **«la brecha crece con la resolución»** si `brecha(128) − brecha(W mínimo
   suficiente) > umbral`. Se reportan las dos aunque sólo una sea cierta: son afirmaciones
   distintas.

## Cómo se separan las TRES causas de que un `W` dé menos `IoU_val` — escrito antes porque es donde este diseño puede engañar

Una caída de `IoU_val` respecto de `W*` puede venir de tres sitios, y los tres **parecen
iguales** mirando sólo `IoU_val`:

| causa | cómo se ve, por construcción | qué la separa |
|---|---|---|
| **(a) menos información**: el dato reducido ya no la tiene | **`IoU_train` también cae** (más del umbral) y la brecha **no crece**: la red no puede ni ajustar `train` | la descomposición `IoU_val = IoU_train − brecha` |
| **(b) peor generalización**: la información está, pero se memoriza | `IoU_train` se mantiene (dentro del umbral) y **la brecha crece** más del umbral | la misma descomposición |
| **(c) más parámetros**: en este diseño `W` alto es también `×55–×152` parámetros (205.348 a 128, 3.748 a 16, 1.348 a 8 con `C = 8`) | **indistinguible de (b) con los 25 brazos**: (b) y (c) suben la brecha juntas | sólo el control `w128-de16` |

**Regla de lectura, por brazo:** se clasifica cada `W ≠ W*` en (a), (b), **mixta** (las dos) o
**ninguna** (dentro del umbral), y se reporta la tabla entera. **Nunca** se resume en «a menos
resolución, mejor generalización»: con este dato se espera que la brecha **baje** al reducir
`W` simplemente porque hay menos que memorizar, **mientras** `IoU_train` también baja — y eso
es (a) con una brecha menor, no (b).

**El control, si se corre:** `w128-de16` tiene los parámetros de `w128` y la información de
`w016`.
- `IoU_val(w128-de16) ≈ IoU_val(w016)` (dentro del umbral) → los parámetros **no** explican la
  diferencia `w128 − w016`: es información.
- `IoU_val(w128-de16) ≈ IoU_val(w128)` → la explican los parámetros (y la información de 16 px
  bastaba).
- En medio → se reporta la fracción, sin redondear a ninguna de las dos.
- Y su brecha contra la de `w016` dice si **más parámetros a igual información** memorizan más.

⚠ **Sin el control, cualquier (b) se escribe como «(b) o (c): sin cerrar (confundido)»**, que
es la etiqueta que el propio `ESTADO.md` del repo central da a este mismo confound (capacidad
contra resolución) en dos estudios anteriores de `foveal-vision`. No se cierra lo que el
diseño no puede cerrar.

## Las tres formas posibles de la curva, escritas antes

| forma | qué se lee |
|---|---|
| **(i)** `IoU_val` sube con `W` y ningún `W < 128` es suficiente | con 100 imágenes, más píxeles siguen ayudando: la resolución **no compromete** la generalización en este rango |
| **(ii)** meseta: `W` mínimo suficiente `< 128`, y 128 no estorba | por encima del `W` suficiente la resolución **sobra**: coste sin ganancia |
| **(iii)** máximo interior: `W* < 128` y 128 estorba | el exceso de píxeles **perjudica**. La descomposición dice si es (b), y el control si es (b) o (c) |

Y **«no hubo señal»** —los cinco `W` dentro del umbral entre sí— **es un resultado**: en este
rango la resolución no cambia la generalización.

## Qué tarea es ésta, dicho antes de mirar

**Localizar un bloque de tinta**, no leer texto. El IoU de una caja aguanta el desenfoque: en
`banco-k`, filtrar con una gaussiana de σ = 1,5 px **subió** el IoU de `eval` (0,7981 → 0,8130)
y bajó la brecha (+0,0421 → +0,0298). Así que **se espera una meseta larga hacia `W` bajo**, y
que la caída llegue por **cuantización**, no por información fina: a `W = 8` el **40,5 %** de
las cajas mide menos de 2 px de alto y el **8,9 %** menos de 1 px; a `W = 16`, el 8,9 % mide
menos de 2 px *(medido el 2026-10-01 con las 1000 etiquetas)*. Una meseta larga **no** es un
fallo del experimento: es la respuesta a la pregunta **para esta tarea**. Lo que **no** dice
nada es sobre tareas que necesiten el trazo (leer, separar palabras).

## Lo que este criterio NO decide

- A qué se debe (c) **sin** el control: se deja escrito como «sin cerrar».
- Nada sobre la red de producción de `foveal-vision`, ni sobre kernels (**los pesos de aquí no
  se importan nunca a `banco-k`**), ni sobre otros datasets.
- Un «ganador» para `ESTADO.md`: lo decide el dueño con la curva delante.

## Riesgos, escritos antes

- **`W = 8`.** Las cajas miden 1,2–6,6 px de ancho y 0,4–6,9 px de alto *(medido)*. Está para
  ver **dónde se rompe**; una caída ahí es (a) por cuantización salvo que la descomposición
  diga otra cosa.
- **Un solo `lr`.** Si en el ensayo no hay un `lr` con el que `W = 32` **y** `W = 128` bajen sin
  oscilar, el plan **para** y vuelve al dueño. No se da un `lr` por `W`: sería otro confound.
- **Deriva entre máquinas.** Medida el 2026-10-01: 0,0012 de IoU entre el dev y una máquina de
  Vast para el mismo run (`banco-k`, fase 1). Por eso **todos los brazos de una misma semilla
  corren en la misma máquina**: toda comparación entre `W` es misma máquina.
- **1000 pasos pueden quedarse cortos** para los kernels grandes (más parámetros, mismo `lr`).
  Se mira la curva; si se alarga, se alarga **para todos los brazos** y se repite todo. Nunca uno.
- **El techo es `W = 128`**: el dato publicado ya es /4 del render (584 → 146). Las
  reducciones /1 y /2 **no se pueden estudiar** con este dato; pedirían publicar uno nuevo.
