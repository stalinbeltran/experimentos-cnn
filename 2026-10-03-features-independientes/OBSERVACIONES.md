# `feat-ind` — Observaciones: una CNN pequeña por feature, entrenadas aparte, y qué dan juntas

**2026-10-03 · SÓLO DOCUMENTO.** Nada de esto está implementado ni medido. Lo que aquí se afirma
sale de tres sitios, y cada afirmación dice de cuál: **(a)** lo ya medido en este repo y en
`foveal-vision`, con fecha; **(b)** lo que se deriva de una fórmula o de la definición, sin medir;
**(c)** lo que se recuerda de la literatura, marcado «de memoria, no comprobado desde aquí». La
sección final dice qué esperaríamos ver, escrita hoy, para que cuando se corra algo «no hubo
señal» sea un resultado y no una decepción.

## 0. La propuesta, dicha con precisión

Hoy la forma normal de reconocer un dígito es **una** CNN con 10 salidas y entropía cruzada: la
pérdida sólo premia lo que **separa** una clase de otra. Lo que el dueño propone es otra cosa:

1. Fijar **de antemano** un vocabulario de `N` features (círculo, recta vertical, arco, cruce…).
2. Entrenar **`N` CNN pequeñas, independientes**, cada una con una sola pregunta: *«¿está la
   feature `i` (y dónde)?»*. Ninguna ve la etiqueta de clase ni la salida de las otras.
3. Un **compositor** que lee las `N` salidas y reconoce el objeto complejo.

La frase que define todo es *«se especializan en reconocer, no en discriminar»*. Un detector de
círculo aprende qué aspecto tiene un círculo **en cualquier dígito**; una capa de un clasificador
aprende lo que le sirve para distinguir el 9 del 4, que puede o no ser un círculo. Son objetivos
distintos y producen features distintas.

## 1. Lo que ya existe aquí y se parece (por `id`, nunca por carpeta)

| | qué es | qué enseña para esto |
|---|---|---|
| `esq-k` (cerrado), `esq-2d` (cerrado), `bor-k` (abierto) | **un kernel** + cabeza mínima que detecta **una** feature (una esquina, los bordes) | que un detector de una sola feature **se puede entrenar solo** y aprende; son exactamente el paso 2 de la propuesta, con `N` = 1 |
| `banco-k` (abierto) | un kernel **fijado antes** de la red, aplicado a la entrada | un «feature» de la capa 0 decidido por una persona y no por la pérdida; mide si eso facilita o transfiere. Hasta hoy ningún kernel declara (medido, `banco-k/README.md`) |
| la red foveada de `foveal-vision` | **dos ramas** (centro y periferia) con sus convoluciones separadas, unidas por `concat` en una `Linear` de 12.800 → 12 | es la versión *acoplada* de la idea: las ramas comparten la pérdida y **el gradiente de la cabeza altera a las dos**. Y esa cabeza es el 91,1 % de los 168.652 parámetros *(medido el 2026-09-01, `plan-strides-rama-2026-09-01.md`)* |
| `ruido-nist` (cerrado) | rasteriza **trazos sintéticos** (rectas, oblicuas, arcos de Bézier, recortes) a 32×32 y los reduce a 8×8 contando bloques | ya hay un generador de «features» dibujadas con control total de posición, grosor y ángulo. Se copiaría, no se importaría (Regla 0) |

Lo nuevo de `feat-ind` no es el detector unitario (ya está) sino **la composición de `N`** y la
comparación neta contra el clasificador monolítico.

## 2. Observaciones sobre la premisa

### 2.1 Reconocer necesita etiquetas de feature, y eso es un coste que discriminar no tiene

Un clasificador se entrena con la etiqueta de clase, que viene con el dataset. Un detector de
círculo necesita saber **dónde hay círculos**, y eso **no viene con NIST**. Hay tres fuentes, y la
elección condiciona todo el experimento:

| fuente de la etiqueta | coste | qué se pierde |
|---|---|---|
| **síntesis**: dibujar la feature (como hace `ruido-nist` con sus trazos) y entrenar sobre lo dibujado | 0 $, ilimitada, posición exacta conocida | el círculo dibujado no es el círculo manuscrito: hay **transferencia** sintético → real que medir, no suponer |
| **anotación a mano** sobre dígitos reales | cara, ambigua (¿el lazo abierto de un 9 es círculo?) | no escala; sirve para un conjunto pequeño de **evaluación** |
| **etiqueta débil por clase**: «0, 6, 8, 9 contienen círculo» | gratis | **reintroduce la discriminación** por la puerta de atrás: el detector aprende lo que separa esas cuatro clases de las otras seis, no un círculo |

Observación: **la síntesis es la única fuente compatible con la premisa** (reconocer sin ver la
clase), y la anotación a mano sirve para medir si lo sintético transfiere. La tercera parece la
cómoda y es la que invalida la idea.

### 2.2 El vocabulario de features es una HIPÓTESIS sobre el dato, y se mide antes de entrenar

Que un 9 «sea» círculo + recta es verdad para muchos escritores y falso para otros (lazo abierto,
cola curva). Antes de entrenar nada conviene medir la **cobertura** del vocabulario: qué
fracción de cada clase se puede describir con él. Se hace con un **oráculo** (anotar a mano, o
sintetizar configuraciones) y no cuesta entrenamiento. Si la cobertura de una clase es baja, el
compositor **no puede** reconocerla por mucho que acierten los detectores, y conviene saberlo
antes de atribuir el fallo a la red.

Vocabulario inicial plausible para dígitos, como hipótesis y no como decisión: lazo cerrado, recta
vertical, recta horizontal, arco abierto (hacia la izquierda / la derecha), cruce de trazos,
extremo libre. Es el vocabulario del reconocimiento **estructural** clásico de caracteres (trazos,
lazos, extremos), anterior a las CNN *(de memoria, no comprobado desde aquí)*.

### 2.3 El ahorro de cómputo es real para las convoluciones, y se puede derivar sin medir

El coste de una capa convolucional es proporcional a `C_in · C_out · k² · H · W` *(fórmula, no
medida)*. Comparar a canales totales iguales:

| | canales por capa | coste por capa (unidades de `k²HW`) |
|---|---|---|
| **una** red ancha | `N·C` | `(N·C)² = N²·C²` |
| **`N`** redes de `C` canales | `C` cada una | `N · C² ` |

El factor es **`N`**: `N` redes estrechas son la red ancha con sus mezclas entre grupos puestas a
cero (es una convolución **por grupos** con `groups = N` y sin mezcla). Lo que se ahorra es
exactamente lo que se renuncia a aprender: **las features cruzadas**. El punto 2.4 es el precio de
eso.

Y la paralelización es trivial: `N` procesos sin gradiente compartido. En el dev (2 vCPU, 3,8 GB,
*medido 2026-09-06*) la red de `ruido-nist` —3 capas 3×3, 8 canales, **1.338 parámetros**— entrena
una corrida en **~25–45 s** *(medido el 2026-10-03)*. Seis detectores de ese tamaño son ≈8.000
parámetros en total y caben en el dev sin alquilar nada *(estimado a partir de lo anterior)*.

### 2.4 La independencia es también el riesgo: ningún detector tiene contexto

Un detector de «recta vertical» no puede usar «hay un círculo encima» para decidir. Debe ser
robusto **solo** a la deformación, al grosor y a la oclusión por otros trazos. En un clasificador
monolítico, las capas altas sí combinan contexto, y eso es parte de por qué funciona. Por tanto:

- es **esperable** que cada detector, solo, acierte menos sobre dígitos reales que la feature
  equivalente implícita en una red monolítica;
- y la pregunta neta no es «¿aciertan los detectores?» sino **«¿lo que pierden por no tener
  contexto lo recupera el compositor?»**.

### 2.5 La salida de un detector debería ser un MAPA, no un escalar

Si el detector termina en un promedio global y un escalar («hay círculo: 0,93»), tira **dónde**
está, y entonces el 6 y el 9 —mismas features, distinta disposición— son indistinguibles para el
compositor. Un detector **totalmente convolucional**, sin pooling global, da un mapa de calor de
la feature. Esto conecta directamente con `feat-pos`, que es dónde se discute cómo usar esas
posiciones. Aquí la observación es: **la decisión mapa-o-escalar se toma al diseñar el detector**,
y tomarla mal deja a `feat-pos` sin materia prima.

⚠ A 8×8 (el dataset publicado `uci-optdigits-8px-r20261002`) un mapa tiene 64 celdas y el lazo
de un 9 ocupa del orden de 3–4 px *(estimación por la geometría del dígito, NO medida)*. Es muy
poca resolución para posiciones. Los bitmaps de NIST son 32×32 y **no vienen con scikit-learn**
(`README.md` del dataset). Qué resolución usar es una **decisión del dueño**, no de este documento.

## 3. Qué es «el resultado neto» y cómo se mediría sin engañarse

Tres brazos sobre el **mismo** dato y el mismo presupuesto de entrenamiento:

| brazo | qué es | qué contesta |
|---|---|---|
| **A** monolítico | una CNN, 10 salidas | la base. Ya medida en esta familia: 0,851 ± 0,025 (`dim-nist` `w8`, 1.258 parámetros) y 0,870 ± 0,032 limpio / 0,912 con ruido en línea (`ruido-nist`), con 180 train / 1617 val *(medido 2026-10-02/03)* |
| **B** detectores congelados + compositor | los `N` detectores entrenados aparte (síntesis), congelados; un compositor pequeño entrenado con la clase | **la propuesta tal cual**. Lo que cuesta la independencia |
| **C** detectores + compositor, afinados juntos | igual que B pero se deja fluir el gradiente a los detectores | cuánto de la diferencia A−B es por **no acoplar**, y cuánto por el **vocabulario** |
| **D** oráculo | las features verdaderas (anotadas o sintéticas) directamente al compositor | el **techo** del compositor con detectores perfectos. Si D no gana a A, la idea no es viable con ese vocabulario, por buenos que sean los detectores |

Y lo que **sólo** esta arquitectura permite medir, y que es su argumento más fuerte: la
**atribución del error**. Cuando B falla en una imagen, o un detector no vio una feature que
estaba (fallo de **reconocimiento**), o todos acertaron y el compositor compuso mal (fallo de
**composición**). Un clasificador monolítico no ofrece esa descomposición. Un criterio que no
mida esto deja fuera la mitad del valor.

Métricas naturales: exactitud sobre val (comparable con A); por detector, precisión/recall contra
el oráculo; y la tabla de atribución reconocimiento/composición sobre los fallos de B.

## 4. La hipótesis que más vale probar: pocos datos

Toda la línea de `dim-nist` y `ruido-nist` es **generalizar desde el 10 %** (180 imágenes). Un
compositor sobre `N` mapas tiene **muy pocos parámetros** comparado con una CNN completa, y los
detectores se entrenan con síntesis **ilimitada**. Es plausible que:

- a datos abundantes, **A ≥ B** (el acoplamiento aprende lo que el vocabulario no previó);
- a datos muy escasos, **B ≥ A**, porque lo que se aprende de las 180 imágenes es sólo la
  composición.

Es la curva **exactitud × tamaño de train** la que distingue las dos cosas, no un único punto.

## 5. Dónde encaja en la literatura, para no reinventarlo en silencio

*(todo de memoria, no comprobado desde aquí; es para saber qué buscar, no para citar)*

- **Reconocimiento estructural / sintáctico de caracteres** (años 70–80): primitivas de trazo y
  reglas de composición. Es la propuesta sin aprendizaje en los detectores.
- **Modelos por partes deformables** (Felzenszwalb y otros, ~2008–2010): detectores de partes +
  un modelo de la disposición relativa. Es la propuesta con detectores lineales.
- **Cápsulas** (Sabour, Frosst, Hinton, 2017): partes con vector de pose y acuerdo parte–todo.
  Es la propuesta con detectores acoplados por el enrutado.
- **Aprendizaje multitarea con tronco compartido**: lo contrario de la premisa (comparte
  cómputo y acopla gradientes). Sirve como control de «¿importa no acoplar?».

## 6. Qué se haría primero, y qué NO

Primero, y todo a 0 $ en el dev:

1. Fijar el vocabulario y **medir su cobertura** con un oráculo (§2.2).
2. Entrenar `N` detectores sobre features **sintéticas** (copiando la idea del rasterizador de
   `ruido-nist`, con posición conocida), salida en **mapa** (§2.5).
3. Medir su transferencia a dígitos reales contra un conjunto pequeño anotado a mano.
4. Los brazos A, B, D de §3 (C después, si B y A se distinguen).

Lo que no se haría: etiquetas débiles por clase (§2.1, invalida la premisa); una sola corrida sin
oráculo (no se podría separar «el vocabulario no cubre» de «el detector no acierta»); decidir la
resolución aquí (§2.5).

Cuando haya un plan de corrida, nace `instrucciones/02-criterio.md` **antes de mirar** y se
declaran `gasta`, `entrada` y `dataset` en `experimento.json`; `comprobar.py` exige entonces que
el script que entrene se llame `entrenar_local.py` (contrato con el freno).

## 7. Qué esperaríamos ver — escrito el 2026-10-03, antes de que exista nada

- Los detectores **aprenden su feature sintética** con facilidad (precedente: `esq-k`, `esq-2d`).
- La transferencia a trazos manuscritos **degrada**, sobre todo en lazos abiertos y cruces.
- A 180 imágenes de train, **B queda por debajo de A** en exactitud, y D (oráculo) por encima de
  A: o sea que el vocabulario sí basta y lo que falta es el reconocimiento.
- La ganancia **medible** de B no es la exactitud sino la **atribución del error**.
- El régimen de muy pocos datos (§4) es lo único donde B podría ganar, y **no sabemos** cómo sale.

Si sale al revés en cualquiera de los cinco puntos, eso es un resultado y se escribe tal cual.
