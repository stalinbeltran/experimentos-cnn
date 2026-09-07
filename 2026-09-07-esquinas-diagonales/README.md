# `esq-2d` — con UN solo kernel, ¿qué tamaño detecta las DOS esquinas en diagonal?

**Corrido el 2026-09-07, 14:11:32 → 14:31 UTC · 5 brazos × 300 épocas · ~20 min de reloj ·
0 máquinas · 0 $** (CPU de este droplet de 2 vCPU). El criterio está congelado en
[`instrucciones/02-criterio.md`](instrucciones/02-criterio.md), escrito **antes** de la primera
época, y **no declara ganador**: se reportan todos los que pasan.

## Qué salió

En la época que guarda su `best.pt` (elegida por `val_loss`, la misma regla en los cinco):

| brazo | params | época | acierto ≤2 px `tl` | `br` | ≤1 px `tl` | `br` | error `tl` | `br` | f1 `tl` | `br` | antisim. | ¿pasa? |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| `k05` | 30 | 295 | 10,3 % | 6,4 % | 2,6 % | 0,0 % | 4,88 px | 6,13 px | 0,218 | 0,000 | 56 % | ❌ |
| `k07` | 54 | 268 | 56,4 % | 5,1 % | 34,6 % | 0,0 % | 2,45 px | 6,07 px | 0,340 | 0,069 | 52 % | ❌ |
| `k09` | 86 | 266 | 65,4 % | 7,7 % | 23,1 % | 2,6 % | 2,29 px | 5,17 px | 0,385 | 0,463 | 62 % | ❌ |
| **`k11`** | 126 | 235 | 62,8 % | **12,8 %** | 25,6 % | 2,6 % | 2,15 px | 4,72 px | 0,408 | 0,621 | 62 % | ✅ |
| **`k13`** | 174 | 109 | **89,7 %** | **23,1 %** | 24,4 % | 2,6 % | **1,73 px** | **4,00 px** | 0,470 | 0,609 | 47 % | ✅ |
| **suelo** (sin entrenar) | — | 0 | 5,1 % | 6,4 % | 1,3 % | 3,8 % | 5,96 px | 6,13 px | 0,334 | 0,334 | 38 % | — |

**Pasan el umbral (12 % a ≤2 px en LAS DOS esquinas) dos de los cinco: `k11` y `k13`.** Un solo
kernel **sí** puede con las dos esquinas en diagonal — pero necesita mucho más kernel que con una
sola, y ni de lejos lo hace igual de bien.

### Las cinco cosas que hay que leer antes que la tabla

1. ⚠⚠ **Compartir el kernel sale carísimo, y el número que lo dice es `k07`.** Ahí `esq-k`
   sacaba **100 %** a ≤2 px con **una** esquina; aquí, con la misma convolución y dos esquinas,
   saca **56,4 % en `tl` y 5,1 % en `br`**. No es un ajuste: es otra tarea.
2. **La asimetría predicha se cumplió, y es enorme.** `tl` va del 10 % al 90 %; `br`, del 6 % al
   23 %. **El kernel se queda con la esquina barata** — que es el desenlace 3 del criterio,
   registrado como el más probable, y por el motivo que estaba medido: `br` pide renunciar a la
   parte del filtro que apaga el interior del párrafo.
3. ⚠ **NADA saturó, al revés que en `esq-k`, y el mejor es el BORDE del rango.** `k13` gana en las
   dos esquinas y en las dos métricas, y la `val_loss` baja monótona con `k` (1,82 → 1,43 → 1,17
   → 1,02 → **0,89**). Es el **desenlace 5**: el eje **no está acotado por arriba** y la respuesta
   no es «13», es «mirar más allá».
4. **`existe` apenas despega.** Con suelo 0,334, el mejor `f1` es 0,470 (`tl`) y 0,621 (`br`), y
   `k05` da **0,000** en `br`: no detecta la esquina inferior-derecha en absoluto. La red aprende
   antes *dónde* que *si*.
5. ⚠ **Una semilla.** Entre brazos vecinos no se puede declarar nada: `k09` (65,4 %) y `k11`
   (62,8 %) en `tl` se dan la vuelta respecto de la tendencia, y con una sola semilla eso no se
   distingue de una fluctuación. Lo que sí es sólido es el extremo: `k05` no aprende y `k13` es el
   mejor de largo.

### Lo que se ve en las muestras

[`muestras/`](muestras/) tiene las 10 muestras congeladas por brazo, **antes**
(`-ep000-sin-entrenar`) y **después** (`-ep300`). En `k13-ep300.png`: **3/3 en `tl` y 1/3 en
`br`**, y el mapa enseña por qué — responde **positivo sobre la tinta** (al revés que el kernel de
`esq-k`, que la suprimía), con la esquina `tl` como máximo local; el mínimo, que es de donde se lee
`br`, cae en zonas mucho menos definidas. Y las probabilidades de `existe` son bajísimas incluso
cuando acierta la posición (0,57 · 0,14 · 0,15 en las tres `tl`), que es la misma historia del
punto 4 vista de cerca.

## Los cinco kernels sobre PÁGINAS ENTERAS

*Medido el 2026-09-07 con `python nn/transformacion.py --paginas 10` — 16 s, 0 $.*

El producto del experimento no es la red: es **el filtro**. `nn/transformacion.py` lo aplica
suelto a las **10 páginas enteras** de la partición `muestra` (200×200 px reducidos, nunca vistas),
una figura por kernel: arriba la página, abajo su mapa de respuesta en las mismas coordenadas, con
las esquinas verdaderas en verde, **el máximo en rojo** (de donde se lee `tl`) y **el mínimo en
naranja** (de donde se lee `br`).

| kernel | máximo a ≤2 px de `tl` | mínimo a ≤2 px de `br` | mediana `tl` | mediana `br` |
|---|--:|--:|--:|--:|
| `k05` | 5/10 | **0/10** | 5,3 px | 67,7 px |
| `k07` | 6/10 | **0/10** | 0,8 px | 55,8 px |
| `k09` | 4/10 | **0/10** | 27,4 px | 57,8 px |
| `k11` | 1/10 | **0/10** | 52,6 px | 31,7 px |
| **`k13`** | **8/10** | **0/10** | 1,2 px | 41,7 px |
| *(`esq-k`, una esquina)* | *10/10* | *—* | *0,5 px* | *—* |

**Cuatro cosas, y ninguna es buena para esta lectura:**

1. ⚠⚠ **`br` es 0/10 en los cinco kernels.** El mínimo del mapa **nunca** marca la esquina
   inferior-derecha sobre una página entera; la mediana va de 32 a 68 px. Es la confirmación más
   dura de lo que la ventana ya insinuaba (23 % como techo) y coincide exactamente con la sonda
   que se hizo **antes de diseñar el experimento** sobre el kernel de `esq-k`: 0/10, mediana 91 px.
2. ⚠ **`tl` tampoco generaliza bien**: el mejor es `k13` con 8/10, contra el **10/10 con mediana
   0,5 px** que daba `esq-k` con una sola esquina. Pedirle las dos degrada también la que sí
   aprendió.
3. ⚠⚠ **La ventana y la página NO se ordenan igual, y eso es un aviso de método.** `k11` es el
   segundo mejor en ventana y da **1/10** en página (mediana 52,6 px); `k07`, que no pasaba el
   umbral, da 6/10. Un brazo puede ser bueno en 32×32 y malo en 200×200 — **el barrido no mide la
   tarea de la página**, y si lo que se quiere es una transformación para páginas, hay que medirla
   ahí.
4. **El signo del kernel se dio la vuelta respecto de `esq-k`.** Allí la suma era **−73,68** (un
   supresor de tinta); aquí es **positiva en los cinco** (+19 a +36), y los mapas salen rojos sobre
   el párrafo. El azul —donde vive el mínimo— se concentra en los **bordes inferior y derecho**,
   así que el kernel sí codifica algo de esa zona, pero su mínimo no pica en el punto.

## Lo que quedó pendiente

- **El eje no está acotado por arriba, y esta vez con más razón que en `esq-k`**: `k13` es el
  borde y gana en todo. El dataset admite hasta **k = 17** sin regenerarlo (lo calcula el
  manifiesto), así que `k15` y `k17` cuestan ~5 min y están a un `BRAZOS` de distancia.
- **`br` sigue siendo malo en términos absolutos** (23 % a ≤2 px). Si lo que hace falta es
  detectar las dos esquinas de verdad, esta lectura no basta y la continuación declarada es
  **`ant`** — el mismo montaje con el kernel forzado antisimétrico—, que está implementada y
  comprobada, sin armar. Ver
  [`instrucciones/03-alternativas-anotadas.md`](instrucciones/03-alternativas-anotadas.md).
- ⚠ **La transformación sobre página entera no funciona para `br`, y eso es lo más importante
  que deja este experimento.** Si el objetivo es un filtro que señale las dos esquinas de un
  párrafo en una página, esta lectura (máximo/mínimo de un mapa) **no vale**, y subir `k` no lo
  arregla: 0/10 en los cinco.
- **`existe` no se ha estudiado.** Su `f1` se queda cerca del suelo en los cinco brazos y nadie ha
  mirado si es la cabeza (2 parámetros por esquina) o el kernel.
- **Una semilla**, como en `esq-k`. Nadie ha medido la variación entre semillas.

## Cómo está montado (lo de abajo se escribió antes de correr)

Hereda de `esq-k` **todo lo que decide la comparabilidad**: misma receta, semilla 1, ventana
32×32, reducción por 4, sin padding, sin bias, cabeza C1 (esperanza bajo `softmax(β·M)`, β
aprendida desde 3,5), Adam `lr` 0,05, lote 128, 300 épocas.

**El diseño tiene UN eje: el tamaño del kernel.** Cinco brazos, una sola estructura, y lo único
que cambia entre ellos es `k`. Las otras lecturas que se propusieron quedan **anotadas y sin
correr**, por orden del dueño del 2026-09-07 — en
[`instrucciones/03-alternativas-anotadas.md`](instrucciones/03-alternativas-anotadas.md), con lo
que cada una contestaría y lo que cuesta.

## El problema, que no es el de ayer con otra etiqueta

Un filtro lineal que pica en una esquina superior-izquierda tiene forma de **cuadrante** y esa
forma es **orientada**: la esquina inferior-derecha es **la misma forma girada 180°**. Partiendo
el kernel bajo ese giro (`W = S + A`, con `S` simétrica y `A` antisimétrica), para el mismo
parche:

```
respuesta_tl = <S,P> + <A,P>          respuesta_br = <S,P> − <A,P>
```

Las dos respuestas son **simétricas respecto de `<S,P>`**, así que **toda la diferencia entre
esquinas vive en `A`**. Las estructuras salen de ahí, no de una lluvia de ideas.

⚠ **Y hay un dato en contra de la estructura obvia, medido antes de diseñar nada.** El kernel
ganador de `esq-k` —que sólo vio esquinas tl— tiene el **74,0 % de su energía en `S`** y suma
**−73,68**: es sobre todo un **supresor de tinta**. Medido sobre las 10 páginas enteras de aquel
experimento, **su mínimo cae en la mancha de tinta, no en la esquina br: 0/10, mediana 91 px**.
El comando está en [`instrucciones/01-encargo.md`](instrucciones/01-encargo.md).

## La estructura: la de `esq-k`, con dos esquinas en vez de una

**Orden del dueño (2026-09-07): «Debe ser idéntica al exper anterior, sólo que en vez de 1 esquina
van a ser 2».** Y lo es: una sola convolución `k×k`, sin bias, sin padding, sin ReLU, y la misma
lectura C1 por esperanza bajo `softmax`. **No se gira nada.**

```
entrada x (1, 32, 32) en TINTA (/255)
    │
    └── conv(x, W)  ──►  M  (m×m, m = 32 − k + 1)   ── UN solo mapa
                          │
                          ├── softmax(+β·M) ──► (x, y) del MÁXIMO  ──► esquina superior-izquierda
                          │   existe_tl = a₁ · logsumexp(+β·M)/β + b₁
                          │
                          └── softmax(−β·M) ──► (x, y) del MÍNIMO  ──► esquina inferior-derecha
                              existe_br = a₂ · logsumexp(−β·M)/β + b₂

β se aprende (arranca en 3,5) y es UNA sola, compartida por las dos lecturas.
```

| brazo | `k` | mapa | kernel | cabeza | **total** | (`esq-k` a ese `k`) |
|---|--:|---|--:|--:|--:|--:|
| `k05` | 5 | 28×28 | 25 | 5 | **30** | 28 |
| `k07` | 7 | 26×26 | 49 | 5 | **54** | 52 |
| `k09` | 9 | 24×24 | 81 | 5 | **86** | 84 |
| `k11` | 11 | 22×22 | 121 | 5 | **126** | 124 |
| `k13` | 13 | 20×20 | 169 | 5 | **174** | — |

Los brazos se llaman como allí (`k05`, `k07`…) porque lo único que varía es `k`, y así las dos
tablas de resultados se leen una al lado de la otra sin traducir.

```bash
python nn/modelo.py     # imprime esta tabla, las alternativas anotadas, y comprueba las dos
```

### Los dos únicos sitios donde «idéntica» no puede ser literal

Son consecuencia de tener dos esquinas, no decisiones de estilo, y conviene tenerlos delante:

1. **La cabeza pasa de 3 a 5 parámetros.** La `β` sigue siendo **una**, y lo único que se duplica
   es el `existe` (`a`, `b`), porque ahora hay **dos** cosas que detectar. Todo lo demás —la
   convolución y la lectura de la posición— es idéntico.
2. **La segunda esquina se lee del MÍNIMO del mapa, y no hay otro sitio de donde sacarla.** Con
   una sola convolución y una cabeza mínima, las únicas dos lecturas distintas de un mapa son su
   **máximo** y su **mínimo**: cualquier otra pediría pesos **por posición** (m² de ellos), y eso
   rompe la restricción que tú mismo pusiste en `esq-k` — *«si la cabeza es grande, el kernel no
   aprende nada»*. No es una elección entre varias: es la única que cabe.

Hay un test que lo fija: sobre un mapa con un pico negativo y otro positivo en sitios distintos,
`tl` cae en el positivo y `br` en el negativo, cada uno en su coordenada (`assert` en
`nn/modelo.py`). Sin eso, «`br` = mínimo» sería una intención escrita en un comentario.

### ⚠⚠ Y una medida que juega EN CONTRA de esta estructura, dicha antes de correrla

El kernel ganador de `esq-k` —el único de esta familia que se ha entrenado— tiene el **74,0 % de
su energía en la parte simétrica** bajo giro de 180° y suma **−73,68**: es sobre todo un
**supresor de tinta**. Y medido sobre las 10 páginas enteras de aquel experimento, **su mínimo cae
en la mancha de tinta, no en la esquina inferior-derecha: 0/10, mediana 91 px.**

- **No significa que esto no pueda funcionar**: aquel kernel se entrenó **sólo para `tl`**, sin
  ninguna presión para poner nada en el mínimo.
- **Sí significa que el gradiente, cuando se le deja elegir, gasta el kernel en apagar el interior
  del párrafo** — y para que `br` sea el mínimo hay que renunciar a buena parte de eso.

**Ésa es exactamente la tensión que este barrido mide**, y por eso el criterio dice que el
desenlace más probable es *acierta en `tl` y falla en `br`*, y que el titular sea **la peor de las
dos esquinas, nunca el promedio**. La fracción simétrica del kernel se registra **en cada época**,
así que la explicación se podrá contrastar contra ese 74,0 % de partida en vez de conjeturarla.

### Lo que se propuso y NO se corre

- ❌ **`rot`** (aplicar el mismo kernel a la entrada girada 180°): **descartada por el dueño** —
  *«No queremos girar el kernel»*. Su código está **borrado**, a propósito: una alternativa
  descartada que se queda en el repo se acaba armando por error.
- **`ant`** (el mismo `sig` con el kernel forzado antisimétrico, 29 parámetros a k=7) y el control
  **`ind-tl`/`ind-br`** (dos redes sin compartir nada): siguen **implementadas y comprobadas** en
  `nn/modelo.py`, sin armar. Armarlas es añadir su línea a `BRAZOS`.

El registro completo —qué contestaría cada una, qué cuesta, y por qué `ant` es la continuación
natural si el barrido falla— en
[`instrucciones/03-alternativas-anotadas.md`](instrucciones/03-alternativas-anotadas.md).

### El eje: `k` ∈ {5, 7, 9, 11, 13}

⚠ **El 3×3 se cae con un dato, no por gusto** (y es el propio dueño quien lo saca): en `esq-k` se
quedó en el 8,3 % de acierto, que es **exactamente su suelo sin entrenar**. No aprendió poco: no
aprendió. Aquí la tarea es más difícil, así que repetirlo sería pagar por re-confirmar al
perdedor. **El 13 entra en su lugar** porque aquel eje no estaba acotado por arriba: el 11 seguía
mejorando y era el borde del rango.

⚠ **Y 13 no es un tope caprichoso: es lo que este dataset admite.** La esquina se sortea entre los
píxeles 8 y 23 de la ventana, y el mapa de un kernel `k` sólo representa de `(k−1)/2` a
`31−(k−1)/2`. Con **k = 17** los dos rangos coinciden exactamente; con k = 19 habría esquinas que
el mapa **no puede** señalar. Cabe 15; a partir de 19 hay que regenerar el dato. Lo calcula y lo
escribe el manifiesto, no se supone.

### Lo que se propuso y NO se corre

`sig` (un mapa: `tl` = máximo, `br` = mínimo), `ant` (lo mismo con el kernel forzado
antisimétrico) y el control `ind-tl`/`ind-br` (dos redes sin compartir nada). **Siguen
implementadas y comprobadas** en `nn/modelo.py` —una estructura implementada es la forma menos
ambigua de anotarla, y así no se pudre en silencio—, pero no están armadas: ponerlas en marcha es
añadir su línea a `BRAZOS`.

⚠ **Lo que se pierde, dicho por delante:** si el barrido sale bien, nada. Si sale mal, este diseño
**no puede distinguir** *«un kernel no da para las dos esquinas»* de *«esta lectura no es la
buena»*. Está escrito en el criterio como desenlace 2.

## El dataset vive en el repo de DATOS, y es siempre el mismo fichero

**Por orden del dueño (2026-09-07): «Guarda los datasets en el repo de data, de modo que sean
siempre los mismos, por consistencia».**

```
foveal-vision-data/experimentos-cnn/esquinas300-32px-r4-r20260907/
    train.npz · val.npz · muestra.npz · muestras-congeladas.npz · manifiesto.json · README.md
```

Se resuelve con `expcnn.exigir_dataset(...)`, que es **la única** puerta: si no está publicado,
el entrenamiento **se niega antes de empezar** en vez de generarse uno equivalente. Ahí está el
punto — un dato re-derivado es el mismo *mientras nada cambie*, y «nada cambia» no es comprobable
hacia el futuro; publicado, es el mismo **porque es el mismo fichero**.

⚠ **No va en este repo y no es una preferencia:** éste es **público** y el de datos es
**privado**. Y no va en `window-datasets/`, que es de `foveal-vision` y lo resuelve su propio
`settings`: meter ahí un dataset de otra forma sería una colisión silenciosa.

⚠ **Y de paso tapa un agujero real.** Las 10 muestras congeladas de `esq-k` se declaraban
commiteadas y **no lo estaban**: el `*.npz` del `.gitignore` de este repo —que existe para que no
se cuele el dato de entrada— se llevaba también la verificación. Medido el 2026-09-07 en el clon
limpio de esta máquina: aquel `nn/muestras.npz` no existe, así que su figura de verificación no
se podía regenerar sin volver a rendir 300 imágenes. Publicadas con el dataset, dejan de perderse.

### Qué trae

Mismas imágenes que `esq-k` (misma receta, misma semilla): 267 válidas de 300. Diez ventanas por
imagen: **2 `tl` + 2 `br`** positivas y **6 negativas** (las esquinas de la otra diagonal `tr` y
`bl`, que son el negativo duro; borde superior, borde inferior, interior y fondo).

⚠ **Se etiquetan LAS CUATRO esquinas**, no sólo la diagonal que mide `esq-2d`. No cuesta nada —
las cuatro coordenadas ya se conocen al recortar— y es lo que hace que *«si pueden usar el mismo
dataset no hay problema»* sea cierto también para el experimento siguiente: el que mire la otra
diagonal, o las cuatro, no tiene que re-rendir nada. `esq-2d` lee `tl` y `br` e ignora el resto.

⚠ **Ninguna ventana contiene más de una esquina** (párrafo ≥ 64 px, ventana 32). Se **cuenta** al
generar y se escribe en el manifiesto, no se supone.

```bash
python nn/datos.py --imagenes 300 --publicar   # genera y publica (no pisa lo publicado)
python nn/datos.py --comprobar                 # ¿el publicado es el del manifiesto? (instantáneo)
python nn/datos.py --rederivar                 # ¿la receta y la semilla lo vuelven a dar? (~6 min)
```

**`--publicar` se niega a pisar un dataset ya publicado**: dato nuevo = nombre nuevo, que es la
regla que el repo de datos ya tiene escrita. Si publicar pudiera sobrescribir, un `--publicar`
distraído cambiaría el dato bajo los pies de todo lo ya medido, sin un solo error.

## El criterio, congelado antes de mirar

En [`instrucciones/02-criterio.md`](instrucciones/02-criterio.md), con los suelos medidos sobre
las redes **sin entrenar**: **3,8–5,1 % (`tl`) y 6,4 % (`br`)** a ≤2 px, que son exactamente el
predictor constante en el centro. Umbral de «ha aprendido algo»: **12 % en las dos esquinas**. El
titular es la **peor** de las dos, nunca el promedio.

Y sin los brazos de control, el techo se lee de `esq-k`, que midió la tarea de **una** esquina con
esta misma convolución: 92,3 % (k=5) · 100 % (7 · 9 · 11). ⚠ Es una **referencia, no un control**,
por dos motivos y no uno: sus **ventanas** son otras (allí 4 `tl` por imagen), y la **red no es
idéntica** —2 parámetros más de cabeza y una lectura del mínimo que allí no existía—. Lo que se
compara es *la misma convolución con el doble de trabajo*, no la misma red.

## Cómo se lanzó, y cómo se le pregunta cómo va

**El lanzador está commiteado**, por orden del dueño (2026-09-07): *«Guarda el script q empleaste
para el vigilante… Así podemos saber cuál se usó, y si funcionó como se espera»*.

```bash
nn/lanzar_barrido.sh            # los 5 brazos x 300 épocas + las figuras ep300 + el aviso
nn/lanzar_barrido.sh --estado   # ¿está viva? ¿por dónde va? ¿falló? ¿se relanzó?
```

Corre como **unidad de systemd** (`esq2d-barrido`, vía `desacoplar-persistente.sh`), o sea con
padre PID 1: sobrevive al fin del turno que la lanzó y al reinicio del coordinador. Y **se niega a
lanzarse dos veces**: dos procesos escribiendo los mismos pesos y el mismo `metrics.jsonl` los
corrompen.

### ✅ Lo que se midió con este mismo lanzamiento (2026-09-07)

La unidad arrancó a las **14:11:32 UTC**. A mitad del barrido, el proceso de Claude Code que la
lanzó terminó:

| | qué pasó |
|---|---|
| la **unidad** | siguió `active`, `Result=success`, **`NRestarts=0`**, y `k09` terminó sus 300 épocas **después** de que muriera la sesión ✅ |
| los **vigilantes** del harness (`run_in_background`) | quedaron marcados **`stopped` sin registro de finalización** ❌ |

Son las dos mitades de la regla del proyecto medidas **en el mismo suceso**: el trabajo sobrevive
porque su padre es PID 1; el vigilante no, porque el suyo es la sesión. Por eso el estado se lee
del **disco** (`--estado` mira `metrics.jsonl`, que se escribe y se cierra en cada época) y no del
log, que además puede verse vacío estando todo bien: `python` bufferiza cuando no es un tty.

⚠ **`NRestarts` se mira siempre.** Una unidad que falló y se relanzó sola *parece* «corriendo» y
está repitiendo trabajo desde cero — pasó el 2026-09-02 y el 2026-09-04, con 62 relanzamientos.

El detalle del mecanismo y la regla general viven donde se dispara, en
[`telegram-coordinator/CLAUDE.md`](https://github.com/stalinbeltran/telegram-coordinator/blob/main/CLAUDE.md).

## Cómo se corre a mano (los pasos sueltos)

```bash
cd ~/src/experimentos-cnn
E=2026-09-07-esquinas-diagonales
.venv/bin/python $E/nn/datos.py --imagenes 300        # ~6 min (generador + Chromium)
.venv/bin/python $E/nn/entrenar_local.py --suelos     # los suelos, sin entrenar nada
for b in k05 k07 k09 k11 k13; do   # o, mejor: nn/lanzar_barrido.sh
  .venv/bin/python $E/nn/entrenar_local.py --brazo $b --epocas 300   # reanudable
done
.venv/bin/python $E/nn/muestras.py --etiqueta ep300
```

Las dependencias del venv están en el README de `esq-k` § «Cómo se repite» (son las mismas, y
`uv pip install -e .` **no** las trae todas).

## Cuánto cuesta (estimado, no medido)

**0 máquinas y 0 $**: entrena en este droplet, como `esq-k`. Lo que cuesta es reloj —
*medido el 2026-09-07 a máquina libre*: de **0,31 s/época** (k=5) a **0,53** (k=11 y 13), así que
los **cinco brazos × 300 épocas ≈ 11 min** *(estimado a partir de cuatro medidas, no medido
entero)*. Es del orden de los 9 min que costó `esq-k` entero.

⚠ Y una lección de medición, porque el primer número que di estaba mal: la misma prueba **con la
máquina rindiendo el dataset a la vez** daba 1,2–2,7 s/época, o sea **~3× más**. En un droplet de
2 vCPU, un tiempo por época medido con algo más corriendo no es el tiempo por época.

Y en disco: 5 brazos × (2 checkpoints de ~17 KB + un `metrics.jsonl` de ~50 KB) ≈ **0,4 MB**,
holgado dentro del tope de ~5 MB por experimento que declara el `CLAUDE.md` del repo. El registro
se guarda **redondeado a 6 cifras** de todas formas: era necesario con 25 brazos y no estorba con
5.

⚠ El freno lo ve: `entrenar_local.py` está en la lista `TRABAJOS` de
`telegram-coordinator/scripts/cerrable.mjs`, así que un entrenamiento vivo aparece en el veredicto
que se lee desde el móvil. No hace falta ejecutor nuevo de Telegram: `gasta` es `entrena-local`,
no `alquila`.

## Lo que este experimento NO contesta

- **Nada sobre señalar las dos esquinas a la vez en la misma vista.** La ventana es 32×32 y el
  párrafo ≥ 64 px, así que ninguna ventana contiene las dos. La pregunta de aquí es *«¿puede un
  kernel servir a las dos?»*; la de la página entera es el experimento siguiente.
- **Nada sobre las otras dos esquinas** (`tr`, `bl`): aquí son negativos.
- **Una semilla.** Igual que en `esq-k`, y con el mismo precio: las diferencias pequeñas entre
  brazos no se van a poder declarar.
