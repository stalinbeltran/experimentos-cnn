# 01 — Encargo: montar el banco de evaluación de kernels

**Fecha:** 2026-09-08 · **Estado:** carpeta montada, **nada corrido**, tres decisiones abiertas.

## Qué pidió el dueño

> «Crea una carpeta para el experimento especificado en
> https://claude.ai/public/artifacts/fe5ff2bb-2393-484a-87d0-cfacf2bfdf60, y dime si necesitas
> algo más. Documenta todo»

O sea: **montar la carpeta y decir qué falta**, no correr el banco. Esto último es lo que este
documento contesta, en su § «Lo que hace falta».

## De dónde salió la especificación, y por qué está copiada aquí

El artifact es la **especificación técnica v1.0** del banco, fechada **2026-09-07** y titulada
`ESPECIFICACION_Banco_Evaluacion_Kernels.md`. Está copiada **verbatim** a
[`ESPECIFICACION.md`](../ESPECIFICACION.md) (21.324 caracteres, 469 líneas), sin editar una
palabra.

**Se copia y no se enlaza porque una URL no es un artefacto.** Estos servidores se rehacen sin
aviso y de ellos sólo sobrevive lo que está empujado; un banco cuyas condiciones viven en una
página web es un banco que en el próximo server no se puede reproducir. Lo que sí se conserva es
la URL, en `experimento.json` → `especificacion.url`, para poder cotejar si algún día sale una v2.

⚠ **Cómo se rescató, que costó y conviene saberlo:** la página es una aplicación cliente y
`WebFetch` sólo ve el armazón vacío; la API (`/api/published_artifacts/<uuid>`) está detrás de
un desafío JavaScript de Cloudflare que devuelve **403** a `curl`. Se leyó con el Chromium de
Playwright del venv de `image-text-sample-generator`, interceptando la respuesta de esa API.
*Medido el 2026-09-08.*

## Qué se hizo

| | |
|---|---|
| la carpeta | `2026-09-08-banco-kernels/`, con la forma `<fecha>-<nombre>` que pide el repo |
| las dos obligaciones | `experimento.json` (`id: banco-k`) y `REGLAS.md` con sus cinco secciones |
| la especificación | copiada verbatim, es **la fuente de verdad** |
| la arquitectura del §7 | `nn/modelo.py`, **autónoma** (sólo `torch`), con sus invariantes comprobables |
| la entrada declarada | `nn/entrenar_local.py`, que hoy **se niega** y dice qué falta |
| la receta de render | `nn/receta.json`, 584 × 584 con el área `[68, 512]` del §3.3 |
| el criterio antes de mirar | [`02-criterio.md`](02-criterio.md) |

**Lo que NO se hizo, a propósito: generar y publicar el dataset.** Un dataset publicado **no se
reescribe nunca** (dato nuevo = nombre nuevo), así que publicarlo ahora congelaría las tres
decisiones de abajo antes de que el dueño las haya tomado. Es la operación irreversible de este
encargo y va después de ellas, no antes.

## Lo que se comprobó ejecutándolo (no leyendo)

*Todo medido el 2026-09-08 en esta máquina (2 vCPU, 3,8 GB RAM), coste 0 $.*

1. ✅ **El recuento de parámetros de la especificación es exacto.** §7.1 dice 5.812 totales y 68
   en la cabeza: salen **5.812 y 68**, clavados (`python nn/modelo.py`).
2. ✅ **El renderizador de párrafos funciona a 584 × 584 en esta máquina.** Medido **dos veces
   por separado**: 3 renders en 2,6 s (**0,87 s/imagen**) y, en una comprobación independiente,
   3 renders en 2,34 s (**0,78 s/imagen**, con extremos de 0,64 y 0,88).
   → **las 1000 imágenes del §3.1 son 13-15 min** *(estimado a partir de esos ritmos medidos)*.
   ⚠ **No hace falta Google Chrome** —que en esta máquina **no está**— porque el Chromium de
   Playwright sí está en caché. Es lo contrario de lo que avisa el `CLAUDE.md` del coordinador
   para renders en `nyc1`, así que conviene no darlo por roto sin probarlo.
3. ✅ **Entrenar es BARATO, y esto cambia la forma del encargo.** Un paso de lote 20 tarda
   **35,1 ms** (100 pasos cronometrados, torch 2.14.0+cpu, 2 hilos), así que **un run entero de
   200 épocas = 1000 pasos ≈ 0,6 min de entrenamiento puro**.
   → el banco completo —calibración de 20 runs (§11.2) más ~5 condiciones × 5 semillas— es del
   orden de **~45 runs ≈ 30-45 min** *(estimado desde el paso medido; no incluye la carga del
   dato ni las 4 paradas de evaluación, que son forward sobre 1000 muestras y son menores)*.
   **⇒ Esto se corre AQUÍ. No hace falta alquilar nada**, y por eso `gasta` es
   `entrena-local` y no `alquila`: no entran las obligaciones de flota (prefijo, vigilante,
   destrucción por etiqueta).
4. ⚠⚠ **`placement.area` acota SÓLO la esquina superior-izquierda de la caja, NO la caja
   entera** — y esto está confirmado por las dos vías, medición **y** código:
   - **el código lo dice literalmente**: `PlacementRecipe.area` en
     `image-text-sample-generator/app/models/recipe.py` se documenta como *«Region of the canvas
     the block's **top-left corner** may land in»*, y `_place()` en `app/core/resolver.py` sólo
     calcula y acota `(x, y)` (`hi_y = max(y0, y1 - rh)`): **nada** recorta el borde
     inferior-derecho, ni contra el área ni contra el lienzo;
   - **medido en 6 renders** (3 + 3 de una comprobación independiente): **los 6** con
     `y0 = 67,97`, o sea pegados al borde del área (que empieza en 67,98). Cajas de la primera
     tanda: `(105,66 · 67,97 · 268,36 · 353,01)`, `(98,69 · 67,97 · 242,40 · 436,08)`,
     `(79,77 · 115,45 · 263,45 · 304,46)`.
5. ⚠⚠⚠ **Y la consecuencia es peor que «fuera de `[68, 512]`»: una caja puede SALIRSE DEL
   LIENZO.** En la segunda tanda, un render dio **`y1 = 641,62` sobre un lienzo de 584** — un
   párrafo **cortado por abajo**. Una caja cortada da una etiqueta que **no describe lo que se
   ve**, que es un daño distinto del que describe el §3.3 (él sólo habla del marco final).
   → hay que **descartar las dos cosas**: caja fuera de `[68, 512]` **y** caja fuera del lienzo.
   Y por tanto **para 1000 muestras válidas hay que pedir más de 1000 imágenes**; cuántas más
   **no está medido**, y sale gratis medirlo al generar.
   ⚠ Además el generador puede devolver solape o ningún bloque (`has_overlap`, `blocks` vacío),
   que son dos descartes más.

## ⚠⚠ Lo que hace falta: una contradicción y tres decisiones

### P1 — el §7.1 se contradice consigo mismo, y la arquitectura es INVARIANTE

**Es lo único que puede invalidar el banco entero, así que va primero.** El §7.1 escribe la
cadena `128 → 64 → 32 → 16` y a la vez dice *«todas las convoluciones son `valid`»*. **Las dos
cosas no pueden ser ciertas** *(comprobado el 2026-09-08 con torch)*:

| lectura | cadena real | rejilla a la cabeza | parámetros |
|---|---|---|---|
| `valid` (la **palabra** del §7.1) | 128 → **62 → 29 → 14** | **14 × 14** | 5.812 |
| padding `same` (las **dimensiones escritas**) | 128 → **64 → 32 → 16** | **16 × 16** | 5.812 |

⚠ **El recuento de parámetros no desempata**: son 5.812 en los dos casos, porque el padding no
añade pesos. La cifra del §7.1 es compatible con ambas lecturas.

**Lo que sí desempata son tres cosas del propio documento, y las tres apuntan a `same`:**

1. §7.3 lee *«cada marginal de **16 elementos**»* y divide por **15** para normalizar a [0,1];
2. §7.5 dice *«la rejilla de 16 × 16 corresponde a **8 px por celda** en el marco de 128»*, y
   128/16 = 8 **exacto** (con 14 saldría 9,14, que no es lo escrito);
3. §7.5 da como arreglo diagnóstico quitar el stride de la tercera conv *«pasando a **32 × 32**»*,
   que es lo que sale de 64 → 32 → 32, no de 62 → 29 → 29.

**Qué se hizo mientras se decide:** `nn/modelo.py` implementa **`same`** —la lectura que
reproduce las dimensiones escritas y las tres corroboraciones— y lo deja como **interruptor de
una línea** (`PADDING_SAME`), con las dos cadenas impresas al ejecutarlo. **No se eligió en
silencio.**

**Qué se necesita del dueño:** confirmar que manda la **cadena de dimensiones** y no la palabra
`valid`. Si manda `valid`, hay que **reescribir el §7.5** (la rejilla sería 14 × 14, a 9,14 px
por celda) y el §7.3 (marginales de 14, dividir por 13). Se pregunta porque el §12 declara la
arquitectura **invariante**: elegir mal no da un error, da una serie de mediciones que hay que
tirar entera.

### P2 — cómo se empaqueta el dataset, que se publica una sola vez

Las imágenes reducidas son **146 × 146 en 1 canal**, y hay 1000. El promedio por área de un
`uint8` sobre bloques de 4 × 4 da múltiplos de 1/16, o sea **valores no enteros**, así que el
`dtype` es una decisión con consecuencias y **no se puede cambiar después** *(cifras calculadas,
no medidas)*:

| opción | tamaño en crudo | ¿exacto? |
|---|---|---|
| `float32` | ~85 MB | sí, pero es el más grande y **git guarda todas las versiones** |
| `uint16` guardando la **suma** del bloque (0…4080) | ~43 MB | **sí, exactamente** |
| `uint8` redondeando | ~21 MB | **no**: pierde hasta ½ nivel por píxel |
| `float16` | ~43 MB | **no** cerca de 255 (el paso es 0,25 ahí) |

**Recomendación: `uint16` con la suma del bloque** — es exacto, cabe en la mitad que `float32`, y
lo que se guarda es un entero reproducible en vez de un flotante. ⚠ **El `.npz` comprime mucho**
(el fondo es blanco uniforme), así que el tamaño real será bastante menor; **no está medido**.

⚠ **Por qué se pregunta en vez de elegir:** este banco existe para medir diferencias **finas**
entre kernels, y un redondeo en la entrada es ruido que entra **antes** del kernel, igual para
todas las condiciones pero no necesariamente inocuo. Y el dataset **no se reescribe nunca**.

### P3 — qué varía el generador, que es lo que la estratificación tiene que equilibrar

§4.1 pide muestreo **estratificado** *«según las características que el generador varíe»*, pero
la especificación **no dice cuáles son**: el §3.1 fija resolución, fondo, reducción y número de
muestras, y nada sobre el tamaño ni la densidad del párrafo.

`nn/receta.json` lleva hoy, **como valor de partida y no como decisión tomada**, ancho
`[200, 300]` px y `[80, 160]` palabras a 14 px — traído de la receta que ya funciona en este
repo y **re-escalado al lienzo de 584**. Hay que confirmar (o cambiar) las tres cosas:

1. **el rango de tamaño y de palabras** del párrafo;
2. **qué variables estratifican** el reparto 100/100/800 (tamaño · densidad · posición son las
   que nombra el §4.1);
3. **cuántas imágenes se piden** para conseguir 1000 válidas, dado el descarte del punto 5 de
   arriba.

⚠ **Y esto es lo que decide si el banco mide algo**, por el §10.1: si el generador coloca los
párrafos con **poca variabilidad**, el control de **caja media** —un predictor constante que
ignora la imagen— saca un IoU alto, todas las condiciones quedan comprimidas en un rango
estrecho y **el banco no discrimina nada**. Por eso la caja media se corre **primero**, y
**si sale alta se corrige el generador y no se continúa**.

### Y una cosa que NO es un bloqueo, pero hay que saber que está pendiente

**El número de semillas no lo decide el diseño: lo fija la calibración** (§8.5, §11.4), midiendo
la desviación real del IoU con **10 semillas** de identidad y aleatorio. La especificación dice
que puede estar en **0,03–0,05**, lo que vuelve exigente el margen de los criterios. Hasta
entonces, «≥ 5» es un suelo, no un plan.

## Y no hay ningún kernel que evaluar todavía

El banco es **agnóstico al origen del kernel** (§1) y los métodos para obtenerlos están **fuera
de alcance** (§15), así que esto no es un hueco del montaje: es el reparto de responsabilidades
que la especificación declara. `kernels/` está vacío a propósito.

Con los controles (caja media · identidad · aleatorio · gauss · sobel) **el banco se puede
calibrar y validar entero sin un solo kernel de verdad**, que es justo lo que el §11 pide hacer
primero.
