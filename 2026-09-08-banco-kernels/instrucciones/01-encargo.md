# 01 — Encargo: montar el banco de evaluación de kernels

**Fecha:** 2026-09-08 · **Estado:** carpeta montada sobre la **v1.2** de la especificación,
**nada corrido**, **cero decisiones abiertas** — lo que queda es trabajo (§ «Lo que falta ahora»).

## Qué pidió el dueño

> «Crea una carpeta para el experimento especificado en
> https://claude.ai/public/artifacts/fe5ff2bb-2393-484a-87d0-cfacf2bfdf60, y dime si necesitas
> algo más. Documenta todo»

O sea: **montar la carpeta y decir qué falta**, no correr el banco.

Y después, el 2026-09-08: **«Actualicé el documento. Revísalo»** → la revisión contra la **v1.2**
está en la § «La v1.2 cerró las tres decisiones».

## De dónde salió la especificación, y por qué está copiada aquí

La copia de [`ESPECIFICACION.md`](../ESPECIFICACION.md) es **verbatim**, sin editar una palabra,
y hoy es la **v1.2** (`ESPECIFICACION_Banco_Kernels_v1.2.md`, 29.959 caracteres, 566 líneas). La
v1.0 con la que se montó la carpeta tenía 21.324 caracteres y 469 líneas.

**Se copia y no se enlaza porque una URL no es un artefacto.** Estos servidores se rehacen sin
aviso y de ellos sólo sobrevive lo que está empujado; un banco cuyas condiciones viven en una
página web es un banco que en el próximo server no se puede reproducir.

⚠⚠ **Y esto quedó demostrado a los dos días, por donde no se esperaba: cada versión se publica en
su PROPIO `uuid`.** El enlace de la v1.0 (`fe5ff2bb-…`) siguió sirviendo la v1.0 **byte a byte**
después de que el documento se actualizara —comprobado tres veces, con respuestas frescas del
origen (`cf-cache-status: DYNAMIC`, sin cabecera `age`)—, y la v1.2 apareció en
`d38a6851-…`. **Un artifact publicado es una instantánea, no un documento vivo**, así que «la
última versión» no es una propiedad de la URL: la URL **es** la versión. Las dos quedan en
`experimento.json` → `especificacion.url` y `url_v1_0`.

⚠ **Consecuencia práctica:** *«mismo link»* no basta para revisar una actualización. Hay que
comprobar el `sha256` o la línea de versión del propio documento, que es lo que se hizo.

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
   → ⚠ **Recalculado para la v1.2, que fija 10 semillas** (§8.5): **10 condiciones × 10 semillas
   = ~100 corridas ≈ 1 h** *(estimado desde el paso medido; no incluye la carga del dato ni las 4
   paradas de evaluación, que son forward sobre 1000 muestras y son menores)*. Coincide con la
   cuenta que hace la propia §8.5, que **cita este mismo número medido**.
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
   → la aserción tiene que cubrir **las dos cosas**: caja fuera de `[68, 512]` **y** caja fuera
   del lienzo. **La v1.2 recogió las dos** como «dos daños distintos» en §3.3, y llama al segundo
   **peor**: *«la etiqueta es directamente falsa»*.
   ⚠⚠ **Y aquí la v1.2 corrige lo que yo había escrito.** Yo concluí «pues hay que pedir más de
   1000 imágenes y descartar»; el §3.4 nuevo lo **prohíbe**: *«No se admite sobre-generar y
   rechazar»*. Ver la § «Lo que la v1.2 cambió» más abajo.

## ✅ La v1.2 cerró las tres decisiones — y corrigió una cosa que yo había escrito

**Revisado el 2026-09-08.** El dueño publicó la **v1.2** de la especificación
(`ESPECIFICACION_Banco_Kernels_v1.2.md`, 566 líneas contra 469, +108/−11), y la copia del repo
está reemplazada.

⚠ **La v1.2 vive en otro `uuid`** (`d38a6851-…`), no en el enlace de la v1.0 (`fe5ff2bb-…`), que
sigue sirviendo la v1.0 byte a byte — comprobado tres veces con respuestas frescas del origen
(`cf-cache-status: DYNAMIC`). **Un artifact publicado es una instantánea**, no un documento vivo,
así que «la especificación» de este experimento es **la copia de este repo**. Las dos URLs quedan
en `experimento.json`.

### Las tres que bloqueaban, resueltas

| | qué decía yo | qué dice la v1.2 |
|---|---|---|
| **P1** padding | el §7.1 se contradice; implementé `same` como interruptor, sin elegir en silencio | **§7.1: «todas las convoluciones del tronco usan padding `same`», cadena 128 → 64 → 32 → 16.** ✅ Coincide |
| **P2** dtype | recomendé `uint16` con la suma del bloque (exacto, ~43 MB) | **§3.8: `uint16`, suma del bloque 4 × 4, ~43 MB, exacto.** ✅ Coincide, y con el mismo motivo |
| **P3** generador | había que decidir qué varía y qué estratifica | **§3.5 (siete factores, ancho y alto críticos con ≥ 2×) y §3.6 (área en 4 bins de cuartil + balance marginal + hipercubo latino).** ✅ Resuelto |

**Y el motivo del padding es mejor que el que yo había deducido.** Yo lo apoyé en tres
corroboraciones textuales; la v1.2 lo apoya en **alcanzabilidad**, que es un argumento de
construcción y no de lectura:

| | rejilla | centros en la entrada | ¿cubre las etiquetas `[8, 119]`? |
|---|---|---|---|
| `same` | 16 × 16 | **0 … 120**, paso 8 | **sí** |
| `valid` | 14 × 14 | **10 … 114**, paso 8 | **no**: `[8,10)` y `(114,119]` quedan **inalcanzables** |

*Comprobado el 2026-09-08 propagando los centros capa a capa, y de forma independiente con
impulsos sobre convoluciones reales.*

⚠ **Un off-by-one de la v1.2, que se anota en vez de heredarlo.** La v1.2 escribe **tres veces**
(§7.1 y §7.5) que §3.3 permite bordes *«en el rango [8, 120]»*. Su propia aritmética da **119**:
§3.3 acota la caja a `[68, 512]` y §6.4 transforma `(coord/4) − 9`, o sea `512/4 − 9 = 119`. Aquí
se usa **119**.

**No cambia ninguna conclusión** —8 y 119 quedan los dos fuera del span `10…114` de `valid`, y los
dos dentro del `0…120` de `same`—, así que el argumento de alcanzabilidad se sostiene igual. Lo
único que cambia es el **margen por arriba: 1 px, no 0**. Sigue siendo ajustado, y por eso §7.5
manda comprobarlo **contra el dataset real** y no contra el rango teórico. **Es lo único de la v1.2
que convendría corregir**, y es cosmético. Un borde inalcanzable es error irreducible — la misma clase que §3.3 existe para evitar.
Y la v1.2 lo vuelve **aserción obligatoria** (§7.1 y §11.8-9), que es justo lo que
`nn/modelo.py` ya hacía y ahora también comprueba el **span**.

### ⚠⚠ Lo que yo tenía MAL: el sobre-generar y rechazar

Del hallazgo de que `placement.area` no acota la caja entera concluí *«hay que pedir más de 1000
imágenes y descartar las inválidas»*. **La v1.2 lo prohíbe**, y tiene razón:

> **§3.4 Orden de muestreo — OBLIGATORIO.** «La colocación debe muestrear **primero el tamaño**
> de la caja y **después** la esquina superior-izquierda, restringida al rango que garantiza que
> la caja completa quede dentro de [68, 512]. **No se admite sobre-generar y rechazar.**»

**Por qué mi versión estropeaba el instrumento sin fallar:** rechazar elimina **selectivamente**
las cajas **grandes** y las **periféricas**, así que sesga la distribución hacia párrafos
**pequeños y centrados** — que es exactamente el factor que §3.5 manda **maximizar**, y lo que
hace **subir el control de caja media** y comprimir el banco (§10.1). Un filtro de aspecto
inocente que ataca justo la variabilidad de la que depende que se mida algo.

Con el orden correcto **no hay rechazos: 1000 generadas son 1000 válidas**, y la aserción de §3.3
se queda como **red de seguridad, no como filtro**. Corregido en `REGLAS.md`, en `nn/receta.json`
y arriba, en el punto 5.

⚠ **Y tiene una consecuencia de implementación:** `placement.area` es un rectángulo **fijo** para
la esquina, así que **la receta no puede expresar el §3.4** —el rango de esquina depende del
tamaño sorteado—. La colocación la tiene que calcular **`nn/datos.py`**: sortea el tamaño, deriva
el rango válido de esquina *para ese tamaño*, y sortea dentro. Anotado en `nn/receta.json`.

### Lo demás que cambió, y qué toca

- **Semillas: 10 fijas** (§8.1, §8.5 y Anexo A), ya no «≥ 5 y lo fija la calibración». ⚠ **Y el
  argumento es el coste que se midió en esta carpeta**: ~35 ms/paso → ~0,6 min por corrida →
  **~1 h para 10 semillas × 10 condiciones**. Actualizado en `REGLAS.md`, `README.md` y
  `02-criterio.md`.
- **§10.1: umbral de arranque `IoU` de caja media ≤ 0,40** *(propuesto, no derivado)*. Y si queda
  por encima, el remedio es **ampliar ancho y alto**, no la posición.
- **§10.1.1 NUEVO — el techo también importa.** Si la **identidad** pasa de ~0,95 tampoco hay
  margen: es el mismo fallo por el extremo opuesto. **El rango útil del banco es la distancia
  entre caja media e identidad**; si es estrecha, ninguna cantidad de semillas produce evidencia.
- **§11: la calibración pasa de 7 a 10 pasos**, con las dos aserciones nuevas (cadena de rejillas
  y span) y el balance marginal.
- **§3.7 NUEVO — reserva contra fuga de distribución.** Se reservan configuraciones del generador
  (≥ 1 familia tipográfica y un rango de densidad) de **uso exclusivo del banco**: un kernel
  obtenido con el mismo generador tiene **fuga aunque las muestras sean distintas**. **Sin elegir
  todavía** — está en `pendiente`.
- **§14: nueve trampas nuevas**, entre ellas las dos que me tocaban: *«tronco con `valid` en vez
  de `same` — el recuento de parámetros no lo detecta»* y *«sobre-generar y rechazar»*.

### Lo que falta ahora, y ya no necesita a nadie: es TRABAJO

`bloqueado_por` está **vacío**. En `pendiente`:

1. **`nn/datos.py`**: el muestreo del §3.4 (tamaño primero), las aserciones del §3.3, el
   empaquetado `uint16` del §3.8, la estratificación del §3.6, y **publicar**.
2. **Elegir la reserva del §3.7** y declararla en el contrato de kernel.
3. **Comprobar que el generador puede variar los siete factores del §3.5** — familia
   tipográfica, tamaño de fuente, interlineado y nivel de gris **no están comprobados**, y de
   ellos depende §3.5. Es lo primero que haría, porque puede obligar a tocar el generador.
4. **`nn/pipeline.py`** (§6) y **`nn/evaluar.py`** (§9.1).

⚠ **Y el rango de tamaño de `nn/receta.json` sigue pendiente de ampliar** al **≥ 2×** que pide
§3.5: hoy tiene ancho `[200, 300]`, que es 1,5×. Se cambia al escribir `nn/datos.py`, que es
quien pasa a decidir la colocación.

## Y no hay ningún kernel que evaluar todavía

El banco es **agnóstico al origen del kernel** (§1) y los métodos para obtenerlos están **fuera
de alcance** (§15), así que esto no es un hueco del montaje: es el reparto de responsabilidades
que la especificación declara. `kernels/` está vacío a propósito.

Con los controles (caja media · identidad · aleatorio · gauss · sobel) **el banco se puede
calibrar y validar entero sin un solo kernel de verdad**, que es justo lo que el §11 pide hacer
primero.
