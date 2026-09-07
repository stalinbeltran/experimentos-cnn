# El criterio de `lim-ab`, congelado ANTES de la primera época

**Escrito el 2026-09-07 con CERO épocas entrenadas.** Los suelos salen de
`nn/entrenar_local.py --suelos` sobre el `val.npz` **de este dataset**, no de memoria y **no de
`esq-cq`**. Nada de aquí se toca después de mirar.

## La pregunta

Orden del dueño (2026-09-07):

> «puesto que las esquinas son ambiguas, vamos a hallar las posiciones y superior e inferior de
> cada párrafo […] se van a probar los mismos valores de kernel, pero ahora serán **límites** en
> vez de esquinas»
>
> «La altura tómala de la pos y de **tl** y **bl** (borde alto y bajo)»

Una salida: **`existe`** y **una altura `y`**, sin decir si el límite es el superior o el
inferior. **No hay `x`**: un límite horizontal es una línea.

## ⚠⚠ Los umbrales NO se heredan, y este es el fallo más caro que se evitó

| | `esq-cq` (2-D) | **`lim-ab` (1-D)** |
|---|--:|--:|
| métrica | distancia euclídea | **\|Δy\|** |
| suelo sin entrenar | 5,8 % | **21,2 %** |
| umbral | 9,6 % | **25,8 %** |
| `λ` de la pérdida | 0,0293 | **0,0522** |

**El umbral de `esq-cq` (9,6 %) está muy por debajo del suelo de aquí (21,2 %).** Copiarlo habría
hecho que los siete brazos «pasaran» en la época 0 sin haber aprendido nada, y el informe lo
habría dicho con un ✅. La `λ` cae en la misma trampa: al perder el término en `x`, el `mse` baja
casi a la mitad y la `λ` que iguala los dos términos sube ×1,78.

## Los suelos, medidos

*`--suelos`, 2026-09-07, sobre las 780 ventanas de `val`.*

| | valor |
|---|--:|
| positivas | **312 / 780 = 40,0 %** (156 sup + 156 inf) |
| celdas | `tl` 78 · `tr` 39 · `bl` 39 · `br` 78 · **puros 39+39** |
| negativos duros | interior 156 · borde vertical 156 · fondo 156 |
| acierto \|Δy\| ≤2 px sin entrenar | **21,2 %** (alto 19,2 · bajo 23,1) |
| error \|Δy\| medio | **4,28 px** |
| `f1` del «siempre sí» | **0,571** |
| **umbral** | **> 25,8 %** (suelo + 2 SE, SE = 2,3 % sobre 312) |

⚠ El `f1` de 0,571 es **el mismo de `esq-cq` (0,572) por construcción, no por suerte**: el reparto
de ventanas se rehízo para dejar el 40 % de positivas justamente para eso.

## ⚠ El desglose es el resultado, no el número global

Seis celdas, y dos grupos que contestan cosas distintas:

- **`tl` · `tr` · `bl` · `br`** — las **mismas 78/39/39/78 ventanas de `esq-cq`**. Es lo único
  comparable columna a columna con aquel experimento.
- **`puro_sup` · `puro_inf`** — el límite **sin ninguna esquina**. Es lo que `esq-cq` no podía
  medir (su offset se sorteaba y se tiraba) y la razón de haber regenerado el dataset.

## La medida hecha antes de correr, y lo que predice

*2026-09-07, sobre las 389 ventanas del `val` de `esq-cq`. Distancia de la etiqueta a la tinta,
como punto (euclídea) y como línea (sólo vertical).*

| celda | como **punto** | como **línea** | |
|---|--:|--:|---|
| `tl` | 1,00 px | **1,00 px** | igual |
| **`tr`** | 6,00 px | **1,00 px** | ✅ **el cambio la arregla** |
| `bl` | 2,00 px | **2,00 px** | igual |
| **`br`** | 8,25 px | **7,00 px** | ❌ **apenas mejora** |

**El cambio de esquina a límite arregla tres celdas de cuatro y falla en la misma que ya falló en
`esq-cq`**, por la misma causa geométrica: la última línea del párrafo es corta, así que el borde
bajo pasa **sobre fondo** en su tramo derecho.

## Los desenlaces, por orden de probabilidad estimada

1. **`alto` sube mucho y `bajo` se atasca en su celda `br`** *(el más probable, y por un mecanismo
   medido)*. El desglose enseñaría `tl`≈`tr`≈`puro_sup` altos y `br` bajo. Cerraría que el
   problema no era «esquina contra límite» sino **dónde hay tinta**.
2. **Las seis celdas suben y el global pasa holgado.** Entonces el cambio de objetivo era la
   respuesta y `esq-cq` medía sobre todo la ambigüedad de `br`.
3. **`fp_interior` se dispara.** El modo de fallo propio de aquí: cada línea de texto tiene arriba
   y abajo un canto horizontal, y hay decenas por párrafo. Si la red no distingue «el borde del
   párrafo» de «el borde de una línea», esto lo enseña.
4. **Nada pasa el 25,8 %.** Entonces la lectura por máximo no vale para una cresta y la
   continuación es `sim` o el perfil 1-D (ver `03-alternativas-anotadas.md`).

## La predicción registrada — y aquí la premisa SÍ se sostiene

**La fracción simétrica bajo VOLTEO VERTICAL debe SUBIR al entrenar.**

`conv(flipud(x), W) = flipud(conv(x, W))` ⟺ `W[a,b] = W[k−1−a, b]`. El par (borde alto, borde
bajo) **sí** es reflejo uno del otro por volteo vertical.

⚠ En `esq-cq` la predicción análoga se escribió sobre `rot180` y su premisa estaba **medida
falsa** (el parche de `br` no era el giro del de `tl`: 8,1 px contra 1,0 px de tinta). Aquí la
simetría es la correcta para el par que hay que tratar igual.

⚠ **Se reportan las DOS fracciones** (`simetrico_vertical` y `simetrico_rot180`). La vertical es la
que predice esto; la de 180° se conserva sólo porque es la única columna comparable con la `S %`
de `esq-cq` y `esq-k`.

## Contra qué se compara, y contra qué NO

| | comparable | **no** comparable |
|---|---|---|
| **`esq-cq`** | la **arquitectura** entera (conv sin bias/padding, cabeza de 3, lectura por máximo, totales `k²+3`, eje `k`, 300 épocas, semilla 1, lr 0,05, lote 128) · las **mismas 267 imágenes** y el mismo split · el **n por celda** (78/39/39/78) · el **suelo de `f1`** (0,571 ≈ 0,572), por construcción | el **acierto titular**: 1-D contra 2-D, con suelos **21,2 %** contra **5,8 %**. Un 60 % aquí no es un 60 % allí · `err_px` (\|Δy\| contra euclídea) · la columna de simetría (vertical contra rot180) salvo mirando la de 180° · el **fichero** de dataset |

**Lo que sí se puede poner en la misma frase que `esq-cq`: la forma de la curva por celda contra
el radio.** Allí `tl` saturó al 100 % desde `k07` y `br` subió monótona 3,8 → 44,9 %. La predicción
de aquí es que `tr` se comporte ahora como `tl` (porque su tinta pasó de 6,00 a 1,00 px) y que
`br` siga necesitando radio.

## Qué NO contesta

- **Nada sobre los límites verticales** (izquierdo/derecho): aquí son negativo duro.
- **Nada sobre la página entera**: se mide aparte, y en `esq-2d` el orden de los brazos **no se
  conservó** entre ventana y página.
- **Una semilla.** Los brazos vecinos no se van a poder declarar.
