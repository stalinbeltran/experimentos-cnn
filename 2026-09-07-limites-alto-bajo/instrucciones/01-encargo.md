# El encargo de `lim-ab`

## La orden, literal (2026-09-07)

> «Copia este experimento. Debe ejecutarse idéntico, pero puesto que las esquinas son ambiguas,
> vamos a hallar las posiciones y superior e inferior de cada párrafo. Seguramente deberás ajustar
> varios scripts para que esto funcione. En resumen, se van a probar los mismos valores de kernel,
> pero ahora serán límites en vez de esquinas»

Y la aclaración que resolvió el punto clave:

> «La altura tómala de la pos y de tl y bl (borde alto y bajo)»

## Por qué: «las esquinas son ambiguas» está MEDIDO

`esq-cq` corrió el 2026-09-07 y su desglose lo dice sin ambigüedad: **`tl` satura al 100 % desde
`k07`, y `br` se queda en 44,9 % incluso con `k17`**, subiendo monótono con el radio del campo
receptivo. La causa se midió **antes** de correrlo: la tinta más cercana a `br` está a **8,1 px**
(contra 1,0 px en `tl`), porque `br` es el vértice inferior-derecho de la **caja del layout** y la
última línea del párrafo es corta — así que ese vértice cae sobre fondo.

## La extensión de la orden, y por qué no es una interpretación libre

El dueño nombró `tl` y `bl`. **`tr` y `br` llevan exactamente las mismas dos líneas**, y eso es
aritmética de la caja `(x,y,w,h)`:

```
tl = (x, y)        tr = (x+w, y)        ← la MISMA y: el borde ALTO
bl = (x, y+h)      br = (x+w, y+h)      ← la MISMA y: el borde BAJO
```

Una ventana centrada en `tr` **contiene el borde alto igual que una de `tl`**; lo único distinto
es por dónde lo corta. Dejarlas fuera no sería fidelidad a la orden: sería etiquetar como
**ausente** un límite que está ahí, con tinta a 1 px, en 78 de 389 ventanas. Eso no es ruido de
etiqueta, es una **anti-etiqueta** — y obligaría a la red a resolver `existe` por el canto
**vertical** de la esquina, o sea a volver a ser `esq-cq` en silencio.

## Lo que se midió antes de diseñar nada

*2026-09-07, sobre las 389 ventanas del `val` de `esq-cq`, 0 $.*

| celda | tinta como **punto** | tinta como **línea** | |
|---|--:|--:|---|
| `tl` | 1,00 px | **1,00 px** | igual |
| **`tr`** | 6,00 px | **1,00 px** | ✅ el cambio la arregla |
| `bl` | 2,00 px | **2,00 px** | igual |
| **`br`** | 8,25 px | **7,00 px** | ❌ apenas mejora |

**Tres celdas de cuatro mejoran; la cuarta no.** El borde bajo, en su tramo derecho, sigue pasando
sobre fondo. Está escrito en el criterio como el desenlace más probable.

## Por qué hubo que REGENERAR el dataset

En `esq-cq`, las ventanas `borde-superior`/`borde-inferior` —un límite horizontal **sin ninguna
esquina**, que es el caso más limpio de esta pregunta— existían como negativos, y su offset se
sorteaba y **se tiraba** (`dict(VACIA)`). Eran positivas conceptuales sin etiqueta: 78 en `val`,
432 en `train`.

Las tres salidas eran regenerar, excluirlas, o dejarlas como negativas. **Dejarlas como negativas
es el fallo silencioso**: el borde alto tiene tinta a 1 px a lo ancho, así que enseñaría al kernel
a responder bajo justo donde la señal es más limpia, y son el 20 % del train.

Se regeneró (**~6 min, 0 $, 0 máquinas**), y de paso se arregló el reparto:

| positivas (8 de 20) | negativas (12 de 20) |
|---|---|
| 2 `tl` · 2 `br` · 1 `tr` · 1 `bl` *(idénticas a `esq-cq`)* | 4 `interior` *(el negativo duro: cada línea de texto es un borde falso)* |
| **1 `borde-superior` · 1 `borde-inferior`** *(el caso puro, nuevo)* | **2 `borde-izquierdo` · 2 `borde-derecho`** *(negativo duro nuevo)* · 4 `fondo` |

**40,0 % de positivas** — el mismo de `esq-cq`, para que el suelo de `f1` vuelva a ser 0,571 ≈
0,572 **por construcción**. Y las celdas de esquina conservan su n exacto (78/39/39/78 en `val`),
que es lo que permite comparar columna a columna.

⚠ **No se tocó** `SEMILLA`, `receta.json`, `REDUCCION`, `VENTANA`, `LADO_MIN`, `OFF_MIN`/`OFF_MAX`
ni el split: por eso salieron **las mismas 267 imágenes válidas** y `k_max_representable` sigue
siendo 17.

## Lo que cambia respecto de `esq-cq`, y lo que no

| | `esq-cq` | `lim-ab` |
|---|---|---|
| objetivo | esquina (`tl`∪`br`) | **límite horizontal** (alto ∪ bajo) |
| salida | `existe` + (x, y) | **`existe` + `y`** — sin `x` |
| métrica | euclídea 2-D | **\|Δy\|** |
| suelo · umbral | 5,8 % · 9,6 % | **21,2 % · 25,8 %** |
| `λ` | 0,0293 | **0,0522** |
| simetría que importa | rot180 | **volteo vertical** |
| negativo duro | otra diagonal (`tr`,`bl`) | **interior · bordes verticales** |
| ventanas por imagen | 10 (40,1 % pos.) | **20 (40,0 % pos.)** |
| cabeza · totales · eje `k` · épocas · semilla | | **iguales** |

**Quitar la `x` no cuesta parámetros**: `x` e `y` salían de las dos marginales del **mismo**
softmax 2-D, sin pesos propios. La cabeza sigue siendo `(β, a, b)` = **3**, y los totales siguen
siendo `k²+3` — o sea los de `esq-k`. La cláusula «idéntico» se conserva donde se puede medir.

## Qué cuesta

**0 máquinas, 0 $.** El dataset tiene ~2× las ventanas de `esq-cq` (4319 de train contra 2157), así
que se estima **~30-35 min** para los 7 brazos × 300 épocas *(estimado escalando los 0,33-0,41
s/época medidos en `esq-cq`, no medido entero)*.

El freno lo ve: `entrenar_local.py` está en la lista `TRABAJOS` de `cerrable.mjs`.
