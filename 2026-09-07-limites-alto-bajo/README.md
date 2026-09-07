# `lim-ab` — con UN kernel, ¿qué tamaño detecta CUALQUIERA de los dos LÍMITES de un párrafo?

**Preparado el 2026-09-07. Criterio congelado en
[`instrucciones/02-criterio.md`](instrucciones/02-criterio.md) antes de la primera época.**

Una salida: **`existe`** («hay límite») y **una altura `y`**, sin decir si es el borde superior o
el inferior. **No hay `x`** — un límite horizontal es una línea.

## Por qué existe: «las esquinas son ambiguas» está medido

`esq-cq` corrió el 2026-09-07: **`tl` satura al 100 % desde `k07` y `br` se queda en 44,9 %** aun
con `k17`. La causa se midió antes: la tinta más cercana a `br` está a **8,1 px** (contra 1,0 px en
`tl`), porque `br` es el vértice de la **caja del layout** y la última línea del párrafo es corta.

**Este experimento cambia el objetivo de esquinas a límites** — pasar de un punto a una línea.

## ⚠ Lo que el cambio arregla y lo que no (medido antes de correr)

| celda | tinta como **punto** | tinta como **línea** | |
|---|--:|--:|---|
| `tl` | 1,00 px | **1,00 px** | igual |
| **`tr`** | 6,00 px | **1,00 px** | ✅ **arreglada** |
| `bl` | 2,00 px | **2,00 px** | igual |
| **`br`** | 8,25 px | **7,00 px** | ❌ **apenas mejora** |

**Tres de cuatro mejoran. La cuarta no**, por la misma razón geométrica de siempre: el borde bajo
pasa sobre fondo en su tramo derecho. Está escrito en el criterio como desenlace más probable.

## ⚠⚠ Los umbrales NO se heredaron, y ese era el fallo caro

| | `esq-cq` (2-D) | **`lim-ab` (1-D)** |
|---|--:|--:|
| suelo sin entrenar | 5,8 % | **21,2 %** |
| umbral | 9,6 % | **25,8 %** |
| `λ` | 0,0293 | **0,0522** |

**El umbral de `esq-cq` está por debajo del suelo de aquí**: copiarlo habría hecho que los siete
brazos «pasaran» en la época 0 sin aprender nada, con un ✅ en el informe. Los tres números se
midieron con `--suelos` sobre **este** dataset y se congelaron antes de entrenar.

## El dataset: nuevo, y por una razón concreta

```
foveal-vision-data/experimentos-cnn/limites300-32px-r4-r20260907/
```

En `esq-cq` las ventanas de **límite puro** —sin ninguna esquina, el caso más limpio— existían
como negativos y su offset **se sorteaba y se tiraba**. Regenerar (~6 min, 0 $) las recupera con su
posición: **78 en `val` que antes eran 78 con cero**.

De paso se rehízo el reparto — 20 ventanas por imagen, **8 positivas (40,0 %)**:

| positivas | negativas |
|---|---|
| 2 `tl` · 2 `br` · 1 `tr` · 1 `bl` *(idénticas a `esq-cq`)* | 4 `interior` *(cada línea de texto es un borde falso)* |
| **1 `borde-superior` · 1 `borde-inferior`** *(el caso puro)* | **2 `borde-izq` · 2 `borde-der`** *(negativo duro nuevo)* · 4 `fondo` |

⚠ **Mismas 267 imágenes, mismo split, mismas celdas de esquina** (78/39/39/78 en `val`): no se
tocó semilla, receta, ventana ni offsets. Y el 40,0 % de positivas es a propósito — hace que el
suelo de `f1` vuelva a ser **0,571 ≈ 0,572 por construcción**, no por suerte.

## La estructura

```
conv(x, W) → M → softmax(β·M) → y del MÁXIMO        ← LA altura (marginal sobre columnas)
                                existe = a·logsumexp(β·M)/β + b
```

Cabeza de **3 parámetros**, totales **28·52·84·124·172·228·292**. Quitar la `x` **no cuesta
parámetros**: `x` e `y` salían de las dos marginales del mismo softmax 2-D. La `x` se sigue
calculando en `diagnostico()` —para ver si el máximo se pega a un extremo de la línea— pero **no
entra en la pérdida ni en el criterio**.

### ⚠ La simetría que importa aquí es otra que en `esq-cq`

`conv(flipud(x), W) = flipud(conv(x, W))` ⟺ `W[a,b] = W[k−1−a, b]`. El par (alto, bajo) es reflejo
**vertical**, no giro de 180°. `simetria()` devuelve **las dos** fracciones: la vertical predice
esto, la de 180° se conserva porque es la única columna comparable con `esq-cq` y `esq-k`.

Hay **tres invariantes** con test, y el tercero existe para que esto no se rompa en silencio:

```bash
.venv/bin/python 2026-09-07-limites-alto-bajo/nn/modelo.py
#  1. la ALTURA sale del máximo (y no se devuelve ninguna x)
#  2. `sim` es EQUIVARIANTE exacta bajo volteo vertical
#  3. las dos simetrías son medidas DISTINTAS (sim vertical da 68 % bajo rot180)
```

⚠ **Y aquí la premisa sí se sostiene**, al revés que en `esq-cq`: allí el álgebra suponía que el
parche de `br` era el giro del de `tl`, y estaba **medido falso**.

## Cómo se corre

```bash
cd ~/src/experimentos-cnn
E=2026-09-07-limites-alto-bajo
.venv/bin/python $E/nn/entrenar_local.py --suelos     # los suelos, sin entrenar
.venv/bin/python $E/nn/entrenar_local.py --init       # los pesos de la época 0 (no pisa)
$E/nn/lanzar_barrido.sh                               # 7 brazos (unidad `limab-barrido`)
$E/nn/lanzar_barrido.sh --estado                      # ¿viva? ¿por dónde? ¿NRestarts?
```

## Cuánto cuesta

**0 máquinas, 0 $.** ~2× las ventanas de `esq-cq` (4319 de train contra 2157), así que **~30-35
min** estimados para 7 brazos × 300 épocas *(escalado de los 0,33-0,41 s/época de `esq-cq`, no
medido entero)*. El freno lo ve.

## Lo que NO contesta

- **Nada sobre los límites verticales**: aquí son negativo duro.
- **Nada sobre la página entera**: se mide aparte, y en `esq-2d` el orden de los brazos **no se
  conservó** entre ventana y página.
- **Una semilla.**
