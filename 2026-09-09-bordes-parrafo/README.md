# `bor-p` — bordes de párrafo

**El dato de entrada para aprender kernels de hasta 19 px que localicen los
cuatro bordes del recuadro que encierra un párrafo.**

Estado: **`abierto`**. El dataset está hecho, publicado y comprobado. **No hay red, ni
criterio, ni ninguna cifra medida sobre kernels**, y eso es el encargo —
*«Por ahora solo obtén las páginas»* — no un hueco.

⚠ Este README **se genera** (`python nn/escribir_readme.py` leyendo el manifiesto). No
se edita a mano: una tabla transcrita es como nacen los números que nadie puede auditar.

## Qué hay

| | |
|---|---|
| dataset | `parrafos1000-pagina1024-r20260909`, en `foveal-vision-data/experimentos-cnn/` (**repo privado**) |
| páginas | **287** de 1024 × 1024 px, gris de 1 canal, fondo blanco sólido |
| párrafos | **1000** (2→78 · 3→66 · 4→69 · 5→74 por página) |
| etiqueta | 4 enteros por párrafo: `izq, der, sup, inf` — la caja de **TINTA** |
| descartadas | **0** de 287 planeadas |
| coste | 574 renders, 658.5 s |

⚠ El `.npz` **no está aquí y no puede estarlo**: este repo es público y el de datos
privado. Se lee con `expcnn.exigir_dataset("parrafos1000-pagina1024-r20260909")`.

## Las tres garantías, todas derivadas de `k_max = 19`

| | mínimo duro | medido | por qué ese mínimo |
|---|---|---|---|
| separación entre párrafos | 19 px | **41.88** (mediana 110.53) | `2·9+1`: las ventanas 19×19 de dos bordes enfrentados no se solapan |
| margen al borde del papel | 9 px | **52.02** (mediana 77.56) | `valid` con k=19 sólo cubre `[9, 1014]` |
| solape entre párrafos | ninguno | **ninguno** | se construye, no se filtra |

⚠ La métrica es **L-infinito**, no euclídea: es la que corresponde a una ventana
**cuadrada**. Dos cajas a 14 px en diagonal están a 19,8 euclídeos y una ventana de
19×19 **sí las mezcla**.

## Cuánto se puede recortar

Las ventanas de entrenamiento se recortan **después**, y su presupuesto ya viaja con el
dato: `derivadas` de `cajas.npz` trae `ventana_limpia` por párrafo.

| mínimo | percentil 10 | mediana |
|---|---|---|
| **81 px** | 107 px | 140 px |

Una ventana de hasta **81×81** centrada
en cualquier borde es limpia para **todos** los párrafos, sin filtrar ninguno.

⚠ Y el reparto `train`/`val`/`eval` es **por página** (201 ·
29 · 57), no por párrafo: dos ventanas de la
misma página comparten fuente, fondo y vecinos, así que repartirlas sería una fuga que no
falla por ningún lado.

## Variación

| factor | rango usado |
|---|---|
| fuente | DejaVuSans, DejaVuSerif, LiberationSans, LiberationSerif |
| cuerpo | 11.0–30.0 px |
| interlineado | 1.101–1.42 |
| ancho de tinta | 146.0–919.0 px |
| alto de tinta | 56.0–493.0 px |
| palabras | 6–532 |
| color del texto | #000000 fijo (no se vario: el encargo pide papel limpio) |

## ⚠ La reserva de `banco-k` (§3.7): RESPETADA

Es **el requisito que motivó todo esto**. `banco-k` reserva `LiberationMono` y el
interlineado `[1.45, 1.6]` para uso exclusivo
suyo; un kernel aprendido sobre datos que los alcancen sale **optimista** al evaluarlo
allí. Este dataset usa las otras cuatro familias e interlineado
`[1.1, 1.42]`.

Comprobado ejecutando el chequeo **del propio banco** (`banco-k/nn/importar_kernel.py`):

```
bor-p    usa_la_reserva=False   ← este experimento
esq-k    usa_la_reserva=True    ← el vecino, que sí filtra
```

## Cómo se repite

```bash
python nn/geometria.py                                   # 27 comprobaciones, sin render
python nn/generar_paginas.py --reserva                   # el contrato §3.7, solo
python nn/generar_paginas.py --parrafos 1000            # ~11 min
python nn/generar_paginas.py --publicar                  # una vez en la vida
python nn/generar_paginas.py --comprobar                 # casa las huellas del publicado
python nn/generar_paginas.py --muestras 6                # la figura con las cajas encima
```

⚠ Se corre con el venv **del generador** (`~/src/image-text-sample-generator/.venv`), que
es el que trae Playwright. `nn/geometria.py` no necesita nada: es stdlib pura.

⚠ Y lo largo va desacoplado, que es la regla del proyecto para cualquier cosa lanzada
desde una sesión de Claude Code:

```bash
COORD_HOME="$HOME/src/telegram-coordinator" \
  "$HOME/src/telegram-coordinator/scripts/desacoplar-persistente.sh" borp-dataset \
  sh -c 'cd <esta carpeta> && <venv>/bin/python nn/generar_paginas.py --parrafos 1000'
```

Las condiciones, las decisiones y qué **no** se hereda de nadie, en
[`REGLAS.md`](REGLAS.md). El encargo literal y lo que se midió antes de escribir código,
en [`instrucciones/01-encargo.md`](instrucciones/01-encargo.md).
