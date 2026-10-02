# Reglas de `bor-p4`

**Escritas el 2026-10-01** al montar la carpeta, como fase 0a del plan
[`docs/plan-kernels-banco-2026-10-01.md`](../docs/plan-kernels-banco-2026-10-01.md). Su
`experimento.json` declara el estado **`abierto`**: produce un dataset y **nada más**. No hay
red ni criterio: los que van a aprender kernels con este dato son `bor-k`, `bor-ae` y
`bor-pca`, cada uno con sus reglas.

⚠ **Estas reglas son de este experimento y de ninguno más.** Se copió de `bor-p` para no
reescribir lo que allí ya estaba especificado (§ «Qué NO hereda»), y **`bor-p` no se tocó**.

**Qué pregunta:** ninguna. Produce el dato: páginas limpias con 2–4 párrafos y la caja de
tinta de cada uno, **a la escala en que `banco-k` aplica el kernel** (/4), con la separación
y el margen que exige un kernel de 19 px **a esa escala**.

## Entradas

- **Dataset que consume:** ninguno publicado. Lo **produce** este experimento con
  `nn/generar_paginas.py --parrafos 1000`, una vez en la vida, a partir de la receta
  `nn/receta.json` y del generador `image-text-sample-generator` (puerta:
  `expcnn.exigir_generador()`).
- **Dataset que publica: `parrafos1000-pagina1024-r4-r20261001`**, en
  `foveal-vision-data/experimentos-cnn/`. Se lee con
  `expcnn.exigir_dataset("parrafos1000-pagina1024-r4-r20261001")`.
- **Qué trae, y es lo que hay que saber antes de leerlo:**
  - `paginas.npz` → `paginas` `(P, 256, 256)` **`uint16`**: la **suma** de cada bloque 4×4
    del render de 1024 (el método del §3.8 de `banco-k`, exacto). Papel blanco = **4080**,
    tinta negra pura = 0. `particion` (0=train 1=val 2=eval, **por página**) y `n_parrafos`.
  - `cajas.npz` → `cajas` `(párrafo, 5)` `int32`: `pagina, izq, der, sup, inf` en px del
    **lienzo de render (1024)**. En la página guardada valen **`coord / 4`**, sin redondear
    (la misma aritmética que el banco aplica a sus etiquetas).
  - `cajas.npz` → `derivadas` `(párrafo, 3)` `float32`: `margen_r4`, `separacion_r4`,
    `ventana_limpia_r4`, en px **guardados** — es donde se recortan las ventanas.
  - `meta.json` → los factores de cada párrafo y el porqué de cada página descartada.
- **Condiciones:**
  - render de **1024 × 1024**, guardado **256 × 256**; gris de 1 canal; **fondo blanco
    sólido** y texto **`#000000`** fijo;
  - **de 2 a 4 párrafos por página** (no 2–5: ver § «Qué NO hereda»), y **ninguno se solapa**;
  - la etiqueta es la **caja de TINTA** (unión de las cajas de palabra), `align: justify`;
  - cuerpo **11–30 px en render → 2,75–7,5 px en la guardada**, que es el rango que ve el
    kernel en el banco (allí: 11–30 px a 584, reducido /4). **Para eso existe este dataset.**
- **Las tres garantías geométricas, derivadas de `k_max = 19` A LA ESCALA GUARDADA** (viven
  en `nn/geometria.py`, que las comprueba solo):

  | | mínimo duro (guardado · render) | se construye con (render) | de dónde sale |
  |---|---|---|---|
  | separación entre párrafos | **19 · 76** | 160 | `2·9+1`: las ventanas 19×19 de dos bordes enfrentados no se solapan |
  | margen al borde del papel | **9 · 36** | 128 | una convolución `valid` k=19 sobre 256 sólo cubre `[9, 246]` |
  | solape | **ninguno** | por construcción | |

  Con la construcción, la tinta queda a **≥ 52 px guardados** del papel y a **≥ 40** de otro
  párrafo, así que **toda** ventana de **79 × 79** (guardados) centrada en un borde es limpia.
  ⚠ La métrica es **L-infinito**, la de una ventana cuadrada (prueba dedicada en
  `nn/geometria.py`).
- **La reserva de `banco-k` (§3.7), que es lo único que se hereda del banco:** este dataset
  **no usa** `LiberationMono` ni interlineado en `[1.45, 1.60]`. Se comprueba en código
  (`comprobar_reserva()`) **antes** de rendir, y `nn/receta.json` lo declara explícito porque
  omitirlo es filtrar (sin `fonts` el generador sortea todas las familias; sin `line_height`
  su defecto solapa con la banda).
- **Qué se normaliza al cargar:** nada. Se guarda la suma cruda; cualquier normalización es
  del experimento que lo consuma.

## Salidas

- **El dataset**, que es la única salida: se publica en
  `foveal-vision-data/experimentos-cnn/parrafos1000-pagina1024-r4-r20261001/` y **nunca se
  copia a este repo** (éste es público; aquél, privado y con `origin` en el almacén).
- **Etapa local:** `datos/` (ignorada por git): `paginas/*.npy` según se rinden, luego
  `paginas.npz`, `cajas.npz`, `meta.json`, `manifiesto.json`. `datos/ensayo/` y
  `datos/rederivar/` son de las pruebas y no se publican.
- **Figuras:** `muestras/paginas-r4-<n>.png` — páginas **guardadas** (256, ampliadas ×2 sin
  suavizar) con la caja /4 dibujada encima. Se regenera con `--muestras N`.
- **Pesos y métricas:** ninguno. Aquí no se entrena.
- **Qué se commitea:** el código, las reglas y las figuras. **No** `datos/` ni ningún `.npz`.

## Procesos

1. ✅ **Generar y publicar el dataset**, una vez en la vida — **hecho el 2026-10-01** (publicado y `--comprobar` → «TODO CASA»). En este orden:
   1. **La reserva §3.7 se comprueba ANTES de rendir.**
   2. **El reparto en celdas GARANTIZA la separación** (guillotina + `ranura()`), no la
      busca. No se sortea-y-rechaza: eso eliminaría los párrafos grandes.
   3. **Cada página se rinde DOS veces**: el pase A apila los párrafos y **mide** su caja
      real; el pase B coloca cada uno donde toca para esa caja.
   4. **Los cuerpos se EMPAREJAN con las ranuras**: se sortean los `n` cuerpos de la página
      y el mayor va a la ranura más ancha (`cuerpos_emparejados()`). ⚠⚠ **Esto no estaba en
      `bor-p`, y se añadió porque el primer render salió sesgado** (2026-10-01): con la
      geometría ×4 hay ranuras de 120 px de ancho, y ahí un cuerpo de 30 px no mete ni las
      6 palabras mínimas sin pasarse de alto. La página se descartaba —91 descartes, 6
      páginas perdidas enteras, **977 párrafos**— y el cuarto de cuerpo más grande quedó un
      **11 % por debajo** del uniforme (`[270, 251, 237, 217]` por cuartos de `[11, 30]`,
      ~244 cada uno). Era sortear-y-rechazar sobre el cuerpo, justo lo que el paso 2
      prohíbe. Aquel render no se publicó; su metadata está en el almacén
      (`foveal-vision-data/temporal/dev/2026-10-01/bor-p4-primer-render/`).
      Emparejar **no cambia ninguna distribución marginal** (los cuerpos se sortean igual y
      las ranuras salen del mismo reparto); lo que cambia es **qué cuerpo cae en qué
      ranura**: dentro de una página, el ancho del párrafo y su cuerpo quedan
      correlacionados en positivo. Es una decisión de este experimento y se dice aquí.
   5. **El descarte existe y es la RED DE SEGURIDAD**, con **10 intentos por página** (eran
      3; los intentos son deterministas, así que subirlo no cambia ninguna página que ya
      saliera). Lo que queda de él es sobre todo la cota **declarada** de ancho de tinta
      (`ancho ≥ 120` px de render): un párrafo cuyas líneas no llenan la columna. El número
      va en el manifiesto (`descartadas`) y el porqué de cada uno en `meta.json`.
      **Medido en el render publicado (2026-10-01, 1412 s): 40 descartes** —26 «no se pudo
      ajustar el tamaño» y 14 de ancho de tinta—, **0 páginas perdidas, 1000 párrafos**, y el
      cuerpo **sin sesgo**: `[253, 255, 239, 253]` por cuartos de `[11, 30]` (uniforme: 250),
      χ² = 0,66 con 3 g.l. (el 5 % está en 7,81), media 20,39.
   6. **Se REDUCE /4 cada página en cuanto se rinde** (suma del bloque 4×4) y se guarda.
   7. **Se comprueba la TINTA contra las cajas en la página reducida**: ningún bloque con
      tinta puede caer fuera de toda caja (con 1 bloque de tolerancia, porque las cajas se
      guardan redondeadas al px de render). Si cae, **no se empaqueta**: la etiqueta no
      describiría la imagen. *Ensayo del 2026-10-01: 0 bloques fuera, incluso sin tolerancia.*
   8. **Publicar es irreversible.** `--publicar` se niega a pisar uno existente.
2. **Comprobar el publicado** (`--comprobar`): huellas, garantías geométricas y tinta,
   recalculadas sobre lo que está en el repo de datos.

- **Qué se mide y con qué umbral:** sólo las garantías del **instrumento** (separación,
  margen, tinta dentro de la caja). No hay hallazgo que medir.
- **Brazos, semillas, «ganar»:** no aplica. La semilla del dataset es **1**.

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/geometria.py` | el reparto, la colocación y **las aserciones**, en las dos escalas. Aritmética pura | `python nn/geometria.py` (27 comprobaciones) |
| `nn/generar_paginas.py` | genera, reduce, empaqueta, comprueba y **publica** | `--parrafos 1000` · `--parrafos 1000 --paginas N` (ensayo, a `datos/ensayo/`) · `--publicar` · `--comprobar` · `--rederivar N` · `--muestras N` · `--reserva` |
| `nn/receta.json` | los factores del generador **y el contrato anti-fuga del §3.7** | lo lee `nn/generar_paginas.py` |

- ⚠ **`generar_paginas.py` se llama así por el FRENO**: `cerrable.mjs` casa
  `generar_paginas\.py` en su lista `TRABAJOS` (entró con `bor-p`). Con otro nombre, el
  veredicto diría «nada corriendo» durante el render.
- **Cómo se corre** (el render tarda más de lo que dura un turno: va como unidad):

  ```bash
  cd 2026-10-01-bordes-parrafo-r4
  COORD_HOME=~/src/telegram-coordinator ~/src/telegram-coordinator/scripts/desacoplar-persistente.sh \
    bor-p4-render sh -c 'ITF_CHROMIUM_PATH=/usr/bin/google-chrome \
      ~/src/image-text-sample-generator/.venv/bin/python -u nn/generar_paginas.py --parrafos 1000; \
      node "$COORD_HOME/scripts/notify.mjs" "bor-p4: render terminado" || true'
  ```

- **Dependencias:** el venv del **generador** (`~/src/image-text-sample-generator/.venv`:
  numpy, pillow, playwright) y **Google Chrome** en `/usr/bin/google-chrome`
  (`ITF_CHROMIUM_PATH`): el Chromium de Playwright no se descarga desde `nyc1` (403, medido
  el 2026-08-27). Ni torch ni `fv`.
- **De dónde sale el código:** copiado de `bor-p` y cambiado (abajo). No importa nada de
  ningún experimento.

## Qué NO hereda

- **Se copió de:** `bor-p` (2026-10-01): `nn/geometria.py`, `nn/generar_paginas.py`,
  `nn/receta.json`. **`bor-p` no se tocó**: su dataset `parrafos1000-pagina1024-r20260909`
  está publicado y sus `--comprobar`/`--rederivar` tienen que seguir valiendo con su código.
- **Qué se cambió a propósito, y por qué:**
  - **la reducción /4 al guardar** (suma del bloque 4×4): el banco aplica el kernel después
    de reducir /4, y un kernel aprendido sobre `bor-p` (sin reducir) vería tinta ~4× más
    gruesa que la que verá allí. Es la decisión 1 del plan, tomada por el dueño;
  - **la geometría ×4** (separación 160, margen 128 en render): las garantías se derivan de
    `k = 19` **a la escala guardada**. `bor-p` no se puede reducir tal cual: su separación
    mínima (41,88 px) queda en 10,5 a /4;
  - **los cuerpos emparejados con las ranuras** y **10 intentos por página** (paso 1.4 y
    1.5 de Procesos): sin eso el primer render salió sesgado contra el cuerpo grande;
  - **2–4 párrafos por página, no 2–5**: con la geometría ×4, 5 párrafos no caben en el
    **22 %** de las páginas (simulado el 2026-10-01 con `celdas()` sobre 2000 repartos), y el
    código de `bor-p` bajaba la densidad en silencio, dejando el total por debajo de 1000;
  - **el tope de ancho de la receta, 608** (era 920): es la ranura más ancha posible con la
    geometría ×4, `(1024 − 2·128) − 160`;
  - **las páginas como array `uint16` y no como PNG concatenados**: a /4 caben en RAM;
  - **las derivadas en px guardados**, y **la comprobación de tinta sobre la página
    reducida**, que `bor-p` no necesitaba.
- **Qué se conservó, y por qué:** el lienzo de **1024** — a /4 da párrafos de 30–152 px de
  ancho, que es la escala del banco (sus párrafos van de 20 a 105 px a /4; leído de su
  manifiesto el 2026-10-01); con un lienzo mayor serían más grandes que los del banco. Y
  todo lo demás de la receta (fuentes, cuerpo, interlineado, `justify`, texto negro, fondo
  blanco, reparto 70/10/20 por página, semilla 1), porque es lo que el plan pide: el mismo
  dato que `bor-p`, a otra escala.
- **Contra qué se compara:** con nada. Es un dataset.
- **Restricciones de otros experimentos que NO aplican aquí:** del banco no se hereda ni su
  lienzo de 584, ni un párrafo por imagen, ni el gris variable, ni las particiones
  100/100/800. Sólo la reserva §3.7, que es una prohibición del sistema.
