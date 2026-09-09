# Encargo — 2026-09-09

## Lo que pidió el dueño, literal

> Vamos a crear un experimento en el repo experimentos cnn. Ahí tenemos un experimento
> banco de conparacion de kernels. Revisa cuales son los requisitos para generar un
> dataset que no afecte las mediciones realizadas ahí

y, tras la revisión:

> Prepara un dataset de parrafos, cumpliendo los requisitos que hallaste. Queremos unos
> 1000 parrafos. Cada parrafo debe identificar la posicion de sus borderes, del recuadro
> que lo encierra. Prueba distintos tamaños, distintos interlineados, etc. No probemos
> fondos sucios por ahora, solo un papel en blanco. Verifica que haya suficiente espacio
> entre párrafos para aplicar un kernel de 19 px y que la imagen resultante aún vea algo
> del borde. Este 19 px es un kernel maximo, vamos a obtener otros menores. Calcula
> tambien q estas paginas serán recortadas para hacer entrenamientos sobre menos pixeles
> (dependiendo del kernel que se quiere entrenar). Por ahora solo obtén las páginas.
> Descarta las q no cumplan algún requisito. Es importante que los párrafos nunca se
> solapen entre sí, y que dejen suficiente espacio entre ellos. Las ventanas de
> entrenamiento serán recortadas posteriormente. Guarda el dataset en el repo de data.

## Cómo se leyó cada frase, y en qué se convirtió

| lo que pidió | en qué se convirtió | dónde vive |
|---|---|---|
| «unos 1000 párrafos» | **exactamente 1000**, repartidos en páginas de 2 a 5 | `plan()` |
| «la posición de sus bordes, del recuadro que lo encierra» | 4 enteros `izq, der, sup, inf` en píxeles de la página, sobre la **caja de tinta** | `cajas.npz` |
| «distintos tamaños, distintos interlineados, etc.» | 7 factores: fuente (4), cuerpo, interlineado, ancho, alto, posición, nº de palabras | `factores()` |
| «solo un papel en blanco» | fondo `solid #ffffff`, texto `#000000`. **Ningún** fondo sucio | `receta.json` |
| «suficiente espacio … kernel de 19 px» | separación **L∞ ≥ 19 px** entre tintas, construida a 40 | `geometria.py` |
| «que la imagen resultante aún vea algo del borde» | margen al papel **≥ 9 px**, construido a 32 | `geometria.py` |
| «19 px es un kernel máximo» | `K_MAX = 19`, y **todas** las constantes se derivan de él | `geometria.py` |
| «serán recortadas … sobre menos píxeles» | `ventana_limpia` por párrafo + reparto **por página** | `cajas.npz`, `_particion()` |
| «descarta las que no cumplan» | `revisar()` sobre la tinta real; la página que falle se tira entera | `generar_paginas.py` |
| «nunca se solapen … suficiente espacio entre ellos» | garantizado **por construcción**, no por filtrado | `celdas()` + `ranura()` |
| «guarda el dataset en el repo de data» | publicado en `foveal-vision-data/experimentos-cnn/` | `--publicar` |

## Los requisitos que la revisión encontró (la primera petición)

Hay **dos** formas de «afectar» a `banco-k`, y piden cosas distintas:

**A. No pisar el dato ya medido** — ya tiene freno y no hubo que hacer nada especial:
dato nuevo = nombre nuevo, `--publicar` se niega a sobrescribir, y el `.npz` nunca se copia
al repo público.

**B. No contaminar la comparación — la reserva del §3.7.** Éste es el requisito real y el
que se rompe en silencio: `LiberationMono` y el interlineado `[1.45, 1.60]` son de uso
**exclusivo** del banco, porque un kernel aprendido sobre datos que los alcancen tiene fuga
de distribución *aunque las muestras sean distintas*.

⚠ **Y omitir la declaración es filtrar, no es neutro.** El chequeo que hace el banco
(`banco-k/nn/importar_kernel.py:60`) lee `nn/receta.json` y falla del lado prudente en los
tres casos: sin fichero, sin `fonts` (el generador sortea **todas** las familias, incluida
la reservada) y sin `line_height` (su defecto `Range(1.15, 1.6)` **solapa** con la banda).
Los tres kernels `esqk-*` que hoy están en el banco llevan `usa_la_reserva: true` por
exactamente eso.

## Lo que se comprobó antes de escribir nada (2026-09-09)

Cinco suposiciones del diseño, medidas contra el generador real en vez de supuestas:

| | resultado |
|---|---|
| ¿el registro de fuentes tiene 6 familias? | **No: 5.** `assets/fonts/` trae 5 `.ttf` y el `_LINUX_FALLBACK` de 6 sólo se usa si esa carpeta está **vacía**. Quitando la reservada quedan **4** |
| ¿`avoid_overlap` garantiza que no se solapen? | **No.** `resolver.py:394`: *«a preference, not a hard constraint»*; `SAMPLE_FORMAT.md` §5: ~1 % de solape y *«se conserva el reparto menos malo en vez de fallar»* |
| ¿mover un párrafo cambia su tinta? | **No**: `dw = dh = 0.00` moviéndolo `(+37, +53)`. **De esto depende que el doble pase sea exacto** |
| ¿un bloque apilado sobre otro mide bien? | **Sí**, idéntico a medirlo separado. Por eso el pase A es **un** render por página y no uno por párrafo |
| ¿el lienzo recorta la etiqueta? | **No**: un párrafo de 764 px reporta 764 px en un lienzo de 300 |

Y dos medidas que fijan el predictor de altura:

- **el paso de línea es exactamente `cuerpo × interlineado`** (12×1,25 → 15,00 px;
  18×1,25 → 22,50; 28×1,25 → 35,00);
- **`C = palabras_por_línea · cuerpo / ancho` es estable por fuente** (3 anchos × 4 fuentes):
  `DejaVuSans` 0,283 · `DejaVuSerif` 0,276 · `LiberationSans` 0,324 · `LiberationSerif` 0,361.

## Lo que NO se hizo, y por qué

- **No hay red, ni criterio, ni semillas.** *«Por ahora solo obtén las páginas»*. El
  criterio escrito antes de mirar (R13) se escribirá cuando haya algo que medir; escribirlo
  ahora sería rellenarlo.
- **No se varió el gris del texto.** El encargo acota a papel limpio. Es el mando obvio
  para un dataset siguiente, y está anotado en `REGLAS.md` como decisión, no como olvido.
- **No se recortaron ventanas.** El encargo las deja para después. Lo que sí se hizo es
  **calcular su presupuesto** (`ventana_limpia`) y **repartir por página** para que ese
  recorte no pueda filtrar entre `train` y `eval`.

## Una decisión que el encargo no pedía y se tomó igual

**El reparto `train`/`val`/`eval` es POR PÁGINA.** No se pidió, y se añade porque su
ausencia es una trampa: dos ventanas recortadas de la **misma** página comparten fuente,
fondo y vecinos, así que repartirlas entre `train` y `eval` sería una fuga que no falla por
ningún lado y que sólo se ve como un resultado demasiado bueno. Repartiendo páginas,
cualquier recorte posterior hereda el reparto.

Es una columna en el `.npz`: si el dueño prefiere otro reparto, se ignora y se hace otro.
