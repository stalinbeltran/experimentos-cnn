# `dim-gen` — reducir las dimensiones de la imagen, ¿qué le hace a la generalización?

**Corrido el 2026-10-02 en Vast (01:49 → 06:25 UTC, 13 alquileres, 1,2023 $) y CERRADO.**
30 brazos: `W ∈ {128, 64, 32, 16, 8}` × 5 semillas, más el control `w128-de16` × 5. Todo con
el 10 % del dataset para entrenar (100 imágenes) y el 90 % para validar (900), `L = 4` capas
`valid` con kernel `n = W/4`, `C = 8`, 4000 pasos, `lr = 1e-3`, sin selección. El criterio se
escribió antes de entrenar ([`instrucciones/02-criterio.md`](instrucciones/02-criterio.md)) y
aquí se aplica tal cual; lo que el criterio no previó va aparte y marcado.

> **Veredicto en una línea:** con 100 imágenes, **la red generaliza mejor cuanto MENOS
> resolución tiene, hasta 16 px**: `W = 16` da el mejor IoU sobre las 900 no vistas
> (0,612), `W = 8` es indistinguible (0,579), y de ahí hacia arriba cae monótono —0,514 a 32,
> 0,395 a 64, 0,305 a 128—. Es la forma **(iii)** del criterio: **la resolución estorba**, y
> el control reparte la caída de 16 a 128 en **≈40 % parámetros y ≈60 % información**.

## Qué salió (media ± desviación entre 5 semillas; `nn/informe.py`, regenerado del disco)

| brazo | W | kernel | parámetros | IoU **val** (900) | IoU train (100) | brecha | clasificación del criterio |
|---|---:|---:|---:|---|---|---|---|
| `w128` | 128 | 32 | 205.348 | 0,305 ± 0,047 | 0,515 | +0,21 ± 0,26 | mixta: cae train **y** crece la brecha (⚠ 3 semillas atascadas, abajo) |
| `w064` | 64 | 16 | 51.748 | 0,395 ± 0,072 | 0,740 | +0,34 ± 0,18 | (b) peor generalización (⚠ 1 semilla atascada) |
| `w032` | 32 | 8 | 13.348 | 0,514 ± 0,061 | 0,850 | +0,34 ± 0,04 | (b) peor generalización |
| **`w016`** | 16 | 4 | 3.748 | **0,612 ± 0,049** | 0,723 | +0,11 ± 0,03 | **W\*** |
| `w008` | 8 | 2 | 1.348 | 0,579 ± 0,076 | 0,669 | +0,09 ± 0,02 | dentro del umbral de W\* |
| `w128-de16` (control) | 128 | 32 | 205.348 | 0,496 ± 0,047 | 0,817 | +0,32 ± 0,03 | parámetros de `w128`, información de `w016` |

Piso (caja media de train sobre las 900): **0,2464**. Umbral entre dos brazos:
`max(2·SE_dif, 0,01)`. Todos los brazos superan el piso en más de 2·SE. Figuras:
[`resultados/iou-vs-w.png`](resultados/iou-vs-w.png) y
[`resultados/brecha-vs-w.png`](resultados/brecha-vs-w.png); el informe entero,
[`resultados/RESULTADOS.md`](resultados/RESULTADOS.md).

### Las dos preguntas del encargo, con el criterio escrito antes

1. **¿Hay un `W` mínimo suficiente?** `W* = 16`. Suficientes (a menos del umbral de W\*):
   `w016` y `w008`. **El `W` mínimo suficiente es 8 px** — con un kernel de 2×2 y 1.348
   parámetros, 152 veces menos que `w128`, y 0,04 s por paso contra 5,2 en el dev.
2. **¿La resolución estorba?** **SÍ**: `w128` queda 0,31 por debajo de W\*, muy por encima del
   umbral (0,05). **Y la brecha crece con la resolución**: de +0,09 a 8 px a +0,21 a 128
   (+0,49 en las semillas que entrenaron), diferencia mayor que el umbral.

Forma de la curva: **(iii) máximo interior**. No es «no hubo señal»: la amplitud de la curva
(0,31 de IoU) es seis veces el umbral.

### El control `w128-de16`: parámetros contra información

El control tiene **los 205.348 parámetros de `w128` y la información de `w016`** (la imagen de
16 px repetida 8×8). Da **0,496**: ni como `w016` (0,612) ni como `w128` (0,305). Fracción del
camino de 16 a 128 que recorre: **0,38** (0,44 si se compara con las semillas de `w128` que
entrenaron). Se lee así, y **no se redondea** a ninguna de las dos:

- **Darle a la red grande sólo 16 px de información le cuesta 0,116 de IoU** respecto de
  `w016`: eso es **capacidad** (c). Y su brecha es +0,21 mayor que la de `w016` (umbral 0,06):
  **más parámetros a igual información memorizan más**.
- **Darle además los 128 px de verdad le cuesta otros 0,19** (0,15 contra las semillas
  entrenadas): eso es la **información fina** (b). A igualdad de red, los píxeles de más
  **empeoran** la generalización en esta tarea con 100 imágenes.

Esto es lo que `ESTADO.md` del repo central deja «sin cerrar (confundido)» para `border_reduce`
en `foveal-vision`: aquí, con el control, los dos efectos **se separan y los dos existen**.

## ⚠ Lo que el criterio NO previó: semillas que no entrenaron

**`w128`: 3 de 5 semillas (2, 4 y 5) se atascaron; `w064`: 1 de 5 (la 2).** Su pérdida de
train se quedó en **0,137** desde las primeras épocas —la pérdida de predecir una caja
constante— y su IoU en train (0,30) es el del piso: la red **no aprendió nada**, no es que
aprendiera y no generalizara. Las 5 semillas del control (misma red que `w128`, otra entrada)
sí entrenaron, y las 15 de `w ≤ 32`, también. Tabla por semilla:

| brazo | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|
| `w128` IoU val · pérdida final | 0,385 · 0,018 | **0,274 · 0,137** | 0,313 · 0,014 | **0,279 · 0,137** | **0,274 · 0,137** |
| `w064` | 0,437 · 0,013 | **0,270 · 0,136** | 0,399 · 0,010 | 0,431 · 0,011 | 0,439 · 0,010 |
| `w032` | 0,486 · 0,009 | 0,571 · 0,013 | 0,564 · 0,011 | 0,423 · 0,022 | 0,524 · 0,012 |
| `w016` | 0,596 · 0,033 | 0,643 · 0,032 | 0,544 · 0,047 | 0,601 · 0,026 | 0,674 · 0,020 |
| `w008` | 0,446 · 0,076 | 0,609 · 0,034 | 0,617 · 0,033 | 0,631 · 0,035 | 0,595 · 0,036 |
| `w128-de16` | 0,540 · 0,014 | 0,427 · 0,020 | 0,517 · 0,014 | 0,527 · 0,014 | 0,468 · 0,015 |

**Qué cambia y qué no**, mirado *después* y por tanto **sin peso en el veredicto**:

- **El veredicto no cambia.** Sólo con las semillas que entrenaron, `w128` da 0,349 ± 0,051
  (n = 2) y `w064` 0,426 ± 0,019 (n = 4): siguen por debajo de `w032` (0,514) y muy por debajo de
  `w016` (0,612). El orden de la curva es el mismo y «estorba» sigue siendo sí.
- **La clasificación de `w128` sí cambia de lectura.** El criterio la llama «mixta» porque
  **cae train**; pero ese «cae train» no es falta de información (a 128 px la hay de sobra),
  es **que no entrenó**. Con las dos semillas que sí lo hicieron, train es 0,84 y la brecha
  +0,49: sería **(b)**, como `w064` y `w032`. El criterio asumió que una caída de train sólo
  podía venir del dato; **puede venir del optimizador**, y eso hay que escribirlo en el
  próximo criterio de este tipo.
- **Es un resultado en sí**: con `lr = 1e-3`, inicialización por defecto y ReLU, **los kernels
  de 32×32 son frágiles de entrenar** (3 de 5 colapsan a una constante) y los de 16×16 ya lo
  insinúan (1 de 5). El ensayo de mecanismo (`W = 128`, semilla 1, 10 épocas) no podía verlo:
  justo la semilla 1 entrena. Un `lr` por `W` habría sido otro confound; el precio de no
  tenerlo es éste, y queda medido.

## ¿Bastaron los 4000 pasos?

En lo esencial, sí. En las semillas que entrenaron, la pérdida de train entre las épocas 700 y
800 baja un 2–4 % en `W = 8` y `16` (0,0352 → 0,0338; 0,0339 → 0,0332) y ya oscila sin
tendencia en `W ≥ 32`. El IoU de validación de la semilla 1 a las épocas 200/400/600/800 va
0,52/0,58/0,62/0,60 en `w016` y 0,37/0,39/0,38/0,38 en `w128`: la curva está asentada mucho
antes del final. No se alarga nada.

## Por factor del dato (IoU val, media entre semillas)

- **Fuente**: todas dentro de ±0,03 en cada brazo; `LiberationMono` es siempre la peor
  (−0,03 a −0,04 respecto de la mejor). No hay una fuente que explique la curva.
- **Área de la caja** (cuartiles): las cajas **pequeñas** son las difíciles en todos los `W`
  —`w016`: q1 0,43 · q4 0,77; `w128`: q1 0,15 · q4 0,45—, y la ventaja de `w016` sobre `w128`
  es parecida en los cuatro cuartiles (+0,28 a +0,32).

## Lo que este estudio NO dice

- **La tarea es localizar un bloque de tinta**, que aguanta el desenfoque (§2.4 de
  `ESPECIFICACION.md`); de tareas que necesiten el trazo no dice nada.
- **El techo es 128 px**: el dato publicado ya es /4 del render.
- **`lr` único para todos los `W`** (decisión de diseño): la fragilidad de `w128`/`w064` es
  parte del resultado y también un límite de él. Separar «no entrena» de «no generaliza» a
  `W ≥ 64` pide repetir esos dos brazos con un `lr` o una inicialización que entrene las cinco
  semillas (p. ej. `3e-4`, que en el ensayo también bajaba, o `LeakyReLU`): **no está hecho**.
- **El control cruzó máquinas** (una por semilla, distinta de la de `w128`): la deriva medida
  entre máquinas es 0,0012 de IoU, cien veces menor que la diferencia que lee.
- **Nada se ha aplicado** a ninguna red de producción.

## Coste y reloj (del libro, `resultados/vast/todo/`)

| máquina | trabajo | alquilada → destruida (UTC) | min | $/h | vCPU · CPU | $ |
|---|---|---|---:|---:|---|---:|
| `s1` (2.º intento) | `w008…w128` s1 | 02:51:00 → 05:37:30 | 166 | 0,0516 | 12 · Xeon E5-2673 v3 | 0,1431 |
| `s2` | s2 | 01:49:02 → 03:51:14 | 122 | 0,0516 | 12 · E5-2673 v3 | 0,1050 |
| `s3` | s3 | 01:49:02 → 02:51:17 | 62 | 0,0528 | 14 · E5-2680 v4 | 0,0547 |
| `s4` (3.º intento) | s4 | 03:52:46 → 06:24:49 | 152 | 0,0516 | 12 · E5-2673 v3 | 0,1307 |
| `s5` | s5 | 01:49:03 → 04:44:46 | 176 | 0,0551 | 18 · E5-2686 v4 | 0,1614 |
| `c1` | `w128-de16` s1 | 01:49:03 → 03:21:08 | 92 | 0,0556 | 12 · Core i7-5820K | 0,0853 |
| `c2` | s2 | 01:49:04 → 03:35:49 | 107 | 0,0570 | 18 · E5-2686 v4 | 0,1015 |
| `c3` | s3 | 01:49:03 → 03:28:32 | 100 | 0,0578 | 14 · E5-2680 v4 | 0,0958 |
| `c4` | s4 | 01:49:03 → 03:43:46 | 115 | 0,0582 | 14 · E5-2680 v4 | 0,1113 |
| `c5` | s5 | 01:49:03 → 04:53:34 | 185 | 0,0649 | 12 · E5-2673 v3 | 0,1995 |
| 3 intentos fallidos (`s1` ×1, `s4` ×2) | — | 5 min cada uno | 16 | — | sshd sin responder ×2, clave rechazada ×1 | 0,0140 |

**Total: 1,2023 $ reales, 13 alquileres (10 trabajaron), 4 h 36 min de reloj** (01:49:00 →
06:24:49). Por brazo, en las máquinas: `w128` 48–140 min, `w128-de16` 88–178 min, `w064`
7–22 min, `W ≤ 32` menos de 7 min. El ensayo estimaba «≈2 h de reloj y 0,4–0,55 $ sin
control»: el control y los dos reintentos lo llevaron a 4,6 h y 1,20 $.

## Dónde está todo

- **Aquí** (repo público, rama `tema-2` hasta que se fusione): `nn/pesos/<brazo>-s<N>/`
  (`metrics.jsonl`, `summary.json`, `log.txt`; **sin** `.pt`), `resultados/` (informe, figuras,
  `criterio-aplicado.json`, el libro de Vast).
- **En el volumen** (`foveal-vision-data`, rama `tema-2`, empujado al almacén el 2026-10-02
  06:25 UTC): `experimentos-cnn-resultados/dim-gen/` con los **30 `last.pt`**, las métricas, el
  informe, el libro, los logs de las unidades y el descriptor. 13 MB.
- Regenerar el informe: `../.venv/bin/python nn/informe.py` (lee los `summary.json`).
- Reporte en el repo central: `estudios-redes-neuronales/reportes/estudios/2026/10-octubre/2026-10-02-dim-gen-resolucion-generalizacion.md`.
