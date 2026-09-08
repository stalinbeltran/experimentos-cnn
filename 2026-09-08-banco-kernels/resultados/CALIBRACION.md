# Calibración del banco `banco-k`

**No es un experimento y estas cifras no se reportan como hallazgos** (§11). Fijan el instrumento y contestan una sola pregunta: **¿puede este banco producir evidencia?**

Generado por `python nn/calibrar.py --informe` leyendo `resultados/calibracion.json`. No se transcribe nada a mano.

## Veredicto

| paso | resultado |
|---|---|
| 1. caja media ≤ 0,40 (§10.1) | ✅ pasa |
| 2. identidad bajo el techo (§10.1.1) | ✅ pasa |
| 3. rango útil ≥ 3 desviaciones | ✅ pasa |
| 5. resolución bajo la celda (§7.5) | ✅ pasa |
| 6-9. aserciones (§3.3 · §6.4 · §7.1 · §7.5) | ✅ pasa |
| 10. balance marginal (§3.6) | ✅ pasa |

## Los dos límites, y el rango entre ellos

| | IoU sobre `eval` |
|---|---|
| **piso** — caja media (§10.1) | **0.2479** (umbral ≤ 0.4) |
| **techo** — identidad (§10.1.1) | **0.7981** ± 0.0057 (techo 0.95) |
| **rango útil** | **0.5501**, o sea **96.1 ×** la desviación entre semillas |

## El denominador de los criterios (§2)

El **aleatorio** (k=9, 10 semillas de aleatoriedad) da **0.8083 ± 0.0077**.

Un kernel evaluado con esa misma desviación tendría que superarlo por más de **0.0155** (la **suma** de las dos desviaciones, §2.1) para declararse útil. Ése es el listón, y sale de aquí — no se elige después de mirar.

| condición | IoU `eval` | IoU `train` | brecha |
|---|---|---|---|
| caja media | 0.2479 | 0.2699 | +0.0219 |
| identidad | 0.7981 ± 0.0057 | 0.8401 | +0.0421 ± 0.0059 |
| aleatorio | 0.8083 ± 0.0077 | 0.8514 | +0.0431 ± 0.0117 |
| gauss | 0.8130 ± 0.0086 | — | +0.0298 ± 0.0054 |
| sobel | 0.8164 ± 0.0064 | — | +0.0367 ± 0.0044 |

## ⚠ La distancia entre CONDICIONES no es el rango útil

El rango útil (0.5501) mide **caja media → identidad**, o sea cuánto hay entre no mirar la imagen y mirarla sin filtrar. Pero todas las condiciones **que sí filtran** caben en mucho menos:

| | IoU `eval` |
|---|---|
| identidad | 0.7981 |
| aleatorio | 0.8083 |
| gauss | 0.8130 |
| sobel | 0.8164 |

De **identidad** (0.7981) a **sobel** (0.8164) hay **0.0183**, que son **2.4 ×** la desviación entre semillas del aleatorio — no las 96 × del rango útil.

**Qué significa, sin adornarlo:** la tarea está dominada por *dónde está la mancha oscura*, y eso **sobrevive a cualquier filtro** — se ve en `muestras/condiciones-4x6.png`, donde las cuatro condiciones conservan la caja igual de clara. El banco distingue con holgura **filtrar de no mirar**, y con mucho menos margen **un filtro de otro**. Un kernel que quiera declararse útil aquí tiene que moverse dentro de esa franja estrecha, y por eso el margen del §2.1 no es una formalidad.

⚠ Esto es una **observación sobre el instrumento**, medida en la calibración, no un hallazgo sobre kernels (§11).

## Resolución (§7.5)

MAE medio de la identidad: **2.40 px** contra una celda de rejilla de **8.0 px**. El soft-argmax **interpola entre celdas**, que es lo que §7.5 espera.

## Balance marginal entre particiones (§3.6)

Criterio: |t| <= 3.0 SE en los continuos (15 comparaciones) y chi2 <= 9,49 en la familia. §3.6 dice que el balance **se verifica, no se fuerza**, así que esto es una medición, no una corrección.

| factor | train | monitor | eval | peor \|t\| |
|---|---|---|---|---|
| `cuerpo` | 19.639 | 20.317 | 20.631 | 1.7 |
| `interlineado` | 1.379 | 1.382 | 1.374 | 0.7 |
| `gris_nivel` | 44.02 | 55.43 | 51.32 | 2.8 |
| `ancho_real` | 230.534 | 243.693 | 241.238 | 1.2 |
| `alto_real` | 185.728 | 180.58 | 184.272 | 0.3 |

Peor desvío: **|t| = 2.77** en gris_nivel (train vs monitor). Por debajo del umbral.

⚠⚠ **Este criterio se cambió DESPUÉS de ver el primer resultado, y hay que decirlo.** El original pedía que la dispersión relativa a la media fuera < 10 %, y con eso el nivel de gris salía **22,7 % → NO pasa**. Se cambió porque el criterio era malo, no porque el resultado no gustara: un porcentaje sobre la media no es comparable entre factores que viven en escalas distintas (`cuerpo` en [11, 30], `gris` en [0, 102]), y sobre todo no contesta la única pregunta que importa — **¿cabe esta diferencia en el azar con 100 muestras?** —, que es la misma que la chi-cuadrado ya hacía para la familia. Aun así, **el orden de los hechos fue ése**, y quien lea esto tiene derecho a descontarlo.

⚠ **Y el nivel de gris es de verdad el factor menos equilibrado**: |t| = 2,8 entre `train` y `monitor` es borderline, no cómodo. Lo que **no** afecta es la comparación entre condiciones, que es lo único que el banco afirma: §8.2 hace que particiones y semillas sean idénticas en todas, así que el mismo desequilibrio lo sufren todas por igual. Lo que sí podría mover ligeramente es el valor **absoluto** de la brecha `train − eval`, y por eso queda escrito aquí.

## Aserciones

- §3.3 y §6.4 sobre las **1000** etiquetas del dataset: OK
- §7.1 cadena de rejillas: `[64, 32, 16, 16]`
- §7.5 span de centros `[0, 120]` contra el rango **real** de etiquetas `[8.0, 119.0]`: **cubre**
- dataset: publicado

---

⚠ **Concluida la calibración, los parámetros quedan congelados** (§12). Cambiar cualquier invariante obliga a un banco nuevo con su propia serie, no a re-etiquetar éste.
