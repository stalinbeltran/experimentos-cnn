# `delg-cnn` — la forma DELGADA y SUAVE del dígito con una CNN entrenada con trazos sintéticos

**Estado: anotado el 2026-10-13, NO ejecutado.** Encargo en `instrucciones/01-encargo.md`, criterio (11 umbrales) en
`instrucciones/02-criterio.md`, reglas en `REGLAS.md`. Antecedente: `dig-delg` (2026-10-10) buscó lo mismo con
respuestas de Gabor y un esqueleto morfológico y **ninguna alternativa pasó**; no probó red ni sintéticos.

## La idea, en una frase

Una red pequeña ve miles de trazos sintéticos gruesos con su línea media conocida y aprende a devolver, para un dígito
real, un **mapa continuo** de por dónde pasa el bolígrafo — sin binarizar, porque el valor dice cuán seguro está y por
dónde exactamente entre dos píxeles —, y de paso **abre los lazos que la tinta tiene macizos** y deja los cruces como
dos líneas que se cruzan.

## Orden de trabajo

1. `nn/sinteticos.py` (copiar `arco`/`recta` de `rizo-2`; 20 % de lazos macizos con hueco en la verdad).
2. `nn/entrenar_local.py --brazo A|B|C` (minutos cada uno, 0 $).
3. `nn/evaluar.py`: A–D, figuras, `metricas.json`, y este README con lo medido contra la predicción.

## Predicción (escrita antes de correrlo, 2026-10-13)

- B (sin reducción) pasa posición y delgadez en sintéticos y es plano en desplazamientos; A y C ven más contexto y
  abren mejor los lazos macizos, pero pueden no ser equivariantes (criterio 11).
- En los 64 reales, el mejor brazo pasa delgadez y cobertura y **se queda cerca** en topología (85–90 %): el cruce
  grueso del 8 es el fallo típico, como en `dig-delg` y como se prevé en `trazos`.
- En los 8 de val, el % con 2 agujeros sube de 78 % a 85–90 %, con ≤ 5 % de 6 y 9 con un agujero inventado.
- Metido en el compositor «todo objetos», val no baja (0,9775) y quizá suba 0,2–0,5 por los 8.

## Relación con `trazos`

`trazos` necesita una línea media como primera etapa y la toma de la transformada de distancia; si esto sale, su
etapa 1 pasa a ser la cresta de `M`, que ya trae resueltos los cruces y los lazos macizos. Por eso se anota ahora:
son dos mitades del mismo objeto (el trazo), y ésta es la que puede «ver» lo que la tinta tapa.

## ✅ Primer resultado (2026-10-13): la red de sólo kernels, y su efecto en el compositor «todo objetos»

Sin seguir aún el plan de brazos A/B/C de arriba, el dueño pidió empezar por lo mínimo: trazos a mano alzada
(`nn/sinteticos.py`: modelo de bolígrafo, grosor 3–6 px, sin lazos de varias vueltas, 30 % casi rectas) y una red de
**sólo kernels** (`nn/entrenar_local.py`: 6 conv 3×3 × 8 canales, ReLU, sigmoide; 2.489 parámetros; 300 trazos, 400
épocas, 109 s). Se conserva como `nn/pesos/k3x3/`; la `k5x5` (6.841 parámetros) sale más borrosa y se descartó.
Figuras: `resultados/0-curvas-muestra.png` (sintéticos), `1-kernels-300-k3x3.png` (vistos y nuevos),
`2-digitos-30-k3x3.png` (30 dígitos de val), `2b-val1327.png` (un 8: los dos huecos se agrandan en los tres umbrales),
`3-estructura-k3x3.png` (la red, sus kernels y sus activaciones).

**Medido en `rizo-2` (§ 40–41 de su README)** con el mapa continuo de los 5.620 dígitos (`nn/exportar_delgados.py`):
el compositor «todo objetos» con rectas y gris leyendo el mapa **sin binarizar** y un detector de **huecos suaves** da
**0,9781 val / 0,9715 a ciegas**, contra 0,9775 / 0,9592 sobre la tinta: +1,2 a ciegas. Binarizado, no aporta.
⚠ Los 11 criterios de `instrucciones/02-criterio.md` (posición, delgadez, topología en sintéticos; desplazamientos)
**siguen sin medirse**; los lazos macizos **no** están en los sintéticos (el dueño pidió dejarlos fuera por ahora).

## SUAVIZADO previo (2026-10-17, el dueño: «vamos a crear otro preproceso que los suavice»)

`python nn/exportar_delgados.py --suavizar σ [--gris]` → `resultados/delgados-k3x3-s<σ>[g].npz`: antes de la red, la tinta binaria
pasa por una gaussiana de σ px y se rebinariza a 0,5 (con `--gris`, no: fuera de lo que vio la red al entrenar). Sin la opción, el
resultado es idéntico al de siempre (comprobado ese día comparando M). Medido en rizo-2 § 113: con σ 1,5 los 1l con entrantes falsos
bajan del 57 al 19 %, a cambio de cerrar ~10 % de los lazos del 8 y el 9.

## Adelgazador para tinta GRIS: `k3x3-g1.5` (2026-10-17, el dueño: «usemos 1.5»)

`python nn/entrenar_local.py --n 300 --capas 6 --canales 8 --suavizar 1.5 --nombre k3x3-g1.5`: el k3x3 de siempre, entrenado con los
trazos sintéticos suavizados por la MISMA gaussiana que `exportar_delgados.py --suavizar 1.5 --gris` aplica a los dígitos. 111 s en
el dev (2 vCPU), pérdida en nuevos 0,1406 (k3x3: 0,1389). Delgados: `exportar_delgados.py --pesos k3x3-g1.5 --suavizar 1.5 --gris` →
`resultados/delgados-k3x3-g1.5-s1.5g.npz`. Las imágenes, en rizo-2 § 115.

## `k3x3-g0.7` y `--dataset` (2026-10-18): el adelgazador para el dataset de MISMO TRAZO

`python nn/entrenar_local.py --n 300 --capas 6 --canales 8 --suavizar 0.7 --nombre k3x3-g0.7` (σ 0,7 = el desenfoque del borde medido
en `uci-optdigits-orig-32px-trazo4-r20261018`); 136 s, pérdida en nuevos 0,1380. `exportar_delgados.py --pesos k3x3-g0.7 --dataset
<nombre>` lee cualquier dataset publicado (gris tal cual). Medido en rizo-2 § 119: los 1l con entrantes, del 57 % al 3 %.

## `k3x3-g0.7x-64` y `--ampliar` (2026-10-18): adelgazar a 64 px

`--dos 0.5` (sintéticos de dos trazos, T y X: `sinteticos.dos_trazos`) y `--ampliar 2` (entrenar y exportar a 64 px; la tinta ampliada
con `entrenar_local.ampliar`, la verdad calculada a 64 px). `python nn/entrenar_local.py --n 600 --capas 8 --canales 8 --suavizar 0.7 --dos 0.5
--ampliar 2 --nombre k3x3-g0.7x-64` (33 min, pérdida en nuevos 0,0862) y `exportar_delgados.py --pesos k3x3-g0.7x-64 --dataset <…> --ampliar 2`
(M de 64×64). A 32 px, entrenar con dos trazos (`k3x3-g0.7x`) NO recuperaba la barra del 7 (46 %); a 64 px, sí (89 %). Rizo-2 § 121–122.

## `afinar_retrazo.py` y `k3x3-g0.7x-64-rt` (2026-10-19): el FEEDBACK DEL RE-TRAZO

Ajuste fino de `k3x3-g0.7x-64` con una pérdida sobre los DÍGITOS REALES (sin línea media de verdad): la cresta suave de M, dilatada
con un disco del grosor del dígito, tiene que cubrir el núcleo de la tinta (`--inventado 0`: el término de trazo inventado enseñaba a
borrar la barra del 7). Sólo train + extra; val queda fuera. `python nn/afinar_retrazo.py --lambda 1 --inventado 0 --pasos 1500`
(1,3 s por paso en el dev). El criterio, las pruebas y el resultado, en rizo-2 § 125.

