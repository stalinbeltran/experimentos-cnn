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
