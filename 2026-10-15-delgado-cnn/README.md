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
