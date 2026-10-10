# Reglas de `dig-delg`

**Qué pregunta:** ¿se puede obtener el mismo dígito pero delgado (trazo de ~1 px, centrado, sin perder ni añadir lazos)
aprovechando las respuestas de un banco de Gabor impar —el borde gris de la figura 19 del boceto curvas-gabor— en vez
de un esqueleto morfológico, y qué alternativa lo hace mejor?

## Entradas

- **Dataset:** `uci-optdigits-orig-32px-r20261005`, leído con `expcnn.exigir_dataset(...)`; no se regenera.
- **Qué se lee de él:** sólo las imágenes de los **64 dígitos** de `entradas/digitos-64.json` (índices sobre el `.npz`
  entero; son de la partición val) y sus etiquetas, para rotular las figuras. No se entrena nada.
- **Condiciones que el dataset ya trae:** 32×32, tinta = valor > 0 (se binariza así al cargar); los dígitos ocupan casi
  los 32 px de alto (sin margen vertical).
- **Qué se transforma al cargar:** el dígito binario (0/1, float32) pasa por los bancos de Gabor de cada brazo. Nada más.
- **El banco impar es el de la figura 19, copiado** (no importado) del boceto: kernel 9×9, λ 10, 8 orientaciones, seno
  con la misma envolvente que el par, norma 1. Aquí se usa **con signo**: la imagen de la figura 19 es su |·| máximo.

## Salidas

- **Pesos:** ninguno (no se aprende nada).
- **Métricas:** `resultados/metricas.json` — por brazo, las cuatro del criterio (media y por dígito).
- **Figuras:** `resultados/1-comparacion.png` (16 de los 64 con todos los brazos), `resultados/2-<A|B8|B12|C|D>.png`
  (los 64 con cada brazo) y `resultados/3-metricas.png`; se regeneran con `python nn/delgado.py`.
- **Tablas / informe:** en el `README.md` de esta carpeta.
- **Qué se commitea:** todo lo de `resultados/` (pesa poco). El dato de entrada no.

## Procesos

1. Cargar los 64 dígitos.
2. Para cada brazo (A, B · λ_p 8, B · λ_p 12, C, y la referencia D), obtener el dígito delgado.
3. Medir las cuatro métricas y dibujar.

- **Qué se mide y con qué umbral:** delgadez ≥ 85 %, dentro de la tinta ≥ 95 %, cobertura ≥ 95 %, topología igual en
  ≥ 90 % de los dígitos. Completo en `instrucciones/02-criterio.md`, escrito antes de mirar.
- **Cuántos brazos y cuántas semillas:** 4 brazos de Gabor + 1 referencia. Sin semillas: todo es determinista.
- **Qué se llama «ganar»:** no se declara ganador; se reportan los que pasan los cuatro umbrales y elige el dueño.

## Desplazamientos

- **Qué se reconoce y a qué nivel:** **nada**. El experimento transforma un dígito en otro (el delgado); no clasifica ni
  detecta. Por eso no hay curva de acierto contra desplazamiento. Y la transformación es, por construcción, equivariante a
  desplazamientos enteros: convolución + comparaciones entre vecinos, sin rejilla fija ni max-pool (regla 6 de la regla
  del dueño: una curva plana por construcción se dice, no se mide). Si el delgado se usa luego para reconocer, la curva
  va en ese experimento.

## Qué NO hereda

- **Se copió de:** ningún experimento. Del boceto `curvas-gabor` se toman tres cosas, a propósito y nada más: los 64
  dígitos (la lista), el banco de Gabor impar de su figura 19 (el código, copiado) y la regla del dueño de no usar
  morfología para el borde (como espíritu, para el adelgazamiento).
- **Qué NO aplica aquí del boceto:** el detector de curvas (giro κ, trozos, orientación), su calibración (TAU, σE,
  coherencia), el compositor, las 3 semillas, la partición train/val/a ciegas y el acierto como métrica. Aquí no se
  clasifica: se transforma el dígito y se mide la forma que sale.
- **Contra qué se compara:** el esqueleto morfológico (referencia D), que es el método clásico para lo mismo.

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/delgado.py` | los cinco brazos sobre los 64 dígitos, las métricas y las dos figuras | `python nn/delgado.py` (desde la carpeta del experimento o desde cualquier sitio) |

- **Dependencias:** el `.venv` del repo con numpy, scipy, torch, matplotlib y **scikit-image** (sólo para la referencia
  D y para medir).
- **De dónde sale el código:** el banco de Gabor impar y par se copian de `docs/bocetos/2026-10-09-curvas-gabor/`
  (`bordes_gabor.kernel_impar`, `curvas.kernel_gabor`). Lo demás es de aquí.
