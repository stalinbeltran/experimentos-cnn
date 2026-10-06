# Criterio — escrito el 2026-10-06 (01:50 UTC), ANTES de entrenar

No existe ningún detector de este experimento. Lo único mirado: la rejilla de muestras (`resultados/muestras-features.png`),
que enseña lo esperable: una curva corta y gruesa (radio 5, 4 px) se parece mucho a una recta corta.

## A. Por detector, val sintético — los umbrales de feat-ind32

«aprendió» F1 ≥ 0,90 y posición ≤ 1 celda ≥ 0,90 · «a medias» F1 ≥ 0,75 · si no, «no aprendió».

**H1:** ≥ 6 de los 8 detectores al menos «a medias».

**H1-cv — ¿se distingue una curva corta de una recta corta?** (la pregunta del dueño, aquí en su versión más difícil): la
tasa media de falsos positivos de los 4 detectores de curva sobre imágenes de rectas cortas, y de los 4 de recta sobre
curvas cortas (val; `fp_por_familia` de cada `summary.json`). **Se distinguen si las dos medias quedan ≤ 0,10.**

## B. Predecir los dígitos — el compositor posicional de feat-ind32 (8 mapas → 512 entradas)

Los 1797 de `windep` con su reparto 180/1617, 3 semillas; y la curva con 36/180/1080 sobre el test de 717. **La vista
principal es `norm3`** *(decido, ahora)*: los detectores se entrenan finos (2–4 px) y el dígito normalizado a 3 px es lo que
se parece a eso; `nada` (crudo) y `nada+norm3` (las dos) se reportan al lado.

Referencia (feat-ind32, las 13 features largas, crudo): **0,949**; curva **0,792 / 0,942 / 0,974**.

**H2:** cortas en `norm3` **≥ 0,949**: predicen al menos como las 13 features. Se dice que **ganan** si ≥ 0,959.
**H3:** cortas (`norm3`) **más** las 13 largas (crudas) en el mismo compositor (21 mapas): **≥ 0,959**.

## Lo que espero

- H1 sí en las rectas; las curvas, a medias (son cortas y casi rectas cuando el radio es 10).
- **H1-cv no**: con 60–100° de apertura y radio hasta 10, una curva corta tiene muy poca flecha; espero 0,10–0,25.
- H2 cerca: 0,93–0,95. Son primitivas locales y el compositor tiene que componerlo todo desde ellas.
- H3 sí (0,96–0,97): lo local y lo largo son informaciones distintas, como fino y grueso en feat-ind.

Si sale al revés, eso es un resultado y se escribe tal cual.
