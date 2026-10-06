# Criterio — escrito el 2026-10-06 (antes de 01:46 UTC; commit `444fdc1`), ANTES de entrenar

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

## Enmienda antes de lanzar (02:10 UTC) — a raíz del revisor; sigue sin existir ningún detector

H2 y H3, tal como estaban, cambiaban **dos** cosas a la vez (el vocabulario y la vista: cortas en `norm3` contra largas en
`nada`), y feat-fallos ya midió que `norm3` solo le cuesta −0,074 a las largas. Se miden ahora **en la misma vista**:

- **H2** (por vista): cortas contra las 13 largas **en la misma vista** —`nada` contra 0,949; `norm3` contra las largas en
  `norm3`; `nada+norm3` contra las largas en las dos vistas—. **Gana** si +0,01 o más, **empata** dentro de ±0,01,
  **pierde** si no. El veredicto principal sigue siendo el de `norm3`.
- **H3:** cortas (`norm3`) + 13 largas (crudas) **≥ largas en las dos vistas + 0,01** (feat-fallos midió 0,956 → **≥ 0,966**):
  si no supera a lo que las largas solas ya dan con dos vistas, las cortas no aportan.
- H1 queda como está, pero la prueba que de verdad dice algo es **H1-cv**.

## Resultado (02:42 UTC) y un diagnóstico escrito ANTES de medirlo (02:47 UTC)

H1 ✅ (8/8) · **H1-cv ✅** (curva sobre rectas cortas 0,012; recta sobre curvas 0,015: se distinguen, contra lo que esperaba)
· **H2 pierde en las tres vistas** (crudo 0,880 contra 0,949; 3 px 0,837 contra 0,876; las dos 0,915 contra 0,956) · **H3 ❌**
por poco (0,9606 contra 0,9663; aun así, la mejor combinación medida hasta hoy).

**¿Por qué pierden?** Dos sospechas, y una prueba que las separa sin entrenar nada:

1. **el LARGO** — un detector corto sólo ve un trozo y el compositor lineal no puede componer trozos; o
2. **el VOCABULARIO** — a las 8 cortas les faltan las 5 features que no son trazos sueltos: el **lazo** (0, 6, 8, 9) y las
   cuatro **esquinas** (4, 5, 7). Un compositor lineal no puede construir un lazo sumando arcos en posiciones fijas.

Ya medido (`nn/firma.py` y una prueba con rectas dibujadas a mano): la recta corta **sí** se enciende sobre rectas largas
(su puntuación baja de ~0,95 a ~0,8 pasados los 16 px), así que el largo, si cuenta, no es por no encenderse.

**La prueba (`nn/vocabulario.py`, compositor de 180, crudo):** las 8 largas que son trazos (4 arcos + 4 rectas) contra las 8
cortas, y las 8 cortas **más** el lazo y las 4 esquinas de las largas contra las 13 largas.
- si **cortas + lazo + esquinas ≥ 13 largas − 0,01** → es el **vocabulario**: lo corto vale tanto como lo largo;
- si no, y **8 largas-trazo > 8 cortas + 0,02** → es el **largo**;
- si ninguna de las dos, no lo separa y se dice.
