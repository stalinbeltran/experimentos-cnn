# Encargo (el dueño, 2026-10-13, conversación de Telegram)

> «Anota otro experimento (si no está ya realizado): crear una CNN que obtenga la forma delgada del dígito. Lo
> entrenamos con trazos sintéticos. Esto podría hacer aparecer agujeros adicionales, aclarar curvas, etc. Los trazos
> resultantes deben ser suaves, pues binarizarlos obliga a desechar información.»

Lo que el encargo da por sabido, escrito aquí:

1. **No está realizado.** `dig-delg` (2026-10-10) buscó el dígito delgado con respuestas de Gabor (valle firmado,
   cresta par, fase local) y un esqueleto morfológico como referencia: **ninguno pasó** sus cuatro umbrales. No probó
   una red entrenada, ni sintéticos. Este experimento es esa vía.
2. **«Forma delgada»**: la línea media del trazo (≈ 1 px de ancho, centrada en la tinta), la misma noción que en
   `dig-delg` y la que `trazos` necesita como primera etapa.
3. **«Suave, sin binarizar»**: la salida es un **mapa continuo** en [0, 1] por píxel —la verdad de entrenamiento es una
   campana de σ ≈ 0,8 px alrededor de la línea media real—, y todo lo que se mida sobre ella se mide **sin umbral**
   (cresta del mapa, integral, fase) o con **varios umbrales** a la vez, para que ningún número dependa de uno.
   Binarizar sería tirar justo lo que el dueño pide conservar: cuán seguro está la red de que ahí pasa un trazo, y por
   dónde exactamente entre dos píxeles.
4. **«Hacer aparecer agujeros, aclarar curvas»**: la hipótesis es que una red que ha visto miles de trazos gruesos con
   su línea media sabe que un lazo de tinta maciza (el lazo de abajo de un 8 con el hueco de 1–2 px, idx 890/898 de
   val) tiene dentro un hueco, y lo abre; y que donde dos trazos se funden (el cruce del 8, el cuello del 6) devuelve
   dos líneas que se cruzan y no un rombo. Son dos cosas medibles: topología (número de agujeros) y cruces.
5. **Sólo sintéticos para entrenar**, como la estrategia C de `rizo-2`: los dígitos reales son únicamente prueba. Así
   el experimento también mide cuánto se parece lo sintético a lo real.
6. **Gasto previsto: 0 $**: red de miles de parámetros a 32×32, se entrena en minutos en este dev (2 vCPU). Si un
   brazo pasara de 30 min, se reconsidera.
