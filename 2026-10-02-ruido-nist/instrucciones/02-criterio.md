# Criterio — escrito el 2026-10-02, ANTES de escribir código y de entrenar

## Medida

Por corrida, con `last.pt`: exactitud y entropía cruzada sobre las **1617 de validación limpias**,
y lo mismo sobre las 180 de train limpias. Por tipo de ruido y semilla:
**Δ = acc_val(ruido, s) − acc_val(limpio, s)** — pareado: mismos pesos iniciales, mismo orden de
lotes. Por tipo: media de los 3 Δ y `SE = sd/√3`.

## Base

`limpio` = las 180 originales **duplicadas** (360, sin ruido): mismo número de pasos e imágenes
que los escenarios con ruido. Lo que diferencia un escenario de la base es sólo el ruido de la
copia.

## Qué se declara

- **Ayuda**: `media(Δ) > max(2·SE, 0,01)` — al menos un punto de exactitud (16 imágenes de 1617).
- **Perjudica**: `media(Δ) < −max(2·SE, 0,01)`.
- **Indistinguible**: lo demás. Es un resultado, no un fallo.
- Se reporta también Δ en **entropía cruzada de val** (no satura) y la exactitud de **train
  limpio**: un ruido que «ayuda» bajando la exactitud de train y subiendo la de val es
  **regularización**; uno que sube las dos, **facilitación**.

## Fase 1 → fase 2

Pasan a la fase 2 los tipos **ayuda** e **indistinguible con media(Δ) > 0**. Si ninguno ayuda,
se dice: con 180 imágenes y esta red, ningún ruido de esta lista mejora la generalización.

## Lo que NO decide

- Ganador global entre fases: la fase 2 decide la intensidad dentro de cada tipo; no se combinan
  ruidos (sería otro estudio).
- Nada sobre otros datos ni otras redes.

## Enmiendas del 2026-10-02, al implementar y ANTES de que ninguna corrida entrenara

Lo único corrido hasta aquí es el ensayo de mecanismo (sólo pérdida de train) y pruebas de 2
épocas en directorios temporales. Nada de val se ha mirado.

1. **El mecanismo se lee en entropía cruzada de train, no en exactitud.** El ensayo dio exactitud
   de train **1,000** con el `lr` congelado: saturada, «baja la exactitud de train» no puede salir
   nunca. Con las 180 de train limpias: `Δce_train > umbral_ce` (ajusta **peor** el train) →
   **regularización**; `Δce_train < −umbral_ce` → **facilitación**; si no, «train sin cambio
   distinguible». `umbral_ce = max(2·SE, δ_ce)` con **`δ_ce = 0,05` nats**. La exactitud de train
   se reporta igual.
2. **Los niveles son cinco por tipo y el medio es el índice 2** (`nn/ruido.py`); el de
   `sal-pimienta`, que no existía, es **0,1**. Los medios del resto no cambian.
3. **El tipo repetido con otra copia es `oblicua@0.6-r2`.** Su regla: `Δ_r = acc_val(r2, s) −
   acc_val(r1, s)` pareado por semilla; si `|media(Δ_r)| > max(2·SE, δ)`, **la realización pesa**
   y la fase 2 no se lanza con copia fija: se pasa al ruido en línea (lo pendiente de S2). Si no,
   una copia fija sirve para esta fase. Se reporta además la amplitud de las medias entre tipos,
   para leer el uno contra el otro.
4. **Integridad antes de comparar**: el informe comprueba que las tres semillas de un escenario
   comparten la copia (huella), que los escenarios de una semilla comparten los pesos iniciales y
   que la val es la congelada. Lo que no case se avisa en voz alta, no se promedia en silencio.
5. **Una corrida con `summary.json` en disco no se repite** (el lanzador la salta): la fase 2
   reutiliza el nivel medio de la fase 1 en vez de volver a pagarlo.

## Lo que se espera, escrito antes

Con 180 imágenes y una red de 1.338 parámetros que ajusta el train a CE 0,0004, lo más plausible es
que algún ruido **regularice** (suba la CE de train y la exactitud de val) y que los de trazo a
α = 0,6 salgan **indistinguibles** por tenues (una recta de 1 px es ¼ de bloque); `borrado` y
`recorte` son los candidatos naturales a ayudar, `sal-pimienta` y `externos` a perjudicar. Es una
expectativa, no un criterio: lo que decide es la regla de arriba, y **«ninguno ayuda» es un
resultado**.

## Fase 3 — ruido EN LÍNEA: enmienda del 2026-10-03, escrita antes de lanzarla

Lo pendiente de S2 («una realización nueva por época, para más variedad»), sobre lo que la fase 2
dejó: **el mejor nivel de cada tipo cuyo mejor nivel fue «ayuda»** (`recorte@0.6`, `curva@0.8`,
`gaussiano@0.2`, `vertical@1`, `oblicua@1`), con el sufijo `-linea`, × 3 semillas = 15 corridas.

- **Qué es «en línea»**: la misma semilla de ruido que la copia fija; la copia de la **época 1 es
  exactamente la fija** (mismo generador, primer sorteo) y cada época siguiente saca otra del
  mismo generador — idéntica entre las tres semillas de pesos, como en las fases 1 y 2. Todo lo
  demás (pesos iniciales, orden de lotes, pasos, `lr`) igual. Tiene test: 1 época en línea ==
  1 época fija bit a bit; nivel 0 en línea == `limpio` bit a bit.
- **Dos medidas, pareadas por semilla**: Δ contra `limpio` (el mismo veredicto de siempre) y
  **Δ contra su copia fija**, con umbral `max(2·SE, δ)`: «en línea mejor», «peor» o
  «indistinguible de la fija».
- **Lo que se espera, escrito antes**: para `recorte`, `curva` y `gaussiano` —los que quitan o
  perturban más— la variedad debería sumar (en línea > fija); para `vertical@1` y `oblicua@1`,
  cuyo efecto roza δ, lo más probable es «indistinguible». Si en línea sale **peor** en todos,
  la lectura es que 222 copias distintas sin repetir ninguna no dejan ajustar la copia, y eso es
  un resultado.
- **No decide** ganador entre tipos (eso ya lo hizo la fase 2) ni combina ruidos.

## Fase 4 — grosor y número de trazos: enmienda del 2026-10-03, escrita antes de lanzarla

El eje que la fase 2 dejó abierto: los trazos suben hasta α = 1, que es el tope de opacidad, así
que «más» sólo puede ser **más grueso** o **más trazos**. Dos variantes del dibujo, con la misma
semilla y en línea (la fase 3 enseñó que en línea nunca es peor): **`-grueso`** = 3–4 px de grosor
a 32 (el plan: 1–2) con 1–2 trazos; **`-doble`** = 3–4 trazos (el plan: 1–2) de 1–2 px. Sobre los
trazos cuya versión en línea dio «ayuda» en la fase 3 (`vertical@1`, `oblicua@1`, `curva@0.8`),
× 3 semillas = 18 corridas. Sólo valen para rectas y curvas: `parsear` se niega con el resto.

- **Medida**: Δ contra `limpio` (el veredicto de siempre) y **Δ contra su base** (el mismo
  escenario en línea sin variante), pareado, umbral `max(2·SE, δ)`: «mejor», «peor» o
  «indistinguible de su base».
- **Lo que se espera, escrito antes**: que `-grueso` y `-doble` sumen algo en `vertical` y
  `oblicua` (siguen la tendencia «más ruido, mejor» de la fase 2) y que en `curva`, que ya tenía
  pico interior en α, no sumen o resten. Si una variante sale **peor** que su base, el eje está
  acotado por ese lado y se dice.
- **No decide** nada sobre combinar tipos ni sobre `recorte`/`gaussiano`, que no tienen trazos.
