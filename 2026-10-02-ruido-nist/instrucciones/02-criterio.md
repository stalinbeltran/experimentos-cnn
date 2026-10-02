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
