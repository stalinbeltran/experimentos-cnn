# Criterio (escrito y commiteado ANTES de mirar ningún resultado, 2026-10-11)

## Qué es un lazo aquí, con números

- **Centro**: donde se acumulan los votos de los píxeles curvos. Cada píxel con giro medible y |κ| ≥ 2 °/px vota en
  p + R·k̂, con R = 57,3/|κ| y k̂ la dirección del vector de curvatura. Los votos se suavizan (σ_v) y cada máximo local
  con al menos V_min votos es un candidato. El centro no puede caer sobre tinta.
- **Radio**: el de la corona alrededor del centro con más tinta (2–12 px).
- **Cobertura**: la fracción de 36 sectores de 10° alrededor del centro que tienen tinta en la corona [0,6 R, 1,4 R].
- **Grado de cierre** = (cobertura − 0,5) / 0,5, recortado a [0, 1]: **media vuelta (una C, una U) = 0 «abierto»**,
  tres cuartos = 0,5, **vuelta completa = 1**. Con cobertura < 0,5 **no es un lazo** (es una curva).
- **Orientación**: el centro del hueco (racha de sectores vacíos) más grande. Apunta a la parte abierta. Con cierre
  ≥ 0,95 no se mide (no hay hueco).

## Cómo se calibra y cómo se prueba

- **Calibrar** (sólo σ_v ∈ {1; 1,5} y V_min ∈ {3, 5, 8, 12}): con lazos SINTÉTICOS de verdad conocida —arcos de
  círculo de radio 4–10, grosor 1–3 px, cobertura 0,5–1 (cierre 0–1), abertura y posición al azar— más negativos que
  NO son lazos (rectas, curvas de menos de 100°, ruido). Puntuación: media de «lazos encontrados» y «negativos sin
  lazo». Semilla 101.
- **Probar** con otros sintéticos (semilla 202) y con los dígitos.
- **El detector de curvas no se re-calibra**: es el delgado del boceto (Gabor par λ 6, σE 0,5, coherencia 0,25, giro
  mínimo 2 °/px) con **TAU 0,6**, el que mejor generalizó en `tau_comparar.py`.

## Qué se mide, y el umbral

**Sintéticos de prueba:**

1. Lazos encontrados (un candidato a ≤ 3 px del centro real): **≥ 90 %**.
2. Negativos con algún lazo (falso positivo): **≤ 10 %**.
3. Error del centro, mediana: **≤ 1,5 px**.
4. Error del grado de cierre, mediana: **≤ 0,15**.
5. Error de la orientación (lazos con cierre < 0,9), mediana: **≤ 30°**.

**Dígitos** (val, 1617; no hay verdad de lazos, así que se compara con lo esperable de cada clase):

6. 0, 6 y 9 con al menos un lazo de cierre ≥ 0,5: **≥ 80 %** de cada clase.
7. 8 con dos lazos: **≥ 60 %**.
8. 1 y 7 sin ningún lazo: **≥ 80 %** de cada clase.
9. 2, 3, 4 y 5: sin umbral (dependen del estilo de cada escritor); se reporta la distribución.

**No se declara ganador** (hay una sola calibración elegida): se dice qué umbrales pasa.

## Predicción

- Sintéticos: pasa 1, 3 y 5; **el cierre (4) falla en los de grosor 3 y radio 4** (la corona es casi tan ancha como el
  radio); falsos positivos ~5 % (pasa 2).
- Dígitos: pasa 6 en el 0 y el 6, y **el 9 se queda cerca de 80 %** (su lazo es pequeño). **El 8 no llega a 60 %** (sus
  dos lazos se tocan y los votos se mezclan). El 1 pasa 8; **el 7 no** (su codo atrae votos).
