# Criterio (escrito y commiteado ANTES de medir, 2026-10-11)

## Qué es «el rizo superior» aquí, con números

Un arco de un solo trazo que cubre **al menos media vuelta** (cobertura ≥ 0,5), de radio 2–7 px (del centro a la mitad
del trazo), con el interior sin tinta. Se describe con **centro** (x, y), **radio**, **cierre** = (cobertura − 0,5)/0,5
en [0, 1] y **abertura** (hacia dónde queda el hueco, grados; 0 = →, 90 = ↓; no se mide con cierre ≥ 0,95). Un rizo
«superior» es el de centro en la mitad de arriba del lienzo (y < 16) de mayor puntuación.

## Las cuatro estrategias (cada una da, por imagen, una lista de rizos con puntuación)

| | estrategia | idea | parámetros que se calibran |
|---|---|---|---|
| **R** | rayos | un píxel de fondo es centro si de 36 rayos (cada 10°) una fracción ≥ F choca con tinta antes de RMAX px **y** las distancias al choque son parejas (un anillo, no tinta suelta); los píxeles que lo cumplen se agrupan y cada grupo es un rizo | F, RMAX, la dispersión máxima |
| **P** | plantillas | banco de rizos sintéticos (radio × grosor × cobertura × abertura) como imágenes con signo (+ en el trazo, − en el hueco); correlación con la imagen en ±1; máximos locales ≥ S | S, el banco |
| **V** | votos de curvatura | el detector de `lazos` (Gabor par + giro + votos por centro de curvatura), copiado, más sus rizos por rayos ≥ 0,97: **la referencia**, lo ya medido | ninguno (σ_v 1, V_min 3, como ayer) |
| **C** | CNN con sintéticos | red convolucional pequeña, entrenada **sólo** con imágenes sintéticas de verdad conocida (rizos con cola y distractores; negativos sin rizo) a predecir un mapa de calor del centro y, en él, radio, cierre y abertura; máximos ≥ p | p; dos brazos de arquitectura/datos |

Pre-proceso: **ninguno** (la imagen binaria tal cual, igual para sintéticos y dígitos). Calibrar: sintéticos semilla 101.

## Qué se mide, y el umbral

**A · Sintéticos de prueba** (semilla 202, 300 positivos con cola y distractores, 300 negativos; grosores 2–6 px):

1. Rizos encontrados (algún candidato a ≤ 2,5 px del centro real): **≥ 90 %**.
2. Negativos con algún rizo: **≤ 10 %**.
3. Error del centro, mediana: **≤ 1,5 px**. 4. Radio, mediana: **≤ 1 px**. 5. Cierre, mediana: **≤ 0,15**.
6. Abertura (cierre real < 0,9), mediana: **≤ 30°**.

**B · Los 48 doses etiquetados** (`etiquetas-2.json`, sin el dudoso: 47):

7. Encontrados (algún candidato a ≤ 2,5 px del centro etiquetado): **≥ 85 %**.
8. Error del centro, mediana: **≤ 1,5 px**. 9. Radio, mediana: **≤ 1 px**. 10. Abertura, mediana: **≤ 40°**.
11. Doses con un rizo DE MÁS en la mitad superior (otro candidato a > 2,5 px del etiquetado y con y < 16): **≤ 20 %**.

**C · Las demás clases** (val, 1617 dígitos): % con algún rizo en la mitad superior.

12. **1 y 4: ≤ 15 %** cada uno. **7: ≤ 30 %** (algunos 7 llevan gancho arriba).
13. 0, 3, 5, 6, 8, 9: sin umbral, se reporta (3, 9 y 8 tienen de verdad un arco arriba; el 0 un óvalo grande).

**D · Desplazamientos** (los 47 doses, d = −4…+4 px, horizontal y vertical por separado; la verdad se mueve d; lo
que sale del lienzo se pierde y el % recortado va en la figura):

14. Encontrados en cada d, sobre los no recortados: **no baja más de 5 puntos** respecto de d = 0.
15. Error del centro respecto de (etiqueta + d), mediana: **≤ 1 px** en todo d.

**Y se reporta** el tiempo por dígito de cada estrategia.

## Qué se llama «ganar»

La estrategia que **pase más de los 15 umbrales**; empate → la de mayor 7 (encontrados en los doses reales); nuevo
empate → menos 11 (rizos de más). Si ninguna pasa 7, no hay ganadora: se dice cuál se acerca y qué le falta.

## Predicción

- **C** gana: pasa A y B (7 ≈ 90 %), y falla 12 en el 7 (los ganchos del 7 se parecen al rizo).
- **P** pasa 7 por poco (≈ 80–85 %: el grosor de los dígitos varía más que el banco) y falla 11 (varios máximos por rizo).
- **R** falla 7 (< 60 %): con cobertura ≈ 0,6–0,7 fallan muchos rayos y para verlos hay que bajar F, que dispara 2 y 11.
- **V** falla 7 (< 40 %): el giro a ±4 px no ve radios < 5 (medido ayer) y los rizos por rayos ≥ 0,97 sólo ven los cerrados.
- Desplazamientos: planas para R, P y V por construcción; C baja ≤ 3 puntos (es convolucional, pero el borde del lienzo
  recorta contexto).
