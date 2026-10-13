# Criterio (escrito y commiteado ANTES de medir, 2026-10-13)

## La salida y la verdad

Por imagen 32×32 binaria, la red devuelve `M` ∈ [0, 1]^(32×32). La verdad en sintéticos es `T = exp(−d²/(2·0,8²))`,
con `d` la distancia a la línea media real (la polilínea con que se pintó el trazo). Pérdida: BCE por píxel con peso 5 en
los píxeles de `T > 0,5` (la línea media es el 3–5 % de la imagen).

## Los brazos (entrenados sólo con sintéticos; semilla 101 calibrar, 202 probar)

| | brazo | qué cambia |
|---|---|---|
| **A** | U pequeña | encoder-decoder de 3 niveles (8–16–32 canales), salida sigmoide; ~20 k parámetros |
| **B** | sin reducción | 6 convoluciones 3×3 (16 canales) a resolución completa, campo receptivo 13 px; ~12 k parámetros |
| **C** | A + grosor | como A, pero la red también predice el grosor local (segunda salida, regresión), para que aprenda «centro» y no «tinta» |

Los sintéticos: 1–3 trazos por imagen (rectas, arcos de radio 2–12, rizos con cola, un cruce en el 25 %, una esquina
en el 20 %), grosor 2–6 px, 20 % con el lazo **macizo a propósito** (hueco de 0–2 px) y su verdad con el hueco: es lo
que enseña a abrir lazos. 6.000 para entrenar, 600 para validar la pérdida, 600 de prueba (202).

## Qué se mide, y el umbral — TODO sin binarizar, o con varios umbrales

**A · Sintéticos de prueba (600):**

1. **Posición**: distancia de la cresta de `M` (máximos locales a lo largo de la normal, `M` ≥ 0,3) a la línea media
   real, mediana **≤ 0,7 px**; y de la línea real a la cresta (cobertura), p90 **≤ 1,5 px**.
2. **Delgadez**: anchura de `M` a media altura, mediana **≤ 1,6 px**, para grosores 2–6 por separado (el de 6 px es el
   que importa).
3. **Topología estable**: el número de agujeros de `M > u` para u ∈ {0,3 · 0,5 · 0,7} coincide con el real en los tres
   **≥ 90 %** de las imágenes; y en las de lazo macizo, **≥ 75 %** (aquí es donde «aparecen» agujeros).
4. **Cruces**: en las imágenes con cruce, `M > 0,5` tiene un nodo de grado 4 (o dos de grado 3 a ≤ 2 px) a ≤ 2 px del
   cruce real **≥ 85 %**.

**B · Los 64 dígitos reales de la figura 19 de `curvas-gabor`** (los mismos de `dig-delg`, para comparar), con las
cuatro medidas de `dig-delg` aplicadas a la cresta de `M` (no a un umbral):

5. Delgadez (% de la cresta de 1 px de ancho): **≥ 85**. 6. Dentro de la tinta: **≥ 95**. 7. Cobertura de la tinta a
   ≤ 2 px: **≥ 95**. 8. Topología igual a la de la tinta para u = 0,5, **o** distinta sólo por agujeros que **abren
   un lazo macizo** (se cuentan aparte): **≥ 90 % de dígitos**. Ninguna alternativa de `dig-delg` pasó las cuatro.

**C · Val entera (1617) — lo que el encargo quiere ver:**

9. **Agujeros que aparecen**: en los 8 de val, % con **2 agujeros** en `M > 0,5` frente al % en la tinta (hoy, con
   área ≥ 3 px, 121 de 156 = 78 %): se reporta y se espera **≥ 85 %**, sin que suba el % de 6 y 9 con 2 (hoy 5/160 y
   13/162): **≤ 5 %** cada uno.
10. El detector de huecos de `rizo-2` (`huecos_separados`) sobre `M > 0,5` en vez de sobre la tinta, metido en el
    compositor «todo objetos» (118): val **no baja** de 0,9775 (se reporta a ciegas).

**D · Desplazamientos** (los 64 de B, d = −4…+4 px, horizontal y vertical, cada signo aparte; sin `np.roll`; el %
recortado va en la figura):

11. La cresta de `M` movida d coincide con la de d = 0 movida d: distancia mediana **≤ 0,5 px** y agujeros iguales
    **≥ 97 %** (una red convolucional sin reducción es equivariante; la U con reducción ×4 puede no serlo: es lo que
    se mide).

**Y se reporta** parámetros, tiempo de entrenamiento y tiempo por dígito de cada brazo.

## Qué se llama «ganar»

El brazo que pase más umbrales; desempate por 3 y 9 (la topología es el motivo del experimento). Si ninguno pasa 1–2,
la red no da la línea media y se para; si pasa A y no B, lo sintético no se parece a lo real y se mira qué falta en
los sintéticos antes de tocar la red.
