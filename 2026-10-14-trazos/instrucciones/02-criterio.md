# Criterio (escrito y commiteado ANTES de medir, 2026-10-13)

## Qué es «un trazo» aquí, con números

Una **polilínea** de ≥ 4 px de longitud por la línea media de la tinta, de grosor ≈ constante, que no cambia de
dirección más de 90° en 3 px (una esquina más cerrada que eso separa dos trazos: el 4 y el 7 tienen esquinas, el 2 de
abajo también). Un trazo **se cruza** a sí mismo o a otro donde la línea media tiene un nodo de grado ≥ 3. Lo que sale
por dígito: `trazos = [{puntos, longitud, extremos, giro_total, cruces}]`, y `levantamientos = len(trazos)`.

**Sin orden ni sentido**: dos listas de trazos se comparan como conjuntos.

## El método (M), y sus parámetros

1. **Línea media**: cresta de la transformada de distancia (los píxeles de tinta cuya distancia al fondo es máximo
   local a lo largo de la normal), adelgazada a 1 px; grosor local = 2 × distancia. Parámetro: ninguno.
2. **Grafo**: nodos = píxeles de la línea media con 1 vecino (extremo) o ≥ 3 (cruce); aristas = cadenas entre nodos.
   **Poda** de espolones (aristas extremo–cruce de longitud < G/2, con G el grosor local): son artefactos del
   adelgazamiento en trazos gruesos, no trazos. Parámetro: el factor de poda (0,5 × grosor; se tantea 0,3–0,8).
3. **Cruces**: en un nodo de grado 4 se emparejan las aristas por **continuidad de tangente** (la tangente de cada
   arista se mide a 3 px del nodo, y se eligen las dos parejas de menor cambio de ángulo); en uno de grado 3, la pareja
   de menor cambio continúa si ese cambio es < 45°, y la tercera arista termina ahí (una T). Grado ≥ 5: se trata como
   dos cruces cercanos fundidos; se reporta. Parámetro: el ángulo máximo de continuación (45°; se tantea 30–60°).
4. **Trazos**: se recorre el grafo encadenando aristas por las parejas elegidas; cada cadena es un trazo. Las
   **esquinas** (cambio de dirección > 90° en 3 px dentro de una arista) cortan la cadena en dos.

## Qué se mide, y el umbral

**A · Sintéticos de prueba** (semilla 202: 300 dígitos sintéticos hechos de 1–3 trazos conocidos —rectas, arcos,
rizos con cola, un cruce en el 20 %—, grosor 2–6 px, lienzo 32×32):

1. Número de trazos correcto: **≥ 90 %**.
2. Cruces: encontrados a ≤ 2 px del real **≥ 90 %**, y **resueltos** (las parejas correctas) **≥ 85 %**.
3. Forma: distancia de Hausdorff entre cada trazo real y el recuperado que le toca (asignación por Hausdorff mínima),
   mediana **≤ 1,5 px**; extremos a ≤ 2 px del real **≥ 85 %**.
4. Con grosor 5–6 px (el peor caso del adelgazamiento): número de trazos correcto **≥ 80 %**.

**B · Los 50 dígitos etiquetados** (`etiquetas-trazos.json`: 5 por clase, de val, elegidos con semilla 0 antes de
mirar; para cada uno: número de trazos, extremos de cada trazo (x, y) y cruces (x, y)):

5. Número de trazos coincide: **≥ 80 %**.
6. Extremos a ≤ 2,5 px: **≥ 80 %**. 7. Cruces a ≤ 2,5 px: **≥ 80 %**, y sin cruces de más en los que no tienen.

**C · Todo val (1617), por clase, sin verdad — se reporta la distribución**:

8. 8: un trazo con un cruce **≥ 70 %**. 9. 0 y 1: un trazo sin cruces **≥ 85 %**. 10. Lo demás, sin umbral: la tabla
   «clase × (número de trazos, número de cruces)» es un resultado por sí misma, porque dice con qué estructura escribe
   cada clase esta población.

**D · Desplazamientos** (los 50 etiquetados, d = −4…+4 px, horizontal y vertical por separado; la verdad se mueve d;
lo que sale del lienzo se pierde y el % recortado va en la figura):

11. Número de trazos igual al de d = 0, sobre los no recortados: **≥ 95 %** en todo d (la línea media es local: mover
    la imagen debe mover los trazos y nada más; si cambia, es un fallo del método, no del dato).
12. Extremos respecto de (etiqueta + d), mediana: **≤ 1 px**.

**Y se reporta** el tiempo por dígito, y los casos de grado ≥ 5.

## Qué se llama «ganar»

No hay brazos: es UN método con dos parámetros tanteados en sintéticos de calibración (semilla 101) **y fijados antes
de mirar B, C y D**. El experimento «sale» si pasan 1–3 y 5–7 y 11; si pasa A y no B, el fallo es de la verdad o del
paso de sintético a real, y se dice cuál mirando los fallos uno a uno (figura). Si no pasa A, el método no sirve y se
anota qué paso falla (línea media, poda, cruces).
