# Criterio (escrito y commiteado ANTES de mirar ningún resultado, 2026-10-10)

## Qué se compara

Tres alternativas que usan **sólo respuestas de Gabor** del dígito, y una referencia:

| brazo | idea | qué usa |
|---|---|---|
| **A · valle firmado** | la línea clara de la figura 19: donde la respuesta del Gabor impar **cambia de signo** a lo largo de la normal, con la polaridad de un trazo (borde de subida y luego de bajada), no la de un hueco entre trazos | el banco impar de la figura 19, **con signo** |
| **B · cresta del Gabor par** | un Gabor **par** del ancho del trazo responde máximo en el centro del trazo: se toma la cresta (máximo a lo largo de su normal). Dos anchos, λ_p 8 y 12 | un banco par de 8 orientaciones, kernel 9×9 |
| **C · fase local** | par e impar de la MISMA escala (λ 10, kernel 9): en el centro del trazo el par domina y el impar se anula (fase ≈ 0). Cresta de la energía donde la fase está cerca de 0 | el banco impar de la figura 19 + su par |
| *D · esqueleto morfológico* | *`skimage.morphology.skeletonize` sobre el dígito binario. **Sólo referencia**: es morfología* | *el dígito binario* |

## Qué se mide en cada dígito (los 64), y el umbral

1. **Delgadez**: % de píxeles del dígito delgado con **≤ 3** píxeles delgados en su 3×3 (él incluido). Una línea de
   1 px da ≤ 3 salvo en los cruces. **Pasa si ≥ 85 %** (media sobre los 64).
2. **Dentro de la tinta**: % de píxeles delgados que caen sobre tinta del dígito original. **Pasa si ≥ 95 %.**
3. **Cobertura**: % de la tinta original a ≤ 3 px de algún píxel delgado (el semiancho de un trazo de UCI es 2–3 px).
   **Pasa si ≥ 95 %.** Esto es lo que impide que un dígito delgado «vacío» o a trozos pase.
4. **Topología**: el número de **lazos** (huecos cerrados) y el de **trozos** (componentes conexas) igual que el dígito
   original. **Pasa si coincide en ≥ 90 % de los 64.**

⚠ Las cuatro métricas usan el dígito binario y la distancia euclídea **para medir**, nunca para obtener el delgado: medir
con morfología no es construir con morfología.

## Qué se llama «ganar»

**No se declara ganador.** Se reportan todos los brazos con sus cuatro números y se dice cuáles pasan los cuatro
umbrales; elige el dueño. La referencia D se mide igual, para saber cuánto falta, pero no compite.

## Predicción (antes de mirar)

- **A** es delgado y centrado (pasa 1 y 2), pero se **rompe en los cruces y en las curvas cerradas** del 8 y del 6,
  donde ninguna orientación ve un trazo limpio: cobertura ~90 % y topología ~70–85 % → **no pasa 3 y 4**.
- **B** con λ_p 8 sale más continuo que A (pasa 3) pero **más grueso en los cruces**; λ_p 12 junta trazos cercanos y
  **cierra lazos pequeños** (el 8 de lazo bajo pequeño, el mismo que confundía al compositor) → topología baja.
- **C** queda entre A y B.
- **D** pasa los cuatro por construcción, salvo espolones (ramitas sueltas) que no miden estas métricas.
- **Ninguna alternativa de Gabor pasa los cuatro en la primera vuelta**, y la que más se acerca es B · λ_p 8.
