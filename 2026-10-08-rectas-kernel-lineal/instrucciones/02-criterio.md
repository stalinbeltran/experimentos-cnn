# Criterio de `rect-lin` — escrito ANTES de medir (2026-10-08)

Se commitea antes de correr la rejilla y antes de pasar el banco a ninguna referencia. Lo que se escriba después de
mirar va al README marcado como tal.

## Definiciones

- **Detecta** = alguna de las 4 salidas con logit > 0. **Orientación** = la salida con mayor logit.
- **Recall** de un grupo de rectas = fracción detectadas **y** con la orientación correcta (centro más cercano, ±22,5°).
- **FP** = fracción de negativos con alguna detección, por tipo (ruido · puntos sueltos · mancha) y total.
- **Fuerza** = el mayor logit. Para curvas se reporta la tasa de detección y la fuerza media **relativa** a la de las
  rectas del mismo grosor y largo del banco.
- Grupos de grosor del banco: **fino** 2–4 px (el de entrenamiento) · **medio** 6–8 · **grueso** 10–14.
- **N90** de una configuración = el menor N cuyo recall fino (media de 3 semillas) llega al 90 % del de N = 1000, con
  FP total ≤ 0,10. Es la medida de «cuántas muestras necesita».
- Toda cifra es la **media de 3 semillas**; se reporta también el rango.

## Hipótesis (lo que espero, y qué cuenta como sí / no)

| # | hipótesis | se cumple si |
|---|---|---|
| **H1** aprende | algún kernel, entrenado con continuas y N = 1000, detecta rectas finas | recall fino ≥ 0,90 **y** FP ≤ 0,05 |
| **H2** pocas muestras | esa configuración no necesita muchas | N90 ≤ 32 |
| **H3** aprender aporta | el kernel aprendido supera al Gabor a mano (N = 0, mismo k y escalas) | recall fino +0,05 o más a igual o menor FP |
| **H4** las escalas resuelven el grosor | k = 7, continuas, N = 1000 | recall grueso (3 escalas) − recall grueso (1 escala) ≥ 0,20 |
| **H5** tamaño | con 3 escalas, 7×7 y 9×9 se parecen y 5×5 queda atrás | \|k7 − k9\| ≤ 0,03 en recall total **y** k5 ≤ k9 − 0,03 |
| **H6** punteadas sin haberlas visto | entrenado con continuas, k ≥ 7, 1 escala | recall en punteadas con separación ≤ 4 px ≥ 0,70 |
| **H6b** las escalas alargan el alcance | ídem, separación 10–12 px | 3 escalas − 1 escala ≥ 0,20 |
| **H7** de puntos a trazo | entrenado con punteadas, N = 1000 | recall fino en continuas ≥ 0,85 **y** FP en puntos sueltos ≤ 0,15 |
| **H8** recta > curva | la mejor configuración de H1 | la detección de curvas no baja al crecer el radio (holgura 0,05), y las curvas de radio ≤ 9 px se detectan ≤ 0,5 × el recall fino de rectas |
| **H9** contra la CNN de `feat-ind32` | la mejor configuración de H1 | recall fino ≥ CNN − 0,05 **y** recall grueso > CNN |

## Lo que haría cambiar la lectura

- Si **H1 falla**, el resto se lee como «qué hace un kernel lineal que no aprende la tarea», y no como capacidad.
- Si **H3 falla** (el Gabor a mano iguala al aprendido), el resultado es que **no hace falta aprender**: también es un
  resultado, y quizá el más barato.
- Si **H8 falla en el sentido contrario** (las curvas se detectan tanto como las rectas), un kernel lineal no separa
  recta de curva por fuerza, y la segunda etapa pendiente (curvas sobre rectas cortas) no tiene de dónde partir.
- **No se declara un único ganador.** Se reportan N90 y recall por grupo de cada configuración.

## Lo que NO contesta

- Si con esto se leen mejor los dígitos (no hay compositor: el dueño pidió el detector solo).
- La curva de N de la CNN: la CNN es un punto fijo (2000 por clase). Reentrenarla a cada N cuesta horas en 2 vCPU y no
  se hace aquí.
