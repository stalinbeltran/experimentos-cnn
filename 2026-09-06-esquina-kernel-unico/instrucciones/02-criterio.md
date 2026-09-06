# El criterio, congelado ANTES de la primera época

Escrito el 2026-09-06 con **cero épocas entrenadas** en cualquier brazo. Los suelos de abajo
están medidos sobre la partición de validación con las redes **sin entrenar**, que es lo único
que se puede medir antes de mirar.

## La métrica principal, y su suelo

**Tasa de acierto: qué fracción de las esquinas se predice a ≤ 2 px de su sitio**, medida
**sólo sobre las ventanas donde la esquina existe de verdad** (la etiqueta, no la predicción).
Val trae **156 ventanas con esquina de 389**.

| | acierto ≤2 px | acierto ≤1 px | error medio |
|---|--:|--:|--:|
| azar puro (área del círculo / área del cuadrado de lado 15) | **5,6 %** | 1,4 % | — |
| predictor **constante** en el centro | 7,7 % | 3,8 % | 5,88 px |
| las cinco redes **sin entrenar** | 7,7 – **8,3 %** | 1,9 – 3,8 % | 5,84 – 5,95 px |

> **Un brazo ha aprendido algo** si su acierto ≤2 px de validación pasa del **13 %**
> (= 8,3 % del techo sin entrenar + 2·SE, con SE = 2,2 % sobre 156 positivas).

### Por qué NO se usa la distancia media como métrica principal

Porque su suelo **parece bueno y no lo es**. Medido el 2026-09-06 sin entrenar nada: las cinco
redes dan 5,84–5,95 px, y ese número está explicado **entero por la geometría** — la distancia
media al centro de un cuadrado uniforme de lado 15 es 0,3826·15 = **5,74 px**. Como la distancia
máxima posible es 10,6 px, el suelo ya *parece* medio camino.

⚠ Y hay algo peor que la estética: **el modo degenerado de la cabeza C1 coincide con el mejor
predictor constante.** Si el mapa sale plano, el softmax es uniforme y la esperanza cae en el
centro del mapa, que es justo donde se concentran las etiquetas. La solución vaga y la solución
decente son la misma, así que hay un óptimo local cómodo exactamente donde la red arranca.

⚠ **Ensanchar el rango de posiciones NO lo arregla**, aunque sea lo que parece: el suelo escala
como 0,3826·L y el máximo también, así que la razón entre «no saber nada» y «acertar» queda
igual. Sólo cambia la escala.

La distancia media **se sigue registrando**, y siempre junto a su suelo (5,88 px), nunca sola.
Como métrica secundaria, un brazo que aprenda debería además bajar de **5,50 px** (5,87 − 2·SE).

## Los desenlaces, escritos antes

1. **Algún brazo pasa del 13 %.** El eje tiene señal. Gana el de **mayor acierto**; si dos caen
   dentro de 2·SE el uno del otro, se declara **empate** y gana el **kernel más pequeño**, por
   ser más barato.
2. **Ningún brazo pasa del 13 %.** *No hay señal en el eje del kernel con esta cabeza.* Es un
   resultado, no un fracaso — y entonces la pregunta siguiente no es «otro k» sino si la cabeza
   C1 puede expresar la tarea.
3. **`existe` no despega.** Su suelo es **f1 = 0,572**, que es lo que da el «siempre sí» con
   40,1 % de positivas — y es exactamente lo que dan las redes sin entrenar (medido: 0,5725). Un
   brazo por debajo de eso no ha aprendido a detectar, aunque acierte coordenadas.
4. **Todos los brazos salen igual de mal.** No se podrá distinguir «ningún kernel sirve» de «la
   cabeza es demasiado estrecha». Por eso hay que mirar **el mapa de respuesta** de las 10
   muestras: si el mapa tiene estructura y la lectura falla, el problema es la cabeza; si el mapa
   es plano, es el kernel.

## Lo que se congela, y es igual en los cinco brazos

dataset · semilla 1 · ventana 32×32 · sin padding · sin bias · cabeza C1 con β inicial 3,5 ·
Adam lr 0,05 · lote 128 · **`λ_coord` 0,03** · las mismas 10 muestras.

⚠ **`λ_coord` está MEDIDO, no elegido a ojo.** Al inicializar, la BCE vale 1,23–1,31 y el MSE de
coordenadas 40,9–42,6 px² en los cinco brazos; la regla es **«los dos términos parten iguales»**
—porque las coordenadas son el objeto de este experimento, no un extra— y eso da 0,0294–0,0317
según el brazo. Se congela en **0,03 para los cinco**: un λ por brazo haría que cada uno
optimizase una función distinta. Comprobado sobre val sin entrenar: 1,30 y 1,19.

**Lo único que varía es `k`.**

## Lo que este experimento NO contesta

- Nada sobre otras esquinas, otras escalas de reducción, ni otros tamaños de ventana.
- Nada sobre el `stride`: es el eje siguiente y no se toca aquí.
- Nada sobre si otra cabeza leería mejor el mismo mapa.
