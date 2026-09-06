# El criterio, congelado ANTES de la primera época

Escrito el 2026-09-06 con **cero épocas entrenadas** en cualquier brazo. Los suelos de abajo
están medidos sobre la partición de validación con las redes **sin entrenar**, que es lo único
que se puede medir antes de mirar.

## La métrica principal, y su suelo

**Error de posición**: distancia media, en píxeles de la ventana, entre la esquina predicha y la
verdadera, medida **sólo sobre las ventanas donde la esquina existe de verdad** (la etiqueta, no
la predicción). Val trae **156 ventanas con esquina de 389**.

| suelo | error |
|---|---:|
| predictor **constante** en la posición media de train (15,7 · 15,7) | **5,87 px** |
| predecir siempre el centro de la ventana | 5,88 px |
| las cinco redes **sin entrenar** | 5,84 – 5,95 px |

⚠ **Las cinco redes sin entrenar son indistinguibles del predictor constante**, y eso es
esperable, no un fallo: con un kernel aleatorio el mapa tiene poco contraste y la lectura por
esperanza colapsa al centro. **Ése es el suelo real: 5,87 px.**

> **Un brazo ha aprendido algo** si su error de validación baja de **5,50 px**
> (= 5,87 − 2·SE, con SE = 0,188 px sobre las 156 positivas). Por encima de eso, no se
> distingue de no haber entrenado.

## Los desenlaces, escritos antes

1. **Algún brazo baja de 5,50 px.** El eje tiene señal. Gana el de **menor error**; si dos caen
   dentro de 2·SE el uno del otro, se declara **empate** y gana el **kernel más pequeño**, por
   ser más barato.
2. **Ningún brazo baja de 5,50 px.** *No hay señal en el eje del kernel con esta cabeza.* Es un
   resultado, no un fracaso — y entonces la pregunta siguiente no es «otro k» sino si la cabeza
   C1 puede expresar la tarea.
3. **`existe` no despega.** Su suelo es **f1 = 0,572** (el «siempre sí», con 40,1 % de
   positivas). Un brazo por debajo de eso no ha aprendido a detectar, aunque acierte coordenadas.
4. **Todos los brazos salen igual de mal.** No se podrá distinguir «ningún kernel sirve» de «la
   cabeza es demasiado estrecha». Por eso hay que mirar **el mapa de respuesta** de las 10
   muestras: si el mapa tiene estructura y la lectura falla, el problema es la cabeza; si el mapa
   es plano, es el kernel.

## Lo que se congela, y es igual en los cinco brazos

dataset · semilla 1 · ventana 32×32 · sin padding · sin bias · cabeza C1 con β inicial 3,5 ·
Adam lr 0,05 · lote 128 · `λ_coord` 0,01 · las mismas 10 muestras.

**Lo único que varía es `k`.**

## Lo que este experimento NO contesta

- Nada sobre otras esquinas, otras escalas de reducción, ni otros tamaños de ventana.
- Nada sobre el `stride`: es el eje siguiente y no se toca aquí.
- Nada sobre si otra cabeza leería mejor el mismo mapa.
