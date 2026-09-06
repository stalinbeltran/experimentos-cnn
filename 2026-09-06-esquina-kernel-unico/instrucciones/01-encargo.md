# El encargo, tal como llegó (2026-09-06, por Telegram)

> «Vamos a diseñar un experimento para hallar una transformación que podría ayudar a mejorar
> el entrenamiento y la inferencia de una nn. […] El objetivo de una cnn es detectar las
> coordenadas de cada párrafo. […] usar estos párrafos directamente exigiría muchos píxeles.
> Puesto que la detección de párrafos no depende de los detalles pequeños de las letras que lo
> conforman, decidí que podía reducirlo. […] luego hallé que la detección realmente se podría
> hacer con menos detalles.
>
> El objetivo de este experimento es hallar cuál es la transformación más efectiva a la hora de
> entrenar una cnn para detección de párrafos. Y el método para lograrlo es entrenar una nn
> para reconocer párrafos empleando 1 kernel en una cnn, y así el kernel debe ser aprendido de
> modo que el resultado del kernel ayude lo más posible a detectar los párrafos.»

Y las precisiones que llegaron después, en el orden en que llegaron:

1. **El dataset se regenera** desde el repo de generación de párrafos, para poder ajustarlo:
   más pequeño, más limpio, más espaciado.
2. **La salida se simplifica a UNA esquina**: existe o no en la ventana, y sus coordenadas. Ya
   no hacen falta cuatro.
3. **Se empieza por el eje del kernel.** Nada más se hace hasta barrer ese eje.
4. **Sin padding**, que reduce el tamaño de la salida de la convolución.
5. **No se compara con nada**: este es un proyecto independiente.
6. **La cabeza se mantiene lo más reducida posible**, porque *«si la cabeza es grande, el kernel
   no aprende nada»*.
7. **10 muestras aleatorias fuera del entrenamiento**, al menos 4 con la esquina buscada, todas
   juntas en una imagen, como forma simple de verificar que funciona.
8. **Reanudable.**

## Lo que se decidió al diseñarlo, y por qué

- **La esquina es la SUPERIOR-IZQUIERDA**, no «cualquier esquina». Un filtro lineal que pique en
  una esquina tiene forma de **cuadrante** —positivo donde espera tinta, negativo donde espera
  fondo— y esa forma es intrínsecamente orientada: no puede picar en las cuatro orientaciones a
  la vez. Pedirle una sola es pedirle lo que su forma sí permite. *(Razonamiento sobre la forma
  del filtro, no medición.)* Y sale un regalo: **las otras tres esquinas son negativos difíciles,
  gratis y bien etiquetados.**
- **Cabeza C1** (3 parámetros): la esquina se lee como la **esperanza de la posición** bajo
  `softmax(β·M)`, y `existe` de `logsumexp`. Define «transformación efectiva» como *la que hace
  que la esquina sea el máximo del mapa de respuesta*.
- **Sin ReLU entre la convolución y la cabeza**, y **sin bias en la convolución**. Las dos van
  juntas y el porqué está medido en la cabecera de [`../nn/modelo.py`](../nn/modelo.py).
- **La entrada es TINTA** (fondo 0, tinta 255), no gris. Con la conv sin bias, un peso positivo
  significa «espero tinta aquí»: la interpretación de cuadrante se vuelve literal.
- **El barrido es k ∈ {3, 5, 7, 9, 11}**, impares para que el campo receptivo tenga centro. Con
  la cabeza C1 los 3 parámetros de cabeza **no dependen de k**, así que entre brazos lo único
  que cambia es el kernel.
