# Encargo (el dueño, 2026-10-11, conversación de Telegram)

> «Ayer anotamos pendiente que me ayudes a crear detectores, observando los dígitos. Concéntrate en idear formas de
> detectar el rizo superior de los números 2. Anota para claude que siempre me muestren los resultados en imágenes (raw
> de github, para evitar problemas de ayer). Anota también que un detector podría emplear un compositor si acaso hace
> falta, o podría crearse empleando entrenamiento de cnn. Lo importante es hallar la mejor solución. Tienes autorización
> para realizar los entrenamientos y pruebas sin mi supervisión, y puedes gastar hasta $10 en entrenamientos paralelos
> con tal de acelerar la obtención de resultados.»

Lo que el encargo da por sabido, escrito aquí:

1. **«El rizo superior del 2»** es la cabeza del 2: el arco que empieza arriba a la izquierda, pasa por arriba y baja
   por la derecha antes del trazo diagonal. Mirados los 48 primeros 2 de val (figura `etiquetar`, 2026-10-11): en los 48
   es un arco **abierto** (una C girada, de media vuelta a tres cuartos), con el hueco interior de 3–6 px, el centro en
   x 11–16 · y 7–9, y la abertura hacia **abajo-izquierda** (≈ 210–230°). Sólo 10 de los 159 doses de val lo cierran
   del todo (hueco cerrado, medido ayer en `lazos`). Un detector del rizo tiene que ver los dos casos.
2. **«Observando los dígitos»** pide una verdad que no dependa de suposiciones por clase. La única que había era el
   hueco cerrado (10 dígitos). Por eso este experimento trae **48 doses etiquetados a mano** (`etiquetas-2.json`):
   centro, radio y abertura del rizo, leídos por Claude sobre una rejilla de píxeles **antes** de correr ningún
   detector. ⚠ Son etiquetas de Claude, no del dueño, con ±1 px de error: la figura que las enseña está para que el
   dueño las corrija si no está de acuerdo.
3. **«Compositor o CNN; lo importante es la mejor solución»**: se comparan varias estrategias con los mismos números y
   se dice cuál gana y por qué, sin preferir la aprendida ni la construida de antemano.
4. **«Hasta 10 $ en paralelo»**: las redes de aquí son de 32×32 y unos miles de parámetros; en este dev (2 vCPU) un
   brazo entrena en minutos. Se corre local y en secuencia; alquilar no acelera nada que importe y mete una flota que
   vigilar. **Gasto previsto: 0 $.** Si un brazo pasara de 30 min se reconsidera con `nn/vast.sh` como los demás.
5. **Las imágenes se enlazan con `raw.githubusercontent.com`** en el reporte, nunca con el visor de GitHub.
