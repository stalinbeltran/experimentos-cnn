# `experimentos-cnn`

Pruebas de estructuras de red. **Un experimento por carpeta, y cada uno con sus propias
reglas**: su código, su criterio, su métrica y sus objetivos. Pueden no parecerse en nada
entre sí — es a propósito.

Separado de [`foveal-vision`](https://github.com/stalinbeltran/foveal-vision) el 2026-09-06,
porque allí las restricciones del proyecto entorpecían especificar pruebas libres.

```bash
python3 comprobar.py           # qué hay, qué puede correr aquí, y qué está mal puesto
python3 comprobar.py --indice  # regenera la tabla de abajo desde los experimento.json
```

Desde Telegram: `/use exp`.

## Los experimentos

La tabla la **genera** `comprobar.py --indice` leyendo cada `experimento.json`. La identidad
de un experimento es su `id`, no el nombre de su carpeta: por eso las carpetas se pueden
renombrar y re-ordenar sin romper nada.

<!-- INDICE: generado por `python3 comprobar.py --indice`. No editar a mano. -->

| experimento | estado | qué pregunta |
|---|---|---|
| [`2026-09-06-esquina-kernel-unico/`](2026-09-06-esquina-kernel-unico/) `esq-k` | cerrado | Con UN solo kernel y una cabeza de 3 parametros, que tamano de kernel produce la transformacion mas efectiva para detectar la esquina superior-izquierda de un parrafo |
| [`2026-09-07-esquinas-cualquiera/`](2026-09-07-esquinas-cualquiera/) `esq-cq` | abierto | Colapsando las dos salidas de esq-2d en UNA -- 'hay esquina' y su posicion, sin importar si es la superior-izquierda o la inferior-derecha -- que tamano de kernel unico lo hace mejor, y cuanto se gana o se pierde respecto de tener que declarar cual esquina es |
| [`2026-09-07-esquinas-diagonales/`](2026-09-07-esquinas-diagonales/) `esq-2d` | cerrado | Una sola convolucion sin bias produce UN mapa, y las dos esquinas en diagonal se leen de sus dos extremos: la superior-izquierda del maximo y la inferior-derecha del minimo. Que tamano de kernel lo hace mejor, y cuanto se pierde respecto del mismo barrido sobre UNA sola esquina de esq-k |
| [`2026-09-08-banco-kernels/`](2026-09-08-banco-kernels/) `banco-k` | abierto | Dado un kernel k x k cualquiera, aplicado a las entradas antes de la red, mejora el IoU sobre eval por encima del kernel aleatorio de igual norma, y ademas REDUCE la brecha train-eval respecto de la identidad (transferencia) en vez de solo subir las dos juntas (facilitacion) |
| [`2026-09-09-bordes-parrafo/`](2026-09-09-bordes-parrafo/) `bor-p` | abierto | Que kernel de convolucion de k <= 19 localiza los cuatro bordes del recuadro que encierra un parrafo, sobre paginas de papel limpio con varios parrafos que nunca se solapan |
| [`2026-09-17-gauss-parametros/`](2026-09-17-gauss-parametros/) `gauss-p` | abierto | Que sigma (y que k para contenerlo) merece la pena meter por la puerta de banco-k, visto sobre 10 muestras de train de su propio dataset, dado que el unico punto gaussiano ya evaluado es el defecto k/6 y nadie lo eligio |
| [`2026-10-01-bordes-autoencoder/`](2026-10-01-bordes-autoencoder/) `bor-ae` | abierto | Con un solo filtro, una penalizacion de dispersion y un decodificador de norma 1, ¿que kernel sale de reconstruir las ventanas de borde de parrafo? ¿Algo distinto de la identidad y de un pasa-bajos? |
| [`2026-10-01-bordes-kernel/`](2026-10-01-bordes-kernel/) `bor-k` | abierto | Un solo kernel de convolucion de k <= 19, leido por los extremos de sus perfiles horizontal y vertical, ¿localiza los cuatro bordes de la caja de tinta de un parrafo? Y por k: ¿cuales aprende y cuales no? |
| [`2026-10-01-bordes-parrafo-r4/`](2026-10-01-bordes-parrafo-r4/) `bor-p4` | abierto | No contesta ninguna: produce el dato. Paginas limpias con 2-4 parrafos y la caja de tinta de cada uno, a la escala en que banco-k aplica el kernel (/4), con la separacion y el margen que exige un kernel de 19 px A ESA ESCALA |
| [`2026-10-01-bordes-pca/`](2026-10-01-bordes-pca/) `bor-pca` | abierto | ¿Cual es la direccion de mayor varianza de los parches de borde de parrafo a la escala del banco, y sirve como kernel? |
| [`2026-10-01-dimension-generalizacion/`](2026-10-01-dimension-generalizacion/) `dim-gen` | cerrado | Con el 10 % del dataset para entrenar y el 90 % para validar, L capas fijas sin padding y un kernel que siempre ve la misma FRACCION f del ancho de la imagen, ¿como cambia el IoU sobre las imagenes no vistas —y la brecha train-val— al reducir la imagen de 128 a 64, 32, 16 y 8 px? ¿Hay un W minimo suficiente, y hay un W a partir del cual la resolucion sobra o estorba? |
| [`2026-10-02-dimension-nist/`](2026-10-02-dimension-nist/) `dim-nist` | cerrado | Con los digitos manuscritos de 8x8 (NIST via UCI/scikit-learn), el 10 % para entrenar (180) y el 90 % para validar (1617), L = 2 capas sin padding con kernel n = ⌈W/2⌉ y cabeza constante, ¿como cambia la exactitud sobre las no vistas —y la brecha train-val— al reducir la imagen de 8 a 7, 6, 5 y 4 px? ¿Que resolucion generaliza mejor, y hay una a partir de la cual la resolucion sobra o estorba? |
| [`2026-10-02-ruido-nist/`](2026-10-02-ruido-nist/) `ruido-nist` | cerrado | Con 180 imagenes de entrenamiento y 1617 de validacion limpias, la misma CNN (3 capas 3x3 sin padding) y los mismos pesos iniciales y orden de lotes por semilla, ¿que tipo de ruido aplicado SOLO al entrenamiento sube la exactitud de validacion respecto de entrenar sin ruido, y cuanto? |
| [`2026-10-03-features-independientes/`](2026-10-03-features-independientes/) `feat-ind` | abierto | Si cada CNN pequena reconoce una unica feature fijada de antemano (reconoce, no discrimina) y las N se entrenan independientes y en paralelo, ¿aprende cada una su feature (presencia y posicion en 8x8), y que resultado neto dan juntas al reconocer un objeto complejo compuesto por varias de esas features? (Sin comparar contra una CNN monolitica, por ahora) |
| [`2026-10-03-posicion-relativa-features/`](2026-10-03-posicion-relativa-features/) `feat-pos` | abierto | ¿Como se codifica la posicion relativa entre las features de un objeto —p. ej. un 9 = circulo + recta vertical unidos en cierta posicion, y esa es solo una de sus codificaciones posibles— de forma que la informacion posicional entre en el resultado del reconocimiento sin pasar por texto, y sirven los embeddings (de posicion, o de la configuracion entera) para eso? |
| [`2026-10-03-ruido-combinado/`](2026-10-03-ruido-combinado/) `ruido-comb` | cerrado | Con el mismo dato, red, semillas y protocolo de ruido-nist (180 train + 180 copias en linea, 1617 val limpias, pesos iniciales identicos por semilla), ¿aplicar recorte@0.6 y gaussiano@0.2 JUNTOS en la copia —en secuencia sobre la misma imagen, o repartidos al 50 % entre imagenes— sube la exactitud de val por encima del mejor de los dos solos (gaussiano@0.2 en linea, 0,912), y cuanto? |
| [`2026-10-05-agrupar-por-features/`](2026-10-05-agrupar-por-features/) `feat-agr` | abierto | Si cada digito se describe por las features que lo componen —que detector se enciende y en que zona— y se agrupan los digitos por esa descripcion sin leer la etiqueta, ¿salen grupos de formas semejantes —variantes de un mismo digito (el 1 que es solo una recta vertical y el 1 con un trazo inclinado arriba) y formas que comparten digitos distintos— y cuanto coinciden esos grupos con las etiquetas de NIST? |
| [`2026-10-05-features-independientes-32px/`](2026-10-05-features-independientes-32px/) `feat-ind32` | abierto | Si los detectores independientes por feature y el compositor de feat-ind trabajan sobre el bitmap 32x32 sin reducir, ¿aprenden mejor su feature (sobre todo arcos y esquinas, que a 8x8 no aprendieron) y sube el resultado neto sobre los mismos digitos? |

<!-- FIN INDICE -->

## Cada experimento es independiente de los demás

**No se hereda nada entre experimentos**: ni las condiciones del dataset, ni los
hiperparámetros, ni la métrica, ni el umbral, ni los nombres de los scripts, ni la forma de la
carpeta. Dos experimentos que hacen lo mismo de dos formas distintas están bien los dos, y una
diferencia entre ellos **no es un bug que haya que arreglar**.

Copiar la carpeta de otro para empezar es normal y está bien — es para ahorrar tecleo, no para
heredar obligaciones: se relee todo y se cambia lo que no aplique, sin justificar nada.

Por eso **cada experimento trae su `REGLAS.md`**: sus entradas, sus salidas, sus procesos, sus
scripts y **qué NO hereda** de aquel del que se copió. Es obligatorio y lo comprueba
`comprobar.py`. El porqué entero está en [`CLAUDE.md`](CLAUDE.md) § «Regla 0».

Y **los datasets viven en el repo de datos** (`foveal-vision-data/experimentos-cnn/`), no aquí:
un dataset se publica una vez y todos los experimentos que lo necesiten lo leen de ahí por su
nombre, en vez de regenerarlo. Cada experimento puede tener el suyo.

## Cómo se añade uno

1. `mkdir 2026-09-06-<nombre>` y copia dentro `experimento.ejemplo.json` como
   `experimento.json`, rellenado.
2. Copia `REGLAS.ejemplo.md` como `REGLAS.md` y rellénalo. Si vienes de copiar otro
   experimento, **reléelo línea por línea**: es el paso que se salta.
3. Escribe el **criterio antes de mirar** en `instrucciones/02-criterio.md`.
4. Lo demás lo decide el experimento. `python3 comprobar.py` te dice si algo está mal puesto.

Las reglas, la frontera de lo que un experimento puede decidir y dónde va cada artefacto
están en [`CLAUDE.md`](CLAUDE.md).
