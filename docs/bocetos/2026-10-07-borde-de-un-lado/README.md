# Boceto — detectores de borde DE UN SOLO LADO, como primera capa (2026-10-07)

**No es un experimento**: no entrena nada ni gasta nada. Es la ilustración que pidió el dueño para decidir si
vale la pena montar uno. Las imágenes salen de `ver_proceso.py` sobre `uci-optdigits-orig-32px-r20261005`, con
kernels FIJOS (Sobel orientado) para enseñar el mecanismo.

## Regla del dueño de este día

*«Al menos por ahora, no se aplica pre-proceso a ningún dígito antes del detector de features.»* El código de
pre-proceso de `feat-bor` (`nn/bordes.py`) y de `feat-fallos` (esqueleto, erosión) se queda, pero no se usa. Aquí
el borde no es un script delante de la red: es su **primera capa** (un kernel + ReLU), y entra la tinta tal cual.

## La idea

Un detector = un kernel 3×3 + ReLU. El kernel del detector θ es la derivada en la dirección θ, así que con ReLU
sólo responde donde la tinta **aumenta** al avanzar en θ: «blanco → negro yendo hacia θ». «Negro → blanco hacia θ»
no es una familia aparte: es el detector de θ + 180°. Con 8 direcciones están los dos tipos.

**Por qué ve un solo borde:** recorrido en θ, un trazo tiene un borde de entrada (+) y uno de salida (−). ReLU
deja pasar uno. La distancia entre los dos (el grosor) deja de importar: el detector `→` dibuja sólo el lado
izquierdo de cada trazo, mida 2 px o 12. Lo enseña `imagenes/1-concepto-barra.png`.

**Y en un 8 grueso:** el contorno exterior sigue entero en los detectores. El hueco que se cierra sólo quita los
bordes del hueco, que además salen en los detectores **opuestos** (el lado izquierdo de un hueco es negro→blanco
yendo a la derecha, o sea `←`). Fila 2 y 4 de `imagenes/3-ochos-fino-a-grueso.png`.

## Lo que se ve y no es bonito (para el criterio, si se monta el experimento)

1. **Los diagonales salen a trozos** en UCI: a 32 px el dígito es una escalera de píxeles, y un 3×3 ve escalones.
   Un kernel 5×5, o un poco de suavizado DENTRO de la red (otra capa, no pre-proceso), lo arreglaría; no medido.
2. **Un trazo de 1 px**: los dos bordes caen a 1 px uno del otro; con ReLU cada detector se queda con uno, pero
   desplazado medio píxel. Comprobado sólo a ojo.
3. **Lo que esto NO resuelve**: en `feat-bor` el borde con signo (4 canales, por pre-proceso) dio el mismo mapa que
   esta capa daría con kernels fijos, y los detectores de features **entrenados con trazos de 2–4 px** perdieron
   recall en trazos de 6–12 px (0,84 → 0,73). El borde de un lado quita el «dos líneas» de cada canal, pero los
   bordes de los OTROS lados se separan con el grosor. La pregunta del experimento sería si los detectores de
   features, entrenados sobre estos canales, aprenden la forma de UN canal.

## Cómo se entrenaría (opciones, de más barata a más libre)

- **A. Kernels fijos** (Sobel orientado, lo de las imágenes). 0 parámetros, no hay nada que entrenar.
- **B. Fijos al arrancar y aprendibles**, con la misma red de features detrás.
- **C. Aprendidos desde cero contra un objetivo sintético**: en los dibujos sintéticos se conoce la geometría, así
  que la etiqueta «aquí empieza la tinta yendo hacia θ» se calcula del dibujo. Es la que más cuesta y la única que
  dice si una red encuentra sola este detector.

    /tmp/vizenv/bin/python ver_proceso.py --datos <dir con datos.npz>   # numpy, scipy, matplotlib
