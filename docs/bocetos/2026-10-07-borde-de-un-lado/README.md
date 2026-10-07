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

## Segunda parte (pedida el mismo día): usar estos bordes como el ÚNICO pre-proceso, delante de los detectores

Propuesta del dueño: aplicar estos 8 (o 4) detectores de un lado como pre-proceso único a todos los dígitos, y
entrenar los detectores de verdad (rectas, curvas…) sobre su salida. La regla se reescribe en consecuencia (ver
`CLAUDE.md` de la raíz).

### Lo que ya está medido y hay que tener delante

`feat-bor` (2026-10-06) **ya probó casi esto**: su representación `signo` son los 4 canales de un lado (→ ↓ ← ↑),
calculados por pre-proceso. Resultado: con trazos gruesos el recall cayó (0,84 → 0,73) y en los dígitos el compositor
bajó de **0,949 a 0,865**. O sea que «poner los bordes delante» a secas ya se pagó y salió peor.

### Por qué falló, y qué cambia aquí — `imagenes/5-un-canal-contra-todos.png`

En `feat-bor` **cada detector miraba los 4 canales a la vez** (la primera convolución tenía 4 canales de entrada).
La figura 5 enseña el problema: **dentro de UN canal** el contorno exterior es el mismo a 2, 6 y 12 px (canal →: el
arco izquierdo, idéntico); **entre canales**, la distancia del lado → al lado ← es justo el grosor. Un detector que
mira los 4 juntos vuelve a ver el grosor — la columna «los 4 juntos» es otra vez el trazo doble.

**El aporte, por tanto: que cada detector de features mire UN canal**, con los mismos pesos para los 4 (u 8)
canales (rotados o compartidos), y que se combine **después** (máximo o suma de sus respuestas). Esto no se probó en
`feat-bor`.

### Lo que la figura también enseña, y no es tan bonito

Cada canal tiene dos clases de borde: el del **contorno exterior** del dígito (no cambia con el grosor) y el del
**hueco** (el borde interior del trazo, que encoge o desaparece al engrosar: fila de 12 px, canales ↓ y ←). Para tu
caso del 8 eso es justo lo que querías — el contorno exterior sobrevive aunque el trazo tape el hueco —, pero un
detector de «curva» verá curvas distintas según lo grueso que sea el trazo **por dentro**. El vocabulario de features
probablemente haya que redefinirlo **por canal** («curva vista desde la izquierda»), no reutilizar el de `feat-ind32`.

### Propuesta concreta (no lanzada; criterio por escribir antes de entrenar)

1. Pre-proceso fijo: los 4 canales de un lado (kernels de la figura 2), igual para entrenamiento, prueba gruesa y
   dígitos UCI.
2. Brazos: (a) un detector por feature mirando los 4 canales a la vez — la réplica de `feat-bor signo`, como
   control —, (b) el mismo detector compartido aplicado a cada canal por separado, y max entre canales.
3. Medidas: recall en la prueba gruesa (6–12 px no vistos) y el compositor en los dígitos, contra 0,949 (tinta) y
   0,865 (`signo`).
4. Coste: del orden de `feat-bor` (1 máquina Vast, ~1 h, ~0,1 $) — estimado por comparación, no medido.

## Tercera parte: prueba rápida de vistas y «compositor de compositores» (medido 2026-10-07, 0 $, 1 min en el dev)

`prueba_vistas.py` → `resultados-vistas.txt` / `.json`. Detectores = los kernels FIJOS (no los de features);
compositor = regresión logística sobre cada vista reducida a 8×8. Mide la información que queda en cada vista, no lo
que haría la red de verdad. Prueba: los 1617 dígitos de `val`, tal cual, engrosados artificialmente (+1/+2 px de
dilatación, estímulo de prueba) y el cuartil con más tinta de cada clase («gruesos reales»).

Lo que sale (entrenado con 3823 dígitos de otros escritores; con 180, el mismo patrón un escalón más abajo):

| | normal | +1 px | +2 px | gruesos reales |
|---|---:|---:|---:|---:|
| tinta | 0,951 | **0,913** | **0,740** | 0,959 |
| 1 vista (rango de las 8) | 0,889–0,910 | 0,71–0,79 | 0,29–0,43 | 0,89–0,93 |
| 2 vistas → ↓ juntas | 0,958 | 0,868 | 0,425 | 0,964 |
| 8 vistas juntas | **0,973** | 0,892 | 0,416 | **0,971** |
| 8 vistas, compositor de compositores | 0,964 | 0,907 | 0,480 | 0,966 |

1. **Una vista sola pierde**: 4–6 puntos por debajo de la tinta con trazo normal.
2. **Dos vistas ya igualan a la tinta** y ocho la superan (0,973 contra 0,951): juntas tienen más información útil.
3. **El compositor de compositores no mejora al que ve todo junto** con trazo normal (0,964 contra 0,973), pero
   **resiste algo mejor el engrosamiento** (+1 px: 0,907 contra 0,892; +2 px: 0,480 contra 0,416). Con pocos datos
   (180) empata con el de todo junto en normal.
4. ⚠ **Contra la hipótesis: con engrosamiento ARTIFICIAL, los bordes caen MUCHO más que la tinta** (+2 px: 0,42
   contra 0,74). Los gruesos REALES no muestran esa caída (las vistas empatan o ganan a la tinta). Lo más probable
   *(no comprobado)*: al dilatar, el contorno se mueve 2 px hacia fuera —media celda del 8×8— y el compositor lineal
   es posicional; la tinta media cambia menos. O sea que **la invariancia al grosor de cada vista no llega al
   compositor si éste compara posiciones finas**, que es el problema 1 de antes visto desde el otro lado. Pide
   pooling (máximo 3×3) o un compositor menos posicional antes de cualquier experimento de pago.

### Y con pooling tolerante al desplazamiento (medido 2026-10-07, mismo guion, `--pool max5` y `--pool max-bloque`)

| entrenado con 3823 | normal | +1 px | +2 px | gruesos reales |
|---|---:|---:|---:|---:|
| tinta · media (lo de arriba) | 0,951 | 0,913 | 0,740 | 0,959 |
| tinta · max5 | 0,907 | 0,783 | 0,440 | 0,887 |
| 8 vistas juntas · media | 0,973 | 0,892 | 0,416 | 0,971 |
| 8 vistas juntas · max5 | **0,977** | **0,919** | 0,448 | **0,983** |
| 8 vistas juntas · max-bloque | 0,967 | 0,900 | 0,503 | 0,978 |
| 8 vistas compositor de compositores · max5 | 0,963 | 0,906 | 0,480 | 0,964 |

**La explicación de arriba («es el desplazamiento») NO queda confirmada.** El pooling mejora un poco las vistas
(lo mejor medido: 8 vistas + max5, 0,977 normal y 0,983 en gruesos reales), pero **no rescata el +2 px** (0,45–0,50),
y a la tinta la hunde. Otra explicación, también sin comprobar: una dilatación de 5×5 cierra huecos y funde trazos
vecinos —cambia la forma, no sólo el grosor—, y la tinta media la tolera porque sólo se oscurece. Los gruesos
**reales** no muestran nada de eso: ahí las vistas ganan. O sea que el +2 px artificial puede no ser un buen modelo
de un trazo grueso real; hace falta una prueba de grosor mejor antes de concluir nada.

## ⏳ PENDIENTE para la próxima sesión (escrito 2026-10-07)

Estado: **nada lanzado, nada pagado**. Todo es boceto con kernels fijos y regresión logística.

1. **Decidir qué es «grueso»** para medir: el +2 px por dilatación cierra huecos y quizá no representa trazos
   gruesos reales. Opciones: el cuartil más grueso de UCI (ya está, y ahí las vistas ganan), o el dataset
   sintético `feat-bor-sinteticas-grueso-32px-r20261006` (2–12 px, generado con grosor de verdad).
2. **Elegir el compositor**: con kernels fijos gana «8 vistas juntas + max5» (0,977 / 0,983). El compositor de
   compositores no gana con trazo normal y sólo resiste un poco mejor el engrosamiento artificial; con 180 de
   entrenamiento empata.
3. **Montar el experimento de verdad** (con su `experimento.json`, `REGLAS.md` y criterio escrito ANTES de entrenar):
   detectores de features entrenados por vista (mismos pesos para todas), etiquetas definidas por vista, y comparar
   contra 0,949 (tinta, `feat-ind32`) y 0,865 (`feat-bor signo`). Estimado ~0,1 $ / ~1 h en Vast (por comparación
   con `feat-bor`, no medido). **Pedir permiso antes de alquilar.**
4. Regla vigente: un solo pre-proceso, el mismo para todos los dígitos (`CLAUDE.md` de la raíz).
5. Para repetir esta prueba: el dataset de dígitos sale del almacén
   (`git show origin/main:experimentos-cnn/uci-optdigits-orig-32px-r20261005/datos.npz` en `foveal-vision-data`),
   y el entorno es `uv venv /tmp/vizenv && uv pip install --python /tmp/vizenv numpy scipy matplotlib scikit-learn`.
