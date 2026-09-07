# `esq-k` — qué kernel único señala la esquina superior-izquierda de un párrafo

**Corrido el 2026-09-06, 23:45 → 23:54 UTC · 5 brazos × 300 épocas · ~9 min de reloj ·
0 máquinas · 0 $** (CPU de este droplet de 2 vCPU). El criterio está congelado en
[`instrucciones/02-criterio.md`](instrucciones/02-criterio.md), escrito **antes** de la primera
época.

## Qué salió

En la época que guarda su `best.pt` (elegida por `val_loss`, la misma regla en los cinco):

| brazo | k | parámetros | época | **acierto ≤2 px** | acierto ≤1 px | error medio | f1 `existe` |
|---|--:|--:|--:|--:|--:|--:|--:|
| suelo (sin entrenar) | — | — | 0 | 7,7–8,3 % | 1,9–3,8 % | 5,88 px | 0,572 |
| **k03** | 3 | 12 | 285 | **8,3 %** | 3,2 % | 5,31 px | 0,639 |
| k05 | 5 | 28 | 141 | 92,3 % | 13,5 % | 1,38 px | 0,935 |
| **k07** | 7 | 52 | 102 | **100,0 %** | 96,8 % | 0,40 px | 0,997 |
| k09 | 9 | 84 | 66 | 100,0 % | 100,0 % | 0,19 px | 0,997 |
| k11 | 11 | 124 | 159 | 100,0 % | 100,0 % | **0,08 px** | 1,000 |

**Se cumple el desenlace 1 del criterio: hay señal en el eje.** Y por la regla congelada
—empate dentro de 2·SE, gana el kernel más pequeño por ser más barato— **el ganador es `k07`**.

### Las tres cosas que hay que leer antes que el ganador

1. ⚠ **`k03` NO aprende: se queda en 8,3 %, que es exactamente el suelo sin entrenar.** Con 12
   parámetros y un campo receptivo de 3 px, la red no puede señalar la esquina. No es que
   aprenda poco: está en el azar. Y sin embargo su `f1 existe` sube a 0,639 (por encima del
   0,572 del «siempre sí»), o sea que **detecta algo y no sabe dónde** — las dos mitades de la
   tarea se separan.
2. ⚠⚠ **La métrica principal SATURÓ.** Tres brazos dan 100 %, así que el empate no es un empate
   medido: es un techo. El criterio se escribió para contestar *«¿aprende algo?»* y esa pregunta
   quedó contestada de sobra; para ordenar a los ganadores hace falta mirar la métrica
   secundaria, que **no empata**: 0,40 → 0,19 → **0,08 px**. El error sigue bajando con `k` sin
   señal de saturar.
3. **`k05` es el punto interesante del eje**: acierta a ≤2 px el 92,3 % de las veces pero sólo el
   13,5 % a ≤1 px. Encuentra la esquina y no la clava. La transición de «no puede» (k=3) a «lo
   clava» (k=7) ocurre entera entre 3 y 7.

## Lo que se ve en las muestras

[`muestras/`](muestras/) tiene las 10 muestras congeladas en una imagen por brazo, antes
(`-ep000-sin-entrenar`) y después (`-ep300`). En `k07-ep300.png`: **5/5 aciertos**, `existe` a
1,00 en las positivas y **0,00 en las cinco negativas** —incluidas las otras dos esquinas del
mismo párrafo, que son el negativo difícil—. Y el mapa de respuesta enseña qué aprendió el
kernel: responde **negativo sobre la tinta**, y el máximo cae justo en la esquina.

## Qué aprendió el kernel: es un detector de CUADRANTE, como se predijo

El producto del experimento no es la red: es **el filtro**. Éstos son los 49 pesos de `k07`
(sin bias, la convolución no lo tiene) — [`muestras/kernel-k07.png`](muestras/kernel-k07.png):

```
  -1.360  -1.572  -1.232  -2.700  -3.661  -3.205  -3.199     <- espera FONDO arriba
  -0.914  -1.064  -0.786  -1.005  -1.109  -1.136  -2.006
  -0.786  -0.741  -0.717  -1.054  -0.632  -0.732  -1.196
  -4.995  -1.169  -0.657  -1.598  -0.516  -1.225  -2.460
  -9.433  -1.655  -2.754  +0.267  +2.319  +0.104  -1.111
  -6.457  -1.506  -4.376  +1.960  -0.245  -0.083  +2.054     <- espera TINTA abajo-derecha
  -5.467  -1.929  -1.792  -0.456  +0.442  -0.482  +0.346
     ^ espera FONDO a la izquierda
```

**Fila superior y columna izquierda muy negativas; cuadrante inferior-derecho positivo.** Es
literalmente la forma que el encargo predijo *antes* de entrenar nada, y por el argumento de que
un filtro lineal orientado no puede picar en las cuatro esquinas a la vez:

> «Un filtro lineal que pique en una esquina superior-izquierda tiene forma de **cuadrante**
> —positivo donde espera tinta, negativo donde espera fondo—.»
> — [`instrucciones/01-encargo.md`](instrucciones/01-encargo.md)

⚠ **Y la asimetría no es decorativa**: el peso más fuerte de todos es el **−9,43 de la columna
izquierda**, casi 4× el positivo más grande. El filtro gasta más en comprobar que *no hay tinta a
la izquierda* que en comprobar que *sí la hay abajo*. Tiene sentido para separar una esquina
superior-izquierda de un borde superior, que es el negativo que sólo se distingue por ese lado.

## La transformación aplicada a 20 entradas nunca vistas

[`nn/transformacion.py`](nn/transformacion.py) expone la función suelta —`cargar_kernel()` y
`aplicar()`—, que acepta **cualquier tamaño de entrada**, no sólo la ventana 32×32 con la que se
entrenó: una transformación que sólo sirve para la forma exacta del entrenamiento no es una
transformación, es una capa. La conversión a tinta y el `/255` van **dentro** de `aplicar()`,
porque pasarle la imagen sin invertir da un mapa con el signo cambiado y una figura que parece
razonable.

[`muestras/transformacion-k07-20-entradas.png`](muestras/transformacion-k07-20-entradas.png) —
20 ventanas de la partición `muestra`, fuera de train y de val:

- **Las 8 entradas `esquina-tl` tienen su máximo (rojo) justo sobre la esquina verdadera.**
- **Las 12 negativas no tienen máximo positivo en ninguna parte** — ni el interior, ni el fondo,
  ni el borde superior, ni las **otras tres esquinas** del párrafo, que son el negativo por
  orientación.

⚠ **Los dos signos se dibujan a escalas independientes, y hay que decirlo.** El kernel suma
**−73,68**, o sea que responde muy negativo a toda la tinta; con una escala única el pico
positivo —que es lo único que la cabeza lee— queda aplastado y la figura parece un detector de
tinta en vez de un detector de esquina. Pasó en la primera versión de esta figura.

## ...y la misma función sobre 10 PÁGINAS ENTERAS: aciertan las 10

*Medido el 2026-09-07 con `.venv/bin/python nn/transformacion.py --paginas 10` — 16 s de reloj,
0 máquinas, 0 $.*

[`muestras/transformacion-k07-10-paginas.png`](muestras/transformacion-k07-10-paginas.png) — las
**10 primeras páginas de la partición `muestra`**, enteras (200×200 px reducidos) y fuera de train
y de val, cada una con su mapa de respuesta en las mismas coordenadas.

**Es la misma `aplicar()` sin tocar nada, y ahí está el punto de que sea una transformación y no
una capa.** Lo que cambia es el problema, y cambia de verdad: entrenando, cada ventana traía
**una** esquina y siempre con contexto a los dos lados; una página entera trae las cuatro
esquinas, los cuatro bordes, el interior, todo el fondo y **el comienzo de cada línea** — y hay
**un solo máximo** para las 37.636 posiciones del mapa (194×194) en vez de para las 676 de una
ventana.

| página (`s`) | 0 | 1 | 2 | 3 | 4 | 6 | 7 | 8 | 9 | 10 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **distancia del máximo a la esquina (px)** | 0,70 | 0,54 | 0,58 | 0,49 | 0,42 | 0,49 | 0,62 | **0,87** | 0,28 | 0,64 |
| pico del mapa | 1,75 | 1,62 | 2,77 | 2,14 | 1,91 | 1,50 | 1,60 | 2,03 | 2,15 | 2,02 |

**En las 10 el máximo cae a ≤1 px de la esquina superior-izquierda verdadera**; el peor, a 0,87 px.

**Lo que esto añade** sobre la figura de las 20 ventanas es justo lo que no estaba medido: la
cabeza que leía el mapa nunca vio más de 32×32, así que *«el máximo de la página entera cae en la
esquina buena»* era una extrapolación razonable — con las otras tres esquinas y todos los
comienzos de línea compitiendo — y ahora es una medida.

⚠ **El suelo de la medida es ~0,5 px, así que estos diez números NO se ordenan entre sí.** La
esquina verdadera sale del DOM y es un decimal; el máximo del mapa es un píxel entero. Un detector
perfecto mediría igualmente entre 0 y ~0,7 px. Lo que dice la tabla es que **las diez están por
debajo de 1 px**, no que la página 9 sea mejor que la 8.

⚠ **Y lo que NO dice:** son 10 páginas de **una** receta (un párrafo por página, limpio, sin girar
y sin ruido) y **una** semilla. Cada página tiene **exactamente una** esquina superior-izquierda,
así que «el máximo» es un lector válido aquí y sólo aquí: con dos párrafos habría que leer picos
sobre un umbral, y ni el umbral ni la separación entre picos están medidos.

## Lo que quedó pendiente

- **El eje no está acotado por arriba.** `k11` sigue mejorando y es el borde del rango. Si
  interesa el error de posición y no sólo el acierto, hay que mirar más allá de 11.
- **Una semilla.** Los cinco brazos comparten la semilla 1. Nadie ha medido la variación entre
  semillas, así que las diferencias pequeñas (k09 contra k11) no se pueden declarar.
- **El techo de la métrica.** Si se repite, el umbral de acierto debería bajar a ≤1 px o ≤0,5 px,
  que es donde los brazos todavía se distinguen.
- **Una página con VARIOS párrafos.** Con uno solo, «el máximo del mapa» es un lector
  válido; con dos o más hay que leer picos sobre un umbral, y ni ese umbral ni la
  separación entre picos están medidos.
- **Nada de esto dice nada del `stride`**, que es el eje siguiente y no se ha tocado.

## Cómo se repite

```bash
cd ~/src/experimentos-cnn
uv venv && uv pip install -e . && uv pip install torch --index-url https://download.pytorch.org/whl/cpu
# El dato se RINDE con el navegador del generador, así que el venv necesita además esto:
uv pip install playwright pydantic
.venv/bin/python -m playwright install chromium               # ~115 MB
sudo .venv/bin/python -m playwright install-deps chromium     # libatk-1.0 y compañía
E=2026-09-06-esquina-kernel-unico
.venv/bin/python $E/nn/datos.py --imagenes 300     # ~2,5 min (necesita el generador y Chromium)
.venv/bin/python $E/nn/datos.py --comprobar        # ¿se re-deriva el mismo dato? SÍ, comprobado
for b in k03 k05 k07 k09 k11; do
  .venv/bin/python $E/nn/entrenar_local.py --brazo $b --epocas 300   # reanudable: se corta y sigue
done
.venv/bin/python $E/nn/muestras.py --etiqueta ep300
.venv/bin/python $E/nn/transformacion.py --kernel --entradas 20 --paginas 10
```

⚠ **Las cuatro líneas de `playwright` faltaban aquí, y se descubrieron a mitad** (2026-09-07,
en una máquina recién hecha). `uv pip install -e .` no las trae porque `pyproject.toml` **no declara
dependencias a propósito**, y sin ellas `datos.py` y `--paginas` no fallan al empezar: fallan al
importar `app.core.renderer`, o al lanzar el navegador con `libatk-1.0.so.0: cannot open shared
object file`. Es la regla del preflight de este proyecto: comprueba estado **utilizable**, no
presencia.

⚠ **Y `cdn.playwright.dev` SÍ sirvió** desde este droplet el 2026-09-07 (115 MB, sin un solo error),
al contrario de lo medido el 2026-08-27 desde `nyc1`, donde daba **403** y hubo que instalar el `.deb`
de Google Chrome. O sea que ese 403 **no es permanente**: se prueba primero el camino normal y sólo
se cae al `.deb` si falla.

⚠ **El dataset SE REPRODUCE**, y está comprobado el 2026-09-06 ejecutando `--comprobar`: las tres
particiones dan la misma huella SHA-256 tras regenerarlas de cero. Por eso el `.npz` no se
commitea y sí su huella. **No se dio por supuesto**: en este sistema ya se dio por reproducible
un dataset que no lo era.

## Por qué esto NO se corrió en Vast

El dueño pidió lanzarlo en Vast. **No se hizo, y el motivo es aritmética**: una época cuesta
**0,15–0,35 s** aquí, o sea los 5 brazos × 300 épocas son **~9 min**; y arrancar *una sola*
máquina de Vast cuesta **8,4 min medidos** de peaje (arranque + subida + instalar torch,
`foveal-vision/scripts/estudio_estimar.py:53`). Con 5 máquinas serían **42 min-máquina de peaje
para 9 min de trabajo**. A eso se suma que la tercera pregunta del freno —*quién apaga las
máquinas si este droplet muere*— hoy **no tiene respuesta automática** para este repo.
