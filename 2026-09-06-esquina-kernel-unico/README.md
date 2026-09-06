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

## Lo que quedó pendiente

- **El eje no está acotado por arriba.** `k11` sigue mejorando y es el borde del rango. Si
  interesa el error de posición y no sólo el acierto, hay que mirar más allá de 11.
- **Una semilla.** Los cinco brazos comparten la semilla 1. Nadie ha medido la variación entre
  semillas, así que las diferencias pequeñas (k09 contra k11) no se pueden declarar.
- **El techo de la métrica.** Si se repite, el umbral de acierto debería bajar a ≤1 px o ≤0,5 px,
  que es donde los brazos todavía se distinguen.
- **Nada de esto dice nada del `stride`**, que es el eje siguiente y no se ha tocado.

## Cómo se repite

```bash
cd ~/src/experimentos-cnn
uv venv && uv pip install -e . && uv pip install torch --index-url https://download.pytorch.org/whl/cpu
E=2026-09-06-esquina-kernel-unico
.venv/bin/python $E/nn/datos.py --imagenes 300     # ~2,5 min (necesita el generador y Chromium)
.venv/bin/python $E/nn/datos.py --comprobar        # ¿se re-deriva el mismo dato? SÍ, comprobado
for b in k03 k05 k07 k09 k11; do
  .venv/bin/python $E/nn/entrenar_local.py --brazo $b --epocas 300   # reanudable: se corta y sigue
done
.venv/bin/python $E/nn/muestras.py --etiqueta ep300
```

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
