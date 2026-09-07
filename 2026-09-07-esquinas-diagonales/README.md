# `esq-2d` — con UN solo kernel, ¿qué estructura detecta las DOS esquinas en diagonal?

**PREPARADO el 2026-09-07, SIN ENTRENAR NI UNA ÉPOCA.** Aquí están las estructuras, el dataset,
los suelos medidos y el criterio congelado. Lo que todavía no hay es resultado — y ésa es la
diferencia entre este documento y el de `esq-k`.

Hereda de `esq-k` **todo lo que decide la comparabilidad**: misma receta, semilla 1, ventana
32×32, reducción por 4, sin padding, sin bias, cabeza C1 (esperanza bajo `softmax(β·M)`, β
aprendida desde 3,5), Adam `lr` 0,05, lote 128, 300 épocas.

**El diseño tiene UN eje: el tamaño del kernel.** Cinco brazos, una sola estructura, y lo único
que cambia entre ellos es `k`. Las otras lecturas que se propusieron quedan **anotadas y sin
correr**, por orden del dueño del 2026-09-07 — en
[`instrucciones/03-alternativas-anotadas.md`](instrucciones/03-alternativas-anotadas.md), con lo
que cada una contestaría y lo que cuesta.

## El problema, que no es el de ayer con otra etiqueta

Un filtro lineal que pica en una esquina superior-izquierda tiene forma de **cuadrante** y esa
forma es **orientada**: la esquina inferior-derecha es **la misma forma girada 180°**. Partiendo
el kernel bajo ese giro (`W = S + A`, con `S` simétrica y `A` antisimétrica), para el mismo
parche:

```
respuesta_tl = <S,P> + <A,P>          respuesta_br = <S,P> − <A,P>
```

Las dos respuestas son **simétricas respecto de `<S,P>`**, así que **toda la diferencia entre
esquinas vive en `A`**. Las estructuras salen de ahí, no de una lluvia de ideas.

⚠ **Y hay un dato en contra de la estructura obvia, medido antes de diseñar nada.** El kernel
ganador de `esq-k` —que sólo vio esquinas tl— tiene el **74,0 % de su energía en `S`** y suma
**−73,68**: es sobre todo un **supresor de tinta**. Medido sobre las 10 páginas enteras de aquel
experimento, **su mínimo cae en la mancha de tinta, no en la esquina br: 0/10, mediana 91 px**.
El comando está en [`instrucciones/01-encargo.md`](instrucciones/01-encargo.md).

## La estructura: `rot`, un kernel y la misma cabeza sobre la vista girada

**Una sola convolución de `k×k` sin bias, aplicada dos veces con los MISMOS pesos**: a la entrada
y a la entrada girada 180°. Una esquina inferior-derecha es, literalmente, una superior-izquierda
en la vista girada — así que la cabeza también es la misma, y son **3 parámetros de cabeza**,
exactamente los de `esq-k` con una sola esquina.

```
entrada x (1, 32, 32) en TINTA (/255)
    │
    ├── conv(x, W)              ──► M⁺  (m×m, m = 32 − k + 1)  ──► cabeza C1 ──► (existe_tl, x_tl, y_tl)
    │
    └── conv(girar180(x), W)    ──► M⁻  (m×m)                  ──► cabeza C1 ──► (existe_br, x_br, y_br)
             ▲ el MISMO W                    ▲ la MISMA cabeza        ▲ y las coordenadas se
                                               (β, a, b)                devuelven al marco original:
                                                                        x = 31 − x_girada

cabeza C1:  p = softmax(β·M) · (x, y) = esperanza de la posición bajo p
            existe = a · logsumexp(β·M)/β + b            β se aprende, arranca en 3,5
```

| brazo | k | mapa | kernel | cabeza | **total** |
|---|--:|---|--:|--:|--:|
| `rot-k05` | 5 | 28×28 | 25 | 3 | **28** |
| `rot-k07` | 7 | 26×26 | 49 | 3 | **52** |
| `rot-k09` | 9 | 24×24 | 81 | 3 | **84** |
| `rot-k11` | 11 | 22×22 | 121 | 3 | **124** |
| `rot-k13` | 13 | 20×20 | 169 | 3 | **172** |

```bash
python nn/modelo.py     # imprime esta tabla, las alternativas anotadas, y comprueba las dos
```

⚠ **Es exactamente la red de `esq-k`, con el doble de tarea.** Mismo `k²+3`, misma cabeza, misma
convolución. Por eso este barrido se puede leer **uno a uno** contra el de allí sin traducir nada
— que es lo que se perdería con cualquier otra de las lecturas propuestas, que necesitan 2
parámetros más de cabeza y una lectura distinta del mapa.

⚠ **La equivarianza es exacta y hay un test que la fija**: girar la entrada 180° intercambia las
dos esquinas predichas (`assert` en `nn/modelo.py`). Si eso se rompiera, el brazo dejaría de ser
lo que dice ser y se leería como «aprende peor» en vez de como un fallo.

⚠ **Y por eso las dos esquinas son la MISMA tarea**, no dos parecidas. Es la propiedad que hace
que un solo kernel pueda con las dos sin que ninguna de las dos partes del filtro tenga que ceder
— que es el problema que tienen las lecturas de un solo mapa, y está medido en el encargo.

### El eje: `k` ∈ {5, 7, 9, 11, 13}

⚠ **El 3×3 se cae con un dato, no por gusto** (y es el propio dueño quien lo saca): en `esq-k` se
quedó en el 8,3 % de acierto, que es **exactamente su suelo sin entrenar**. No aprendió poco: no
aprendió. Aquí la tarea es más difícil, así que repetirlo sería pagar por re-confirmar al
perdedor. **El 13 entra en su lugar** porque aquel eje no estaba acotado por arriba: el 11 seguía
mejorando y era el borde del rango.

⚠ **Y 13 no es un tope caprichoso: es lo que este dataset admite.** La esquina se sortea entre los
píxeles 8 y 23 de la ventana, y el mapa de un kernel `k` sólo representa de `(k−1)/2` a
`31−(k−1)/2`. Con **k = 17** los dos rangos coinciden exactamente; con k = 19 habría esquinas que
el mapa **no puede** señalar. Cabe 15; a partir de 19 hay que regenerar el dato. Lo calcula y lo
escribe el manifiesto, no se supone.

### Lo que se propuso y NO se corre

`sig` (un mapa: `tl` = máximo, `br` = mínimo), `ant` (lo mismo con el kernel forzado
antisimétrico) y el control `ind-tl`/`ind-br` (dos redes sin compartir nada). **Siguen
implementadas y comprobadas** en `nn/modelo.py` —una estructura implementada es la forma menos
ambigua de anotarla, y así no se pudre en silencio—, pero no están armadas: ponerlas en marcha es
añadir su línea a `BRAZOS`.

⚠ **Lo que se pierde, dicho por delante:** si el barrido sale bien, nada. Si sale mal, este diseño
**no puede distinguir** *«un kernel no da para las dos esquinas»* de *«esta lectura no es la
buena»*. Está escrito en el criterio como desenlace 2.

## El dataset vive en el repo de DATOS, y es siempre el mismo fichero

**Por orden del dueño (2026-09-07): «Guarda los datasets en el repo de data, de modo que sean
siempre los mismos, por consistencia».**

```
foveal-vision-data/experimentos-cnn/esquinas300-32px-r4-r20260907/
    train.npz · val.npz · muestra.npz · muestras-congeladas.npz · manifiesto.json · README.md
```

Se resuelve con `expcnn.exigir_dataset(...)`, que es **la única** puerta: si no está publicado,
el entrenamiento **se niega antes de empezar** en vez de generarse uno equivalente. Ahí está el
punto — un dato re-derivado es el mismo *mientras nada cambie*, y «nada cambia» no es comprobable
hacia el futuro; publicado, es el mismo **porque es el mismo fichero**.

⚠ **No va en este repo y no es una preferencia:** éste es **público** y el de datos es
**privado**. Y no va en `window-datasets/`, que es de `foveal-vision` y lo resuelve su propio
`settings`: meter ahí un dataset de otra forma sería una colisión silenciosa.

⚠ **Y de paso tapa un agujero real.** Las 10 muestras congeladas de `esq-k` se declaraban
commiteadas y **no lo estaban**: el `*.npz` del `.gitignore` de este repo —que existe para que no
se cuele el dato de entrada— se llevaba también la verificación. Medido el 2026-09-07 en el clon
limpio de esta máquina: aquel `nn/muestras.npz` no existe, así que su figura de verificación no
se podía regenerar sin volver a rendir 300 imágenes. Publicadas con el dataset, dejan de perderse.

### Qué trae

Mismas imágenes que `esq-k` (misma receta, misma semilla): 267 válidas de 300. Diez ventanas por
imagen: **2 `tl` + 2 `br`** positivas y **6 negativas** (las esquinas de la otra diagonal `tr` y
`bl`, que son el negativo duro; borde superior, borde inferior, interior y fondo).

⚠ **Se etiquetan LAS CUATRO esquinas**, no sólo la diagonal que mide `esq-2d`. No cuesta nada —
las cuatro coordenadas ya se conocen al recortar— y es lo que hace que *«si pueden usar el mismo
dataset no hay problema»* sea cierto también para el experimento siguiente: el que mire la otra
diagonal, o las cuatro, no tiene que re-rendir nada. `esq-2d` lee `tl` y `br` e ignora el resto.

⚠ **Ninguna ventana contiene más de una esquina** (párrafo ≥ 64 px, ventana 32). Se **cuenta** al
generar y se escribe en el manifiesto, no se supone.

```bash
python nn/datos.py --imagenes 300 --publicar   # genera y publica (no pisa lo publicado)
python nn/datos.py --comprobar                 # ¿el publicado es el del manifiesto? (instantáneo)
python nn/datos.py --rederivar                 # ¿la receta y la semilla lo vuelven a dar? (~6 min)
```

**`--publicar` se niega a pisar un dataset ya publicado**: dato nuevo = nombre nuevo, que es la
regla que el repo de datos ya tiene escrita. Si publicar pudiera sobrescribir, un `--publicar`
distraído cambiaría el dato bajo los pies de todo lo ya medido, sin un solo error.

## El criterio, congelado antes de mirar

En [`instrucciones/02-criterio.md`](instrucciones/02-criterio.md), con los suelos medidos sobre
las redes **sin entrenar**: **3,8–5,1 % (`tl`) y 6,4 % (`br`)** a ≤2 px, que son exactamente el
predictor constante en el centro. Umbral de «ha aprendido algo»: **12 % en las dos esquinas**. El
titular es la **peor** de las dos, nunca el promedio.

Y sin los brazos de control, el techo se lee de `esq-k`, que midió la tarea de **una** esquina con
esta misma red: 92,3 % (k=5) · 100 % (7 · 9 · 11). ⚠ Es una **referencia, no un control**: sus
ventanas son otras, así que una diferencia de pocos puntos no se puede atribuir a compartir el
kernel.

## Cómo se corre (cuando se ordene)

```bash
cd ~/src/experimentos-cnn
E=2026-09-07-esquinas-diagonales
.venv/bin/python $E/nn/datos.py --imagenes 300        # ~6 min (generador + Chromium)
.venv/bin/python $E/nn/entrenar_local.py --suelos     # los suelos, sin entrenar nada
for b in rot sig mag sig11 ind-tl ind-br; do
  .venv/bin/python $E/nn/entrenar_local.py --brazo $b --epocas 300   # reanudable
done
.venv/bin/python $E/nn/muestras.py --etiqueta ep300
```

Las dependencias del venv están en el README de `esq-k` § «Cómo se repite» (son las mismas, y
`uv pip install -e .` **no** las trae todas).

## Cuánto cuesta (estimado, no medido)

**0 máquinas y 0 $**: entrena en este droplet, como `esq-k`. Lo que cuesta es reloj —
*medido el 2026-09-07 a máquina libre*: `rot` va de **0,55 s/época** (k=5) a **1,02** (k=13), así
que los **cinco brazos × 300 épocas ≈ 20 min** *(estimado a partir de tres medidas, no medido
entero)*.

⚠ Y una lección de medición, porque el primer número que di estaba mal: la misma prueba **con la
máquina rindiendo el dataset a la vez** daba 1,2–2,7 s/época, o sea **~3× más**. En un droplet de
2 vCPU, un tiempo por época medido con algo más corriendo no es el tiempo por época.

Y en disco: 5 brazos × (2 checkpoints de ~17 KB + un `metrics.jsonl` de ~50 KB) ≈ **0,4 MB**,
holgado dentro del tope de ~5 MB por experimento que declara el `CLAUDE.md` del repo. El registro
se guarda **redondeado a 6 cifras** de todas formas: era necesario con 25 brazos y no estorba con
5.

⚠ El freno lo ve: `entrenar_local.py` está en la lista `TRABAJOS` de
`telegram-coordinator/scripts/cerrable.mjs`, así que un entrenamiento vivo aparece en el veredicto
que se lee desde el móvil. No hace falta ejecutor nuevo de Telegram: `gasta` es `entrena-local`,
no `alquila`.

## Lo que este experimento NO contesta

- **Nada sobre señalar las dos esquinas a la vez en la misma vista.** La ventana es 32×32 y el
  párrafo ≥ 64 px, así que ninguna ventana contiene las dos. La pregunta de aquí es *«¿puede un
  kernel servir a las dos?»*; la de la página entera es el experimento siguiente.
- **Nada sobre las otras dos esquinas** (`tr`, `bl`): aquí son negativos.
- **Una semilla.** Igual que en `esq-k`, y con el mismo precio: las diferencias pequeñas entre
  brazos no se van a poder declarar.
