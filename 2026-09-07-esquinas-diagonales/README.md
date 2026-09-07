# `esq-2d` — con UN solo kernel, ¿qué estructura detecta las DOS esquinas en diagonal?

**PREPARADO el 2026-09-07, SIN ENTRENAR NI UNA ÉPOCA.** Aquí están las estructuras, el dataset,
los suelos medidos y el criterio congelado. Lo que todavía no hay es resultado — y ésa es la
diferencia entre este documento y el de `esq-k`.

Hereda de `esq-k` **todo lo que decide la comparabilidad**: misma receta, semilla 1, ventana
32×32, reducción por 4, sin padding, sin bias, cabeza C1 (esperanza bajo `softmax(β·M)`, β
aprendida desde 3,5), Adam `lr` 0,05, lote 128, 300 épocas.

**El diseño tiene DOS ejes: la estructura de lectura × el tamaño del kernel.** El que contesta la
pregunta es el primero; el segundo está para que *«esta lectura no puede»* no se confunda nunca
con *«con este kernel no cabe»* — que es justo la duda que `esq-k` dejó abierta al quedarse en el
borde de su rango.

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

## Las estructuras

| brazo | idea | kernel | cabeza | total |
|---|---|--:|--:|--:|
| **`rot`** | el **mismo** kernel sobre la entrada **y sobre la entrada girada 180°**. Una br es una tl en la vista girada, así que la cabeza también es la misma | 49 | 3 | **52** |
| **`sig`** | un mapa: **tl = máximo, br = mínimo**. Pide que mande `A` | 49 | 5 | **54** |
| **`ant`** | igual que `sig`, pero el kernel se **proyecta antisimétrico** (`W = (V − rot180(V))/2`): `S = 0` por construcción, así que `respuesta_br = −respuesta_tl` deja de ser una esperanza y pasa a ser **exacta**. La mitad de grados de libertad | **24** | 5 | **29** |
| `sig11` | `sig` con k = 11. **Punto de seguro, no parte del eje**: separa «esta lectura no puede» de «con 49 pesos no cabe» | 121 | 5 | 126 |
| `ind-tl` · `ind-br` | **control, no compite**: dos redes de un kernel cada una, sin compartir nada. Es el **techo** contra el que se mide lo que cuesta compartir | 49 | 3 | 52 ×2 |

```bash
python nn/modelo.py       # imprime esta tabla y comprueba que rot es equivariante
```

⚠ **`rot` es exactamente equivariante y hay un test que lo fija**: girar la entrada 180°
intercambia las dos esquinas predichas. Si eso se rompiera, el brazo dejaría de ser lo que dice
ser y se leería como «aprende peor» en vez de como un fallo.

⚠ **`ant` es antisimétrico exacto y hay un test que lo fija**: el kernel cumple
`W = −rot180(W)` con tolerancia 1e-7, y responde a un parche y a su giro con el mismo número
cambiado de signo. El precio de forzarlo está declarado: pierde la parte del kernel que **apaga
el interior del párrafo**, que es justo donde el kernel de `esq-k` gastaba el 74 % de su energía.

⚠ **Una cuarta estructura se descartó EN PAPEL, sin pagarla:** *«las dos esquinas son máximos y
se distinguen por el VALOR de la respuesta»*. No es expresable con esta cabeza — `existe` es una
función **lineal de un solo escalar**, o sea monótona, y esa hipótesis pide fondo 0 · `br` medio ·
`tl` alto, donde «`br`» sería una **banda** que ninguna recta separa. Darle dos escalares (el
máximo y el mínimo) es ya `sig`. El argumento entero está en la cabecera de
[`nn/modelo.py`](nn/modelo.py).

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
las redes **sin entrenar**. El titular es la **peor** de las dos esquinas, nunca el promedio.

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
*medido el 2026-09-07 a máquina libre, una época en nueve brazos*: **0,31–1,02 s/época** (sube
con `k` y con el número de convoluciones: `rot` hace dos). Los **25 brazos × 300 épocas ≈ 60–70
min** *(estimado a partir de esas nueve medidas, no medido entero)*.

⚠ Y una lección de medición, porque el primer número que di estaba mal: la misma prueba **con la
máquina rindiendo el dataset a la vez** daba 1,2–2,7 s/época, o sea **~3× más**. En un droplet de
2 vCPU, un tiempo por época medido con algo más corriendo no es el tiempo por época.

Y en disco: 25 brazos × (2 checkpoints de ~17 KB + un `metrics.jsonl` de ~50 KB) ≈ **2,1 MB**,
dentro del tope de ~5 MB por experimento que declara el `CLAUDE.md` del repo. Por eso el registro
se guarda **redondeado a 6 cifras** y las figuras de muestras **no salen para los 25** por
defecto: sin las dos cosas, sólo el historial serían ~3 MB.

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
