# `esq-2d` — con UN solo kernel, ¿qué estructura detecta las DOS esquinas en diagonal?

**PREPARADO el 2026-09-07, SIN ENTRENAR NI UNA ÉPOCA.** Aquí están las estructuras, el dataset,
los suelos medidos y el criterio congelado. Lo que todavía no hay es resultado — y ésa es la
diferencia entre este documento y el de `esq-k`.

Hereda de `esq-k` **todo lo que decide la comparabilidad**: misma receta, semilla 1, ventana
32×32, reducción por 4, sin padding, sin bias, cabeza C1 (esperanza bajo `softmax(β·M)`, β
aprendida desde 3,5), Adam `lr` 0,05, lote 128, 300 épocas. **`k` = 7**, que es el que ganó allí.
Lo único que varía es **cómo se lee el mapa**.

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

## El dataset

El mismo de `esq-k` en imágenes —misma receta y misma semilla—, con **etiqueta de seis números**
por ventana: `(existe_tl, x_tl, y_tl, existe_br, x_br, y_br)`. Diez ventanas por imagen: **2 tl +
2 br** positivas y **6 negativas** (las esquinas de la otra diagonal `tr` y `bl`, que son el
negativo duro; borde superior, borde inferior, interior y fondo).

267 imágenes válidas de 300 · **2.157 ventanas de train** (432 tl · 432 br), **389 de val**
(78 · 78) y 120 de muestra. ⚠ Y está **comprobado, no supuesto, que ninguna ventana contiene las
dos esquinas**: lo cuenta el propio generador y lo escribe en el manifiesto.

El `.npz` **no se commitea**: se re-deriva de la receta y la semilla, y su huella SHA-256 vive en
[`nn/manifiesto.json`](nn/manifiesto.json). ✅ **Comprobado el 2026-09-07** regenerándolo entero
en un directorio aparte: las tres particiones dan la misma huella
(`python nn/datos.py --comprobar`, ~6 min). En este sistema ya se dio por reproducible un dataset
que no lo era, así que esto se ejecuta, no se supone.

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

**0 máquinas y 0 $**: entrena en este droplet, como `esq-k`. Lo que sí cuesta es reloj —
*medido el 2026-09-07 con una época de prueba por brazo y la máquina ocupada rindiendo el
dataset*: **1,2–2,7 s/época**, o sea **~40–80 min los seis brazos** a 300 épocas. Es bastante más
que los 9 min de `esq-k` (0,36 s/época) porque el mapa es mayor y hay dos lecturas por paso; con
la máquina libre bajará, pero **no se ha medido libre**.

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
