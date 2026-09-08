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
