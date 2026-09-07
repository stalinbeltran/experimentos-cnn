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
| [`2026-09-07-esquinas-diagonales/`](2026-09-07-esquinas-diagonales/) `esq-2d` | cerrado | Una sola convolucion sin bias produce UN mapa, y las dos esquinas en diagonal se leen de sus dos extremos: la superior-izquierda del maximo y la inferior-derecha del minimo. Que tamano de kernel lo hace mejor, y cuanto se pierde respecto del mismo barrido sobre UNA sola esquina de esq-k |

<!-- FIN INDICE -->

## Cómo se añade uno

1. `mkdir 2026-09-06-<nombre>` y copia dentro `experimento.ejemplo.json` como
   `experimento.json`, rellenado.
2. Escribe el **criterio antes de mirar** en `instrucciones/02-criterio.md`.
3. Lo demás lo decide el experimento. `python3 comprobar.py` te dice si algo está mal puesto.

Las reglas, la frontera de lo que un experimento puede decidir y dónde va cada artefacto
están en [`CLAUDE.md`](CLAUDE.md).
