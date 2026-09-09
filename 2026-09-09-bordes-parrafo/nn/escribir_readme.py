#!/usr/bin/env python3
"""Escribe el README.md del experimento LEYENDO el manifiesto. No se transcribe nada.

    python nn/escribir_readme.py

Existe por la misma razon que `banco-k` genera sus informes en vez de copiarlos: una
tabla transcrita a mano es como nacen los numeros que nadie puede auditar, y aqui hay
dos sitios que podrian divergir -- el README del dataset (que vive en el repo de datos)
y este, que vive en el repo publico y NO puede incluir el dato.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
sys.path.insert(0, str(EXP.parent))
sys.path.insert(0, str(AQUI))

import geometria as G  # noqa: E402

from expcnn import ruta_dataset  # noqa: E402

NOMBRE = json.loads((EXP / "experimento.json").read_text(encoding="utf-8")).get(
    "dataset") or json.loads(
    (EXP / "experimento.json").read_text(encoding="utf-8"))["dataset_previsto"]


def main() -> int:
    p = ruta_dataset(NOMBRE) or (EXP / "datos")
    m = json.loads((p / "manifiesto.json").read_text(encoding="utf-8"))
    g, med, f, rep = m["geometria"], m["medido"], m["factores"], m["reparto"]
    ppp = " · ".join(f"{k}→{v}" for k, v in sorted(m["parrafos_por_pagina"].items()))

    (EXP / "README.md").write_text(f"""# `bor-p` — bordes de párrafo

**El dato de entrada para aprender kernels de hasta {g['k_max']} px que localicen los
cuatro bordes del recuadro que encierra un párrafo.**

Estado: **`abierto`**. El dataset está hecho, publicado y comprobado. **No hay red, ni
criterio, ni ninguna cifra medida sobre kernels**, y eso es el encargo —
*«Por ahora solo obtén las páginas»* — no un hueco.

⚠ Este README **se genera** (`python nn/escribir_readme.py` leyendo el manifiesto). No
se edita a mano: una tabla transcrita es como nacen los números que nadie puede auditar.

## Qué hay

| | |
|---|---|
| dataset | `{m['nombre']}`, en `foveal-vision-data/experimentos-cnn/` (**repo privado**) |
| páginas | **{m['paginas']}** de {m['lienzo']} × {m['lienzo']} px, gris de 1 canal, fondo blanco sólido |
| párrafos | **{m['parrafos']}** ({ppp} por página) |
| etiqueta | 4 enteros por párrafo: `izq, der, sup, inf` — la caja de **TINTA** |
| descartadas | **{m['descartadas']}** de {m['paginas_planeadas']} planeadas |
| coste | {m['renders']} renders, {m['segundos']} s |

⚠ El `.npz` **no está aquí y no puede estarlo**: este repo es público y el de datos
privado. Se lee con `expcnn.exigir_dataset("{m['nombre']}")`.

## Las tres garantías, todas derivadas de `k_max = {g['k_max']}`

| | mínimo duro | medido | por qué ese mínimo |
|---|---|---|---|
| separación entre párrafos | {g['separacion_dura']} px | **{med['separacion_min']}** (mediana {med['separacion_mediana']}) | `2·{g['radio']}+1`: las ventanas {g['k_max']}×{g['k_max']} de dos bordes enfrentados no se solapan |
| margen al borde del papel | {g['margen_duro']} px | **{med['margen_min']}** (mediana {med['margen_mediana']}) | `valid` con k={g['k_max']} sólo cubre `[{g['radio']}, {m['lienzo']-g['radio']-1}]` |
| solape entre párrafos | ninguno | **ninguno** | se construye, no se filtra |

⚠ La métrica es **L-infinito**, no euclídea: es la que corresponde a una ventana
**cuadrada**. Dos cajas a 14 px en diagonal están a 19,8 euclídeos y una ventana de
{g['k_max']}×{g['k_max']} **sí las mezcla**.

## Cuánto se puede recortar

Las ventanas de entrenamiento se recortan **después**, y su presupuesto ya viaja con el
dato: `derivadas` de `cajas.npz` trae `ventana_limpia` por párrafo.

| mínimo | percentil 10 | mediana |
|---|---|---|
| **{med['ventana_limpia_min']} px** | {med['ventana_limpia_p10']} px | {med['ventana_limpia_mediana']} px |

Una ventana de hasta **{med['ventana_limpia_min']}×{med['ventana_limpia_min']}** centrada
en cualquier borde es limpia para **todos** los párrafos, sin filtrar ninguno.

⚠ Y el reparto `train`/`val`/`eval` es **por página** ({rep['paginas']['train']} ·
{rep['paginas']['val']} · {rep['paginas']['eval']}), no por párrafo: dos ventanas de la
misma página comparten fuente, fondo y vecinos, así que repartirlas sería una fuga que no
falla por ningún lado.

## Variación

| factor | rango usado |
|---|---|
| fuente | {', '.join(f['fuentes'])} |
| cuerpo | {f['cuerpo_px_usado'][0]}–{f['cuerpo_px_usado'][1]} px |
| interlineado | {f['interlineado_usado'][0]}–{f['interlineado_usado'][1]} |
| ancho de tinta | {med['ancho_tinta'][0]}–{med['ancho_tinta'][1]} px |
| alto de tinta | {med['alto_tinta'][0]}–{med['alto_tinta'][1]} px |
| palabras | {f['palabras_usado'][0]}–{f['palabras_usado'][1]} |
| color del texto | {f['color_texto']} |

## ⚠ La reserva de `banco-k` (§3.7): RESPETADA

Es **el requisito que motivó todo esto**. `banco-k` reserva `LiberationMono` y el
interlineado `{m['reserva_banco_k_§3.7']['interlineado_prohibido']}` para uso exclusivo
suyo; un kernel aprendido sobre datos que los alcancen sale **optimista** al evaluarlo
allí. Este dataset usa las otras cuatro familias e interlineado
`{m['reserva_banco_k_§3.7']['interlineado_usado']}`.

Comprobado ejecutando el chequeo **del propio banco** (`banco-k/nn/importar_kernel.py`):

```
bor-p    usa_la_reserva=False   ← este experimento
esq-k    usa_la_reserva=True    ← el vecino, que sí filtra
```

## Cómo se repite

```bash
python nn/geometria.py                                   # 27 comprobaciones, sin render
python nn/generar_paginas.py --reserva                   # el contrato §3.7, solo
python nn/generar_paginas.py --parrafos {m['parrafos']}            # ~{round(m['segundos']/60)} min
python nn/generar_paginas.py --publicar                  # una vez en la vida
python nn/generar_paginas.py --comprobar                 # casa las huellas del publicado
python nn/generar_paginas.py --muestras 6                # la figura con las cajas encima
```

⚠ Se corre con el venv **del generador** (`~/src/image-text-sample-generator/.venv`), que
es el que trae Playwright. `nn/geometria.py` no necesita nada: es stdlib pura.

⚠ Y lo largo va desacoplado, que es la regla del proyecto para cualquier cosa lanzada
desde una sesión de Claude Code:

```bash
COORD_HOME="$HOME/src/telegram-coordinator" \\
  "$HOME/src/telegram-coordinator/scripts/desacoplar-persistente.sh" borp-dataset \\
  sh -c 'cd <esta carpeta> && <venv>/bin/python nn/generar_paginas.py --parrafos 1000'
```

Las condiciones, las decisiones y qué **no** se hereda de nadie, en
[`REGLAS.md`](REGLAS.md). El encargo literal y lo que se midió antes de escribir código,
en [`instrucciones/01-encargo.md`](instrucciones/01-encargo.md).
""", encoding="utf-8")
    print(f"escrito {EXP / 'README.md'} desde {p / 'manifiesto.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
