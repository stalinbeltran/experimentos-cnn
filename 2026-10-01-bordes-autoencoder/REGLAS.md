# Reglas de `bor-ae`

**Escritas el 2026-10-01** al montar la carpeta (fase 0d del plan
[`docs/plan-kernels-banco-2026-10-01.md`](../docs/plan-kernels-banco-2026-10-01.md), E2).
Estado **`abierto`**: código comprobado; **`λ` sin fijar** (lo fija el tanteo) y nada entrenado.

⚠ **Estas reglas son de este experimento y de ninguno más.** Las ventanas (`nn/datos.py`) se
**copiaron** de `bor-k` el mismo día; el resto se escribió aquí (§ «Qué NO hereda»).

**Qué pregunta:** con un solo filtro, una penalización de dispersión y un decodificador de
norma 1, ¿qué kernel sale de **reconstruir** las ventanas de borde de párrafo? ¿Algo distinto
de la identidad y de un pasa-bajos?

## Entradas

- **Dataset: `parrafos1000-pagina1024-r4-r20261001`** (`bor-p4`), leído con
  `expcnn.exigir_dataset(...)`. No se regenera.
- **Las ventanas son las de `bor-k`, copiadas:** lado 55, cuatro por párrafo (una por borde,
  desplazadas ±12) más una al azar por párrafo, semilla fija. **Sin etiqueta**: se usan todas
  como entrada a reconstruir. Partición del dataset, por página.
- **Qué se normaliza:** `x = 1 − suma/4080` (papel 0, tinta 1). Con el papel en 0, un código
  `ReLU` sin bias puede quedarse a cero donde no hay nada: es lo que hace que «disperso»
  signifique algo.

## Salidas

- **Pesos:** `nn/pesos/kNN-sS/` → `best.pt`, `last.pt`, `metrics.jsonl`. El **codificador** es
  `conv.weight` `(1, 1, k, k)`: lo que lee `importar_kernel.py --de bor-ae`.
- **El tanteo de `λ`:** `resultados/tanteo-lambda.json` (una fila por `λ`, con el kernel final y
  el veredicto de la regla). Se corre **una** vez y manda.
- **El libro de Vast:** `resultados/vast/fase2/`.
- **Qué se commitea:** todo lo de arriba. Nunca el dataset.

## Procesos

1. **El tanteo de `λ`** en el dev (`nn/entrenar_local.py --tanteo-lambda`: `k = 9`, semilla 1,
   60 épocas, `λ ∈ {0, 0,01, 0,03, 0,1, 0,3, 1}`) con la regla escrita **antes** en
   `instrucciones/02-criterio.md` y codificada tal cual en `regla_lambda()`. Si ningún `λ`
   cumple, `bor-ae` **no va al banco** y se le dice al dueño: no se rediseña sin él.
2. **Entrenar los 27 brazos en Vast** (`nn/vast.sh fase2`), con el `λ` congelado: una máquina
   por `k`, tres semillas a la vez, 300 épocas, Adam `lr` 0,01, lote 128.
3. **Importar al banco** los brazos cuyo codificador final **no** sea una delta
   (`delta ≤ 0,5`): una delta es la condición `identidad` del banco con otro nombre.

- **Qué se mide:** `R²` de la reconstrucción en la región `R`, `delta` del codificador,
  fracción de código activo. Criterio en `instrucciones/02-criterio.md`.
- **Brazos y semillas:** `k ∈ {3, …, 19}` × `{1, 2, 3}` = **27**.
- **«Ganar»:** no se declara. La utilidad la dice el banco.

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/datos.py` | las ventanas (copiadas de `bor-k`) | `python nn/datos.py` |
| `nn/modelo.py` | codificador + ReLU + decodificador de norma 1; `delta()` | `python nn/modelo.py` (identidad exacta en `R`) |
| `nn/entrenar_local.py` | un brazo, reanudable; y el tanteo | `--brazo k09-s1 [--hilos N]` · `--tanteo-lambda` · `--comprobar` |
| `nn/vast.json` · `nn/vast.sh` | la fase 2 en Vast | `fase2 [k…]` · `--estado` · `apagar` · `VAST_SECO=1 …` |

- ⚠ **`entrenar_local.py` se llama así por el FRENO**, y por eso el tanteo es un modo suyo y
  no un script aparte: mientras corre, `cerrable.mjs` lo ve.
- **Dependencias:** el `.venv` de la raíz (torch CPU).

## Las dos regiones, iguales para todo `k`

- **`R = [18, 36]`** (19×19): donde se mide la reconstrucción. Es donde el campo receptivo
  **entero** de codificador + decodificador (`k − 1 ≤ 18` px a cada lado) cae dentro de la
  ventana para todo `k ≤ 19`. Fuera de ahí, un `k` grande reconstruiría con menos soporte.
- **`O = [9, 45]`** (37×37): donde se cuenta la dispersión — las posiciones de código que
  alimentan `R`, que es además la región que lee `bor-k`.

## Qué NO hereda

- **Se copió de:** `bor-k`, sólo `nn/datos.py` (las ventanas), para que los tres experimentos
  aprendan de **las mismas** ventanas: es una decisión de este experimento, no una herencia.
- **Qué se cambió a propósito:** nada de las ventanas. Todo lo demás es propio: la red, la
  pérdida, la `lr` (0,01) y el criterio.
- **La restricción que SÍ es propia y no es de estilo:** el **decodificador de norma 1**. Sin
  ella, escalar el codificador por `s` y el decodificador por `1/s` divide `|z|` por `s`, y
  `λ` deja de significar nada.
- **Contra qué se compara:** con la `identidad` y el `gauss` del banco (es lo que más
  probablemente salga), y en el banco contra el aleatorio de igual norma y `k`.
- **Restricciones de otros experimentos que NO aplican aquí:** la sonda L1 de `foveal-vision`
  (2026-09-03) **no** se hereda; sólo se cita como la medida de que con `λ = 0` sale la
  identidad.
