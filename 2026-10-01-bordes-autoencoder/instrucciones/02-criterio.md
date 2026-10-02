# Criterio — escrito el 2026-10-01, ANTES de correr el tanteo de `λ` y ANTES de entrenar

Este fichero se escribe **antes de mirar** (R13). Lo que diga aquí manda sobre lo que se
vea después; si un número sale raro, se dice, pero la regla no se mueve.

## A. Cómo se fija `λ` (el tanteo)

**Por qué hace falta.** Con un solo filtro, el óptimo de un autoencoder sin dispersión es la
**identidad**: codificador y decodificador deltas reconstruyen perfecto (`R² = 1`). Lo midió
la sonda L1 de `foveal-vision` el 2026-09-03 (`estudios-redes-neuronales`, reporte
`2026-09-03-sonda-l1-tanteo-eje-k`, líneas 39–54 y 86–88). Una delta en el banco es la
condición `identidad`, que el banco ya tiene: importarla no mide nada. Así que `λ` no puede
elegirse a ojo, y la regla tiene que **detectar** la delta.

**El tanteo** (`python nn/entrenar_local.py --tanteo-lambda`): `k = 9`, semilla 1, 60 épocas,
el mismo bucle, la misma `lr` (0,01) y las mismas ventanas que el entrenamiento de verdad,
para `λ ∈ {0 · 0,01 · 0,03 · 0,1 · 0,3 · 1}`. Mide sobre `val`, al final:

| medida | qué es |
|---|---|
| `R²` | `1 − MSE/Var(x)` en la región `R` (19×19 central) |
| `delta` | fracción de la energía del codificador en su píxel más fuerte: **1 = delta** |
| `activa` | fracción del código > 0 en la región `O` |

**La regla:**

1. Son **candidatos** los `λ` con **`delta ≤ 0,5`**, **`R² ≥ 0,5`** y **`activa ≥ 1 %`**:
   ya no es la identidad, todavía reconstruye, y el código no está muerto.
2. Si hay candidatos, se elige **el MENOR**: la mínima presión de dispersión que ya aparta
   al filtro de la identidad. Un `λ` mayor sólo comprime más, y eso es otra pregunta.
3. **Si no hay ninguno, `bor-ae` no va al banco.** Querría decir que este autoencoder, con un
   filtro, da la identidad o nada, y eso **es** el resultado del tanteo. Se le dice al dueño
   con la tabla; no se cambia el diseño (código con ruido, *stride*, filtro atado…) sin que
   él lo decida, porque sería otro experimento.

La regla está codificada tal cual en `regla_lambda()` (`nn/entrenar_local.py`), y el tanteo
se niega a correr dos veces: su resultado, `resultados/tanteo-lambda.json`, manda.

⚠ **Lo que esta regla NO hace:** no elige el `λ` que dé el mejor kernel para el banco. No
puede: eso sólo lo sabe el banco, y elegir `λ` mirando el banco sería ajustar al instrumento.

## B. El entrenamiento (si el tanteo da un `λ`)

- **Brazos:** `k ∈ {3, 5, …, 19}` × semillas `{1, 2, 3}` = 27. 300 épocas, Adam `lr` 0,01,
  lote 128, sin parada temprana; `best.pt` = mínima pérdida de `val`.
- **Qué se reporta por brazo** (de `best.pt`, sobre `val` y sobre `eval`): `R²`, `delta`,
  `activa`, y la forma del kernel.
- **Qué va al banco:** el **codificador** (`conv.weight`) de cada brazo **cuyo `delta` final
  sea ≤ 0,5**. Un brazo que acabe en delta se reporta y **no se importa**, por lo mismo que
  en A: sería la `identidad` del banco con otro nombre.
- **No se declara ganador.** El veredicto de utilidad es el del banco (§2.1 de su
  especificación). Aquí sólo se dice qué filtro salió.

## C. Los riesgos, escritos antes

- **Pasa-bajos.** Con un solo filtro, lo que no es delta probablemente sea un pasa-bajos
  parecido a `gauss`. Si sale eso, es un resultado: se compara contra el `gauss` del banco.
- **`k = 3`.** Con 9 píxeles, un kernel razonable puede tener `delta` cerca de 0,5 sin ser
  una identidad. El umbral es el mismo para todos los `k` a propósito (un umbral por `k`
  sería elegirlo mirando); si `k = 3` cae fuera por eso, se dice así.

## Enmienda — 2026-10-02, ANTES de volver a correr el tanteo: el instrumento estaba roto

**El primer tanteo no midió nada del diseño: midió un fallo del entrenador.** Con `λ = 0`, donde
la identidad es una solución perfecta y alcanzable, salió `R² = −0,001`, `activa 0,0 %`: el
código **muerto**. Se paró tras ese primer `λ` (9 min por valor en el dev; los cinco restantes
sólo podían morir más rápido). Su log queda en `resultados/tanteo-lambda-ROTO-2026-10-02.log`.

**La causa, medida** (600 ventanas, `k = 9`, semilla 1): al iniciar, codificador y decodificador
salían con **signos opuestos** (`<e, d> = −0,14`); en la primera época el optimizador apagó el
código —suma del kernel de +0,34 a −2,5, código activo de 26 % a 1,9 %— porque, con un 3,4 % de
tinta por ventana, reconstruir sólo con el sesgo era más barato que reconstruir con el signo
cambiado. Un ReLU sin pre-activaciones positivas no recibe gradiente: no vuelve.

**El arreglo, y por qué éste**: el decodificador se **inicializa igual** al codificador
(`nn/modelo.py`). Es sólo inicialización: la arquitectura escrita arriba —decodificador libre de
norma 1— no cambia. Probado en el mismo diagnóstico: código vivo (24–35 %), R² 0,80–0,86 en 12
épocas con `λ = 0` y `λ = 0,1`. La alternativa de **atar** el decodificador al codificador
también lo resolvía, pero cambia el modelo; ésta no.

**Lo que NO cambia**: la regla de la sección A (candidatos, el menor, y qué pasa si no hay
ninguno), el `k`, las épocas, la `lr` y los `λ` del tanteo. Se vuelve a correr **una** vez con
el entrenador arreglado, y su resultado manda.
