# Kernels evaluados en `banco-k`

Generado por `python nn/evaluar_kernel.py --informe` leyendo los `resultados/*/criterios.json`. **No se transcribe nada a mano.**

⚠ Esto **no** es la calibración: aquéllas son cifras del **instrumento** y no se reportan como hallazgos (§11). [`CALIBRACION.md`](CALIBRACION.md) es la otra.

| kernel | `k` | IoU `eval` | IoU `train` | brecha | §2.1 | §2.2 | mecanismo | veredicto |
|---|---|---|---|---|---|---|---|---|
| **esqk-k11** | 11 | 0.8256 ± 0.0087 | 0.8516 ± 0.0113 | **+0.0260 ± 0.0043** | +0.0162 / 0.0208 | (+0.0161 / 0.0103) *pasaría, pero es condicional* | transferencia | **NO DECLARA** |
| **esqk-k09** | 9 | 0.8181 ± 0.0079 | 0.8443 ± 0.0066 | **+0.0261 ± 0.0035** | +0.0098 / 0.0156 | (+0.0159 / 0.0094) *pasaría, pero es condicional* | transferencia | **NO DECLARA** |
| **esqk-k07** | 7 | 0.8176 ± 0.0072 | 0.8457 ± 0.0084 | **+0.0280 ± 0.0049** | +0.0123 / 0.0149 | (+0.0141 / 0.0109) *pasaría, pero es condicional* | transferencia | **NO DECLARA** |
| **laplaciano** | 9 | 0.7963 ± 0.0051 | 0.8459 ± 0.0051 | **+0.0496 ± 0.0080** | -0.0120 / 0.0128 | -0.0075 / 0.0140 | sin efecto o perjudica | **NO DECLARA** |

Contra los controles del §10 (mismo dataset, mismas 10 semillas):

| control | IoU `eval` | brecha |
|---|---|---|
| identidad | 0.7981 ± 0.0057 | +0.0421 ± 0.0059 |
| aleatorio | 0.8083 ± 0.0077 | +0.0431 ± 0.0117 |
| gauss | 0.8130 ± 0.0086 | +0.0298 ± 0.0054 |
| sobel | 0.8164 ± 0.0064 | +0.0367 ± 0.0044 |

⚠ **Cada kernel se compara contra el aleatorio de SU `k`** (§2.1: «igual norma y mismo `k`»), que puede no ser el de la tabla de arriba. El que se usó está en `comparado_contra` de cada `criterios.json`.

## ⚠⚠ Fuga de distribución (§3.7): estos resultados salen OPTIMISTAS

Estos kernels se aprendieron con **el mismo generador** que el banco, y sobre datos que **alcanzan la reserva** del §3.7. El §3.7 es explícito: *«hay fuga aunque las muestras sean distintas»*.

| kernel | viene de | por qué hay fuga |
|---|---|---|
| `esqk-k07` | `esq-k` | familia: sin fijar -> sortea TODAS, incluida LiberationMono · interlineado: (1.15, 1.6) contra la reserva (1.45, 1.6) -> SOLAPA |
| `esqk-k09` | `esq-k` | familia: sin fijar -> sortea TODAS, incluida LiberationMono · interlineado: (1.15, 1.6) contra la reserva (1.45, 1.6) -> SOLAPA |
| `esqk-k11` | `esq-k` | familia: sin fijar -> sortea TODAS, incluida LiberationMono · interlineado: (1.15, 1.6) contra la reserva (1.45, 1.6) -> SOLAPA |

**No invalida la medición y no se ocultó**: la reserva se declaró el 2026-09-08 y esos kernels son anteriores, así que nadie rompió ninguna regla. Pero la advertencia **tiene que viajar con el número**: un kernel que vio la reserva parte con ventaja sobre uno que no la vio, y comparar los dos como iguales sería exactamente el error que el §3.7 existe para evitar.
