# Kernels evaluados en `banco-k`

Generado por `python nn/evaluar_kernel.py --informe` leyendo los `resultados/*/criterios.json`. **No se transcribe nada a mano.**

⚠ Esto **no** es la calibración: aquéllas son cifras del **instrumento** y no se reportan como hallazgos (§11). [`CALIBRACION.md`](CALIBRACION.md) es la otra.

| kernel | `k` | IoU `eval` | IoU `train` | brecha | §2.1 | §2.2 | mecanismo | veredicto |
|---|---|---|---|---|---|---|---|---|
| **bork-k17-s3** | 17 | 0.8531 ± 0.0105 | 0.8870 ± 0.0089 | **+0.0339 ± 0.0048** | ✅ | +0.0077 / 0.0109 | facilitacion | **UTIL (sin efecto de generalizacion)** |
| **bork-k17-s2** | 17 | 0.8517 ± 0.0087 | 0.8821 ± 0.0084 | **+0.0304 ± 0.0049** | ✅ | ✅ | facilitacion | **UTIL Y GENERALIZA** |
| **bork-k17-s1** | 17 | 0.8515 ± 0.0095 | 0.8829 ± 0.0076 | **+0.0314 ± 0.0055** | ✅ | +0.0102 / 0.0116 | facilitacion | **UTIL (sin efecto de generalizacion)** |
| **bork-k15-s1** | 15 | 0.8499 ± 0.0114 | 0.8812 ± 0.0112 | **+0.0314 ± 0.0051** | ✅ | ✅ | facilitacion | **UTIL Y GENERALIZA** |
| **bork-k15-s2** | 15 | 0.8497 ± 0.0124 | 0.8817 ± 0.0133 | **+0.0321 ± 0.0050** | ✅ | +0.0095 / 0.0101 | facilitacion | **UTIL (sin efecto de generalizacion)** |
| **bork-k15-s3** | 15 | 0.8485 ± 0.0111 | 0.8795 ± 0.0112 | **+0.0310 ± 0.0056** | ✅ | +0.0106 / 0.0107 | facilitacion | **UTIL (sin efecto de generalizacion)** |
| **borpca-k15** | 15 | 0.8475 ± 0.0118 | 0.8802 ± 0.0123 | **+0.0327 ± 0.0074** | ✅ | +0.0089 / 0.0125 | facilitacion | **UTIL (sin efecto de generalizacion)** |
| **borpca-k17** | 17 | 0.8463 ± 0.0077 | 0.8798 ± 0.0116 | **+0.0335 ± 0.0057** | ✅ | +0.0080 / 0.0118 | facilitacion | **UTIL (sin efecto de generalizacion)** |
| **bork-k13-s3** | 13 | 0.8454 ± 0.0133 | 0.8781 ± 0.0113 | **+0.0327 ± 0.0048** | ✅ | +0.0093 / 0.0100 | facilitacion | **UTIL (sin efecto de generalizacion)** |
| **bork-k13-s2** | 13 | 0.8436 ± 0.0136 | 0.8769 ± 0.0134 | **+0.0332 ± 0.0046** | ✅ | +0.0088 / 0.0099 | facilitacion | **UTIL (sin efecto de generalizacion)** |
| **bork-k13-s1** | 13 | 0.8434 ± 0.0135 | 0.8763 ± 0.0132 | **+0.0329 ± 0.0051** | ✅ | +0.0091 / 0.0103 | facilitacion | **UTIL (sin efecto de generalizacion)** |
| **borpca-k13** | 13 | 0.8360 ± 0.0105 | 0.8755 ± 0.0093 | **+0.0395 ± 0.0059** | ✅ | +0.0025 / 0.0111 | facilitacion | **UTIL (sin efecto de generalizacion)** |
| **bork-k11-s3** | 11 | 0.8321 ± 0.0150 | 0.8652 ± 0.0093 | **+0.0331 ± 0.0085** | +0.0224 / 0.0275 | +0.0089 / 0.0142 | facilitacion | **NO DECLARA** |
| **bork-k11-s2** | 11 | 0.8315 ± 0.0149 | 0.8676 ± 0.0121 | **+0.0361 ± 0.0063** | +0.0217 / 0.0273 | +0.0059 / 0.0120 | facilitacion | **NO DECLARA** |
| **bork-k11-s1** | 11 | 0.8307 ± 0.0144 | 0.8645 ± 0.0094 | **+0.0338 ± 0.0084** | +0.0209 / 0.0268 | +0.0082 / 0.0141 | facilitacion | **NO DECLARA** |
| **borpca-k11** | 11 | 0.8287 ± 0.0058 | 0.8744 ± 0.0096 | **+0.0457 ± 0.0054** | ✅ | -0.0037 / 0.0110 | facilitacion | **UTIL (sin efecto de generalizacion)** |
| **esqk-k11** | 11 | 0.8256 ± 0.0087 | 0.8516 ± 0.0113 | **+0.0260 ± 0.0043** | +0.0162 / 0.0208 | (+0.0161 / 0.0103) *pasaría, pero es condicional* | transferencia | **NO DECLARA** |
| **borpca-k09** | 9 | 0.8222 ± 0.0065 | 0.8680 ± 0.0074 | **+0.0458 ± 0.0044** | +0.0140 / 0.0148 | -0.0024 / 0.0108 | facilitacion | **NO DECLARA** |
| **esqk-k09** | 9 | 0.8181 ± 0.0079 | 0.8443 ± 0.0066 | **+0.0261 ± 0.0035** | +0.0098 / 0.0156 | (+0.0159 / 0.0094) *pasaría, pero es condicional* | transferencia | **NO DECLARA** |
| **borae-k13-s3** | 13 | 0.8180 ± 0.0141 | 0.8436 ± 0.0151 | **+0.0255 ± 0.0052** | +0.0109 / 0.0255 | (+0.0165 / 0.0104) *pasaría, pero es condicional* | transferencia | **NO DECLARA** |
| **esqk-k07** | 7 | 0.8176 ± 0.0072 | 0.8457 ± 0.0084 | **+0.0280 ± 0.0049** | +0.0123 / 0.0149 | (+0.0141 / 0.0109) *pasaría, pero es condicional* | transferencia | **NO DECLARA** |
| **bork-k09-s2** | 9 | 0.8146 ± 0.0066 | 0.8557 ± 0.0075 | **+0.0410 ± 0.0050** | +0.0064 / 0.0148 | +0.0023 / 0.0114 | facilitacion | **NO DECLARA** |
| **borae-k11-s3** | 11 | 0.8130 ± 0.0082 | 0.8382 ± 0.0096 | **+0.0251 ± 0.0075** | +0.0033 / 0.0206 | (+0.0169 / 0.0131) *pasaría, pero es condicional* | transferencia | **NO DECLARA** |
| **bork-k09-s1** | 9 | 0.8129 ± 0.0099 | 0.8569 ± 0.0064 | **+0.0441 ± 0.0074** | +0.0046 / 0.0181 | -0.0007 / 0.0138 | facilitacion | **NO DECLARA** |
| **bork-k09-s3** | 9 | 0.8124 ± 0.0093 | 0.8568 ± 0.0075 | **+0.0444 ± 0.0054** | +0.0041 / 0.0175 | -0.0011 / 0.0118 | facilitacion | **NO DECLARA** |
| **borae-k09-s3** | 9 | 0.8111 ± 0.0056 | 0.8379 ± 0.0089 | **+0.0268 ± 0.0043** | +0.0029 / 0.0138 | (+0.0166 / 0.0107) *pasaría, pero es condicional* | transferencia | **NO DECLARA** |
| **borae-k11-s2** | 11 | 0.8100 ± 0.0091 | 0.8365 ± 0.0088 | **+0.0265 ± 0.0056** | +0.0002 / 0.0216 | (+0.0155 / 0.0113) *pasaría, pero es condicional* | transferencia | **NO DECLARA** |
| **borae-k09-s1** | 9 | 0.8072 ± 0.0046 | 0.8366 ± 0.0070 | **+0.0294 ± 0.0078** | -0.0010 / 0.0128 | +0.0140 / 0.0142 | transferencia | **NO DECLARA** |
| **borpca-k07** | 7 | 0.8071 ± 0.0075 | 0.8559 ± 0.0104 | **+0.0488 ± 0.0061** | +0.0022 / 0.0160 | -0.0075 / 0.0124 | facilitacion | **NO DECLARA** |
| **bork-k07-s3** | 7 | 0.8063 ± 0.0067 | 0.8528 ± 0.0066 | **+0.0465 ± 0.0091** | +0.0013 / 0.0152 | -0.0052 / 0.0154 | facilitacion | **NO DECLARA** |
| **bork-k07-s2** | 7 | 0.8059 ± 0.0063 | 0.8525 ± 0.0079 | **+0.0466 ± 0.0091** | +0.0010 / 0.0148 | -0.0052 / 0.0154 | facilitacion | **NO DECLARA** |
| **borae-k05-s1** | 5 | 0.8057 ± 0.0047 | 0.8406 ± 0.0069 | **+0.0349 ± 0.0070** | +0.0109 / 0.0121 | +0.0071 / 0.0122 | transferencia | **NO DECLARA** |
| **bork-k07-s1** | 7 | 0.8041 ± 0.0037 | 0.8507 ± 0.0063 | **+0.0465 ± 0.0051** | -0.0008 / 0.0122 | -0.0052 / 0.0114 | facilitacion | **NO DECLARA** |
| **borae-k05-s3** | 5 | 0.8034 ± 0.0035 | 0.8357 ± 0.0045 | **+0.0323 ± 0.0042** | +0.0086 / 0.0108 | (+0.0097 / 0.0094) *pasaría, pero es condicional* | transferencia | **NO DECLARA** |
| **borae-k07-s1** | 7 | 0.8030 ± 0.0107 | 0.8346 ± 0.0092 | **+0.0315 ± 0.0068** | -0.0019 / 0.0192 | +0.0098 / 0.0131 | transferencia | **NO DECLARA** |
| **borae-k09-s2** | 9 | 0.8025 ± 0.0069 | 0.8326 ± 0.0094 | **+0.0301 ± 0.0099** | -0.0058 / 0.0152 | +0.0132 / 0.0163 | transferencia | **NO DECLARA** |
| **borae-k07-s2** | 7 | 0.8011 ± 0.0084 | 0.8345 ± 0.0072 | **+0.0335 ± 0.0080** | -0.0039 / 0.0169 | +0.0079 / 0.0143 | transferencia | **NO DECLARA** |
| **borpca-k03** | 3 | 0.7972 ± 0.0100 | 0.8401 ± 0.0090 | **+0.0429 ± 0.0074** | +0.0042 / 0.0170 | +0.0004 / 0.0138 | sin efecto o perjudica | **NO DECLARA** |
| **laplaciano** | 9 | 0.7963 ± 0.0051 | 0.8459 ± 0.0051 | **+0.0496 ± 0.0080** | -0.0120 / 0.0128 | -0.0075 / 0.0140 | sin efecto o perjudica | **NO DECLARA** |
| **borae-k07-s3** | 7 | 0.7952 ± 0.0098 | 0.8348 ± 0.0106 | **+0.0396 ± 0.0062** | -0.0098 / 0.0183 | +0.0017 / 0.0125 | sin efecto o perjudica | **NO DECLARA** |
| **bork-k05-s1** | 5 | 0.7937 ± 0.0095 | 0.8404 ± 0.0122 | **+0.0467 ± 0.0082** | -0.0011 / 0.0168 | -0.0047 / 0.0134 | sin efecto o perjudica | **NO DECLARA** |
| **bork-k05-s3** | 5 | 0.7935 ± 0.0097 | 0.8387 ± 0.0119 | **+0.0453 ± 0.0062** | -0.0014 / 0.0171 | -0.0033 / 0.0114 | sin efecto o perjudica | **NO DECLARA** |
| **bork-k05-s2** | 5 | 0.7931 ± 0.0096 | 0.8400 ± 0.0117 | **+0.0468 ± 0.0072** | -0.0017 / 0.0169 | -0.0048 / 0.0124 | sin efecto o perjudica | **NO DECLARA** |
| **borpca-k05** | 5 | 0.7931 ± 0.0074 | 0.8438 ± 0.0087 | **+0.0507 ± 0.0050** | -0.0017 / 0.0147 | -0.0087 / 0.0102 | sin efecto o perjudica | **NO DECLARA** |
| **borae-k11-s1** | 11 | 0.7896 ± 0.0082 | 0.8306 ± 0.0106 | **+0.0409 ± 0.0107** | -0.0201 / 0.0207 | +0.0011 / 0.0164 | sin efecto o perjudica | **NO DECLARA** |
| **bork-k03-s2** | 3 | 0.7894 ± 0.0073 | 0.8478 ± 0.0081 | **+0.0584 ± 0.0085** | -0.0036 / 0.0143 | -0.0150 / 0.0148 | sin efecto o perjudica | **NO DECLARA** |
| **bork-k03-s1** | 3 | 0.7884 ± 0.0067 | 0.8458 ± 0.0087 | **+0.0574 ± 0.0079** | -0.0045 / 0.0137 | -0.0140 / 0.0143 | sin efecto o perjudica | **NO DECLARA** |
| **bork-k03-s3** | 3 | 0.7882 ± 0.0069 | 0.8459 ± 0.0077 | **+0.0577 ± 0.0085** | -0.0047 / 0.0140 | -0.0143 / 0.0149 | sin efecto o perjudica | **NO DECLARA** |
| **borae-k15-s1** | 15 | 0.7588 ± 0.0128 | 0.8050 ± 0.0117 | **+0.0462 ± 0.0066** | -0.0506 / 0.0244 | -0.0047 / 0.0117 | sin efecto o perjudica | **NO DECLARA** |

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
