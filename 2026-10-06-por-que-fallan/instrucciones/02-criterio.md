# Criterio — por iteraciones, cada una escrita ANTES de correrla

Las métricas son siempre las mismas cuatro (`nn/evaluar.py`): el **compositor** posicional (180/1617, 3 semillas; y la
curva con 36/180/1080 sobre el test de 717), la **firma** (arcos en los 1), la **κ** de los grupos al engrosar 2 px y al
adelgazar 1 px (zonas, K = 30), y la **prueba gruesa** sintética de `feat-bor` (F1, recall y falsos positivos en 2–4 y
6–12 px).

**La referencia** es el banco de líneas de `feat-ind32` sin tocar nada (`lineas-nada`), medido en `feat-bor`
(`resultados/referencia-lineas.json` de allí) y repetido aquí: compositor **0,949**; curva **0,792 / 0,942 / 0,974**
(README de feat-ind32); κ **0,114** al engrosar y **0,345** al adelgazar; arco-E y arco-W en el **76 %** y el **70 %** de los
1; prueba gruesa F1 **0,795 → 0,567** con FP **0,030 → 0,126**.

## Iteración 1 (2026-10-06, 01:25 UTC) — S3: normalizar el grosor del dígito, sin reentrenar nada

**La causa que ataca, medida** (`nn/medir_grosor.py` → `resultados/grosor-digitos.json`): grosor medio de trazo (tinta ÷
esqueleto) **5,8** en los dígitos (p5 4,0 · p95 10,3; los 1, **9,8**) contra **2,1** en las features con que se entrenaron
los detectores (p95 3,5). Los dos rangos casi no se solapan.

**La solución:** esqueleto (Zhang–Suen) y volver a engrosar a **3 px** —el centro del rango de entrenamiento, 2–4 px; se
fija ahora y no se elige mirando el acierto— a todo lo que entra al detector: los dígitos y, en la prueba gruesa, también
las features.

Lo único mirado antes de escribir esto: `resultados/normalizacion.png` (12 dígitos). Normalizar deja dígitos finos y
limpios; adelgazar 1 px y normalizar da casi lo mismo; **pero engrosar 2 px y normalizar rompe la forma**, porque la
dilatación **cierra los huecos** (6, 8, 9) y eso ya no lo recupera nada.

| | se confirma si | la referencia |
|---|---|---|
| **S3a** el síntoma | arco-E y arco-W en ≤ 30 % de los 1 | 76 % / 70 % |
| **S3b** robustez | κ al adelgazar ≥ 0,545 (+0,20) | 0,345 |
| **S3c** no perder | compositor ≥ 0,939 (−0,01); se dice que **gana** si ≥ 0,959 (+0,01) | 0,949 |
| **S3d** sintético | F1 en 6–12 px ≥ F1 en 2–4 px − 0,05 | 0,567 contra 0,795 |

κ al **engrosar** se reporta sin umbral: la transformación misma destruye los huecos.

**Lo que espero:** S3a sí, S3b sí, S3c **gana** (los detectores ven los dígitos en el régimen en que se entrenaron), S3d sí
salvo lazos y arcos pequeños, a los que el grosor ya les cerró el hueco.

### Resultado de la iteración 1 (01:29 UTC)

`lineas-norm3`: **S3a ✅** (arcos en los 1: 12 % / 5 %) · **S3b ✅** (κ adelgazar 0,649; engrosar 0,275) · **S3c ❌ pierde**
(compositor **0,876**, −0,074; curva 0,685 / 0,878 / 0,928) · **S3d ❌** (F1 0,572 → 0,394: normalizar estropea hasta los
trazos finos sintéticos, cuyo ruido se esqueletiza en puntos y espinas).

**Diagnóstico, medido después:** la logística sobre píxeles también pierde al normalizar (0,908 → 0,884), así que se
destruye algo de información; los detectores pierden el triple. Los pares nuevos: **4→1 ×24**, 2→1 ×18, 2→7 ×13, 8→9 ×11.
Los 4 de este dataset son un **triángulo relleno**, y el esqueleto de una zona rellena es un palo: adelgazar destruye las
zonas rellenas igual que engrosar cerraba los huecos.

## Iteración 2 (01:31 UTC) — S3+: las DOS vistas, la cruda y la normalizada, en el mismo compositor

**Por qué:** la normalización arregla el grosor y rompe las zonas rellenas; la vista cruda, al revés. En `feat-ind`
(corrida 5) dar al compositor dos versiones de los detectores (fino + grueso) fue lo mejor medido: aprende a cuál creer. Aquí
son **los mismos 13 detectores sobre dos entradas** (26 mapas). Sin reentrenar.

| | se confirma si |
|---|---|
| **I2a** | compositor ≥ **0,959** (gana) · entre 0,949 y 0,959, empata · por debajo, pierde |
| **I2b** | κ al adelgazar ≥ **0,50** (conserva parte de la robustez de S3; la cruda sola da 0,345) |

**Lo que espero:** compositor 0,955–0,965 y κ 0,45–0,50.

### Resultado de la iteración 2 (01:35 UTC)

`lineas-nada+norm3`: compositor **0,956** → **I2a empata** (+0,007, no llega a +0,01); pero la curva sube en los tres tamaños
(**0,820 / 0,955 / 0,986** contra 0,792 / 0,942 / 0,974). **I2b ✅** κ adelgazar **0,562** (engrosar 0,303). Las dos vistas
conservan la mayor parte de la robustez de S3 sin perder acierto.

## Iteración 3 (escrita antes de 01:46 UTC; commit `444fdc1`) — S2: entrenar las features con el grosor de los dígitos

**Por qué:** S3 quita el grosor del dígito y rompe las zonas rellenas; la otra salida es que el detector **vea gruesos al
entrenar**. Los 13 detectores, misma receta, sobre `feat-fallos-sinteticas-grueso-32px-r20261006` (2–12 px). En `feat-ind`
(a 8×8) esto arregló el 1↔8 y abrió un 4→1 (la recta vertical gruesa se encendía en el cuerpo relleno del 4).

| | se confirma si |
|---|---|
| **S2a** síntoma | arcos en ≤ 30 % de los 1 (crudo) |
| **S2b** robustez | κ al adelgazar ≥ 0,545 |
| **S2c** no perder | compositor (crudo) ≥ 0,939; **gana** si ≥ 0,959 |
| **S2d** sintético | F1 en 6–12 px ≥ F1 en 2–4 px − 0,05 (lo vio al entrenar) |
| **S2e** fino + grueso | `lineas` y `lineas-grueso` crudos en el mismo compositor (26 mapas) ≥ 0,959 |

**Lo que espero:** S2d sí (es lo que vio); S2a sí; S2b a medias; **S2c pierde o empata** (vuelve el 4→1); **S2e sí** —es la
corrida 5 de feat-ind (0,972 a 8×8) a 32×32—.

*Enmienda antes de lanzar (02:10 UTC), a raíz del revisor:* S2e se mide con `nn/evaluar.py lineas+lineas-grueso nada` (dos
**bancos** juntos), que el código no hacía; y la máquina pasa a ≥ 26 vCPU para 13 procesos (en feat-bor, 26 procesos en 28
vCPU fueron 4 veces más lentos por núcleo que el dev). Ningún umbral cambia.

## Iteración 4 (02:24 UTC, ANTES de ver el diagnóstico) — más allá del grosor: inclinación y posición

S2 sigue entrenando. Mientras, la pregunta es qué **más** falla, aparte del grosor. `nn/diagnostico.py` (D1–D3: confusiones,
el mismo compositor sobre los **píxeles**, y el error por tercil de inclinación, descentrado, alto y ancho) es descriptivo y
no lleva umbral. Las dos soluciones que siguen se escriben **antes** de verlo, porque atacan las dos causas clásicas que el
grosor no explica:

**S4 — desinclinar** (`desinc`): cizallar cada dígito para que su eje quede vertical (α = cov(fila, col) / var(fila) de su
tinta; la cizalla pasa por su centro de masas y usa el vecino más cercano, así que la imagen sigue siendo 0/1). Un 1 inclinado
es una `recta-S`/`recta-B` donde el compositor esperaba una `recta-V`, y cada escritor inclina distinto.

| | se confirma si |
|---|---|
| **S4a** | `lineas` con `desinc` ≥ `lineas-nada` + 0,005 (≥ 0,954) |
| **S4b** | el error del tercil MÁS inclinado baja ≥ 25 % (relativo) respecto de `lineas-nada` |
| **S4c** | sumado a las dos vistas (`desinc+desinc_norm3`) ≥ `lineas-nada+norm3` + 0,005 (≥ 0,961) |

**S5 — compositor tolerante a la posición**: los mapas 8×8 pasan por un máximo 3×3 (paso 1) antes del lineal, así que una
feature corrida una celda sigue cayendo en el mismo peso. Sobre la combinación de referencia y sobre la mejor del momento.

| | se confirma si |
|---|---|
| **S5a** | la curva con **36** de train sube ≥ 0,02 (es donde pesa la posición: pocos ejemplos por celda) |
| **S5b** | el compositor de 180 no baja (≥ el mismo sin tolerancia − 0,002) |

**Lo que espero:** S4a sí, con el grueso de la ganancia en 1, 7 y 4↔9; S4b sí; S4c a medias. S5a sí y S5b empata; con 1080
de train, S5 perdería algo (la posición exacta es información cuando sobran ejemplos), y eso se mira pero no decide.
