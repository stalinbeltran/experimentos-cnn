# Criterio — escrito el 2026-10-06, ANTES de entrenar ningún detector de borde

No existe ningún detector de borde: nada de lo de abajo se decidió mirando uno. Lo que SÍ se miró antes de escribirlo, y
hay que decirlo:

1. la **sonda de grosor** (`nn/sonda_grosor.py` → `resultados/sonda-grosor.json`), que compara representaciones y no
   entrena nada;
2. los **detectores de líneas de `feat-ind32`** (ya entrenados, la referencia) pasados por esta misma evaluación
   (`nn/evaluar.py`) sobre la prueba gruesa y sobre los dígitos. Sus números son el listón de abajo, y **cambiaron §B
   antes de escribirla en limpio**: la primera versión medía el grosor por el *recall*, y las líneas **no pierden recall**
   con trazos gruesos (0,863 → 0,865); lo que ganan son **falsos positivos** (0,030 → 0,126). Medir el recall habría
   medido lo que no falla. §B mide el **F1**, que recoge las dos cosas.

Si al correr hace falta cambiar algo, va como **enmienda fechada** al final.

## A. Por detector, val sintético (2–4 px, como al entrenar) — los umbrales de `feat-ind32`

| veredicto | condición |
|---|---|
| **aprendió** | F1 ≥ 0,90 **y** posición ≤ 1 celda ≥ 0,90 |
| **a medias** | F1 ≥ 0,75, o F1 ≥ 0,90 con posición < 0,90 |
| **no aprendió** | lo demás |

**H1** (contorno) y **H1** (signo): **≥ 11 de los 13** detectores, al menos «a medias», **y el F1 medio de los 13 a no más
de 0,05 del de las líneas**. Referencia: las líneas, 13 de 13 (4 rectas y el lazo «aprendió»; arcos y esquinas «a
medias»).

## B. La prueba de GROSOR (`feat-bor-sinteticas-grueso-32px-r20261006`)

Por detector, **r = F1 con grosores no vistos (6–12 px) ÷ F1 con grosores vistos (2–4 px)**, contando en cada tramo los
positivos y los negativos de ese grosor; **R = la media de los detectores con ≥ 20 positivos en los dos tramos** (12: el
lazo nunca pasa de 6 px en el perfil grueso).

Referencia medida, las líneas: **R = 0,696** (F1 0,795 → 0,567). Por familias: arcos r 0,38–0,45 (lo que se hunde), rectas
0,89–1,01 (aguantan), esquinas 0,64–0,80. Y el porqué: el recall no baja (0,863 → 0,865), **los falsos positivos se
multiplican por cuatro** (0,030 → 0,126; +0,096).

**H2** (signo) y **H2** (contorno) — la hipótesis del dueño, en sintético: con el borde, un trazo grueso deja de confundir.
Se confirma si se cumplen **las tres**:
1. los falsos positivos suben con el grosor **como mucho la mitad que en las líneas** (≤ +0,048);
2. el recall **no baja más de 0,05** de 2–4 px a 6–12 px;
3. el F1 en 6–12 px **no es peor que el de las líneas** (≥ 0,567).

R se reporta al lado, pero ya no decide: es un cociente de F1, y el F1 de cada tramo depende de una prevalencia que cambia
entre tramos (del 9 % al 22 % de positivos en recta-V, del 11,5 % al 5,8 % en arco-E). Un borde peor en todo podría
«confirmar» con un cociente bueno, y por eso el punto 3 es absoluto.

## C. Los dígitos (los 5620; la firma, sobre los 1797 de `windep`)

**H3** (signo) — el síntoma: los arcos se encienden en un 1, que es una recta. Con las líneas, arco-E en el **76 %** de los
1 y arco-W en el **70 %**. Se confirma si con el borde con signo los dos quedan **≤ 30 %**.

**H4** (contorno) y **H4** (signo) — lo que refutó `feat-agr`: la consistencia corregida por azar (κ) de los grupos al
**engrosar 2 px** y al **adelgazar 1 px** (zonas 13 × 9, K = 30, la partición de menor inercia de 5 semillas). Se confirma
si **κ ≥ κ(líneas) + 0,20 en los dos**. Líneas, medido: 0,114 y 0,345 → umbrales **0,314 y 0,545**. Al lado, los píxeles
(0,525 y 0,816).

**H5** (contorno) y **H5** (signo) — que no se pierda lo que había: el compositor posicional de `feat-ind32` (180 de train,
1617 de val, 3 semillas) **≥ líneas − 0,03**. Líneas, medido: 0,949 → umbral **0,919**.

## Lo que espero hoy (antes de entrenar)

- **H1** sí en los dos, con arcos y esquinas otra vez los más flojos. Puede costar algo de F1 respecto de las líneas: el
  ruido del dataset (puntos sueltos encendidos, agujeros en el trazo) se convierte en **bordes espurios**.
- **H2 contorno, no.** La sonda dice que el contorno pierde la forma al engrosar (0,59 con 12 px, peor que las líneas, 0,78):
  un trazo grueso tiene **dos** bordes, y su contorno se parece a un lazo o a dos rectas.
- **H2 signo, al borde.** Cada lado conserva su forma (0,79), pero el detector ve los cuatro lados juntos y sólo vio trazos
  finos, donde el izquierdo y el derecho casi coinciden; si aprendió «los dos juntos», el grosor lo rompe igual. Espero R
  entre 0,75 y 0,85.
- **H3 signo, sí.** Los lados de un 1 grueso son rectas, cada una en su canal: no hay arco que ver.
- **H4**: el contorno, no (peor que las líneas); el borde con signo, mejora, pero dudo de que llegue a +0,20.
- **H5**: el contorno pierde 0,02–0,05; el borde con signo, 0,01–0,03.
- **Lo que más probablemente salga mal:** los bordes espurios del ruido sintético, que los dígitos de NIST (limpios) no
  tienen; y que el ancla (la línea media del trazo) queda a medio grosor de cada borde, o sea a más de una celda en los
  trazos muy gruesos.

Si sale al revés en cualquiera de estos puntos, eso es un resultado y se escribe tal cual.

## Enmienda del 2026-10-06, ANTES de entrenar — a raíz del revisor

Sigue sin existir ningún detector de borde. Cambian H1 y H2, y los dos se endurecen:

- **H1** añade que el F1 medio no quede a más de 0,05 del de las líneas. «≥ 11 de 13 a medias» no detectaría que los 13
  empeoraran a la vez.
- **H2** pasa de «R ≥ R(líneas) + 0,10» a las tres condiciones de arriba. El cociente premiaba ir peor en 2–4 px y dependía
  de la prevalencia de cada tramo; el código además pedía R ≥ 0,80, que el criterio no decía. Ahora hay una sola
  definición, en los dos sitios.

Y una limitación que no se arregla con una sola semilla: con `signo` la primera convolución tiene 4 canales, así que la
inicialización de **todas** las capas cambia respecto de las líneas. El efecto de la representación no se puede separar del
de la inicialización.
