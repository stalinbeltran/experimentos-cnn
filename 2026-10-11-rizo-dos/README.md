# `rizo-2` — el RIZO SUPERIOR del 2: cuatro estrategias contra 48 doses etiquetados (2026-10-11)

**Pregunta** (`experimento.json`): ¿cuál de cuatro estrategias detecta mejor el rizo superior de los 2 —centro, radio,
cierre y abertura— en los dígitos de UCI de 32 px: **R** rayos, **P** plantillas, **V** votos de curvatura (la referencia
de `lazos`), o **C** una CNN entrenada sólo con sintéticos? Encargo en `instrucciones/01-encargo.md`; criterio, escrito y
commiteado **antes** de medir (`23d72f4`), en `instrucciones/02-criterio.md`. 0 $ gastados (todo local, 2 vCPU; ~2 h de
cómputo en tres vueltas). ⚠ Todas las figuras de abajo son del **estado final** (tercera vuelta).

## Resultado en una frase

**Ninguna estrategia pasa el criterio (≥ 85 % de los doses reales con pocos falsos).** Lo mejor, barriendo el umbral de
cada una (figura 6): la CNN genérica **`c-base` encuentra el 57 % con un 32 % de falsos** en 1/4/7, o el **74 % con un
70 %**; los rayos, el 51 % con un 37 %; las plantillas encuentran el 91 % pero **disparan en el 100 % de los dígitos**
(no sirven solas). En sintéticos la CNN llega al 92–94 %: **lo que falla es el paso de sintético a dígito real**, y
el porqué está medido abajo.

![6 · lo que encuentra contra lo que inventa](resultados/6-curvas.png)

## Lo que se midió, vuelta a vuelta (todo contra la misma verdad)

| vuelta | qué cambió | doses encontrados (47, ≤ 2,5 px): R · P · V · C |
|---|---|---|
| 1 | sintéticos genéricos (rizo + cola corta), etiquetas a ojo | 13 · 45 · 19 · **34** (`c-base`) |
| 2 | + `c-dos`, CNN entrenada con un **compositor de doses** (cabeza + diagonal + base) | 13 · 45 · 19 · 38 (`c-dos`) |
| 3 | **etiquetas afinadas** (el centro a ojo caía 1–2 px sobre la tinta) · huecos de **rendija** (≥ 0,5 px) en sintéticos, plantillas y rayos · `c-dos2` | **51 · 91 · 34 · 57** (`c-base`) |

Los 16 umbrales del criterio, por estrategia, en `resultados/metricas.json` (`criterio`). ⚠ El conteo de «umbrales
pasados» (`c-dos` 12/16) **engaña**: pasa los de falsos (no ve casi nada) y falla el que importa, el 7 (encontrados).
Por eso el eje honesto es la figura 6, y por eso **no se declara ganadora** (el criterio lo exigía: nadie pasa el 7).

### Las tres cosas que se aprendieron, y que valen para cualquier detector de esta feature

1. **El hueco del rizo de un 2 real es una RENDIJA, no un círculo.** Medido en las 47 etiquetas: radio inscrito del
   hueco **mediana 1 px, 90 % ≤ 2,2 px** (`hueco_px` en `etiquetas-2.json`), con trazos de 4–6 px alrededor. Los
   sintéticos de la vuelta 1 exigían hueco ≥ 1,25 px y radio uniforme 2–7: la red aprendió un rizo redondo con aire
   dentro, y no lo encuentra en un 2. Bajar el hueco a 0,5 px (vuelta 3) subió `c-base` del 34 % al 57 % **sin
   reentrenar** sólo por las etiquetas, y los rayos del 13 % al 51 %.
2. **Las etiquetas a ojo sobre una rejilla se van 1–2 px.** El píxel anotado caía **sobre tinta** en la mitad de los
   casos (vuelta 2). Se afinaron al píxel de fondo más **encerrado** a ≤ 3 px del anotado (fracción de 36 rayos que
   chocan antes de 8 px; `centro_manual` conserva lo leído). La primera afinación, por distancia a la tinta, se iba al
   fondo abierto: se vio en la figura 0 y se cambió. ⚠ **Siguen siendo etiquetas de Claude**: la figura 0 está para
   que el dueño las corrija.
3. **El compositor de doses NO cerró el hueco sintético→real** (`c-dos2` 43 % contra `c-base` 57 %): parecerse a un 2
   no basta si la cabeza sintética sigue siendo un arco de círculo limpio. Las figuras 5 y 2 enseñan dónde se va: la
   CNN ve el **codo inferior** del 2 (entre la diagonal y la base) como rizo, y no ve una cabeza cuyo hueco es una
   rendija de 1 px.

## Las figuras

- **0 · las 48 etiquetas** (para corregir): `resultados/0-etiquetas.png`
- **1 · sintéticos de prueba y lo que ve cada estrategia**: `resultados/1-sinteticos.png` · muestras de los dos
  generadores: `1-sinteticos-muestra.png`, `1b-sinteticos-dos-muestra.png`
- **2 · los 48 doses, las siete columnas** (R · P · V · c-ancha · c-base · c-dos · c-dos2; ✓ = encontrado):
  `resultados/2-doses-etiquetados.png`
- **3 · % con rizo en la mitad superior por clase**: `resultados/3-por-clase.png` — P 100 % en todas; V 84–94 % en
  0/5/7/8/9; `c-base` 67 % en el 2, 61 % en el 3, 76 % en el 9, 6 % en el 1
- **4 · desplazamientos**: `resultados/4-desplazamientos.png` — **planas** para todas en horizontal (recorte 0 %), como
  se predijo; en vertical los dígitos tocan arriba y abajo, así que el «recortado» se mide sobre el rizo (tinta
  perdida a ≤ R + 3 px del centro): sin recorte, el % no cambia; con d = −4 queda el 13 % de la muestra
- **5 · los 20 doses que `c-base` no encuentra**: `resultados/5-fallos.png`
- **6 · barrido del umbral de cada estrategia**: `resultados/6-curvas.png` (`6-curvas.json`)

## Predicción contra lo medido

Se predijo que **C ganaría** pasando A y B con ≈ 90 % en los doses y fallando en el 7; que P pasaría por poco; que R y V
fallarían. **Acerté en A** (C 92–94 % en sintéticos, P/R/V por debajo), en que R y V fallan, y en que el 7 se parece al
rizo (47–58 % de los 7 con `c-base`). **Fallé en lo central**: C no pasa de sintéticos a dígitos (57 %), y P no «pasa
por poco», dispara en todo (S 0,8 → 100 % de falsos; S 0,95 → 21 % encontrados).

## Qué probar en la siguiente vuelta (sugerencias, ninguna probada)

1. **Entrenar la CNN con dígitos reales**, que es lo que el hueco sintético→real pide: verdad = el píxel más
   encerrado del rizo (lo que afinó las etiquetas, calculable **para los 159 doses de val y los ~1.200 de extra sin
   etiquetar a mano**) más las 47 etiquetas como control; negativos = las demás clases, con el 3 y el 9 como positivos
   «de arriba». Entra en 32 px en minutos en el dev. ⚠ Cambia la regla de este experimento (*«ninguna se entrena con
   dígitos»*): va como experimento aparte, con su criterio.
2. **Compositor de dos niveles: P propone y la CNN verifica.** P encuentra el 91 % (recall) pero no discrimina; la
   CNN discrimina y no encuentra. Puntuar cada candidato de P con el mapa de calor de C (o con el encierro por rayos).
3. **El encierro por rayos como feature directa**: *«fracción de rayos que chocan antes de 8 px»* es la medida que mejor
   localizó el hueco real (es la que afinó las etiquetas). Como detector dio 51 %/37 % porque exige anillo parejo
   (`disp`); sin esa exigencia y con un **mínimo de rendija** (hueco ≥ 1 px y ≤ 3 px) puede ser el mejor candidato
   barato. Y combinar con V, que es el único que ve la **curvatura** del trazo y no el hueco.
4. **Que el dueño corrija la figura 0**, o etiquete 20 más: con 47 doses, un punto son 2 %.

## Cómo repetirlo

    cd 2026-10-11-rizo-dos
    ../.venv/bin/python nn/rayos.py            # calibra R (~15 min)   → resultados/cal-rayos.json
    ../.venv/bin/python nn/plantillas.py       # calibra P (~10 s)     → resultados/cal-plantillas.json
    ../.venv/bin/python nn/entrenar_local.py c-base --epocas 12                     # ~36 min
    ../.venv/bin/python nn/entrenar_local.py c-dos2 --datos dos --epocas 8 --n 16000  # ~17 min
    ../.venv/bin/python nn/evaluar.py          # ~10 min → metricas.json y figuras 1–5
    ../.venv/bin/python nn/etiquetas.py        # figura 0

Los brazos `c-ancha` (genérico, cortado en la época 5 al ver la vuelta 1) y `c-dos` (compositor, huecos ≥ 1,25) se
conservan con sus pesos. `resultados/vuelta2/` guarda `metricas.json` y las calibraciones de la vuelta 2 (etiquetas
sin afinar, huecos ≥ 1,25); la vuelta 1 está en el commit `2a75a7e`.
