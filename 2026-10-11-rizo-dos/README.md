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

## Reevaluación con el criterio del dueño (2026-10-11, tarde): SÓLO cuenta encontrar el rizo

> «No importa, para este experimento, diferenciar los 2 de los otros números. Lo importante es hallar el mayor
> porcentaje posible de esos rizos que hallamos en muchos 2. Pero también se van a hallar en otros números.»

Con eso, los umbrales 2, 11 y 12 del criterio (falsos y rizos de más) dejan de decidir; cada estrategia se lleva al
umbral que **más rizos etiquetados encuentra** (≤ 2,5 px), y se reporta cuántos candidatos pone por 2 para que se vea
el precio. `nn/evaluar.py` no cambia; esto es `resultados/7-solo-encontrar.{png,json}` (barrido hecho a mano, mismos
47 doses, mismas funciones de medida).

![7 · sólo encontrar](resultados/7-solo-encontrar.png)

| estrategia (umbral) | **encontrados** | centro px | radio px | abertura ° | candidatos arriba por 2 |
|---|---:|---:|---:|---:|---:|
| **P plantillas (S 0,8)** | **91,5 %** | 1,41 | 1,0 | 40 | 9,0 |
| R rayos (F 0,5) | 83,0 % | 0,0 | 0,75 | 30 | 6,5 |
| C:c-base (p 0,2) | 74,5 % | 1,41 | 1,21 | 17 | 1,0 |
| C:c-ancha (p 0,2) | 68,1 % | 1,21 | 1,05 | 16 | 1,0 |
| C:c-dos2 / c-dos (p 0,02) | 61,7 % | 1,0 | 1,0 | 31–33 | 0,9 |
| V votos (V_min 1) | 36,2 % | 1,0 | 2,0 | 22 | 1,0 |

**Con este criterio ganan las plantillas (91,5 %), y los rayos pasan el 85 % por poco no (83 %).** Pero el precio está
en la última columna: P y R lo consiguen diciendo que hay **6–9 rizos por dígito** en la mitad de arriba (uno de ellos
cae donde está el de verdad). La CNN sigue dando **un** candidato por 2 y acierta 3 de cada 4. Son dos cosas distintas:
P/R **localizan** (el de verdad está entre sus candidatos), la CNN **decide**. Por eso la sugerencia 2 de arriba —P o R
proponen, la CNN puntúa— sigue siendo la que junta las dos virtudes, y con este criterio es la primera que probaría.

Dos matices medidos: el **centro** de R es exacto (0 px: el píxel más encerrado por rayos es la misma medida con la que
se afinaron las etiquetas, así que esto es en parte circular y hay que decirlo); y la abertura de P/R (30–40°) es peor
que la de la CNN (16–17°).

## El COMPUESTO de los tres primeros (P + R + V), pedido por el dueño el 2026-10-11

Antes de componer, lo que P y R ponen **de más** (figura 8): cruces a lo largo de **todo el contorno** —son detectores
de concavidad, no de rizo— con 16,9 (P) y 10,4 (R) candidatos sobrantes por dígito.

![8 · extras](resultados/8-extras-plantillas-rayos.png)

`nn/compuesto.py`: **candidatos** = unión de los de P (S 0,8), R (F 0,5) y los lazos de V, fundidos a ≤ 2 px; en cada
uno se miden las tres evidencias en su píxel (p = mejor plantilla, r = encierro por rayos a 8 px, v = votos de curvatura
relativos); **puntuación = r**; el rizo es el de más puntuación de la mitad superior.

| compuesto | rizos encontrados | candidatos por 2 |
|---|---:|---:|
| todos los candidatos | 100 % | 10,1 |
| con acuerdo ≥ 2 de los 3 | 97,9 % | 8,6 |
| acuerdo = 3 | 23 % | 1,2 |
| **el mejor (top-1)** | **74,5 %** | 1 |
| los dos mejores (top-2) | 78,7 % | 2 |
| top-1 de P solo · R solo · V solo | 25,5 · 46,8 · 6,4 % | 1 |

![9 · compuesto](resultados/9-compuesto.png)

**Qué dice.** Componer **sí** sirve, y se ve en la última fila (⚠ esas tres cifras y la de acuerdo ≥ 2 estaban mal
copiadas de la corrida anterior en el commit `d32ef53`; las de aquí son las de `9-compuesto.json`, y lo cazó el
`verificador`): ninguno de los tres solo elige bien (25 / 47 / 6 %), y
juntos eligen el 74,5 % con un candidato por dígito — **igual que la mejor CNN** (`c-base`, p 0,2, 74,5 %) y sin
entrenar nada. Pero el reparto de méritos no es el esperado: P y R **proponen** (el rizo está entre sus candidatos el
100 % de las veces), el **encierro elige**, y P y V **no ayudan a elegir**: la puntuación empezó como p + r + v (40 %)
y, probadas nueve combinaciones sobre los 47 doses, el encierro solo sube al 70–74 % mientras sumarle p o v baja
(p vale 0,90 en el rizo y 0,86 en el resto del contorno; v ≈ 0 en los dos). ⚠ Dos cosas que hay que decir: la regla se
eligió **mirando** los 47 etiquetados, así que está ajustada a ellos; y el encierro es la misma medida con la que se
afinaron las etiquetas, lo que favorece su **centro** (no su elección entre regiones, que es lo que mide el top-1).

**Lo que falla en el 25 % restante** (figura 9, en naranja): el top-1 cae en **otra concavidad de la misma cabeza** (el
hueco entre la cabeza y la diagonal, 1–3 px por debajo del rizo) o en el codo inferior. Para separar esas dos hace falta
algo que P, R y V no miden: que el trazo **rodee** al punto de forma continua (la estrategia 1 de la lista de `CLAUDE.md`,
seguir el trazo), o la CNN como juez de los candidatos del compuesto —que es lo contrario de lo que se probó hoy, y
queda como siguiente vuelta—.
