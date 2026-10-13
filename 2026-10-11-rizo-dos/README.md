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

## Plantillas MÁS RODEADAS (pedido por el dueño, 2026-10-11)

El banco actual, dibujado (`nn/plantillas_rodeadas.py` → figura `11-banco-actual.png`): 851 plantillas = radio
{2,5…7} × grosor {3,4,5} × cobertura {0,55 · 0,7 · 0,85 · 1} × abertura cada 30°; trazo +1, hueco −1, vano −½. La
figura 10 enseñó que las de **media vuelta** (0,55) casan con cualquier curva. Se probaron bancos que exigen más
vuelta y/o más hueco, con los mismos 47 doses (`11-bancos.json`, figura `11-bancos-rodeados.png`):

| banco (plantillas) | S | encontrados | candidatos por dígito | **top-1 arriba acierta** |
|---|---:|---:|---:|---:|
| actual, cob ≥ 0,55 (851) | 0,8 | 91,5 % | 18,2 | 29,8 % |
| cob ≥ 0,7 (575) | 0,8 | 85,1 % | 4,7 | 61,7 % |
| cob ≥ 0,7 (575) | 0,7 | 89,4 % | 10,1 | 61,7 % |
| **cob ≥ 0,85 (299)** | 0,8 | 76,6 % | **2,5** | **72,3 %** |
| cob ≥ 0,85 (299) | 0,6 | 87,2 % | 8,4 | 72,3 % |
| cob 0,6–0,8, «como los 2» (828) | 0,8 | 91,5 % | 12,5 | 36,2 % |

**Sí: quitar las de media vuelta arregla la mayor parte.** Con cob ≥ 0,85 el top-1 pasa del 30 % al **72 %** —a dos
puntos del compuesto (74,5 %) y de la CNN— con 2,5 candidatos por dígito en vez de 18. Exigir hueco ≥ 1 px no cambia
nada (las plantillas con hueco de medio píxel apenas pesan). Y la sorpresa: el banco «como los 2» (cob 0,6–0,8, que es
la cobertura que **tienen** los rizos etiquetados) es casi tan malo como el actual (36 %), porque una C de 0,6–0,7
sigue casando con cualquier curva desde su lado cóncavo. **Lo que discrimina no es que la plantilla se parezca al
rizo; es que la plantilla exija ser rodeada** —que es lo mismo que mide el encierro por rayos—. Un rizo real de
cobertura 0,65 casa con una plantilla de 0,85 porque la diagonal del 2 cierra el resto de la vuelta.

## El compuesto con las plantillas RODEADAS como proponente (`nn/compuesto2.py`, 2026-10-11)

Candidatos = plantillas rodeadas (cob ≥ 0,85, S 0,6) ∪ rayos (F 0,5); se elige por encierro, por la plantilla, o por
la suma (`12-compuesto2.json`):

| elige por | top-1 acierta | top-2 |
|---|---:|---:|
| **encierro r** | **78,7 %** | 89,4 % |
| plantilla rodeada p' | 70,2 % | 80,9 % |
| r + p' | 70,2 % | 83,0 % |
| (sólo candidatos de P', por p') | 59,6 % | 74,5 % |

**78,7 % con un candidato por dígito**: cuatro puntos más que el compuesto anterior (74,5 %) y que la CNN, y el top-2
llega al 89 %. Los candidatos son 8,2 por dígito (97,9 % de recall) y sigue eligiendo mejor el encierro que la propia
plantilla rodeada. Los 10 fallos, de cerca, en `12-compuesto2-fallos.png`; los 48, en `12-compuesto2.png`.

## TODOS los dígitos: el compuesto como detector de curvas cerradas + los otros detectores (`nn/digitos_cerradas.py`, 2026-10-11)

Cadena del boceto curvas-gabor (detector DELGADO: recto · vector de curvatura · golpe, 384 características en celdas
8×8 → compositor lineal, 180 train / 1617 val / 3823 a ciegas, 3 semillas) **más** dos mapas de curvas cerradas del
compuesto (campana de altura = encierro en cada candidato, y otra sólo con el mejor; 128 características). Dos corridas:

| características | val (1617) | a ciegas (3823) | fallos val s0 | arregla / rompe |
|---|---:|---:|---:|---:|
| otros (delgado), la referencia | **0,929** | **0,921** | 114 | — |
| cerradas solas (encierro ≥ 0,5, rayos 8 px) | 0,733 | 0,723 | | |
| otros + cerradas (≥ 0,5, 8 px) | 0,925 | 0,913 | 119 | 41 / 46 |
| cerradas solas (encierro ≥ 0,85, rayos 12 px) | 0,502 | 0,495 | | |
| otros + cerradas (≥ 0,85, 12 px) | 0,914 | 0,899 | 140 | 34 / 60 |

**No ayuda: añadir las curvas cerradas deja el compositor igual o peor.** Arregla tantos como rompe (41/46) y con los
candidatos más estrictos rompe más (34/60). Las cerradas solas clasifican el 73 % (y el 50 % si sólo se dejan las muy
encerradas). Figuras: `13-digitos-fallos.png` (los 119 fallos de la primera corrida, con la cruz azul del mejor
candidato), `13-digitos-cerradas.png` y `13b-*` (lo que el detector ve en tres dígitos de cada clase).

**Por qué, mirando las figuras:**
1. **El compositor lineal ya tiene esa información.** Sus vectores de curvatura → ← ↓ ↑ por celda dicen desde qué
   lado gira cada trozo; un lazo es «curvatura hacia dentro por los cuatro lados», y el compositor lo aprende sin
   necesitar que nadie le marque el centro. La feature nueva es redundante con las que había, y 128 números más sobre
   180 ejemplos de entrenamiento meten ruido (de ahí que rompa).
2. **El encierro premia estar PEGADO a la tinta, no estar en el centro del hueco** —lo que el dueño vio como «un punto
   arriba y afuera de lo que un humano consideraría una curva cerrada»—: en un 0 el mejor candidato cae en el borde del
   óvalo (donde todos los rayos chocan cerca), no en su centro (figura 13b, fila del 0). Con rayos a 12 px pasa igual.
   Para una feature de dígito haría falta el centro del hueco (distancia a la tinta) además del encierro.
3. **Los fallos del compositor no son de lazos** (figura 13): 8→9 (14), 1→3 (7), 8→3 (6), 9→3 (6), 2→1 (5)… son
   dígitos gruesos, unos sin hueco y treses con panza cerrada — justo donde un detector de cerradas tampoco distingue.

Lo que sí queda: el detector de curvas cerradas sirve como **feature para localizar** (el 78,7 % de los rizos del 2, y
los lazos del 6, 8, 9 y 0 en la figura 13b), no para **clasificar** dígitos por encima de lo que ya hacen las curvas.

## Más FONDO en el centro (`nn/cerradas_hueco.py`, pedido por el dueño el 2026-10-11)

Al candidato del compuesto se le añade **h = el radio inscrito del hueco** (la mayor distancia a la tinta a ≤ 2 px del
candidato; el candidato se MUEVE a ese píxel, el centro del hueco) y se exige h ≥ H. El encierro se mide ahí, con
rayos de 8 px (como hasta ahora) o de **14 px**. Medido en los 47 doses (top-1) y dibujado sobre los 119 dígitos que
falla el compositor (`14-hueco-*.png`, `14-hueco.json`):

| h ≥ | rayos | doses: top-1 | doses: encontrados | candidatos por dígito (en los 119) |
|---:|---:|---:|---:|---:|
| 0 (actual) | 8 | **80,9 %** | 89,4 % | 2,5 |
| 1,5 | 8 | 55,3 % | 59,6 % | 1,5 |
| 2,5 | 8 | 21,3 % | 21,3 % | 0,5 — y pierde los 0 |
| 0 | 14 | 78,7 % | 85,1 % | 3,1 |
| 1,5 | 14 | 53,2 % | 55,3 % | 2,2 |
| **2,5** | **14** | 21,3 % | 21,3 % | **0,7 — y sólo en lazos de verdad** |

⚠ Primera versión fallida, que hay que contar: h medido **en el píxel propuesto** (que casi siempre está pegado a la
tinta: h = 1) tiraba el 75 % de los rizos con h ≥ 1,5. El hueco se mide en su centro.

**Tres cosas que salen:**
1. **Mover el candidato al centro del hueco ya sube el top-1 de los doses de 76,6 a 80,9 %** sin exigir nada (fila 1).
2. **Pedir fondo en el centro es un dial entre «rizo» y «lazo».** Con h ≥ 2,5 y rayos de 14 px el detector marca **sólo
   lo que un humano llamaría curva cerrada**: los dos huecos del 8, el lazo del 6, 9 y 0, la panza cerrada de algunos 3 y
   5; y **nada** en 1, 4, 7 ni en la mayoría de los 2 (figura `14-hueco-h2.5-a0-r14.png`). Pero los rizos del 2 son
   rendijas (radio inscrito mediana 1 px) y se pierden casi todos (21 %). Con h ≥ 1,5 se queda a medio camino (55 %).
3. **Los rayos largos hacen falta para los lazos grandes**: con 8 px, los 0 (centro a más de 8 px de la tinta) tienen
   encierro bajo y con h ≥ 2,5 **desaparecen**; con 14 px vuelven (figura `14-hueco-h2.5-a0-r14.png`, fila de los 0).

Qué se propone: **dos salidas del mismo detector**, no un umbral: «lazo» (h ≥ 2,5, rayos 14) y «rizo» (h ≥ 0, rayos 8,
candidato en el centro del hueco), que es lo que el 2 tiene y el 8 no.

## Cada salida como feature APARTE del compositor (`nn/digitos_variantes.py`, 2026-10-11)

Cinco variantes del detector sobre los mismos candidatos (hueco medido en su centro; encierro ahí con rayos de 8, 14 o
18 px): **rizo** (h ≥ 0, 8 px), **medio** (h ≥ 1,5, 14 px), **lazo** (h ≥ 2,5, 14 px), **amplia** (h ≥ 3,5, 18 px,
para los 0) y **amplia floja** (encierro ≥ 0,7). Lo que cada una ve, en un dígito de cada clase: `15-variantes-mapas.png`.
% de dígitos de val con algún candidato: la amplia marca el 97 % de los 0 y ≤ 34 % de todo lo demás; el lazo, el 99 %
de los 0, 70–75 % de 6/8/9 y 0 % de los 1; el rizo, 100 % de los 8, 99 % de 6/9, 70 % de los 2 y 13 % de los 1.

**Con la codificación de antes (dos mapas en celdas 8×8 = 128 características por variante): todas empeoran**, y
cuantas más, peor (`15-variantes.json`): otros 0,929 → + rizo 0,912 · + medio 0,923 · + lazo 0,889 · + amplia 0,904 ·
+ rizo + lazo + amplia 0,876 · + las cinco **0,868**. Con 180 dígitos de entrenamiento, 128 números más por variante son
ruido que el compositor lineal ajusta.

**Con una codificación COMPACTA** (sólo el mapa del mejor candidato en celdas 2×2 o 4×4, más el número de candidatos:
5 o 17 características por variante; `15b-compactas.json`, con los candidatos ya calculados):

| compositor | val (1617) | a ciegas (3823) |
|---|---:|---:|
| otros (delgado), referencia | 0,9287 | 0,9209 |
| otros + rizo (2×2) | **0,9349** | **0,9246** |
| otros + medio (4×4) | **0,9355** | 0,9227 |
| otros + las cinco (2×2) | 0,9347 | 0,9227 |
| otros + rizo + lazo + amplia (2×2) | 0,9322 | 0,9239 |
| otros + lazo (2×2) | 0,9293 | 0,9181 |
| otros + amplia (2×2) | 0,9287 | 0,9188 |

**Lo que sale:** codificadas de forma compacta, las curvas cerradas **sí suman, poco**: +0,6 puntos en val y +0,4 a
ciegas con el rizo solo (0,9349 / 0,9246), y el medio igual; las que más ayudan son las que ven **más** dígitos (rizo,
medio), no las más selectivas (lazo, amplia), que apenas mueven nada y a ciegas restan. Juntar las cinco no supera a la
mejor sola. ⚠ El margen entre semillas es ≈ 0,002, así que +0,006 es una señal pequeña, no una declaración. **El cuello
de botella es el compositor** (lineal, 180 ejemplos): una feature de 128 números lo rompe y una de 5 apenas le cabe; con
más entrenamiento o un compositor que no sea lineal, las mismas salidas podrían valer más. Los fallos de la mejor
combinación de 8×8: `15-variantes-fallos.png`.

## Compositores NO lineales (`nn/compositores.py`, pedido por el dueño el 2026-10-11)

Mismo protocolo (180 train / 1617 val / 3823 a ciegas, 3 semillas) y mismas características; cambia el compositor
(`16-compositores.json`, figura `16-compositores.png`). Val / a ciegas:

| compositor | otros (384) | + rizo compacto (5) | + las cinco (25) | + rizo 8×8 (128) |
|---|---:|---:|---:|---:|
| lineal (referencia) | 0,929 / 0,921 | 0,935 / 0,925 | 0,935 / 0,923 | 0,912 / 0,908 |
| MLP 64 | 0,937 / 0,924 | 0,941 / 0,930 | 0,940 / 0,927 | 0,913 / 0,908 |
| **MLP 256** | 0,943 / 0,932 | **0,947 / 0,935** | 0,942 / 0,934 | 0,917 / 0,919 |
| SVM RBF | 0,927 / 0,919 | 0,933 / 0,924 | 0,930 / 0,924 | 0,882 / 0,875 |
| **bosque aleatorio (500)** | 0,946 / 0,930 | **0,956 / 0,933** | 0,942 / 0,919 | 0,919 / 0,888 |
| gradient boosting | 0,885 / 0,864 | 0,890 / 0,874 | 0,890 / 0,871 | 0,902 / 0,886 |
| k vecinos (5) | 0,856 / 0,837 | 0,860 / 0,841 | 0,856 / 0,832 | 0,824 / 0,793 |

**Lo que sale:**
1. **Un compositor no lineal sube ~1,5 puntos sobre el lineal con las mismas características**: bosque 0,946 y MLP 256
   0,943 contra 0,929 (a ciegas 0,930 / 0,932 contra 0,921). Boosting y k vecinos, peor que el lineal con 180 ejemplos.
2. **El rizo compacto suma en TODOS los compositores que funcionan**: +1,0 punto con el bosque (0,946 → **0,956**, el
   mejor val de todo el experimento), +0,45 con el MLP 256 (→ **0,947**, y el mejor a ciegas, **0,935**), +0,6 con el
   lineal y el SVM. Cinco características bastan; las cinco variantes juntas (25) y el mapa 8×8 (128) siguen restando,
   también con los no lineales: el cuello de botella no era sólo la linealidad, sino los 180 ejemplos.
3. **La ganancia a ciegas es menor que en val** (bosque: +1,0 en val, +0,3 a ciegas), así que parte es ajuste a val.
   ⚠ Rangos entre semillas ≈ 0,005 en el bosque y el MLP: el +1,0 del bosque está fuera del rango; el +0,3 a ciegas no.

Lo que se lleva: **MLP 256 + otros + rizo (0,947 / 0,935)** es la combinación más honesta (mejor a ciegas, poca
varianza), y el bosque la que más sube en val. En ambos casos la salida «rizo» del detector de curvas cerradas, codificada
como mejor candidato en 2×2 más la cuenta, es una feature que vale: la primera de este experimento que mejora el
reconocimiento de dígitos.

## Los detectores YA PROBADOS, solos y combinados (`nn/detectores.py`, pedido por el dueño el 2026-10-11)

Cuatro familias, cada una el detector tal como quedó publicado en el boceto curvas-gabor, sin tocar su calibración:
**rectas** (4 Gabor par integrados, 512), **delgado** (curvas sobre el dígito tal cual, 384), **gris** (curvas sobre el
borde Gabor gris λ_b 10 · K 9, 384) y **rizo** (la salida compacta de este experimento, 5). Con tres compositores
(`17-detectores.json`, figura `17-detectores.png`). Val / a ciegas:

| características | lineal | MLP 256 | bosque 500 |
|---|---:|---:|---:|
| rectas | 0,955 / 0,947 | 0,961 / 0,958 | 0,956 / 0,943 |
| delgado | 0,929 / 0,921 | 0,943 / 0,932 | 0,946 / 0,930 |
| **gris** | 0,960 / 0,957 | 0,967 / 0,960 | 0,967 / 0,955 |
| rizo solo | 0,374 / 0,361 | 0,424 / 0,391 | 0,471 / 0,428 |
| rectas + rizo | 0,958 / 0,950 | 0,964 / 0,959 | 0,958 / 0,944 |
| delgado + rizo | 0,935 / 0,925 | 0,947 / 0,935 | 0,956 / 0,933 |
| **gris + rizo** | 0,965 / 0,958 | **0,971** / 0,963 | **0,970** / 0,960 |
| rectas + gris | 0,963 / 0,960 | 0,971 / 0,968 | 0,962 / 0,949 |
| gris + delgado | 0,967 / 0,961 | 0,970 / 0,965 | **0,975** / 0,962 |
| **rectas + gris + rizo** | 0,965 / 0,962 | **0,972 / 0,969** | 0,965 / 0,951 |
| rectas + gris + delgado | 0,964 / 0,958 | 0,971 / 0,970 | 0,964 / 0,951 |
| todo | 0,965 / 0,957 | 0,970 / **0,970** | 0,968 / 0,952 |

**Lo que sale:**
1. **El gris es el mejor detector solo** (0,960 lineal, 0,967 no lineal), como ya decía el boceto; el delgado, que es el
   que se usó en las secciones anteriores como «otros», es el **peor** de los tres viejos. Lo que allí parecía poco
   margen era en parte eso.
2. **El rizo contribuye en todas las combinaciones y con los tres compositores**: +0,3 a +1,0 en val al añadirlo a
   cualquier familia (gris: 0,960 → 0,965 lineal, 0,967 → 0,971 MLP, 0,967 → 0,970 bosque). Con 5 números. A ciegas
   suma menos (+0,1 a +0,5) pero no resta en ningún caso.
3. **Rectas y gris se complementan** (0,963 lineal, 0,971 MLP, por encima de cada una), y el delgado aporta sobre el
   gris con el bosque (0,975 val, el máximo de la tabla) pero poco a ciegas (0,962).
4. **Lo mejor a ciegas**: MLP 256 con **rectas + gris + rizo (0,969)** o con todo (0,970) —a ciegas, que es lo que no se
   ha mirado al elegir—. El bosque gana en val y pierde a ciegas en todas las filas: ajusta más a val.

Resumen: de los detectores previos contribuyen **gris** (el que más), **rectas** (complementa al gris) y, poco pero
siempre, el **rizo**; el **delgado** sólo aporta con el bosque y no a ciegas. La mejor combinación honesta hoy:
**MLP 256 · rectas + gris + rizo: 0,972 val / 0,969 a ciegas** (la referencia del boceto era 0,960 / 0,957).

## Resistencia a los DESPLAZAMIENTOS, y cómo dársela (`nn/resistencia.py`, pedido por el dueño el 2026-10-12)

Compositor MLP 256 · rectas + gris + rizo. Val movida d = −4…+4 px (sin recortados; en vertical los dígitos tocan
arriba y abajo, así que sólo d = ±1 es medible), 3 semillas (`18-resistencia.json`, figura `18-resistencia.png`):

| método | d = 0 | x ±1 | x ±2 | x ±3 | x ±4 | y ±1 | cambian a x ±4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A base (180 tal cual, celdas 8×8) | 0,972 | 0,963 / 0,962 | 0,894 / 0,918 | 0,618 / 0,804 | **0,301 / 0,561** | 0,951 / 0,921 | 70 % / 44 % |
| **B aumento** (copias movidas ±1–3 px, 180 × 13) | **0,985** | 0,985 / 0,985 | 0,982 / 0,977 | 0,977 / 0,968 | **0,957 / 0,939** | 0,979 / 0,952 | 4 % / 6 % |
| C celdas 4×4 + aumento | 0,984 | 0,985 / 0,984 | 0,984 / 0,980 | 0,975 / 0,973 | 0,953 / 0,956 | 0,958 / 0,968 | 4 % / 4 % |
| D conjunto en prueba (±1) + aumento | **0,986** | 0,985 / 0,985 | 0,983 / 0,981 | 0,975 / 0,969 | 0,959 / 0,941 | 0,979 / 0,976 | 4 % / 5 % |

**Lo que sale:**
1. **El compositor base NO resiste**: a 3 px pierde 15–35 puntos y a 4 px lee mal 1 de cada 2 dígitos (70 % cambian de
   lectura a −4). Los detectores son convolucionales y se mueven con el dígito; lo que no se mueve son las **celdas fijas**
   en las que se leen (el mismo fallo que el boceto encontró en los gemelos del 5).
2. **El aumento lo arregla casi entero y, de paso, sube el acierto sin mover: 0,972 → 0,985.** Con 13 copias movidas de
   cada uno de los 180 dígitos, a ±4 px queda en 0,94–0,96 y sólo cambian de lectura el 4–6 %. Es la mejora más grande
   de todo el experimento, y es gratis: mismo detector, mismas features, más ejemplos de los que ya había.
3. Celdas 4×4 y el conjunto en prueba **añaden poco sobre el aumento** (±0,3 puntos; D da el mejor d = 0, 0,986, y la
   mejor y ±1). Lo que da la resistencia es ver los dígitos movidos al entrenar, no leerlos más grueso.
4. ⚠ Sólo val: el aumento no se ha medido a ciegas (3823), y el +1,3 en d = 0 hay que confirmarlo ahí.

### Y pasándole lo CERRADO de la curva (métodos E/F, 2026-10-12)

Al rizo compacto se le añaden 4 números del mejor candidato: el **cierre** de la plantilla que casa (0 = media vuelta …
1 = vuelta entera), cos y sin de su **abertura**, y la media del cierre de todos los candidatos (`compositores.cerrado`).
Misma medida (`18-resistencia.json`):

| método | d = 0 | x ±2 | x ±4 | y ±1 | cambian a x ±4 |
|---|---:|---:|---:|---:|---:|
| A base | 0,972 | 0,894 / 0,918 | 0,301 / 0,561 | 0,951 / 0,921 | 70 % / 44 % |
| E base + cerrado | 0,972 | 0,899 / 0,919 | 0,298 / 0,564 | 0,944 / 0,929 | 70 % / 44 % |
| B aumento | **0,985** | 0,982 / 0,977 | 0,957 / 0,939 | 0,979 / 0,952 | 4 % / 6 % |
| F aumento + cerrado | 0,984 | 0,983 / 0,978 | 0,954 / 0,938 | 0,972 / 0,952 | 5 % / 6 % |

**No cambia nada**: ni el acierto sin mover (0,972 → 0,972; 0,985 → 0,984) ni la resistencia (diferencias de ±0,3
puntos, dentro del margen entre semillas). Dos lecturas posibles, y no se distinguen con esto: (a) lo cerrado ya está
implícito en el encierro (altura del rizo) y en los vectores de curvatura del gris; (b) el cierre que da la plantilla
es grueso (cuatro valores: 0,1 · 0,4 · 0,7 · 1) y mal medido en rendijas, así que no aporta señal nueva. Para saber
cuál, haría falta un cierre medido sobre el trazo (cuánto arco rodea de verdad al hueco), que ninguno de los detectores
de hoy calcula.

## RASTREO de un fallo: el idx 784, un 8 leído como 6 (`nn/rastrear.py`, pedido por el dueño el 2026-10-12)

`python nn/rastrear.py --idx 784` entrena el mejor compositor (MLP 256 · rectas + gris + rizo, con aumento; semilla 0)
y para ese dígito pinta: candidatos del rizo y logits; **ablación** por grupo (sustituir rectas / gris / rizo por la media
de train, de los 8 o de los 6); **atribución** gradiente × entrada del margen logit 6 − logit 8, celda a celda sobre el
dígito; la feature del rizo frente a las medias de clase; y los 6 vecinos más cercanos en train
(`20-rastrear-784.json`, figura `20-rastrear-784.png`). Lo medido:

| | |
|---|---|
| logits | 6: **4,01** · 8: 3,12 (margen +0,89; p(6) 0,69, p(8) 0,28). El resto por debajo de 0 |
| rizo | encuentra **los dos huecos** (arriba en (12,9), abajo en (16,25), encierro 1,0 en los dos) · 5 rizos frente a 4,2 de media en los 8 y 2,7 en los 6 |
| atribución por grupo | **rectas +3,95** (hacia el 6) · gris −2,44 (hacia el 8) · rizo −0,51 (hacia el 8) |
| ablación | rectas ← media train: lee **8** (margen −2,7) · gris ← media train: lee 6 con más margen (+3,0) · rizo ← cualquier media: sigue 6 |
| sólo un grupo (los otros dos a la media) | sólo rectas → **6** · sólo gris → **8** · sólo rizo → 9 |
| vecinos en train | 1.º un **6** (idx 66), luego 8, 8, 6, 8, 6 · la clase más cercana en media es el 6 (33,1) y después el 8 (35,6) |

**Por qué 6:** los dos detectores que miran la curva —el gris (vectores de curvatura) y el rizo— leen un 8, pero el grupo de
**rectas** (Gabor en 4 orientaciones × 2 escalas, leído en celdas 8×8) lo lee como 6 con más fuerza, y gana por 0,89. Lo
que más empuja son las rectas a **135° a escala ×2** (+1,52) y a 0° (+0,79), en las celdas del lado derecho del lazo
inferior y del remate superior: el lazo de arriba es un bloque casi macizo (13 px de ancho con un hueco de 4×4) y a escala
gruesa su contorno se lee como el trazo diagonal grueso con que un 6 arranca. Es un fallo de **composición**, no de
detección: las features que distinguen (hueco arriba, curvatura) están, y el compositor las pesa menos que la textura de
rectas. Falla en las 3 semillas, y su vecino más cercano en feature es un 6.

⚠ Rastreo de **un** dígito y una semilla; la atribución grad × entrada es local (el margen es de 0,89 y la suma de las
atribuciones, +1,0, no coincide exactamente con él porque la red no es lineal). Para generalizarlo habría que correrlo en
los 23 fallos de la figura 19 (los 8→9 sobre todo).

### ¿Tiene sentido que sea la recta 135°? (2026-10-12, figura `20b-gabor135-784.png`)

Sí, y es una ambigüedad real a esa escala, no un artefacto. Medido con el kernel: en este proyecto «135°» es la cresta
que sube hacia la derecha (`/`). El 784 es un 8 **inclinado** (lazo de arriba a la izquierda, lazo de abajo a la derecha),
así que su silueta exterior tiene dos tramos `/` largos: el flanco superior-izquierdo del lazo de arriba y el inferior-
derecho del de abajo. En las celdas 8×8 a escala ×2 esos dos tramos caen **exactamente donde la media de los 6 de train
tiene su 135°** (el gancho con que arranca el 6 arriba a la izquierda, 1,1–1,2, y el cierre abajo a la derecha), y el 784
da 1,0 ahí. Lo que la media de los 8 tiene y el 784 no es el 135° **en el centro** (el cruce de la cintura, 0,9–1,0 en los
8; 0,1–0,3 en el 784, cuya cintura es casi horizontal). A escala ×2 además el hueco de arriba (4×4 px) se difumina y el
lazo se ve como una barra `/` gruesa, como el trazo del 6. O sea que la feature hace lo que es: energía de orientación por
posición; lo que distingue un 8 inclinado y macizo de un 6 es el hueco, que las rectas no ven y el gris y el rizo sí.

## Por qué el compositor pesa MÁS las rectas que el gris y el rizo (`nn/peso_grupos.py`, pedido por el dueño el 2026-10-12)

Cinco medidas sobre el compositor MLP 256 · rectas + gris + rizo con aumento, 3 semillas (`21-peso-grupos.json`, figura
`21-peso-grupos.png`; la comparación con/sin aumento en `21b-con-sin-aumento.json`):

| medida | rectas (512) | gris (384) | rizo (5) |
|---|---:|---:|---:|
| ‖W‖² de la 1.ª capa, total | **7,39** | 5,51 | 0,35 |
| ‖W‖² **por feature** | 0,0144 | 0,0144 | **0,071** |
| sensibilidad media en val, Σ\|grad × entrada\| | **12,9** | 9,2 | 0,39 |
| acierto en val SIN ese grupo (a la media) | **0,952** (−3,3) | 0,969 (−1,6) | 0,984 (−0,1) |
| acierto en val SÓLO con ese grupo | 0,961 | 0,952 | 0,290 |
| 8 contra 6, logística con ese grupo solo | 0,988 | **0,991** | 0,881 |
| ruido/señal del aumento (var. entre copias / entre dígitos) | 0,71 | **1,34** | 0,16 |
| ceros en train | 13,6 % | **73,5 %** | 29,4 % |
| F de Fisher por feature (mediana) | **13,2** | 4,2 | 9,1 |
| F de Fisher, suma del grupo | **7.717** | 2.158 | 54 |

**Lo que sale, en orden de importancia:**
1. **Rectas y gris pesan IGUAL por feature (0,0144 las dos).** La red no prefiere una recta a una curva: lo que hay es que
   las rectas son 512 números y el gris 384, y con el mismo peso por número el grupo grande suma más (1,34× en ‖W‖² y en
   sensibilidad, que es casi exactamente 512/384 = 1,33). **La ventaja de las rectas es de cuenta, no de preferencia.**
2. **El gris es RALO: el 73,5 % de sus features son cero** (sólo hay curvatura donde hay un borde curvo medible; la media
   de un 8 ocupa pocas celdas), y por feature separa las clases **3 veces peor** que una recta (F 4,2 frente a 13,2). Su
   información está concentrada en pocas celdas decisivas; la de las rectas, repartida en muchas celdas que informan
   un poco cada una. Con `weight_decay` y 400 pasos, el MLP reparte el peso por utilidad por feature, así que un grupo
   de muchos votos débiles acumula más peso que uno de pocos votos fuertes **aunque el segundo sea mejor detector solo**
   (gris 0,960 contra rectas 0,955 con el lineal, tabla de `detectores.py`). En 8 contra 6 los dos separan igual de bien
   (0,991 / 0,988): el 784 no se pierde por falta de información en el gris, sino porque su voto tiene menos masa.
3. **El rizo pesa 5 veces más por feature (0,071) y es el más estable al mover (ruido/señal 0,16)**: la red se fía de
   cada uno de sus números más que de ninguna recta. Pero son **5 números** que resumen sólo el mejor candidato (2×2) y
   la cuenta, así que su voto total es el 3 % de la sensibilidad y no puede dar la vuelta a un margen de 0,89. Solo
   separa 8 de 6 al 0,881: la cuenta de rizos (5 en el 784, 4,2 de media en los 8, 2,7 en los 6) solapa.
4. ⚠ **El aumento NO es lo que inclina hacia las rectas, al contrario** (`21b`): sin aumento el peso por feature era
   0,0119 rectas / 0,0094 gris / 0,025 rizo, y la sensibilidad 11,6 / 6,6 / 0,19. El aumento **igualó** rectas y gris
   por feature y multiplicó por 2,8 el peso del rizo. La hipótesis «el gris se mueve más al desplazar (1,34) y por eso
   se le desconfía» se midió y es **falsa**: el desequilibrio ya está en los 180 sin mover, y viene de la cuenta y de la
   ralez.

**Qué se sigue de aquí** (sin probar): si se quiere que el compositor escuche más al gris y al rizo, lo barato no es
cambiar el compositor sino el **tamaño relativo** de los grupos: o reducir las rectas (una escala, o celdas 4×4: 128–256
números), o dar al rizo más números (un mapa 8×8 por candidato, como la variante de 128 de `digitos_variantes.py`, que
en la tabla 15 no mejoró al compositor **lineal**; con el MLP y con aumento no está medido). Lo otro es no
estandarizar por feature sino por grupo, para que cada grupo entre con la misma norma total.

### Las rectas del 784 dibujadas como rectas (`nn/rectas_dibujo.py`, pedido por el dueño el 2026-10-12)

`python nn/rectas_dibujo.py --idx 784` → `22-rectas-784.png`: cada celda 8×8 de la feature «rectas» se pinta como un
segmento en la posición del máximo dentro de la celda, con la orientación de su Gabor (la dirección de la cresta se lee
del kernel: 0° horizontal, 45° `\`, 90° vertical, 135° `/`), y con longitud y tono proporcionales al valor (el máximo del
dígito, 2,13, se pinta de 15 px, que es la longitud del integrador). Suma de celdas por orientación, escala ×1 / ×2: 0°
29,4 / 22,2 · 45° 31,5 / 23,7 · 90° 33,5 / 25,3 · 135° 29,8 / 19,7. Lo que se ve: a 135° ×2 una sola recta larga y
oscura cruza el lazo de arriba por su flanco superior-izquierdo (la que la media de los 6 tiene ahí) y otra el cierre
inferior-derecho; a 45° las rectas fuertes van por el cuerpo inclinado del dígito; a 0° las más largas están en el remate
inferior y el superior, y a 90° en los dos flancos exteriores.

## Que las rectas cuenten MENOS: tres opciones a probar (orden del dueño, 2026-10-12)

De `peso_grupos.py`: rectas y gris pesan igual por feature, pero son 512 contra 384, y una recta larga enciende de 4 a 8
celdas seguidas, así que **una recta se cuenta varias veces**. Las rectas que ve son válidas; lo que sobra es la cuenta.
El dueño decidió probar, en este orden:

3. **Celdas 4×4 en vez de 8×8 sólo para las rectas** (512 → 128). `nn/menos_rectas.py`. ✅ empezado el 2026-10-12.
4. **Un solo voto por recta**: supresión del no máximo a lo largo de la recta (en la perpendicular, picos; a lo largo,
   un máximo por segmento conexo), para que una recta larga valga una celda y no cuatro. Mismo tamaño, mucho más ralo.
   ⏳ pendiente.
5. **Rectas como OBJETOS, como los rizos**: extraer segmentos (posición, orientación, longitud, grosor) y resumirlos por
   orientación y escala (cuántos, longitud total, 2×2 del más largo). ⏳ pendiente; con más números que el rizo, que a
   5 apenas vota (3 %).

Criterio para las tres, escrito antes de mirar: que **val y ciega no bajen** respecto de la referencia (0,985 val con
aumento; ciega con aumento no medida aún: la mide la primera corrida) **y** que se arreglen fallos de la figura 19 sin
abrir otros tantos. Que el 784 pase a 8 no es el criterio: es un dígito.

### Opción 3, medida (`nn/menos_rectas.py`, 2026-10-12): el 784 pasa a 8, pero val BAJA y se abren más fallos de los que cierra

`23-menos-rectas.json`, figura `23-menos-rectas.png`. MLP 256 con aumento, 3 semillas, gris 8×8 y rizo iguales:

| rectas | n | val | a ciegas | cuota de peso rectas / gris / rizo | fallos fig. 19 arreglados | fallos nuevos | 784 |
|---|---:|---:|---:|---|---:|---:|---|
| 8×8 (referencia) | 512 | **0,9852** | 0,9785 | 0,56 / 0,42 / 0,03 | 0–2 | 0–4 | 6, 6, 6 |
| **4×4** | 128 | 0,9825 | **0,9789** | 0,41 / 0,56 / 0,03 | 4–6 | **10–11** | **8, 8, 8** |
| 4×4, una escala | 64 | 0,9810 | 0,9772 | 0,28 / 0,68 / 0,04 | 4–5 | 12–13 | 8, 8, 8 |

1. **Reequilibra como se predijo**: con 128 rectas la cuota de peso pasa de 0,56 a 0,41 y el gris de 0,42 a 0,56. Y el
   784 se lee como 8 en las 3 semillas.
2. **Pero no mejora**: val baja 2,7 milésimas (0,9852 → 0,9825, fuera del rango entre semillas de la referencia,
   0,9845–0,9858) y a ciegas queda igual (0,9785 → 0,9789, +0,4 milésimas, dentro del ruido). De los 23 fallos arregla 4–6
   y **abre 10–11 nuevos**: las rectas a 8×8 resolvían más dígitos de los que confundían.
3. Con una sola escala (64) es peor aún: val 0,9810, 12–13 fallos nuevos. La escala ×2 aporta.

**Lectura**: el criterio escrito antes (val y ciega no bajen, y arreglar sin abrir otros tantos) **no se cumple**. Lo
que el 784 necesitaba —que las rectas pesen menos— se lo quita a otros dígitos que las necesitaban finas. Reducir celdas
quita posición, no repeticiones: la opción 4 (un voto por recta) es la que ataca lo segundo sin tocar lo primero, y es
la siguiente. ⚠ Primera medida a ciegas del compositor con aumento: 0,9785 (en val 0,985; la tabla de `detectores.py`
daba 0,969 a ciegas sin aumento, así que el aumento también sube a ciegas, +1,0).

**Quiénes son** (`nn/menos_rectas_fallos.py`, figura `23b-menos-rectas-fallos.png`, semilla 0; `n` en la tabla es el
número de features de rectas que entran al compositor): la 4×4 **arregla 5** —idx 27 (7, leído 9), 420 (6, leído 5),
**784 (8, leído 6)**, 1662 (9, leído 5) y 1765 (3, leído 5)— y **estropea 11**, que son casi todos trazos finos e
inclinados donde lo que distingue es la posición fina de las rectas: cinco 9 (683 y 738 → 1; 1562 y 1572 → 7; 1658 → 8),
dos 2 → 1 (75, 461), 4 → 1 (247), dos 5 → 7 (633, 1575) y 1 → 8 (1723). En dos de los 9 (1562, 683) el rizo no ve el
lazo o lo ve flojo, así que lo único que separaba 9 de 7 o de 1 era la recta fina del lazo, que a 4×4 se funde con el
palo. Siguen mal en las dos variantes 18 de los 23.

La misma figura con la opción 3 (`--celdas 4`): `22-rectas-784-4x4.png`. A 4×4 cada orientación y escala queda en 16
segmentos (uno por celda) en vez de 64; la suma de celdas baja de 124 a 50 (×1) y de 91 a 36 (×2). La recta `/` larga del
135° ×2 sobre el flanco superior-izquierdo sigue ahí, pero ahora es **una** celda, no cuatro: el mismo voto que en la media
de los 6, sin la repetición que lo multiplicaba.

### Opción 4, medida (`nn/menos_rectas.py --opcion 4`, 2026-10-12): UN VOTO por recta reequilibra más, arregla más y estropea más

Supresión del no máximo con ventana alargada sobre el mapa integrado de cada orientación (3 px de ancho, L px de largo
en la dirección de la recta; a ×2 la ventana se escala con la imagen): una recta larga deja un pico cada L px. Celdas 8×8,
512 números, pero ralos: en el 784 quedan 14 celdas no nulas de 256 a ×1 (la referencia tiene 226), y dibujadas
(`22-rectas-784-voto15.png`) son un trazado de 16 segmentos que se lee como el dígito. `24-un-voto.json`, figuras
`24-un-voto.png` y `24b-un-voto-fallos.png`; MLP 256 con aumento, 3 semillas:

| rectas | val | a ciegas | cuota de peso rectas / gris / rizo | fallos fig. 19 arreglados | fallos nuevos | 784 |
|---|---:|---:|---|---:|---:|---|
| 8×8 (referencia) | **0,9852** | **0,9785** | 0,56 / 0,42 / 0,03 | 0–2 | 0–4 | 6, 6, 6 |
| un voto, L = 15 | 0,9831 | 0,9740 | 0,25 / 0,71 / 0,04 | **7–9** | 12–13 | 8, 8, 8 |
| un voto, L = 31 | 0,9821 | 0,9735 | 0,25 / 0,71 / 0,04 | 7–9 | 14–15 | 8, 8, 8 |

1. **Reequilibra de verdad**: la cuota de peso de las rectas cae de 0,56 a 0,25, y el gris pasa a mandar (0,71). Es el
   efecto buscado, mayor que con 4×4 (0,41).
2. **Arregla más que la 4×4** (7–9 de 23, frente a 4–6; semilla 0: 27, 449, 784, 1100, 1152, 1662, 1765: cuatro 9, un 8, un
   7 y dos 3), **pero estropea más** (12–13 nuevos; semilla 0: dos 8 → 9 y 8 → 1, tres 5 → 7/4, dos 9, dos 3, un 1).
   Val baja 2 milésimas y **a ciegas baja 4,5** (0,9785 → 0,9740): es la peor a ciegas de las tres variantes.
3. L = 31 (un voto por recta entera) es igual o peor que L = 15 en todo: el voto único no aporta sobre el voto cada 15 px.

**Por qué estropea, leído de los nuevos fallos**: un mapa ralo leído en **celdas fijas** es frágil por construcción. Un
pico cae en una celda o en la vecina según 1 px, y con 14 celdas no nulas cada una de ellas pesa mucho; el aumento lo
compensa en parte en val (los 12 nuevos) y menos a ciegas. Es el mismo fallo que la figura 18 midió en el compositor base
sin aumento, concentrado en las pocas celdas que sobreviven. Las rectas densas de la referencia son redundantes, y esa
redundancia es lo que las hace estables a 1 px: contar menos quita la repetición **y** la estabilidad a la vez.

**Lectura para la opción 5**: convertir los picos en OBJETOS (posición continua, orientación, longitud) y resumirlos sin
rejilla fija (cuántos por orientación, longitud total, posición del más largo) es justo lo que evita que un pico cambie
de celda; por eso el rizo compacto, con 5 números, es el grupo más estable al mover (ruido/señal 0,16 en `peso_grupos`).
El criterio sigue siendo el mismo: val y ciega no bajen.

### Cómo se obtiene la feature del rizo, etapa a etapa (`nn/rizo_dibujo.py`, pedido por el dueño el 2026-10-12)

`python nn/rizo_dibujo.py --idx 784` → `25-rizo-784.png`, ocho paneles: 1 dígito · 2 plantillas rodeadas (299 anillos con
cob ≥ 0,85, mejor correlación por píxel y sus picos, P') · 3 rayos (36 rayos de 8 px desde cada píxel de fondo, fracción que
choca; regiones ≥ 0,5, R) · 4 unión P' ∪ R ordenada por encierro y sin duplicados a 2 px (24 candidatos crudos) · 5 el
hueco: cada candidato salta al máximo de la distancia al trazo a ≤ 2 px y toma su altura h · 6 encierro en ese centro con
rayos de 8 px, filtro h ≥ 0 y encierro ≥ 0,85, sin duplicados (5 rizos) · 7 mapa de campanas σ 1,5 con altura = encierro
(todos, y sólo el mejor) · 8 lo que entra al compositor: máximo 2×2 del mapa del mejor + la cuenta = 5 números, aquí
`[1,0 · 0,03 · 0 · 0 · 5]`. ⚠ Lo que enseña el último panel: de los dos huecos del 8, al compositor sólo le llega la
**posición del mejor** (arriba a la izquierda); el de abajo sólo cuenta en el `5`, mezclado con los duplicados del mismo
hueco. Es lo que limita al rizo a 0,881 en 8 contra 6.

## Pasarle TODOS los agujeros al compositor, en 3×3 (`nn/rizo_huecos.py`, pedido por el dueño el 2026-10-12)

Al ver la figura 25 (*«logra identificar los agujeros del 8, pero sólo pasa 1»*): los rizos filtrados se **agrupan por
agujero** —dos candidatos cuyo centro cae en la misma componente conexa del fondo son el mismo agujero, se queda el de
mayor encierro; los del fondo exterior (concavidades abiertas) se agrupan por distancia ≤ 6 px—, un agujero → una campana
σ 1,5 de altura = encierro, el mapa de **todos** se lee en **3×3** y se añade la **cuenta de agujeros**: 10 números. Sin
agujeros, nueve ceros y un 0. Figura `26-huecos.png` (784: 5 rizos → 2 agujeros, cuenta 2 en vez de 5, y los dos huecos en
el 3×3; el 6: 1; el 3 y el 9 idx 1562: 0). ⚠ Ese 9 tiene lazo cerrado y el detector **no lo ve** (0 candidatos pasan):
eso es aguas arriba de esta feature y no lo arregla.

Medido con el compositor (`26-huecos.json`, `26-huecos-medida.png`; MLP 256, rectas + gris + …, con aumento, 3 semillas):

| grupo del rizo | n | val | a ciegas | cuota de peso del rizo | fallos fig. 19 arreglados | nuevos | 784 |
|---|---:|---:|---:|---:|---:|---:|---|
| actual: 2×2 del mejor + cuenta | 5 | 0,9852 | 0,9785 | 0,027 | 0–2 | 0–4 | 6 |
| huecos 3×3 + cuenta de agujeros | 10 | 0,9847 | 0,9787 | 0,030 | 1–2 | 3 | 6 |
| actual + huecos 3×3 | 15 | **0,9854** | **0,9792** | 0,048 | 2 | 2–3 | 6 |

**No cambia nada medible**: ±0,5 milésimas en val y +0,7 a ciegas con los dos juntos, dentro del margen entre semillas
(0,9845–0,9858). El compositor sí le da más peso (0,027 → 0,048) pero el 784 sigue en 6: con 10–15 números frente a 896
de rectas y gris, el grupo sigue siendo el 3–5 % del voto. La feature es más fiel —cuenta agujeros de verdad, y los dos
del 8 llegan— y eso vale por sí mismo; lo que no hace es cambiar el equilibrio del compositor, que es el problema de
`peso_grupos.py`. Mismo criterio: se queda si no baja, y no baja.

### Opción 5, medida (`nn/rectas_objetos.py`, 2026-10-12): las rectas como OBJETOS, con la receta de los agujeros

Los picos de «un voto» (L = 15) de la misma orientación, colineales (≤ 2 px en la perpendicular) y a ≤ 15 px a lo largo
se agrupan: una recta. Cada recta es un objeto (centro, orientación, longitud = extensión de sus picos + 15, fuerza = el
pico mayor), y por orientación se lee el mapa de todas (campana σ 1,5, altura = fuerza/2,5) en 3×3 + cuenta + longitud
total: 44 números por escala. Figura `27-objetos-784.png`: el 784 queda en **14 rectas** a ×1 (2 · 5 · 3 · 4 por
orientación) y 4 a ×2, dibujadas con su longitud y su fuerza, frente a las 226 celdas de la referencia. Medida con gris
8×8 y rizo = actual + huecos 3×3 (15), MLP 256 con aumento, 3 semillas (`27-objetos.json`, `27-objetos-medida.png`):

| rectas | n | val | a ciegas | cuota rectas / gris / rizo | fallos fig. 19 arreglados | nuevos | 784 |
|---|---:|---:|---:|---|---:|---:|---|
| 8×8 (referencia) | 512 | 0,9854 | **0,9792** | 0,55 / 0,41 / 0,05 | 2 | 2–3 | 6 |
| objetos ×1 | 44 | 0,9829 | 0,9689 | 0,17 / 0,77 / 0,06 | 6–7 | 10–12 | 8 |
| objetos ×1 y ×2 | 88 | 0,9843 | 0,9710 | 0,25 / 0,70 / 0,06 | **9** | 10–13 | 8 |
| **8×8 + objetos ×1 y ×2** | 600 | **0,9878** [0,9876–0,9882] | 0,9787 | 0,60 / 0,36 / 0,04 | 5–6 | **2** | 6 |

1. **Solas, las rectas-objeto cumplen lo que se buscaba y no el criterio**: con 88 números el grupo baja a 0,25 de la
   masa, el 784 se lee como 8 en las 3 semillas y arregla 9 de los 23; pero abre 10–13 nuevos y **a ciegas cae 8
   milésimas** (0,9792 → 0,9710), la peor caída de las tres opciones. Son 14 objetos por dígito: cada uno pesa mucho y
   uno que se parta o se funda cambia la lectura.
2. **Añadidas a las rectas 8×8 dan el mejor val hasta hoy: 0,9878**, 2,4 milésimas sobre la referencia y fuera de su
   rango entre semillas; arregla 5–6 y abre sólo 2. A ciegas, igual (0,9787). O sea que las rectas-objeto **aportan
   información que las celdas no tienen** (cuánto mide una recta y cuántas hay), pero no sustituyen a las celdas, y el
   784 sigue en 6 porque la cuota de rectas incluso sube (0,60).
3. **Las tres opciones dicen lo mismo**: todo lo que hace que las rectas cuenten menos arregla el 784 y baja a ciegas.
   La redundancia de las 512 celdas no es sólo cuenta: es estabilidad, y a ciegas se paga quitarla.

⚠ La longitud de un objeto de un solo pico es 15 px por construcción (L); es un suelo, no una medida. Y `NORMA` = 2,5
se fijó a mano (el 784 da 2,13): si alguna vez un pico pasa de ahí, satura.

### ✅ Decisión del dueño (2026-10-12): para las rectas, **objetos ×1** (44), porque baja el peso de las rectas

Con eso el **gris** pasa a dominar el compositor (cuota 0,77). Queda pendiente medir esa configuración a ciegas con
intención (hoy sale 0,9689; ver tabla de la opción 5) y qué hacer con la caída.

### Cómo funciona el detector GRIS, etapa a etapa (`nn/gris_dibujo.py`, pedido por el dueño el 2026-10-12)

`python nn/gris_dibujo.py --idx 784` → `28-gris-784.png`. Es la cadena del boceto curvas-gabor con la calibración «gris
re-calibrado» (τ 0,1 · σ_E 0,5 · coh 0,25 · λ 3):
1. el dígito; 2. **borde**: banco de Gabor **impar** (seno, λ_b 10, 9×9, 8 orientaciones), |respuesta| máxima normalizada
por un escalón ideal → borde **gris** en la frontera del trazo (2–3 px, no binario); 3. **campo**: Gabor **par** (coseno,
12 orientaciones, λ 3) sobre ese borde → energía E_max (dónde hay borde: umbral τ), orientación θ (media en ángulo doble)
y coherencia (cuánto se cree θ); 4. **giro**: en cada píxel del borde se lee θ a ±4 px a lo largo de la tangente, κ = Δθ/8
en °/px, medible sólo si los dos extremos siguen en el borde y son coherentes (en el 784, 222 de 493 px); 5. el **vector
de curvatura** k = κ·(−sin θ, cos θ), que apunta al centro de la curva; 6. **seis mapas**: recto (|κ| < 2 °/px) · k→ ·
k← · k↓ · k↑ (cada sentido, max(0, ·)) · golpe (coherencia < 0,25: esquina o cruce); 7. máximo en celdas 8×8 → 384.
Lo que se ve en el 784: el lazo de abajo es casi todo «k←» y «k↑» (el centro queda a la izquierda y arriba de su borde
derecho e inferior), el de arriba «k→» y «k↓»; «recto» sólo tiene 13 celdas (los dos flancos del cuello), y «golpe»
marca el cruce de la cintura y las esquinas de los huecos.

### Sólo el gris (2026-10-12, `29-gris-solo.json`)

MLP 256 con aumento, 3 semillas, sin rectas ni rizo: **val 0,9779** (0,9765–0,9790) · **a ciegas 0,9690**. Arregla 7–8
de los 23 fallos (el 784 entre ellos: 8 en las 3 semillas) y abre 19–22 nuevos. Sin aumento (`detectores.py`) era
0,967 / 0,960, así que el aumento le da +1,1 / +0,9. Contra la configuración elegida (objetos ×1 + gris + rizo, 0,9829 /
0,9689): a ciegas son **iguales**, o sea que en esa configuración las rectas-objeto y el rizo sólo aportan en val.
Contra la referencia (0,9852 / 0,9792): −0,7 / −1,0 puntos.

### ¿Quitar el canal «recto» del gris, ya que las rectas están en otro grupo? (2026-10-12, `30-gris-sin-recto.json`)

El «recto» del gris son 64 de sus 384 (|κ| < 2 °/px y medible). Medido quitándolo, MLP 256 con aumento, 3 semillas:

| configuración | n | val | a ciegas | fallos fig. 19 arreglados | nuevos | 784 |
|---|---:|---:|---:|---:|---:|---|
| gris solo | 384 | 0,9779 | 0,9690 | 7–8 | 19–22 | 8 |
| gris SIN recto, solo | 320 | 0,9722 | 0,9640 | 6 | 27–30 | 8 |
| objetos ×1 + gris + rizo (elegida) | 443 | 0,9829 | 0,9689 | 6–7 | 10–12 | 8 |
| objetos ×1 + gris SIN recto + rizo | 379 | 0,9823 | **0,9629** | 6–8 | 12–13 | 8 |
| rectas 8×8 + gris + rizo (referencia) | 911 | 0,9854 | 0,9792 | 2 | 2–3 | 6 |
| rectas 8×8 + gris SIN recto + rizo | 847 | 0,9856 | 0,9776 | 3 | 2–4 | 6 |

**No conviene quitarlo.** Solo, el gris pierde 0,6 / 0,5 puntos sin él. En la configuración elegida, val queda igual pero
**a ciegas baja 0,6** (0,9689 → 0,9629): las rectas-objeto no sustituyen al «recto» del gris. Y hasta con las 512 rectas
densas baja a ciegas (0,9792 → 0,9776). El motivo es que no son la misma cosa: las rectas Gabor miden **dónde hay un trazo
recto** (energía de orientación en el trazo); el «recto» del gris dice **dónde el borde no gira**, medido en la misma
cadena y con la misma máscara que las curvas, así que es el complemento de «k→ k← k↓ k↑»: para una celda, «hay borde y
no es curvo» es información que los otros cuatro canales no llevan (su cero puede ser «recto» o «no medible»).

### El gris (sin «recto») como OBJETOS, con la receta de los agujeros (`nn/gris_objetos.py`, 2026-10-12)

Cada píxel curvo del borde se asigna al **sentido dominante** de su vector k (→ ← ↓ ↑); las componentes conexas dentro de
cada sentido son **una curva** (objeto: centro, longitud en px, fuerza = |κ| máx / 6); por sentido, 3×3 de todas + cuenta +
longitud; el golpe igual, 3×3 + cuenta: **54 números** (contra 320 del gris sin recto en 8×8). Figura `31-gris-objetos-784.png`:
el 784 queda en 14 curvas (6 → · 3 ← · 3 ↓ · 2 ↑) y 7 golpes. Medida (`31-gris-objetos.json`, `31-gris-objetos-medida.png`;
MLP 256 con aumento, 3 semillas; rizo = actual + huecos):

| configuración | n | val | a ciegas | fallos fig. 19 arreglados | nuevos | 784 |
|---|---:|---:|---:|---:|---:|---|
| objetos ×1 + gris + rizo (elegida) | 443 | 0,9829 | 0,9689 | 6–7 | 10–12 | 8 |
| objetos ×1 + gris sin recto + rizo | 379 | 0,9823 | 0,9629 | 6–8 | 12–13 | 8 |
| objetos ×1 + **gris-objetos** + rizo (sustituye) | 113 | 0,9678 | 0,9529 | 10–11 | 39–40 | 8, 8, 6 |
| objetos ×1 + gris sin recto + gris-objetos + rizo (añade) | 433 | 0,9800 | **0,9711** | 8 | 16–18 | 8 |
| rectas 8×8 + gris-objetos + rizo | 581 | 0,9870 | 0,9774 | 7 | 4–6 | 6 |
| rectas 8×8 + gris + gris-objetos + rizo | 965 | 0,9862 | **0,9801** | 3–4 | 2–3 | 6 |
| gris-objetos solo | 54 | 0,9419 | 0,9326 | 7–8 | 78–79 | 6 |

1. **Como sustituto, no**: 54 números en vez de 320 pierden 1,5 / 1,6 puntos (0,9678 / 0,9529) y abren 39 fallos. Es el
   mismo patrón que con las rectas: los objetos resumen demasiado y cada uno pesa demasiado.
2. **Como añadido, sí, y a ciegas**: sobre el gris sin recto en la configuración elegida sube a ciegas 0,8 puntos
   (0,9629 → 0,9711, por encima incluso de la elegida con recto, 0,9689) aunque baje 0,2 en val; y sobre la referencia
   completa da **0,9801 a ciegas, lo mejor medido hasta hoy** (0,9792 antes), con val 0,9862. Cuántas curvas hay de cada
   sentido y cuánto miden es información que las celdas no llevan y que generaliza a otros escritores.
3. El 784 sigue en 8 en toda configuración sin rectas densas, y en 6 con ellas: no es cosa del gris.

### Cómo queda repartido el peso (2026-10-12, `32-reparto-peso.json`)

Cuota de ‖W‖² de la 1.ª capa y de sensibilidad (Σ|grad × entrada| en val) por grupo, 3 semillas:

| configuración | rectas | gris celdas | gris-objetos | rizo |
|---|---|---|---|---|
| elegida: objetos ×1 (44) + gris (384) + rizo (15) | 0,17 / 0,15 | **0,77 / 0,80** | — | 0,06 / 0,06 |
| objetos ×1 + gris sin recto (320) + gris-objetos (54) + rizo | 0,17 / 0,15 | 0,56 / 0,61 | **0,21 / 0,20** | 0,06 / 0,05 |
| referencia: rectas 8×8 (512) + gris + rizo | **0,55 / 0,57** | 0,41 / 0,40 | — | 0,05 / 0,03 |
| rectas 8×8 + gris + gris-objetos + rizo | 0,50 / 0,52 | 0,35 / 0,36 | 0,11 / 0,09 | 0,04 / 0,03 |

(peso / sensibilidad). Y el peso **por feature**: en la elegida, rectas-objeto 0,070 · gris celdas 0,038 · rizo 0,079; con
gris-objetos, éstos 0,072 (igual que las rectas-objeto y el rizo), y el gris en celdas 0,032.

Lo que dice: **los tres grupos de objetos pesan lo mismo por número (≈0,07) y el doble que una celda del gris (≈0,035)**;
la red se fía más de cada número-objeto que de cada celda. Lo que decide la cuota es la cuenta: el gris en celdas sigue
siendo el 56 % con 320 números, y los 54 del gris-objetos valen el 21 %, más que los 44 de las rectas-objeto (17 %). El rizo
se queda en el 6 % con 15. Si el objetivo es que el reparto sea más parejo sin quitar información, el mando es el número
de celdas del gris, no el peso.

## ✅ Decisión del dueño (2026-10-12): TODO OBJETOS — «responsables» identificables (`nn/todo_objetos.py`)

*«La idea es tener una especie de responsables claramente identificables. Cuando un dígito no sea correctamente
identificado, vamos a los objetos y hallamos cuáles son los responsables de la falla. Por supuesto, un porcentaje no va a
ser identificado, porque depende de los detalles que estamos dejando de lado en esta etapa.»*

Compositor sólo con objetos: **rectas-objeto ×1 (44) + gris-objetos (54) + agujeros 3×3 + cuenta (10) = 108 números**
(sin celdas de ninguna clase; el rizo «actual» 2×2 también queda fuera). MLP 256 con aumento, 3 semillas
(`33-todo-objetos.json`):

| | val | a ciegas | cuota de peso rectas / gris-obj / agujeros | fallos val (s0) | en las 3 semillas |
|---|---:|---:|---|---:|---:|
| todo objetos (108) | **0,9637** (0,9629–0,9647) | **0,9527** | 0,38 / 0,54 / 0,08 | 59 | 53 |
| referencia celdas (911) | 0,9852 | 0,9792 | | 23 | 21 |

Es decir, **2,2 puntos menos en val y 2,7 a ciegas**: el precio de dejar los detalles, que el dueño da por asumido en esta
etapa. Confusiones más frecuentes: **8→9 ×9, 9→7 ×8**, 2→1 ×4, 3→9 ×4, 5→7 ×4, 2→9 ×3. Los 59 fallos, cada uno con sus
objetos encima (segmentos = rectas, círculos = curvas por sentido, × = golpe, + = agujero), en `33-todo-objetos-fallos.png`;
la lista con «real → predicho» y en cuántas semillas falla, en el JSON. De los 23 de la referencia arregla 8–10 (el 784
entre ellos, en las 3 semillas) y abre 44–45.

### Los 9 ochos que «todo objetos» lee como 9 (`33b-ochos-como-nueve.png`, 2026-10-12)

Arriba todos los objetos, abajo sólo el detector de agujeros. ⚠ **Corrige lo que se dijo al presentar la figura 33**
(«en varios el agujero de abajo no aparece»): **en 7 de los 9 el detector encuentra los DOS agujeros** (816, 1423, 1468,
1491, 1511, 1529, 1542). Sólo en 890 (lazo de abajo macizo, sin hueco) y 898 (hueco de abajo de 1–2 px) hay uno. O sea
que el responsable **no es el detector de agujeros**: la cuenta dice 2 y el compositor lee 9 igual. Lo que tienen en
común los nueve es la forma: lazo de arriba grande y redondo (círculos → y ← grandes arriba), lazo de abajo pequeño y
pegado a la derecha, con el trazo izquierdo descendiendo en diagonal — la silueta de un 9 con cola. Con la cuota de peso
de esta configuración (gris-objetos 0,54 · rectas 0,38 · agujeros 0,08), dos agujeros no bastan contra esa silueta.
El responsable es el **reparto**, y la salida barata es subirle el peso al grupo de agujeros (o darle más números), no
tocar el detector.

### Los mismos 9 ochos SIN rectas (`33c-ochos-sin-rectas.{png,json}`, 2026-10-12)

Compositor «todo objetos» con rectas (108), sin rectas (gris-objetos + agujeros, 64) y sólo agujeros (10); media de 3
semillas del margen logit 8 − logit 9:

| idx | con rectas | sin rectas | sólo agujeros |
|---:|---|---|---|
| 816 | 9 (−3,7) | 9 (−5,5) | **8** (+2,4) |
| 890 | 9 (−7,4) | 9 (−6,7) | 9 (−3,2) · un agujero |
| 898 | 9 (−5,4) | 9 (−5,9) | 9 (−2,7) · un agujero |
| 1423 | **8** (+0,4) | **8** (+2,0) | **8** (+3,0) |
| 1468 | 9 (−0,7) | 2 (−1,7) | 3 (+3,0) |
| 1491 | 9 (−2,4) | 9 (−3,2) | **8** (+5,1) |
| 1511 | 9 (−2,7) | **8** (+0,9) | **8** (+2,8) |
| 1529 | 9 (−3,9) | 9 (−4,0) | **8** (+3,1) |
| 1542 | 9 (−1,2) | **8** (+1,6) | **8** (+2,0) |
| val / ciega | 0,9637 / 0,9527 | 0,9441 / 0,9394 | 0,4945 / 0,4817 |

**Las rectas no son el responsable principal**: sin ellas sólo cambian 2 de los 9 (1511 y 1542 pasan a 8; 1468 se va a
2), y 5 siguen en 9 con márgenes iguales o peores. Quien dice 9 es el **gris-objetos** (la silueta: lazo grande arriba,
lazo chico abajo a la derecha, diagonal a la izquierda). Y **sólo los agujeros** leen 8 en 7 de los 9 —los dos que no,
890 y 898, son los que de verdad tienen un agujero—, aunque solos clasifiquen al 0,49. O sea que el grupo pequeño tiene
la respuesta correcta para estos y el grande lo tapa: el responsable es el reparto, como decía la figura 33b.

### Con MÁS dígitos de entrenamiento (`nn/mas_digitos.py`, 2026-10-13, unidad `rizo2-mas-digitos2`, 37 min)

«Todo objetos» (108) entrenado con los 180 de train y con 180 + los 3823 extra (otros escritores; 4003 dígitos, 52.039
filas con aumento), evaluado en val; 3 semillas (`34-mas-digitos.json`, figura `34-mas-digitos.png`). ⚠ Sin conjunto a
ciegas en la segunda fila: extra está en train.

| train | val | fallos val | de los 59 arreglados | nuevos | cuota agujeros | ochos bien (de 9) |
|---|---:|---:|---:|---:|---:|---:|
| 180 | 0,9637 | 57–60 | — | — | 0,079 | 1 |
| **4003** | **0,9833** (0,9827–0,9845) | **25–28** | 41–42 | 8–10 | 0,072 | **4** |

1. **Con 22× más dígitos, todo objetos sube 2 puntos** (0,9637 → 0,9833) y queda a 2 milésimas de la referencia en
   celdas entrenada con 180 (0,9852): los objetos no son peores features, son features que **necesitan más ejemplos**
   para que el compositor aprenda a combinarlas. Arregla 41–42 de los 59 fallos.
2. **Los ochos: de 1 a 4 bien** (1423, 1491, 1511, 1542), y en 890 y 898 —los de un solo agujero— el margen pasa de
   −7,4 / −5,4 a casi 0. Pero **la cuota de peso de los agujeros NO sube** (0,079 → 0,072): no aprende «los agujeros
   importan más» como regla general, aprende con más ejemplos **cuándo** dos agujeros con silueta de 9 son un 8. 816 y
   1529 siguen en 9 con los dos agujeros vistos, y 1468 se va a 3.
3. Lo que queda por decidir: para esta línea, el conjunto de entrenamiento de 180 era una **regla del boceto** (el
   compositor se entrena con pocos para que las features hagan el trabajo). Con objetos, 180 no bastan; con 4003 se
   pierde la medida a ciegas. Una partición nueva (p. ej. 2000 train / 1617 val / ~2000 ciegas) daría las dos cosas.
