# CLAUDE.md — `experimentos-cnn`

Este repo es para **probar estructuras de red sueltas**. Cada experimento trae su propio
código, sus propias reglas y su propio criterio, y puede no parecerse en nada al de al lado.
Léete esto antes de proponer nada: aquí la libertad es real, pero tiene una frontera escrita,
y la frontera es lo que evita que un experimento cueste dinero sin que nadie lo vea.

## Qué es, y por qué está separado de `foveal-vision`

Nació el **2026-09-06** para sacar los experimentos de `foveal-vision`, con el motivo que dio
el dueño al crearlo: *«foveal tiene muchas restricciones propias que entorpecen la
especificación de las pruebas»*.

**Y separarlos no es una preferencia: ya costó una vez.**
`foveal-vision/pyproject.toml:59` lleva `norecursedirs = ["experimentos", …]` porque los
snapshots congelados de los experimentos chocaron de nombre con los tests vivos y `pytest -q`
pasó a recoger **cero** tests. Un experimento es una ráfaga —se escribe en dos días y no se
toca más—; `src/fv` cambia todos los días. Relojes opuestos, piezas distintas (R1).

⚠ **Aquí no vive código de producción.** Si un experimento necesita cambiar `fv`, se copia
lo que haga falta a su `nn/` y se deja `foveal-vision/src/fv/` intacto. Es instrucción del
dueño del 2026-09-03: *«estos son experimentos… si hay que hacer cambios al código tendremos
que copiarlo localmente (pero si vale la pena, y eso depende de nuestras pruebas)»*.

## ⚠ Reglas del dueño (2026-10-11): los resultados se ENSEÑAN en imágenes, y un detector se construye como haga falta

> «Anota para claude que siempre me muestren los resultados en imágenes (raw de github, para evitar problemas de ayer).
> Anota también que un detector podría emplear un compositor si acaso hace falta, o podría crearse empleando
> entrenamiento de cnn. Lo importante es hallar la mejor solución. Tienes autorización para realizar los entrenamientos
> y pruebas sin mi supervisión, y puedes gastar hasta $10 en entrenamientos paralelos con tal de acelerar la obtención
> de resultados.» — el dueño, 2026-10-11

1. **Todo resultado se enseña con una IMAGEN**, y el enlace es el del fichero crudo:
   `https://raw.githubusercontent.com/stalinbeltran/experimentos-cnn/<hash>/<ruta>.png`. El visor de GitHub
   (`github.com/…/blob/…`) le falló al dueño el 2026-10-10 desde el móvil; el `raw` abre el PNG a secas. Con `<hash>`
   del commit, no `main`: un enlace a `main` cambia de contenido cuando cambia el fichero. La figura tiene que estar
   **commiteada y empujada** antes de dar el enlace.
2. **Un detector de features se construye con lo que mejor funcione**: puede ser un **compositor** (juntar detectores
   más pequeños —curvas, rectas, rayos— con reglas) o una **CNN entrenada** (con sintéticos de verdad conocida o con
   dígitos), o las dos cosas comparadas. No hay preferencia a priori: se mide cada estrategia con la misma verdad y los
   mismos números y gana la que pase más umbrales. El primero que lo aplica es `rizo-2` (`rizo-2`).
3. **Autorización para entrenar y probar sin supervisión, con tope de gasto: 10 $ por encargo** en entrenamientos
   paralelos (Vast, `nn/vast.sh` con el modo `trabajo` del lanzador) si eso acorta el tiempo hasta tener resultados.
   ⚠ Es una autorización para **acelerar**, no para gastar: una red de 32×32 entrena en minutos en el dev y alquilar no
   la acelera. Lo que gaste lleva su libro commiteado al alquilar y su freno (`/use exp-vast`), como siempre.

## ✅ El mejor detector de CURVAS CERRADAS hasta ahora (dueño, 2026-10-11): el compuesto de `rizo-2`

> «El resultado está MUY BIEN. Anota que este detector es el mejor hasta ahora para detectar curvas cerradas (nota que
> al final, el radio no importa).» — el dueño, 2026-10-11

Es `nn/compuesto2.py` de `rizo-2`: **proponen** las plantillas RODEADAS (cobertura ≥ 0,85; las
de media vuelta casan con cualquier curva) y los rayos (F 0,5); **elige** el ENCIERRO (fracción de 36 rayos que chocan
con tinta antes de 8 px), sin entrenar nada. En los 47 doses etiquetados a mano acierta el rizo el 78,7 % con un
candidato por dígito (89 % con dos), y 6 de los 10 fallos son la misma rendija con el centro en el fondo en vez de en la
boca. **El radio no importa**: lo que decide es que el trazo rodee al punto, no cuánto mide el hueco (el banco «con el
radio de los 2» no ayudó; el banco rodeado sí). ⚠ Lo que sí falla: en 4 de 47 elige otra concavidad de la misma cabeza
(el hueco cabeza–diagonal), y a veces un punto arriba y fuera de lo que un humano llamaría curva cerrada; separar eso
pide seguir el trazo, que ninguno de los tres mide. Detalle y figuras en su README.

## 🏆 EL MEJOR RESULTADO DE RECONOCIMIENTO DE DÍGITOS HASTA HOY (dueño, 2026-10-12)

> «Guarda que este es nuestro mejor resultado para reconocimiento de dígitos hasta el momento.» — el dueño, 2026-10-12

**MLP 256 · rectas + gris + rizo, entrenado con AUMENTO: 0,9858 en val (1617) con la semilla 0, 0,9852 con 3 semillas;
23 fallos, 21 de ellos en las tres semillas** (8→9 ×7, 3→9 ×3, 3→5 ×2). Movido ±4 px sigue en 0,94–0,96. Cómo se
reproduce: `nn/resistencia.py` de `rizo-2` (método B) y `nn/fallos_mejor.py` (la lista y la figura 19,
`resultados/19-fallos-mejor.{png,json}`). Qué es cada pieza:

- **rectas**: 4 Gabor par (0/45/90/135) integrados en línea, 2 escalas, máximo en celdas 8×8 (boceto curvas-gabor,
  `digitos.caract_rectas`; 512).
- **gris**: el detector de curvas sobre el borde Gabor impar gris λ_b 10 · K 9 re-calibrado (τ 0,1 · σ_E 0,5 · coh 0,25 ·
  λ 3): recto · κ→←↓↑ · golpe, celdas 8×8 (`digitos_bordes.caract` con `bordes_gabor.BordeGabor`; 384).
- **rizo**: la salida «rizo» del compuesto de curvas cerradas de `rizo-2` (`compuesto2.py` + candidatos al centro del
  hueco), compacta: mejor candidato en celdas 2×2 con su encierro, más la cuenta (5).
- **compositor**: MLP de una capa oculta de 256, ReLU, dropout 0,3, Adam 3e-3, L2 1e-3, 400 pasos, entradas
  estandarizadas; entrenado con los 180 de train **y 12 copias movidas** (±1, ±2, ±3 px en x y en y) de cada uno.

✅ **A ciegas (3823) con aumento, medido el 2026-10-12: 0,9785** (`nn/menos_rectas.py`; sin aumento era 0,969). La
referencia anterior era el boceto: 0,960 / 0,957 con el compositor lineal.

**Y el mismo 2026-10-12, por la tarde, lo mejor en VAL subió a 0,9878** (3 semillas, rango 0,9876–0,9882; a ciegas
0,9787, igual) añadiendo a lo de arriba dos grupos nuevos **sin quitar nada**: las **rectas como OBJETOS** (picos de «un
voto» agrupados por colinealidad; por orientación y escala, 3×3 de todas + cuenta + longitud; 88) y los **agujeros**
(un rizo por componente del fondo, 3×3 de todos + cuenta; 10). `nn/rectas_objetos.py` de `rizo-2`, fila «rectas
8×8 + objetos» de `resultados/27-objetos.json`. ⚠ En val, no a ciegas: ahí no mueve. Lo que se probó para que las rectas
**contaran menos** (celdas 4×4, un voto por recta, sólo objetos) reequilibra los grupos y arregla el 784, pero **baja a
ciegas** (hasta 0,971): las 512 rectas densas son redundantes y esa redundancia es lo que las hace estables. El detalle,
en el README de `rizo-2` § «Que las rectas cuenten MENOS».

## 🧩 LA LÍNEA «TODO OBJETOS» (decisión del dueño, 2026-10-12/13): responsables identificables

El dueño eligió trabajar con un compositor hecho **sólo de objetos** —rectas como objetos (×1, 44), curvas del gris como
objetos (54) y **huecos cerrados y concavidades exteriores por separado** (3×3 + cuenta cada uno, área mínima 3 px; 20)—,
118 números, para que cuando un dígito falle se pueda ir a los objetos y nombrar al responsable. Vigente:
**0,9775 val / 0,9592 a ciegas** (180 de train, aumento, 3 semillas), contra 0,9852 / 0,9792 de la referencia en celdas:
ese es el precio asumido de dejar los detalles en esta etapa. Lo medido por el camino y lo que NO ayudó (celdas 4×4, un
voto por recta, quitar el «recto» del gris, 22× más dígitos de entrenamiento, el prior por cuenta de huecos) está en el
README de `rizo-2` § «Que las rectas cuenten MENOS» y siguientes. ⚠ **Los huecos van separados de las
concavidades** (`rizo_huecos.huecos_separados`): mezclados, «cuenta 2» era un 8 el 36 % de las veces; separados, el 79 %.

## 📌 LOS MEJORES HASTA HOY (2026-10-12): detectores y compositores — para recordárselo al dueño cuando pregunte

Medido en `rizo-2` (`nn/detectores.py`, `resultados/17-detectores.json`; 180 train / 1617 val /
3823 a ciegas, 3 semillas; val / a ciegas):

- **Detectores de features, de mejor a peor solo**: **gris** (curvas del boceto curvas-gabor sobre el borde Gabor impar
  gris λ_b 10 · K 9 re-calibrado; 0,960 / 0,957 lineal) · **rectas** (4 Gabor par integrados, 2 escalas; 0,955 / 0,947)
  · **delgado** (curvas sobre el dígito tal cual; 0,929 / 0,921) · **rizo** (la salida «rizo» del compuesto de curvas
  cerradas, compacta: mejor candidato en celdas 2×2 + cuenta, 5 números; solo no clasifica, 0,37, pero **suma en toda
  combinación**, +0,3 a +1,0 en val).
- **Detector de curvas cerradas**: el **compuesto** `compuesto2.py` (plantillas rodeadas cob ≥ 0,85 ∪ rayos proponen; el
  encierro por rayos elige; el candidato se mueve al centro del hueco). Rizos del 2: 80,9 % top-1. Dos salidas: «rizo»
  (hueco libre, rayos 8 px) y «lazo» (hueco ≥ 2,5 px, rayos 14 px: sólo lazos de verdad, 0,7 por dígito).
- **Compositores**: **MLP 256** (una capa oculta, dropout 0,3, Adam, 400 pasos) es el más honesto —mejor a ciegas y
  poca varianza—; el **bosque aleatorio 500** gana en val y pierde a ciegas (ajusta a val); el lineal del boceto queda
  ~1,5 puntos por debajo; boosting y k vecinos, peor que el lineal con 180 ejemplos.
- **La mejor combinación honesta: MLP 256 · rectas + gris + rizo = 0,972 val / 0,969 a ciegas** (la referencia del
  boceto era 0,960 / 0,957). Con todo (+ delgado): 0,970 / 0,970. El máximo en val, bosque · gris + delgado 0,975, cae
  a 0,962 a ciegas.
- **Y con AUMENTO (2026-10-12, `resistencia.py`): 0,985 en val sin mover, y 0,94–0,96 movido ±4 px** (el base caía a
  0,30–0,56): entrenar el MLP con 13 copias movidas ±1–3 px de cada dígito. Es la mejora más grande del experimento.
  ⚠ Medido sólo en val; a ciegas, pendiente.

## ⚠ Regla del dueño (2026-10-15): todo README de experimento lleva un GLOSARIO, y cada término nuevo entra en él

El dueño lo pide siempre y hasta hoy no estaba escrito en ningún sitio. Cada README de experimento tiene una sección
`## 📖 GLOSARIO` con los términos del experimento en lenguaje llano: la cadena de datos (qué entra, qué sale), cada
objeto/feature con su definición en una línea, cómo se codifica, y qué significa cada nombre de corrida. **Cuando se
inventa un término nuevo** (entrante, cono, resto, fundido/aparte…), **se añade al glosario en el mismo commit** que lo
usa. Al describir corridas al dueño, cada nombre de corrida se explica con lo que lleva, no se da por sabido. El primero
está en `2026-10-11-rizo-dos/README.md`.

## ⚠ Regla del dueño (2026-10-15): TODO detector lleva POSICIÓN; si uno no la lleva, se le DICE

> «Normalmente quiero posición en todos.» — el dueño, 2026-10-15

Un feature que entra al compositor **sin posición** (sólo presencia, cuenta o fuerza, como los «entrantes a 4
direcciones» del § 59 de `rizo-2`) es una **excepción de prueba**, no la forma normal. Vale para medir cuánto aporta
la posición (allí: 3 puntos, § 59 contra § 58), y nada más. Dos obligaciones:

1. **Al proponer o correr un feature, se dice explícitamente si lleva posición y cuál** (hoy: rejilla 3×3 con el
   máximo por celda, § 55 y § 63 de `rizo-2`). Si no la lleva, se avisa **en la misma frase** en que se presenta, y
   se dice por qué (qué se está midiendo con ello).
2. **La forma normal de cualquier detector nuevo es CON posición.** Lo medido en `feat-ind` manda: presencia sola
   0,718, posicional 0,959 (+0,24). Reducir a «sin posición» es una decisión que se toma a propósito y se anota, no
   un atajo.

Lo que esto NO fija: **cómo** se codifica la posición. La rejilla 3×3 es lo que hay medido; sus límites están en
`rizo-2` § 55 (4×4 no mejora), § 60–61 (empate diagonal de la dirección) y § 63 (una mancha cae en dos celdas; el
60 % del bloque son ceros). Alternativas no probadas: centro en píxeles en una lista ordenada de objetos, reparto blando
entre celdas o direcciones vecinas.

## ⚠ Regla del dueño (2026-10-16): los GOLPES NO van por defecto en un compositor

> «No queremos usar golpes por default.» — el dueño, 2026-10-16, al ver que «aprobada + vértice» los lleva

Un **golpe** es el pico de curvatura |κ| ≥ 0,45 de la cresta (`rizo-2` README, tabla de bloques; bloque `golpes` =
rejilla 3×3 + cuenta, 10 números). **Hoy las CINCO recetas del revisor de `rizo-2` lo llevan** —las cuatro base
(`RECETAS_BASE` en `web/servidor.py`) y «aprobada + vértice»—, heredado de las corridas de 144 del § 40 sin que nadie
lo decidiera aparte. A partir de ahora:

1. **Un compositor nuevo se propone SIN `golpes`.** Meterlo es una decisión que se toma a propósito y se escribe con su
   motivo (qué se está midiendo con él), igual que la excepción de «sin posición» de la regla anterior.
2. **Las recetas ya entrenadas NO se tocan por esto**: sus pesos están en el almacén y sus números en el README (§ 85i,
   § 95, § 96). Cambiarlas es otro entrenamiento y otra tabla, y lo ordena el dueño.
3. ⚠ Lo que NO está medido: cuánto aporta `golpes` por sí solo en las recetas con el 4 en dos. La cuota del bloque la
   enseña el revisor (pantalla del compositor); medir la receta «aprobada + vértice − golpes» es el experimento que
   diría si el defecto nuevo cuesta algo.

## ⚠ Regla del dueño (2026-10-17): POR DEFECTO, TODO se calcula sobre el DELGADO, no sobre la tinta

> «No sabía que el vértice se calculaba sobre la tinta. Anota que por default todo se calcula sobre el delgado. Debemos
> modificar todos los features, pues hay un divorcio entre lo que yo esperaba y lo que el sistema hace.» — el dueño, 2026-10-17

### ⏳ PENDIENTE (el dueño, 2026-10-17): pasar al delgado lo que hoy va sobre la tinta — NO recalculado todavía

Inventario hecho ese día leyendo el código, y revisado por el agente `verificador`. **En lo que usa el revisor** (`caract` de
`rizo-2/nn/cresta_curvas.py` y `rizo-2/web/servidor.py`) el divorcio es **sólo del vértice**; el resto ya sale del delgado
(`caract(M, …)` sólo pasa la tinta al bloque del vértice):

| qué | dónde | entrada hoy |
|---|---|---|
| bloque `vertice` (§ 95) | `cresta_curvas.vertices` / `bloque_vertice`, llamado desde `caract` | ❌ **tinta** |
| bloque `vertice_1` (§ 98) | `web/servidor.py: bloque_vertice_1` → `CC.vertices` | ❌ **tinta** |
| el vértice que dibuja el revisor en la ficha del dígito | `web/servidor.py` (líneas de `ob["vertices"] = CC.vertices(d.tinta[i])`) | ❌ **tinta** |
| rectas, arcos, lazos (y región/área), entrantes, golpes, restos, cruces | `cresta_curvas.objetos(M)` | ✅ delgado |
| el 4 partido en 4l/4a (`subgrupo4`) | `web/servidor.py` | ✅ delgado |

⚠ **Fuera del revisor la tinta sigue en muchos sitios, y eso NO está auditado**: 33 scripts antiguos de `rizo-2/nn/` leen
`imagenes` del dataset y calculan sobre la tinta (p. ej. `todo_objetos.features` = rectas + gris + huecos, `detectores.py`,
`menos_rectas.py`, `resistencia.py`, `compuesto*.py`, `evaluar.py`), y otros 28 experimentos del repo también la leen. Son
anteriores al delgado y no alimentan el revisor; si se van a reutilizar, primero se pasan al delgado. Lista sacada con `grep`,
no leída fichero a fichero.

El sustituto natural es el **vértice libre** sobre la cresta del delgado (`rizo-2` § 102–103), pero **cambiarlo cambia la
huella de los compositores guardados** y el dueño ordenó el 2026-10-17 **no entrenar ningún compositor por ahora**. Cuando
se haga: la tabla de arriba es la lista; quitar `tinta` de `caract` y del revisor; y medir la curva de desplazamientos.

### ⏳ PENDIENTE (el dueño, 2026-10-17): la CURVATURA del vértice — afilado o redondeado

> «El delgado nos permite apreciar si el vértice es afilado o redondeado, y ese dato puede ser útil (lo vamos a probar más
> adelante).»

Un dato más por vértice, con la misma posición única que su presencia y su amplitud. **No implementado ni definido por el
dueño**; dos propuestas para elegir: (a) la distancia entre el punto de la cresta y el cruce de las dos rectas de los brazos
(≈ 0 px en una V, varios px en una U), escalada a 0–1; (b) cuánto se abre el ángulo al mirarlo de cerca (a 3 px) contra de
lejos (a 6 px), que el prototipo ya calcula para descartar curvas.

### La regla

> «La entrada default no es la tinta sino delgada (le aplicamos una transformación).» — el dueño, 2026-10-17, al ver una
> figura con la tinta de fondo

El **delgado** es la salida de `delg-cnn` (`2026-10-15-delgado-cnn`): una CNN entrenada con trazos sintéticos que convierte
el dígito en un **mapa continuo** 32×32 de por dónde pasa el bolígrafo (abre los lazos macizos, deja los cruces como dos
líneas). Fichero: `2026-10-15-delgado-cnn/resultados/delgados-k3x3.npz`, clave `M` (float, 5620 dígitos, mismo orden que el
dataset); lo regenera `nn/exportar_delgados.py` en segundos. La **cresta** (`rizo-2/nn/cresta_curvas.tramos`) es el
esqueleto de `M > 0,4`.

1. **Toda feature, detector o figura nueva se calcula sobre el delgado.** La tinta (`imagenes > 0` del dataset) sólo entra
   si se decide a propósito, y entonces se **dice** en el código, en la figura y en el README («ENTRADA: la tinta, porque…»).
2. **Las figuras enseñan el delgado de fondo**, no la tinta: si el fondo es tinta, el dueño lee (con razón) que el cálculo
   también lo fue. Pasó con `rizo-2` § 102: la detección iba sobre el delgado y el dibujo, sobre la tinta.
3. ⚠ **Lo que hoy NO cumple la regla, y está en uso**: el bloque `vertice` del revisor (`cresta_curvas.vertices`, § 95) mira
   los bordes de la **tinta**. El dueño ordenó el 2026-10-17 pasarlo al delgado (pendiente de arriba, sin recalcular aún). Medido el
   2026-10-17 (`rizo-2` § 101): ese mismo detector sobre el delgado (`M > 0,4`) dispara N en 10 cuatros de val, contra 16
   de 26 cuatros cerrados sobre la tinta; el vértice sobre la cresta (§ 102) da 23 de 26.

## ⚠ Regla del dueño (2026-10-17): UNA rejilla por DIRECCIÓN, y sólo hay 4 — N, E, S, O

> «Cada rejilla debe corresponder a solo una dirección de las 4. No hay ninguna dirección NE, sino que hasta cierto rango
> (ej. hasta 45 grados) un vértice NE será considerado N o E, pues no quiero meter tanto detalle.»

1. Un feature con dirección lleva **una rejilla 3×3 por dirección**, y las direcciones son **4**: N, E, S, O (antes
   arriba/derecha/abajo/izquierda). Cada una es un **sector de 90°** del rumbo: N [315°, 45°), E [45°, 135°), S [135°, 225°),
   O [225°, 315°). **No hay diagonales**: un vértice que apunta a 30° es N y uno a 60° es E.
2. **Las celdas de una rejilla NO se rotulan con letras de brújula.** Una celda es un SITIO del dígito (tercio de arriba/medio/
   abajo × izquierda/centro/derecha); rotularla «NE» se lee como una dirección diagonal. Pasó en `rizo-2` § 101. Si hay que
   nombrarlas, por fila y columna («arriba-der»), nunca con N/S/E/O.
3. Se aplica de aquí en adelante a todo feature con dirección (vértices, entrantes…); renombrar en el código los ya existentes
   (los conos de entrantes usan `→ der`, `← izq`, …) va con el pendiente del delgado: es tocar la huella de lo guardado.

## ⏳ PENDIENTE (anotado el 2026-10-15, pedido por el dueño): AUTOMATIZAR LA BÚSQUEDA DE FEATURES — no empezado

> «Partimos de que un píxel es un feature, y al poner un problema (por ej. la identificación de un dígito) obtenemos la
> idoneidad de un píxel como feature, calculando qué tan bien sirve un píxel como feature. Notar que un píxel es siempre
> la entrada básica disponible en visión artificial, pero eso no significa que sea el mejor feature. Una vez evaluada la
> idoneidad del píxel, pasamos a proponer otros features (cualquiera), y también los evaluamos. Probablemente sirva
> evaluar los features con pocas muestras para una rápida evaluación. Esto nos permitirá, con cada propuesta, contar con
> un set de features calificado (o sea, cada uno tendrá una cierta calificación), donde los mejores son los más aptos para
> identificar dígitos. Luego podemos proponer features compuestos creados de alguna forma a partir de features simples, y
> también los evaluaremos. También proponemos features compuestos combinados con features simples, etc. Así continuamos
> ampliando gradualmente la lista de features y obteniendo los mejores para un trabajo dado.» — el dueño, 2026-10-15

**Qué es**: un ciclo que **califica** features para un trabajo dado y **crece** la lista a partir de las mejores, en vez
de proponerlas una a una a mano como se ha hecho hasta hoy (rectas, arcos, lazos, concavidades, entrantes…). Cuatro
piezas, en este orden:

1. **La calificación de un feature.** Un número por feature, para un trabajo (hoy: identificar el dígito). Es la
   decisión de diseño de la que depende todo lo demás, y hay que escribirla con su criterio **antes** de medir (R13).
   Candidatas, ninguna elegida: información mutua con la clase · acierto de un clasificador de **un** feature ·
   ganancia marginal al añadirlo al mejor conjunto actual (la única que castiga la redundancia: dos features iguales
   valen como uno). ⚠ La tercera es la que corresponde a «los mejores para un trabajo», pero depende del conjunto y
   cambia en cada vuelta.
2. **El suelo: el píxel.** Los 1024 píxeles de 32×32 (o los 64 de 8×8) calificados uno a uno. Ya hay un número para
   el conjunto entero —el compositor sobre píxeles crudos lee 0,890 en `feat-ind` y 0,908 en `por-que-fallan`—, pero
   nunca se ha calificado **cada** píxel. Es la referencia contra la que cualquier feature propuesto tiene que ganar.
3. **Evaluación rápida con pocas muestras.** Calificar con 36–180 dígitos, que es lo que ya hace el compositor, y
   comprobar que el ranking con pocas muestras **predice** el ranking con muchas (es la condición para que el ciclo sea
   barato; si no se cumple, se sabe antes de ampliar nada). La curva de `feat-ind` corrida 11 (36 → 1080 dígitos) es
   el antecedente.
4. **La composición.** Proponer features compuestos a partir de los mejores simples —y mezclas de compuestos con
   simples— y volver a calificar. Cómo se componen es una decisión abierta: por operaciones fijas (máximo, producto,
   diferencia, «A y B en celdas vecinas», como el combinante de la corrida 10), por posición relativa
   (`posicion-relativa-features`), o aprendidas.

**Qué hay ya que sirve de pieza**: todos los detectores de `rizo-2`, `feat-ind32` y `rect-bor` son features
candidatos con su cuota de peso medida (`cuota_peso` en los `40-entrada-*.json` es una calificación *dentro de un
conjunto*, no absoluta); el protocolo 180 train / 1617 val / 3823 a ciegas con 3 semillas; y la regla de los
desplazamientos, que aquí sería una calificación más (qué tan estable es el feature al mover la entrada).

**Qué NO está decidido, y hay que preguntar antes de empezar**: la calificación (punto 1), si «feature» es un número
por dígito (como la cuenta de lazos) o un mapa (como un píxel o un detector convolucional, que son 1024 features o uno
según cómo se mire), y cuánto cuesta una vuelta del ciclo (R-gasto: en CPU local es gratis mientras el feature sea
barato; un feature aprendido ya no).

## ⏳⏳ PENDIENTE PARA EL 2026-10-12 (encargo del dueño del 11): BUSCAR ESTRATEGIAS PARA DETECTAR LAZOS Y CURVAS — empieza aquí

> «Necesito que me ayudes observando los dígitos y aplicando distintas estrategias hasta hallar unas que nos permitan
> detectar features como estos lazos y curvas. Debes evaluar varios, y medir su efectividad, y luego sugerir otros. Luego
> me reportas tus avances y así continuamos.» — el dueño, 2026-10-11

**Qué se pide**: un ciclo, no un experimento cerrado — **observar** los dígitos, **probar varias estrategias**, **medir** cada
una con los mismos números, **sugerir** otras, y **reportar** al dueño antes de seguir. El reporte enseña figuras (con el
dibujo que él pidió para los lazos: dígito en gris, cruz azul en el centro, arco delgado con el hueco hacia la abertura) y
los enlaces directos al PNG (`raw.githubusercontent.com`, porque el visor de GitHub le fallaba).

**De dónde se parte** (todo el 2026-10-10/11, 0 $):
- **Curvas**: el detector DELGADO del boceto `docs/bocetos/2026-10-09-curvas-gabor/` (`delgadas.py`): 12 Gabor par λ 6
  sobre la imagen, giro κ a ±4 px, trozos rectos/curvos con radio, centro y orientación (vector de curvatura). Sintéticos de
  1–3 px: 99,7 % rectas, 97,7 % arcos. TAU recomendado 0,6 (`tau_comparar.py`), **sin confirmar por el dueño**.
- **Lazos**: el experimento `lazos`. Votos por centro de curvatura: bien en lazos sintéticos limpios
  (centro 0,6 px, cierre 0,04, orientación 2°), **mal en dígitos** (duplicados en óvalos, falsos en 7, el 8 sin sus dos
  lazos). Caso señalado por el dueño, **el 2 nº 50 del dataset** (`nn/caso.py 50`): un «lazo» de radio 12 (el máximo
  permitido) con 5 votos de 85 cuya corona ancha junta la cabeza y la base del 2 — **dos curvas distintas leídas como un
  lazo**. Y el **radio mínimo que ve es ≈ 5 px** (giro a ±4 px): los rizos de los 2 se pierden; un **sondeo por rayos** los
  recoge (`nn/rizos.py`: 11/12 huecos de los 2, 95 % de los 8).

✅ **El 2026-10-11 se empezó por el RIZO SUPERIOR DEL 2** (encargo del dueño de ese día): el experimento `rizo-2`
(id `rizo-2`) compara cuatro estrategias —rayos abiertos, plantillas, votos de curvatura (la referencia) y
una CNN entrenada con sintéticos— contra **48 doses etiquetados a mano** por Claude. **Lo que salió (tres vueltas, 0 $): NINGUNA pasa**. Lo mejor es la CNN genérica, 57 % de los doses con 32 % de falsos
en 1/4/7 (figura 6 de su README); en sintéticos llega al 94 %. Dos hallazgos que valen para la siguiente: el hueco del
rizo real es una **rendija de 1 px de radio inscrito** (mediana), y las etiquetas a ojo se van 1–2 px (afinadas por
encierro). Siguiente vuelta sugerida: entrenar con dígitos reales usando el píxel más encerrado como verdad, y P como
proponente + CNN como verificador. Detalle y figuras en su README.

**Estrategias para empezar a probar** (de lo medido; las 2, 5 y 8 se prueban en `rizo-2`):
1. **Exigir CONTINUIDAD**: un lazo es UN trazo que gira, no tinta en sectores — seguir el trazo a lo largo del arco (la
   cadena de trozos curvos con el mismo centro), en vez de contar sectores con cualquier tinta.
2. **Votos relativos**: el pico debe juntar una fracción de los píxeles curvos (p. ej. ≥ 30 %), no ≥ 3 votos absolutos.
3. **Corona estrecha y radio máximo relativo** al tamaño del dígito.
4. **Explicar la tinta**: aceptar un lazo, quitar sus votos y su tinta, buscar el siguiente (óvalos, el 8).
5. **Rizos por rayos** para los lazos de radio < 5 px, combinados con los de curvatura.
6. **Multiescala**: giro a ±2/±4/±6 según el grosor y el radio.
7. **Signo de la curvatura a lo largo del trazo** (C contra S): para separar la cabeza y la base del 2.
8. **Aprendido**: un detector pequeño entrenado con sintéticos de verdad conocida (lazos, curvas, rectas a varios
   grosores) — entrena-local, 0 $; la red ve la imagen, no reglas.

**Cómo medir, para que las estrategias se puedan comparar**: (a) sintéticos con verdad (los de `lazos/nn/lazos.py`, más
grosores 4–6 px como los dígitos); (b) en dígitos, lo esperable por clase (0/6/9 un lazo cerrado, 8 dos, 1/7 ninguno) y los
huecos cerrados como verdad medible de los lazos cerrados; (c) **proponer al dueño etiquetar a mano ~50 dígitos** (lazos y
curvas, con centro aproximado): es la única verdad de dígitos que no depende de suposiciones por clase. Cada estrategia
nueva, con su criterio escrito antes (R13) si va a decidir algo.

## ⏳ PENDIENTE (2026-10-09): la línea de RECTAS — empieza aquí

Dos experimentos cerrados el 2026-10-08 (`rect-lin`, `rect-bor`; reportes #38 y #39 del central) y un tanteo del 09 que
cambia el rumbo: **4 kernels Gabor FIJOS, sin entrenar, conservando el MAPA (no su max) e integrados a lo largo de su
orientación en 2 escalas, leen los dígitos 0,955** (180/1617, compositor lineal) contra 0,949 de los 13 detectores CNN de
`feat-ind32`. Detalle en el README de `rect-bor` § «Tanteo». Nada vivo ni pagado pendiente.

Lo que falta, por orden de lo que pidió o aceptó el dueño:

1. **Convertir el tanteo en experimento** con su criterio escrito antes: Gabor fijo + mapa + integración + 2 escalas →
   compositor. Con curva de N (180 → 900+), más semillas, y las referencias re-ejecutadas con el MISMO protocolo (el
   `verificador` sólo las comprobó en los documentos). Y ampliar el vocabulario FIJO (arcos, esquinas) para ver si suma.
2. **Cómo leer el mapa en vez del max** (el dueño: «el problema es que usamos el máximo»). Probada sólo la opción 1
   (integrar a lo largo de la orientación). Faltan la **2** (segmentos: umbral + componentes conexas → largo, ancho,
   orientación de cada trozo) y la **3** (votación tipo Hough), que es la única que podría unir **puntos muy separados**.
3. **Curvas como segunda etapa sobre rectas cortas** (el dueño, 2026-10-08): una curva = cadena de detecciones cortas cuya
   orientación gira. ✅ **Boceto hecho el 2026-10-09** (`docs/bocetos/2026-10-09-curvas-gabor/`, 0 $): 12 Gabor fijos →
   orientación local → giro κ a lo largo de la tangente → recta/curva (con radio)/esquina/mancha. Sobre el banco de
   `rect-lin`: 0 % de rectas finas leídas como curva, 99 % de los arcos R ≤ 27 como curva; radio útil 5–30 px. Falla en
   esquinas de brazos cortos, trazo grueso (≥ 8 px) y punteadas. Lo que sería el experimento está en su README.
   ⚠ **Desde el 2026-10-10, la idea del dueño: el detector trabaja SÓLO sobre los BORDES de la imagen real** (un trazo
   grueso = dos curvas, su contorno de fuera y el de dentro). Hecho en el mismo boceto (`bordes.py`, 0 $): contorno de
   1 px → Gabor λ = 3 (el más estrecho que funciona; λ = 2 no ve ningún arco) → giro por trozo. Banco de `rect-lin`,
   mismo criterio: arcos R ≤ 27 90,6 % (sobre el trazo, 86,9 %), rectas finas sin curva falsa 96,9 % (94,8 %). Radio
   útil 5–20 px. Umbrales elegidos, no calibrados. Pendiente: emparejar las dos curvas de un trazo.
   **Orientación** (2026-10-10): el vector de curvatura κ·(−sin θ, cos θ) apunta al centro; error mediano 0° en la galería.
   ⚠ **Desde el 2026-10-10 el borde es de GABOR IMPAR, no morfológico** (regla de arriba; `bordes_gabor.py`): dígitos
   **0,938** val / **0,925** a ciegas sin suavizar. Lo de abajo se midió con el borde morfológico, ya retirado.
   ⚠ **Y SIN binarizar, con el detector re-calibrado con la galería** (`gris_recalibrado.py`): kernel 9 · λ_b 8–12 →
   **0,960–0,963** val / **0,946–0,957** a ciegas, por encima del binarizado; supera a rectas (0,955 / 0,947) en val, y a
   ciegas sólo con λ_b 10–12. Verificado ejecutando (agente `verificador`) λ_b 8 y 10 · K 9.
   **Dígitos** (2026-10-10, `digitos_bordes.py`, REEMPLAZA a la variante «curvas» de `digitos.py`): sólo este detector →
   **0,889** val / 0,868 a ciegas (la vieja, 0,867 / 0,836; rectas, 0,955). La orientación aporta +0,011. Con ±1 px
   cambia el 10–11 % de las lecturas (rectas, 4–5 %).
4. **Grosor.** (a) Gabor más ancho o con 2 escalas como detector (en `rect-lin` daba 0,88 en 6–8 px); (b) entrenar con
   trazo 2–8 px y Sobel delante, para comprobar la hipótesis «aprendió dos bordes juntos, no un borde» (`rect-bor`);
   (c) bordes de un solo lado, que necesitan un detector de varios canales.
5. **El kernel APRENDIDO salió ruidoso y responde desplazado del borde** (visto por el dueño en `rect-bor`,
   `resultados/ejemplo-curva.png`). Arreglos propuestos y sin probar: kernel simétrico, kernel paramétrico (3–4
   números), o supervisar DÓNDE está la recta (objetivo = mapa) y no sólo «hay recta». Causa probable, sin comprobar:
   la supervisión sólo dice «en alguna parte».
6. **El banco de prueba tiene figuras que no son lo que dice su etiqueta**: «rectas» de 10 × 14 px (un bloque) y
   «curvas» de radio 9 y 8 px (una mancha). Quitarlas y re-medir; 0 $, sin re-entrenar.
7. **Rectas suavizadas (antialias) contra binarias**: todo lo medido es binario, como los dígitos de NIST. Medir si
   entrenar suavizado ayuda en el banco binario.
8. **Método**: N equilibrado por orientación en la curva de aprendizaje (con N = 4 hay orientaciones sin ejemplo), y
   guardar los kernels SIN redondear (con 4 decimales, el control re-evaluado de `rect-bor` difiere hasta 0,028).
9. **Punteadas**: un kernel no las generaliza en ningún sentido (`rect-lin`). Es la opción 3 de arriba.

⚠ El pendiente de `feat-1lado` («ningún detector de recta ve la MITAD de las rectas reales») queda **contestado en
parte** por `rect-lin`: el hueco era el ÁNGULO entre clases, y un kernel lineal con 4 orientaciones lo cubre (0,98 a
15–22,5° de su centro, contra 0,30 de la CNN). Lo de los arcos de esa lista sigue abierto.

⚠ Fuera de este repo, y conocido: el freno (`cerrable.mjs`) dice `NO SÉ` con una máquina de Vast lanzada desde `~/src`
(sin `WORKSPACE.json` no hay prefijo). Es el P1/P2 de `telegram-coordinator/docs/freno-prefijos-2026-09-01.md`.

## ⏳ PENDIENTE (2026-10-07): detectores de borde de un solo lado

Línea de trabajo abierta con el dueño: bordes de un lado como único pre-proceso, una «vista» por dirección, y qué
compositor usar. Todo lo hecho (figuras, pruebas rápidas a 0 $, resultados) y la lista de lo que falta están en
`docs/bocetos/2026-10-07-borde-de-un-lado/README.md` § «PENDIENTE». Nada lanzado ni pagado. Grosor y compositor
ya decididos (§ «Cuarta parte»). El experimento corrió: `feat-1lado`. ⚠ **Para el 2026-10-09:** su README §
«PENDIENTE para mañana» — la auditoría mostró que ningún detector de recta ve la MITAD de las rectas reales de los dígitos.

## ⚠ Regla del dueño (2026-10-07): UN SOLO pre-proceso, y el mismo para TODOS los dígitos

> Primera versión (2026-10-07): «no se aplica pre-proceso a ningún dígito antes del detector de features».
> Reescrita el mismo día: «si se aplica un pre-proceso debe aplicarse a todos los dígitos antes de los detectores».

1. **Como mucho UN pre-proceso**, y se aplica **igual a todo** lo que vean los detectores: entrenamiento sintético,
   pruebas y dígitos. Nada de transformar sólo los dígitos al evaluar (que es lo que hizo `por-que-fallan` con el
   esqueleto y la erosión).
2. El candidato vigente es el de **bordes de un solo lado** (`docs/bocetos/2026-10-07-borde-de-un-lado/`). El código
   viejo (`nn/bordes.py` de `feat-bor`, esqueleto, erosión) se queda y **no se usa** salvo que el dueño
   lo elija.
3. Un experimento declara en su `REGLAS.md` qué pre-proceso usa (o «ninguno»). Es una regla de esta línea de trabajo;
   no hereda la Regla 0: un experimento que quiera saltársela lo pregunta.

## ⚠ Regla del dueño (2026-10-10): el BORDE se saca con un filtro GABOR, nunca con morfología

> «No hagas más la obtención de bordes que haces, anota eso. Reemplázalo con filtro Gabor.»

Hasta ese día el borde era `x AND NOT erosión₃ₓ₃(x)` (`docs/bocetos/2026-10-09-curvas-gabor/bordes.py`, función
`bordes`). **No se usa más en trabajo nuevo**; el código se conserva sólo para reproducir las figuras ya publicadas. El
borde se saca con un banco de **Gabor IMPAR** (seno: responde a un escalón, no a una línea), y se binariza en 0,5 si el
detector que lo lee lo necesita: un umbral sobre la salida del Gabor no es morfología. La implementación de referencia
es `BordeGabor` en `docs/bocetos/2026-10-09-curvas-gabor/bordes_gabor.py` (λ 6, kernel 5×5, 8 orientaciones,
binarizado). Medido ese día con el compositor de dígitos: 0,938 en val y 0,925 a ciegas sin suavizar, contra 0,889 y
0,868 del borde morfológico. Si un Gabor no se puede aplicar a un caso, **se dice y se analiza la alternativa con el
dueño**; no se vuelve a la erosión en silencio.

## ⚠ Regla del dueño (2026-10-09): la resistencia a los DESPLAZAMIENTOS se mide SIEMPRE

> «Siempre que hagamos reconocimiento debemos evaluar la resistencia de cualquier experimento a las variaciones de
> posición. Probablemente se obtengan curvas, probablemente distintas según los píxeles desplazados. Eso para números
> enteros. Para features individuales también queremos curvas, pues los desplazamientos siempre están presentes en
> reconocimiento visual.»

**Por qué, medido el mismo día** (boceto `docs/bocetos/2026-10-09-curvas-gabor/`, § «Los gemelos» y
`imagenes/9-desplazamientos.png`). El compositor lineal de dígitos que leía val a 0,958 falla un 5 nítido que tiene un
gemelo casi idéntico en entrenamiento: la barra media está 1–1,5 px más alta y cruza una frontera de celda. Moviendo
todo val:

| compositor combinado, movido en horizontal | d = 0 | ±1 px | ±2 px | ±4 px |
|---|---:|---:|---:|---:|
| acierto | 0,958 | 0,945 / 0,936 | 0,842 / 0,870 | 0,267 / 0,491 |

**Ningún experimento de reconocimiento de este repo había medido esto**, y un acierto a d = 0 no dice nada de lo que
pasa con 1 px. Desde hoy, todo experimento que reconozca algo (el objeto entero, cada feature, o los dos) trae su curva.

### Qué se mide

1. **Los dos niveles, cada uno con su curva.** El **objeto entero** (p. ej. el acierto del dígito) y **cada feature
   individual** (el recall de cada detector, y sus falsos positivos). Para una feature, el desplazamiento que importa es
   también el de su posición **relativa a la rejilla de quien la lee**: un detector puede ser invariante y su compositor
   no.
2. **Métrica contra d, con d = −k…+k px enteros**, horizontal y vertical **por separado** y **cada signo aparte**: las
   curvas salen asimétricas (medido: −1 px arregla el 5 que +1 px empeora). k ≥ 4, y nunca menor que la celda o el
   stride más grande de la red.
3. **Además del acierto, el % que CAMBIA de lectura respecto de d = 0.** El acierto puede moverse poco mientras muchas
   lecturas cambian en los dos sentidos (medido: con ±1 px el acierto baja 0,013 y cambian 121 de 1617).
4. **Sólo evaluación: se mueve la entrada, no se re-entrena.** Entrenar con copias desplazadas es un **arreglo**, y si
   se prueba va como brazo propio, declarado en el criterio y medido con la misma curva.
5. **Lo que sale por un borde se PIERDE y se CUENTA.** No se usa `np.roll`: hace reaparecer la tinta por el lado
   contrario y mide otra cosa. Si mover corta tinta, la curva mide el recorte y no la tolerancia, así que el % recortado
   va en la misma figura. ⚠ **Los dígitos de UCI ocupan los 32 px de alto** (1555 de 1617 en val): con ese dataset la
   curva vertical está recortada desde d = ±1 y no vale. **Un dataset nuevo de reconocimiento deja un margen ≥ k px**
   en el lienzo para que las dos direcciones se puedan medir.
6. **Una curva plana es un resultado, y se dice por qué.** Una convolución es exactamente equivariante a desplazamientos
   enteros, así que un detector sin rejilla fija (convolución + max sobre toda la imagen, o estadísticas por
   componente) sale plano **por construcción**: medido, el detector de curvas del boceto da 1,00 en las 9 posiciones.
   Lo que rompe la invariancia es lo que lee **en celdas fijas** (max-pool por celda, un peso por celda, stride).
7. **La forma esperada se escribe en el criterio ANTES de medir** (R13), como el resto.

### Cómo se sostiene

- `REGLAS.md` lleva una sección **`## Desplazamientos`** (está en `REGLAS.ejemplo.md`): qué se reconoce y a qué nivel,
  rango y direcciones, qué curva, cómo se trata el recorte y dónde queda la figura. Si el experimento no reconoce nada,
  se escribe eso y el porqué, y la sección queda cerrada.
- **`comprobar.py` la exige a todo experimento con `creado` ≥ 2026-10-09** y falla si falta. Los anteriores no la tienen
  y no se les exige: están cerrados, y un aviso que sale siempre se deja de leer. El que se reabra para reconocer algo
  nuevo, se la añade. Test: `python3 -m unittest tests/test_desplazamientos.py`.
- **La primera aplicación, para copiar la forma**: `docs/bocetos/2026-10-09-curvas-gabor/desplazamientos.py`
  (`mover()` sin `np.roll`, las dos direcciones, % que cambia, % recortado, objeto y feature en una figura).

## ⚠⚠ Regla 0 — cada experimento es INDEPENDIENTE de los demás

**Orden del dueño (2026-09-07), y es la regla que manda sobre todas las de este fichero:**

> «aunque es un mismo repo, cada experimento es totalmente independiente de los otros. No
> debemos tratar de que uno se comporte como alguno de los otros.»

**Y no es una preferencia de estilo: es el fallo observado.** Lo que el dueño describe
haber visto es que *«Claude empieza a meter restricciones de otros experimentos como si se
tratara de un único proyecto»*. Ese fallo no da error por ningún lado — da un experimento
que contesta una pregunta que nadie hizo, con condiciones que nadie pidió, y **parece
correcto**: es el fallo silencioso otra vez, sólo que sobre el diseño en vez de sobre el
dinero.

### Qué NO se hereda entre experimentos: nada

Nada de esta lista viaja de un experimento a otro. Cada uno la decide entera, desde cero:

| | |
|---|---|
| **el dataset** y sus condiciones | qué se sortea, cuántas ventanas, qué se descarta, qué margen, qué reducción |
| **la etiqueta** | cuántos números, qué significan, qué es positivo |
| **la red** | arquitectura, número de parámetros, qué se aprende y qué se congela |
| **los hiperparámetros** | épocas, `lr`, lote, semillas, los pesos de la pérdida |
| **la métrica y el umbral** | qué se mide, qué se llama «aprendió», qué se llama «ganar» |
| **el criterio** | cuántos brazos, si declara ganador o reporta todos los que pasan |
| **los scripts y su interfaz** | qué ficheros hay, cómo se llaman sus banderas, qué imprimen |
| **la forma de la carpeta** | qué subdirectorios tiene y cómo se llaman |

**Lo único que un experimento no decide** son las cinco cosas de la § siguiente
(«La libertad, y su frontera exacta»), y las cinco protegen dinero o trabajo del
**sistema**, no la coherencia entre experimentos — que no es un objetivo aquí.

### La consecuencia que hay que interiorizar: una diferencia NO es un bug

Dos experimentos que hacen lo mismo de dos formas distintas **están bien los dos**. No se
unifican, no se «arreglan», y **no se toca el de al lado** para que cuadre con el que se
está escribiendo.

⚠ **Las frases que hay que reconocer como el fallo**, porque suenan a rigor y son justo lo
contrario:

- *«pero en `esq-2d` esto se hacía así»* → allí, sí. Aquí no se ha pedido.
- *«para ser consistente con el otro experimento…»* → la consistencia entre experimentos
  **no es un valor de este repo**. Lo es dentro de uno.
- *«esto rompería la comparabilidad»* → sólo si **este** experimento declaró querer
  comparar. Si no lo declaró en su `REGLAS.md`, no hay nada que romper.
- *«ya que estamos, lo alineo con el patrón del repo»* → aquí no hay patrón del repo. Hay
  una frontera (cinco cosas) y todo lo demás es libre.

**Si de verdad parece que una condición de otro experimento debería aplicarse aquí**: se
**dice** y se **escribe en las reglas de este experimento** como decisión propia, con su
motivo. Lo que no vale es importarla en silencio porque el vecino la tenía.

### Copiar un experimento es ahorrar tecleo, NUNCA heredar obligaciones

Copiar la carpeta de otro experimento para empezar es **lo normal y está bien** — es lo que
se hizo con `esq-cq`, copiado de `esq-2d` (`instrucciones/01-encargo.md`, orden literal del
dueño). Lo que se copia son las **especificaciones ya escritas**, para no volver a
escribirlas.

> «En ocasiones vamos a copiar un experimento para ahorrar tiempo y no repetir cosas ya
> especificadas, pero eso no le resta independencia al proyecto. […] No hay obligación de
> hacer las cosas tal como las hizo el proyecto copiado (sólo queremos ahorrar tiempo en las
> especificaciones).» — el dueño, 2026-09-07

Así que al copiar:

1. **Se cambia el `id`** y se reescribe el `experimento.json` entero. Un `id` duplicado lo
   caza `comprobar.py`, pero un título heredado no.
2. **Se relee `REGLAS.md` línea por línea y se cambia lo que no aplique.** Es el paso que se
   salta: un fichero copiado que nadie releyó es una especificación **de otro experimento**
   con el nombre de éste.
3. **Cambiar cualquier cosa del original es gratis y no hay que justificarlo.** Al revés:
   **conservar** algo del original porque estaba ahí, sin decidirlo, es lo que sí es un
   fallo.
4. **El experimento copiado no se toca.** Ni para arreglarle nada, ni para «alinearlos». Si
   está cerrado, sus números ya se reportaron con el código que tenía.

⚠ **Qué significa `hereda_de` en `experimento.json`, y qué NO significa.** Es **linaje**:
*«de aquí salió la pregunta, y contra esto se comparan los números»*. No es herencia de
obligaciones ni de restricciones. Un experimento con `hereda_de` puede cambiar todo lo que
quiera de su padre — y si cambia algo que afecte a la comparación, **lo dice en su
`REGLAS.md`** y ya está: dejar de ser comparable es una decisión legítima, no un error.

### Cómo se sostiene, y no es sólo esta página

- **Ningún experimento importa código de otro.** Ya lo comprueba `comprobar.py` (la
  comprobación de R16 recorre **todo** el repo, así que un fichero de un experimento que
  nombre la carpeta de otro falla igual que uno de fuera). El código común se copia, y el
  precio —dos ficheros que pueden divergir— es **menor** que el de un `comun/` inventado con
  el segundo caso.
- **`comun/` no existe a propósito**, y crearlo antes de tiempo es la forma estructural de
  romper esta regla: en cuanto dos experimentos comparten una pieza, cambiarla por uno
  cambia al otro. Ver § «Los huecos conocidos».
- **Cada experimento declara sus reglas por escrito** en su `REGLAS.md` (§ «`REGLAS.md`: el
  set de reglas PROPIO de cada experimento»), que es donde se contesta *«¿qué aplica aquí?»*
  sin tener que mirar al vecino.

## La libertad, y su frontera exacta

Esto es lo que el dueño pidió que quedara anotado: **la estructura de este repo se puede
sugerir y re-ordenar según convenga**, y cada experimento puede tener sus propias reglas y
objetivos. Pero hay cinco cosas que un experimento **no** puede decidir por su cuenta, y
las cinco protegen dinero o trabajo, no gusto.

| Un experimento SÍ decide | Un experimento NO puede cambiar | Por qué |
|---|---|---|
| su arquitectura, su bucle, su métrica, su dataset, su umbral, qué llama «ganar», cuántas semillas, qué dependencias usa, si reusa `fv` o no, y **la forma interna de su carpeta** | que **el freno lo vea** si tarda o alquila | el veredicto «¿se puede apagar este server?» se lee desde el móvil y decide una factura |
| en qué lenguaje lo escribe, si publica figuras, si guarda snapshot de código | que exista un **criterio escrito antes de mirar** (R13) | escrito después no se distingue de una racionalización, y «no hubo señal» deja de ser un resultado |
| dónde deja sus artefactos **dentro** de su carpeta | que el veredicto que mueva `ESTADO.md` acabe en el **repo central** (R7) | si no, `estudios-redes-neuronales` deja de contestar qué se pagó ya |
| cuántos ficheros tiene, cómo se llaman y qué imprimen | que **sus reglas estén escritas** en su `REGLAS.md` (§ más abajo) | sin ellas, el siguiente que llegue rellena las condiciones con las del experimento de al lado — que es el fallo que la Regla 0 evita |
| — | que los secretos no se commiteen · que todo vaya a `main` · que el dato de entrada **no se copie aquí** (se **publica** en el repo de datos) | reglas del sistema, no del experimento |

**Lo que NO está en esa lista es libre.** No hace falta pedir permiso para inventarse una
estructura de carpeta distinta, ni para no usar `expcnn`, ni para escribir el experimento
entero en un solo fichero.

⚠ **Y ninguna de las cinco es «parecerse a los demás experimentos».** La frontera la ponen el
freno, el criterio, el repo central, los secretos y las reglas escritas — el sistema. La
coherencia entre experimentos **no está en la lista y no es un objetivo** (Regla 0).

## Se puede RE-ORDENAR — y por eso el nombre de la carpeta NO es la identidad

El dueño pidió explícitamente poder re-ordenar los ficheros en el futuro. Eso sólo sigue
siendo barato si se respeta **una** convención:

> **Nada de fuera de una carpeta de experimento la nombra por su ruta: se la pide al registro
> por su `id`. Y ningún experimento importa de otro experimento.**

Y una regla de forma que la sostiene: **la carpeta empieza siempre por su fecha**
(`<AAAA-MM-DD>-<nombre>`). No es estética. La comprobación de arriba busca el nombre de la
carpeta dentro de los ficheros, así que una carpeta llamada `k3s1` casaría con cualquier
docstring o tabla que hable del brazo `k3s1` — el aviso que sale siempre y se deja de leer.
Con la fecha delante el nombre es inequívoco, y renombrar sigue siendo gratis: lo que se fija
es la **forma** del nombre, no el nombre.

Con eso, re-ordenar es `git mv` y nada más: renombrar, agrupar por tema (`planas/`,
`cabezas/`), meterlas por trimestre o cambiar la forma interna de una son **gratis**.

La identidad la da un dato comprobable —el `id` de `experimento.json`— y no el nombre de la
carpeta (R16). **No es una precaución teórica**: en `foveal-vision/experimentos/`,
`comun/serie.py:19-26` y `comun/preproceso.py:69-71` cablean nombres de carpeta, así que allí
renombrar una carpeta rompe el evaluador.

**Cómo se comprueba** (las dos mitades, medidas el 2026-09-06 al escribir esto):

```bash
python3 comprobar.py                 # lista, valida, y falla si algo de fuera nombra una carpeta
git mv 2026-09-06-algo 2026-09-06-otro-nombre && python3 comprobar.py   # tiene que salir 0
```

## La forma de un experimento

Es una **sugerencia con motivo**, no un molde: lo único obligatorio son
`experimento.json` y `REGLAS.md`.

```
<fecha>-<nombre>/
  experimento.json     OBLIGATORIO. La identidad y las obligaciones que hereda
  REGLAS.md            OBLIGATORIO. Las reglas de ESTE experimento: entradas, salidas,
                       procesos, scripts, y qué NO hereda de nadie (§ siguiente)
  README.md            qué se preguntó, qué salió, y cómo repetirlo
  instrucciones/       01-encargo.md · 02-criterio.md  ← el criterio, ANTES de mirar (R13)
  nn/
    entrenar_local.py  ⚠ EL NOMBRE ES UN CONTRATO si entrena largo (ver abajo)
    modelo.py          la red AUTOCONTENIDA: no importa nada de nadie
    pesos/<brazo>/     best.pt · last.pt · config.json · metrics.jsonl · summary.json
  resultados/          métricas, figuras, kernels
  codigo/              SNAPSHOT congelado del código que lo produjo, si hizo falta
```

Las cuatro reglas se heredan de `foveal-vision/experimentos/README.md`, que ya las tiene
escritas con su porqué, y siguen valiendo aquí:

1. **`nn/modelo.py` no importa nada del repo.** Es lo único que garantiza poder cargar los
   pesos dentro de un año.
2. **`codigo/` es un snapshot, no una copia de trabajo.** No se edita nunca: dos copias vivas
   del mismo código divergen y nadie se entera.
3. **Los pesos SÍ entran en git aquí, y es una excepción con su tope.** La regla general del
   proyecto es que no se guardan (862 runs × 2,7 MB ≈ 2,3 GB, medido 2026-08-31); estas redes
   son pequeñas —de 304 a 19.656 parámetros, 12 checkpoints = 1,1 MB— así que la razón de la
   regla no aplica. **El tope es ≈5 MB por experimento**; por encima, se pregunta.
   ⚠ Se comprueba que **cargan**, no que están: un `.pt` que no abre ocupa sitio y parece un
   respaldo.
   ⚠ Esto **no** cambia la regla para la red de producción, que sigue pidiendo que el dueño lo
   ordene y va aprobada una a una en `inferencia.json`.
4. **Lo que no se puede regenerar se guarda; lo que sí, se enlaza.**

## `REGLAS.md`: el set de reglas PROPIO de cada experimento

**Orden del dueño (2026-09-07):**

> «Cada experimento debe tener un set de reglas propio, donde se especifican las entradas,
> las salidas, los procesos, los scripts, etc, de modo que cada experimento use su propio
> código.»

Es **obligatorio** y lo comprueba `comprobar.py`. Un experimento sin sus reglas escritas es
un experimento cuyas condiciones sólo existen en la cabeza de quien lo montó — y en cuanto
esa sesión termina, el siguiente que llegue las rellena con las del vecino, que es
exactamente el fallo de la Regla 0.

### Las cinco secciones, y por qué esas

`REGLAS.ejemplo.md` (en la raíz, commiteada) es la plantilla: se copia y se rellena.

| sección | qué contesta | por qué es obligatoria |
|---|---|---|
| **Entradas** | qué dataset publicado consume, con qué condiciones y qué etiqueta | es lo primero que se hereda por error del experimento copiado |
| **Salidas** | qué produce y dónde queda: pesos, métricas, figuras, tablas | «¿dónde está el resultado?» no puede ser una pregunta que haya que investigar |
| **Procesos** | qué pasos hay, en qué orden, y qué se decide en cada uno | sin esto, repetir el experimento es leer los scripts |
| **Scripts** | qué fichero hace qué, y **su interfaz exacta** | los nombres y banderas son de este experimento, no del repo |
| **Qué NO hereda** | de qué experimento se copió y **qué se cambió a propósito** | es la única sección que ataca la Regla 0 de frente |

⚠ **La quinta es la que no se puede omitir aunque el experimento no venga de ninguno**: si
no se copió de nadie, se escribe *«no se copió de ningún experimento»*. Un hueco se lee como
«todavía no lo he pensado» y una ausencia no se distingue de un olvido.

### Por qué un fichero aparte, y no dentro de lo que ya había

Los tres documentos de un experimento contestan tres preguntas distintas y en tres momentos
distintos (R8: estado e historial son documentos distintos):

- **`experimento.json`** — la identidad **legible por máquina**: `id`, `estado`, `gasta`,
  `dataset`, `entrada`. Lo lee `comprobar.py` y el índice del README. No es prosa.
- **`REGLAS.md`** — el **contrato de trabajo**, y es lo que hay que leer **antes de tocar
  nada**. Vale para todo el experimento, de la primera línea a la última.
- **`instrucciones/`** — el **encargo** y el **criterio escrito antes de mirar** (R13) de
  una corrida concreta. Un experimento puede tener varias; las reglas siguen siendo unas.
- **`README.md`** — **qué salió**. Se escribe después, y por eso no puede ser el sitio donde
  vivan las condiciones: lo que se escribe después de mirar no se distingue de una
  racionalización.

⚠ **Y las reglas se ACTUALIZAN cuando el experimento cambia**, en el mismo commit. Unas
reglas que describen lo que el experimento era hace tres días son peores que no tenerlas: se
leen como vigentes.

## ⚠ El contrato de nombre con el freno: `entrenar_local.py`

`telegram-coordinator/scripts/cerrable.mjs:137` **ya casa `entrenar_local.py`** en su lista
declarada `TRABAJOS` (comprobado el 2026-09-06). Ese es el mecanismo por el que un
entrenamiento de este repo aparece en el veredicto que el dueño lee desde el móvil antes de
destruir la máquina.

**Con otro nombre, el freno dice «nada corriendo» con 37 épocas vivas.** Por eso el script que
entrena se llama así, y por eso `comprobar.py` se niega si un experimento declara
`gasta: "entrena-local"` y su entrada se llama de otra forma.

`experimento.json` declara qué obligaciones hereda:

| `gasta` | Qué significa | Qué obliga |
|---|---|---|
| `"no"` | no tarda ni cuesta | nada |
| `"entrena-local"` | entrena en esta máquina, tarda | la entrada se llama `entrenar_local.py` |
| `"alquila"` | alquila máquinas: **cuesta dinero** | además, ejecutor de Telegram en el **mismo commit**, `desacoplar-persistente.sh`, y `notify.mjs … \|\| true` |

## Dónde va cada artefacto

| Qué | Dónde | Por qué |
|---|---|---|
| pesos, métricas, figuras, kernels | **aquí**, en la carpeta del experimento | vive donde su productor (R7) |
| el dataset de entrada | **se PUBLICA y se lee de `foveal-vision-data/experimentos-cnn/`; NUNCA se copia aquí** | este repo es **público** y el de datos es **privado** (medido el 2026-09-06: anónimo, 200 contra 404). Git no olvida. Detalle en la § siguiente |
| el reporte de un estudio | **`estudios-redes-neuronales`**, con su fila en `reportes/README.md` | el central es quien contesta «qué se pagó ya» |
| qué se cree hoy de un parámetro | `ESTADO.md` del central (se reescribe) | estado e historial son documentos distintos (R8) |

**Cuándo un experimento merece reporte en el central — dos preguntas mecánicas**, para no
discutirlo cada vez:

1. **¿Cambia lo que `ESTADO.md` dice de algún parámetro?** Sí → reporte + fila + edición de
   `ESTADO.md`, **aunque haya costado 0 $** (hay precedente: el reporte #12 es un estudio con
   0 máquinas y 0,00 $). No → se queda en su carpeta con su `README.md`.
2. **La unidad del reporte es la PREGUNTA, no la carpeta.** Seis carpetas que contestan una
   pregunta son **un** reporte que enlaza las seis.

⚠ Esta frontera es una **propuesta pendiente de confirmar**: `reportes/README.md` del central
dice hoy *«un reporte por cada barrido, estudio o medición que se termine, venga de donde
venga»*, y la práctica ya lo contradice (de los 11 experimentos de `foveal-vision`, sólo 2
tienen reporte). Si se confirma, la enmienda se escribe **allí**, no aquí en silencio.

## Los datasets viven en el repo de DATOS, y se reusan por NOMBRE

⚠ **Desde el 2026-10-01 el `origin` de ese repo es el ALMACÉN** —el volumen del mini—, no
GitHub: `git push` desde `~/src/foveal-vision-data` va al mini, y allí **nadie borra**. Las
rutas de aquí no cambian —el repo se sigue llamando `foveal-vision-data` y está al lado—;
cambia a dónde empuja, y que **lo temporal también se guarda** (`temporal/<máquina>/<fecha>/`
del repo de datos, en vez de `/tmp`). La regla entera, en `telegram-coordinator/CLAUDE.md`
§ «EL ALMACÉN».

**Orden del dueño, dos veces y con dos motivos distintos:**

> «Guarda los datasets en el repo de data, de modo que sean siempre los mismos, por
> consistencia.» — 2026-09-07

> «los datasets se guardan siempre en el repo de data, para que todos los experimentos que
> usen un dataset dado puedan tomarlo de ahí en vez de regenerarlos. Los datasets pueden ser
> distintos en cada experimento.» — 2026-09-07

Las dos mitades importan y dicen cosas distintas: **el mismo dataset se comparte** (no se
regenera por experimento), y **cada experimento puede tener el suyo** (compartir el sitio no
es compartir el dato).

```
foveal-vision-data/experimentos-cnn/<nombre-del-dataset>/
    train.npz · val.npz · muestra.npz · muestras-congeladas.npz
    manifiesto.json     la huella SHA-256 de cada partición, y con qué se generó
    README.md           qué etiqueta, cuántas ventanas, quién lo usa
```

### Las cinco reglas, con su porqué

1. **Un dataset se PUBLICA una vez y se LEE muchas.** Un experimento nuevo que necesite un
   dataset que ya existe **no lo regenera**: lo pide por su nombre. Regenerar da un dato que
   es el mismo *mientras nada cambie*, y «nada cambia» no se puede comprobar hacia el
   futuro; **publicado, es el mismo porque es el mismo fichero**.
2. **La puerta es `expcnn.exigir_dataset("<nombre>")`**, y se llama en la primera línea. Se
   **niega antes de empezar** si no está publicado, en vez de re-derivarlo al vuelo (R2): un
   dataset re-derivado a mitad daría números incomparables **sin fallar por ningún lado**.
3. **Sólo se añaden: dato nuevo = nombre nuevo**, con su `r<fecha>` de render. Un dataset
   publicado **no se reescribe nunca** y `--publicar` se niega a pisar uno existente — si
   pudiera, un `--publicar` distraído cambiaría el dato bajo los pies de todo lo ya medido.
4. **El nombre del dataset se DECLARA** en `experimento.json` (`"dataset"`), y
   `comprobar.py` comprueba que esté publicado de verdad. No vale que sólo lo sepa un
   `DATASET = ...` dentro de un script: entonces *«¿sobre qué se midió esto?»* se contesta
   leyendo código.
5. **Nunca se copia a este repo.** Éste es **público** y el de datos es **privado** (medido
   el 2026-09-06 contra la API de GitHub: anónimo, 200 contra 404), y git no olvida. Por eso
   el `.gitignore` de aquí lleva `*.npz`.

⚠ **Compartir dataset NO es compartir condiciones** — y ésta es la Regla 0 aplicada al dato,
que es por donde más fácil se cuela. Dos experimentos sobre el mismo `.npz` pueden leer
columnas distintas de la etiqueta, quedarse con subconjuntos distintos, normalizar distinto y
medir distinto. Lo único que garantiza el dataset compartido es que **las ventanas son las
mismas**; todo lo demás lo decide cada experimento y lo escribe en su `REGLAS.md`.

⚠ **Y un dataset parecido no es el mismo dataset.** Medido el 2026-09-07: dos datasets de
este repo salen de las **mismas imágenes** —misma receta, misma semilla, misma reducción— y
aun así **las ventanas son otras**, porque uno sortea 4 esquinas `tl` por imagen y el otro
2 `tl` + 2 `br`. Los números de un experimento sólo son comparables contra el dataset que
usó, y por eso el nombre lleva qué etiqueta trae y no sólo la fecha.

⚠ **El dataset sobrevive al experimento, y es la prueba de que este sitio es el correcto.**
`lim-ab` se borró el 2026-09-07 porque el diseño cambió; su dataset `limites300-…` sigue
publicado y utilizable por el que venga. Si el dato hubiera vivido en la carpeta del
experimento, se habría ido con él.

### La verificación se conserva como PRUEBA, no como sustituto

`python nn/datos.py --rederivar` sigue existiendo y sigue teniendo que pasar: contesta *«¿la
receta y la semilla lo vuelven a dar?»*. Pero es una **prueba de que la receta es honesta**,
no una alternativa a publicar. Lo mismo `--comprobar`, que casa las huellas del publicado
contra el manifiesto.

⚠ **Qué costaba no publicarlo, medido el 2026-09-07:** las 10 muestras congeladas de `esq-k`
se declaraban commiteadas y **no lo estaban** — el `*.npz` del `.gitignore` de este repo se
las llevaba —, así que en un clon limpio su figura de verificación no se podía regenerar sin
volver a rendir 300 imágenes.

## Cómo se conecta con `foveal-vision` (y qué pasa si no está)

Un experimento **autónomo no necesita nada**: `comprobar.py` y `expcnn` son stdlib pura, así
que un clon limpio corre sin instalar nada.

Hay **tres** puertas a los repos hermanos, y son las tres únicas (`expcnn/entorno.py`).
Se llaman **en la primera línea** del experimento que las necesite:

| puerta | para qué | cómo resuelve |
|---|---|---|
| `exigir_fv()` | reusar código de `foveal-vision` | `EXPCNN_FV` > el hermano `../foveal-vision` > se **niega** |
| `exigir_dataset("<nombre>")` | **un dataset publicado** (lo normal) | sobre la raíz de abajo; sin `manifiesto.json`, se **niega** |
| `exigir_datos()` | la raíz del repo de datos, para algo que no sea un dataset | `EXPCNN_DATOS` > `FV_DATA_ROOT` > el hermano `../foveal-vision-data` > se **niega** |
| `exigir_generador()` | rendir párrafos con `image-text-sample-generator` | `EXPCNN_GENERADOR` > el hermano > se **niega** |

⚠ **Para leer el dato de entrada, la puerta es `exigir_dataset(nombre)`**, no
`exigir_datos()` + componer la ruta a mano: si cada experimento arma su propia ruta, *«el
mismo dataset»* deja de estar garantizado por nada (R4). El subdirectorio lo decide
`expcnn.SUBDIR_DATASETS` en un solo sitio.

⚠ **La puerta al dato respeta `FV_DATA_ROOT`** —la variable con la que `foveal-vision`
resuelve ESE MISMO repo (`src/fv/settings.py:27`)— a propósito: dos mandos para un solo hecho
pueden discrepar sin que nada falle (R15).

⚠ **Y NO cae al repo de código como hace `fv.settings.data_root()`.** Allí ese respaldo es
correcto: quien no ha clonado el repo de datos sigue funcionando. Aquí sería un directorio sin
`windows.npz`, o sea fallar a mitad en vez de negarse al empezar (R2).

Se niega **antes de empezar**, no a la época 30 (R2). Y hay una sola indirección a propósito:
en `foveal-vision/experimentos/` hay **72 `sys.path.insert`** repartidos por 41 ficheros
(medido el 2026-09-06), cada uno deduciendo del disco dónde está el repo hermano — que es el
antipatrón de R4, y que al mudarse aquí dejaría de ser frágil para ser **falso**.

## Cómo se corre

```bash
python3 comprobar.py                  # qué hay, qué puede correr, qué está mal puesto
uv venv .venv && uv pip install -e .  # sólo si un experimento lo necesita
```

**Un entorno por experimento sale casi gratis con `uv`, así que no lo pienses dos veces:**
medido el 2026-09-06 en esta máquina, dos venvs con `numpy` ocupan 57 MB cada uno por
separado pero **58 MB los dos juntos** (comparten por enlace duro), y crear el segundo tardó
**0 s**. Lo caro es torch: el `.venv` de `foveal-vision` pesa **1,1 GB**, y ése conviene
reusarlo (`EXPCNN_FV`) en vez de copiarlo.

⚠ **Esta máquina es pequeña: 2 vCPU y 3,8 GB de RAM** (medido 2026-09-06). Aquí caben
tanteos; lo que entrene de verdad alquila máquina — y entonces entra todo lo de la fila
`alquila` de arriba.

## Desde Telegram

`/use exp` — qué experimentos hay, en qué estado y qué está mal puesto. Llega solo, sin
copiar nada ni reiniciar el bot: el coordinador escanea `~/src/*/telegram`.

⚠ Un ejecutor nuevo de este repo se llama **`exp-<algo>`**, para no colisionar con los de
`foveal-vision` (que ya tiene `estudio`, `entrenar`, `vigilante`…). Las colisiones se avisan
pero las gana la primera fuente, y eso no es un criterio que quieras (R15).

## Los huecos conocidos, sin adornar

- **No hay CI, y por tanto `comprobar.py` no lo corre nadie solo** (R17). Ninguno de los siete
  repos lo tiene. Hoy se corre a mano o por `/use exp`.
- **Este repo no tiene `.claude/`**, así que una sesión abierta con el cwd aquí **no dispara**
  el triage, ni el registro de sesión, ni el archivado de conversación — esos hooks viven
  sólo en `telegram-coordinator`. No es un freno que se pueda dar por supuesto.
- **`comun/` no existe, y es a propósito.** Se crea cuando haya **dos** experimentos que
  tengan que medir con la misma regla, nunca con el primero: un `comun/` creado el primer día
  es un `utils/` con mejor nombre (antipatrón de R1). En `foveal-vision` se ganó su sitio con
  seis gemelos.
- **⏳ ABIERTO: los 11 experimentos de `foveal-vision/experimentos/` siguen allí.** Mientras
  convivan los dos sitios, *«¿dónde está un experimento?»* tiene dos respuestas, que es la
  definición de migración a medias (R19). Está sin decidir a propósito: **el dueño no ha
  pedido moverlos**. Si se mueven, lo caro no son los ficheros sino los 72 `sys.path.insert`
  y los nombres de carpeta cableados en `comun/`.

## Las reglas de escritura y de diseño se heredan enteras

Las cinco reglas de redacción (todo número con su procedencia, «sobrevive» con complemento,
caducidad de los cerrojos, terminado = invocable desde Telegram, el preflight crece con cada
fallo) y las 19 reglas de diseño viven en `telegram-coordinator`
(`CLAUDE.md` y `docs/reglas-de-diseno.md`) y **valen aquí igual**. Se entra por la tabla de
disparadores del § 0, no se lee entero.

Y las dos operativas que más muerden:

- **Lo que no está empujado, no existe.** Estos servidores se rehacen sin aviso: cambio o
  documentación terminada → commit → push, el mismo día. Todo a `main`.
- **Cada respuesta al usuario termina con la línea del freno**:
  `node ~/src/telegram-coordinator/scripts/cerrable.mjs --breve`.

Comentarios y mensajes al usuario: en **español**, como el resto del proyecto.
