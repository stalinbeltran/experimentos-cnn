# Reglas de `feat-agr`

**ESPECIFICACIÓN del 2026-10-05, sin implementar.** No hay código, ni grupos, ni un solo número propio. El criterio,
escrito **antes de calcular ningún grupo**, está en `instrucciones/02-criterio.md`. Al implementar se reescribe aquí lo
que cambie, en el mismo commit.

**Qué pregunta:** si cada dígito se describe por las features que lo componen —qué detector se enciende y en qué zona—
y se agrupan los dígitos por esa descripción **sin leer la etiqueta**, ¿salen grupos de formas semejantes —variantes de
un mismo dígito (el 1 que es sólo una recta vertical y el 1 con un trazo inclinado arriba) y formas que comparten
dígitos distintos— y cuánto coinciden esos grupos con las etiquetas de NIST?

**La idea en una línea:** en todo lo de hoy la etiqueta organizaba (los detectores se componían *para* predecirla).
Aquí se agrupa sin ella, y la etiqueta sólo entra **después**, como lente para leer unos grupos que se formaron solos.

## Entradas

### 1. Los dígitos

- **Dataset:** `uci-optdigits-orig-32px-r20261005` (publicado por `feat-ind32`; se lee con `exigir_dataset`, no se
  regenera). Los **5620** dígitos de NIST en 32×32 binario, de 43 escritores: 3823 de 30 (`tra`, `cv`, `wdep`) y 1797 de
  otros 13 (`windep`).
- **Qué se lee:** `imagenes` y `origen`; `etiquetas` **sólo en la lectura** (§5), nunca al agrupar. `particion` (el
  180/1617) **no se usa**: agrupar no entrena nada contra la etiqueta, así que no hay train/val que proteger.
- ⚠ **No hay identificador de escritor por dígito** (`optdigits-orig.names` no lo trae), y el orden del fichero no lo
  delata (mirado el 2026-10-05: los `windep` empiezan con tres series 0–9 y luego van sin patrón). Así que un grupo **no
  se puede atribuir a un escritor**; lo más fino que se puede decir es **en qué población aparece** (los 30 o los 13).
- **Condiciones que trae:** NIST ya **centra y normaliza** cada dígito al marco de 32×32 (*«Inputs are centered and
  normalized as 32x32 bitmaps»*, `optdigits-orig.names`). Por eso una zona del marco ya es una zona *del dígito*, y no
  hace falta la posición relativa al centro (E3 de `feat-pos`).
- **Al cargar:** el 32×32 en 0/1 para el banco de 32; reducido por bloques 4×4 (0..16, ÷16) para el banco de 8 y para el
  control de píxeles. Es la reducción de NIST: para los 1797 `windep` da exactamente `uci-optdigits-8px-r20261002`
  (comprobado por `feat-ind32`, `nn/datos.py --comprobar`).

### 2. Las features: los detectores de hoy, congelados y COPIADOS

- **Banco principal, de 32×32:** los 13 detectores de `feat-ind32` (C3), entrada 32×32 → mapa 8×8, entrenados **sólo con
  dibujos sintéticos**. Sobre su val sintético: rectas F1 0,965–0,971, lazo 0,962, arcos 0,862–0,897, esquinas
  0,879–0,890 (README de `feat-ind32`). **(decido)** que sea el principal porque es el que mejor **significa** su
  feature —precisión 0,88–0,98, contra 0,61–0,67 de los de 8×8 en arcos y esquinas— y aquí lo que importa es que «arco-E
  encendido» quiera decir arco, no que el compositor acierte el dígito. Y porque confunde menos por grosor: 1↔8, 5 veces
  contra 22 (compositor posicional, semilla 1).
- **Banco de contraste, de 8×8:** los 13 `fino` de `feat-ind` (corrida 2), sobre la reducción 4×4. Sirve para **una**
  pregunta: ¿los grupos son de los dígitos, o del banco?
- **Cómo se usan** (Regla 0: ningún experimento importa de otro): se **copian** los `best.pt` a `nn/detectores-32px/` y
  `nn/detectores-8px/` (13 × 173 KB + 13 × 61 KB ≈ **3,0 MB**, medido; bajo el tope de 5 MB del repo) y las dos clases
  `Detector`, autocontenidas, con un `nn/detectores.json` que dice de qué `id` y de qué commit sale cada uno, su huella y
  su umbral `u_f`. Al cargar se recalcula la huella y **se niega** si no casa. **(decido)** copiar en vez de leerlos del
  otro experimento por su `id`: `feat-ind32` sigue abierto y le queda «más semillas de detectores»; si sus pesos cambian,
  estos grupos no deben cambiar con ellos.
- **Que las copias son las de hoy, comprobado:** la firma por clase de los 1797 `windep`, recalculada aquí, tiene que ser
  **idéntica** a la que guardaron `feat-ind32` y `feat-ind` (`resultados/firma-por-clase.json` de cada uno, pedidos al
  registro por su `id`). Si no, se niega.
- ⚠ **Lo que NO entra, y por qué:** los detectores **ajustados de C** (`ajuste13`, el mejor modelo de hoy) se ajustaron
  **con las etiquetas** de los dígitos: agrupar con ellos metería la etiqueta por la puerta de atrás —la «etiqueta débil»
  que `feat-ind` (`OBSERVACIONES.md` §2.1) señaló como lo que invalida la premisa—. Tampoco los bancos `grueso`, `dig` y
  `cae*`: dos bancos bastan para la pregunta del banco, y `dig`/`cae` no tienen nombre de feature, así que la composición
  de un grupo no se podría leer.

### 3. La descripción de un dígito: qué feature, y en qué zona

De los 13 mapas σ (8×8) de cada dígito salen tres resoluciones del «dónde». Son el eje del experimento:

| | qué es | tamaño | qué conserva |
|---|---|---:|---|
| **P** · presencia | el máximo de cada mapa | 13 | *qué* hay. Ciega a la posición: el 6 y el 9 deberían caer juntos |
| **Z** · zonas — **la principal** | el máximo de cada mapa en 9 zonas solapadas: ventanas de 4×4 celdas con paso 2 (filas y columnas 0–3, 2–5, 4–7) → NW, N, NE, W, C, E, SW, S, SE | 13 × 9 = 117 | *qué y en qué zona*: «recta-V en el centro + recta-S arriba a la izquierda» |
| **M** · mapa | los 13 mapas enteros | 832 | *qué y en qué celda exacta* |

**(decido)** Z es la principal, por tres motivos:

1. Es la resolución de la descripción humana: *«una línea inclinada **en la parte superior**»*.
2. Un máximo sobre una ventana tolera ±1 celda, y hoy se midió que hace falta: desplazar el dígito 1 celda dejó al
   compositor posicional en **0,556** (8×8) y **0,608** (32×32) de acierto, y leer tras un máximo 3×3 lo subió a
   **0,774** y **0,733** (`feat-ind` corrida 14 y `feat-ind32` C6, `resultados/curva-fino.json` y
   `resultados/generalizacion.json`). Dos dígitos iguales desplazados una celda no pueden caer en grupos distintos.
3. Cada zona tiene 16×16 px del marco y se solapa a la mitad con sus vecinas: una feature en una frontera no salta de
   zona por un píxel.

**El control sin features, X** (píxeles): el 8×8 contado, 64 números ÷16. **(decido)** el 8×8 y no el binario de 32×32:
el conteo 4×4 suaviza, y es la representación clásica de estos dígitos; al control se le da su mejor versión.

**Los cinco brazos que se agrupan:** **Z32** (principal), **P32** y **M32** (las otras resoluciones, mismo banco), **Z8**
(el banco de 8×8) y **X8** (píxeles).

**La composición de un grupo** —lo que se lee de él, y es una matriz, no un texto—: 13 × 9 con la fracción de sus
miembros en que cada (feature, zona) pasa el umbral `u_f` de su detector (el guardado en su checkpoint, elegido sobre el
train **sintético**: no vio ningún dígito). Lo que pase de 0,5 es «su composición». El texto («recta-V·C + recta-S·NW»)
sólo se escribe en las figuras, para leerlas.

## Salidas

- **Pesos:** las copias de los detectores (`nn/detectores-32px/`, `nn/detectores-8px/`, ≈3,0 MB) y `nn/detectores.json`.
  Se commitean. No se entrena ninguna red.
- **Los grupos:** la partición principal (Z32, K = 30) en `resultados/grupos-Z32-K30.json`: a qué grupo va cada uno de
  los 5620, centroides, medoides, composición y árbol. **Se commitea**: es el producto del experimento. Las demás
  particiones, en `resultados/grupos.npz` (no se commitea: `*.npz`; se regeneran con las mismas semillas).
- **Métricas:** `resultados/lecturas.json` — L1–L6 por brazo, K y semilla, y la evaluación del criterio.
- **Figuras** (`nn/figuras.py`):
  - `grupos-Z32-K30.png` — una fila por grupo, en el orden del árbol: tamaño, mezcla de etiquetas, el medoide grande y
    15 miembros al azar. **Es lo primero que se mira**, antes que cualquier número;
  - `composicion-Z32-K30.png` — por grupo, los 13 mapas de zona (3×3) medios;
  - `arbol-Z32-K30.png` — el dendrograma de los 30 grupos;
  - `variantes-1.png` — los 1 de cada variante, con su índice de bandera (§5, L2b);
  - `grupos-X8-K30.png` — la misma rejilla para los píxeles, para comparar a ojo;
  - `consistencia.png` — L4 por brazo y transformación.
- **Qué se commitea:** todo menos `*.npz` (las representaciones y las particiones secundarias se regeneran en minutos).
- **Reporte en el central:** **no** — 0 $ y no cambia `ESTADO.md` (la misma decisión que `feat-ind`). Si el dueño lo
  quiere, se escribe.

## Procesos

### 4. Cómo se agrupa

- **k-means euclídeo** con siembra k-means++ (la del obtenedor de `feat-ind`, corrida 6, copiada y pasada a distancia
  euclídea), hasta que ninguna asignación cambie o 300 iteraciones; un grupo vacío se re-siembra en el punto más lejano.
  **5 semillas** por (brazo, K).
- **K = 30 el principal** *(decido)*: deja sitio a unas 3 variantes por dígito, y 30 filas son una rejilla que se puede
  mirar en el móvil. Se reportan además **K = 10** —tantos grupos como etiquetas: ¿coinciden?—, 20 y 50.
- **La partición que se dibuja y se lee es la de menor inercia de las 5 semillas**: se elige **sin mirar etiquetas**.
- **Árbol:** Ward sobre los 30 centroides, pesados por tamaño → qué grupos se juntan primero. Es la lectura de «familias
  de formas», y ordena las filas de las rejillas para que lo parecido quede junto.
- **Dos ajustes:** (a) sobre los **5620**, que es lo que se dibuja y se lee; (b) sobre los **3823** de 30 escritores,
  para medir si los grupos valen para los 13 que no vio —cada uno se asigna al centroide más cercano— y para elegir a
  quién etiquetar (L6).
- ⚠ **La etiqueta no puede entrar por accidente, y eso lo garantiza el código, no el cuidado:** el cargador de dígitos
  **no devuelve** `etiquetas` salvo que se pidan explícitamente, y sólo la lectura las pide. `--comprobar` falla si el
  script que agrupa las toca.

### 5. Las lecturas — la etiqueta entra aquí, y sólo aquí

| # | qué | cómo |
|---|---|---|
| **L1** | coincidencia con las etiquetas | pureza (cada grupo → su etiqueta mayoritaria), NMI y ARI, por brazo y K. El azar se **mide**, permutando las etiquetas. Y con **K = 10** —tantos grupos como etiquetas— qué clase se parte y qué clases comparten grupo: es donde una forma pesa más que una etiqueta |
| **L2** | **variantes**: una clase partida en varios grupos | variante de *c* = grupo donde *c* es mayoría (≥ 50 %) y que tiene ≥ 10 % de los *c*. Cuenta sólo si es **estable**: reaparece en ≥ 4 de las 5 semillas (algún grupo con Jaccard ≥ 0,5). Con 30 grupos para 10 dígitos partir clases es inevitable; lo que no es inevitable es que el corte se repita |
| **L2b** | **el ejemplo del dueño, medido SIN detectores** | por cada 1, tres índices de píxeles sobre su bitmap de 32×32 (abajo). Entre dos variantes del 1, el AUC de cada índice dice **por qué** se partió |
| **L3** | **formas compartidas**: grupos mezclados | mayoría < 70 % y ≥ 20 miembros: qué pares de dígitos comparten forma, y si ahí caen los 54 dígitos que hoy falla C (`resultados/errores-c.json` de `feat-ind32`, pedido al registro por su `id`; sus índices son los de este mismo dataset). Y los 30 dígitos más lejos de su centroide —formas que no se parecen a ninguna—: cuántos son fallos de C |
| **L4** | **¿forma o apariencia?** — sin etiquetas | consistencia *c*: fracción de dígitos que, **desplazados 4 px, engrosados 2 px o adelgazados 1 px**, caen en su mismo grupo (centroide más cercano); y corregida por azar, κ = (*c* − Σ *p*²) / (1 − Σ *p*²) con *p* la fracción de dígitos de cada grupo —un brazo con un grupo gigante sería «consistente» sin agrupar nada—. Las transformaciones son las de C6 de `feat-ind32`, copiadas |
| **L5** | estabilidad | ARI entre las 5 semillas; en los 1797, entre «asignados desde los centroides de los 3823» y «agrupados aparte»; entre Z32 y Z8; y qué grupos de los 3823 casi no aparecen en los 1797 |
| **L6** | **¿para qué sirve?** | elegir a quién etiquetar: los K medoides del ajuste (b), etiquetados → el compositor lineal de hoy sobre M32 → acierto en los 1797. Contra K al azar y K al azar estratificado (5 sorteos cada uno), para K = 10, 20, 30, 50 y 100. Y la propagación: cada dígito toma la etiqueta del medoide de su grupo |

**Los tres índices de L2b**, con las filas de tinta del dígito r0…r1 (alto *h*) e *I(r)* = la columna de tinta más a la
izquierda de la fila *r*:

- **bandera** *b* = mediana de *I* en el tercio central (filas r0 + h/3 … r0 + 2h/3) − mínimo de *I* en el cuarto
  superior (r0 … r0 + h/4), en px. Un 1 recto da *b* ≈ 0; uno con bandera, *b* grande; uno inclinado como «/», *b* < 0;
- **grosor** = píxeles de tinta ÷ filas con tinta;
- **inclinación** = ángulo del eje principal de la tinta respecto de la vertical (momentos de segundo orden).

El AUC se toma sin dirección, max(AUC, 1 − AUC), para que los tres compitan en igualdad. *b* tiene un punto ciego
conocido —un 1 inclinado como «\» también adelanta la tinta de arriba a la izquierda—, y para eso está la inclinación al
lado: si separa mejor que *b*, el corte fue por inclinación.

### 6. Pasos

1. Copiar los detectores y escribir `nn/detectores.json`. `--comprobar`: huellas, firma por clase idéntica a la de
   origen, métricas contra casos hechos a mano, y que el agrupador no toca la etiqueta.
2. **Representar** los 5620, limpios y con las 3 transformaciones, en los dos bancos → `resultados/representaciones.npz`.
3. **Agrupar:** 5 brazos × K ∈ {10, 20, 30, 50} × 5 semillas sobre los 5620; Z32 además sobre los 3823, con K hasta 100.
4. **Mirar** las rejillas, antes que los números.
5. **Leer:** L1–L6 → `resultados/lecturas.json`, con la evaluación del criterio. README con lo que salió.

- **Qué se mide y con qué umbral:** `instrucciones/02-criterio.md` (H0–H6), escrito antes.
- **Cuántos brazos y cuántas semillas:** 5 brazos; 5 semillas de k-means; el compositor de L6, 3 semillas del
  optimizador.
- **Qué se llama «ganar»:** no se declara ganador entre brazos. Cada hipótesis se confirma o se refuta; el producto
  principal son **los grupos, mirados**.

### 7. Dónde corre y cuánto cuesta

**0 $, en el dev, sin alquilar nada.** Estimado, **no medido**: ≤ 10 min en total. Los mapas de los 5620 con el banco de
32 tardaron 41 s (medido el 2026-10-05 en `feat-ind32`, `nn/aplicar.py`, con pesos de prueba de la misma red), ×4 con
las transformaciones ≈ 3 min; el banco
de 8, ~1 min; los k-means, ~5 min (lo caro es M32, 832 dimensiones); L6, < 1 min.

Al implementar, `gasta` pasa a `entrena-local` con la entrada en `nn/entrenar_local.py` —el contrato con el freno: si
esto tarda, el freno tiene que verlo—. Si la primera pasada entera pasa de 10 min, se lanza como unidad
(`desacoplar-persistente.sh`) con un lanzador que imprime la orden, tiene modo seco y se niega a lanzar dos veces.

## Scripts

Previstos. Los nombres y las banderas son de este experimento, y si cambian al implementar, esta tabla cambia en el
mismo commit.

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/detector32.py`, `nn/detector8.py` | las dos redes de detector, copiadas y autocontenidas: cargar y huella | — |
| `nn/datos.py` | los 5620 por `exigir_dataset`, con sus huellas; **sin etiquetas** salvo `con_etiquetas=True` | `python nn/datos.py --comprobar` |
| `nn/representar.py` | P, Z y M de los dos bancos, y X8; limpios y transformados | `python nn/representar.py` → `resultados/representaciones.npz` |
| `nn/metricas.py` | k-means++, Ward, pureza, NMI, ARI y AUC en numpy, probados contra casos hechos a mano | `python nn/metricas.py --comprobar` |
| `nn/entrenar_local.py` | los k-means de todos los brazos (el nombre es el contrato con el freno) | `python nn/entrenar_local.py [--brazos Z32,X8] [--K 10,30]` → `resultados/grupos.npz`, `grupos-Z32-K30.json` |
| `nn/leer.py` | L1–L6 y el criterio | `python nn/leer.py` → `resultados/lecturas.json` |
| `nn/figuras.py` | las figuras de § Salidas | `python nn/figuras.py` → `resultados/*.png` |

- **Dependencias:** el `.venv` de la raíz (numpy 2.5.3, torch 2.14.1 CPU, Pillow 12.3.0, matplotlib 3.10.9; comprobado el
  2026-10-05). **No hay scipy ni scikit-learn**, y no se instalan: k-means, Ward, NMI, ARI y AUC son ~100 líneas de numpy
  con su prueba, y así no se toca el venv que comparten los demás experimentos.
- **De dónde sale el código:** autónomo, con piezas copiadas (abajo). Ningún import de otro experimento.

## Qué NO hereda

- **Se copió de:** ningún experimento como carpeta. Se copian **piezas**: las redes y los pesos de los detectores (de
  `feat-ind32` y `feat-ind`), las transformaciones y el compositor lineal (de `feat-ind32`), y el k-means++ (del
  obtenedor de `feat-ind`, pasado a distancia euclídea).
- **Qué se cambió a propósito:** la etiqueta deja de ser el objetivo —no se entrena nada contra ella y sólo se lee al
  final—; no hay reparto train/val para agrupar; la posición se lee en 9 zonas en vez de 64 celdas; y el resultado
  principal es una **partición**, no un acierto.
- **Qué se conservó, y por qué:** los detectores exactos (con su huella) y los 5620 en su orden, para que los grupos se
  lean contra la firma por clase y los errores de C de hoy; las transformaciones de C6, para que «consistencia» mida lo
  mismo que allí «robustez»; el compositor lineal en L6, para que «acierto con K etiquetas» use el mismo lector que las
  curvas de hoy.
- **Contra qué se compara:** consigo mismo — features contra píxeles, un banco contra otro, tres resoluciones del dónde,
  elegir por grupos contra elegir al azar. Los aciertos de hoy (curvas, N(ε), G) son **contexto**, no brazo: allí se
  sorteaba de datasets balanceados de los 43 escritores, y aquí se evalúa en los 13 de `windep`.
- **Restricciones de otros experimentos que NO aplican aquí:** el reparto 180/1617 y el test fijo de 717 (aquí nada se
  entrena contra la etiqueta al agrupar); los umbrales de «aprendió» de los detectores (no se entrena ninguno); Vast y
  *«todo lo que es paralelo debe correrse en paralelo»*, que era para lo que se alquilaba (aquí no se alquila); las 3
  semillas del compositor como semillas del experimento (aquí mandan las 5 de k-means).

### Qué se usa de lo de hoy, y qué no

| de hoy | ¿aquí? | por qué |
|---|---|---|
| los 13 detectores de 32×32 (`feat-ind32`, C3) | ✅ banco principal | entrenados sin dígitos; los que mejor significan su feature |
| los 13 `fino` de 8×8 (`feat-ind`, corrida 2) | ✅ contraste | ¿los grupos son del dígito o del banco? |
| los mapas σ 8×8 y la firma por clase (`nn/aplicar.py`) | ✅ | son la descripción; la firma, la prueba de que las copias son las de hoy |
| el máximo 3×3 contra el desplazamiento (corridas 13–14, C6) | ✅ convertido en las zonas | la lección medida de hoy |
| las 3 transformaciones de C6 | ✅ L4 | «misma forma» se pone a prueba igual que allí |
| los 54 dígitos que falla C (`errores-c.json`) | ✅ L3 | ¿los grupos mezclados explican los errores? |
| 3823 (30 escritores) / 1797 (13) | ✅ L5, L6 | la única separación de escritores que existe |
| el compositor lineal | ✅ L6 | mismo lector que las curvas |
| N(ε), las curvas de aprendizaje | ✅ como pregunta de L6 | ¿agrupar ahorra etiquetas? |
| los detectores ajustados de C | ❌ | vieron las etiquetas |
| las CNN (LeNet-5, la del repo, A, B, igualada) | ❌ | aquí no se entrena ninguna red |
| Vast y el lanzador `nn/vast.sh` | ❌ | minutos en el dev, 0 $ |
| las particiones T/p de la ganancia | ❌ | no hay entrenamiento supervisado que repartir |

### Lo que vendría después, y NO es de aquí

- Usar los grupos como **etiquetas nuevas**: reconocer la *forma* y luego `forma → dígito`, la lectura por prototipos
  que `feat-pos` (§5) propuso para «varias codificaciones de un 9».
- La posición **relativa** entre features (E2/E3 de `feat-pos`) en vez de la zona del marco.
- Agrupar **dentro** de cada clase (sólo los 1, sólo los 7): daría variantes más finas, pero ya no sería independiente
  de la etiqueta, que es lo que se pidió. Las variantes salen igual de L2.
