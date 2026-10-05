# Reglas de `feat-agr`

**ESPECIFICACIÓN del 2026-10-05, sin implementar.** No hay código, ni grupos, ni un solo número propio. El criterio,
escrito **antes de calcular ningún grupo**, está en `instrucciones/02-criterio.md`. Esta versión es la segunda del mismo
día: la primera (`6d42bda`) la pasaron por el `verificador` y el `revisor`, y lo que corrigieron está aplicado aquí.
Al implementar se reescribe lo que cambie, en el mismo commit.

**Qué pregunta:** si cada dígito se describe por las features que lo componen —qué detector se enciende y en qué zona—
y se agrupan los dígitos por esa descripción **sin leer la etiqueta**, ¿salen grupos de formas semejantes —variantes de
un mismo dígito (el 1 que es sólo una recta vertical y el 1 con un trazo inclinado arriba) y formas que comparten
dígitos distintos— y cuánto coinciden esos grupos con las etiquetas de NIST?

**La idea en una línea:** en todo lo de hoy la etiqueta organizaba (los detectores se componían *para* predecirla).
Aquí se agrupa sin ella, y la etiqueta sólo entra **después**, como lente para leer unos grupos que se formaron solos.

## ⏳ Lo que decide el dueño antes de implementar (el coste no cambia con ninguna)

| # | qué | por defecto | por qué hay que decidirlo |
|---|---|---|---|
| D1 | **qué banco de detectores es el principal** | **los dos, en pie de igualdad** (Z32 y Z8) | los números apuntan en direcciones opuestas (tabla de §2): ninguno de los dos «significa» su feature sobre manuscrito |
| D2 | **qué significa «los agrupamos por los features que los componen»** | **parecido continuo**: k-means sobre *cuánto* se enciende cada feature en cada zona | la otra lectura —mismo grupo si tienen *las mismas* features en las mismas zonas, sí/no— es más literal a tu ejemplo, pero depende de umbrales que sobre dígitos reales están mal calibrados (el arco-E de 32×32 tiene umbral 0,25 y se enciende en el 76 % de los 1). Si la quieres, es un brazo más (**Zb**, Z binarizado) |
| D3 | **L6 — ¿sirven los grupos para elegir qué etiquetar?** | **dentro**, marcada como opcional | no la pediste. Cuesta < 1 min y es la única lectura que dice si los grupos sirven para algo; si no la quieres, se quita y nada más cambia |

## Entradas

### 1. Los dígitos

- **Dataset:** `uci-optdigits-orig-32px-r20261005` (publicado por `feat-ind32`; se lee con `exigir_dataset`, no se
  regenera). Los **5620** dígitos de NIST en 32×32 binario, de 43 escritores: 3823 de 30 (`tra`, `cv`, `wdep`) y 1797 de
  otros 13 (`windep`).
- **Qué se lee:** `imagenes` y `origen`; `etiquetas` **sólo en la lectura** (§5), nunca al agrupar. `particion` (el
  180/1617) **no se usa**: agrupar no entrena nada contra la etiqueta, así que no hay train/val que proteger.
- ⚠ **No hay identificador de escritor por dígito** (`optdigits-orig.names` no lo trae). **Pero los 1797 de `windep`
  vienen en 14 bloques** de 122 a 130 dígitos que siguen **la misma plantilla de 130 etiquetas**, 13 por clase (medido el
  2026-10-05): 8 bloques la siguen entera, uno con una etiqueta distinta y 5 con entre 2 y 8 casillas de menos; 12 de los
  14 empiezan con 0–9 tres veces seguidas. Es la huella de **un formulario impreso, rellenado por una persona**. Pero son
  14 bloques para 13 escritores, así que «bloque = escritor» **no está comprobado**: se usa como *posible* escritor (L5),
  nunca como dato. Los 3823 no tienen esa estructura (ninguna racha de etiquetas consecutivas de más de 4; la fracción de
  pares consecutivos +1, 0,089–0,112, es la del azar).
- **Condiciones que trae:** NIST ya **centra y normaliza** cada dígito al marco de 32×32 (*«Inputs are centered and
  normalized as 32x32 bitmaps»*, `optdigits-orig.names`). Por eso una zona del marco ya es una zona *del dígito*, y no
  hace falta la posición relativa al centro (E3 de `feat-pos`).
- **Al cargar:** el 32×32 en 0/1 para el banco de 32; reducido por bloques 4×4 (0..16, ÷16) para el banco de 8 y para el
  control de píxeles. Es la reducción de NIST: para los 1797 `windep` da exactamente `uci-optdigits-8px-r20261002`
  (comprobado por `feat-ind32`, `nn/datos.py --comprobar`).

### 2. Las features: los detectores de hoy, congelados y COPIADOS

Dos bancos de 13 detectores, entrenados **sólo con dibujos sintéticos** (ninguno vio un dígito):

- **De 32×32:** los de `feat-ind32` (C3), entrada 32×32 → mapa 8×8. Sobre su val **sintético**: rectas F1 0,965–0,971,
  lazo 0,962, arcos 0,862–0,897, esquinas 0,879–0,890; precisión en arcos y esquinas 0,88–0,92 (0,88–0,98 contando las
  13).
- **De 8×8:** los `fino` de `feat-ind` (corrida 2), sobre la reducción 4×4. Sobre su val sintético, precisión en arcos y
  esquinas 0,61–0,67.

⚠ **El nombre de un detector es lo que aprendió sobre dibujos, NO lo que significa sobre manuscrito**, y la firma por
clase de hoy lo mide (fracción de cada dígito en que se enciende; `resultados/firma-por-clase.json` de cada uno):

| se enciende en… | banco de 32×32 | banco de 8×8 |
|---|---:|---:|
| arco-E, en los **1** | 76 % | 70 % |
| arco-W, en los **1** | 70 % | 77 % |
| «/» (recta-S), en los **1** | 28 % | 12 % |
| «/» (recta-S), en los **7** —cuyo trazo principal es un «/»— | **12 %** | **59 %** |
| esquina «┐» (SW), en los **7** | 44 % | 82 % |

Un 1 es una recta y los arcos se encienden en tres de cada cuatro, **en los dos bancos**; el «/» de 32×32 casi no ve el
trazo del 7, y el de 8×8 sí. Lo que se enciende en los 1 a 32×32 (el 28 % del «/») puede ser la bandera o la inclinación:
**no se sabe**. Una explicación posible, **no medida**: a 32×32 el detector está afinado a ±12° de 45° y la diagonal del 7
es más empinada; a 8×8 el conteo 4×4 emborrona el ángulo. Y sobre dígitos, como **entrada de un compositor**, el de 8×8
ganó en todo N (README de `feat-ind32`). Por eso D1: por defecto **los dos son principales**, y cada hipótesis se evalúa
en cada uno.

- **Cómo se usan** (Regla 0: ningún experimento importa de otro): se **copian** los `best.pt` a `nn/detectores-32px/` y
  `nn/detectores-8px/` (13 × 173 KB + 13 × 61 KB = **3,05 MB**, medido; bajo el tope de ≈5 MB por experimento del repo, que con las
  figuras se cuenta en § Salidas) y las dos clases
  `Detector`, autocontenidas, con un `nn/detectores.json` que dice de qué `id` y de qué commit sale cada uno, su huella y
  su umbral `u_f`. Al cargar se recalcula la huella y **se niega** si no casa. **(decido)** copiar en vez de leerlos del
  otro experimento por su `id`: `feat-ind32` sigue abierto y entrenó sus detectores con una sola semilla (lo anota como
  pendiente), así que sus pesos pueden cambiar; estos grupos no deben cambiar con ellos.
- **Cómo se comprobará que las copias son las de hoy:** la firma por clase de los 1797 `windep`, recalculada con las
  copias, tiene que ser **idéntica** a la guardada en `feat-ind32` (clave `firma_windep`) y en `feat-ind` (clave
  `firma`), pedidas al registro por su `id`. ✅ La precondición ya se comprobó el 2026-10-05 (el `verificador`, con los
  pesos de origen): firma recalculada = guardada, diferencia máxima 0,0, y huellas de `best.pt` iguales, en los dos bancos.
- ⚠ **Lo que NO entra, y por qué:** los detectores **ajustados de C** (`ajuste13`, el mejor modelo de hoy) se ajustaron
  **con las etiquetas** de los dígitos: agrupar con ellos metería la etiqueta por la puerta de atrás —la «etiqueta débil»
  que `feat-ind` (`OBSERVACIONES.md` §2.1) señaló como lo que invalida la premisa—. Tampoco los bancos `grueso`, `dig` y
  `cae*`: `dig`/`cae` no tienen nombre de feature y la lectura de un grupo se quedaría sin vocabulario. ⚠ Eso no impide
  medir con ellos si dos descripciones agrupan igual (un ARI no necesita nombres): queda como posible brazo extra, no
  incluido.
- ⚠ **Fuga leve, de diseño:** la forma de la descripción (las zonas, qué bancos) se eligió con lo que hoy midieron
  compositores **supervisados sobre estos mismos dígitos**. Ninguna etiqueta entra en el cálculo de los grupos, pero el
  diseño no es ciego a ellas.

### 3. La descripción de un dígito: qué se enciende, y en qué zona

De los 13 mapas σ (8×8) de cada dígito salen tres resoluciones del «dónde»:

| | qué es | tamaño | qué conserva |
|---|---|---:|---|
| **P** · presencia | el máximo de cada mapa | 13 | *qué* se enciende. Ciega a la posición: el 6 y el 9 deberían caer juntos |
| **Z** · zonas — **la principal** | el máximo de cada mapa en 9 zonas solapadas: ventanas de 4×4 celdas con paso 2 (filas y columnas 0–3, 2–5, 4–7) → NW, N, NE, W, C, E, SW, S, SE | 13 × 9 = 117 | *qué y en qué zona*: «recta-V en el centro + «/» arriba a la izquierda» |
| **M** · mapa | los 13 mapas enteros | 832 | *qué y en qué celda exacta* |

**(decido)** Z es la principal, por tres motivos:

1. Es la resolución de la descripción humana: *«una línea inclinada **en la parte superior**»*.
2. Un máximo sobre una ventana **tolera** el desplazamiento de ±1 celda, y hoy se midió que hace falta: desplazar el
   dígito 1 celda dejó al compositor posicional en **0,556** (8×8) y **0,608** (32×32) de acierto, y leer tras un máximo
   3×3 lo subió a **0,774** y **0,733** (`feat-ind` corrida 14 y `feat-ind32` C6: `resultados/curva-fino.json` y
   `resultados/generalizacion.json`). ⚠ **Tolera, no garantiza**: un rasgo de una sola celda que pasa de la columna 3 a
   la 4 puede cambiar hasta 4 de sus 9 zonas, y de la 1 a la 2, hasta 2 (según en cuántas ventanas caiga su fila). Por
   eso L4 lo **mide** en vez de suponerlo.
3. Cada zona tiene 16×16 px del marco y se solapa a la mitad con sus vecinas.

**El control sin features, X8** (píxeles): el 8×8 contado, 64 números ÷16. **(decido)** el 8×8 y no el binario de 32×32:
el conteo 4×4 suaviza, y es la representación clásica de estos dígitos; al control se le da su mejor versión. Es además
la **única descripción independiente** de los detectores: Z32 y Z8 comparten vocabulario y dibujos sintéticos (uno es el
otro reducido 4×4), así que compararlos entre sí mide la **resolución**, no si los grupos «son de los dígitos».

**Los brazos que se agrupan:** **Z32** y **Z8** (los principales, D1), **P32** y **M32** (el eje de resolución, en un
solo banco porque contesta una pregunta de la descripción y no del banco; si D1 elige sólo el de 8, se hace en ése),
**X8** (píxeles) y, si D2 lo pide, **Zb**. Y uno condicional, decidido ahora: **Z−a** (§5, L7).

**Lo que se enciende en un grupo** —su composición *aparente*, una matriz y no un texto—: 13 × 9 con la fracción de sus
miembros en que cada (feature, zona) pasa el umbral `u_f` de su detector (el guardado en su checkpoint, elegido sobre el
train **sintético**). Lo que pase de 0,5 se escribe en las figuras («recta-V·C + «/»·NW»), **sólo para leerlas**, y con
la advertencia de la tabla de §2: el nombre es el del dibujo.

## Salidas

- **Pesos:** las copias de los detectores (`nn/detectores-32px/`, `nn/detectores-8px/`, 3,05 MB) y `nn/detectores.json`.
  Se commitean. No se entrena ninguna red.
- **Los grupos:** las particiones principales (Z32 y Z8, K = 30) en `resultados/grupos-<brazo>-K30.json`: a qué grupo
  va cada uno de los 5620, centroides, medoides, lo que se enciende y el árbol. **Se commitean**: son el producto del
  experimento. Las demás particiones, en `resultados/grupos.npz` (no se commitea: `*.npz`; se regeneran con las mismas
  semillas).
- **Métricas:** `resultados/lecturas.json` — L1–L7 por brazo, K y semilla, y la evaluación del criterio.
- **Figuras** (`nn/figuras.py`):
  - `grupos-<brazo>-K30.png` para Z32, Z8 y X8 — una fila por grupo, en el orden del árbol: tamaño, mezcla de
    etiquetas, el medoide grande y 15 miembros al azar. **Es lo primero que se mira**, antes que cualquier número;
  - `enciende-<brazo>-K30.png` — por grupo, los 13 mapas de zona (3×3) medios;
  - `arbol-<brazo>-K30.png` — el dendrograma de los 30 grupos;
  - `los-1.png` — todos los 1, ordenados por su grupo, con su índice de bandera (L2b);
  - `consistencia.png` — L4 por brazo y transformación.
- **Qué se commitea:** todo menos `*.npz` (las representaciones y las particiones secundarias se regeneran en minutos).
- **Presupuesto de tamaño:** el tope del repo es ≈5 MB por experimento. Pesos 3,05 MB + figuras ≤ 1,5 MB (en grises o
  con paleta, ≤ 150 KB cada una; las 12 de `feat-ind32` suman 2,3 MB) + JSON < 0,5 MB. Si el total pasa de 5 MB, se
  pregunta antes de commitear.
- **Reporte en el central:** **no** — 0 $ y no cambia `ESTADO.md` (la misma decisión que `feat-ind`). Si el dueño lo
  quiere, se escribe.

## Procesos

### 4. Cómo se agrupa

- **k-means euclídeo** con siembra k-means++ (la del obtenedor de `feat-ind`, corrida 6, copiada y pasada a distancia
  euclídea), hasta que ninguna asignación cambie o 300 iteraciones; un grupo vacío se re-siembra en el punto más lejano.
  **5 semillas** por (brazo, K).
- **K = 30 el principal** *(decido)*: deja sitio a unas 3 variantes por dígito, y 30 filas son una rejilla que se puede
  mirar en el móvil. Se calculan además **K = 10** —tantos grupos como etiquetas—, **20 y 50**, que son los que deciden
  si un corte es real o un efecto de K (§5, L2).
- **La partición que se dibuja y se lee es la de menor inercia de las 5 semillas**: se elige **sin mirar etiquetas**.
- **Árbol:** Ward sobre los 30 centroides, pesados por tamaño → qué grupos se juntan primero. Es la lectura de «familias
  de formas», y ordena las filas de las rejillas para que lo parecido quede junto.
- **Dos ajustes:** (a) sobre los **5620**, que es lo que se dibuja y se lee; (b) sobre los **3823** de 30 escritores
  —Z32, Z8 y X8—, para medir si los grupos valen para los 13 que no vio (cada uno se asigna al centroide más cercano) y
  para elegir a quién etiquetar (L6).
- ⚠ **La etiqueta no puede entrar por accidente, y eso lo garantiza el código, no el cuidado:** el cargador de dígitos
  **no devuelve** `etiquetas` salvo que se pidan explícitamente, y sólo la lectura las pide. `--comprobar` falla si el
  script que agrupa las toca.

### 5. Las lecturas — la etiqueta entra aquí, y sólo aquí

| # | qué | cómo |
|---|---|---|
| **L1** | coincidencia con las etiquetas | pureza, **NMI** y ARI por brazo y K, **siempre al lado de X8 con el mismo K** (la pureza crece con K por sí sola). El azar se **mide**, permutando las etiquetas. Y con **K = 10** —tantos grupos como etiquetas— qué clase se parte y qué clases comparten grupo |
| **L2** | **cortes de una clase** | *grupos de c* = los que tienen ≥ 10 % de los dígitos de *c*, **sea cual sea su mayoría** (si los 1 con bandera caen en un grupo de 7, cuentan). Un **corte** es un par de grupos de *c*. **Persiste en K′** si, con K′ grupos, los *c* de uno caen mayoritariamente en un grupo distinto del de los *c* del otro. **Es de forma** si persiste en K′ = 20 **y** 50 y no lo explican el grosor ni la inclinación (los dos con AUC < 0,80 entre los *c* de un lado y del otro) |
| **L2b** | **los tres índices de píxeles**, medidos **sin detectores** en todos los dígitos | *bandera*, *grosor* e *inclinación* (abajo). Entre los dos lados de un corte, el AUC de cada uno dice **por qué** se partió |
| **L3** | **formas compartidas** | *grupo mezclado* = mayoría < 70 % y ≥ 20 miembros; qué pares de dígitos comparten forma. Y el **enriquecimiento** en los fallos de C, contado **por evaluación**: fracción de sus 67 evaluaciones falladas (de 6000: 3 semillas × 2000 de test; un dígito cuenta tantas veces como se evaluó; `resultados/errores-c.json` de `feat-ind32`, pedido por `id`, índices de este mismo dataset) que cae en grupos mezclados ÷ fracción de las 6000 que cae en ellos. Las 6000 se regeneran con la `particion()` de `feat-ind32` copiada (T = 4000, p = 0,5, semillas 1–3). ✅ Ya reproducida el 2026-10-05 (el `verificador`): da su `huella_particiones`, `fda771968953b23d`, y los 67 fallos caen en el test de su semilla. En Z32, Z8 **y X8** |
| **L4** | **¿forma o grosor?** — sin etiquetas | consistencia *c*: fracción de dígitos que, **engrosados 2 px, adelgazados 1 px o desplazados 4 px**, caen en su mismo grupo (centroide más cercano); y corregida por azar, κ = (*c* − Σ *p*²) / (1 − Σ *p*²), con *p* la fracción de dígitos de cada grupo —un brazo con un grupo gigante sería «consistente» sin agrupar nada—. Las transformaciones son las de C6 de `feat-ind32`, copiadas |
| **L5** | estabilidad | ARI entre las 5 semillas; en los 1797, entre «asignados desde los centroides de los 3823» y «agrupados aparte»; qué grupos de los 3823 casi no aparecen en los 1797; ARI entre brazos (Z32–Z8 mide la resolución, Z–X8 compara dos descripciones independientes); y, con los 14 bloques de `windep`, los grupos cuyos miembros de `windep` salen ≥ 50 % de un solo bloque (lo esperable es ~7 %): **posible estilo de una persona**, no comprobado |
| **L6** | *(opcional, D3)* **¿para qué sirven?** | los K medoides del ajuste (b), etiquetados → el compositor lineal de hoy sobre los mapas M32 → acierto en los 1797; medoides de Z32, de Z8 y de X8, contra K al azar y K al azar estratificado (5 sorteos cada uno), para K = 10, 20, 30, 50 y 100. Y la propagación: cada dígito toma la etiqueta del medoide de su grupo |
| **L7** | **¿qué manda en los grupos?** — sin etiquetas | qué fracción de la separación entre grupos (la suma de cuadrados entre grupos, por dimensión) aporta cada feature y cada zona |

**Los tres índices de L2b**, con las filas de tinta del dígito r0…r1 (alto *h*) e *I(r)* = la columna de tinta más a la
izquierda de la fila *r*:

- **bandera** *b* = mediana de *I* en el tercio central (filas r0 + h/3 … r0 + 2h/3) − mínimo de *I* en el cuarto
  superior (r0 … r0 + h/4), en px. Un 1 recto da *b* ≈ 0; uno con bandera, *b* grande; uno inclinado como «/», *b* < 0;
- **grosor** = píxeles de tinta ÷ filas con tinta;
- **inclinación** = ángulo del eje principal de la tinta respecto de la vertical (momentos de segundo orden).

El AUC se toma sin dirección, max(AUC, 1 − AUC), para que los tres compitan en igualdad. *b* tiene un punto ciego
conocido —un 1 inclinado como «\» también adelanta la tinta de arriba a la izquierda—, y para eso está la inclinación al
lado. Fuera del 1 no hay testigo positivo de forma: el grosor y la inclinación sólo **descartan** cortes, y lo demás lo
dicen las rejillas.

**El brazo condicional Z−a, decidido ahora:** si en un banco los 4 arcos aportan **más de la mitad** de la separación
entre grupos (L7), se repite ese brazo **sin los arcos** y se reportan H1, H2 y H4 con él. La regla va escrita antes
porque los arcos se encienden en tres de cada cuatro 1 (§2), y es el camino más probable por el que el grosor acabe
mandando sobre la forma.

### 6. Pasos

1. Copiar los detectores y la `particion()`, y escribir `nn/detectores.json`. `--comprobar`: huellas, firma por clase
   idéntica a la de origen, `huella_particiones` de `feat-ind32`, métricas contra casos hechos a mano, y que el
   agrupador no toca la etiqueta.
2. **Representar** los 5620, limpios y con las 3 transformaciones, en los dos bancos → `resultados/representaciones.npz`.
3. **Agrupar:** los brazos × K ∈ {10, 20, 30, 50} × 5 semillas sobre los 5620; Z32, Z8 y X8 además sobre los 3823, con
   K hasta 100; Z−a si L7 lo pide.
4. **Mirar** las rejillas, antes que los números.
5. **Leer:** L1–L7 → `resultados/lecturas.json`, con la evaluación del criterio. README con lo que salió, y
   `feat-agr` en «Quién lo usa» del README del dataset.

- **Qué se mide y con qué umbral:** `instrucciones/02-criterio.md` (comprobaciones de cordura y H0–H6), escrito antes.
- **Cuántos brazos y cuántas semillas:** los de §3; 5 semillas de k-means; el compositor de L6, 3 semillas del
  optimizador.
- **Qué se llama «ganar»:** no se declara ganador entre brazos. Cada hipótesis se confirma, se refuta o empata, en cada
  banco; el producto principal son **los grupos, mirados**.

### 7. Dónde corre y cuánto cuesta

**0 $, en el dev, sin alquilar nada.** Estimado, **no medido**: ≤ 15 min en total. Los mapas de los 5620 con el banco de
32 tardaron 41 s (medido el 2026-10-05 en `feat-ind32`, `nn/aplicar.py`, con pesos de prueba de la misma red) y 33 s al
repetirlo ese día (el `verificador`); ×4 con las transformaciones ≈ 2–3 min. El banco de 8, ~1 min; los k-means,
~8 min (lo caro es M32, 832 dimensiones); L6, < 1 min.

Al implementar, `gasta` pasa a `entrena-local` con la entrada en `nn/entrenar_local.py` —el contrato con el freno: si
esto tarda, el freno tiene que verlo—. Si la primera pasada entera pasa de 10 min, se lanza como unidad
(`desacoplar-persistente.sh`) con un lanzador que imprime la orden, tiene modo seco y se niega a lanzar dos veces.

## Scripts

Previstos. Los nombres y las banderas son de este experimento, y si cambian al implementar, esta tabla cambia en el
mismo commit.

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/detector32.py`, `nn/detector8.py` | las dos redes de detector, copiadas y autocontenidas: cargar y huella | — |
| `nn/datos.py` | los 5620 por `exigir_dataset`, con sus huellas; **sin etiquetas** salvo `con_etiquetas=True`; los bloques de `windep`; la `particion()` copiada | `python nn/datos.py --comprobar` |
| `nn/representar.py` | P, Z y M de los bancos, y X8; limpios y transformados; los tres índices de L2b | `python nn/representar.py` → `resultados/representaciones.npz` |
| `nn/metricas.py` | k-means++, Ward, pureza, NMI, ARI, AUC y κ en numpy, probados contra casos hechos a mano | `python nn/metricas.py --comprobar` |
| `nn/entrenar_local.py` | los k-means de todos los brazos (el nombre es el contrato con el freno) | `python nn/entrenar_local.py [--brazos Z32,Z8,X8] [--K 10,30]` → `resultados/grupos.npz`, `grupos-<brazo>-K30.json` |
| `nn/leer.py` | L1–L7 y el criterio | `python nn/leer.py` → `resultados/lecturas.json` |
| `nn/figuras.py` | las figuras de § Salidas | `python nn/figuras.py` → `resultados/*.png` |

- **Dependencias:** el `.venv` de la raíz (numpy 2.5.3, torch 2.14.1 CPU, Pillow 12.3.0, matplotlib 3.10.9; comprobado el
  2026-10-05). **No hay scipy ni scikit-learn**, y no se instalan: k-means, Ward, NMI, ARI, AUC y κ son ~150 líneas de
  numpy con su prueba, y así no se toca el venv que comparten los demás experimentos.
- **De dónde sale el código:** autónomo, con piezas copiadas (abajo). Ningún import de otro experimento.

## Qué NO hereda

- **Se copió de:** ningún experimento como carpeta. Se copian **piezas**: las redes y los pesos de los detectores (de
  `feat-ind32` y `feat-ind`), las transformaciones, el compositor lineal y la `particion()` (de `feat-ind32`), y el
  k-means++ (del obtenedor de `feat-ind`, pasado a distancia euclídea).
- **Qué se cambió a propósito:** la etiqueta deja de ser el objetivo —no se entrena nada contra ella y sólo se lee al
  final—; no hay reparto train/val para agrupar; la posición se lee en 9 zonas en vez de 64 celdas; y el resultado
  principal es una **partición**, no un acierto.
- **Qué se conservó, y por qué:** los detectores exactos (con su huella) y los 5620 en su orden, para que los grupos se
  lean contra la firma por clase y los errores de C de hoy; las transformaciones de C6, para que la consistencia ponga a
  prueba lo mismo que allí el acierto bajo transformación (el G3 de la corrida 12 de `feat-ind`, que C6 reproduce); el compositor lineal en L6, para que «acierto con K etiquetas» use el mismo lector que las
  curvas de hoy.
- **Contra qué se compara:** consigo mismo — features contra píxeles (X8 es el control de **todas** las hipótesis), un
  banco contra otro, tres resoluciones del dónde, elegir por grupos contra elegir al azar. Los aciertos de hoy (curvas,
  N(ε), G) son **contexto**, no brazo: allí se sorteaba de datasets balanceados de los 43 escritores, y aquí se evalúa en
  los 13 de `windep`.
- **Restricciones de otros experimentos que NO aplican aquí:** el reparto 180/1617 y el test fijo de 717 (aquí nada se
  entrena contra la etiqueta al agrupar); los umbrales de «aprendió» de los detectores (no se entrena ninguno); Vast y
  *«todo lo que es paralelo debe correrse en paralelo»*, que era para lo que se alquilaba (aquí no se alquila); las 3
  semillas del compositor como semillas del experimento (aquí mandan las 5 de k-means).

### Qué se usa de lo de hoy, y qué no

| de hoy | ¿aquí? | por qué |
|---|---|---|
| los 13 detectores de 32×32 (`feat-ind32`, C3) | ✅ banco principal (D1) | entrenados sin dígitos |
| los 13 `fino` de 8×8 (`feat-ind`, corrida 2) | ✅ banco principal (D1) | entrenados sin dígitos; sobre manuscrito, ninguno de los dos significa su nombre (§2) |
| los mapas σ 8×8 y la firma por clase (`nn/aplicar.py`) | ✅ | son la descripción; la firma, la prueba de que las copias son las de hoy |
| el máximo 3×3 contra el desplazamiento (corridas 13–14, C6) | ✅ convertido en las zonas | la lección medida de hoy |
| las 3 transformaciones de C6 | ✅ L4 | la misma forma se pone a prueba igual que allí |
| los 54 dígitos que falla C (`errores-c.json`) y la `particion()` | ✅ L3 | ¿los grupos mezclados explican los errores?, con la tasa base exacta |
| 3823 (30 escritores) / 1797 (13) | ✅ L5, L6 | la única separación de escritores que existe |
| el compositor lineal | ✅ L6 | mismo lector que las curvas |
| N(ε), las curvas de aprendizaje | ✅ como pregunta de L6 | ¿agrupar ahorra etiquetas? |
| los detectores ajustados de C | ❌ | vieron las etiquetas |
| las CNN (LeNet-5, la del repo, A, B, igualada) | ❌ | aquí no se entrena ninguna red |
| Vast y el lanzador `nn/vast.sh` | ❌ | minutos en el dev, 0 $ |
| las particiones T/p de la ganancia, para entrenar | ❌ | no hay entrenamiento supervisado que repartir (sólo se regenera la de C, para L3) |

### Lo que vendría después, y NO es de aquí

- Usar los grupos como **etiquetas nuevas**: reconocer la *forma* y luego `forma → dígito`, la lectura por prototipos
  de `feat-pos` (§5, *«Sólo una de las codificaciones posibles de un 9»*).
- La posición **relativa** entre features (E2/E3 de `feat-pos`) en vez de la zona del marco.
- Agrupar **dentro** de cada clase (sólo los 1, sólo los 7): daría variantes más finas, pero ya no sería independiente
  de la etiqueta, que es lo que se pidió. Los cortes salen igual de L2.
