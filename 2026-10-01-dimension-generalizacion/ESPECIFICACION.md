# `dim-gen` — reducir las dimensiones de la imagen y medir qué le hace a la generalización

**2026-10-01 · ES UN PLAN. Nada ejecutado, nada entrenado, ningún script escrito.** Lo
escribió Claude a petición del dueño (literal y completa en
[`instrucciones/01-encargo.md`](instrucciones/01-encargo.md)); **no es una transcripción del
dueño** como la especificación de `banco-k`. Es **historial**: si algo de aquí discrepa de
[`REGLAS.md`](REGLAS.md) o de [`instrucciones/02-criterio.md`](instrucciones/02-criterio.md),
mandan ésos, que son lo vigente y se actualizan con el experimento.

Todo número lleva su procedencia: **medido** (con fecha y cómo) o **estimado**. Lo medido hoy
se midió **sin entrenar nada**: tamaños de caja, piso y huellas con las etiquetas e imágenes
publicadas, y el coste por paso con pesos aleatorios.

Revisado el 2026-10-01 con los agentes `arquitecto` (estructura: §6 y la carpeta) y
`revisor` (RESERVAS, todas recogidas: §1, §2.4, §3.2 y el criterio).

## 0. Lo primero: cuatro SUPUESTOS que el dueño tiene que confirmar, y el plan en una tabla

El encargo admite más de una lectura en cuatro sitios, y el experimento cambia con cada una.
Se ha elegido una; **si alguna no es la que el dueño quería, se cambia el plan, no se corre**.

| | supuesto elegido | la otra lectura |
|---|---|---|
| **S1** | **«ventana» = el kernel** de cada convolución (`n × n`) | el recorte de entrada, que es lo que «ventana» suele significar en este proyecto |
| **S2** | **`n = f · W`**, con `W` el ancho de la imagen que **entra en la red**; el mismo `n` en las `L` capas | `n` a partir del ancho de la entrada de **cada capa** (encoge capa a capa; §3.3) |
| **S3** | **la tarea es localizar la caja de tinta** de un párrafo, medida con **IoU**: es lo que el dataset etiqueta | otra etiqueta pediría publicar otro dato |
| **S4** | **«reducción» = el lado `W`**: 128 → 64 → 32 → 16 → 8, por suma exacta de bloques | otros `W` (96, 48…) exigirían interpolar |

| # | qué | dónde | reloj | coste | depende de |
|---|---|---|---|---|---|
| 0 | **los 4 supuestos y las 6 decisiones del dueño** (§7) | — | — | — | — |
| 1 | `nn/datos.py`: cargar, recortar a 128, reducir a `W` por suma exacta, control `w128-de16`, etiquetas; `--comprobar` con las **huellas congeladas** de `REGLAS.md` | dev | ~1 h de código | 0 $ | 0 |
| 2 | `nn/modelo.py` + `nn/probar.py`: la red `(W, L, f, C)` autocontenida; tests de mapa `L×L`, parámetros, huellas, y que un acumulador `uint16` **cae** | dev | ~1 h de código | 0 $ | 1 |
| 3 | `nn/entrenar_local.py`, `nn/informe.py`, `nn/vast.sh` + `nn/vast.json` (o `nn/lanzar.sh` si va al dev); **`gasta` pasa a `alquila`/`entrena-local` en ese commit** | dev | ~2–3 h de código | 0 $ | 2 |
| 4 | **ensayo de mecanismo**: `W = 32` (1000 pasos, 15 s) y `W = 128` (50 épocas, ~22 min con `C = 8`), sólo pérdida de `train`; congela `lr` | dev | ~25 min | 0 $ | 3 |
| 5a | **los brazos en Vast** (recomendado): 5 máquinas, una por semilla con todos sus brazos | Vast | **≈2 h** de reloj sin control · **≈3,5 h** con él *(estimado del paso medido)* | **≈0,4–0,55 $** · **≈0,8–1,05 $** *(estimado)* | 4 |
| 5b | **o en el dev**, serializados en una unidad de systemd | dev | **≈8,5 h** con `C = 8` · **≈4,5 h** con `C = 4` *(derivado del paso medido)*; con control, casi el doble | 0 $ | 4 |
| 6 | `nn/informe.py`: tabla, figuras, criterio aplicado; `README.md`; reporte al central según §6 | dev | ~1 h | 0 $ | 5 |

## 1. De dónde se parte (leído y medido el 2026-10-01)

- **El dataset que sirve ya está publicado** y en el almacén: `parrafos1000-584px-r4-r20260908b`
  (§2). Su reparto `train 100 / monitor 100 / eval 800` **es** el 10 % / 90 % del encargo, sin
  tocar nada.
- **Qué da este dato con 100 imágenes, con otra red**: `banco-k` (calibrado el 2026-09-08
  sobre este mismo dataset, 10 semillas): caja media **0,2479** de IoU en `eval`; identidad
  (su CNN de 5.812 parámetros, `same`, stride 2) **0,7981 ± 0,0057** en `eval`, **0,8401** en
  `train`, **brecha +0,0421 ± 0,0059**. Es **contexto**, no comparación (otra red); dice que el
  dato tiene rango (0,55) y que la desviación entre semillas en este régimen es ≈0,006–0,008.
- **Nadie ha medido esto**: en ninguno de los repos hay una medida del efecto de la
  **resolución de la imagen** sobre la generalización con pocos datos. `foveal-vision` barre
  tamaños de ventana, de borde y strides a 80 px, y su `fovea_px` está **atado al dataset**
  («no barrible», `ESTADO.md`); el banco fija 128. El §3.2 de la especificación del banco
  (*«un factor agresivo (/8) convierte el párrafo en un rectángulo gris casi uniforme»*) es una
  expectativa de diseño: el banco **nunca corrió /8**.
- ⚠ **Lo que SÍ está medido, dos veces, es que este tipo de diseño deja el resultado
  confundido.** `ESTADO.md` del repo central, línea 54, sobre `border_reduce`: *«sin cerrar
  (confundido) … Capacidad y resolución están confundidas en este diseño»*; y línea 131:
  *«Desconfundirlo pide un diseño que suba N sin subir el área, o que compare a parámetros
  igualados»*. El reporte #11 (E10b: *«a igual área, ¿ayuda verla con más resolución?»*) ganó
  con `p` = 0,008 **y no pudo atribuirlo**, porque la resolución traía +156 % de parámetros; el
  #21 deja la caída de `s = 4` *«sin atribuir entre resolución y capacidad»*. **Este experimento
  tiene exactamente ese confound** (§3.2), y por eso trae un control (§3.6) y un criterio que lo
  nombra antes de mirar.
- **La máquina**: dev de 2 vCPU y 3,8 GB, sin swap; `.venv` con torch 2.14.1+cpu y numpy 2.5.3
  en `~/src/experimentos-cnn` *(el de este workspace no existe todavía)*; freno en verde al
  empezar. **Vast hoy** (fase 1 de `banco-k`, 2026-10-01): máquina lista en 1 min 48 s,
  0,049 $/h, y **~13 % más rápida que el dev** en ese run (36,0 s contra 41,5 s): lo que compra
  Vast es paralelismo y no ocupar el server del bot, no velocidad.

## 2. El dato

### 2.1 Por qué `parrafos1000-584px-r4-r20260908b` y no otro

| candidato | qué es | por qué sí / no |
|---|---|---|
| **`parrafos1000-584px-r4-r20260908b`** | 1000 imágenes de **un** párrafo, 146×146 `uint16` = suma exacta de bloques 4×4 del lienzo de 584; etiqueta = caja de tinta (4 enteros); reparto 100/100/800 estratificado por área de caja | ✅ el reparto **es** 10/90; la etiqueta es **geométrica** (no cambia al reducir); la suma exacta permite reducir más **sin interpolar**; piso de caja media bajo (0,25) |
| `parrafos1000-584px-r4-r20260908` | el mismo diseño con anchos/altos uniformes | ❌ piso 0,5261: las cajas son demasiado parecidas y un predictor constante acierta la mitad; comprimiría todo igual que comprimió al banco |
| `parrafos1000-pagina1024-r20260909` | 287 páginas de 1024 con 2–5 párrafos | ❌ varios objetos por imagen: la etiqueta deja de ser una caja; habría que recortar y **publicar** un dato nuevo. Es la salida si algún día hacen falta `W > 128` |
| `esquinas300-32px-…`, `limites300-…` | ventanas de 32 px con esquinas/límites | ❌ 32 px es el techo: sólo quedarían 16 y 8 |
| `window-datasets/*` de `foveal-vision` | ventanas de 80 px con máscaras | ❌ otra tarea y otro pipeline; el encargo pide «la cnn» de este repo, no `fv` |

### 2.2 Qué se lee y cómo se reduce

- Recorte central **146 → 128** (9 px por lado): es el marco en el que viven las etiquetas
  (`(c/4) − 9`, rango `[8, 119]`: no corta ninguna caja), y **128 es potencia de 2**, con lo
  que las reducciones a 64, 32, 16 y 8 son sumas de bloques `2×2`, `4×4`, `8×8` y `16×16`
  **exactas**. En **`int64`**: a 16 y 8 la suma no cabe en `uint16` (261.120 y 1.044.480,
  medido) y un acumulador corto desbordaría **sin avisar**.
- **Las reducciones NO se publican como datasets nuevos** (decisión con el `arquitecto`, R4/R15):
  son pipeline, como el recorte 146 → 128 que el propio README del dataset declara «parte del
  pipeline, no del dato». Lo que las hace «el mismo dato» es la **huella congelada** de cada `W`
  en `REGLAS.md` (calculadas hoy) que `datos.py` recalcula y exige.
- Entrada: **fracción de tinta** por píxel, `x = 1 − suma / (255·16·b²) ∈ [0, 1]`, igual a todo
  `W`. ⚠ Y es la razón por la que a `W` pequeño todavía hay información: la suma de bloque
  **codifica en el gris cuánta tinta cae dentro**, así que el borde de una caja de 3 px sigue
  teniendo posición subpíxel en los valores. No es lo mismo que umbralizar.
- Etiqueta: la caja normalizada a `[0, 1]`, **la misma para los cinco `W`**.

### 2.3 Qué tamaño tienen las cajas a cada `W` (medido el 2026-10-01 con las 1000 etiquetas)

| `W` | ancho de caja (mín · mediana · máx) px | alto de caja (mín · mediana · máx) px | cajas con alto < 2 px |
|---:|---|---|---:|
| 128 | 19,5 · 56,9 · 105,0 | 6,5 · 38,5 · 110,8 | 0 % |
| 64 | 9,8 · 28,4 · 52,5 | 3,2 · 19,2 · 55,4 | 0 % |
| 32 | 4,9 · 14,2 · 26,3 | 1,6 · 9,6 · 27,7 | 0 % |
| 16 | 2,4 · 7,1 · 13,1 | 0,8 · 4,8 · 13,9 | **8,9 %** |
| **8** | **1,2 · 3,6 · 6,6** | **0,4 · 2,4 · 6,9** | **40,5 %** (y 8,9 % < 1 px) |

A `W = 8` hay cajas **más finas que un píxel**. Es el extremo que enseña dónde se rompe, y por
eso entra (decisión 4), pero se lee con eso delante: su caída es **cuantización**, no
generalización.

**El piso de este reparto** —predecir siempre la caja media de `train`— da **0,2464** de IoU
sobre las 900 *(medido con las etiquetas)*. Coincide con el 0,2479 que midió el banco sobre
`eval`, como tiene que ser: es una propiedad del dato.

### 2.4 Qué tarea es ésta, y qué NO se puede medir con este dato

- **Localizar un bloque de tinta.** El IoU de una caja aguanta el desenfoque: en `banco-k`,
  filtrar con una gaussiana de σ = 1,5 px **subió** el IoU (0,7981 → 0,8130) y bajó la brecha.
  Así que **se espera una meseta larga hacia `W` bajo** y que la caída llegue por cuantización.
  Eso no es un fallo del experimento: es la respuesta **para esta tarea**. Sobre tareas que
  necesiten el trazo (leer, separar palabras) este dato no dice nada, y el plan lo dice.
- **El techo es `W = 128`.** El publicado ya es /4 del render; /1 y /2 no existen aquí. Pedirían
  publicar un dato nuevo (de las páginas de 1024, recortando un párrafo por imagen).

## 3. La arquitectura, y la elección de `f`

### 3.1 La regla

`L` capas `Conv2d(C_in, C, n)` + ReLU, **sin padding, sin stride, sin pooling**: cada capa
reduce el lado en `n − 1`. Después, `Flatten` y `Linear(mapa² · C → 4)`. El kernel es
**relativo al ancho de la imagen que entra en la red** (S2): `n = f · W`, el mismo en las `L`
capas.

Con eso el lado final es `W − L·(n − 1)`, y hay **un solo `f` que lo hace independiente de `W`**:

    f = 1/L   ⇒   n = W/L   ⇒   lado final = W − L·(W/L − 1) = L     (para todo W múltiplo de L)

Con `L = 4`: **mapa final `4×4` para los cinco `W`**, cabeza de **516 parámetros constante**, y
campo receptivo `L·(n − 1) + 1 = W − 3`: **casi la imagen entera a `W` alto** (98 % a 128,
95 % a 64, 91 % a 32), y menos a `W` bajo (81 % a 16, 62 % a 8), porque los 3 px que faltan
pesan más cuanto menor es `W`. La red «ve» lo mismo a todas las escalas salvo ese borde; lo que
cambia son los píxeles —y, con ellos, los parámetros (§3.2).

### 3.2 Las tablas: `f = 0,1` (el encargo) contra `f = 0,25` (lo propuesto), `L = 4`, `C = 8`

*Calculado el 2026-10-01 (`n = round(f·W)`, mínimo 1; MMAC = millones de multiplicaciones-sumas
por imagen en el forward).*

**`f = 0,1`:**

| `W` | `n` | `n/W` | mapas por capa | mapa final | campo receptivo | p. conv | **p. cabeza** | total | MMAC |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 128 | 13 | 0,102 | 116 → 104 → 92 → 80 | **80** | 49 (38 % de W) | 33.832 | **204.804** | 238.636 | 295,9 |
| 64 | 6 | 0,094 | 59 → 54 → 49 → 44 | 44 | 21 (33 %) | 7.232 | 61.956 | 69.188 | 17,7 |
| 32 | 3 | 0,094 | 30 → 28 → 26 → 24 | 24 | 9 (28 %) | 1.832 | 18.436 | 20.268 | 1,2 |
| 16 | **2** | **0,125** | 15 → 14 → 13 → 12 | 12 | 5 (31 %) | 832 | 4.612 | 5.444 | 0,1 |
| 8 | **1** | **0,125** | 8 → 8 → 8 → 8 | 8 | **1** | 232 | 2.052 | 2.284 | 0,0 |

**`f = 0,25`:**

| `W` | `n` | `n/W` | mapas por capa | mapa final | campo receptivo | p. conv | p. cabeza | total | MMAC |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 128 | 32 | 0,250 | 97 → 66 → 35 → 4 | **4** | 125 (98 %) | 204.832 | **516** | 205.348 | 443,9 |
| 64 | 16 | 0,250 | 49 → 34 → 19 → 4 | 4 | 61 (95 %) | 51.232 | 516 | 51.748 | 30,0 |
| 32 | 8 | 0,250 | 25 → 18 → 11 → 4 | 4 | 29 (91 %) | 12.832 | 516 | 13.348 | 2,2 |
| 16 | 4 | 0,250 | 13 → 10 → 7 → 4 | 4 | 13 (81 %) | 3.232 | 516 | 3.748 | 0,2 |
| 8 | 2 | 0,250 | 7 → 6 → 5 → 4 | 4 | 5 (62 %) | 832 | 516 | 1.348 | 0,0 |

**Los cuatro problemas de `f = 0,1` con `L = 4`, que son los que el dueño pidió buscar:**

1. **La cabeza es el experimento.** Cuatro capas al 10 % sólo quitan el 40 % del lado: queda un
   mapa de `0,6·W + 4`, y la `Linear` que lo lee escala con **`W²`**: 204.804 parámetros a 128
   contra 4.612 a 16, el **86 %** del total a 128. Lo que se mediría contra `W` es sobre todo
   **tamaño de cabeza**, no resolución.
2. **El campo receptivo no llega al objeto.** Cubre el 28–38 % de la imagen, y la caja de un
   párrafo mide hasta el 82 % del lado: localizarla la hace la cabeza densa con pesos por
   posición, que es lo contrario de lo que se quiere observar.
3. **El redondeo rompe la regla donde más importa.** A `W = 16`, `n = 1,6 → 2` (12,5 %); a
   `W = 8`, `n = 0,8 → 1`: una convolución `1×1` es puntual, **no reduce ni ve vecinos**, así que
   el brazo de 8 no es una red del mismo tipo que los demás.
4. **No es «cada capa reduce»** en ningún sentido útil: a 128 el mapa pasa de 128 a 80 en cuatro
   capas.

Con `f = 1/L` los cuatro desaparecen: cabeza constante, campo receptivo casi completo, `n = W/L`
entero para todo `W` múltiplo de 4 (`n = 2` a `W = 8`, que sigue siendo una convolución), y la
reducción por capa es la que colapsa el mapa.

⚠⚠ **Lo que `f = 0,25` NO quita, y hay que decirlo con la misma claridad: el confound de
capacidad.** Los parámetros **crecen con `W²`** sea cual sea `f` —con `f = 0,25` el total a 128
es 205.348 y a 8 es 1.348, **×152**; con `f = 0,1` sería 238.636: casi lo mismo—. `f = 0,25` no
elimina el confound, lo **mueve de la cabeza al tronco**: un kernel que ve el 25 % de una imagen
más grande tiene más píxeles que pesar. **Más resolución ES más parámetros en el diseño que pide
el encargo** (`n = f·W` con `L` y `C` fijos), y ésa es la clase de resultado que `ESTADO.md`
etiqueta *«sin cerrar (confundido)»* (§1). No se presenta `f = 0,25` como «lo que aísla la
resolución»: aísla la **forma** de la red (mapa, cabeza, campo receptivo), no su tamaño. El
tamaño lo separa el control (§3.6) y lo lee el criterio (descomposición en «información» /
«generalización» / «parámetros»).

### 3.3 La otra lectura de «ancho de entrada» (S2), calculada y descartada

Si `n` se tomara como fracción del ancho **de la entrada de cada capa** (no de la imagen), el
kernel encogería capa a capa y el mapa **nunca colapsaría**: con `f = 0,25` quedan 44 px de 128
(kernels 32, 24, 18, 14) y 8 de 16; con `f = 0,1`, 87 de 128 y 14 de 16 *(calculado hoy)*.
Vuelve el problema 1. Por eso `n = f · W`, con `W` el ancho de **la imagen que entra en la red**.

### 3.4 Si el dueño prefiere mantener `f = 0,1`: la regla dice `L = 10`

`f = 1/L` deja elegir **uno** de los dos. Las tres combinaciones calculadas, todas sobre este
dato (para `L = 5` y `L = 10` hace falta un recorte de **120** en vez de 128 —13 px por lado,
etiquetas en `[4, 115]`, caben—, porque `W` tiene que ser múltiplo de `L`):

| `L` | `f` | recorte | `W` posibles (`n = W/L ≥ 2`) | mapa final | ms por paso a `W` máx (medido, `C = 8`) |
|---:|---:|---:|---|---:|---:|
| **4** | **0,25** | 128 | **128 · 64 · 32 · 16 · 8** (5 puntos, escalera ×2 exacta) | 4×4 | 5.162 |
| 5 | 0,20 | 120 | 120 · 60 · 30 · 20 · 15 · 10 (6 puntos, no es ×2) | 5×5 | 4.386 |
| 10 | 0,10 | 120 | 120 · 60 · 30 · 20 (4 puntos; a 10 ya es `n = 1`) | 10×10 | 2.423 |

**Recomendada `L = 4, f = 0,25`**: es la `L` del encargo, da cinco puntos en una escalera ×2
exacta desde el marco natural del dato, y es la red menos profunda (menos cosas que confundir
con la resolución). `L = 10` conserva el «10 %» del dueño al precio de una red de diez capas y
sólo cuatro puntos. **Las tres tienen el confound de §3.2**: no es cosa de `f`.

### 3.5 Canales, inicialización, salida

- **`C` canales por capa, constante entre `W`** (es un parámetro molesto, no la variable):
  recomendado **`C = 8`**; **`C = 4`** es la variante barata (§5). `C = 16` cuesta 3,4 h por
  corrida a 128 y no entra.
- **Inicialización**: la de PyTorch por defecto (`U(±1/√fan_in)`, con `fan_in = n²·C_in`), que
  **ya escala con el tamaño del kernel**: las activaciones iniciales tienen magnitud comparable
  a todos los `W` sin tocar nada.
- **Salida**: los 4 números de la `Linear` son la caja normalizada `(izq, der, sup, inf)`.
  Pérdida L1 contra la etiqueta normalizada. Para el IoU la predicción se recorta a `[0, 1]`.

### 3.6 El control que separa parámetros de información: `w128-de16`

La imagen de `W = 16` **repetida 8×8** hasta 128 (cada píxel de 16 es un bloque uniforme de
8×8), entrenada con **la red de `w128`**: mismos 205.348 parámetros que `w128`, misma
información que `w016`. Cinco corridas más (una por semilla), al coste de `w128` (§5).

| si sale | se lee |
|---|---|
| `IoU_val(w128-de16) ≈ IoU_val(w016)` | los parámetros no explican `w128 − w016`: es **información** |
| `IoU_val(w128-de16) ≈ IoU_val(w128)` | lo explican los **parámetros**, y 16 px de información bastaban |
| en medio | se reporta la fracción, sin redondear |

Y su brecha contra la de `w016` dice si **más parámetros a igual información** memorizan más.
La alternativa «a parámetros igualados» —ensanchar `w016` hasta ≈200.000 parámetros
(`C ≈ 64`)— se descarta: cambia la **anchura** de la red, que es otra clase de capacidad, y
compararía dos arquitecturas distintas. `w128-de16` compara la **misma** red con dos entradas.

## 4. Protocolo de entrenamiento

| | valor | por qué |
|---|---|---|
| pasos | **1000** (lote 20 → 5 pasos por época → 200 épocas) | el régimen que ya se sabe suficiente sobre este dato con 100 imágenes (identidad del banco) — decidido, no heredado |
| optimizador | Adam, **un `lr` para todos los brazos**, congelado en el ensayo (§0 paso 4) | un `lr` por `W` sería otro confound |
| pérdida | L1 sobre las 4 coordenadas normalizadas | robusta con `n = 100` |
| aumento / parada temprana / selección | **ninguno** | el sobreajuste es lo que se quiere ver; elegir por `val` contaminaría la medida |
| semillas | **5** por brazo (decisión 3) | `SE = sd/√5`; 10 doblan el coste |
| evaluación | `train` cada época; **las 900** cada 10 épocas y al final, por lotes de 100 (24 s cada vez a `W = 128`, medido) | la curva del propio conjunto de validación, sin usarla para decidir |
| orden de los lotes | el mismo para todos los brazos de una semilla | lo único de la semilla que puede ser común entre brazos |
| máquina | **todos los brazos de una semilla, en la misma máquina** | deriva dev↔Vast medida hoy: 0,0012 de IoU |
| cada corrida | **un proceso `python -u nn/entrenar_local.py --brazo …`** | el freno casa ese nombre en la línea de comando; importarlo lo haría invisible |

## 5. Coste y reloj — MEDIDO el 2026-10-01

Medido en el dev (2 vCPU, torch 2.14.1+cpu, 2 hilos) con pesos aleatorios, lote 20, 3 pasos
tras 1 de calentamiento; el coste no depende de los valores de los píxeles. **Sin optimizar
nada** (conv por defecto de torch; no se probó FFT ni `channels_last`).

| configuración | ms por paso | **1000 pasos** | eval de 900 imágenes |
|---|---:|---:|---:|
| `L=4 f=0,25 C=8` **W=128** (n=32) | **5.162** | **86 min** | 24,0 s |
| `L=4 f=0,25 C=8` W=64 (n=16) | 352 | 5,9 min | 1,6 s |
| `L=4 f=0,25 C=8` W=32 (n=8) | 15 | 0,3 min | 0,2 s |
| `L=4 f=0,25 C=8` W=16 / W=8 | 7 / 6 | 0,1 min | 0,1 s |
| `L=4 f=0,25` **C=4** W=128 | 2.619 | **44 min** | *(no medido; menos)* |
| `L=4 f=0,25` C=6 W=128 | 3.788 | 63 min | — |
| `L=4 f=0,25` C=16 W=128 | 12.378 | 206 min | 47,9 s |
| `L=4 f=0,1 C=8` W=128 (el encargo literal) | 2.646 | 44 min | 18,1 s |
| `L=5 f=0,2 C=8` W=120 | 4.386 | 73 min | — |
| `L=10 f=0,1 C=8` W=120 | 2.423 | 40 min | — |

**Lo que domina es `W = 128`**: el 97 % del reloj, y el control cuesta lo mismo que él.
Derivado de la tabla, con la evaluación de las 900 cada 10 épocas (20 veces):

| opción | reloj | coste |
|---|---|---|
| **Vast, `C = 8`**, 5 máquinas (una por semilla, sus brazos en serie) | ≈1,7 h de trabajo por máquina al ritmo del dev + 2 min de arranque → **≈2 h de reloj**; con el control, **≈3,5 h** | 5 × ≈1,8 h × 0,049–0,062 $/h ≈ **0,4–0,55 $**; con el control **≈0,8–1,05 $** *(estimado)* |
| **dev, `C = 8`** | 5 × (86 + 8) min a 128 + 5 × 6,5 a 64 + ~5 ≈ **8,5 h** de las 2 vCPU que también usa el bot; con el control ≈16 h | 0 $ |
| **dev, `C = 4`** | ≈ **4,5 h** *(la eval a `C = 4` no se midió; estimada a la mitad)*; con el control ≈8 h | 0 $ |

⚠ **La regla del propio repo decide**: *«aquí caben tanteos; lo que entrene de verdad alquila
máquina»*. 4,5–8,5 h de las dos vCPU del server del bot no es un tanteo. Por eso la
recomendación es Vast.

### 5.1 Las tres preguntas del gasto, contestadas antes de que haya nada que lanzar

| | |
|---|---|
| **qué se alquila** | 5 máquinas de Vast (4–6 vCPU, ≥ 8 GB, con GPU que no se usa: en Vast no hay otra forma), una por semilla, etiqueta `expc-dimgen-s<N>`, ≈2 h de vida (≈3,5 h con el control), **tope `--horas-max 5`** |
| **cuánto cuesta** | ≈0,4–0,55 $ sin control, ≈0,8–1,05 $ con él *(estimado: 0,049–0,062 $/h medidos hoy × horas derivadas del paso medido en el dev; si la máquina rinde como en la fase 1 del banco, un 13 % menos)* |
| **quién apaga si este dev muere** | la unidad y su `finally` mueren con él y las de Vast **siguen facturando**. Lo que sobrevive es la **cuenta de Vast**: desde cualquier máquina con el token (el mini, por su bot) `vast_instance.py trabajo --apagar expc-dimgen-`, o desde Telegram `/use exp-vast` → `apagar expc-dimgen-`. El **libro se commitea y empuja al alquilar** (como `bor-k`) para que el server siguiente sepa qué etiquetas eran suyas; `--horas-max 5` acota cada máquina aunque nadie mire; y `cerrable.mjs` ya cuenta las máquinas `expc-*` |

El modo `trabajo` del lanzador está **validado hoy** (fase 1 de `banco-k`); el descriptor se
escribe como el de `bor-k`, con una máquina por semilla en vez de por `k`. **No hace falta
ejecutor nuevo**: `exp-vast` ya frena cualquier `expc-*`.

## 6. Qué sale, dónde, y la regla de reporte que hay que confirmar

Todo en esta carpeta (R7): `nn/pesos/<brazo>-s<semilla>/{metrics.jsonl, summary.json}`,
`resultados/RESULTADOS.md` (regenerado), `resultados/iou-vs-w.png`, `resultados/brecha-vs-w.png`,
y el libro de Vast en `resultados/vast/`. **Los pesos no se commitean** (superan el tope del
repo; el detalle en `REGLAS.md` § Salidas).

⚠ **Sobre el reporte en `estudios-redes-neuronales` hay dos reglas escritas que se
contradicen**, y lo tiene que confirmar el dueño: el `CLAUDE.md` del coordinador y
`reportes/README.md` del central piden *«un reporte por cada barrido, estudio o medición que se
termine, venga de donde venga»*; el `CLAUDE.md` de este repo pide reporte **sólo si mueve
`ESTADO.md`**, y él mismo lo marca como *«propuesta pendiente de confirmar»*. Hoy ningún
experimento de este repo tiene fila en el índice central. La respuesta mecánica previsible es
que **no** mueve `ESTADO.md` (describe parámetros de `foveal-vision`); si alquila, el central
es quien contesta «qué se pagó ya», y entonces la fila con horas, máquinas y coste **sí** hace
falta para eso aunque no mueva ningún parámetro. Recomendado: **fila y reporte si alquila**;
sólo `README.md` si corre en el dev.

## 7. Decisiones que tiene que tomar el dueño ANTES de escribir código

Primero **los cuatro supuestos de §0** (S1–S4). Después:

1. **`L` y, con él, `f = 1/L`** (§3.4): **recomendado `L = 4, f = 0,25`**. Alternativas
   calculadas: `L = 5` (`f = 0,2`, 6 puntos) y `L = 10` (`f = 0,1`, 4 puntos). Si quiere
   mantener `f = 0,1` **con** `L = 4`, los cuatro problemas de §3.2 quedan dentro y hay que
   decir que se aceptan.
2. **`C`**: **`C = 8`** (recomendado si corre en Vast) o **`C = 4`** (si corre en el dev: 4,5 h
   en vez de 8,5).
3. **Semillas**: **5** (recomendado) o 10 (doble coste, `SE` ÷ 1,4).
4. **¿Entra `W = 8`?** Recomendado **sí**: es el único punto donde las cajas bajan de un píxel,
   y «dónde se rompe» es parte de la pregunta. Cuesta 0,1 min.
5. **El control `w128-de16`** (+5 corridas al coste de `w128`: casi dobla el reloj; ≈+0,45 $ en
   Vast). Recomendado **sí en la primera pasada si corre en Vast**: sin él, cualquier «la
   resolución estorba» se escribe como *«sin cerrar (confundido)»*, que es lo que ya pasó dos
   veces (§1). Si corre en el dev, se añade sólo si aparece la lectura (iii), **con las mismas
   semillas**.
6. **Dónde corre**: **Vast** (≈0,4–1,05 $ según el control, ≈2–3,5 h, 5 máquinas; recomendado
   por la regla del repo) o **dev** (0 $, 4,5–16 h compartiendo CPU con el bot). Las dos
   respetan «todos los brazos de una semilla en la misma máquina».

## 8. Riesgos, y lo que este plan NO cubre

- **Parámetros ∝ `W²`**: inherente al diseño del encargo (§3.2). El criterio lo lee como
  «(b) o (c): sin cerrar» salvo que se corra el control (decisión 5).
- **La tarea aguanta el desenfoque** (§2.4): se espera meseta larga y caída por cuantización.
  Es la respuesta para esta tarea, no un fallo; y no dice nada de leer texto.
- **Un `lr` para todos**: si el ensayo no encuentra uno, el plan se para (criterio).
- **1000 pasos**: pueden quedar cortos para los kernels de 32×32. Se mira la curva de `train`;
  si se alarga, se alarga para todos y se repite todo.
- **El dev es el server del bot**: 8,5 h con las dos vCPU ocupadas lo vuelven lento. Es un
  motivo real para Vast, no un detalle.
- **No se ha ejecutado nada**: ni código, ni venv de este workspace, ni una sola corrida. Lo
  único medido son los números de §2.3, el piso, las huellas de `REGLAS.md` y §5, con los
  comandos dichos.
- **Fuera de alcance**: `W > 128` (pediría publicar un dato nuevo a partir de las páginas de
  1024), reducciones que no sean potencias de 2 (interpolar no es el mismo dato), otras tareas
  que no sean la caja, y cualquier cambio a la red de producción.
