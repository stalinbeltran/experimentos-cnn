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
