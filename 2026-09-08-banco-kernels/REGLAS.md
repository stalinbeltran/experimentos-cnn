# Reglas de `banco-k`

**Escritas el 2026-09-08**, al montar la carpeta, a partir de la
[especificación v1.0 del dueño](ESPECIFICACION.md) (fechada 2026-09-07). Su
`experimento.json` declara el estado **`abierto`** y **nada se ha corrido**: no hay dataset
publicado, no hay pesos y no hay ninguna cifra medida de este experimento.

⚠ **Estas reglas son de este experimento y de ninguno más.** **No se copió de ningún
experimento** (§ «Qué NO hereda»), así que no hay condiciones heredadas que releer — pero sí
hay tres experimentos vecinos que miden esquinas en ventanas de 32 px, y **ninguna de sus
condiciones aplica aquí**. Lo que manda es `ESPECIFICACION.md`.

⚠⚠ **La especificación es la fuente de verdad, y estas reglas NO la reemplazan.** Donde estas
reglas y la especificación digan cosas distintas, gana la especificación y estas reglas están
mal. Lo único que estas reglas añaden es **qué se decidió al aterrizarla en este repo**, que es
lo que la especificación no puede saber.

**Qué pregunta:** dado un kernel `k × k` cualquiera, aplicado a las entradas antes de la red,
¿mejora el IoU sobre `eval` por encima del **kernel aleatorio de igual norma**, y además
**reduce la brecha `train − eval`** respecto de la identidad (transferencia) en vez de sólo
subir las dos juntas (facilitación)?

## Entradas

- **Dataset:** **ninguno todavía.** `experimento.json` declara `"dataset": null` a propósito, y
  eso no es un hueco: es el estado real. El dataset que pide la especificación (§3) **no
  existe** en `foveal-vision-data/experimentos-cnn/`, donde hoy sólo hay tres datasets de
  ventanas de 32 px que **no sirven** para esto (*comprobado el 2026-09-08*: `ls` del repo de
  datos, y ninguno es de párrafos a 146 × 146).
  - El nombre previsto es **`parrafos1000-584px-r4-rAAAAMMDD`** (`dataset_previsto` en el
    manifiesto). La `r<fecha>` es la época de render y se rellena **al publicar**.
  - ⚠ **`dataset_previsto` no lo lee ningún script y no es un segundo mando** (R15): el mando
    es `dataset`, y mientras sea `null` la puerta `expcnn.exigir_dataset(...)` no se puede
    llamar. Al publicar se mueve el nombre a `dataset` y `dataset_previsto` desaparece.
  - Cuando exista se lee con `expcnn.exigir_dataset("<nombre>")` **en la primera línea**, que
    se niega antes de empezar si no está publicado (R2). **No se re-deriva al vuelo.**
- **Qué se lee de él:** las tres particiones `train` (100) · `monitor` (100) · `eval` (800), y
  para cada muestra la imagen **reducida a 146 × 146** en escala de grises y **las cuatro
  coordenadas enteras de la caja en el marco de 584** (`borde_izq`, `borde_der`, `borde_sup`,
  `borde_inf`). No se ignora nada: las cuatro coordenadas son la etiqueta entera.
  - ⚠ **Se guarda el marco de 146, NO el de 128.** El recorte a 128 depende del kernel (§6.2),
    así que un dataset ya recortado impediría aplicar cualquier kernel. El dataset es
    **pre-kernel** por diseño.
  - ⚠ **Las etiquetas se guardan en el marco de 584**, sin transformar, porque la
    transformación `(coord/4) − 9` es parte del **pipeline** (§6.4) y no del dato.
- **Condiciones que el dataset tiene que traer** (§3, y van aquí aunque estén en el
  manifiesto, porque aquí es donde se leen antes de tocar nada):
  - párrafo de tinta negra sobre **fondo blanco uniforme**, 1 canal, generado a **584 × 584**;
  - **reducción /4 por promedio por área** → 146 × 146. **No /8**: a /8 el párrafo es un
    rectángulo gris casi uniforme y todas las condiciones convergen (§3.2);
  - **la caja del párrafo contenida en `[68, 512]` en los dos ejes del marco de 584** (§3.3).
    Es **obligatoria** y va como **aserción**, no como supuesto: el pipeline descarta 9 px por
    lado y una caja fuera del marco final deja una coordenada **inalcanzable** para el
    soft-argmax, o sea un error irreducible que desplaza el IoU medio. Con 1000 muestras, 20
    mal colocadas ya se notan.
    ⚠⚠ **Y la receta NO puede darlo: `placement.area` acota sólo la esquina superior-izquierda**
    (*leído del código del generador y medido en 6 renders el 2026-09-08*), así que hay que
    **descartar y aseverar** — y descartar **dos** cosas, no una: caja fuera de `[68, 512]`
    **y caja fuera del lienzo de 584**, porque un párrafo cortado da una etiqueta que no
    describe lo que se ve. Detalle en `instrucciones/01-encargo.md` § «Lo que se comprobó»;
  - reparto **estratificado** (§4.1) según lo que el generador varíe.
- **Qué se normaliza o transforma al cargar:** nada del dato crudo. Todo lo que se le hace a
  la entrada es el **pipeline** (§6), que es idéntico para las tres particiones: convolución
  `valid` con el kernel → recorte central a 128 × 128 → **estandarización con la media y la
  desviación de `train` filtrado** (§6.5). Los estadísticos salen **sólo de `train`** y se
  aplican a las tres.
  - ⚠ **La estandarización es un CONTROL, no una variable**: no se corren condiciones con y sin
    ella (§6.5). Sin ella, un kernel de suma positiva (tipo Gauss) hereda la media alta del
    fondo blanco y un kernel de suma cero (tipo Sobel) no, y esa diferencia de distribución se
    contaría como calidad del kernel.
- **El kernel que se evalúa** entra como dato por el contrato de la especificación §5:
  `.npy`, forma `(k, k)`, `float32`, `k` **impar** con `3 ≤ k ≤ 19`, un canal. El banco lo
  **normaliza en norma L2** al recibirlo (`k / sqrt(sum(k**2))`), así que la única diferencia
  entre dos kernels es su **forma**. Los kernels a evaluar van en `kernels/`.
  - ⚠ **El banco es agnóstico al origen del kernel** (§1): de dónde salió no es asunto de este
    experimento, y los métodos para obtenerlos están **fuera de alcance** (§15).

## Salidas

Nada de esto existe todavía. Es dónde va a caer, declarado antes de producirlo.

- **Pesos:** `nn/pesos/<condicion>-s<semilla>/best.pt` y `last.pt`.
  - ⚠ **Con el tope del repo (≈5 MB por experimento) esto NO cabe entero y hay que decidirlo
    antes de correr**: 5.812 parámetros son ~23 KB por checkpoint, pero el banco tiene ~5
    condiciones × ≥5 semillas × 2 ficheros ≈ 50 checkpoints ≈ 1,2 MB *(calculado, no medido)* —
    eso sí cabe. Lo que no cabe es la calibración (§11) con 10 semillas si además se guardan sus
    pesos: **la calibración no guarda pesos**, sólo métricas, porque es puesta a punto del
    instrumento y sus resultados no se reportan como hallazgos (§11).
- **Métricas:** `resultados/<condicion>/metricas.csv`, una fila por **(semilla × parada ×
  métrica)**. Las paradas son **25, 50, 100 y 200** y en cada una se registra, **para cada
  semilla**: IoU en `train`, IoU en `monitor`, IoU en `eval`, brecha `train − eval`, y MAE por
  borde sobre `eval` (§9.2).
  - ⚠⚠ **El IoU de `train` NO es opcional.** Sin él no se distingue **facilitación** de
    **transferencia** (§2.3), que es la distinción central de este banco. Es la trampa §14 con
    su propia fila.
- **Resumen y criterios:** `resultados/<condicion>/resumen.json` (media ± desviación **entre
  semillas** por parada) y `resultados/<condicion>/criterios.json` (la evaluación de §2.1 y
  §2.2, por separado).
  - ⚠ **Toda cifra se reporta como media ± desviación entre semillas. Una cifra de una sola
    semilla no es un resultado en este banco** (§9.3).
- **Trazabilidad:** `resultados/<condicion>/config.json` con el kernel, su `k`, **su norma
  original antes de normalizar**, su hash **antes** de normalizar, las semillas, el hash del
  dataset y la **versión de la especificación** (§13.3).
- **Figuras:** ninguna es obligatoria. Si se hacen, van a `resultados/figuras/` y su comando de
  regeneración se anota aquí en el mismo commit.
- **Qué se commitea y qué no:** se commitean métricas, resúmenes, criterios, configs y los
  `kernels/*.npy` evaluados. **No** se commitea `datos/` (etapa local de render) ni ningún
  `.npz`: el `.gitignore` de la raíz del repo ya excluye `*.npz` porque **este repo es público
  y el de datos es privado**. El dataset **se publica** en `foveal-vision-data`, nunca se copia
  aquí.

## Procesos

Los pasos, en orden. **Hoy sólo está hecho el 0.**

0. ✅ **Montar la carpeta y dejar la especificación a salvo.** `ESPECIFICACION.md` es la copia
   **verbatim** del artifact del dueño. Se rescató porque un artifact es una URL y esta máquina
   se rehace sin aviso: *lo que no está empujado, no existe*.
1. ⛔ **Cerrar las tres decisiones bloqueantes** (`bloqueado_por` en `experimento.json`, y el
   detalle en [`instrucciones/01-encargo.md`](instrucciones/01-encargo.md)). Las tres tocan
   **invariantes** (§12) o un dataset publicado, y las dos cosas son irreversibles.
2. ⛔ **Generar y publicar el dataset** (§3, §4). Una vez en la vida, con su `manifiesto.json`
   y la huella SHA-256 de cada partición. Las **aserciones** de §3.3 (caja dentro de
   `[68, 512]` **y dentro del lienzo**) y §6.4 (la transformación de coordenadas sobre una
   muestra conocida) van en el código de generación, no en un comentario.
   ⚠ **Se piden más de 1000 imágenes para conseguir 1000 válidas** (descartes por caja fuera,
   por solape y por render sin bloque). **Cuántas más no está medido**, y medirlo sale gratis
   aquí: el contador de descartes por motivo va en el `manifiesto.json`.
   ⚠ **Publicar es irreversible**: un dataset publicado **no se reescribe nunca**, y dato nuevo
   es nombre nuevo. Por eso va después del paso 1 y no antes.
3. ⛔ **Escribir el criterio operativo antes de mirar** (R13) en
   [`instrucciones/02-criterio.md`](instrucciones/02-criterio.md). Ya está escrito lo que la
   especificación §2 fija; lo que falta es el número de semillas, que **lo fija la calibración**.
4. ⛔ **Calibrar el banco** (§11), que **no es un experimento** y cuyos resultados **no se
   reportan como hallazgos**:
   1. **caja media primero, antes de cualquier kernel.** Si ese predictor trivial saca un IoU
      alto, el generador coloca los párrafos con poca variabilidad, todas las condiciones
      quedan comprimidas y el banco no discrimina nada. **Si sale alto se corrige el generador
      y NO se continúa** (§10.1);
   2. **identidad y aleatorio con 10 semillas**, para medir la desviación real del IoU;
   3. **fijar el número definitivo de semillas** a partir de ese dato (no por defecto);
   4. **verificar la resolución de coordenada** (§7.5): si el MAE se estanca cerca de **8 px**,
      quitar el stride de la tercera convolución (rejilla 32 × 32) **antes de tocar cualquier
      otra cosa** de la arquitectura, y **reiniciar la calibración**.
   Concluida la calibración, los parámetros quedan **congelados**.
5. ⛔ **Correr las condiciones** con el protocolo de §8, idéntico para todas.
6. ⛔ **Informe.** Y **la decisión de si esto lleva reporte al repo central** se toma con la
   pregunta mecánica del repo: ¿cambia lo que `ESTADO.md` dice de algún parámetro? Este banco
   no barre un parámetro de la red de producción, así que **por defecto se queda en su carpeta
   con su `README.md`** — salvo que un kernel resulte útil y eso mueva algo del central.

- **Qué se mide, y con qué umbral:** IoU de la caja predicha contra la real (**titular**), MAE
  por borde en píxeles (**diagnóstico**) y la brecha `IoU_train − IoU_eval` (**criterio 2.2**).
  El criterio completo escrito **antes de mirar** está en `instrucciones/02-criterio.md`; en una
  línea: **un kernel es útil si supera al aleatorio de igual norma y mismo `k` por más que la
  suma de las desviaciones entre semillas de las dos condiciones**, y **muestra generalización
  si además su brecha es menor que la de la identidad bajo el mismo margen**.
- **Cuántas condiciones y cuántas semillas:** condiciones de control **caja media · identidad ·
  aleatorio** (obligatorias) y **gauss · sobel** (opcionales, §10), más un brazo por kernel
  evaluado. **Semillas ≥ 5, idénticas en todas las condiciones**, y el número definitivo lo
  fija la calibración (§8.5). El **aleatorio necesita varias semillas de aleatoriedad**, no una:
  su desempeño también varía (§10.2).
- **Qué se llama «ganar»:** **no se declara un ganador**; se reporta si cada kernel cumple
  **2.1 y 2.2 por separado**. Un kernel que cumple 2.1 y falla 2.2 es un **resultado válido** y
  se registra como tal (§2.3). La cadena que el banco permite sostener es
  `kernel > aleatorio > identidad > caja media`, y **cada `>` tiene que superar la desviación
  entre semillas, o no es un `>`** (§10.3).
  - ⚠ **Si el kernel supera a la identidad pero no al aleatorio, la conclusión es «filtrar
    funciona», no «el kernel funciona»** (§10.3). Son afirmaciones distintas.
- **Lo que este experimento NO hace** (§15): no propone métodos para obtener kernels, no
  optimiza la CNN de referencia, no admite kernels multicanal, y **no compara eficiencia por
  parámetro** entre kernels de distinto `k` (comparar `k=3` con `k=19` es comparar 9 con 361
  parámetros: legítimo bajo el objetivo declarado, pero no es una comparación de eficiencia).

## Scripts

Los de este experimento, con su interfaz exacta. **Los nombres y las banderas son de aquí.**

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/modelo.py` | La CNN de referencia (§7), **autónoma**: sólo importa `torch`. Trae la comprobación de sus invariantes | `python nn/modelo.py` → imprime la cadena de dimensiones y comprueba 5.812 / 68 / rejilla 16 × 16 |
| `nn/entrenar_local.py` | La **entrada declarada**. Hoy **se niega a entrenar** porque no hay dataset publicado, y dice qué falta | `python nn/entrenar_local.py` (se niega, código 2) · `--comprobar` (comprueba la arquitectura sin dataset, código 0) |
| `nn/receta.json` | La receta de render 584 × 584 con el área de colocación de §3.3. **Dato, no script** | la lee el generador de dataset del paso 2 |

- ⚠ **`entrenar_local.py` se llama así porque es un contrato, no por estética.** `experimento.json`
  declara `gasta: "entrena-local"` y el freno del coordinador (`cerrable.mjs`) casa **ese
  nombre exacto** en su lista de trabajos: con otro nombre, el veredicto *«¿se puede apagar este
  server?»* diría «nada corriendo» con un barrido vivo. `comprobar.py` del repo lo verifica.
- ⚠ **Y existe hoy, negándose, a propósito.** No es un hueco pendiente: es la regla 4 de
  escritura del proyecto —*el freno nunca llega después del acelerador*— cumplida por
  adelantado. El nombre visible para el freno está puesto **antes** de que haya algo que gastar
  tiempo, en vez de añadirse en el commit que lo estrena.
- **Qué falta por escribir** (y se declara para que su ausencia no se lea como olvido):
  `nn/datos.py` (generar, verificar y publicar el dataset), `nn/pipeline.py` (kernel + recorte +
  estandarización), `nn/evaluar.py` (IoU, MAE por borde, brecha) y `nn/informe.py`. Sus
  interfaces se anotan **en esta tabla** en el mismo commit en que se escriban.
- **Dependencias:** el venv del repo, `~/src/experimentos-cnn/.venv` — **torch 2.14.0+cpu y
  numpy 2.5.3** *(comprobado el 2026-09-08)*. No hace falta nada más para entrenar. Para
  **generar** el dataset hace falta además el generador de párrafos, que trae su propio venv
  (`~/src/image-text-sample-generator/.venv`, con Playwright).
  - ⚠ **Son dos venvs y es a propósito**: el generador arrastra Playwright y un navegador, y la
    generación pasa **una vez en la vida**. La puerta al generador es
    `expcnn.exigir_generador()`, que se niega si no está.
- **De dónde sale el código:** **autónomo**. `nn/modelo.py` no importa nada de este repo ni de
  `foveal-vision` (`usa_fv: false`), que es lo único que garantiza poder cargar estos pesos
  dentro de un año. Ningún experimento importa de otro: si algún día hace falta algo del
  vecino, **se copia**.

## Qué NO hereda

- **Se copió de:** **no se copió de ningún experimento.** La carpeta se montó desde cero el
  2026-09-08 leyendo `ESPECIFICACION.md`, que es un documento del dueño y no un experimento de
  este repo. Lo único que se tomó del repo son sus **dos obligaciones** (`experimento.json` y
  este `REGLAS.md`) y el **contrato de nombre** con el freno.
- **Qué se cambió a propósito:** nada, porque no hay original del que cambiar. Lo que sí se
  **decidió** al aterrizar la especificación, y que la especificación no dice:
  - **`dataset: null`** en vez de inventar un nombre: `comprobar.py` verifica que el dataset
    declarado esté **publicado de verdad**, así que un nombre puesto por adelantado sería un
    fallo del preflight — o peor, un nombre que parece un dato y no lo es;
  - **`gasta: "entrena-local"`** desde el primer commit, con su `entrenar_local.py` negándose,
    en vez de `gasta: "no"` hasta que hubiera algo que correr;
  - **la especificación se copia al repo verbatim** en vez de sólo enlazar el artifact;
  - **`nn/modelo.py` implementa el padding que reproduce las dimensiones escritas** en §7.1
    (64 → 32 → 16), y **deja la contradicción anotada y conmutable** en vez de elegir en
    silencio. Ver la § siguiente y `instrucciones/01-encargo.md` § P1.
- **Qué se conservó, y por qué se decidió conservarlo:** **todo lo que dice la especificación**,
  sin excepción — que aquí no es inercia sino el encargo: §12 declara **invariantes** y §1.2 da
  el criterio único del que sale cada decisión (*maximizar la sensibilidad de la medición al
  kernel*). En concreto se conservan, habiéndolo decidido y **pudiendo haberse aflojado**:
  - **100 muestras de `train` y 80 % en `eval`.** Es un régimen artificialmente duro **a
    propósito** (§8.3): es donde el aporte del kernel es medible;
  - **cabeza de 68 parámetros.** Una cabeza grande **compensa un kernel malo** y comprime las
    diferencias hasta el ruido entre semillas (§7.2, §14);
  - **sin aumento de datos, sin parada temprana, sin schedule de `lr`.** El aumento es en sí
    mismo un mecanismo de generalización y competiría con el kernel por el mismo efecto (§8.3);
    la parada temprana **evitaría el régimen de sobreajuste**, que es justo lo que hay que
    observar porque la brecha `train − eval` **es** la evidencia del criterio 2.2 (§8.4);
  - **`K_max = 19` y el descarte fijo de 9 px por lado.** Si cada `k` fijara su propio tamaño
    de salida, las condiciones verían **distinta cantidad de imagen** y la comparación mezclaría
    calidad del kernel con campo de visión (§6.2). Si algún día hace falta `k > 19`, **no se
    eleva `K_max`: se construye un banco nuevo** (§5.2).
- **Contra qué se compara, y qué haría que dejara de ser comparable:** **sólo consigo mismo.**
  Cada kernel se compara contra los **controles corridos en este mismo banco, con las mismas
  semillas, el mismo dataset y la misma arquitectura** (§8.2). **No se compara con ningún otro
  experimento de este repo** ni con nada de `foveal-vision`: nadie más mide esto, sobre este
  dato, con esta red.
  - Dejaría de ser comparable con sus propias mediciones previas si se toca **cualquier
    invariante de §12**: `K_max` y el descarte de 9 px, la entrada 128 × 128 × 1, la
    arquitectura y su recuento de parámetros, las particiones y su contenido, el conjunto de
    semillas, los hiperparámetros y las épocas, la definición de las métricas, o los criterios
    de §2. **Tocar uno obliga a un banco nuevo con su propia serie**, no a re-etiquetar el viejo.
- **Restricciones de otros experimentos que NO aplican aquí** (nombradas porque son de las que
  se cuelan solas — los tres vecinos miden **esquinas en ventanas**, y esto mide **una caja
  entera en la página**):
  - **ventana de 32 px, reducción a un mapa pequeño, kernel único aprendido:** aquí la entrada
    es **128 × 128** y el kernel **no se aprende, entra como dato ya hecho**;
  - **`k` barrido de 5 a 17 como brazos del experimento:** aquí `k` **no es la variable**. La
    variable es **qué kernel**, y su `k` viene dado por el kernel que llegue (3 ≤ k ≤ 19,
    impar);
  - **etiqueta de «hay esquina» + posición, acierto a ≤ 2 px, `f1`:** aquí la etiqueta son
    **cuatro bordes** y la métrica titular es **IoU**, con MAE por borde como diagnóstico;
  - **300 épocas, `lr` 0,05, lote 128:** aquí son **200 épocas, `lr` 3e-4 constante, lote 20**
    (§8.1). La pérdida es **L1**, no L2, porque con n=100 unas pocas muestras atípicas
    dominarían el gradiente;
  - **cabeza de 3 parámetros y lectura por máximo de un mapa:** aquí la cabeza son **68
    parámetros** (conv 1 × 1) y la lectura es **soft-argmax sobre marginales 1D** con
    `tau = 1.0`, **un canal por borde**.
    ⚠ **Y un canal por borde es crítico, no un detalle:** con un solo mapa de columnas para
    izquierda y derecha la distribución es bimodal, el soft-argmax devuelve la esperanza de los
    dos modos —el **centro** del párrafo— y las dos predicciones **colapsan al mismo valor**
    (§7.4, y su fila en §14).
