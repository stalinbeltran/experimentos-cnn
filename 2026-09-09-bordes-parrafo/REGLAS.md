# Reglas de `bor-p`

**Escritas el 2026-09-09** al montar la carpeta y generar el dataset. Su
`experimento.json` declara el estado **`abierto`**: el **dataset está hecho, publicado y
comprobado**, y **nada más**. No hay red, no hay criterio, no hay ninguna cifra medida
sobre kernels — y eso es el encargo, no un hueco: *«Por ahora solo obtén las páginas»*.

⚠ **Estas reglas son de este experimento y de ninguno más.** **No se copió de ningún
experimento** (§ «Qué NO hereda»). Hay cuatro vecinos en el repo y **ninguna de sus
condiciones aplica aquí** — ni siquiera las de `banco-k`, que es a donde se piensa llevar
lo que salga de aquí.

**Qué pregunta:** ¿qué kernel de convolución de `k ≤ 19` localiza los cuatro bordes del
recuadro que encierra un párrafo, sobre páginas de papel limpio con varios párrafos que
nunca se solapan?

**Qué contesta HOY:** nada todavía. Lo que hay es **el dato de entrada** para poder
preguntarlo.

## Entradas

- **Dataset: `parrafos1000-pagina1024-r20260909`**, publicado en
  `foveal-vision-data/experimentos-cnn/`. Se lee con
  `expcnn.exigir_dataset("parrafos1000-pagina1024-r20260909")` **en la primera línea**,
  que se niega antes de empezar si no está publicado (R2). **No se re-deriva al vuelo.**
  - Lo produjo **este mismo experimento** con `nn/generar_paginas.py --parrafos 1000`, una
    vez en la vida. `--rederivar N` sigue existiendo y sigue teniendo que pasar, pero es
    una **prueba de que la receta es honesta**, no una alternativa a publicar: publicado
    es el mismo dato **porque es el mismo fichero**.
- **Qué se lee de él:**
  - `paginas.npz` → `png` + `png_offsets` (las páginas, PNG concatenados), `particion`
    (0=train 1=val 2=eval, **por página**) y `n_parrafos`.
  - `cajas.npz` → `cajas` `(párrafo, 5)` `int32`: `pagina, izq, der, sup, inf`. Las cuatro
    coordenadas **son la etiqueta entera**: no se ignora ninguna.
  - `cajas.npz` → `derivadas` `(párrafo, 3)` `float32`: `margen`, `separacion`,
    `ventana_limpia`. No son etiqueta: son **el presupuesto del recorte** (ver Procesos 2).
  - `meta.json` → los factores de cada párrafo (fuente, cuerpo, interlineado, palabras) y
    **el porqué de cada página descartada**.
- **Condiciones que el dataset trae**, y van aquí aunque estén en el manifiesto porque
  aquí es donde se leen antes de tocar nada:
  - páginas de **1024 × 1024**, gris de 1 canal, **fondo blanco sólido**. **Ningún fondo
    sucio** — es el encargo, y está escrito para que nadie lo «mejore» luego;
  - **de 2 a 5 párrafos por página**, y **ninguno se solapa con otro**;
  - la etiqueta es la **caja de TINTA** (unión de las cajas de palabra), no la de
    maquetación. Por eso `align: justify`: con alineación a la izquierda el borde derecho
    lo marca el final *ragged* de cada línea y la tinta **no llega** al borde de la caja,
    así que ese borde no tendría evidencia visual y el kernel buscaría algo que no está;
  - **texto negro `#000000` fijo**. El nivel de gris **no se varió**, y es una decisión con
    su motivo: el encargo acota el escenario a papel limpio, así que la primera versión no
    mete un factor que nadie pidió. Es el mando obvio para un dataset siguiente.
- **Las tres garantías geométricas, todas derivadas de `k_max = 19`** y ninguna de gusto.
  Viven en `nn/geometria.py`, que las comprueba solo:

  | | mínimo duro | se construye con | de dónde sale |
  |---|---|---|---|
  | separación entre párrafos | **19 px** | 40 px | `2·9+1`: las ventanas 19×19 centradas en dos bordes enfrentados **no se solapan**, y ninguna ve tinta del otro párrafo |
  | margen al borde del papel | **9 px** | 32 px | una convolución `valid` con `k=19` sólo cubre `[9, 1014]`: un borde más cerca del papel **no existe** en la salida |
  | solape | **ninguno** | por construcción | |

  ⚠⚠ **La métrica es L-INFINITO (Chebyshev), no euclídea**, y no es un detalle: una ventana
  **cuadrada** de lado `2r+1` centrada en `p` contiene a `q` si `|px−qx| ≤ r` **y**
  `|py−qy| ≤ r`. Dos cajas separadas 14 px en diagonal están a **19,8** euclídeos —pasarían
  un filtro de 19— y una ventana de 19×19 **sí las mezcla**. Tiene su prueba dedicada en
  `nn/geometria.py`, y es la que distingue las dos métricas.

- **Qué se normaliza o transforma al cargar:** **nada** del dato crudo. Las páginas se
  guardan tal como salieron del render, en gris de 8 bits. Cualquier convolución, recorte o
  estandarización es del experimento que las consuma, y **este experimento todavía no
  declara ninguna** porque no ha llegado a esa parte.

### ⚠⚠ La reserva de `banco-k` (§3.7), que es la única condición que este experimento SÍ hereda

`banco-k` reserva la familia **`LiberationMono`** y el interlineado **`[1.45, 1.60]`** para
uso **exclusivo suyo**. Un kernel aprendido sobre datos que los alcancen tiene **fuga de
distribución aunque las muestras sean distintas**, y al evaluarlo allí sale **optimista**.

**No es una condición del experimento vecino: es una prohibición del sistema**, y por eso
es lo único de `banco-k` que aplica aquí (Regla 0 del repo: lo demás no se hereda).

Este dataset usa las **otras cuatro** familias (`DejaVuSans`, `DejaVuSerif`,
`LiberationSans`, `LiberationSerif`) e interlineado **`[1.10, 1.42]`**, con 0,03 de holgura
bajo la banda reservada.

⚠ **Y se declara EXPLÍCITO en `nn/receta.json` porque omitirlo es filtrar**, no es neutro:

- sin `fonts`, el generador sortea **todas** las familias registradas, incluida la reservada;
- sin `line_height`, su defecto es `Range(1.15, 1.6)`, que **solapa** con `[1.45, 1.60]`.

Es exactamente lo que le pasó a los tres kernels `esqk-*` que hoy están en el banco: su
`receta.json` omite las dos claves, y los tres llevan `usa_la_reserva: true`.

⚠ **Se comprueba en código, no se supone.** `comprobar_reserva()` corre **antes de generar
nada** y se niega si la receta alcanza la reserva; `--reserva` la comprueba sola. Una
receta es un fichero editable, y un descuido ahí no falla por ningún lado: sólo hace
optimista cualquier kernel que salga de aquí.

## Salidas

- **El dataset**, que es la única salida que existe hoy: se **publica** en
  `foveal-vision-data/experimentos-cnn/parrafos1000-pagina1024-r20260909/` y **nunca se
  copia a este repo** — éste es público y aquél privado, y el `.gitignore` de la raíz ya
  excluye `*.npz`.
- **Etapa local:** `datos/` (ignorada por git): `paginas/*.png` según se renderizan,
  y luego `paginas.npz`, `cajas.npz`, `meta.json`, `manifiesto.json`.
- **Figuras:** `muestras/paginas-<n>.png`, una rejilla de páginas con **la caja de cada
  párrafo dibujada encima**. Se regenera con `nn/generar_paginas.py --muestras N`. Es la
  comprobación que ningún número da: que la caja abraza la tinta por los cuatro lados.
- **Pesos: ninguno**, y no por decisión sino porque **no hay entrenamiento todavía**.
  Cuando lo haya, la regla del proyecto sigue siendo que los pesos de una red sólo se
  guardan si el dueño lo ordena.
- **Métricas: ninguna.** Nada se ha medido sobre kernels.
- **Qué se commitea y qué no:** se commitean el código, las reglas y las figuras. **No** se
  commitea `datos/` ni ningún `.npz`.

## Procesos

Los pasos, en orden. **Hoy sólo está hecho el 1.**

1. ✅ **Generar y publicar el dataset**, una vez en la vida, con su `manifiesto.json` y la
   huella SHA-256 de cada fichero. Dentro de este paso, y en este orden:
   1. **La reserva del §3.7 se comprueba ANTES de renderizar** (`comprobar_reserva()`). Si
      la receta la alcanza, no se genera nada: 20 minutos de renders que producen un dato
      contaminado son peores que un error inmediato.
   2. **El reparto en celdas GARANTIZA la separación**, no la busca. `nn/geometria.py`
      parte el papel en celdas de guillotina y mete cada párrafo `SEPARACION/2` hacia
      dentro de la suya; dos párrafos de celdas distintas están separados por al menos una
      línea de corte, así que quedan a `SEPARACION` o más **por construcción**.
      ⚠⚠ **NO se sortea-y-rechaza**, y es obligatorio. Rechazar elimina selectivamente los
      párrafos **grandes** —los que menos hueco libre encuentran— y el tamaño es justo el
      factor que el encargo manda variar. El resultado sería un dataset sesgado hacia
      párrafos pequeños **sin que nada falle**.
      ⚠ **Y el `avoid_overlap` del generador NO se usa**: su propio código dice que el
      margen es *«a preference, not a hard constraint»* (`resolver.py:394`) y que cuando no
      cabe nada conserva *«el reparto menos malo en vez de fallar»* (`SAMPLE_FORMAT.md` §5:
      ~1 % de solape). El encargo pide que **nunca** se solapen, y una preferencia no puede
      dar un nunca.
   3. **Cada página se renderiza DOS veces**, y no es desperdicio. El alto de un párrafo
      **no se puede pedir**: emerge del ancho, la fuente, el cuerpo, el interlineado y el
      número de palabras. Así que el pase A apila los N párrafos en `(0,0)` y **mide** su
      caja real, y el pase B los coloca cada uno donde toca **para esa caja medida**.
   4. **El descarte existe, y es la RED DE SEGURIDAD, no el mecanismo.** Cada página
      renderizada se revisa contra la tinta **real** (`geometria.revisar()`), y la que falle
      **se tira entera** y se reintenta. Con la construcción del paso 2, el descarte real es
      **0** — que es como tiene que ser: si empezara a descartar, sería señal de que la
      construcción está rota, no de que el filtro trabaja.
   5. **Publicar es irreversible.** Un dataset publicado **no se reescribe nunca** y
      `--publicar` se niega a pisar uno existente: dato nuevo es **nombre nuevo**
      (cambia la `r<fecha>`).
2. ⛔ **Recortar las ventanas de entrenamiento.** El encargo lo deja explícitamente para
   después: *«Las ventanas de entrenamiento serán recortadas posteriormente»*.
   **El presupuesto de ese recorte ya está calculado y viaja con el dato**: `ventana_limpia`
   de `cajas.npz` es, por párrafo, el lado **impar** de la ventana más grande que se puede
   centrar en **cualquier** punto de su borde sin salirse del papel y sin ver tinta de otro
   párrafo. Quien recorte no tiene que re-derivarlo ni mirar los píxeles.
   ⚠ **Y el reparto train/val/eval ya está hecho POR PÁGINA**, no por párrafo. Es una
   decisión de este experimento y va aquí con su motivo: dos ventanas de la **misma** página
   comparten fuente, fondo y vecinos, así que repartirlas entre `train` y `eval` sería una
   fuga que **no falla por ningún lado** y que sólo se ve como un resultado demasiado bueno.
   Repartiendo páginas, cualquier recorte posterior **hereda** el reparto y no puede filtrar.
3. ⛔ **La red, sus brazos y sus semillas.** Nada decidido.
4. ⛔ **El criterio, escrito ANTES de mirar** (R13), en `instrucciones/02-criterio.md`. No
   existe todavía **a propósito**: escribirlo antes de saber qué se va a medir sería
   rellenarlo, y un criterio rellenado no distingue «no hubo señal» de una racionalización.
5. ⛔ **Informe.** Y si esto llega a mover lo que `ESTADO.md` dice de algún parámetro, su
   reporte va al repo **central** con su fila; si no, se queda aquí con su `README.md`.

- **Qué se mide, y con qué umbral:** todavía nada. Lo único con umbral hoy son las tres
  garantías geométricas de arriba, y son del **instrumento**, no un hallazgo.
- **Cuántos brazos y cuántas semillas:** sin decidir.
- **Qué se llama «ganar»:** sin decidir. Cuando se decida, se escribe en
  `instrucciones/02-criterio.md` **antes** de mirar ningún número.

## Scripts

Los de este experimento, con su interfaz exacta. **Los nombres y las banderas son de aquí.**

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/geometria.py` | el reparto en celdas, la colocación y **las aserciones**. Aritmética pura: ni renderiza ni importa nada del repo | `python nn/geometria.py` (27 comprobaciones) |
| `nn/generar_paginas.py` | genera, empaqueta, comprueba y **publica** el dataset | `--parrafos 1000` · `--publicar` · `--comprobar` · `--rederivar N` · `--muestras N` · `--reserva` |
| `nn/receta.json` | los factores del generador **y el contrato anti-fuga del §3.7** | lo lee `nn/generar_paginas.py`, y `banco-k/nn/importar_kernel.py` |

- ⚠⚠ **`generar_paginas.py` se llama así por el FRENO, no por estética.**
  `telegram-coordinator/scripts/cerrable.mjs` casa `generar_paginas\.py` en su lista
  declarada `TRABAJOS`, y **entró ahí en el mismo commit que este experimento**. Sin eso,
  el veredicto *«¿se puede apagar este server?»* diría «nada corriendo» durante los ~11 min
  de renders — y un dataset a medias **no es reanudable**: se vuelve a pagar entero.
  ⚠ Y **no** se llama `datos.py` a propósito: ese nombre existe en varios repos y casarlo
  en el freno daría 🔴 permanentes, que es el aviso que se deja de leer. **Dos tests** en
  `telegram-coordinator/tests/cerrable-procesos.test.mjs` fijan las dos mitades —que la
  generación cuente, y que un `datos.py` cualquiera **no**—; el primero **falla con el
  código anterior**.
- ⚠ **`receta.json` MANDA, y se comprueba que mande.** Sus rangos no son decoración: al
  empaquetar se verifica que **todo lo generado cae dentro** y, si no, se niega a publicar.
  Hizo falta: la primera versión declaraba `width: [120, 460]` mientras se generaban
  párrafos de **887 px**, porque `ancho` se deriva de la celda y nadie leía ese rango. Un
  rango que no lee nadie es decoración que se lee como especificación (R15).
- **Dependencias:** **el venv del generador**, `~/src/image-text-sample-generator/.venv`
  (numpy, pillow, playwright + Chromium). No hace falta torch para nada de lo que hay hoy.
  La puerta es `expcnn.exigir_generador()`, que se niega si el repo no está.
  ⚠ `nn/geometria.py` **no necesita ni eso**: es stdlib pura y corre en un clon limpio.
- **De dónde sale el código:** **autónomo**. No importa nada de `foveal-vision`
  (`usa_fv: false`) ni de ningún otro experimento. `nn/geometria.py` no importa nada en
  absoluto.

## Qué NO hereda

- **Se copió de:** **no se copió de ningún experimento.** La carpeta se montó desde cero el
  2026-09-09 a partir del encargo del dueño (`instrucciones/01-encargo.md`).
- **Qué se decidió al montarlo**, y que nadie más dice:
  - **la página es de 1024 px y lleva de 2 a 5 párrafos.** Los cuatro vecinos trabajan con
    **una** figura por imagen; aquí el encargo pide explícitamente varios párrafos que no se
    solapen, así que la unidad de dato es la **página** y la unidad de etiqueta el
    **párrafo**;
  - **las páginas se guardan como PNG dentro del `.npz`**, no como un array. Un array
    `(P, 1024, 1024)` `uint8` son ~300 MB en RAM y esta máquina tiene 3,8 GB; y con PNG en
    disco desde el primer render, una caída a los 10 minutos no pierde nada. PNG es **sin
    pérdida** y `--comprobar` lo demuestra casando la huella del array **decodificado**;
  - **el reparto es por página** (ver Procesos 2);
  - **`gasta: "no"`**, que es lo honesto hoy —no hay entrenamiento—, con el freno cubierto
    por el nombre del script en vez de por el campo. Cuando haya entrenamiento pasa a
    `entrena-local` y su entrada **tiene que** llamarse `nn/entrenar_local.py`.
- **Restricciones de otros experimentos que NO aplican aquí**, nombradas porque son de las
  que se cuelan solas:
  - de **`banco-k`**: ni el lienzo de 584, ni la reducción **/4**, ni un párrafo por imagen,
    ni las particiones 100/100/800, ni las 10 semillas, ni la cabeza de 68 parámetros, ni el
    IoU como métrica titular, ni el descarte de 9 px por lado como **transformación del
    dato** (aquí los 9 px son un **cálculo** que fija el margen, y las páginas se guardan
    enteras y **pre-kernel**). Lo único suyo que aplica es la **reserva del §3.7**, y porque
    es una prohibición del sistema;
  - de **`esq-k`/`esq-2d`/`esq-cq`**: ni las ventanas de 32 px, ni la etiqueta de esquina,
    ni el `f1`, ni el barrido de `k` como brazos. Aquí la etiqueta son **cuatro bordes en
    píxeles de la página** y `k` **no es todavía la variable**.
- **Contra qué se compara, y qué haría que dejara de ser comparable:** hoy, **contra nada**:
  no hay ninguna medida. Lo que sí está fijado es que cualquier medida futura sobre este
  dataset deja de ser comparable si se toca **el dataset**, y el dataset **no se toca**: dato
  nuevo es nombre nuevo.
  - ⚠ **Y lo que rompería la comparabilidad con `banco-k`** —que es a donde van los kernels—
    **no es cambiar algo de aquí, sino alcanzar su reserva.** Un dataset siguiente que
    variara el gris del texto, el tamaño de la página o el número de párrafos seguiría
    siendo válido para el banco; uno que usara `LiberationMono` **no**, por bonito que fuera.
