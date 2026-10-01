# Reglas de `dim-gen`

**Escritas el 2026-10-01 al especificar el experimento.** Estado **`abierto` y SÓLO PLAN**: no
hay código, nada entrenado, ningún resultado. Lo que se fija aquí es el **contrato** con el
que se escribirá el código; los nombres de scripts y banderas son los **previstos**, y si al
implementar cambian, se cambian **aquí en el mismo commit**. Las decisiones todavía abiertas
van marcadas `⏳ abierta (decisión N)` con su valor por defecto; las toma el dueño
(`ESPECIFICACION.md` §7).

⚠ **Estas reglas son de este experimento y de ninguno más.** No se copió de ningún
experimento. Lee **el mismo dataset** que `banco-k`, y eso no le hereda ni una condición
(§ «Qué NO hereda»). Las tablas derivadas (mapas, parámetros, coste) viven **sólo** en
`ESPECIFICACION.md`; aquí va la regla que las produce.

**Qué pregunta:** con el 10 % del dataset para entrenar y el 90 % para validar, `L` capas
fijas sin padding y un kernel que siempre ve la misma fracción `f` del ancho de la imagen,
¿cómo cambia el IoU sobre las imágenes no vistas —y la brecha train−val— al reducir la imagen
de 128 a 64, 32, 16 y 8 px? ¿Hay un `W` mínimo suficiente, y hay un `W` a partir del cual la
resolución sobra o estorba?

## Entradas

- **Dataset: `parrafos1000-584px-r4-r20260908b`**, publicado en `foveal-vision-data/experimentos-cnn/`
  y leído con `expcnn.exigir_dataset(...)` en la primera línea de `nn/datos.py`. **No se
  regenera y no se publica ninguna variante reducida**: las reducciones son una transformación
  determinista y exacta del publicado, se derivan **en memoria** al cargar, y lo que las hace
  «el mismo dato» es que **su huella está congelada aquí** (abajo) y `datos.py` **se niega** si
  no casa. Por qué éste y no otro: `ESPECIFICACION.md` §2.1.
- **Qué se lee:** `train.npz`, `monitor.npz`, `eval.npz` → `imagenes` `(N, 146, 146) uint16`
  (suma de cada bloque 4×4 del lienzo de 584), `etiquetas` `(N, 4) int32` en el marco de **584**
  (`izq, der, sup, inf`), `indices` (índice de la imagen en la receta). `meta.npz` (JSON por
  imagen: `fuente`, `cuerpo`, `interlineado`, `gris_nivel`, `ancho_real`, `alto_real`, …) **sólo**
  para el desglose por factor del resultado, nunca para entrenar.
- **El reparto 10 / 90:** `train` = las 100 de `train` del dataset. **`val`** = `monitor ∪ eval`
  = 900 (la «validación» del encargo). `monitor` y `eval` **se tratan igual**: aquí no hay nada
  que monitorizar aparte. **Ninguna decisión usa las 900**: ni parada temprana, ni selección de
  pesos, ni elección de `lr`. Si alguna lo hiciera, el reparto dejaría de ser 10/90.
- **Condiciones que el dataset ya trae:** lienzo 584 reducido /4 → 146; el marco útil de 128
  deja 9 px por lado; etiquetas en el marco de 128 en `[8, 119]` (el recorte no corta ninguna
  caja); el reparto está estratificado por área de caja en cuartiles. La **reserva §3.7** del
  banco (`LiberationMono`, interlineado `[1,45, 1,6]`) **no aplica aquí**: este experimento no
  produce ningún kernel y sus pesos **no se importan nunca** al banco. Se anota para que nadie
  la «arregle».
- **Qué se transforma al cargar** (todo en `nn/datos.py`, determinista, sin azar):
  1. **recorte central 146 → 128** (9 px por lado). Motivo propio: 146 = 2·73 no admite más
     reducciones exactas y 128 = 2⁷ sí; y es el marco de las etiquetas, que se transforman con
     `(c/4) − 9`;
  2. **reducción a `W ∈ {128, 64, 32, 16, 8}`** por **suma de bloques `b × b`** con `b = 128/W`,
     acumulando en **`int64`** (a `W ≤ 16` la suma no cabe en `uint16` y desbordaría **sin
     avisar**). **Sólo potencias de 2**: cualquier otro `W` exigiría interpolar y dejaría de ser
     el mismo dato reducido;
  3. **valor de entrada** `x = 1 − suma / (255 · 16 · b²)` = **fracción de tinta** del píxel, en
     `[0, 1]`, papel 0 y tinta 1 — **el mismo significado a todo `W`**. Sin estandarizar por
     `μ, σ`: una `μ, σ` por `W` meterían una diferencia entre brazos que no es resolución;
  4. **etiqueta** = caja normalizada `((c/4) − 9) / 128 ∈ [0, 1]`, **idéntica para todo `W`**:
     el objetivo no cambia, cambian sólo los píxeles de entrada;
  5. **el control `w128-de16`** (⏳ abierta, decisión 5; por defecto **sí** si corre en Vast):
     la matriz de `W = 16` repetida 8×8 hasta 128 (`np.repeat` en los dos ejes), con la red de
     `w128`.
- **Huellas congeladas** *(calculadas el 2026-10-01 sobre los `.npz` publicados)*: `sha256` de
  la matriz **`int64` de sumas** `(1000, W, W)` en el orden `train, monitor, eval` tal como
  vienen, bytes little-endian (`astype('<i8').tobytes()`), 16 hex. `datos.py --comprobar` las
  recalcula y **se niega** si una no casa (R2); además comprueba que la suma total de cada `W`
  es la del recorte (eso es lo que «exacto» significa) y que la de `W = 128` **es** el recorte.

  | `W` | huella | máximo de la suma |
  |---:|---|---:|
  | 128 | `a53caf6cf1d7527c` | 4.080 |
  | 64 | `6ba56079c36e8089` | 16.320 |
  | 32 | `0d9dd6a70fcf496e` | 65.280 |
  | 16 | `db69ac6f091f289b` | 261.120 |
  | 8 | `27e1978cf167f46b` | 1.044.480 |
  | etiquetas normalizadas (`float64`, `'<f8'`) | `36ca489668552b90` | rango `[0,0625, 0,9297]` |

## Salidas

- **Pesos: NO se commitean**, como en `banco-k`. Motivo medido: con `C = 8` los `last.pt` de
  los cinco `W` suman ≈1,1 MB por semilla (275.540 parámetros × 4 B), o sea ≈5,5 MB las cinco
  semillas y ≈6,6 MB con el control: por encima del tope del repo (≈5 MB por experimento). El
  resultado aquí es **la curva**, no los pesos; lo que se querría leer después de los pesos (el
  desglose por factor) va ya en `summary.json`. Se quedan en disco bajo `nn/pesos/` (ignorado
  por git: `*.pt`), regenerables con la semilla en la misma máquina. ⏳ Si el dueño quiere
  conservar alguno (p. ej. `w128`), se dice y se mide el tamaño antes.
- **Por corrida**, en `nn/pesos/<brazo>-s<semilla>/`: `metrics.jsonl` (una línea por época,
  **según ocurre**: pérdida y IoU de `train`; cada 10 épocas y al final, IoU de las 900 y
  brecha) y `summary.json` (IoU_train, IoU_val, brecha finales; IoU_val **por factor**: `fuente`,
  cuartil de `cuerpo`, cuartil de `gris_nivel`, cuartil de área de caja; número de parámetros;
  segundos; máquina (`lscpu`), versión de torch; las huellas de las entradas usadas). **No hay
  `best.pt` a propósito**: elegir una época mirando las 900 contaminaría la medida.
- **Figuras:** `resultados/iou-vs-w.png` (IoU de `val` y de `train` contra `W`, media ± sd, eje
  `log2`; el control como punto aparte) y `resultados/brecha-vs-w.png`. Las regenera
  `nn/informe.py`.
- **Tabla / informe:** `resultados/RESULTADOS.md`, regenerado por `nn/informe.py` de los
  `summary.json` (nunca transcrito a mano), con el criterio de `instrucciones/02-criterio.md`
  aplicado **tabla por tabla**: piso, umbral, `W` mínimo suficiente, «estorba», la clasificación
  (a)/(b)/mixta/ninguna de cada `W`, y el control. El veredicto se pega al `README.md` al cerrar.
- **Si alquila:** el libro de Vast en `resultados/vast/` (un JSON por máquina, paso a paso, con
  coste), **commiteado y empujado al alquilar**.
- **Qué se commitea:** métricas, resúmenes, figuras, informe y libro, siempre. Pesos, no.
  **Nunca** el dataset ni las matrices reducidas (`*.npz` está en el `.gitignore` del repo).

## Procesos

0. **Los 4 supuestos y las 6 decisiones del dueño** (`ESPECIFICACION.md` §0 y §7). Sin ellos no
   se escribe código.
1. **`nn/datos.py --comprobar`**: huellas de las tres particiones contra el `manifiesto.json`;
   las cinco reducciones y el control; **las huellas congeladas de arriba**; aserciones
   (etiquetas en `[0, 1]`; suma conservada; `W = 128` idéntico al recorte); imprime la tabla de
   tamaños de caja por `W` y el piso de caja media (**0,2464**, medido ya).
2. **`nn/modelo.py`**: construye la red para cada `W` y comprueba que el mapa final es `L × L`
   en todos, que los parámetros coinciden con la tabla de `ESPECIFICACION.md` §3.2, y hace un
   forward. **`nn/probar.py`** fija las dos cosas con tests, y además que un acumulador `uint16`
   **hace caer** la comprobación de huellas (R14: lo que puede fallar en silencio, se prueba).
3. **Ensayo de mecanismo** (`nn/entrenar_local.py --ensayo`): `W = 32`, semilla 1, 1000 pasos,
   mirando **sólo** la pérdida de `train` (que baje sin oscilar) → fija `lr`. Después `W = 128`,
   50 épocas, igual. **No se mira `val`.** El `lr` se congela aquí y se escribe en estas
   reglas. Si no hay un `lr` que sirva para los dos, se para y se vuelve al dueño.
4. **Entrenar los brazos**: `nn/vast.sh todo` (⏳ decisión 6; por defecto Vast: una máquina por
   semilla con todos sus brazos) o `nn/lanzar.sh todo` (el dev, una unidad de systemd por
   `desacoplar-persistente.sh`). En los dos casos **cada corrida es un proceso
   `python -u nn/entrenar_local.py --brazo …`**: el freno casa ese nombre en la línea de
   comando, y un script que sólo lo *importara* sería invisible para él. De `W` pequeño a
   grande: lo barato primero, para que un fallo se vea en minutos y no en horas.
5. **`nn/informe.py`**: tabla, figuras, criterio aplicado; `README.md` con el veredicto; reporte
   en `estudios-redes-neuronales` según lo que el dueño confirme (`ESPECIFICACION.md` §6).

- **Qué se mide y con qué umbral:** en `instrucciones/02-criterio.md`, escrito **antes**.
  Resumen: IoU sobre las 900, IoU de `train` y brecha, por brazo, media ± sd de 5 semillas; una
  diferencia cuenta si supera `max(2·SE_dif, 0,01)`; cada caída se clasifica en «menos
  información» / «peor generalización» / «sin cerrar» por la descomposición y el control.
- **Cuántos brazos y cuántas semillas:** 5 `W` × 5 semillas = **25**, +5 con el control.
- **Qué se llama «ganar»:** **no se declara ganador.** Se describe la curva y se contestan las
  dos preguntas del criterio: cuál es el `W` mínimo suficiente, y si la resolución estorba — y
  de qué causa.

## Scripts

**Previstos, no escritos.** Los nombres y las banderas son de aquí.

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/datos.py` | carga el dataset publicado, recorta, reduce a `W` (y el control), normaliza; etiquetas; huellas | `python nn/datos.py --comprobar` |
| `nn/modelo.py` | la red parametrizada por `(W, L, f, C)`, **autocontenida** | `python nn/modelo.py` (tabla de mapas y parámetros, un forward) |
| `nn/probar.py` | los tests del dato y del modelo (huellas, suma conservada, `uint16` cae, mapa `L×L`, parámetros) | `python nn/probar.py` |
| `nn/entrenar_local.py` | entrena **un** brazo; `metrics.jsonl` según ocurre; `summary.json` al final | `--brazo w064-s3 [--pasos N] [--hilos N]` · `--ensayo` · `--comprobar` |
| `nn/informe.py` | tabla, figuras y criterio aplicado, del disco | `python nn/informe.py` |
| `nn/lanzar.sh` | en el dev: los brazos en serie como unidad de systemd; guarda el modo al entrar; imprime la orden; el caso final se niega; se niega a lanzar dos veces | `todo [brazo…]` · `--estado` (disco, con `NRestarts`) · `SECO=1 …` (antes del guardia) |
| `nn/probar_lanzador.sh` | el despacho del lanzador por modo, con el seco | `sh nn/probar_lanzador.sh` |
| `nn/vast.sh` + `nn/vast.json` | en Vast: una máquina por semilla con todos sus brazos; libro commiteado al alquilar | `todo` · `--estado` · `apagar` · `VAST_SECO=1 …` |

- ⚠ **`entrenar_local.py` se llama así por el FRENO** (`cerrable.mjs` → `TRABAJOS`): con otro
  nombre, el veredicto «¿se puede apagar este server?» no lo ve. El commit que lo traiga pone
  `gasta` en `alquila` o `entrena-local` y `entrada` en `nn/entrenar_local.py`.
- **Dependencias:** el `.venv` de la raíz de `experimentos-cnn` (torch 2.14.1+cpu y numpy 2.5.3,
  *los que hay hoy en el del dev*) y matplotlib para las figuras. En Vast, el mismo torch fijado
  por versión; la evaluación de las 900 va por lotes de 100 (la máquina del dev tiene 3,8 GB y
  no tiene swap).
- **De dónde sale el código:** se escribe aquí. **No importa nada de ningún experimento**: el
  recorte y la transformación `(c/4) − 9` se reescriben como fórmula, no se importan.

## Qué NO hereda

- **Se copió de:** ningún experimento.
- **Qué se cambió a propósito respecto de `banco-k`, que lee el mismo dataset:** convolución
  `valid` (allí `same`); sin stride ni pooling (allí stride 2); kernel **relativo al ancho**
  (allí 5/5/3 fijos); cabeza `Linear` (allí conv 1×1 + soft-argmax); entrada en fracción de
  tinta sin estandarizar (allí `μ, σ`); **la resolución es la variable** (allí 128 fijo); 5
  semillas (allí 10); `val = monitor ∪ eval` (allí separados); no hay kernel de preproceso ni
  recorte dependiente de `k`; **sus pesos no se importan nunca al banco**.
- **Qué se conservó, y por qué se decidió conservarlo:** 1000 pasos, lote 20, Adam, L1 sobre
  coordenadas normalizadas, sin aumento, sin parada temprana, `last.pt`, y el recorte 146 → 128.
  Porque el régimen que este experimento quiere observar es el **mismo** (100 imágenes,
  sobreajuste a la vista) y esos mandos ya están medidos como suficientes para converger sobre
  este dato (identidad: 0,84 en `train`, 0,80 en `eval` a los 1000 pasos; `banco-k`,
  2026-09-08); el recorte, por el motivo propio de arriba (potencia de 2). Se **decidió**;
  cambiarlos sería gratis.
- **Contra qué se compara:** **entre sus propios brazos** (`W` contra `W`, y el control) y contra
  **su piso** (caja media de `train` sobre las 900: 0,2464). **No** contra la identidad del banco
  (0,7981): otra red y otra cabeza; se cita sólo como contexto de qué da este dato con 100
  imágenes. Dejaría de ser comparable entre brazos si cambiara `C`, `L`, `f`, `lr`, los pasos,
  el reparto o la máquina entre ellos — por eso son constantes y todos los brazos de una
  semilla corren en la misma máquina.
- **Restricciones de otros experimentos que NO aplican aquí:** la reserva §3.7 del banco (no se
  producen kernels); su recorte dependiente de `k`; su techo 0,95 y su umbral de arranque 0,40
  (aquí el piso se mide y se reporta); su obligación de 10 semillas.
