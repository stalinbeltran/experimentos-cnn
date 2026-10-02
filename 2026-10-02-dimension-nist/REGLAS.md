# Reglas de `dim-nist`

**Escritas el 2026-10-02 al preparar el experimento. Estado `abierto`, PREPARADO y NO lanzado**:
hay código, dataset publicado, criterio escrito y ensayo de mecanismo; ningún brazo entrenado.
Se lanza cuando el dueño confirme los supuestos de `ESPECIFICACION.md` §0.

⚠ **Estas reglas son de este experimento y de ninguno más.** **Se copió de `dim-gen`** (orden del
dueño: «otro experimento idéntico a este») y se releyó línea por línea: lo que cambia está en
§ «Qué NO hereda», con su motivo.

**Qué pregunta:** con los dígitos manuscritos de 8×8, el 10 % para entrenar y el 90 % para
validar, `L = 2` capas sin padding con kernel `n = ⌈W/2⌉` y cabeza constante, ¿cómo cambia la
exactitud sobre las no vistas —y la brecha train−val— al reducir la imagen de 8 a 7, 6, 5 y 4
px? ¿Qué resolución generaliza mejor?

## Entradas

- **Dataset: `uci-optdigits-8px-r20261002`**, publicado en `foveal-vision-data/experimentos-cnn/`
  por `nn/datos.py --publicar` (una vez; se niega a pisar) y leído con `expcnn.exigir_dataset`.
  Es la copia de scikit-learn (`load_digits`) del *test set* de UCI *Optical Recognition of
  Handwritten Digits*: **1797 imágenes de 8×8 con valores 0..16** (bits encendidos por bloque
  4×4 del bitmap de 32×32 de NIST), 10 clases equilibradas. ⚠ **8×8 ya es /4 del original de
  NIST**; el 32×32 no está en scikit-learn, así que `W > 8` no existe aquí. Y son **13
  escritores** (el *test set* de UCI): train y val los comparten.
- **Qué se lee:** `datos.npz` → `imagenes` (1797, 8, 8) uint8, `etiquetas` (1797,) uint8,
  `particion` ('train'/'val'). Las tres huellas se casan contra el manifiesto al cargar.
- **El reparto 10 / 90 está en el dataset**: `train` = 18 por clase = **180** (semilla 0,
  estratificado); `val` = **1617**. El mismo train para todos los brazos; **ninguna decisión usa
  las 1617**.
- **Qué se transforma al cargar** (`nn/datos.py`, determinista):
  1. **reducción 8 → `W ∈ {8, 7, 6, 5, 4}` por promedio de área**: `S = A·x·Aᵀ` con `A` entera
     (filas que suman 8; los pesos son múltiplos exactos de 1/8), en `int64`. Para `W = 4` es la
     media de bloques 2×2; para 7, 6 y 5 cada píxel de salida promedia el área fraccionaria que
     cubre. Exacta e independiente de la máquina;
  2. **valor de entrada** `x = S / 1024` = fracción de tinta en `[0, 1]` (0 fondo, 1 tinta), igual
     a todo `W`; sin estandarizar;
  3. **etiqueta** = el dígito (0..9), igual para todo `W`;
  4. **el control `w8-de4`**: la matriz de `W = 4` repetida 2×2 hasta 8, con la red de `w8`.
- **Huellas congeladas** *(2026-10-02, sha256 de `S` int64 en el orden del dataset, 16 hex;
  `cargar` se niega si no casan)*:

  | `W` | huella `S` | fila 0 de `A` |
  |---:|---|---|
  | 8 | `ee9218e7323cc76a` | `[8,0,0,0,0,0,0,0]` |
  | 7 | `6b9f3ef6b1ef852a` | `[7,1,0,0,0,0,0,0]` |
  | 6 | `91ee3970a6036d50` | `[6,2,0,0,0,0,0,0]` |
  | 5 | `3bdbbc60af71beec` | `[5,3,0,0,0,0,0,0]` |
  | 4 | `39483d316ebda8eb` | `[4,4,0,0,0,0,0,0]` |

  Imágenes `8f26b2bd9d135c25` · etiquetas `8ba4f891220f5e4c`.

## Salidas

- **Por corrida**, en `nn/pesos/<brazo>-s<semilla>/`: `last.pt` (época final; **sin `best.pt`**),
  `metrics.jsonl` (una línea por época; cada 10, exactitud de train y val y brecha),
  `summary.json` (exactitud train/val, brecha, **por dígito**, parámetros, segundos, máquina,
  huellas), `log.txt`.
- **Pesos**: no se commitean aquí (`.gitignore`), **sí van al volumen** con todo lo demás
  (`foveal-vision-data/experimentos-cnn-resultados/dim-nist/`, lo hace el cierre de
  `nn/lanzar.sh`), por la orden del dueño del 2026-10-02.
- **Figuras** `resultados/acc-vs-w.png`, `resultados/brecha-vs-w.png` e informe
  `resultados/RESULTADOS.md` + `criterio-aplicado.json`, regenerados por `nn/informe.py`.
- **Qué se commitea:** métricas, resúmenes, logs, figuras, informe. Nunca el dataset.

## Procesos

0. **El dueño confirma los supuestos** S1–S4 (`ESPECIFICACION.md` §0). Sin eso no se lanza.
1. `nn/datos.py --comprobar`: manifiesto, huellas de las cinco reducciones, control, reparto,
   la conservación de la media y el piso (1/10). `nn/probar.py`: lo mismo con el modelo.
2. **Ensayo de mecanismo** (`nn/entrenar_local.py --ensayo`), **hecho el 2026-10-02** mirando
   **sólo** la pérdida de train, 4000 pasos: con **`lr = 3e-3`** baja sin ninguna subida >20 % en
   `W = 8` (CE final 0,0031, exactitud de train 1,0) y `W = 4` (0,486; 0,83); con `1e-3` también
   baja pero `W = 4` se queda en 0,89 (0,70), lejos de la meseta. **`lr = 3e-3`, congelado**;
   `C = 8` se queda: la cabeza de 8 features deja llegar a 1,0 en train. Cada brazo: **26 s**
   medidos en el dev (2 hilos).
3. **Entrenar los 30 brazos**: `nn/lanzar.sh todo` (una unidad de systemd en el dev, padre PID 1,
   en serie: cada brazo es un proceso `nn/entrenar_local.py`). Al terminar la unidad corre
   `informe.py`, commitea, copia todo al volumen y avisa.
4. `README.md` con el veredicto; reporte en `estudios-redes-neuronales` (es un estudio con reloj,
   aunque sin factura).

- **Qué se mide y con qué umbral:** `instrucciones/02-criterio.md`, escrito antes.
- **Brazos y semillas:** 5 `W` × 5 + 5 del control = **30**.
- **Qué se llama «ganar»:** no se declara ganador; se contesta cuál es el `W` mínimo suficiente y
  si la resolución estorba, y de qué causa.

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/datos.py` | publica (una vez) y carga el dataset; reducciones exactas; control; huellas | `--publicar` · `--comprobar` |
| `nn/modelo.py` | la red `(W, L=2, f=0,5, C=8)` con promedio global, autocontenida | `python nn/modelo.py` |
| `nn/probar.py` | las pruebas sin entrenar | `python nn/probar.py` |
| `nn/entrenar_local.py` | un brazo; `metrics.jsonl` según ocurre; `summary.json` al final | `--brazo w6-s3 [--pasos N] [--hilos N]` · `--ensayo --w 8 --lr …` · `--comprobar` |
| `nn/informe.py` | tabla, figuras y criterio aplicado, del disco | `python nn/informe.py` |
| `nn/lanzar.sh` | los 30 en serie como unidad; imprime la orden; se niega a lanzar dos veces; cierre | `todo [brazo…]` · `--estado` · `SECO=1 …` |
| `nn/probar_lanzador.sh` | el despacho del lanzador, con el seco | `sh nn/probar_lanzador.sh` |

- ⚠ **`entrenar_local.py` se llama así por el freno** (`cerrable.mjs` → `TRABAJOS`).
- **Dependencias:** el `.venv` de la raíz del repo (torch 2.14.1+cpu, numpy 2.5.3, matplotlib) y
  **scikit-learn sólo para `--publicar`** (ya hecho: entrenar no lo necesita).
- **De dónde sale el código:** copiado de `dim-gen` y reescrito para clasificación. No importa
  nada de ningún experimento.

## Qué NO hereda

- **Se copió de:** `dim-gen`.
- **Qué se cambió a propósito, y por qué:**
  - **El dato**: dígitos de 8×8 publicados desde scikit-learn, en vez de párrafos de 128; el
    reparto 10/90 es por clase (18 por dígito).
  - **La tarea y la métrica**: clasificación de 10 clases con entropía cruzada y **exactitud**, en
    vez de regresión de caja con L1 e IoU. El piso es **1/10** (azar), no la caja media: la clase
    mayoritaria de train no existe (18 por clase).
  - **`L = 2` en vez de 4, con la misma regla `f = 1/L`**. ⚠ No porque `L = 4` «no quepa» (en
    `dim-gen` el brazo `w008` era `L = 4`, `n = 2`, y entrenó), sino por aritmética: con
    `W ∈ {8, 7, 6, 5, 4}` **no se pueden tener a la vez una `f` constante y una cabeza constante**
    —`n = f·W` entero exige `W` múltiplo de `1/f`, y un mapa final igual entre `W` consecutivos
    exige `L = 1`— y `dim-gen` tenía las dos. `L = 2` es la que deja el campo receptivo completo en
    los cinco `W`; `n = ⌈W/2⌉` hace que `f` valga 0,50 en los pares y 0,57 / 0,60 en los impares
    (riesgo de zigzag, escrito en el criterio). La alternativa exacta son sólo `{8, 6, 4}`.
  - **Cabeza: promedio global + `Linear(C → 10)`** en vez de `Flatten + Linear`: el mapa final es
    2×2 (par) o 1×1 (impar), y el promedio global la hace constante entre `W`. Para clasificar no
    hace falta la posición.
  - **Reducciones fraccionarias** (7, 6, 5) por promedio de área exacto, en vez de sólo potencias
    de 2: «menos dramáticas», pedido literal. ⚠ `dim-gen` las excluyó a propósito («interpolar no
    es el mismo dato»); aquí se aceptan porque los pesos son exactos (múltiplos de 1/8), la media
    se conserva (prueba en `--comprobar`) y la huella está congelada.
  - **La descomposición (a)/(b) va en entropía cruzada**, no en la métrica principal: la exactitud
    de train satura en 1,0 (medido en el ensayo).
  - **Corre en el dev** (`entrena-local`): segundos por brazo. Sin Vast, sin libro.
  - **δ = 0,01 de exactitud** (16 imágenes de 1617) en vez de 0,01 de IoU.
- **Qué se conservó, y por qué:** 10 % / 90 %, 5 semillas, 4000 pasos de lote 20, Adam con un `lr`
  único congelado en un ensayo de sólo train, sin selección (`last.pt`), evaluación cada 10
  épocas, `C = 8`, el control de información a parámetros fijos, el criterio entero (W\*, W
  mínimo suficiente, «estorba», brecha, descomposición (a)/(b)/(c)) y la regla nueva de
  `dim-gen`: mirar las semillas atascadas. Se decidió: es lo que hace comparables las dos lecturas.
- **Contra qué se compara:** entre sus brazos y contra su piso. **No** contra `dim-gen` en
  números (otra tarea, otro dato): sí en **forma** de la curva, que es la pregunta del dueño.
- **Restricciones de otros experimentos que NO aplican:** la reserva §3.7 del banco (otro dato),
  el tope de 128 px y las potencias de 2 de `dim-gen`, Vast.
