# Reglas de `rect-bor`

**Qué pregunta:** si un filtro de bordes convierte el trazo grueso en dos bordes finos, ¿el kernel lineal de `rect-lin`
entrenado con rectas finas detecta también las gruesas, sin perder las finas, sin subir los falsos positivos y sin
necesitar más muestras?

**Pre-proceso: UN filtro de bordes, el mismo para todo** (regla del dueño del 2026-10-07): entrenamiento, banco de prueba
y negativos de calibración del Gabor pasan por él. Dos candidatos, cada uno un brazo: `contorno` (x − erosión con cruz
3×3: borde interior de 1 px) y `sobel` (magnitud del gradiente, normalizada). Detalle en `nn/prepro.py`.
⚠ El candidato vigente del repo es el de **bordes de un solo lado** (`feat-1lado`); aquí **no** se usa porque da 8
canales y este detector es de UN canal. Decisión de este experimento, no un olvido.

## Entradas

- **Datasets:** los de `rect-lin`, sin tocar: `rect-lin-entreno-r20261008` (sólo se usan las 1000 CONTINUAS y los 1000
  negativos) y `rect-lin-banco-r20261008`. Mismas condiciones (REGLAS.md de `rect-lin`): entrenamiento con grosor 2–4 px,
  banco con 2–14 px. Ángulo con y hacia abajo; 4 orientaciones, etiqueta al centro más cercano.
- **Qué se transforma al cargar:** `uint8` → `float32` y el filtro de bordes del brazo.

## Salidas

- `resultados/trozos/rejilla-*.jsonl` (traídos de Vast) o `resultados/rejilla.jsonl` (dev): una línea por brazo con las
  métricas y los 2 kernels. `resultados/referencias.json` (Gabor por filtro y tamaño), `resultados/analisis.json`,
  figuras `resultados/*.png`. Todo se commitea; los datos, no.

## Procesos

1. `nn/prepro.py`: ver los filtros en ASCII. 2. `nn/modelo.py --comprobar`. 3. `nn/evaluar.py --referencias`.
4. La rejilla (`nn/vast.sh rejilla`). 5. `nn/evaluar.py --tablas --figuras`, README y reporte.

- **Rejilla:** pre-proceso ∈ {contorno, sobel} × k ∈ {5, 7, 9} × N ∈ {4, 8, 16, 32, 64, 128, 256, 1000} × 3 semillas =
  **144**. **1 escala** (en `rect-lin` las escalas empeoraban al kernel aprendido) y **sólo trazo continuo**.
- **Control sin filtro:** NO se re-entrena: es el brazo «continua, 1 escala» de `rect-lin`, cuyos kernels y métricas son
  bit a bit reproducibles (medido allí). Se lee de su `resultados/trozos/` por id (`por_id("rect-lin")`).
- **Entrenamiento:** el de `rect-lin` sin cambios (Adam, lr 0,03, 400 épocas, entropía cruzada de 5 clases).
- **Métrica nueva:** `recall_grueso_largo` = recall en grosor 10–14 sólo con largo 16 y 22. Una «recta» de 10 × 14 px es
  un bloque sin orientación, y en `rect-lin` contaba en el grueso. Se reportan las dos.

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/prepro.py` | los filtros de bordes | `python nn/prepro.py` (ASCII) |
| `nn/datos.py` | carga los datasets de `rect-lin` por huella | `--comprobar` |
| `nn/modelo.py` | el kernel lineal y el Gabor (copia de `rect-lin`) | `--comprobar` |
| `nn/entrenar_local.py` | la rejilla, reanudable | `[--parte i/n] [--salida f] [--hilos h] [--solo pre,k,N,semilla]` |
| `nn/evaluar.py` | métricas, referencias, análisis y figuras | `--referencias` · `--tablas` · `--figuras` |
| `nn/vast.sh` | la rejilla en UNA máquina de Vast | `rejilla` · `--estado` · `apagar` · `VAST_SECO=1 …` |

- **Dependencias:** el `.venv` de la raíz (torch CPU, numpy, matplotlib).

## Qué NO hereda

- **Se copió de:** `rect-lin` (todo `nn/`). **Cambiado:** el eje «entrenamiento continua/punteada» y el de escalas se
  quitan; entra el eje de pre-proceso; `datos.py` ya no publica nada (los datasets son los de `rect-lin`); la métrica
  `recall_grueso_largo`; sin referencia CNN (ya medida en `rect-lin`, y no vio bordes).
- **Conservado a propósito:** detector, entrenamiento, banco y métricas, para que la única diferencia con el control sea
  el filtro.
- **Contra qué se compara:** el control de `rect-lin` (sin filtro, 1 escala, continua) y el Gabor con cada filtro.
- **Restricciones de otros experimentos que NO aplican:** las conclusiones de `feat-bor` (los bordes empeoraban el
  compositor de dígitos): aquí no hay compositor ni dígitos; se mide el detector solo.
