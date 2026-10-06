# Reglas de `feat-bor`

**Escritas el 2026-10-06, antes de entrenar.** El criterio, en `instrucciones/02-criterio.md`; el encargo literal, en
`instrucciones/01-encargo.md`. Al correr se reescribe aquí lo que cambie, en el mismo commit.

**Qué pregunta:** si los 13 detectores de features de `feat-ind32` se entrenan igual pero viendo los **bordes** de la
tinta en vez de la tinta, ¿dejan de depender del grosor del trazo? En sintético: ¿reconocen trazos más gruesos que los que
vieron? En los dígitos: ¿dejan de encender arcos en los 1, y los grupos de `feat-agr` dejan de romperse al engrosar?

## Es factible, y qué se midió ANTES de entrenar

Factible sí: el vocabulario, el sorteo, la red y la receta son los de `feat-ind32`; sólo cambia la imagen que entra
(`nn/bordes.py`). Lo que no estaba claro es que **el borde sea independiente del grosor**, y se midió sin entrenar
(`nn/sonda_grosor.py`, 2026-10-06). Cada feature se dibujó con la misma geometría a 3, 6, 9 y 12 px:

| coseno con la misma geometría a 3 px (σ 1,5 px) | 6 px | 9 px | 12 px | píxeles 3 → 12 px |
|---|---:|---:|---:|---:|
| líneas (el trazo relleno, lo de hoy) | 0,96 | 0,88 | 0,77 | 74 → 291 |
| contorno (borde sin signo) | 0,83 | 0,48 | 0,27 | 47 → 64 |
| signo (cada lado aparte) | 0,81 | 0,47 | 0,24 | 67 → 89 |
| … **admitiendo desplazar ± 6 px**: líneas | 0,96 | 0,88 | 0,78 | |
| contorno | 0,83 | 0,64 | **0,59** | |
| signo, cada lado con su desplazamiento | 0,95 | 0,87 | **0,79** | |

La **cantidad** de borde sí es casi independiente del grosor (47 → 64 px, contra 74 → 291 del trazo). Pero un trazo tiene
**dos** bordes y se separan al engrosarlo: el contorno de un trazo grueso son dos líneas, o un lazo, donde el fino era una.
El borde **con signo** pone cada lado en su canal, y cada lado conserva la forma (desplazado medio grosor). Por eso se
entrenan **las dos**: el contorno, que es lo pedido literalmente, y el borde con signo.

## Entradas

- **Entrenar:** `feat-ind32-sinteticas-32px-r20261005` (publicado por `feat-ind32`; se lee con `exigir_dataset`, no se
  regenera): las **mismas** 32.400 features de 32×32 con las que se entrenaron las líneas, grosores 2–4 px, con su ruido
  (`p_on` ≤ 0,02, `p_off` ≤ 0,10) y una segunda feature en la mitad. Particiones `train`/`val` tal cual.
- **La prueba de grosor:** `feat-bor-sinteticas-grueso-32px-r20261006`, **publicada desde aquí** (`nn/datos.py
  --generar-grueso --publicar`): el mismo vocabulario con el perfil `grueso` de `features.py` (grosores 2, 3, 4, 6, 8, 10 y
  12 px), 400 por familia + 200 vacías, otras semillas (20261006 + 10·familia). Sin reparto: sólo se evalúa. 3112
  principales de 2–4 px (vistos) y 2088 de 6–12 px (no vistos).
- **Los dígitos:** `uci-optdigits-orig-32px-r20261005` (los 5620; los 1797 de `windep` con su reparto 180/1617).
- **Qué se lee:** `imagenes`, `principal`, `secundaria`, `ancla`, `grosor`, `radio`, `particion`; de los dígitos,
  `imagenes`, `etiquetas`, `particion` y `origen`.
- **Al cargar:** la imagen 0/1 pasa por `bordes.bordes(x, repr)`:
  - **`contorno`** (1 canal): la tinta con algún vecino-4 de fondo;
  - **`signo`** (4 canales): la tinta con fondo a la izquierda · a la derecha · arriba · abajo.

  Fuera del lienzo cuenta como fondo. Se guarda en uint8 y se pasa a float en cada lote: los mismos valores 0/1 que
  float32, sin cuatro copias de 32.400 imágenes en memoria.
- **Los negativos** de cada detector, como en `feat-ind32`: ni la principal ni la secundaria pueden ser la familia, ni una
  que la **contenga** (`features.CONTIENE`).

## Salidas

- **Pesos:** `nn/pesos-contorno/<f>/` y `nn/pesos-signo/<f>/`: `best.pt` (mejor F1 de val), `config.json`,
  `metrics.jsonl`, `summary.json` y `log.txt`. Se commitean `best.pt`, `config.json` y `summary.json` (que trae las
  métricas del mejor y del último, la curva P/R y los FP por familia); `last.pt`, `metrics.jsonl` y `log.txt` **no**
  (`.gitignore` de la carpeta): con 26 detectores los dos `.pt` pasarían de 9 MB, y el tope del repo es ≈5 MB por
  experimento.
- **Evaluación:** `resultados/evaluacion.json` (§A, §B, §C y el criterio), `resultados/grosor.png` (recall por grosor) y
  `resultados/consistencia.png` (κ de los dígitos). La sonda, en `resultados/sonda-grosor.json`; la referencia de las
  líneas, calculada con el mismo código antes de entrenar, en `resultados/referencia-lineas.json`.
- **El libro de Vast:** `resultados/vast/detectores/` — se commitea **al alquilar** (lo hace `nn/vast.sh`).
- **Reporte en el central:** **sí**, al terminar — alquila, y todo lo que alquila deja su reporte en
  `estudios-redes-neuronales` (como `feat-ind32`, #30–#33).
- **Presupuesto de tamaño:** 26 `best.pt` ≈ 4,5 MB + resúmenes, código y resultados ≈ 0,4 MB: **al límite de los ≈5 MB**
  (estimado por el revisor con los de `feat-ind32`). Si lo pasa, se pregunta antes de commitear.

## Procesos

1. **Sonda** sin entrenar (`nn/sonda_grosor.py`) — hecha; es la que decidió entrenar las dos variantes.
2. **Publicar la prueba de grosor** (`nn/datos.py --generar-grueso --publicar`) y empujarla al almacén.
3. **Comprobar** el mecanismo aquí: `nn/bordes.py`, `nn/datos.py --comprobar`, `nn/entrenar_local.py --comprobar` (dos
   variantes, determinismo, carga), `nn/metricas.py --comprobar`, `nn/probar_vast.sh` y `VAST_SECO=1 nn/vast.sh detectores`.
4. **Entrenar en Vast:** `nn/vast.sh detectores` — los **26** detectores (13 por representación) **a la vez** en UNA
   máquina de 26–48 vCPU y ≥ 48 GB (1,29 GB de pico medido por proceso), un proceso de 1 hilo cada uno; trae los pesos y **destruye la máquina**; tope 3 h. La receta es
   la de `feat-ind32` C3 sin tocar: 80 épocas, lotes 32 + 32, lr 2·10⁻³, BCE por celda (objetivo ×8) + BCE del máximo,
   umbral elegido en train, `best` por F1 de val, semilla 1. La red, la misma; con `signo`, 4 canales de entrada (+432
   parámetros en la primera convolución).
5. **Evaluar** aquí (`nn/evaluar.py`, ~10 min): §A, §B y §C de `instrucciones/02-criterio.md`, con las **líneas** de
   `feat-ind32` como referencia (leídas por su id y comprobadas por su huella; no se copian, por el tope de tamaño).
6. README con lo que salió, reporte en el central, commit y push.

- **Qué se mide y con qué umbral:** `instrucciones/02-criterio.md` (H1–H5).
- **Cuántos brazos y semillas:** 2 representaciones nuevas × 13 detectores × 1 semilla (como `feat-ind32`); la
  referencia, las líneas; en §C, 5 semillas de k-means y 3 del compositor.
- **Qué se llama «ganar»:** no se declara ganador; cada hipótesis se confirma o se refuta por representación.

### Dónde corre y cuánto cuesta

**Vast** (`nn/vast.sh detectores`): 26 procesos de ~15–17 min (1 época con 2 hilos en el dev: 8,7 s contorno, 10,0 s signo,
medido el 2026-10-06) en UNA máquina ≤ 0,40 $/h → **≈ 20–30 min y ≈ 0,02–0,05 $** *(estimado, por `feat-ind32` C3: 13
procesos, 15,5 min, 0,0172 $)*. En el dev serían ~3,5 h (2 vCPU). **El freno**: `nn/vast.sh apagar` aquí, o desde Telegram
`/use exp-vast` → `apagar expc-fbor-`. El libro se commitea al alquilar. La evaluación, en el dev.

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/features.py` | el vocabulario y el rasterizador: COPIA SIN CAMBIOS del de `feat-ind32` | — |
| `nn/bordes.py` | lo único nuevo: `lineas` · `contorno` · `signo`; se prueba contra casos hechos a mano | `python nn/bordes.py` |
| `nn/datos.py` | carga con huellas, `conjunto(…, repr)`, genera y publica la prueba de grosor, los dígitos | `--generar-grueso [--publicar]` · `--comprobar` |
| `nn/modelo.py` | el detector de `feat-ind32` con `entrada` = 1 o 4 canales | `python nn/modelo.py` |
| `nn/entrenar_local.py` | entrena UN detector (el nombre es el contrato con el freno); la receta de `feat-ind32` | `--repr contorno\|signo --feature f [--hilos 1]` · `--comprobar` |
| `nn/sonda_grosor.py` | la sonda sin entrenar | → `resultados/sonda-grosor.json` |
| `nn/evaluar.py` | §A, §B, §C y el criterio, con las líneas de referencia | → `resultados/evaluacion.json`, `grosor.png`, `consistencia.png` |
| `nn/metricas.py` | k-means, κ y demás: COPIA del de `feat-agr` | `--comprobar` |
| `nn/vast.sh`, `nn/vast.json` | Vast, modo `detectores`; se niega si el dataset no está empujado o la comprobación falla | `detectores` · `--estado` · `apagar` · `VAST_SECO=1 …` |
| `nn/probar_vast.sh` | prueba en seco del lanzador | `nn/probar_vast.sh` |

- **Dependencias:** el `.venv` de la raíz (numpy 2.5.3, torch 2.14.1 CPU, Pillow 12.3.0, matplotlib 3.10.9); el de Vast,
  las mismas sin matplotlib.

## Qué NO hereda

- **Se copió de:** `feat-ind32` — `features.py` (sin cambios), `modelo.py` (más `entrada`), `entrenar_local.py` (más
  `--repr` y uint8), `vast.sh`/`vast.json`/`probar_vast.sh` (un modo), las transformaciones de C6 y el compositor de C4; de
  `feat-agr`, `metricas.py` y las zonas. Se copia, no se importa.
- **Qué se cambió a propósito:** la imagen de entrada (bordes); 4 canales en `signo`; los datos en uint8; el número de
  detectores por máquina (26); y la métrica de grosor de §B (F1, no recall: ver el criterio).
- **Qué se conservó, y por qué:** todo lo demás de la receta, el sorteo y la red, **para que la única diferencia con las
  líneas sea lo que ve el detector**; y las medidas de `feat-agr` (zonas, K = 30, κ), para comparar con lo que allí falló.
- **Contra qué se compara:** con las líneas de `feat-ind32`, por id y con su huella, en las tres secciones; y con los píxeles
  (X8) en H4.
- **Restricciones de otros experimentos que NO aplican aquí:** el banco de 8×8 y sus corridas (`feat-ind`); la etiqueta
  como objetivo de los detectores (siguen sin ver un dígito); el reparto T/p de la ganancia; las CNN de punta a punta.

## Lo que cambió al correr (2026-10-06)

- **La máquina:** 28 vCPU (Xeon E5-2680 v4, 62,8 GB, 0,0956 $/h). **69 min y 0,1098 $**, contra 20–30 min y 0,02–0,05 $
  estimados: 26 procesos de 1 hilo en 28 vCPU fueron ~4× más lentos por núcleo que el dev. Para los siguientes
  (`feat-fallos`, `feat-cortas`) se pidió 2 vCPU por proceso.
- **El tamaño:** lo commiteado de este experimento ocupa **4,96 MB** (4.959.893 B, medido con `git ls-files | xargs du -cb`
  antes del README), dentro del tope de ≈5 MB; los `best.pt` son 26 × 174 KB.
- **El pendiente que pidió el dueño** (curvas contra rectas) se anotó en el README y se mide en `feat-fallos`.
