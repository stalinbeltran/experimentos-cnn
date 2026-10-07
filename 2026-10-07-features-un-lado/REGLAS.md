# Reglas de `feat-1lado`

**Escritas el 2026-10-07, antes de entrenar.** El criterio, en `instrucciones/02-criterio.md`; el encargo literal, en
`instrucciones/01-encargo.md`. Al correr se reescribe aquí lo que cambie, en el mismo commit.

**Qué pregunta:** si el único pre-proceso son **8 bordes de un solo lado** y cada detector de features mira **un canal cada
vez** (mismos pesos para los 8 y máximo después), ¿deja el grosor del trazo de confundir a los detectores —lo que `feat-bor`
no consiguió mirando los canales juntos—, sin perder lo que acertaba el compositor de dígitos? Y el compositor entrenado con
su entrada desplazada, ¿resiste un desplazamiento ligero y sólo pierde cuando el dígito sale del lienzo?

## Pre-proceso: UNO, y dentro del modelo

Regla del dueño del 2026-10-07 (`CLAUDE.md` de la raíz): **un solo pre-proceso, el mismo para todo lo que ven los
detectores**. Aquí es la **primera capa del modelo**, fija (un buffer, no un parámetro): los 8 kernels 3×3
k_θ = cos θ·Sx + sin θ·Sy (Sobel / 4), θ = 0°, 45°, …, 315°, más ReLU — los del boceto. Va dentro del modelo para que sea
**imposible** aplicar a los dígitos otro distinto que al entrenamiento. Su huella (`c979cfba642520e2`) va en cada
`config.json`, y `modelo.cargar` se niega si no casa.

**La aumentación del compositor no es pre-proceso**: desplazar la imagen sólo cambia con qué ejemplos se ENTRENA el
compositor; los detectores ven lo mismo (la convolución es equivariante) y la prueba va sin desplazar. Aceptado por el
dueño («tal como dices», encargo §4).

## Brazos

| | detector | entrada de la primera convolución | parámetros |
|---|---|---|---:|
| **control** | el tronco mira los 8 canales a la vez | 8 | 42.833 |
| **compartido** | el MISMO tronco en cada canal por separado; máximo de los 8 mapas de logits | 1 | 41.825 |

El tronco es la red de `feat-ind32` / `feat-bor` sin cambios. El compartido hace **8 pasadas por imagen**: cuesta ~4,5
veces el control (medido abajo). Un tercer brazo con **suma** en vez de máximo no se entrena (pasaría del tope de tamaño
del repo); si hace falta, se mira al componer.

## Entradas

- **Entrenar:** `feat-ind32-sinteticas-32px-r20261005` (las mismas 32.400 features de 2–4 px de `feat-ind32` y `feat-bor`),
  con `exigir_dataset` y sus huellas. Particiones `train`/`val` tal cual. Entra la TINTA, sin transformar.
- **Prueba de grosor:** `feat-bor-sinteticas-grueso-32px-r20261006` (publicado por `feat-bor`, 2–12 px).
- **Dígitos:** `uci-optdigits-orig-32px-r20261005` (5620; windep con su reparto 180/1617; los 3823 restantes, «otros escritores»).
- **Negativos** de cada detector, como en `feat-ind32`: ni la principal ni la secundaria pueden ser la familia, ni una que la
  contenga.

## Salidas

- **Pesos:** `nn/pesos-control/<f>/` y `nn/pesos-compartido/<f>/`: se commitean `best.pt`, `config.json` y `summary.json`;
  `last.pt`, `metrics.jsonl` y `log.txt` no (`.gitignore`). 26 `best.pt` ≈ 26 × 175 KB ≈ 4,6 MB: **al límite de los ≈5 MB**,
  como `feat-bor` (4,97 MB). Si lo pasa, se pregunta antes de commitear. Los pesos del compositor no se guardan.
- **Mapas** del compositor: `nn/mapas/` (fuera de git; se regeneran).
- **Evaluación:** `resultados/evaluacion.json` (§A, §B, §C y el criterio) y `resultados/componer.json` / `.txt` (el
  compositor y la curva). La referencia de las líneas, calculada antes de entrenar: `resultados/referencia-lineas.json`.
- **Libros de Vast:** `resultados/vast/control/` y `resultados/vast/compartido/`, commiteados **al alquilar** (`nn/vast.sh`).
- **Reporte en el central:** sí, al terminar (alquila).

## Procesos

1. **Comprobar** aquí: `nn/modelo.py --comprobar` (kernels = `correlate` de scipy, cada canal un solo borde, kernels fijos,
   compartido = máximo de 8 pasadas), `nn/datos.py --comprobar`, `nn/entrenar_local.py --comprobar --brazo <b>`,
   `nn/probar_vast.sh` y `VAST_SECO=1 nn/vast.sh <b>`.
2. **Referencia** sin entrenar nada: `nn/evaluar.py --solo-lineas` y `nn/entrenar_local.py --componer --bancos lineas`.
3. **Entrenar en Vast**, con permiso del dueño y una máquina por brazo: `nn/vast.sh control` (13 procesos × 2 hilos, tope
   3 h) y `nn/vast.sh compartido` (13 × 2 hilos, tope 3 h; ver «Lo que cambió al correr»). Traen los pesos y **destruyen la máquina**. Receta abajo.
4. **Componer** en el dev, como unidad: `nn/entrenar_local.py --componer --bancos control,compartido`.
5. **Evaluar** en el dev: `nn/evaluar.py` (lee `componer.json` para H5–H7).
6. README con lo que salió, reporte en el central, commit y push.

### Receta (decisión de este experimento, no herencia)

80 épocas, lotes 32 positivas + 32 negativas, Adam lr 2·10⁻³, BCE por celda contra la gaussiana del ancla (σ 0,6, objetivo
×8) + BCE del máximo del mapa, umbral por detector elegido sobre train, `best` por F1 de val (cada 5 épocas), semilla 1.
**Se conserva la de `feat-ind32` / `feat-bor` a propósito**: así la única diferencia con ellos es lo que ve el detector.
Único cambio: los forwards de evaluación van a trozos de 512 (el compartido haría 48.000 pasadas de golpe).

### Compositor (decisión de este experimento)

Linear(832 → 10) sobre los 13 mapas 8×8 (σ), Adam 300 épocas a lote completo, lr 10⁻², L2 10⁻³, 3 semillas: el de
`feat-ind32`, para que sus números sean comparables con 0,949. **Posición tal cual, sin máximo**, y entrenado con la imagen
desplazada 0–2 px en 8 direcciones (el elegido en el boceto). La curva: «sólo s px» y «0..s px», s ∈ {1, 2, 3, 4, 6, 8}.

## Coste — medido en el dev, ESTIMADO para Vast

| | medido en el dev (2026-10-07, 2 hilos, CPU compartida con otra prueba) | Vast, estimado |
|---|---|---|
| control | 12,8 s/época | 13 × 2 hilos, ≥26 vCPU a ~0,10–0,15 $/h, ~1–2 h → **~0,1–0,3 $** (tope 3 h) |
| compartido | 54–58 s/época + ~66 s de evaluación cada 5; **1,09 GB** de pico por proceso | 13 × 4 hilos, ≥52 vCPU a ~0,18–0,35 $/h, **≈3–6 h** → **~0,6–2,1 $** (tope 8 h) — **estimación previa, superada: ver «Lo que cambió al correr»** |

La conversión dev → Vast sale de `feat-bor`. ⚠ **Corregida por el revisor antes de alquilar**: la primera versión de este
fichero decía «un hilo de Vast ~2,5 veces más lento que uno del dev», que sale de dividir por 2 el «~5× por proceso» y
supone que 2 hilos van el doble que 1. Hilo con hilo (las notas de `feat-bor/nn/vast.json`: 11–13 s/época con 1 hilo en el
dev contra 46–50 s en Vast) es **~4×**, y con eso el compartido sale 3–6 h: por eso el tope pasó de 6 a **8 h**, y cada
detector escribe `progreso.json` en cada evaluación (si el tope corta, no vuelve sin nada).

**Orden de lanzamiento, también del revisor:** primero el **control** (barato), y con su s/época REAL en Vast se recalcula
el compartido antes de pedir permiso para él. Precios de `vast_instance.py offers` el 2026-10-07: 64 vCPU desde 0,18 $/h;
28 vCPU desde 0,095 $/h. **Peor caso con los topes: 3 h × 0,2 + 8 h × 0,35 ≈ 3,4 $.** Quién apaga si este server muere:
las unidades de Vast son de systemd (padre PID 1) y destruyen solas al terminar o al tope; a mano, `nn/vast.sh apagar` o
`/use exp-vast` → `apagar expc-f1l-` desde cualquier máquina con el token.

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/features.py` | el vocabulario y el rasterizador: COPIA SIN CAMBIOS del de `feat-bor` | — |
| `nn/datos.py` | carga con huellas, `conjunto(d, f, partición)` (la tinta tal cual), los dígitos | `--comprobar` |
| `nn/modelo.py` | la capa fija de 8 bordes de un lado, los brazos `control` y `compartido`, y `Tinta` para leer la referencia | `python nn/modelo.py` · `--comprobar` |
| `nn/entrenar_local.py` | entrena UN detector (el nombre es el contrato con el freno) y lanza el compositor | `--brazo control\|compartido --feature f [--hilos N]` · `--comprobar [--brazo b]` · `--componer [--bancos …]` |
| `nn/componer.py` | el compositor de dígitos (C1, C2) y la curva de desplazamiento (C3); se llama desde `entrenar_local.py --componer` | → `resultados/componer.json`, `componer.txt` |
| `nn/evaluar.py` | §A, §B, §C y el criterio, con las líneas de referencia | `--solo-lineas` · → `resultados/evaluacion.json` |
| `nn/vast.sh`, `nn/vast-control.json`, `nn/vast-compartido.json` | Vast, un modo por brazo; se niega si el dataset o el criterio no están empujados, o la comprobación falla | `control` · `compartido` · `--estado` · `apagar` · `VAST_SECO=1 …` |
| `nn/probar_vast.sh` | prueba en seco del lanzador | `nn/probar_vast.sh` |

- **Dependencias:** el `.venv` de la raíz (numpy 2.5.3, torch 2.14.1 CPU, Pillow 12.3.0, scipy para `modelo.py
  --comprobar`); el de Vast, numpy, Pillow y torch.

## Qué NO hereda

- **Se copió de `feat-bor`:** `features.py` (sin cambios), `datos.py` (sin bordes ni generación), `modelo.py` (más la capa
  fija y los brazos), `entrenar_local.py` (`--brazo`, evaluación a trozos, `--componer`), `vast.sh`/`vast.json`/`probar_vast.sh`
  (dos modos, dos descriptores) y §B de `evaluar.py`. **De `feat-ind32`**: el compositor (`ajustar`). **Del boceto**: los
  kernels y `mover`. Se copia, no se importa.
- **Qué se cambió a propósito:** el pre-proceso (8 bordes de un lado, dentro del modelo), los dos brazos, la evaluación a
  trozos, §C (sin κ ni engrosamiento artificial: el boceto mostró que dilatar cambia la forma; se mide con los gruesos
  REALES) y el compositor con desplazamiento.
- **Qué se conservó, y por qué:** el vocabulario, el sorteo, la red, la receta y la medida de grosor de §B, para que la
  diferencia con `feat-bor` y `feat-ind32` sea lo que ve el detector.
- **Contra qué se compara:** las líneas de `feat-ind32` (por id y huella), con este mismo código.
- **Restricciones de otros experimentos que NO aplican aquí:** las medidas de grupos de `feat-agr` (κ, zonas, K = 30);
  el pre-proceso `bordes.py` de `feat-bor`; las transformaciones artificiales de C6 de `feat-ind32`.

## Lo que cambió al correr (2026-10-07)

- **El control, medido en Vast:** 28 vCPU (Xeon E5-2660 v4, 31 GB, Vietnam, 0,0948 $/h), 13 procesos × 2 hilos, **8,1 s
  por época**, ~13 min por detector, **16,3 min y 0,0257 $** de alquiler, destruida sola (`resultados/vast/control/control.json`).
  Contra la estimación de ~1–2 h y ~0,1–0,3 $: **en Vast fue más RÁPIDO que en el dev** (8,1 contra 12,8 s/época), porque la
  medida del dev se hizo con la CPU compartida con otras dos pruebas y esta máquina tenía núcleos de sobra (26 hilos en 28
  vCPU). La corrección «hilo con hilo, Vast ~4 veces más lento» que el revisor sacó de `feat-bor` no valió para esta máquina.
- **El compartido, recalculado con ese factor (~0,63):** 54–58 s/época del dev → ~35 s en Vast con 2 hilos → ~1 h con la
  evaluación, ~0,1 $. Por eso pasa de 13 × 4 hilos (≥52 vCPU, ≤0,35 $/h, tope 8 h) a **13 × 2 hilos (≥26 vCPU, ≤0,2 $/h,
  tope 3 h)**: la misma máquina que el control, y dentro de lo que el dueño aprobó (~0,1 $ / ~1 h). Peor caso: 3 h × 0,2 = 0,6 $.
- **El compositor de referencia** (las líneas de `feat-ind32` con el compositor de aquí) da **0,9491**, el 0,949 de
  `feat-ind32`, con sus tres semillas idénticas: el compositor copiado mide igual.
