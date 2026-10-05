# Reglas de `feat-ind32`

**Plan del 2026-10-05, IMPLEMENTADO ese mismo día** con las dos decisiones del dueño: **mapa de salida 8×8**
(opción A del §2) y **Vast**, con *«todo lo que es paralelo debe correrse en paralelo»*. El criterio, escrito
antes de entrenar, está en `instrucciones/02-criterio.md`.

**Qué pregunta:** lo mismo que `feat-ind`, pero sin reducir la imagen. Cada CNN pequeña reconoce una sola
feature fijada de antemano. Las 13 se entrenan por separado y un compositor las junta sobre dígitos
manuscritos. Allí todo vivía en 8×8 (bloques 4×4 contados); aquí el detector ve el **bitmap binario de
32×32** del que salía ese 8×8.

**Por qué puede cambiar algo** (contexto, no criterio): a 8×8, `feat-ind` dejó arcos y esquinas en «no
aprendió» (F1 0,70–0,74, se disparan entre sí: corrida 2) y atribuyó un tercio del error sobre dígitos
al grosor (1↔8). Las dos cosas son de resolución: un arco de radio 4 px ocupa 1 celda a 8×8 y 8 px a
32×32.

## Entradas

### 1. El dato: comparable, y comprobado

- **Dígitos:** UCI *optdigits-orig*, el original sin reducir (bitmaps binarios 32×32, preprocesado NIST).
  Se descargó de `archive.ics.uci.edu/static/public/80/…zip` (HTTP 200, 591 KB) el 2026-10-05. Trae 4
  ficheros: `tra` 1934 · `cv` 946 · `wdep` 943 (30 escritores) · `windep` **1797** (13 escritores
  distintos).
- ⚠ **`windep` son LOS MISMOS 1797 dígitos que `uci-optdigits-8px-r20261002`.** Medido el 2026-10-05:
  al reducir cada bitmap de `windep` contando bloques 4×4, las **1797 de 1797** filas coinciden exactas
  (los 64 valores y la etiqueta) con `optdigits.tes`, en el mismo orden. Así que **el reparto de
  `feat-ind` (180/1617, y el 900+717 de la corrida 11) se puede reproducir índice por índice**, y los
  números de los dos experimentos se pueden comparar directamente. ✅ Y comprobado contra el dataset **publicado** (`uci-optdigits-8px-r20261002`, que salió de scikit-learn): los
  1797 coinciden en imagen, etiqueta y orden, así que su reparto 180/1617 se copia índice por índice
  (`nn/datos.py --comprobar`).
- **Extra que a 8×8 no se usó:** 3823 dígitos 32×32 de **otros 30 escritores** (`tra`+`cv`+`wdep`). Sirven
  para la curva de datos (corrida 11 llegó a 0,99 con 900) sin tocar el test. Se usan sólo en un brazo
  aparte (§4, C5), para que el brazo principal siga siendo comparable.
- **Features sintéticas:** `feat-ind` ya las dibuja a 32×32 y luego las reduce (`ESPECIFICACION.md` §1).
  Aquí **no se reduce**: se guarda la máscara binaria 32×32 con el mismo ruido `p_on`/`p_off`. Vocabulario,
  rangos, segunda feature, relación `CONTIENE` y la cantidad por familia (2000+400) son los mismos, **por
  decisión, no por herencia**: si no, no se sabría si un cambio en el resultado viene de la resolución
  o del vocabulario.
- **Publicación:** dos datasets nuevos en `foveal-vision-data/experimentos-cnn/` (que vive en el
  almacén), nunca copiados aquí: `feat-ind32-sinteticas-32px-r<fecha>` (**2,2 MB** en disco, medido) y `uci-optdigits-orig-32px-r<fecha>` (**340 KB**, medido). ✅ Las sintéticas, reducidas 4×4, dan **bit a bit** `feat-ind-sinteticas-8px-r20261003`:
  se generan con el mismo sorteo y sólo se omite la reducción. ⚠ Antes de publicar, `/use almacen` → `estado`, porque se llenó el 2026-10-03.

## Salidas

Pesos en `nn/pesos/<feature>/` (best/last, config, metrics, summary), igual de forma que en `feat-ind` para comparar; `resultados/*.json` por corrida; los `*.npz` no se commitean. Toda corrida que alquila deja su reporte en el repo central (`estudios-redes-neuronales`), cambie o no `ESTADO.md` (regla del coordinador; #30, #31…).

### 2. La salida del detector: ¿mapa 8×8 o 32×32? — DECIDIR

| opción | qué es | a favor | en contra |
|---|---|---|---|
| **A (propuesta)** | entrada 32×32, dos convoluciones con stride 2 → **mapa 8×8**; mismo objetivo gaussiano σ 0,6 celdas | el compositor es **idéntico** (832 entradas) y toda la cadena de `feat-ind` (compositor, curva, desplazamiento) se compara uno a uno: **sólo cambia lo que ve el detector** | la posición sigue a resolución de celda |
| B | mapa 32×32, objetivo σ 2,4 px | posición 4× más fina | compositor de 13 × 1024 = 13.312 entradas con 180 dígitos de train: se compara contra otra cosa |

**Elegida A** (el dueño, 2026-10-05). B queda escrito y sin correr.

**Red de A** (`nn/modelo.py`, 41.825 parámetros): conv 3×3 16 → conv 3×3/2 32 → conv 3×3/2 32 → 3 × conv 3×3 32 →
1×1 → 1 canal 8×8, con padding. ⚠ **El ensayo C2 del plan NO se hizo**: la profundidad se fijó para que el campo
receptivo (33 px) cubra lo mismo que el de `feat-ind` (7×7 celdas = 28 px), y así no hay que esperar a una
corrida antes de lanzar las 13 en paralelo. Si C3 sale mal por capacidad, ese es el primer sospechoso. Pérdida: BCE por celda (objetivo ×8) + BCE sobre el máximo, umbral por detector elegido en
train, mejor F1 de val. Las dos enmiendas de `feat-ind` (`CONTIENE` y umbral por detector) entran desde
el principio, porque allí se midió que sin ellas el criterio mide otra cosa.

## Procesos

### 3. Criterio — se escribe en `instrucciones/02-criterio.md` ANTES de generar nada

Lo que propongo escribir allí (§A y §B con los mismos umbrales que `feat-ind`, porque la pregunta es
justo «¿cambia al subir la resolución?»):

- **Por detector:** «aprendió» si F1 ≥ 0,90 y posición ≤ 1 celda ≥ 0,90. Más las tres lecturas
  obligatorias (radio, FP por familia, grosor).
- **Hipótesis H1 (detectores):** arcos y esquinas pasan de «no aprendió» a ≥ «a medias». Se dice
  confirmada si al menos 6 de los 8 suben de categoría.
- **H2 (compositor, el reparto 180/1617 de la corrida 2):** posicional ≥ 0,959 + 0,01. Con 3 semillas y
  sd ~0,001 en 8×8, 0,01 es mucho más que el ruido.
- **H3 (techo, el reparto 900/717 de la corrida 11):** mejor que 0,992 — o, si no se puede, decir que el
  techo no era de resolución.
- **H4 (1↔8):** las confusiones 1↔8 bajan de 22 (corrida 2) a ≤ 5 sin que suba otro par (el 4→1 de la
  corrida 4).
- **Lo que espero hoy:** H1 sí en arcos, esquinas a medias; H2 marginal (+0,005 a +0,015), porque a 8×8 el
  compositor ya compensa. Si sale peor que 8×8, también es un resultado: a 32×32 el manuscrito es más
  irregular que el sintético y la transferencia puede empeorar.

### 4. Las corridas, en orden (cada una con su criterio antes)

| # | qué | reproduce de `feat-ind` | coste *(estimado)* |
|---|---|---|---|
| C1 | generar y publicar los 2 datasets; rejilla de muestras; comprobar el §1 | — | minutos |
| ~~C2~~ | ~~ensayo de red~~: sustituido por el criterio del campo receptivo (§2) | — | — |
| C3 | los 13 detectores, 1 semilla, **los 13 a la vez en una máquina de Vast** (`nn/vast.sh detectores`) | corrida 2 | medido en el dev: **11 s por época** con 1 hilo → ~15 min por detector; en Vast, 13 procesos a la vez ≈ 20–40 min, **≈ 0,1–0,3 $** *(estimado)* |
| C4 | `nn/aplicar.py` (los 5620) + `nn/componer.py`: presencia/posicional 180/1617 y pares 6↔9, 1↔8, 4→1 | corrida 2 | 41 s + 72 s medidos (con pesos de prueba) |
| C5 | curva por N (test fijo de 717) y con los 3823 de otros escritores — en `componer.py` | corrida 11 | ídem |
| C6 | G1/G2/G3 con A y B (máx 3×3), y píxeles crudos 32×32 — en `componer.py` | corridas 12 y 13 | ídem |

**Lo que NO se reproduce:** corrida 3 (contra-casos, no se adoptó), 4/5 (trazos gruesos: a 32×32 el grosor es
otra cosa; se decide tras C3 por la lectura de grosor), 6–8 (grupos `dig`/`cae`: no aportaron, corrida 11),
9–10 (sobreajuste y compositor combinante: no movieron nada). Si el dueño quiere alguna, se añade.

**Dónde corre:** C3 en **Vast** (decisión del dueño), con el modo `trabajo` del lanzador: una unidad de systemd
alquila, sube el repo y el dataset, corre los 13 procesos, trae los pesos y **destruye la máquina en un
`finally`**, con tope de **3 h** (`--horas-max 3`). El libro (`resultados/vast/detectores/`) se commitea al alquilar.
Apagar desde cualquier máquina: `nn/vast.sh apagar`, o desde Telegram `/use exp-vast`. C4–C6 en el dev (minutos).

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/features.py` | vocabulario + rasterizador (copiado de `feat-ind`; aquí la muestra es el bitmap 32×32) | `python nn/features.py [--muestras]` |
| `nn/datos.py` | genera/publica las sintéticas y los dígitos; comprueba las dos igualdades con los de 8 px | `--generar [--publicar]` · `--digitos <dir> [--publicar]` · `--comprobar` |
| `nn/modelo.py` | el detector (autocontenido) 32×32 → mapa 8×8 | `python nn/modelo.py` |
| `nn/entrenar_local.py` | entrena UN detector; `--comprobar` el mecanismo (incluye 1 época real) | `--feature f [--semilla s] [--epocas n] [--hilos h]` · `--comprobar` |
| `nn/vast.sh` | Vast, cuatro modos: `detectores` (C3), `cnn` (curvas de CNN), `compositor` (A, B y C, dos máquinas) e `igualada` (la CNN del repo, tres peldaños); se niega si el dataset del modo no está empujado o si su comprobación falla | `detectores` · `cnn` · `compositor` · `igualada` · `--estado` · `apagar` · `VAST_SECO=1 …` |
| `nn/probar_vast.sh` | prueba en seco del lanzador: cada modo construye SU orden y uno desconocido se niega (con el `vast.sh` de un solo modo, los 4 casos de `cnn` fallan) | `nn/probar_vast.sh` |
| `nn/vast.json` | el descriptor del trabajo (máquina, qué se envía, qué se trae) | — |
| `nn/vast-cnn.json` | el descriptor de las curvas de CNN: una máquina de 24–48 vCPU, un proceso de 1 hilo por vCPU pagada | — |
| `nn/vast-compositor.json` | el descriptor de A, B y C: dos trabajos (B sola; C + A), cada uno en una máquina de 32–64 vCPU | — |
| `nn/vast-igualada.json` | el descriptor de la escalera: una máquina de 32–64 vCPU, los tres peldaños | — |
| `nn/init-detectores-8px.pt` | los 13 detectores de la corrida 2 de `feat-ind`, apilados para el banco agrupado (punto de partida de C); huella 792edc8b65501166, con el origen de cada uno | lo escribe `nn/curvas_cnn.py --exportar-init` |
| `nn/aplicar.py` | los 13 sobre los 5620 dígitos → `resultados/mapas-digitos.npz`, firma, rejilla | `python nn/aplicar.py` |
| `nn/componer.py` | C4, C5 y C6 de una vez | `python nn/componer.py` |
| `nn/factor.py` | factor F(N) = no vistos acertados ÷ N (4540 no vistos), detectores y píxeles de 32×32 | `python nn/factor.py` → `resultados/factor.json` |
| `nn/figura_factor.py` | la figura de F(N) con los 4 casos (lee el `factor.json` de `feat-ind` por su id) | `python nn/figura_factor.py` |
| `nn/ganancia.py` | G = acierto en val ÷ (train/(train + val)) con datasets balanceados de 500–4000 dígitos; `particion()` copiada de `feat-ind` | `python nn/ganancia.py` → `resultados/ganancia.json` |
| `nn/figura_ganancia.py` | la figura de G en tres paneles; se niega si los dos experimentos no evaluaron los mismos dígitos | `python nn/figura_ganancia.py` |
| `nn/muestras_necesarias.py` | N(ε): muestras para llegar a un acierto ε (%) sobre los nuevos, desde los datos de la ganancia (no entrena); tabla, rango entre semillas, el ejemplo leído en horizontal y en vertical, comprobación por tamaño de dataset y figura de 3 paneles (N(ε), su inversa y la comprobación) | `python nn/muestras_necesarias.py` → `resultados/muestras-necesarias.json` y `.png` |
| `nn/comparar_cnn.py` | comparación con las CNN ya entrenadas del repo (`dim-nist`, `ruido-comb`, por id; mismo 180/1617, comprobado) + logística sobre píxeles; muestras equivalentes con las curvas de la ganancia; no entrena CNN | `python nn/comparar_cnn.py` → `resultados/comparacion-cnn.json` y `.png` |
| `nn/cnn.py` | las redes de punta a punta, autocontenidas: la CNN de 3 capas de `ruido-nist` (copiada), LeNet-5 sobre 32×32, A (`cnn3pos`), B y C (`Banco13`: los 13 detectores como convoluciones agrupadas + el compositor) y la escalera de la CNN del repo (`CNN3Igualada`: + padding, + cabeza densa, + capacidad) | `python nn/cnn.py` |
| `nn/curvas_cnn.py` | las curvas de las redes de punta a punta sobre las 60 particiones de la ganancia (comprobadas por huella); un json por entrenamiento, se salta lo hecho; exporta el punto de partida de C comprobando que el banco agrupado reproduce los detectores | `--comprobar` · `--exportar-init` · `--todas --modelos a,b [--procesos K]` · `--una m T p s` · `--resumen` |
| `nn/figura_curvas_cnn.py` | las seis curvas (4 de la ganancia + 2 CNN) sobre las mismas particiones (comprobado por huella), N(ε) y la evaluación del criterio | `python nn/figura_curvas_cnn.py` → `resultados/curvas-cnn-comparacion.json` y `resultados/curvas-cnn.png` |
| `nn/figura_compositor.py` | A, B y C contra los detectores congelados y las CNN (mismas particiones, comprobado), la fase congelada de C aparte, y la evaluación del criterio | `python nn/figura_compositor.py` → `resultados/compositor-comparacion.json` y `resultados/compositor.png` |
| `nn/figura_igualada.py` | la escalera de la CNN del repo (+ padding, + cabeza densa, + capacidad y el control 3b) contra C, A y LeNet-5 (mismas particiones, comprobado) y la evaluación del criterio | `python nn/figura_igualada.py` → `resultados/igualada-comparacion.json` y `resultados/igualada.png` |

- ⚠ El que entrena **se llama `entrenar_local.py`** (contrato con el freno), aunque aquí entrene en Vast.
- **Dependencias:** el `.venv` de la raíz (torch 2.14.1 CPU, numpy 2.5.3, Pillow 12.3.0); el de Vast, las mismas.

## Qué NO hereda

- **Se copia de `feat-ind`** el vocabulario, los generadores y la estructura de `nn/`, para no
  reescribirlos. Se copian, no se importan.
- **Se cambia a propósito:** la resolución de entrada, la red (strides), los datasets, el reparto extra con
  otros escritores.
- **Se conserva a propósito**, para poder comparar: vocabulario, cantidades, umbrales del criterio y
  repartos de dígitos.
- **`feat-pos` no se reproduce:** es sólo un documento. La parte que se puede medir aquí (cuánto aporta la
  posición) sale de C4 (presencia contra posicional), igual que en `feat-ind`.
