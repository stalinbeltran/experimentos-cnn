# Plan: kernels de TODOS los tamaños para `banco-k`, entrenados y evaluados en Vast

**2026-10-01 · ES UN PLAN.** ⚠ **Desde el 2026-10-01 por la noche, las fases 0 y 1 están
EJECUTADAS** —el dueño dijo «Ejecútalo» con sus decisiones (1 recomendado · 2 un kernel · 3 tres
semillas · 4 todos · 5 recomendado)—: lo medido está en el
[reporte #25](https://github.com/stalinbeltran/estudios-redes-neuronales/blob/main/reportes/infraestructura/2026/10-octubre/2026-10-01-kernels-banco-fases-0-1.md).
Lo que cambió respecto de lo de abajo: el peaje medido es 1 min 48 s (no 8,4), la deriva Vast↔dev
pasó del umbral (la fase 3 corre los controles en cada máquina), y la fase 2 son 9 máquinas **por
experimento**, no 9 en total. El texto de abajo es el plan tal como se escribió. Lo pidió el
dueño así:

> «Ahí tenemos un banco de kernels. Ahora necesito que crees varios experimentos para
> generar los kernels de los distintos tamaños esperados por el banco. Detalla el plan de
> implementación. Debes usar servers contratados para poder correr esos experimentos en
> paralelo. Aún no ejecutes, solo arma el plan.»

Los experimentos se nombran por su `id` (`banco-k`, `bor-p`, `esq-k`…), nunca por su
carpeta: es la regla R16 del repo y `comprobar.py` la vigila. Los tres experimentos
nuevos todavía no existen; sus ids propuestos son `bor-k`, `bor-ae` y `bor-pca`.

## 0. El plan en una tabla

| # | qué | dónde corre | máquinas | reloj (estimado) | coste (estimado) | depende de |
|---|---|---|---|---|---|---|
| 0a | **dataset a la escala del banco**: `bor-p` con geometría ×4 y páginas reducidas /4, publicado con nombre nuevo | dev | 0 | ~15 min render + publicar | 0 $ | decisión 1 |
| 0b | **lanzador Vast**: modo `trabajo` en `vast_instance.py` (N trabajos → N máquinas, trae un directorio, destruye en `finally`, libro, seco) + descriptor JSON por experimento + puerta `exigir_lanzador()`; **freno + ejecutor Telegram en el mismo commit** | dev | 0 | ~300–400 líneas + tests | 0 $ | — |
| 0c | `banco-k/nn/importar_kernel.py` lee la reserva §3.7 del **manifiesto del dataset** declarado (hoy sin `nn/receta.json` asume fuga) | dev | 0 | pequeño | 0 $ | — |
| 0d | las **tres carpetas**: `bor-k` (supervisado, 4 bordes), `bor-ae` (autoencoder de un filtro), `bor-pca` (componente principal) — cada una con `REGLAS.md`, criterio antes de mirar y `entrenar_local.py` | dev | 0 | trabajo de código | 0 $ | 0a |
| 1 | **validar el lanzador** con una corrida ya conocida del banco (`identidad`, semilla 3) y medir peaje, reloj y deriva numérica Vast↔dev | Vast | 1 | ~15 min | ≈0,01 $ | 0b |
| 2 | **generar**: una máquina por `k` corre `bor-k` y `bor-ae` de ese `k`; `bor-pca` en el dev | Vast + dev | 9 | ~30–40 min | ≈0,25 $ | 0a 0b 0d 1 |
| 3 | **evaluar en el banco**: una máquina por `k`: control aleatorio de ese `k` si falta (6 de 9) + los 3 kernels de ese `k` | Vast | 9 | ~45 min | ≈0,4 $ | 0c 2 |
| 4 | traer `resultados/`, `--informe`, commit; reporte al central **sólo si** mueve `ESTADO.md` | dev | 0 | ~20 min | 0 $ | 3 |

**Total estimado: ≈0,7 $ y ~2 h de reloj de máquinas**, más el trabajo de código de la
fase 0, que es lo grande. Todo «estimado» de aquí sale de números medidos en otros sitios
(§5); nada de esto está medido para estos trabajos.

## 1. De dónde se parte (leído del disco el 2026-10-01)

- **`banco-k` está calibrado y tiene 4 kernels evaluados** (`esqk-k07`, `esqk-k09`,
  `esqk-k11`, `laplaciano`); **ninguno declara**. Controles aleatorios sólo para
  **k = 7, 9, 11**: faltan 6 (3, 5, 13, 15, 17, 19), a ~9 min cada uno *(medido el
  2026-09-08 en el dev: 35 ms/paso, 10 semillas × 1000 pasos)*.
- **Kernels aprendidos que ya existen en el repo**: `esq-k` k03–k11, `esq-2d` k05–k13,
  `esq-cq` k05–k17 (sus 7 brazos terminaron: 301 épocas cada uno, pesos en disco).
  **Todos con fuga §3.7**: sus recetas no fijan fuente ni interlineado, así que
  alcanzan la reserva y salen **optimistas** en el banco (el propio `KERNELS.md` lo
  avisa). Ninguno cubre k = 19. No son lo que se pide, pero están (§6, decisión 5).
- **`bor-p` es el dataset pensado para esto**: 1000 párrafos en 287 páginas de 1024²,
  caja de tinta de 4 bordes, reserva §3.7 **respetada** (`usa_la_reserva=False`),
  reparto `train/val/eval` por página, `ventana_limpia` ≥ 81 px. **Sin red, sin
  criterio, sin ventanas recortadas** — es el encargo que lo dejó así.
- ⚠⚠ **La escala no cuadra, y es la primera decisión.** El banco aplica el kernel
  **después de reducir /4** (§6.1): cuerpo 11–30 px a 584 → **2,75–7,5 px** en el marco
  que ve la red. `esq-k`/`esq-2d`/`esq-cq` aprendieron sus kernels a **/4** (sus datasets
  son `-r4`). `bor-p` está a **1024 sin reducir, con cuerpo 11–30 px**: la tinta es
  **~4× más gruesa** que la que verá el kernel en el banco. *Leído el 2026-10-01 de los
  manifiestos publicados (`reduccion`, `cuerpo_px`).* Y **no vale reducir `bor-p` tal
  cual**: su separación mínima entre párrafos es 41,88 px → 10,5 px a /4, por debajo de
  los 19 que exige una ventana de k = 19. Hace falta un dataset con la geometría ×4 en
  render y guardado a /4 (fase 0a).
- **La máquina**: dev de 2 vCPU / 3,8 GB; `experimentos-cnn` recién clonado hoy y **sin
  `.venv`**; `VAST_AI_API_TOKEN`, `DO_TOKEN` y `GITHUB_TOKEN` presentes; clave
  `~/.ssh/vast` existente y registrada; **Vast vacío** (`list`: ninguna instancia);
  freno `🟢 CERRABLE`.
- **Vast hoy**: 4–6 vCPU con ≥ 8 GB a **0,049–0,062 $/h** *(medido el 2026-10-01 con
  `vast_instance.py offers --cpus 4 --max-cpus 8 --min-ram 8`)*. **Peaje por máquina:
  8,4 min** (arranque + subida + instalar torch; *medido, `estudio_estimar.py:53`*).
  Consecuencia: un trabajo de 9 min paga casi otro tanto de peaje, así que las máquinas
  **se agrupan por `k`**, no por trabajo. `esq-k` ya descartó Vast por esta aritmética
  (42 min-máquina de peaje para 9 min de trabajo); aquí el dueño lo pide y se amortiza
  agrupando.
- **Lo que existe para Vast**: `vast_instance.py` (`launch/list/ssh/destroy`, y `bench`
  de UNA máquina con descriptor `envia/install/run/recoge` que destruye en `finally`;
  `sweep` es secuencial) y `estudio_flota.py` (flota entera, pero atada a los sweeps de
  `fv`). **Nada reparte N trabajos arbitrarios en paralelo ni trae un directorio.** Es la
  fase 0b.

## 2. Las fases, en orden

### Fase 0 — construir (0 $, es lo grande)

**0a · El dataset a la escala del banco.** Ampliar `bor-p` con una reducción declarada:
las constantes de `nn/geometria.py` se derivan de `K_MAX` a la escala **guardada**, así
que con reducción /4 la separación dura en render pasa a 76 px y el margen a 36 px; las
páginas se guardan reducidas por **promedio por área /4** (el mismo método que el banco)
y las cajas se transforman con la misma aritmética. Dato nuevo = nombre nuevo
(`parrafos1000-pagina1024-r4-r<fecha>`), publicado con su manifiesto; el de hoy no se
toca. ⚠ Con celdas 4× más holgadas caben menos párrafos por página (hoy 2–5): se ajusta el
lienzo o se acepta, y se dice en el manifiesto. La reserva §3.7 se sigue comprobando antes
de renderizar. Render: ~11 min *(medido para el dataset actual: 658 s)*.

**0b · El lanzador.** Contrato en §4. Lleva en el **mismo commit** (regla 4 de escritura,
R11): su nombre en `TRABAJOS` de `cerrable.mjs`, y un ejecutor de Telegram `exp-vast`
con `estado` y `apagar <prefijo>` — el freno nunca llega después del acelerador.

**0c · `importar_kernel.py` del banco.** Hoy decide la fuga §3.7 leyendo
`nn/receta.json` del experimento de origen, y sin ese fichero *asume fuga*. Los tres
experimentos nuevos **consumen** un dataset publicado y no tienen receta propia: la
fuente única de qué datos vieron es el **manifiesto del dataset declarado** en su
`experimento.json` (`factores.fuentes`, `interlineado`). Cambio pequeño, con test que
compruebe las dos ramas (con receta y con manifiesto) y que sin ninguna de las dos siga
asumiendo fuga.

**0d · Las tres carpetas**, cada una con lo obligatorio del repo (`experimento.json`,
`REGLAS.md`, `instrucciones/01-encargo.md`, `02-criterio.md` **antes de correr**),
`gasta: "alquila"` (porque van a Vast: obliga a ejecutor de Telegram y
`desacoplar-persistente.sh`), `nn/entrenar_local.py` (el nombre es el contrato con el
freno), `metrics.jsonl` por época (R12: rastro según ocurre), checkpoints con
`conv.weight` de forma `(1, 1, k, k)` para que `importar_kernel.py` los lea sin cambios,
y `nn/lanzar.sh` con seco, `--estado` y negativa a lanzar dos veces. Diseño de cada una en
§3.

### Fase 1 — validar el lanzador con UNA corrida conocida (1 máquina, ≈0,01 $)

Antes de alquilar nueve máquinas se alquila **una** y se corre algo cuyo resultado ya está
en git: `banco-k`, `nn/entrenar_local.py --condicion identidad --semilla 3`. Mide tres cosas
que hoy nadie sabe para este trabajo:

1. que la cadena entera funciona (subir, instalar, correr, traer, destruir);
2. el **reloj real** de un run del banco en Vast (en el dev son ~0,6 min);
3. la **deriva numérica** Vast ↔ dev: `metricas.csv` remoto contra
   `resultados/identidad-s3/metricas.csv`. *Precedente medido en `foveal-vision`: entre
   familias de CPU distintas el entrenamiento diverge (~7·10⁻⁴ en f1); dentro de E5-26 es
   bit a bit.*

**Regla escrita antes de medir**: si |Δ IoU eval| ≤ 0,001 (un quinto de la desviación
entre semillas del banco, 0,006), los kernels se comparan contra los controles ya
calculados en el dev; si es mayor, en la fase 3 **cada máquina corre también `identidad`
y el aleatorio de su `k`**, aunque ya existan, para que todo lo que se compara salga de
la misma máquina (+18 min por máquina, ≈+0,15 $ en total).

### Fase 2 — generar los kernels (9 máquinas, una por `k`)

Cada máquina recibe el repo (sin `.git`, `.venv`, `datos/`, pesos de otros), el dataset
reducido, y corre **el brazo `kNN` de `bor-k` y el de `bor-ae`** en secuencia; `bor-pca`
no entrena y corre en el dev. Vuelven `nn/pesos/kNN/` (`best.pt`, `last.pt`,
`metrics.jsonl`) de cada experimento, se commitean (son ~20 KB por brazo; el tope del repo
es ≈5 MB por experimento) y se importan al banco:

```
python nn/importar_kernel.py --de bor-k  --brazos k03 k05 k07 k09 k11 k13 k15 k17 k19
python nn/importar_kernel.py --de bor-ae --brazos k03 … k19
python nn/importar_kernel.py --de bor-pca --brazos k03 … k19
```

→ `kernels/bork-kNN.npy`, `borae-kNN.npy`, `borpca-kNN.npy` con su `.json`
(`usa_la_reserva: false`, que es toda la razón de usar `bor-p`).

### Fase 3 — evaluar en el banco (9 máquinas, una por `k`)

Cada máquina corre, con el `nn/evaluar_kernel.py` del banco **sin tocar**: el control
`aleatorio-k<k>` si no existe (k = 3, 5, 13, 15, 17, 19) y después los 3 kernels de su
`k`. Vuelven `resultados/<nombre>*/`, `resultados/aleatorio-k*/` y los
`kernels/aleatorio-k*-r*.npy` que haya generado (semilla fija: tienen que dar el mismo
sha que si se generaran aquí; se comprueba). Por máquina: 3–4 grupos × ~9 min + peaje.

### Fase 4 — cerrar

`nn/evaluar_kernel.py --informe` reescribe `KERNELS.md`; se comprueba que el
`contrato.sha256_16` de cada `criterios.json` es el del `.npy` local; commit y push el
mismo día. Reporte en `estudios-redes-neuronales` **sólo si** algún kernel declara y eso
mueve `ESTADO.md` (la pregunta mecánica del repo); si no, se queda en las carpetas.

## 3. Los tres experimentos (lo que se decide aquí; el criterio exacto va en cada `02-criterio.md`, antes de correr)

Los tres leen el **mismo dataset reducido** por `expcnn.exigir_dataset(...)` y recortan
**las mismas ventanas**, de la misma forma: lado **W = 81 px**, el mínimo de
`ventana_limpia` del dataset de hoy (se recalcula para el reducido), para **no filtrar
ningún párrafo** — sortear-y-rechazar sesga hacia párrafos pequeños, y el repo ya pagó esa
lección dos veces. Ventana centrada en un punto sorteado **sobre** el borde (4 positivas por
párrafo, una por borde) más negativas de interior y fondo; el reparto `train/val/eval` se
hereda **por página** del dataset. Con k ≤ 19 el mapa `valid` es ≥ 63×63. Un `nn/datos.py`
por experimento (Regla 0: se copia, no se comparte).

### E1 · `bor-k` — un kernel, cuatro bordes (supervisado; linaje `esq-2d`: un mapa, dos extremos)

- **Estructura**: `M = conv_k(x)` sin bias ni padding → marginal por columnas `c` y por
  filas `r` → `izq = softargmax(+β·c)`, `der = softargmax(−β·c)`, `sup = softargmax(+β·r)`,
  `inf = softargmax(−β·r)`; `existe_t = a_t·logsumexp(±β·marginal)/β + b_t` por borde.
  Cabeza: β + 8 = **9 parámetros**. Es la lectura de `esq-2d` (máximo y mínimo del mismo
  mapa) llevada a los dos ejes: fuerza un kernel con signo **opuesto** en bordes
  enfrentados, que es lo que un filtro lineal puede dar para cuatro orientaciones.
- **Etiqueta**: tipo ∈ {izq, der, sup, inf, ninguno} y **una** coordenada (x o y) del
  borde dentro de la ventana. **Pérdida**: L1 sobre la coordenada del borde presente +
  BCE de los cuatro `existe`.
- **Brazos**: `k03 … k19` (9). **1 semilla** fija, como toda la familia `esq-*` (opción
  en §6). Épocas/`lr`/lote se fijan en su `REGLAS.md` al implementar, no aquí.
- **Métrica propia** (para `best.pt` y su README): error en px por borde y acierto ≤ 2 px
  sobre `val`; «aprendió» = supera el suelo sin entrenar + 2 SE. El veredicto que importa
  lo da el banco, no esto.
- **Riesgo escrito antes**: `esq-2d` midió que un kernel se queda con la esquina barata
  (`tl` 90 %, `br` 23 %). Aquí puede quedarse con un eje. Se reporta por borde, nunca el
  promedio.

### E2 · `bor-ae` — el filtro que mejor reconstruye (autoencoder disperso de UN filtro)

- `z = ReLU(conv_k(x))`, `x̂ = convT_k(z) + b`; pérdida `MSE(x̂, x) + λ·|z|₁` sobre las
  mismas ventanas, **sin etiqueta**. Es el procedimiento que la propia especificación del
  banco cita como productor de kernels (§1).
- Brazos `k03 … k19`, 1 semilla. `λ` se fija **antes** en su `REGLAS.md` (sin barrerlo: un
  barrido de `λ` es otro experimento).
- **Riesgo escrito antes**: con un solo filtro el óptimo tiende a un pasa-bajos parecido
  a `gauss`. Si sale eso, es un resultado y se compara contra `gauss` del banco.

### E3 · `bor-pca` — la dirección principal de los parches de borde (0 $, sin entrenar)

- Parches `k×k` centrados en puntos de borde, media restada; kernel = **primer vector
  propio** de la covarianza. Nueve `k`, sin semilla, segundos en el dev.
- `gasta: "no"`. Sirve de referencia barata entre los clásicos (`gauss`, `sobel`) y los
  aprendidos. Se guarda el PC1 **y** el PC2 (el PC1 de parches de borde suele ser el
  escalón, pero no se da por hecho); al banco entra PC1, y PC2 sólo si se decide antes.

## 4. El lanzador (fase 0b): contrato

| | |
|---|---|
| entrada | un **descriptor** por trabajo (dato, no código): `id` del experimento, comando, qué subir, qué traer, `k`/brazo, etiqueta |
| sube | el repo (sin `.git`, `.venv`, `datos/`, `resultados/` ajenos, pesos ajenos) + el/los datasets publicados, como **tar por SSH** — nunca `git clone`, nunca un token: a una máquina de Vast no se le da ningún secreto |
| en la máquina | `EXPCNN_DATOS=/root/trabajo/foveal-vision-data`, que es la puerta que `expcnn.exigir_dataset` ya respeta: ningún script cambia por correr fuera |
| instala | venv + `torch` del índice CPU + numpy + pillow (el patrón de `benchmarks/foveal-cpu.json`) |
| trae | un **directorio** (`tar` por SSH): `nn/pesos/kNN/` o `resultados/<nombre>*` — no un JSON suelto, que es lo único que `bench` sabe traer hoy |
| destruye | en `finally`, siempre; si falla, **lo dice con el comando** `destroy` a mano, como `medir_en_oferta` |
| paralelo | **una unidad de systemd por trabajo** vía `desacoplar-persistente.sh` (padre PID 1: sobrevive al fin del turno y al restart del bot) |
| etiqueta | `expc-<id>-<brazo>` con prefijo del `WORKSPACE.json` si lo hay: es lo que `cerrable.mjs` usa para contar máquinas **mías** |
| `--estado` | lee **disco** (un JSON por trabajo que se escribe en cada paso: alquilada / subida / instalada / corriendo / traída / destruida, con hora y coste), nunca el log — R12 |
| seco | imprime unidad, oferta y orden **sin alquilar**; va **antes** del guardia de doble lanzamiento |
| se niega | si el dataset no está publicado, si falta el token, o si ya hay una unidad viva con la misma etiqueta: **antes** de alquilar (R2) |
| aviso | `notify.mjs … \|\| true` al final de cada unidad |
| tests (R10, primero lo caro) | 1) la destrucción corre aunque falle el `run` (simulada); 2) el seco no toca la API; 3) el payload no lleva `.git`, `.venv`, `datos/` ni `.env`; 4) el despacho por modo (copia de `probar_lanzador.sh`) |

**Dónde vive** — decisión de estructura (§0 de `reglas-de-diseno.md`: R1, R4, R7, R18).
Revisado con el agente `arquitecto` el 2026-10-01: ver §4.1.

### 4.1 Dónde vive el lanzador

**Recomendación del `arquitecto` (2026-10-01): opción A, partida en dos.** El
**mecanismo** —alquilar N máquinas físicas distintas, subir, instalar, correr, traer un
directorio, destruir en `finally`, libro— va como modo nuevo **`trabajo`** en
`vast_instance.py` del lanzador: es el único programa que ya sabe hablar con Vast y el
que ya protege el dinero (`elegir_ofertas_distintas`, máquinas bloqueadas, tope de
precio, destrucción en `finally`, arnés de tests). Lo que **cambia por experimento** —qué
se envía, qué se instala, qué se corre, qué directorio vuelve, cuántos trabajos por
máquina— va como **un JSON dentro de cada experimento** (dato, no código: R18; el patrón
de `benchmarks/*.json`, con el descriptor fuera del lanzador). Los resultados y el libro
vuelven a la **carpeta del experimento** (R7), no a `results/` del lanzador. Pasar de «una
máquina por brazo» a «una por `k`» es editar ese JSON.

- **R4/R2**: cuarta puerta en `expcnn/entorno.py`, `exigir_lanzador()`
  (`EXPCNN_LANZADOR` > hermano `../digital-ocean-dropplet-auto-launching` > se niega),
  que además comprueba que el subcomando `trabajo` **existe**: un clon viejo del lanzador
  no lo tiene y tiene que negarse antes de alquilar.
- **R5**: el contrato es la CLI (un descriptor JSON de entrada, un libro JSON de
  salida). Nada de `sys.path.insert` ni de leer un log pensado para humanos.
- **R6**: el dataset se lee de `foveal-vision-data` y se deja en una ruta remota
  distinta, donde `exigir_dataset` lo encuentra por `EXPCNN_DATOS`; se escribe en los
  dos lados y con test, y lo primero que hace la `run` remota es casar las huellas contra
  `manifiesto.json`.
- **R11**: en el mismo commit, `vast_instance\.py trabajo` entra en `TRABAJOS` **y en
  `VIGILANTES`** de `cerrable.mjs`; si no, el freno diría «SIN VIGILANTE» y propondría
  `adoptar_vast.py`, que no conoce estos trabajos. Y el ejecutor `exp-vast`.
- **R10**, con un doble de `api()`: N trabajos → N `machine_id` distintos; cada iid se
  destruye aunque fallen `run` o la recogida; el libro está en disco **antes** de que
  acabe el primer trabajo; `--dry-run` hace 0 `PUT /asks/`; sin `--prefijo` se niega
  sin llamar a la API; si falta el subcomando, la puerta se niega antes de alquilar.

**Descartadas**: (B) un script propio en la raíz de `experimentos-cnn` por `subprocess`
— reescribe tar, subida, recogida, paralelo y libro; `launch` coge `ofertas[0]`, así que
N lanzamientos chocan en la misma oferta; y sale con 0 aunque la máquina no llegue a
`running` (`vast_instance.py:1213,1226`) —; (C) una copia del lanzador por experimento —
N `finally: destruir` sin tests que divergen; la Regla 0 protege condiciones, no obliga a
copiar el acelerador —; (D) extender `estudio_flota.py` — arrastra `fv`, `windows.npz` y
2.075 líneas sin tests al repo que se creó para salir de ahí —.

Coste estimado del modo `trabajo`: ~300–400 líneas más tests, en 3 repos, sin reiniciar
nada.

### 4.2 Las tres preguntas del gasto (las que el triage exige antes de lanzar)

| | |
|---|---|
| **qué se alquila** | fase 1: 1 máquina; fases 2 y 3: 9 máquinas cada una, 4–6 vCPU, ≥ 8 GB, con GPU que no se usa (en Vast no hay otra forma), ~0,05 $/h cada una, ~0,5–0,75 h de vida |
| **cuánto cuesta** | ≈0,7 $ en total con 1 semilla; ≈2 $ con 3 (§5, estimado) |
| **quién apaga si el dev muere** | la unidad y su `finally` mueren con él y las de Vast **siguen facturando**. Lo que sobrevive es la **cuenta de Vast**, que es la fuente de verdad: desde cualquier máquina con el token (el mini, por su bot) `vast_instance.py list` las enseña y `destroy <etiqueta>` o `--all` las apaga. El libro se **commitea y empuja al alquilar** (como el `--git` de `estudio_flota.py`) para que el server siguiente sepa qué etiquetas eran suyas; y mientras el dev vive, `--horas-max 2` acota cada máquina |

## 5. Coste y reloj, con la procedencia de cada número

| número | valor | de dónde sale |
|---|---|---|
| precio Vast | 0,049–0,062 $/h | medido 2026-10-01, `offers --cpus 4 --max-cpus 8 --min-ram 8` (6 primeras) |
| peaje por máquina | 8,4 min | medido, `foveal-vision/scripts/estudio_estimar.py:53` |
| un grupo del banco (10 semillas) | ~9 min en el dev | medido 2026-09-08 (35 ms/paso); **en Vast no medido** → fase 1 |
| un brazo de `bor-k` (300 épocas, W=81) | **10–17 min, estimado** | escalado de `esq-cq` (0,31–0,53 s/época a 32×32, ×6,4 píxeles); no medido |
| fase 2 | 9 máq × (~25 min + 8,4) ≈ 0,5 h × 0,055 $/h | ≈ **0,25 $** |
| fase 3 | 9 máq × (3–4 grupos × 9 min + 8,4) ≈ 0,75 h × 0,055 $/h | ≈ **0,4 $** |
| con 3 semillas en E1/E2 | fase 2 ×3 de cómputo; fase 3 ×3 de kernels (27 por experimento) | ≈ **2 $** en total |

## 6. Decisiones que tiene que tomar el dueño ANTES de la fase 0

1. **Escala** (§1): ¿dataset nuevo a /4 con geometría ×4 (recomendado: es la escala del
   banco y de toda la familia `esq-*`), o aprender a 1024 y que el banco mida si cruza de
   escala? Lo segundo contradice el §6.1 de la especificación («la escala es coherente»).
2. **E1**: ¿un kernel para los cuatro bordes (lectura ± por eje, 9 kernels) o un kernel por
   borde (36 kernels, el banco los evalúa uno a uno)? Recomendado: uno.
3. **Semillas**: 1 (como la familia, ≈0,7 $) o 3 (≈2 $, y permite barras de error del
   productor, no sólo del banco).
4. **Cuáles**: los tres, o sólo `bor-k`. `bor-pca` cuesta 0 $; `bor-ae` es el que más
   probablemente dé un pasa-bajos.
5. **Los ya entrenados** (`esq-cq` k13–k17, `esq-k` k03/k05): importarlos al banco cuesta
   0 de entrenamiento y 9 min por kernel, pero **todos salen optimistas** (fuga §3.7).
   Recomendado: no, salvo como curiosidad marcada.

## 7. Riesgos y lo que este plan NO cubre

- **Peaje**: con trabajos de 10–20 min, el 30–45 % del coste es arranque. Agrupar por `k`
  lo reparte; no lo quita.
- **Deriva numérica entre máquinas**: medida en la fase 1, con la regla ya escrita.
- **El dataset del banco viaja a máquinas ajenas** (como ya hace `estudio_flota.py` con
  `windows.npz`): es dato derivado del generador, no un secreto; los tokens no viajan.
- **Ofertas que desaparecen entre buscar y alquilar** (`no_such_ask`): el lanzador
  reintenta con la siguiente, como `elegir_ofertas_distintas`.
- **Lo que no se mide aquí**: nada sobre página entera, nada sobre `tr`/`bl`, y el banco
  no compara eficiencia entre `k` (§15 de su especificación).
- **No se ha ejecutado nada**: ni fase 0, ni venv, ni una sola máquina. Ejecutar empieza
  por las decisiones de §6 y por la fase 0a/0b, que son 0 $.
