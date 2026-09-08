# Banco de evaluación de kernels — `banco-k`

**Un instrumento de medida, no un experimento particular.** Mide cuánto aporta un kernel de
convolución, aplicado como **preprocesamiento de las entradas**, a la **generalización** de una
CNN de referencia de 5.812 parámetros entrenada con **100 muestras**.

El banco es **agnóstico al origen del kernel**: el kernel entra como **dato** (`.npy`), y los
procedimientos que producen kernels quedan **fuera de alcance**.

> 📄 **La fuente de verdad es [`ESPECIFICACION.md`](ESPECIFICACION.md)** — la especificación
> **v1.2** del dueño, copiada verbatim. Este README no la resume para reemplazarla; donde
> difieran, gana ella.
>
> ⚠ **Cada versión se publica en su propio `uuid`**: el enlace de la v1.0 sigue sirviendo la v1.0.
> Por eso la especificación de este experimento es **esta copia**, no una URL.

## ✅ Estado: CALIBRADO. El banco discrimina y está listo para evaluar kernels

**2026-09-08.** La calibración del §11 pasa **los seis pasos**, con **41 corridas** (caja
media + identidad ×10 + aleatorio ×10 + gauss ×10 + sobel ×10) sobre el dataset publicado
`parrafos1000-584px-r4-r20260908b`. Informe completo, regenerado del disco y sin
transcribir nada a mano: [`resultados/CALIBRACION.md`](resultados/CALIBRACION.md).

| | IoU sobre `eval` |
|---|---|
| **piso** — caja media (§10.1) | **0.2479** (umbral ≤ 0,40) |
| **techo** — identidad (§10.1.1) | **0.7981 ± 0.0057** (techo 0,95) |
| **rango útil** | **0.5501** = **96 ×** la desviación entre semillas |

**El listón que un kernel tiene que superar para declararse útil sale de aquí, no de una
elección posterior:** el aleatorio (k=9, 10 semillas) da
**0.8083 ± 0.0077**, así que el margen del §2.1
—la **suma** de las dos desviaciones— es de **0.0155**.

⚠ **Y hay una cosa que la calibración ya dice sobre los kernels clásicos:** gauss
(0,8130 ± 0,0086) y sobel (0,8164 ± 0,0064) **no superan al aleatorio**
(0.8083 ± 0.0077) por el margen del §2.1. O
sea que en este banco **filtrar con un detector de bordes clásico no es mejor que filtrar
con ruido** — que es exactamente la confusión que el control aleatorio existe para separar
(§10.3).

### Las muestras, para revisar sin correr nada

Tres PNG en [`muestras/`](muestras/), condensados a propósito — una rejilla por fichero, no
un fichero por muestra:

| | |
|---|---|
| `parrafos-20-marco146.png` | 20 muestras **como se guardan** (146 × 146), con la etiqueta en rojo y el recorte a 128 en azul |
| `parrafos-20-marco128.png` | las mismas **como las ve la red** en la identidad, con la etiqueta ya transformada por `(coord/4) − 9` |
| `condiciones-4x6.png` | **6 muestras × las 4 condiciones** (identidad · gauss · sobel · aleatorio): lo que cada kernel le hace a los mismos píxeles |

⚠ **La tercera es la que enseña el invariante del §6.2**: las cuatro condiciones ven
*exactamente* los mismos píxeles del original —el descarte es siempre 9 px por lado, sea cual
sea `k`—, así que cualquier diferencia entre filas es del kernel y de nada más.

⚠ **Y enseña también por qué las condiciones están tan juntas** (0,7981 a 0,8164, o sea 2,4 ×
la desviación, contra las 96 × del rango útil): la caja del párrafo **sobrevive a cualquier
filtro**. El banco separa con holgura *filtrar* de *no mirar*, y con mucho menos margen *un
filtro de otro*.

## Si estás leyendo esto en un server NUEVO: cómo se retoma

**Todo lo que hace falta está en git.** *Verificado el 2026-09-08, justo antes de destruir la
máquina donde se produjo.*

```bash
# 1. los dos repos (el de datos es PRIVADO: hace falta GITHUB_TOKEN)
git clone …/experimentos-cnn && git clone …/foveal-vision-data
# 2. el entorno: torch + numpy
cd experimentos-cnn && uv venv .venv && uv pip install torch numpy pillow
# 3. comprobar que la cadena está entera, sin entrenar nada
python3 comprobar.py                      # el dataset declarado tiene que estar publicado
python nn/entrenar_local.py --comprobar    # modelo + pipeline + métricas
python nn/probar_lanzador.sh               # el despacho del lanzador
```

**Y lo que prueba que las comparaciones pueden seguir**, que es distinto de que los ficheros
estén: recalcular una corrida guardada y ver que sale **idéntica**.

```bash
python nn/entrenar_local.py --condicion esqk-k11 --kernel kernels/esqk-k11.npy --semilla 3
# tiene que dar iou_eval=0.842944 y brecha=+0.023870, que es lo que hay en
# resultados/esqk-k11-s3/metricas.csv
```

*Comprobado el 2026-09-08: idéntico hasta el último decimal.* Es lo que permite **comparar un
kernel nuevo contra los ya medidos sin volver a correr ninguno** — y también rehacer cualquiera
si hiciera falta.

| qué | dónde | |
|---|---|---|
| el **dataset** (1000 párrafos, 2,2 MB) | `foveal-vision-data/experimentos-cnn/parrafos1000-584px-r4-r20260908b/` | ✅ empujado |
| los **kernels** evaluados y los controles | `kernels/*.npy` + su `.json` de procedencia y hash | ✅ commiteados |
| las **métricas** de las 91 corridas | `resultados/*/metricas.csv` · `resumen.json` · `criterios.json` | ✅ commiteadas |
| los **informes** | `resultados/CALIBRACION.md` y `resultados/KERNELS.md` | se **regeneran** del disco |
| los **pesos entrenados** | **no existen, a propósito** — ver `REGLAS.md` § Salidas | — |

⚠ **Los pesos de los que salieron los kernels sí están**, que son los que no se regeneran: viven
en el experimento `esq-k` de este mismo repo. *Comprobado el 2026-09-08 que los tres **cargan**,
y que el `.npy` evaluado es bit a bit el `conv.weight` de su checkpoint.*

## Cómo se mete un kernel a probar

Es lo único que hay que saber para usar el banco. Todo lo demás está fijado y **congelado**
(§12).

```bash
python nn/evaluar_kernel.py --contrato kernels/mio.npy   # sólo valida el §5, no entrena
python nn/evaluar_kernel.py --kernel   kernels/mio.npy   # lo evalúa entero (~9 min)
nn/lanzar.sh kernel kernels/mio.npy                      # igual, pero como unidad de systemd
```

**El contrato de entrada (§5.1), y se comprueba antes de entrenar nada:**

| | |
|---|---|
| fichero | `.npy` |
| forma | `(k, k)`, cuadrado |
| tipo | `float32` (se convierte si hace falta, avisando) |
| `k` | **impar**, `3 ≤ k ≤ 19` |
| canales | 1 → 1 |

**No hace falta normalizarlo.** El banco normaliza la norma L2 al recibirlo (§5.4) y guarda
la norma **original** y el hash de **antes** de normalizar (§13.3). Dos kernels que sólo se
diferencian en escala son **el mismo detector** y dan exactamente el mismo resultado.

**Qué sale:** [`resultados/KERNELS.md`](resultados/KERNELS.md) —la tabla de todo lo evaluado,
que **se regenera** con `--informe`— y `resultados/<nombre>/` con `metricas.csv` por semilla,
`config.json`, `resumen.json` y **`criterios.json`** — la evaluación de §2.1 y §2.2 por separado, más el
mecanismo del §2.3 (facilitación o transferencia) y un veredicto.

⚠ **El control aleatorio tiene que ser del mismo `k`** (§2.1: «igual norma y **mismo `k`**»).
La calibración corrió el suyo con `k=9`; si tu kernel tiene otro, el script **corre primero**
un aleatorio nuevo con ese `k` y sus 10 semillas. Compararte contra un aleatorio de otro
tamaño mezclaría la **forma** del kernel con su **campo receptivo**, que es otra pregunta.
Cuesta 10 corridas más y se paga una sola vez por `k`.

⚠ **De dónde salga el kernel no es asunto del banco** (§1, §15): sólo se pide el contrato.
**Pero si lo obtuviste con el mismo generador de párrafos, respeta la reserva del §3.7** —
`LiberationMono` y el interlineado `[1,45 · 1,60]` son de uso exclusivo del banco — o tendrás
**fuga de distribución aunque las muestras sean otras**.

⚠ **Hoy `kernels/` sólo tiene los controles** y es correcto: el banco es agnóstico al origen
del kernel (§1) y los métodos para obtenerlos están **fuera de alcance** (§15).

**Concluida la calibración, los parámetros quedan congelados** (§12).

### El generador: comprobado, y NO hizo falta cambiarlo

**2026-09-08.** Los **siete factores del §3.5** salen del generador tal cual está: las **5
familias tipográficas** registradas, cuerpo, interlineado, nivel de gris, ancho y densidad. Lo
que hubo que cambiar es **cómo se le pide**.

⚠⚠ **El alto de un párrafo no se puede pedir: emerge.** Medido: 240 palabras a 28 px dan
**2195 px** de alto sobre un lienzo de 584. Así que «muestrear primero el tamaño» (§3.4) se hace
al revés de como suena — se sortea el **alto objetivo** y se **deriva** el número de palabras —, y
cada muestra se renderiza **dos veces**: una en la esquina mínima para **medir** la caja real, y
otra en la esquina sorteada **para esa caja**. Sin el primer pase no se conoce el rango de esquina
válido, y sin ese rango la única salida sería rechazar, que es lo que §3.4 prohíbe.

**Resultado: 0 rechazadas.** Lo que hay son **reparos de densidad** (~1,5 %), que **conservan** la
muestra bajándole las palabras hasta que cabe, y se cuentan en el manifiesto.

Y dos correcciones de lectura, las dos por leer el código del generador:

- **`Box` es `(x, y, w, h)`**, no `(x0, y0, x1, y1)`. El cuarto número es el **alto**.
- **`placement.area` sí resta el tamaño** (`hi_y = max(y0, y1 - rh)`), pero uno **estimado**, y
  para párrafos la estimación se queda corta. No es que no acote la caja: **la acota con un número
  equivocado**. Da igual para la decisión —no se puede confiar en él— pero el diagnóstico correcto
  es otro. La esquina se fija con `placement.x/y`, que el resolver respeta literalmente.

**Dos decisiones de este experimento**, con su motivo: la **etiqueta es la caja de TINTA** (unión
de cajas de palabra), no la de maquetación, y el texto va **`justify`**. Las dos por lo mismo:
cada uno de los cuatro bordes tiene que tener **evidencia visual**, o el soft-argmax busca algo
que no está ahí.

## Lo que ya está comprobado ejecutándolo

*Medido el 2026-09-08 en esta máquina (2 vCPU, 3,8 GB RAM). Coste: 0 $.*

- ✅ **El recuento del §7.1 es exacto:** 5.812 parámetros y **68** en la cabeza, clavados.
- ✅ **El renderizador funciona a 584 × 584 aquí:** 0,87 y 0,78 s/imagen en **dos mediciones
  independientes** → las 1000 imágenes son **13-15 min** *(estimado desde esos ritmos)*. **No hace
  falta Google Chrome**: basta el Chromium de Playwright, que ya está en caché.
- ✅ **Entrenar es barato: 35,1 ms por paso** (100 pasos cronometrados) → **un run de 200 épocas
  ≈ 0,6 min**. El banco completo es del orden de **30-45 min** *(estimado)*.
  **⇒ se corre aquí; no hace falta alquilar nada.**
- ⚠⚠ **`placement.area` acota sólo la esquina superior-izquierda, no la caja entera** — lo dice
  el código del generador y lo confirman 6 renders (los 6 pegados al borde del área). **Y una caja
  puede salirse del lienzo**: uno dio `y1 = 641,62` sobre 584, o sea un párrafo cortado. La v1.2
  recogió las dos cosas en §3.3 como «dos daños distintos», y llama al segundo **peor**: *«la
  etiqueta es directamente falsa»*.
  ⚠⚠ **Pero NO se arregla descartando**, y eso corrige lo que yo había escrito: el §3.4 nuevo
  **prohíbe** sobre-generar y rechazar, porque rechazar elimina selectivamente las cajas grandes y
  periféricas y sesga hacia párrafos **pequeños y centrados** — justo lo que sube el control de
  caja media y comprime el banco. Se muestrea **el tamaño primero** y la esquina después.
- ✅ **La cadena de rejillas y el span de centros ya se comprueban**, que la v1.2 vuelve
  aserciones obligatorias (§7.1, §7.5, §11.8-9): cadena `128/64/32/16` y centros **0…120**, que
  cubren las etiquetas `[8, 119]` *(la v1.2 escribe `[8, 120]`; su propia aritmética
  —`512/4 − 9`— da 119. Off-by-one anotado; no cambia ninguna conclusión)*. Con `valid` serían **10…114** y los bordes extremos quedarían
  **inalcanzables por construcción** — el motivo real del `same`, mejor que las tres
  corroboraciones textuales con las que se había deducido.

## Cómo está pensado (las cuatro decisiones que más sorprenden al leerlo)

Todas salen del **criterio único** del §1.2: *maximizar la sensibilidad de la medición al
kernel*. Cualquier cosa que dé holgura a la CNN **reduce la evidencia** que el banco puede
producir.

1. **La cabeza tiene 68 parámetros a propósito.** Una cabeza grande **compensa un kernel malo** y
   comprime las diferencias hasta el ruido entre semillas. Esta red **no se mejora**.
2. **100 de entrenamiento y 800 de evaluación** — la proporción invertida es deliberada: reduce la
   varianza de la métrica titular y endurece el régimen, que es donde el kernel se ve.
3. **Sin aumento de datos, sin parada temprana, sin schedule.** El aumento competiría con el
   kernel por el mismo efecto; la parada temprana **evitaría el sobreajuste**, que es justo lo que
   hay que observar, porque la brecha `train − eval` **es** la evidencia del criterio 2.
4. **Todas las condiciones ven exactamente los mismos píxeles**: el descarte es **siempre 9 px por
   lado** sea cual sea `k` (un kernel pequeño convoluciona menos y recorta más). Así la diferencia
   entre condiciones no mezcla calidad del kernel con cantidad de imagen vista.

## Los dos criterios, y por qué se reportan separados

Congelados **antes de mirar** en [`instrucciones/02-criterio.md`](instrucciones/02-criterio.md):

- **Utilidad (§2.1):** el IoU sobre `eval` supera al del **aleatorio de igual norma y mismo `k`**
  por más que **la suma** de las desviaciones entre semillas de las dos condiciones.
- **Generalización (§2.2):** además, su **brecha `train − eval`** es menor que la de la
  **identidad**, bajo el mismo margen.

Un kernel puede subir el IoU por **facilitación** (hace el problema más fácil: suben `train` y
`eval` juntos) o por **transferencia** (la representación generaliza mejor: la brecha baja). **Sólo
la segunda es el objetivo**, y un kernel que cumple 2.1 y falla 2.2 **es un resultado válido que se
registra como tal**.

⚠ **Si un kernel supera a la identidad pero no al aleatorio, lo que se ha demostrado es que
«filtrar funciona», no que «el kernel funciona».**

## Qué hay en esta carpeta

```
ESPECIFICACION.md     la especificación v1.2 del dueño, VERBATIM. Manda sobre todo lo demás
experimento.json      identidad legible por máquina: id, estado, gasta, lo que bloquea
REGLAS.md             las reglas de ESTE experimento: entradas, salidas, procesos, scripts
instrucciones/
  01-encargo.md       qué se pidió, qué se hizo, qué se midió y QUÉ FALTA
  02-criterio.md      los criterios, congelados antes de mirar (R13)
nn/
  modelo.py           la CNN del §7, AUTÓNOMA (sólo torch) + sus invariantes comprobables
  datos.py            genera · publica · comprueba · muestras
  pipeline.py         el §6: kernel → recorte → estandarización
  evaluar.py          el §9: IoU, MAE por borde, brecha y los dos criterios
  kernels.py          los controles del §10
  calibrar.py         los 10 pasos del §11, reanudable
  entrenar_local.py   entrena una condición (el nombre es el contrato con el freno)
  lanzar.sh           lo que tarda, como unidad de systemd
kernels/              los CONTROLES del §10 (aleatorio ×10, gauss, sobel). Ningún kernel
                      EVALUADO todavía, y es correcto: §1 y §15
resultados/           las 41 corridas + CALIBRACION.md (se regenera, no se transcribe)
```

⚠ **Que no haya ningún kernel *evaluado* no es un olvido.** El banco se **calibra y se valida
entero** con sus controles (caja media · identidad · aleatorio · gauss · sobel) **sin un solo
kernel de verdad**, que es lo que el §11 manda hacer primero — y ya está hecho.

## Antes de correr nada: la calibración (§11)

**No es un experimento y sus cifras no se reportan como hallazgos.** Fija el instrumento:

Son **10 pasos** en la v1.2 (antes 7). Los que deciden si el banco sirve:

1. **caja media primero, con umbral: IoU ≤ 0,40.** Si queda por encima, se amplía el rango de
   **ancho y alto** de caja (§3.5) —no el de posición— y **no se continúa**.
2. **identidad con las 10 semillas**, y que **deje margen bajo el techo**: por encima de ~0,95
   tampoco hay sitio para demostrar nada (§10.1.1).
3. **que el rango entre caja media e identidad sea amplio.** Si es estrecho, **ninguna cantidad de
   semillas produce evidencia**.
4. **aleatorio con las 10 semillas**, cuya desviación es **el denominador de los dos criterios**.
5. **verificar la resolución:** si el MAE se estanca cerca de **8 px** —el tamaño de una celda—,
   quitar el stride de la tercera conv **antes de tocar nada más** y reiniciar la calibración.
6-10. las **aserciones**: caja dentro del marco **y del lienzo**, la transformación de
   coordenadas, la **cadena 128/64/32/16**, el **span de centros** contra el rango real del
   dataset, y el **balance marginal** de factores entre particiones. ✅ Las dos de la arquitectura
   ya las corre `python nn/modelo.py`.

Concluida la calibración, los parámetros quedan **congelados** (§12).
