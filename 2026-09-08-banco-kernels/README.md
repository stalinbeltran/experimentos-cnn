# Banco de evaluación de kernels — `banco-k`

**Un instrumento de medida, no un experimento particular.** Mide cuánto aporta un kernel de
convolución, aplicado como **preprocesamiento de las entradas**, a la **generalización** de una
CNN de referencia de 5.812 parámetros entrenada con **100 muestras**.

El banco es **agnóstico al origen del kernel**: el kernel entra como **dato** (`.npy`), y los
procedimientos que producen kernels quedan **fuera de alcance**.

> 📄 **La fuente de verdad es [`ESPECIFICACION.md`](ESPECIFICACION.md)** — la especificación v1.0
> del dueño (2026-09-07), copiada verbatim. Este README no la resume para reemplazarla; donde
> difieran, gana ella.

## ⚠ Estado: montado, NADA corrido

**2026-09-08.** Existe la carpeta, sus dos obligaciones, la arquitectura comprobable y el
criterio congelado. **No hay dataset publicado, ni pesos, ni una sola cifra medida de este
banco.** `experimento.json` declara `estado: "abierto"` y `dataset: null`, que es el estado real.

```bash
python nn/entrenar_local.py              # se NIEGA y lista qué falta (código 2)
python nn/entrenar_local.py --comprobar  # comprueba la arquitectura del §7 (código 0)
python nn/modelo.py                      # lo mismo, con las dos lecturas del padding
```

### Lo que falta, y quién lo tiene que decidir

Tres cosas necesitan al **dueño** antes de poder correr, porque las tres son **irreversibles**:
una toca un **invariante** (§12) y dos congelan un **dataset publicado**, que no se reescribe
nunca. Están enteras, con su evidencia, en
[`instrucciones/01-encargo.md`](instrucciones/01-encargo.md):

| | qué | por qué bloquea |
|---|---|---|
| **P1** | el §7.1 dice `valid` pero sus dimensiones (64/32/16) **sólo salen con `same`** | la arquitectura es invariante: elegir mal no da un error, da una serie que hay que tirar |
| **P2** | `dtype` del dataset (exacto en `uint16` vs `uint8` redondeado) | un dataset publicado **no se reescribe nunca** |
| **P3** | qué varía el generador y qué estratifica el reparto 100/100/800 | decide si el §10.1 (caja media) deja margen para medir algo |

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
  puede salirse del lienzo**: uno dio `y1 = 641,62` sobre 584, o sea un párrafo cortado. Así que la
  restricción **obligatoria** del §3.3 va como **aserción + descarte** —de las dos cosas— y no
  confiando en la receta.

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
ESPECIFICACION.md     la especificación v1.0 del dueño, VERBATIM. Manda sobre todo lo demás
experimento.json      identidad legible por máquina: id, estado, gasta, lo que bloquea
REGLAS.md             las reglas de ESTE experimento: entradas, salidas, procesos, scripts
instrucciones/
  01-encargo.md       qué se pidió, qué se hizo, qué se midió y QUÉ FALTA
  02-criterio.md      los criterios, congelados antes de mirar (R13)
nn/
  modelo.py           la CNN del §7, AUTÓNOMA (sólo torch) + sus invariantes comprobables
  entrenar_local.py   la entrada declarada. Hoy se niega y dice qué falta
  receta.json         la receta de render 584×584 con el área [68, 512] del §3.3
kernels/              los kernels a evaluar (.npy). Vacío: no hay ninguno todavía, y es correcto
resultados/           métricas, resúmenes y criterios por condición. Vacío
```

⚠ **`kernels/` vacío no es un olvido.** El banco se **calibra y se valida entero** con sus
controles (caja media · identidad · aleatorio · gauss · sobel) **sin un solo kernel de verdad**,
que es lo que el §11 manda hacer primero.

## Antes de correr nada: la calibración (§11)

**No es un experimento y sus cifras no se reportan como hallazgos.** Fija el instrumento:

1. **caja media primero.** Si ese predictor constante saca un IoU alto, **se corrige el generador
   y no se continúa**.
2. **identidad y aleatorio con 10 semillas**, para medir la desviación real del IoU.
3. **fijar el número definitivo de semillas** con ese dato (§8.5 avisa de que puede estar en
   0,03–0,05, lo que vuelve exigente el margen).
4. **verificar la resolución:** si el MAE se estanca cerca de **8 px** —el tamaño de una celda—,
   quitar el stride de la tercera conv **antes de tocar nada más** y reiniciar la calibración.

Concluida la calibración, los parámetros quedan **congelados** (§12).
