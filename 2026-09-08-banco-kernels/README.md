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

## ⚠ Estado: montado sobre la v1.2, NADA corrido

**2026-09-08.** Existe la carpeta, sus dos obligaciones, la arquitectura comprobable y el
criterio congelado. **No hay dataset publicado, ni pesos, ni una sola cifra medida de este
banco.** `experimento.json` declara `estado: "abierto"` y `dataset: null`, que es el estado real.

✅ **Las tres decisiones que bloqueaban están cerradas por la v1.2** — padding del tronco `same`
(§7.1), almacenamiento `uint16` con la suma del bloque (§3.8), y qué varía el generador con su
estratificación (§3.5-§3.6). `bloqueado_por` está **vacío**.

```bash
python nn/entrenar_local.py              # se NIEGA y lista qué falta (código 2)
python nn/entrenar_local.py --comprobar  # comprueba la arquitectura del §7 (código 0)
python nn/modelo.py                      # lo mismo, con las dos lecturas del padding
```

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
