# 02 — Criterio, escrito ANTES de mirar

**Congelado el 2026-09-08**, con **cero runs corridos**, **cero épocas** y **ningún kernel en
`kernels/`**. Esto es comprobable: no existe `resultados/`, ni `nn/pesos/`, ni un dataset
publicado del que sacar una cifra. **Actualizado el mismo día a la v1.2** de la especificación,
que fijó las semillas en 10 y añadió el techo del §10.1.1 — **sin aflojar ningún criterio**.

**No es un criterio nuevo.** Lo fija la [especificación §2](../ESPECIFICACION.md), que ya lo
declara *«antes de ejecutar cualquier corrida»*. Este documento lo **operacionaliza** —lo deja en
la forma exacta en que se va a computar— y **no lo afloja en ningún punto**. Donde éste y la
especificación difieran, gana la especificación.

⚠ **Por qué se escribe antes** (R13): un criterio escrito después de ver los números no se
distingue de una racionalización, y entonces *«no hubo señal»* deja de poder ser un resultado. En
este banco eso importa el doble, porque el desenlace más probable de una condición cualquiera es
**no superar al control**, y eso tiene que poder anotarse como hallazgo y no como decepción.

## Qué se compara, y contra qué

La única variable es **el kernel aplicado a las entradas**. Semillas, arquitectura,
hiperparámetros, particiones y **orden de los lotes** son idénticos en todas las condiciones
(§8.2). Todo se lee en la **época 200** y como **media ± desviación estándar entre semillas**
(§9.3); una cifra de una sola semilla **no es un resultado**.

**Son 10 semillas, fijas** (§8.5 de la v1.2), no «≥ 5 y lo fija la calibración». ⚠ **Y eso
endurece los dos criterios a propósito**: la desviación entre semillas es **el denominador** de
los dos, así que cada semilla adicional **estrecha** el margen que un kernel tiene que superar.
Con ~0,6 min por corrida, 10 semillas × 10 condiciones son **~1 h** de cómputo local.

## Criterio 1 — UTILIDAD (§2.1)

> Un kernel es **útil** si su **IoU medio sobre `eval` en la época 200** supera al del **kernel
> aleatorio de igual norma y mismo `k`** por un margen **mayor que la suma de las desviaciones
> estándar de ambas condiciones entre semillas**.

```
IoU_eval(kernel) − IoU_eval(aleatorio)  >  σ(kernel) + σ(aleatorio)
```

- El comparador es **el aleatorio**, no la identidad. El aleatorio lleva **igual norma L2 y el
  mismo `k`**, y **varias semillas de aleatoriedad**, porque su desempeño también varía (§10.2).
- ⚠ **El margen es la SUMA de las desviaciones, no la mayor ni la del kernel.** Es el criterio
  más exigente de los tres posibles y es el que está escrito.

## Criterio 2 — GENERALIZACIÓN (§2.2)

> Un kernel muestra **efecto de generalización** si, **además de cumplir el criterio 1**, su
> brecha `IoU_train − IoU_eval` en la época 200 es **menor que la de la condición identidad**,
> bajo el mismo margen.

```
brecha(kernel)  <  brecha(identidad) − [ σ_brecha(kernel) + σ_brecha(identidad) ]
```

- **El criterio 2 es condicional al 1.** Un kernel que reduce la brecha pero no supera al
  aleatorio **no cumple el 2**, porque el 2 dice «además de cumplir 2.1».
- El comparador aquí **sí** es la identidad: la pregunta es si la representación generaliza
  mejor que **no filtrar nada**.

## Los dos se reportan SEPARADOS, y las cuatro combinaciones son resultados

Ésta es la distinción central del banco (§2.3) y **la razón por la que registrar el IoU de
`train` no es opcional**: sin él, los dos mecanismos son indistinguibles.

| | ¿cumple 1? | ¿cumple 2? | qué significa | ¿es un resultado? |
|---|---|---|---|---|
| **Transferencia** | sí | sí | el kernel produce una representación **más generalizable**: la brecha se reduce. **Es el objetivo del banco** | sí |
| **Facilitación** | sí | no | el kernel **hace el problema más fácil**: `train` y `eval` suben juntos y la brecha se mantiene | **sí, y se registra como tal** |
| **Sin efecto** | no | — | no se distingue del aleatorio | sí |
| **Perjudica** | no | — | queda por debajo | sí |

⚠ **«Facilitación» no es un fallo del experimento ni del kernel.** Es un desenlace previsto que
la especificación nombra y obliga a registrar. Lo que **no** vale es informar de un kernel útil
sin decir por cuál de los dos mecanismos.

## La cadena que hay que poder sostener (§10.3)

```
kernel evaluado  >  aleatorio  >  identidad  >  caja media
```

**Cada `>` tiene que superar la desviación entre semillas, o no es un `>`.**

⚠ **Y la lectura que no se puede saltar:** si el kernel supera a la **identidad** pero **no al
aleatorio**, la conclusión correcta **no** es *«el kernel funciona»* sino **«filtrar
funciona»**. Son afirmaciones distintas y sólo el control aleatorio las separa. Omitirlo es una
trampa con su propia fila en el §14.

## Lo que invalida la corrida, antes de leer ningún kernel

Se comprueba **en este orden** y **cada uno detiene el banco**:

1. **La caja media saca un IoU alto** — concretamente **por encima de 0,40** (§10.1; umbral
   *propuesto, no derivado*, que se valida en calibración). Ese predictor constante ignora la
   imagen: si acierta mucho, el generador coloca los párrafos con poca variabilidad y todas las
   condiciones quedan comprimidas. → **se amplía el rango de ANCHO y ALTO de caja** (§3.5), no el
   de posición, **y NO se continúa.**
2. **La identidad se pega al TECHO** (§10.1.1, nuevo en la v1.2): por encima de ~0,95 **no queda
   margen** para que ningún kernel demuestre mejora, y las condiciones se comprimen igual que con
   un piso alto. **El rango útil del banco es la distancia entre caja media e identidad**: si es
   estrecha, **ninguna cantidad de semillas produce evidencia**.
3. **El MAE se estanca cerca de 8 px** (§7.5). Es el tamaño de una celda de la rejilla: significa
   que el soft-argmax no está interpolando. → **quitar el stride de la tercera conv** (rejilla
   32 × 32) **antes de tocar cualquier otra cosa**, y **reiniciar la calibración**.
4. **Alguna caja queda fuera del marco final o fuera del lienzo de 584** (§3.3, «dos daños
   distintos»; el segundo es **peor**, porque la etiqueta es **directamente falsa**).
   ⚠ **Y no se arregla descartando** (§3.4): se muestrea el **tamaño primero** y la esquina
   después. Rechazar sesga hacia párrafos pequeños y centrados, que es lo que sube la caja media
   del punto 1. Es un error irreducible que
   desplaza el IoU medio. → **aserción en la generación**; no se entrena con muestras así.
5. **La transformación de coordenadas no cuadra** en una muestra conocida (§6.4). Omitir el
   `−9` sesga **todas** las condiciones por igual, o sea que no se ve como error: se ve como que
   todos los kernels son uniformemente malos. → **aserción**.

## Qué NO se decide con estos criterios

- **No se declara un ganador entre kernels.** Se reporta, por kernel, si cumple 1 y si cumple 2.
- **No se compara eficiencia por parámetro** entre kernels de distinto `k` (§15).
- **Nada de aquí mueve la red de producción de `foveal-vision`.** Este banco mide kernels; que un
  kernel sea útil aquí no es una instrucción para aplicarlo en ningún sitio.
- **La calibración (§11) no produce hallazgos.** Es puesta a punto del instrumento y sus cifras
  no se reportan como resultado — sólo fijan el número de semillas.

## Lo que se puede predecir hoy, para que conste antes de medir

- **La red va a memorizar las 100 muestras de `train`**, y eso es lo buscado, no un problema: la
  brecha `train − eval` en régimen de sobreajuste **es** la evidencia del criterio 2 (§8.4). Por
  eso no hay parada temprana.
- **La desviación entre semillas puede caer en 0,03–0,05**, y con la suma de dos desviaciones
  como margen, **el criterio 1 es difícil de cumplir**. Si con las 10 semillas ninguna condición
  declara, la respuesta es **un margen honesto y decirlo**, nunca un margen más flojo elegido
  después de ver los números.
- **Y el banco puede resultar no medir nada**, que también es un desenlace previsto: si la
  distancia entre caja media e identidad sale estrecha (§10.1.1), lo que hay que arreglar es el
  **generador** (§3.5), y eso **no es un resultado del banco** sino de su calibración.
