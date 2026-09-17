# Encargo de `gauss-p`

## Lo que pidió el dueño, el 2026-09-17

> «En el repo experimentos tenemos definido un experimento banko kernels. Necesito crear un
> experimento que tome el dataset empleado en ese experimento y le aplique un filtro
> gaussiano. Asumo que dicho filtro tiene parámetros. Crea una mini app en este server donde
> podamos aplicar varios parametros a unas 10 muestras. Esto me permitirá elegir los
> parámetros más apropiados, y luego la idea es generar ese filtro como un kernel que
> podamos aplicar en el experimento banco de kernels para evaluarlo. Si gauss no tiene
> parámetros, nada de esto tiene sentido (y en ese caso no hagas nada de esto).»

## La condición de auto-cancelación: NO se dispara

**`gauss` sí tiene parámetros.** `banco-k/nn/kernels.py` define `gauss(k, sigma)`, con
`sigma` por defecto `k/6`. Y el parámetro **sobrevive a la normalización L2 del §5.4**, que
es lo que de verdad había que comprobar: el banco normaliza el kernel al recibirlo, así que
un factor de escala sería un no-op y no contaría como parámetro.

*Medido el 2026-09-17, distancia L2 del kernel normalizado a la delta de Dirac:*

| σ | 0,2 | 0,6 | 1,0 | 1,5 | 3,0 | 10 | 50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dist. a la identidad | 0,000 | 0,470 | 0,934 | 1,117 | 1,269 | 1,328 | 1,333 |

O sea: **σ → 0 es exactamente la identidad** y **σ → ∞ satura en una caja plana**. Entre
medias hay eje de verdad.

## Cómo se leyó la petición, y por qué así

*«Crear un experimento que tome el dataset y le aplique un filtro gaussiano»* admite dos
lecturas, y una de ellas es lo que `banco-k` **ya hace** (su condición `gauss` es
literalmente eso). La frase siguiente del encargo la desambigua: *«elegir los parámetros más
apropiados, y luego generar ese filtro como un kernel que podamos aplicar en el experimento
banco de kernels»*. O sea que lo que se pide es **elegir `sigma` y meter el kernel por la
puerta del banco**, no publicar un dataset pre-filtrado.

Se descartó lo segundo además por dos motivos: un dataset pre-filtrado **no es evaluable como
kernel** en el banco, y el repo tiene una regla de que un dataset publicado no se reescribe.

## Las tres cosas que el encargo no decía y hubo que decidir

1. **De qué muestras.** `banco-k` reserva `LiberationMono` y el interlineado `[1.45, 1.60]`
   para uso exclusivo suyo (§3.7), y **elegir un parámetro mirando muestras es un
   procedimiento que produce kernels**. Se usan sólo las libres: 55 de las 100 de `train`.
   *Y de `train`, no de `eval`*, que es la partición con la que el banco declara.
2. **Qué se enseña.** La imagen **tal como la red la recibirá** —kernel normalizado,
   convolución válida, recorte a 128, estandarizada— y no una versión legible. Enseñar la
   imagen filtrada «cruda» mentiría en la dirección más fácil de creer: la gaussiana apaga el
   contraste y el §6.5 lo devuelve entero.
3. **Quién escribe el `.npy`.** El **banco**, con su propio código, no esta app. Un kernel que
   no se regenera desde código commiteado es un dato huérfano — la lección que este repo ya
   pagó con el laplaciano.

## Lo que se dijo al dueño como reserva

Que el paso 3 del encargo —evaluar el gauss re-afinado— **cae sobre un banco donde hoy no
declara ningún kernel**, y que eso está medido y razonado, no supuesto. El detalle y el
criterio están en `02-criterio.md`, escrito antes de abrir la app.
