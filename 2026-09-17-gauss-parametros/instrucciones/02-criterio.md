# Criterio de `gauss-p` — escrito ANTES de mirar

**Fecha: 2026-09-17, antes de abrir la app por primera vez.** Lo de aquí se escribe ahora
justamente para que «no había nada que rascar» pueda ser **un resultado** y no una
decepción. Si se escribiera después de mover el mando, sería una justificación.

## Lo que este experimento NO decide

No declara nada. El criterio que declara es el **§2.1/§2.2 de `banco-k`**, que está escrito
desde antes y es invariante. Aquí sólo se decide **si merece la pena pagar una evaluación**,
que cuesta ~9 min de CPU de esta máquina y 0 $.

## El punto de partida, medido y no opinado

*Leído del disco el 2026-09-17 de `banco-k/resultados/*/resumen.json`, 10 semillas cada uno,
IoU sobre `eval` en la época 200:*

| condición | IoU eval | sd | efecto sobre su aleatorio | margen §2.1 | ¿declara? |
|---|---:|---:|---:|---:|:--:|
| identidad | 0,7981 | 0,0057 | −0,0103 | 0,0135 | no |
| laplaciano | 0,7963 | 0,0051 | −0,0120 | 0,0128 | no |
| **gauss** (k=9, σ=1,5) | **0,8130** | 0,0086 | **+0,0047** | 0,0163 | **no** |
| sobel | 0,8164 | 0,0064 | +0,0081 | 0,0142 | no |
| esqk-k09 | 0,8181 | 0,0079 | +0,0098 | 0,0156 | no |
| esqk-k11 | 0,8256 | 0,0087 | +0,0162 | 0,0208 | no |

**Ninguno de los siete declara.** El más cercano es `esqk-k11`, que necesitaría **1,28×**
su efecto; el gauss vigente necesitaría **3,47×**.

⚠ **Y eso no es un problema de precisión, así que no se arregla midiendo mejor.** El margen
del §2.1 es la **desviación estándar**, no el error estándar, y una desviación **converge**
al crecer la muestra en vez de encoger: con 100 semillas el margen seguiría valiendo ~0,016.
Está razonado en el commit `64b8fbb` de este repo, cuya conclusión es literal: *«en ESTE
banco, nada lo consigue»*, porque la caja del párrafo sobrevive a cualquier filtro.

## Lo que se espera de mover `sigma`, dicho antes

**Lo más probable es que no aparezca ningún `sigma` que merezca una evaluación**, y hay una
razón concreta además de la de arriba:

1. **El eje es más corto de lo que parece.** *Medido el 2026-09-17 (`nn/forma.py`):* el
   defecto `k/6` del banco resulta ser **la gaussiana más ancha que cabe en `k` sin
   truncarse**. Por arriba el tope es `k` ≤ 19 (§5.2, congelado), o sea σ ≈ 3,2; por abajo,
   σ → 0 **es la identidad**, que ya está medida y es **peor** que el gauss vigente
   (0,7981 contra 0,8130). O sea que el vigente ya está cerca de un extremo útil del eje.
2. **Todo el rango observado entre «no filtrar» y «el mejor filtro medido» es 0,0275**
   (0,7981 → 0,8256). Un `sigma` re-afinado a k=9 tendría que llegar a **≈0,8246** para
   declarar, o sea **+0,0082 por encima de `sobel`**, que es el mejor k=9 jamás medido aquí.

## El criterio, entonces

**Se pide una evaluación al banco sólo si se cumple lo primero, y se recomienda además lo
segundo:**

1. **El `sigma` elegido tiene que ser visiblemente distinto del vigente en algo que importe
   al problema**, y «lo que importa» está definido: el banco mide **los cuatro bordes de la
   caja**. O sea que la pregunta al mirar no es «¿se ve bien?» sino **«¿sigue habiendo borde
   donde la caja roja dice que está, y no se ha salido la tinta fuera?»**. La app dibuja esa
   caja precisamente para que esto sea mirable y no opinable.
2. **Y tiene que cumplir el contrato sin truncarse**: la app avisa cuando la ventana `k`
   corta más del 1 %. Un `sigma` que sólo cabe recortado **no es la gaussiana que se eligió**;
   es otra cosa que se parece a una caja plana, y evaluarla contesta otra pregunta.

**Se cierra sin evaluar nada —y eso es un resultado— si:**

- todos los `sigma` mirables o bien se parecen al vigente, o bien sacan la tinta fuera de la
  caja; **o**
- el único rango que se ve distinto es el que no cabe en `k` ≤ 19.

## Y si se evalúa de todas formas

**Es legítimo, y el dueño puede pedirlo sabiendo lo de arriba.** En ese caso queda escrito
desde ahora que **el desenlace más probable es `NO DECLARA`**, y que eso **no invalidará el
experimento**: un punto más del eje gaussiano, medido con las 10 semillas del banco y
comparable con todo lo anterior, es un dato que hoy no existe. Lo que no vale es leer un
`NO DECLARA` previsto como si fuera una sorpresa.

⚠ **Y lo que el informe que lo cite tiene que decir:** que ese `sigma` **no se eligió a
ciegas**, sino mirando muestras de `train` de este mismo dataset. Cumple el §3.7 —las
muestras miradas excluyen la familia y el interlineado reservados— pero no es un kernel
ciego. El aviso viaja dentro del `.json` del kernel, para que llegue hasta el veredicto sin
depender de que alguien se acuerde.
