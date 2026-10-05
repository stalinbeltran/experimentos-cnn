# Criterio — escrito el 2026-10-05, ANTES de entrenar ningún detector

Nada de lo de abajo se ha decidido mirando un detector entrenado de este experimento. Lo único que se ha visto son
detectores de **1 época** (prueba de que la cadena corre: posicional 0,900 sobre 180/1617), y esos pesos se borraron.
Si al correr hace falta cambiar algo, va como **enmienda fechada** al final.

Los números de referencia son de `feat-ind` (su `README.md`), con los **mismos dígitos, el mismo reparto y el mismo
compositor**. Allí entraban reducidos a 8×8; aquí, en 32×32.

## A. Por detector (13), val sintético — igual que `feat-ind` §A

| veredicto | condición |
|---|---|
| **aprendió** | F1 ≥ 0,90 **y** posición ≤ 1 celda ≥ 0,90 |
| **a medias** | F1 ≥ 0,75, o F1 ≥ 0,90 con posición < 0,90 |
| **no aprendió** | lo demás |

Umbral por detector elegido sobre train (enmienda 2 de `feat-ind`, aquí desde el principio). Lecturas obligatorias:
recall por tramo de radio, FP por familia, recall por grosor.

**H1.** A 8×8 (corrida 2): arcos E/W/S y las 4 esquinas, **«no aprendió»**; arco-N y las 4 rectas, «a medias»; lazo,
«aprendió». **H1 se da por confirmada si al menos 6 de esos 8 «no aprendió» suben de categoría.** Y **se da por
refutada** si ninguna esquina pasa de 0,75 de F1: entonces lo que las confundía no era la resolución.

## B. Compositores sobre dígitos (C4) — 180 train / 1617 val, 3 semillas

**H2.** Posicional de `feat-ind`: **0,959 ± 0,001**. Gana la resolución si aquí sale **≥ 0,969**; es empate si
cae en (0,949, 0,969); pierde si ≤ 0,949. Se reportan presencia, Δ posicional − presencia y los pares 6↔9, 2↔5.

**H4 (el par del grosor).** A 8×8 el posicional confundía **1↔8 22 veces** (atribución del error, corrida 2). Se
confirma si aquí son **≤ 5**, **sin que 4→1 pase de 5** (el precio que pagaron los trazos gruesos en la corrida 4).

⚠ **Referencia de suelo, medida al probar la cadena**: con detectores de **1 época** el posicional ya da 0,900. O sea
que los 0,06 que separan eso de 0,959 son lo que aportó entrenar en `feat-ind`, y la comparación H2 se lee contra
ese rango, no contra el azar.

## C. Curva por N (C5) — test fijo de 717, el de la corrida 11

Referencia a 8×8, con el **mismo test**: fino+grueso **0,974 con N = 180 y 0,992 con N = 900**. ⚠ Aquí sólo hay el
banco `fino` (13 detectores, no 26), así que la comparación es **desfavorable** para 32×32 a propósito: no se
reprodujo el grueso.

**H3.** Si con N = 900 sale **≥ 0,992**, el banco fino a 32×32 iguala o supera a fino+grueso a 8×8: la resolución
sustituye al banco grueso. Si sale < 0,985, no.

**Lo nuevo (sin referencia a 8×8):** el mismo compositor entrenado con los **3823 dígitos de otros 30 escritores**
(solos, y sumados a los 1080), medido sobre el mismo test. Se reporta; se espera ≥ 0,99 y no se declara nada.

## D. Generalización (C6) — test de 717

Referencia `fino` a 8×8 (corrida 12): test 0,964 · G1 0,036 · G2 0,844 · G3 medio 0,762 · desplazado 0,577.
Se reportan A (posicional) y B (máx 3×3) con los detectores, y los píxeles crudos de 32×32.

⚠ **Las 5 transformaciones NO son las de `feat-ind`**: aquí se aplican al bitmap de 32×32 (desplazar 4 px = 1 celda,
ruido σ 0,15, dilatar 2 px, erosionar 1 px, ocluir 12×12 px). Los G3 se comparan **en tendencia, no en valor**.

**H5.** B sube el desplazado respecto de A en ≥ 0,10, como a 8×8 (+0,25 con fino+grueso).

## E. Lo que espero hoy (antes de mirar)

- H1 sí en los arcos y a medias en las esquinas: a 32×32 un arco de radio 4 px sigue siendo pequeño.
- H2 empate o ganancia pequeña (+0,005 a +0,015): a 8×8 el compositor ya compensaba mucho.
- H4 sí: el grosor era el problema y a 32×32 se ve.
- H3 no: 13 detectores no llegan a lo que daban 26.
- Lo que más probablemente salga mal: **la transferencia sintético → manuscrito empeora**, porque a 32×32 el
  trazo manuscrito binarizado de NIST es más irregular que el sintético, y a 8×8 el conteo lo suavizaba.

## Curvas de CNN (2026-10-05) — escrito ANTES de correr los 120 entrenamientos

Pedido del dueño: comparar las curvas con modelos tradicionales. Dos CNN entrenadas de punta a punta (`nn/cnn.py`): la
**CNN de 3 capas** del repo (la de `ruido-nist`, 1.338 parámetros, 8×8) y una **LeNet-5** sobre 32×32 (61.706
parámetros). Mismas 60 particiones que la ganancia (comprobado por huella), 3996 pasos de 20, sin selección ni aumento.

⚠ **Lo que ya se vio antes de escribir esto**, al medir el tiempo de un entrenamiento (semilla 1, 4 puntos): LeNet-5 98,8 %
con N = 2000 y 60,4 % con N = 10; CNN de 3 capas 96,8 % y 43,5 %. Las predicciones de abajo los tienen en cuenta.

Se compara contra las curvas de la ganancia (detectores 8×8: 64,6 % con N = 10, 95,8 % con 160–200, 98,1 % con 2000;
N(95 %) = 133), con las mismas funciones (`nn/muestras_necesarias.py`: media por N, isotónica, sin extrapolar).

1. **Cruce detectores 8×8 – LeNet-5.** Por los 4 puntos vistos, LeNet-5 gana con muchos datos y pierde con muy pocos: hay
   un cruce. **Predicción: entre N = 200 y N = 1000.** Se reporta el N del cruce. Si LeNet-5 queda por encima de los
   detectores para todo N ≥ 40, la ventaja de los detectores es sólo del régimen de muy pocos datos, y se dice así.
2. **N(95 %) de LeNet-5**: predicción **250–500** (los detectores, 133: entre 2× y 4× menos muestras).
3. **CNN de 3 capas contra la logística sobre píxeles**: en N = 180 / 1617 quedó por debajo (86,9 % contra 90,4–90,8 %).
   Predicción: **por debajo hasta N ≈ 500 y por encima desde ~1000** (en N = 2000 ya se vio 96,8 %, por encima de los
   píxeles 8×8, 95,4 %).
4. **Coherencia con lo medido en N = 180 / 1617** (`ruido-comb`): la curva de la CNN de 3 capas en N = 160–200 tiene que
   caer cerca de 86,9 ± 2,4 %. Si se aleja más de 5 puntos, el protocolo distinto (datasets mezclados de 43 escritores)
   pesa más de lo que se suponía en `comparar_cnn.py`, y se dice.
