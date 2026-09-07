# El criterio, congelado ANTES de la primera época

Escrito el **2026-09-07 con cero épocas entrenadas en cualquier brazo**. Los suelos de abajo
están medidos sobre la partición de validación con las redes **sin entrenar**, que es lo único
que se puede medir antes de mirar.

```bash
python nn/entrenar_local.py --suelos      # reproduce la tabla de abajo
```

## La métrica principal, y su suelo

**Tasa de acierto: qué fracción de las esquinas se predice a ≤ 2 px de su sitio**, medida
**por esquina** y sólo sobre las ventanas donde esa esquina existe de verdad (la etiqueta, no la
predicción). Val trae **389 ventanas: 78 con `tl` y 78 con `br`**.

**El titular es la PEOR de las dos esquinas, nunca el promedio.** Un promedio escondería
exactamente el desenlace más probable —que una estructura resuelva `tl` y no `br`—, que es la
pregunta del experimento.

| | acierto ≤2 px `tl` | acierto ≤2 px `br` | ≤1 px | error medio |
|---|--:|--:|--:|--:|
| predictor **constante** en el centro del mapa | 5,1 % | 6,4 % | 1,3 % · 3,8 % | 5,98 · 6,19 px |
| **los cinco brazos sin entrenar** | 3,8 – **5,1 %** | **6,4 %** | 0 – 1,3 % · 2,6 – 3,8 % | 5,96 – 6,06 · 6,13 – 6,22 px |

⚠ **El suelo no depende del brazo, y eso es lo que permite un único umbral.** Se midió sobre las
25 redes de las cuatro estructuras antes de reducir el experimento a una, y las 25 caen en el
mismo sitio: `ant` con 17 parámetros y `sig-k13` con 174 dan el mismo 5,1 % / 6,4 %. No es que se
parezcan — es que todas arrancan en el mismo modo degenerado.

⚠ Las redes sin entrenar dan **exactamente** el predictor constante, igual que en `esq-k`: si el
mapa sale plano, el softmax es uniforme y la esperanza cae en el centro del mapa, que es justo
donde se concentran las etiquetas. **La solución vaga y la solución mediocre son la misma**, así
que hay un óptimo local cómodo exactamente donde la red arranca.

> **Un brazo ha aprendido algo** si su acierto ≤2 px de validación pasa del **12 % EN LAS DOS
> esquinas**. Sale de suelo + 2·SE con SE = 2,5 % (`tl`) y 2,8 % (`br`) sobre 78 positivas:
> 10,1 % y 12,0 %. Se toma el mayor de los dos para las dos, que es la dirección estricta.

**Métrica secundaria**, siempre junto a su suelo y nunca sola: el **error medio**, que además
debe bajar de **5,45 px** (= 5,96 − 2·SE, con SE = 0,26 px; se usa el suelo más bajo de los cinco
brazos, que es la dirección estricta, y el mismo número para las dos esquinas aunque el de `br`
daría 5,61). Y `existe`, cuyo suelo es
**f1 = 0,334** por esquina — lo que da el «siempre sí» con 78 positivas de 389, y que es
exactamente lo que dan las redes sin entrenar.

## Los desenlaces, escritos antes

**El eje es UNO: `k` ∈ {5, 7, 9, 11, 13}, con la misma estructura en los cinco brazos** — una
convolución, un mapa, `tl` = máximo y `br` = mínimo. Las otras lecturas quedan anotadas y sin
correr, y `rot` (girar la entrada) queda **descartada**; las dos cosas por orden del dueño del
2026-09-07. Lo que eso permite y lo que impide está en
[`03-alternativas-anotadas.md`](03-alternativas-anotadas.md).

0. **NO SE DECLARA UN GANADOR.** Orden del dueño (2026-09-07): *«No importa quién gana, quiero
   ver todos los ganadores. Es un experimento, no un concurso»*. Así que se reportan **todos** los
   `k` que pasan el umbral, cada uno con sus cinco números (acierto ≤2 px y ≤1 px por esquina,
   error medio, `f1` de `existe`, y la fracción simétrica del kernel), y **no hay regla de
   desempate**.
   ⚠ **Y eso disuelve el problema de la saturación en vez de esconderlo.** En `esq-k` la métrica
   principal saturó —tres brazos al 100 %— y hubo que elegir uno por «más barato», tirando por el
   camino que uno de ellos colocaba la esquina 5× más fino (0,08 px contra 0,40). Reportando
   todos, el que quiera el más barato y el que quiera el más preciso leen la misma tabla.
   ⚠ **El umbral SIGUE HACIENDO FALTA**, y no es lo mismo que un ganador: separa «aprendió» de «se
   quedó en el suelo», que es la única pregunta que un suelo medido puede contestar.
1. **Algún `k` pasa el 12 % en las dos esquinas.** Un solo kernel sirve para las dos, y el
   experimento tiene señal. Se listan **todos** los que pasan, ordenados por `k`, no por
   resultado.
2. **Ningún `k` pasa el 12 %.** *Con esta lectura, un kernel compartido no resuelve las dos
   esquinas.* Es un resultado, no un fracaso.
   ⚠ **Y es exactamente donde este diseño toca su límite, que hay que decir antes**: con un solo
   brazo por `k`, *«un kernel no da para las dos esquinas»* y *«esta lectura no es la buena»* se
   ven **igual**. Distinguirlas es lo que contestarían las alternativas anotadas, y entonces la
   pregunta siguiente no es otro `k`: es otra estructura.
3. **Pasa en `tl` y no en `br`.** El kernel se queda con la mitad que ya sabía hacer.
   ⚠ **Éste es el desenlace que hay que esperar de verdad, no una rareza**, y es la razón de que
   el titular sea la peor de las dos: `tl` es la tarea de `esq-k`, que ya salió al 100 %, y `br`
   es la que pide que el kernel renuncie a su parte simétrica. Un 100 % en `tl` y un 20 % en `br`
   **no** es «medio resuelto»: es el kernel eligiendo la esquina barata.
4. **El eje no es monótono.** En `esq-k` subía monótono. Si aquí no lo hace, **se dice**: con una
   sola semilla no se puede distinguir de una fluctuación, y ésa es una limitación declarada, no
   un fallo del análisis.
5. **El `13` está entre los que pasan y es el borde del rango.** Entonces el eje **sigue sin
   estar acotado por arriba**, que es exactamente lo que dejó abierto `esq-k`, y la respuesta no es «13» sino
   «mirar más allá». El dataset admite hasta **k = 17** sin regenerarlo (lo calcula el
   manifiesto), así que la continuación es barata y está declarada de antemano.
6. **Todos los `k` salen igual de mal, incluido el 13.** No se podrá distinguir «ningún kernel
   sirve» de «la cabeza es demasiado estrecha». Por eso hay que mirar **el mapa de respuesta** de
   las 10 muestras: si el mapa tiene estructura y la lectura falla, el problema es la cabeza; si
   el mapa es plano, es el kernel.

## La comparación con `esq-k`, que es lo que sustituye al control

Sin los brazos `ind-*`, el techo no se mide aquí: se **lee de `esq-k`**, que midió la tarea de
**una** esquina con esta misma convolución y la misma cabeza C1.

| `k` | 5 | 7 | 9 | 11 |
|---|--:|--:|--:|--:|
| `esq-k`, acierto ≤2 px con **una** esquina | 92,3 % | 100 % | 100 % | 100 % |

⚠ **Es una referencia, no un control, y hay DOS motivos —no uno— por los que no es una
comparación limpia:**

1. **Las ventanas son otras.** Allí se sortean 4 `tl` por imagen; aquí 2 `tl` + 2 `br`. Una
   diferencia de pocos puntos no se puede atribuir a nada.
2. **La red no es idéntica**, aunque casi. La convolución sí, y la lectura C1 también; pero la
   cabeza pasa de **3 a 5 parámetros** (un `existe` por esquina) y `br` se lee del **mínimo** del
   mapa, que allí no se leía. O sea que lo que se compara es *la misma convolución con el doble de
   trabajo*, no la misma red.

Lo que sí se puede leer es lo grueso: si aquí sale 100 % donde allí salía 100 %, compartir no
costó nada visible; si aquí sale 40 %, costó, y mucho.

## La predicción registrada, para que pueda fallar

*Escrita antes de entrenar, el 2026-09-07.*

⚠⚠ **Esta estructura tiene una medida EN CONTRA, y hay que dejarlo escrito antes y no después.**
El kernel ganador de `esq-k` —el único kernel de esta familia que se ha entrenado— tiene el
**74,0 % de su energía en la parte simétrica** y suma **−73,68**: es sobre todo un **supresor de
tinta**. Y su **mínimo cae en la mancha de tinta, no en la esquina `br`**: 0/10 páginas, mediana
**91 px** (medido el 2026-09-07, comando en el encargo).

Lo que eso significa exactamente, y lo que no:

- **No significa que `sig` no pueda funcionar.** Aquel kernel se entrenó **sólo para `tl`**, sin
  ninguna presión para poner nada en el mínimo. Uno entrenado para las dos podría colocar `br`
  ahí.
- **Sí significa que el gradiente, cuando se le deja elegir, gasta el kernel en apagar el
  interior del párrafo.** Y para que `br` sea el mínimo hay que renunciar a buena parte de eso.
  Ésa es la tensión, y es lo que este barrido mide.

**Así que la predicción honesta es: NO LO SÉ, y el desenlace 3 (`tl` sí, `br` no) es el más
probable de los cinco.** Lo que sí está predicho:

- **`tl` debería acercarse a lo de `esq-k`** (100 % a partir de k=7). Si `tl` tampoco pasa, lo
  primero que hay que sospechar es el montaje, no la hipótesis.
- **Si `br` falla, la continuación NO es otro `k`: es `ant`** —el mismo `sig` con el kernel
  forzado antisimétrico—, que es lo único que separa «esta lectura no sirve» de «el gradiente no
  llega hasta ella». Está anotada y lista.
- **Y si `br` sale bien, el kernel resultante debe haber bajado mucho su fracción simétrica.** Se
  registra en cada época (`simetrico` / `antisimetrico` en `metrics.jsonl`), así que esta
  predicción se puede contrastar directamente contra el 74,0 % de partida.

## Lo que se congela, y es igual en todos los brazos

dataset **publicado** `esquinas300-32px-r4-r20260907` (en `foveal-vision-data`, con su huella en
`nn/manifiesto.json`) · semilla 1 · ventana 32×32 · sin padding · sin bias · cabeza C1 con β
inicial 3,5 · Adam lr 0,05 · lote 128 · **300 épocas** · las mismas 10 muestras ·
**`λ_coord` = 0,038**.

⚠ **El dato de entrada ya no se re-deriva al empezar: se LEE del repo de datos**, y si no está,
el entrenamiento **se niega** en vez de generarse uno equivalente. Es lo que hace que «el mismo
dataset» sea comprobable y no una intención: los cinco brazos leen el mismo fichero, y el que venga
detrás también.

⚠ **`λ` cambia respecto del 0,03 de `esq-k`, y NO es un descuido.** Lo que se hereda es la
**regla** —«los dos términos de la pérdida parten iguales»—, no el número: la pérdida ya no es la
misma (dos BCE y dos MSE) y la proporción de positivas por esquina pasó del 40 % al 20 %, así que
la BCE inicial sube de ~1,27 a ~1,64. Medido el 2026-09-07 sobre los cinco brazos sin entrenar,
la regla da **0,0366–0,0391**; se congela en **0,038 para todos**. Un `λ` por brazo haría que cada
uno optimizase una función distinta.

**Varía UNA sola cosa: el TAMAÑO DEL KERNEL** (5 · 7 · 9 · 11 · 13 — los mismos de `esq-k` menos
el 3×3, y uno más grande en su lugar). La estructura es la misma en los cinco brazos y todo lo
demás es idéntico. Las otras lecturas quedan **anotadas y sin correr**, y `rot` **descartada**;
las dos cosas por orden del dueño del 2026-09-07.

## Una nota de higiene, para que esto se pueda creer dentro de un año

Este criterio se escribió con **cero épocas corridas**. Después de escribirlo se corrió **una
sola época** en nueve brazos (las cuatro estructuras en varios `k`, cuando todavía estaban las
cuatro) como prueba del mecanismo —que el bucle no se rompe con ninguna— y **sus pesos y sus
métricas se borraron**: `nn/pesos/` no
existe en el commit que trae este fichero. Ninguno de los números de arriba viene de ahí, y los
de esa prueba no se leen: una época no es una medida.

## Lo que este experimento NO contesta

- **Nada sobre señalar las dos esquinas a la vez en la misma vista.** Ninguna ventana contiene
  las dos (párrafo ≥ 64 px, ventana 32) y está comprobado en el manifiesto, no supuesto.
- Nada sobre las otras dos esquinas (`tr`, `bl`), que aquí son negativos duros.
- Nada sobre otras escalas de reducción, otros tamaños de ventana ni el `stride`.
- **Nada que se pueda declarar entre brazos parecidos: una sola semilla.** Es la misma limitación
  que `esq-k` dejó anotada, y se hereda a propósito para no cambiar dos cosas a la vez.
