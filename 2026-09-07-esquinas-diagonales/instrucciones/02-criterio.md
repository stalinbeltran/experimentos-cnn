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
| **los cinco brazos sin entrenar** | 3,8 – **5,1 %** | **6,4 %** | 0 – 1,3 % · 2,6 – 3,8 % | 5,96 – 6,06 · 6,20 – 6,28 px |

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
debe bajar de **5,46 px** (= 5,98 − 2·SE, con SE = 0,26 px). Y `existe`, cuyo suelo es
**f1 = 0,334** por esquina — lo que da el «siempre sí» con 78 positivas de 389, y que es
exactamente lo que dan las redes sin entrenar.

## Los desenlaces, escritos antes

**El eje es UNO: `k` ∈ {5, 7, 9, 11, 13}, con la estructura `rot` en los cinco brazos.** Las otras
lecturas quedan anotadas y sin correr (orden del dueño del 2026-09-07); lo que eso permite y lo
que impide está en [`03-alternativas-anotadas.md`](03-alternativas-anotadas.md).

1. **Algún `k` pasa el 12 % en las dos esquinas.** Un solo kernel sirve para las dos, y el
   experimento tiene señal. **Gana el de mayor peor-esquina**; si dos caen dentro de 2·SE el uno
   del otro, **empate**, y gana el **`k` más pequeño**, por ser más barato — misma regla que en
   `esq-k` y por el mismo motivo.
2. **Ningún `k` pasa el 12 %.** *Con esta lectura, un kernel compartido no resuelve las dos
   esquinas.* Es un resultado, no un fracaso.
   ⚠ **Y es exactamente donde este diseño toca su límite, que hay que decir antes**: con un solo
   brazo por `k`, *«un kernel no da para las dos esquinas»* y *«esta lectura no es la buena»* se
   ven **igual**. Distinguirlas es lo que contestarían las alternativas anotadas, y entonces la
   pregunta siguiente no es otro `k`: es otra estructura.
3. **Pasa en `tl` y no en `br`** (o al revés). El kernel compartido se queda con una de las dos
   mitades. ⚠ En `rot` esto sería **sorprendente y hay que mirarlo dos veces antes de creerlo**:
   la red es literalmente la misma sobre la entrada y sobre la entrada girada, así que las dos
   esquinas son la MISMA tarea. Una asimetría grande aquí apunta antes al dato —¿son igual de
   difíciles las dos esquinas en este dataset?— que a la red.
4. **El eje no es monótono.** En `esq-k` subía monótono. Si aquí no lo hace, **se dice**: con una
   sola semilla no se puede distinguir de una fluctuación, y ésa es una limitación declarada, no
   un fallo del análisis.
5. **El `13` gana y es el borde del rango.** Entonces el eje **sigue sin estar acotado por
   arriba**, que es exactamente lo que dejó abierto `esq-k`, y la respuesta no es «13» sino
   «mirar más allá». El dataset admite hasta **k = 17** sin regenerarlo (lo calcula el
   manifiesto), así que la continuación es barata y está declarada de antemano.
6. **Todos los `k` salen igual de mal, incluido el 13.** No se podrá distinguir «ningún kernel
   sirve» de «la cabeza es demasiado estrecha». Por eso hay que mirar **el mapa de respuesta** de
   las 10 muestras: si el mapa tiene estructura y la lectura falla, el problema es la cabeza; si
   el mapa es plano, es el kernel.

## La comparación con `esq-k`, que es lo que sustituye al control

Sin los brazos `ind-*`, el techo no se mide aquí: se **lee de `esq-k`**, que midió la tarea de
**una** esquina con esta misma red (`k²+3` parámetros, cabeza C1 de 3) sobre el mismo tipo de dato.

| `k` | 5 | 7 | 9 | 11 |
|---|--:|--:|--:|--:|
| `esq-k`, acierto ≤2 px con **una** esquina | 92,3 % | 100 % | 100 % | 100 % |

⚠ **Es una referencia, no un control, y la diferencia importa:** las ventanas de aquel dataset son
otras (allí 4 `tl` por imagen; aquí 2 `tl` + 2 `br`), así que una diferencia de pocos puntos entre
los dos barridos **no se puede atribuir** a compartir el kernel. Lo que sí se puede leer es lo
grueso: si aquí sale 100 % donde allí salía 100 %, compartir no costó nada visible; si aquí sale
40 %, costó, y mucho.

## La predicción registrada, para que pueda fallar

*Escrita antes de entrenar.* **`rot` pasa, y con holgura**, porque por simetría es la MISMA tarea
de `esq-k` resuelta dos veces con los mismos pesos: si no pasara, lo primero que hay que sospechar
es el montaje, no la hipótesis. Y **el `k` ganador debería parecerse al de allí** (7 por el
criterio de empate; 9-13 si se mira el error de posición). ⚠ Lo que **no** está predicho es el
suelo del rango: el 5 allí acertó el 92,3 % con una esquina, y aquí tiene que repartir 25 pesos
entre dos.

## Lo que se congela, y es igual en todos los brazos

dataset **publicado** `esquinas300-32px-r4-r20260907` (en `foveal-vision-data`, con su huella en
`nn/manifiesto.json`) · semilla 1 · ventana 32×32 · sin padding · sin bias · cabeza C1 con β
inicial 3,5 · Adam lr 0,05 · lote 128 · **300 épocas** · las mismas 10 muestras ·
**`λ_coord` = 0,038**.

⚠ **El dato de entrada ya no se re-deriva al empezar: se LEE del repo de datos**, y si no está,
el entrenamiento **se niega** en vez de generarse uno equivalente. Es lo que hace que «el mismo
dataset» sea comprobable y no una intención: los 25 brazos leen el mismo fichero, y el que venga
detrás también.

⚠ **`λ` cambia respecto del 0,03 de `esq-k`, y NO es un descuido.** Lo que se hereda es la
**regla** —«los dos términos de la pérdida parten iguales»—, no el número: la pérdida ya no es la
misma (dos BCE y dos MSE) y la proporción de positivas por esquina pasó del 40 % al 20 %, así que
la BCE inicial sube de ~1,27 a ~1,64. Medido el 2026-09-07 sobre las seis redes sin entrenar, la
regla da **0,0375–0,0400**; se congela en **0,038 para todos**. Un `λ` por brazo haría que cada
uno optimizase una función distinta.

**Varía UNA sola cosa: el TAMAÑO DEL KERNEL** (5 · 7 · 9 · 11 · 13 — los mismos de `esq-k` menos
el 3×3, y uno más grande en su lugar). La estructura es `rot` en los cinco brazos, y todo lo demás
es idéntico. Las otras lecturas quedan **anotadas y sin correr**, las dos cosas por orden del
dueño del 2026-09-07.

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
