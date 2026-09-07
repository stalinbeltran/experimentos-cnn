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
| **las 25 redes sin entrenar** | 3,8 – **5,1 %** | **6,4 %** | 0 – 1,3 % · 2,6 – 3,8 % | 5,89 – 6,06 · 6,13 – 6,28 px |

⚠ **El suelo no depende del brazo, y eso es lo que permite un único umbral para los 25.** Los 25
caen en el mismo sitio porque todos arrancan en el mismo modo degenerado, no porque se parezcan:
`ant` con 17 parámetros y `sig-k13` con 174 dan el mismo 5,1 % / 6,4 %.

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

0. **Cómo se lee un resultado de DOS ejes, para no elegir a ojo.** A cada estructura se le
   asigna **su mejor `k`** por la peor-esquina, y las estructuras se comparan **en su mejor `k`**.
   Dentro de una estructura, si dos `k` caen dentro de 2·SE el uno del otro, **gana el más
   pequeño** — misma regla que en `esq-k`, y por el mismo motivo: es más barato. **La
   comparación entre estructuras nunca se hace a `k` fijo**, porque una lectura puede necesitar
   más kernel que otra y eso es parte de lo que cuesta, no un artefacto.
1. **Alguna estructura pasa el 12 % en las dos esquinas.** Un solo kernel sirve para las dos.
   Gana la de **mayor peor-esquina** en su mejor `k`; si dos caen dentro de 2·SE la una de la
   otra, **empate**, y gana la de **menos parámetros totales** (a k = 7: `ant` 29 < `rot` 52 <
   `sig` 54). A igualdad, gana la que **también sirve sobre una página entera**, que es hacia
   dónde va esto.
2. **Sólo pasa `rot`.** Entonces compartir el kernel funciona **compartiendo la VISTA, no el
   MAPA**: un solo mapa no puede llevar las dos esquinas con una cabeza de 5 parámetros. Es un
   resultado, y es el que la descomposición `S`/`A` hace más probable.
3. **`sig` falla y `ant` pasa.** La lectura por signo era correcta y lo que fallaba era llegar
   hasta ella: con `S` libre, el óptimo se queda en el supresor de tinta (que es lo medido en
   `esq-k`: 74 % de la energía en `S`). Forzar `A` es entonces la forma de expresar la tarea.
4. **`sig` pasa y `ant` falla.** La parte simétrica **hace falta** —es la que apaga el interior
   del párrafo— y quitarla entera es demasiado. El eje siguiente sería cuánta `S` conviene, no si
   sí o no.
5. **Ninguna pasa en `br` pero todas en `tl`.** El kernel compartido se queda con la tarea que ya
   sabía hacer. **Aquí es donde el eje del kernel se gana su sitio**: si el fallo se arregla al
   subir `k`, el problema era de **capacidad** y el eje siguiente es cuánto kernel hace falta; si
   no se arregla **en ninguno de los cinco tamaños** —incluido el 13, que tiene 3,4× los pesos
   del 7— es de **estructura**, y eso ya no admite la excusa del borde del rango.
   ⚠ **Y una monotonía que se rompa es un resultado, no ruido que redondear**: en `esq-k` el eje
   subía monótono. Si aquí no lo hace, hay que decirlo, porque con una sola semilla no se puede
   distinguir de una fluctuación — y esa es una limitación declarada, no un fallo del análisis.
6. **Ni siquiera el control `ind-tl` pasa.** Entonces el fallo es **del montaje, no de la
   hipótesis**: `ind-tl` es la red de `esq-k` con otro nombre, y allí llegó al 100 % a ≤2 px.
   Ése es todo el papel del control, y por eso se corre aunque su resultado se dé por sabido.

## La predicción registrada, para que pueda fallar

*Escrita antes de entrenar, a partir de la descomposición y de la sonda del kernel de `esq-k`.*

- **`rot` pasa**, y con holgura: por simetría es la MISMA tarea de `esq-k`, resuelta dos veces
  con los mismos pesos. Si `rot` no pasara, lo que está roto es el montaje.
- **`sig` es la que está en duda.** El kernel de `esq-k` tiene el 74 % de su energía en `S` y su
  mínimo cae en la mancha de tinta (0/10 páginas, mediana 91 px): la lectura por signo pide un
  kernel muy distinto del que el gradiente encontró cuando sólo había una esquina.
- **Compartir NO debería costar nada frente al control** si la estructura es la adecuada: `rot`
  con 52 parámetros hace lo que `ind-tl` + `ind-br` hacen con 104. Si `rot` queda por debajo de
  `min(ind-tl, ind-br)` por más de 2·SE, compartir cuesta, y ese número es el resultado.

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

**Varían dos cosas y sólo dos: la ESTRUCTURA DE LECTURA y el TAMAÑO DEL KERNEL** (5 · 7 · 9 · 11
· 13, por orden del dueño del 2026-09-07: los mismos de `esq-k` menos el 3×3, y uno más grande en
su lugar). Todo lo demás es idéntico en los 25 brazos.

## Una nota de higiene, para que esto se pueda creer dentro de un año

Este criterio se escribió con **cero épocas corridas**. Después de escribirlo se corrió **una
sola época** en nueve brazos (las cuatro estructuras en varios `k`) como prueba del mecanismo —que
el bucle no se rompe con ninguna— y **sus pesos y sus métricas se borraron**: `nn/pesos/` no
existe en el commit que trae este fichero. Ninguno de los números de arriba viene de ahí, y los
de esa prueba no se leen: una época no es una medida.

## Lo que este experimento NO contesta

- **Nada sobre señalar las dos esquinas a la vez en la misma vista.** Ninguna ventana contiene
  las dos (párrafo ≥ 64 px, ventana 32) y está comprobado en el manifiesto, no supuesto.
- Nada sobre las otras dos esquinas (`tr`, `bl`), que aquí son negativos duros.
- Nada sobre otras escalas de reducción, otros tamaños de ventana ni el `stride`.
- **Nada que se pueda declarar entre brazos parecidos: una sola semilla.** Es la misma limitación
  que `esq-k` dejó anotada, y se hereda a propósito para no cambiar dos cosas a la vez.
