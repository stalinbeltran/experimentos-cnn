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
| las seis redes **sin entrenar** | 3,8 – **5,1 %** | **6,4 %** | 1,3 % · 2,6-3,8 % | 5,96-5,99 · 6,13-6,22 px |

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

1. **Alguna estructura pasa el 12 % en las dos esquinas.** Un solo kernel sirve para las dos.
   Gana la de **mayor peor-esquina**; si dos caen dentro de 2·SE la una de la otra, **empate**, y
   gana la de **menos parámetros** (`ant` 29 < `rot` 52 < `sig` 54). A igualdad de parámetros,
   gana la que **también sirve sobre una página entera**, que es hacia donde va esto.
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
   sabía hacer. Sólo entonces se lee `sig11`: si con k = 11 sí pasa, el problema era de
   **capacidad**; si tampoco, es de **estructura**.
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

dataset (huella en `nn/manifiesto.json`) · semilla 1 · ventana 32×32 · sin padding · sin bias ·
cabeza C1 con β inicial 3,5 · Adam lr 0,05 · lote 128 · **300 épocas** · las mismas 10 muestras ·
**`λ_coord` = 0,038**.

⚠ **`λ` cambia respecto del 0,03 de `esq-k`, y NO es un descuido.** Lo que se hereda es la
**regla** —«los dos términos de la pérdida parten iguales»—, no el número: la pérdida ya no es la
misma (dos BCE y dos MSE) y la proporción de positivas por esquina pasó del 40 % al 20 %, así que
la BCE inicial sube de ~1,27 a ~1,64. Medido el 2026-09-07 sobre las seis redes sin entrenar, la
regla da **0,0375–0,0400**; se congela en **0,038 para todos**. Un `λ` por brazo haría que cada
uno optimizase una función distinta.

**Lo único que varía entre los brazos que compiten es la ESTRUCTURA DE LECTURA.** `sig11` varía
además `k`, y por eso está declarado como punto de seguro y no como parte del eje: sólo se lee en
el desenlace 5.

## Una nota de higiene, para que esto se pueda creer dentro de un año

Este criterio se escribió con **cero épocas corridas**. Después de escribirlo se corrió **una
sola época por brazo** como prueba del mecanismo —que el bucle no se rompe con ninguna de las
cuatro estructuras— y **sus pesos y sus métricas se borraron**: `nn/pesos/` no existe en el
commit que trae este fichero. Ninguno de los números de arriba viene de ahí.

## Lo que este experimento NO contesta

- **Nada sobre señalar las dos esquinas a la vez en la misma vista.** Ninguna ventana contiene
  las dos (párrafo ≥ 64 px, ventana 32) y está comprobado en el manifiesto, no supuesto.
- Nada sobre las otras dos esquinas (`tr`, `bl`), que aquí son negativos duros.
- Nada sobre otras escalas de reducción, otros tamaños de ventana ni el `stride`.
- **Nada que se pueda declarar entre brazos parecidos: una sola semilla.** Es la misma limitación
  que `esq-k` dejó anotada, y se hereda a propósito para no cambiar dos cosas a la vez.
