# Las alternativas ANOTADAS, y por qué no se ejecutan

**Dos órdenes del dueño, 2026-09-07:**

> *«Veo q propones varias estructuras para probar distintas alternativas. Anótalas pero no vamos a
> ejecutarlas. Sólo variamos los kernels.»*
>
> *«No queremos girar el kernel. Elimina esa parte. Debe ser idéntica al exper anterior, sólo que
> en vez de 1 esquina van a ser 2.»*

Así que este experimento corre **una sola estructura** —`sig`, un mapa y sus dos extremos— con
`k ∈ {5, 7, 9, 11, 13}`, y lo de aquí es el registro de lo que se deja fuera: qué contestaría cada
una, qué costaría, y cómo se pone en marcha si algún día toca.

⚠ **Siguen IMPLEMENTADAS en [`../nn/modelo.py`](../nn/modelo.py) a propósito.** Una estructura
implementada es la forma menos ambigua de anotarla —un párrafo de prosa se puede leer de dos
maneras, un `forward` no—, y `python nn/modelo.py` las construye y las comprueba en cada
ejecución. Una alternativa que se guarda «por si acaso» y deja de ejecutarse se pudre en
silencio, y el día que se arme habría que depurarla desde cero.

**Armar una es una línea**: añadirla a `BRAZOS`. Nada más.

## Lo que las ordena a todas: la descomposición bajo giro de 180°

Todo kernel se parte en `W = S + A` (simétrica y antisimétrica bajo el giro), y como el parche de
una esquina `br` es el giro del de una `tl`:

```
respuesta_tl = <S,P> + <A,P>          respuesta_br = <S,P> − <A,P>
```

Las dos respuestas son **simétricas respecto de `<S,P>`**, y toda la diferencia entre esquinas
vive en `A`. Cada alternativa es una apuesta distinta sobre qué parte manda.

## ❌ `rot` — DESCARTADA por el dueño, y no se implementa

*«No queremos girar el kernel. Elimina esa parte.»* (2026-09-07)

Era: aplicar el **mismo** kernel a la entrada **y a la entrada girada 180°**, y leer las dos
esquinas con la misma cabeza. **Su código está borrado**, y es deliberado: una alternativa
descartada que se queda en el repo se acaba armando por error. Queda aquí, y sólo aquí, para que
el que la vuelva a proponer sepa que ya se decidió.

Lo que ofrecía, para que la decisión se pueda revisar con lo que costaba: hacía que las dos
esquinas fueran **exactamente la misma tarea** (equivarianza exacta), así que ninguna mitad del
filtro tenía que ceder — que es justo la tensión que sí tiene la estructura elegida.

## Las que quedan fuera pero siguen implementadas

### `ant` — `sig` con el kernel forzado antisimétrico · **29 parámetros a k=7**

`W = (V − rot180(V))/2` en cada paso: `S = 0` por construcción, así que `respuesta_br =
−respuesta_tl` deja de ser una esperanza y pasa a ser **exacta**. La mitad de grados de libertad
(29 a k=7 contra 54).

- **Qué contestaría:** separa *«esta lectura no funciona»* de *«el gradiente no llega hasta
  ella»*. Si el barrido falla con `S` libre —que es como corre ahora— no se puede saber cuál de
  las dos cosas pasó, porque el óptimo cómodo (el supresor de tinta) se la come.
- **Es la continuación natural si el barrido sale mal**, y por eso es la que conviene tener a
  mano: contesta la pregunta que el fracaso deja abierta.
- **Coste:** 5 brazos más (~15 min, 0 $).

### `ind-tl` · `ind-br` — el CONTROL: dos redes sin compartir nada · **52 + 52 a k=7**

Cada esquina con su propio kernel, como en `esq-k`. **No es una candidata**: es el **techo**
contra el que se mide qué cuesta compartir.

- **Qué contestaría:** si `rot` queda por debajo del control por más de 2·SE, compartir el kernel
  cuesta, y ese número **es** el resultado.
- **Coste:** 10 brazos (dos esquinas × 5 `k`), que dobla el experimento.

## ⚠ Lo que se pierde al no correrlas, dicho por delante

**Si el barrido de `k` sale bien, no se pierde nada**: un solo kernel con `k²+5` parámetros
resuelve las dos esquinas, y ésa es la respuesta.

**Si sale mal, este diseño no puede distinguir dos cosas**: *«un solo kernel no da para las dos
esquinas»* de *«esta lectura no es la buena»*. La primera es una conclusión sobre el problema; la
segunda, sobre mi elección de estructura. Con un solo brazo por `k`, las dos se ven igual.

Y hay un sustituto **parcial** para el control, que conviene usar con su advertencia: `esq-k`
midió la tarea de **una** esquina en `k` ∈ {5,7,9,11} sobre el mismo tipo de dato —92,3 % · 100 %
· 100 % · 100 % a ≤2 px—, así que sirve de referencia aproximada del techo. ⚠ **Aproximada y no
exacta**: sus ventanas son otras (allí 4 `tl` por imagen, aquí 2 `tl` + 2 `br`), así que una
diferencia de pocos puntos entre los dos barridos **no se puede atribuir** a compartir el kernel.
