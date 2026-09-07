# Las alternativas ANOTADAS, y por qué no se ejecutan

Se arma **`libre`** y sólo `libre`, con `k ∈ {5, 7, 9, 11, 13}`: es lo que *«copia este mismo
experimento»* significa literalmente — la red de `esq-k` con otra etiqueta, ni un parámetro de
diferencia.

⚠ Las otras dos siguen **implementadas** en [`../nn/modelo.py`](../nn/modelo.py) a propósito. Una
estructura implementada es la forma menos ambigua de anotarla —un párrafo de prosa se lee de dos
maneras, un `forward` no— y `python nn/modelo.py` las construye y **comprueba** en cada ejecución.
Una alternativa que se guarda «por si acaso» y deja de ejecutarse se pudre en silencio.

**Armar una es una línea**: añadirla a `BRAZOS`.

## `sim` — el kernel forzado SIMÉTRICO · 28 parámetros a k=7 (la mitad del kernel)

`W = (V + rot180(V))/2` en cada paso. Entonces `respuesta_tl = respuesta_br` **por construcción**,
no por suerte: las dos esquinas dan literalmente el mismo número.

- **Qué contestaría:** separa *«el gradiente encontró la simetría»* de *«la simetría era la
  respuesta y el gradiente no llegaba»*. Y si `libre` funciona, `sim` debería igualarlo con **la
  mitad de grados de libertad** — un `sim` de 7×7 cuesta lo mismo que un `libre` de 5×5.
- **Es la continuación natural**, y el criterio la declara como tal.
- **Coste:** 5 brazos (~15 min, 0 $).

## `abs` — la cabeza lee `|M|` · 52 parámetros a k=7

El kernel sigue libre; lo que cambia es que un pico **negativo** cuenta igual que uno positivo. Un
kernel **orientado** (antisimétrico) da `+v` en una esquina y `−v` en la otra, y `|M|` convierte
las dos en un pico: **sirve para las dos sin ser simétrico**.

- **Qué contestaría:** si la vía buena no era «un filtro que responde igual a las dos» sino «un
  filtro que las distingue, y una lectura a la que le da igual». Son dos soluciones distintas al
  mismo problema, y sólo una de las dos conserva la información de **cuál** es cada esquina — que
  es lo que haría falta si algún día se quiere volver a distinguirlas.
- ⚠ **Su riesgo, escrito antes:** `|M|` hace que el interior del párrafo compita si el kernel tiene
  suma grande, porque ahí la respuesta es grande en valor absoluto. `libre` puede resolverlo con
  un mapa negativo sobre la tinta; `abs` no.
- **Coste:** 5 brazos.

## ⚠ Lo que se pierde al no correrlas

**Si `libre` funciona, poco**: la predicción registrada (que el kernel se vuelve simétrico) se
puede contrastar sin `sim`, porque la fracción simétrica se registra en cada época.

**Si `libre` falla**, este diseño no distingue *«un kernel no da para las dos formas»* de *«la vía
era la simetría forzada y el gradiente no la encontró»* de *«la vía era ignorar el signo»*. Las
tres se ven igual con un solo brazo por `k`, y son las tres preguntas que contestan `sim` y `abs`.
