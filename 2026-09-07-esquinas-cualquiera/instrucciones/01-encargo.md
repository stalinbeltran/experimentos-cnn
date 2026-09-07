# El encargo, tal como llegó (2026-09-07)

> «Ahora copia este mismo experimento, pero ahora no queremos diferenciar las esquinas entre sí.
> Si detecta una esquina, cualquiera (por ahora limitadas a las mismas tl y br) el resultado es
> válido. Dame las estructuras para tratar este caso»

## Qué cambia, exactamente

**Nada de la red, y todo de la etiqueta.** La salida vuelve a ser **una**: `(existe, x, y)`, como
en `esq-k`. `existe` es 1 si la ventana contiene **la superior-izquierda o la inferior-derecha**, y
`(x, y)` es la de la que haya.

- `tr` y `bl` siguen siendo **negativos**: el encargo dice *«por ahora limitadas a las mismas tl y
  br»*. Y son el mejor negativo que hay, porque tienen la misma forma local girada 90°.
- **La cabeza vuelve a 3 parámetros.** En `esq-2d` tuvo que pasar a 5 porque había dos cosas que
  detectar; aquí hay una. La red es **la de `esq-k` sin tocar un parámetro**.
- **No se genera dataset**: se lee el mismo publicado, `esquinas300-32px-r4-r20260907`. Es
  exactamente para lo que aquél se etiquetó con **las cuatro** esquinas.

## Por qué esto no es `esq-2d` con menos exigencia: quita la tensión que lo hundió

Con la descomposición bajo giro de 180°, `W = S + A`, y para el mismo parche:

```
respuesta_tl = <S,P> + <A,P>          respuesta_br = <S,P> − <A,P>
```

`esq-2d` necesitaba que las dos fueran **distintas y en extremos opuestos** (una el máximo, la
otra el mínimo), o sea que mandara `A`. **Aquí hace falta lo contrario**: que las dos sean el
**mismo tipo de extremo**, y eso pasa cuando `<A,P> ≈ 0` — es decir, cuando el kernel es
**simétrico** bajo el giro, y entonces las dos esquinas dan literalmente la misma respuesta.

De ahí salen las tres estructuras, y ninguna es un capricho: son **las tres formas de conseguir
que las dos esquinas sean el mismo extremo**.

| | cómo | grados de libertad del kernel |
|---|---|---|
| **`libre`** | dejar que el gradiente lo encuentre solo | `k²` |
| `sim` | forzar `A = 0` por construcción (`W = (V + rot180(V))/2`) | `(k²+1)/2` |
| `abs` | no tocar el kernel y doblar el **signo** en la lectura: `|M|` | `k²` |

⚠ La tercera es **distinta en especie**: no pide que el kernel sea simétrico, pide que la **cabeza**
no distinga el signo. Un kernel antisimétrico da `+v` en una esquina y `−v` en la otra, y `|M|`
convierte las dos en un pico.

## Lo que se decidió al diseñarlo

- **Se arma `libre` y sólo `libre`**, con `k ∈ {5, 7, 9, 11, 13}`. Es lo que *«copia este mismo
  experimento»* significa literalmente: la red de `esq-k` con otra etiqueta, ni un parámetro de
  diferencia. `sim` y `abs` quedan implementadas, comprobadas y **sin armar**.
- **La comparación con `esq-k` es exacta, y es el regalo de este montaje**: mismas ventanas
  (2.157 / 389 / 120), **mismas 156 positivas en validación**, misma red, mismo `λ` = 0,03
  *medido*, mismo suelo de `f1` = 0,572. Lo único distinto es **qué cuenta como positivo**. Así que
  la diferencia entre los dos barridos **es** el precio de admitir las dos esquinas.
- **Se mide el desglose por clase aunque la tarea no lo pida.** Un 50 % global con `tl` al 100 % y
  `br` a cero no es medio éxito: es `esq-2d` otra vez con otro nombre, y en un número global no se
  vería.
