# Las alternativas: qué se armó, qué se anotó y qué se borró

Registro de lo que **no** se corre, con lo que cada cosa contestaría y lo que cuesta. Se mantiene
porque una alternativa que sólo vive en la cabeza de alguien se vuelve a proponer cada mes.

## ✅ Armada: `max` — la que se corre

Un mapa, **una** salida, leída del **máximo**. Cabeza de 3 parámetros. Es lo que pidió el dueño y
lo que describe [`02-criterio.md`](02-criterio.md).

## 📝 Anotada y NO armada: `sim` — el kernel forzado simétrico

`W = (V + rot180(V))/2`, o sea `A = 0` por construcción. **(k²+1)/2 grados de libertad** en el
kernel (28 a `k`=7 contra 52 de `max`).

**Qué contestaría:** separa *«el máximo no basta para esta tarea»* de *«el gradiente no llegó a la
simetría por su cuenta»*. Si `max` falla y `sim` también, el problema es la lectura; si `sim`
mejora, el problema era la optimización.

**Qué cuesta:** un brazo más, ~4 min, 0 $. Armarlo es añadir su línea a `BRAZOS` en
`nn/modelo.py` — ya está implementado y **comprobado en cada ejecución** de `nn/modelo.py`, con
un test de equivarianza exacta bajo giro de 180°.

**Por qué no se arma ya:** el experimento tiene **un** eje (`k`), por la misma orden que rigió en
`esq-2d` — *«Anótalas pero no vamos a ejecutarlas. Sólo variamos los kernels»*. `sim` es un eje de
estructura, no de tamaño.

## ❌ BORRADA: `ant` — el kernel forzado antisimétrico

**Era «la continuación declarada» de `esq-2d`, y con esta lectura es incapaz por construcción.**

Un kernel antisimétrico responde a un parche y a su giro de 180° con **el mismo número cambiado de
signo**. Con dos salidas eso era exactamente lo que hacía falta: `tl` en el máximo y `br` en el
mínimo. Con **una** salida leída del máximo, sólo una de las dos esquinas puede ser el máximo — la
otra es, por construcción, el mínimo. Pedirle a `ant` que resuelva `esq-cq` es pedirle lo
contrario de lo que hace.

**Se borra el código, no se deja anotado**, y es la regla del propio repo: *«una alternativa
descartada que se queda en el repo se acaba armando por error»*. Sigue existiendo en `esq-2d`,
donde tiene sentido.

## ❌ Descartada por el dueño (2026-09-07): `rot` — girar la entrada 180°

*«No queremos girar el kernel».* Se mantiene descartada.

⚠ **Y ahora, además, ha perdido su razón de ser:** lo que `rot` compraba era equivarianza exacta
bajo giro, y eso sale **gratis** forzando `W` simétrico (`sim`), sin girar ninguna entrada ni
duplicar el coste del forward.

## ❌ Ya no aplica: el control `ind` (una red por esquina)

En `esq-2d` era el techo: dos redes independientes, sin compartir kernel, cada una con **una**
esquina. **Eso es exactamente lo que `esq-cq` hace ahora en el caso normal** —una salida—, sólo
que sin decir cuál esquina es. Como control ya no distingue nada, así que sale.

El techo de esta tarea se lee de `esq-k`, que midió **una** esquina con esta **misma** cabeza de 3
parámetros: 92,3 % (k=5) y 100 % (7·9·11). ⚠ Sigue siendo una **referencia, no un control** — otro
dataset y sólo `tl` —, pero por primera vez **la red es idéntica parámetro a parámetro**, que era
uno de los dos defectos que `esq-2d` declaraba en esa comparación.

## 🔓 Abierta, y es la decisión con coste: extender el eje a `k` ∈ {15, 17}

**No decidida.** El rango es hoy {5,7,9,11,13}, heredado de `esq-2d`.

**El argumento para extenderlo** es una medida, no una corazonada: la tinta más cercana a `br`
está a **8,1 px** de mediana, y los radios de campo receptivo del rango actual son **2·3·4·5·6
px**. **Ningún brazo puede ver la tinta de `br`.** El primer `k` con radio ≥ 8 es **17**, que es
justo el `k_max_representable` del dataset (`k`=19 ya pediría regenerarlo).

**El argumento para no hacerlo**: con {5..13} las cinco filas se leen columna a columna contra las
de `esq-2d` sin traducir nada.

**Coste de extender:** 2 brazos más, **≈ +6 min de reloj, 0 $**. Es un cambio de una línea
(`K_BARRIDO` en `nn/modelo.py`).
