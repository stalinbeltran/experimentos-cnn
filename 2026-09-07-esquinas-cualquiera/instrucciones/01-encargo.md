# El encargo de `esq-cq`

## La orden, literal (2026-09-07)

> «ahora no se quieren detectar las 2 esquinas por separado, sino cualquiera de ellas. Las
> esquinas siguen siendo las mismas, pero las salidas ahora son una sola. Decimos "hay esquina"
> si la hay, y la posición de ella, sin importar si es tl o br.»

Y antes, al montar la carpeta:

> «copia la carpeta esquinas diagonales, pero con el nombre esquinas cualquiera [...] deja la
> carpeta como si recién se fuera a entrenar (porque se quiere empezar un nuevo experimento, no
> lo lances aún)»

## De dónde sale

De `esq-2d`, que preguntaba si **un** kernel puede servir a **dos** salidas —`tl` leída del
máximo del mapa y `br` del mínimo—. Corrió el 2026-09-07 y su respuesta fue *«sí, pero
malamente y sólo en `tl`»*: `br` se quedó en el 23 % como techo en ventana y **0/10 sobre páginas
enteras en los cinco kernels**.

`esq-cq` **quita la mitad que consistía en distinguir**. No es «`esq-2d` más fácil»: quita la
dificultad barata (decir cuál es) y **conserva entera la cara** (que en `br` no hay señal donde
está la etiqueta).

## Lo que se midió ANTES de diseñar nada

Es la parte que hay que leer, porque cambia lo que se puede esperar. *2026-09-07, sobre las 389
ventanas del `val.npz` publicado, 0 $.*

**`br` no es una esquina de tinta.** La distancia del punto etiquetado al píxel de tinta más
cercano es de **1,0 px en `tl` y 8,1 px en `br`** (p90 12,2), y el cuadrante propio de `br`
(radio 6 px) está **vacío en el 82 %** de sus 78 ventanas. La causa es geométrica: la caja del
párrafo es el rectángulo del layout y **la última línea es corta**, así que su vértice
inferior-derecho cae sobre fondo.

Esto explica `esq-2d` entero sin conjeturas, y **acota este experimento por adelantado**: los
radios del campo receptivo son 2·3·4·5·6 px para `k` = 5..13, y ninguno llega a 8.

```bash
# se reproduce con esto (sólo lectura, instantáneo)
.venv/bin/python 2026-09-07-esquinas-cualquiera/nn/entrenar_local.py --suelos
```

## Lo que cambia respecto de `esq-2d`, y lo que no

| | `esq-2d` | `esq-cq` |
|---|---|---|
| salidas | 2 (`tl`, `br`) | **1** (`existe`, `x`, `y`) |
| lectura | máximo **y** mínimo del mapa | **sólo el máximo** |
| cabeza | 5 parámetros | **3** — la de `esq-k` |
| totales | 30/54/86/126/174 | **28/52/84/124/172** — los de `esq-k` |
| pérdida | 2 BCE + 2 MSE, `λ` = 0,038 | 1 BCE + 1 MSE, **`λ` = 0,0293** (re-medida) |
| métrica titular | la **peor** de dos tasas sobre 78 | **una** tasa sobre 156 |
| alternativa anotada | `ant` (antisimétrico) | **`sim`** (simétrico) |
| dataset | `esquinas300-32px-r4-r20260907` | **el mismo fichero**, sin regenerar |
| eje, semilla, épocas | `k` ∈ {5..13}, semilla 1, 300 | **iguales** |

**Lo que no cambia es deliberado**: mismo dato, mismo eje, misma semilla y mismas épocas es lo
que permite comparar el desglose por esquina columna a columna con `esq-2d`.

## Por qué `ant` se BORRA y entra `sim`

Con dos salidas, toda la diferencia entre esquinas vivía en la parte **antisimétrica** del kernel
(`respuesta_tl = ⟨S,P⟩ + ⟨A,P⟩`, `respuesta_br = ⟨S,P⟩ − ⟨A,P⟩`). Con **una** salida leída del
máximo hay que pedir las **dos** altas, y eso exige `⟨S,P⟩ ≫ |⟨A,P⟩|`, o sea **`A → 0`**.

`ant` no es «la otra opción»: con esta lectura es **incapaz por construcción** — responde a una
esquina con el signo cambiado de la otra, y sólo una de las dos puede ser el máximo. Se borra en
vez de quedarse anotada, que es la regla del propio repo: *«una alternativa descartada que se
queda se acaba armando por error»*.

Su contrapartida, `sim` (`W = (V + rot180(V))/2`), **sí** entra: da equivarianza **exacta** bajo
giro de 180°, hay un test que lo fija, y separa *«el máximo no basta»* de *«el gradiente no llegó
a la simetría»*.

⚠ **Y de paso desaparece la razón de `rot`** (la alternativa que el dueño ya había descartado):
la equivarianza que compraba girando la entrada sale gratis forzando `W` simétrico, sin girar
nada.

## Qué cuesta

**0 máquinas, 0 $.** Entrena en este droplet. `esq-2d` midió ~20 min de reloj para 5 brazos × 300
épocas y 0,4 MB en disco; este montaje es el mismo con **dos parámetros menos** por brazo.

El freno lo ve: `entrenar_local.py` está en la lista `TRABAJOS` de `cerrable.mjs`.
