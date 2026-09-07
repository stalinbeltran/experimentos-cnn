# El criterio de `esq-cq`, congelado ANTES de la primera época

**Escrito el 2026-09-07 con CERO épocas entrenadas.** Los suelos de aquí abajo salen de
`nn/entrenar_local.py --suelos` sobre el `val.npz` publicado, no de memoria. Nada de este
documento se toca después de mirar los resultados: si algo no estaba previsto, se anota en el
reporte como no previsto, que es distinto de reescribir el criterio.

## La pregunta

Orden del dueño (2026-09-07), literal:

> «ahora no se quieren detectar las 2 esquinas por separado, sino cualquiera de ellas. Las
> esquinas siguen siendo las mismas, pero las salidas ahora son una sola. Decimos "hay esquina"
> si la hay, y la posición de ella, sin importar si es tl o br.»

Una convolución `k×k` sin bias produce un mapa; la posición sale de su **máximo** y `existe` de
la altura de ese máximo. **Cabeza de 3 parámetros** (`β`, `a`, `b`) — exactamente la de `esq-k`.

## Los suelos, medidos

*`nn/entrenar_local.py --suelos`, 2026-09-07, sobre las 389 ventanas de `val`.*

| | valor |
|---|--:|
| positivas (`tl` ∪ `br`) | **156 / 389 = 40,1 %** — 78 `tl` + 78 `br` |
| negativo duro (`tr`/`bl`) | 78 |
| acierto ≤2 px **sin entrenar** | **5,8 %** (5,1 % en `tl` · 6,4 % en `br`) |
| error medio sin entrenar | **6,08 px** (el predictor constante en el centro) |
| `f1` del «siempre sí» | **0,572** |
| `λ` que iguala los dos términos | **0,0293** (congelada; el 0,038 era de otra pérdida) |

## El umbral: qué cuenta como «ha aprendido algo»

**Acierto a ≤2 px > 9,6 %**, que es el suelo (5,8 %) más 2 SE, con SE = √(p(1−p)/156) = 1,9 %.

- Secundario: error medio **< 5,70 px** y `f1` de `existe` **> 0,572**.
- **No se declara ganador.** Se reportan **todos** los que pasan, como en `esq-2d`.

## ⚠⚠ Lo que este experimento puede aparentar y no ser

**Las positivas son mitad `tl` y mitad `br`. Un resultado en torno al 50 % es exactamente lo que
sale de "`tl` entero, `br` nada".** No es una sospecha: es lo que predice la medida de abajo.

Por eso **el desglose por esquina verdadera se reporta siempre, desde la primera época**, y por
eso la métrica titular va acompañada de sus dos columnas. Un número global sin desglose, en este
experimento, es un autoengaño con forma de éxito.

## La medida que se hizo ANTES de diseñar nada

*2026-09-07, sobre las 389 ventanas del `val.npz` publicado. Distancia del punto etiquetado al
píxel de tinta más cercano (robusta al umbral de binarización: 0, 32 y 64 dan lo mismo).*

| esquina | mediana | p90 | máx |
|---|--:|--:|--:|
| `tl` | **1,00 px** | 1,00 | 1,41 |
| `bl` | 2,00 px | 3,00 | 3,16 |
| `tr` | 6,00 px | 9,22 | 10,63 |
| **`br`** | **8,06 px** | **12,23** | 17,00 |

Y la masa de tinta en el cuadrante propio (radio 6 px): `tl` **0,431** · `br` **0,014**, con el
cuadrante **completamente vacío en 64 de las 78** ventanas (**82 %**).

**`br` no es una esquina de tinta: es el vértice inferior-derecho de la caja del layout.** La
última línea de un párrafo es corta, así que ese vértice cae sobre fondo.

✅ **Y por eso el eje se extendió a `k` ∈ {5,7,9,11,13,15,17}** (decisión del dueño, 2026-09-07,
tomada con esta medida delante). El radio del campo receptivo es `(k−1)/2` = **2·3·4·5·6·7·8 px**.
Ningún brazo de `k` ≤ 13 alcanza los 8,1 px donde empieza la tinta de `br`; **`k17` es el primero
que llega**, y es además el techo del dataset (su mapa cubre [8,23], justo el rango donde se
sortean las esquinas).

⚠ **Lo que cuesta**: las filas de `k05`..`k13` se siguen leyendo columna a columna contra
`esq-2d`; `k15` y `k17` **no tienen contraparte allí** y sólo se comparan entre sí.

## Los desenlaces, por orden de probabilidad estimada

1. **`tl` sí, `br` no** *(el más probable, y por un mecanismo medido, no por intuición)*. El
   global se queda cerca del 50 %, el desglose enseña ~100 % / ~0 %. **Cuenta como resultado, no
   como fracaso**: cierra que el problema no es la lectura ni el reparto de salidas, sino que en
   `br` no hay nada que ver a esa escala.
2. **Los dos suben con `k` hasta el borde.** Con el eje ya extendido a 17 —que es el techo del
   dataset— si `k17` sigue siendo el mejor, la continuación **ya no es subir `k`**: haría falta
   regenerar el dato con otra ventana. Es una diferencia real con `esq-2d`, donde subir era
   gratis.
3. **El global supera el umbral pero el kernel se vuelve simétrico y dispara los falsos positivos
   en `tr`/`bl`.** Un kernel simétrico bajo giro de 180° responde igual a las cuatro esquinas si
   no aprende otra cosa; por eso `fp_otra_diagonal` se mide en cada época.
4. **No pasa nada del umbral en ningún brazo.** Entonces la lectura por máximo no sirve para esta
   tarea y la continuación declarada es `sim` (ver `03-alternativas-anotadas.md`).

## La predicción registrada, que se contrasta gratis

**La fracción SIMÉTRICA del kernel debe SUBIR al entrenar.** Con lectura por máximo hay que pedir
las dos esquinas altas, y eso exige `⟨S,P⟩ ≫ |⟨A,P⟩|`, o sea `A → 0`. Es la inversa exacta de
`esq-2d`, donde toda la diferencia entre esquinas vivía en `A`.

Contra qué se contrasta, sin gastar nada: `esq-k` acabó en **74,0 %** simétrico y `esq-2d` en
**38–53 %**. Se registra en cada época (`simetrico` / `antisimetrico` en `metrics.jsonl`).

⚠ **Y el álgebra descansa en un supuesto que hoy sabemos FALSO**: `esq-2d` la construía sobre
«el parche de una `br` es el giro del de una `tl`». Con 1,0 px de tinta en una y 8,1 px en la
otra, **no lo es**. Simetrizar ayuda pero no puede cerrar la tarea.

## Contra qué se compara, y contra qué NO

| | comparable | no comparable |
|---|---|---|
| **`esq-k`** | la **arquitectura**: misma conv, misma cabeza de 3, misma lectura por máximo, **mismos totales** (28/52/84/124) y mismo suelo de `f1` (0,572) | el **dato** (otro dataset) y la **tarea** (allí sólo `tl`) |
| **`esq-2d`** | el **dato**, de forma exacta: mismo fichero publicado, mismo split, misma semilla, mismo eje, mismas 300 épocas | la **métrica titular**: allí es *la peor de dos tasas sobre 78*, aquí *una tasa sobre 156*. **Un 50 % aquí no es un 50 % allí** |

⚠ **Lo único comparable columna a columna con `esq-2d` es el desglose por esquina verdadera**,
sobre las mismas 78 + 78 ventanas. Por eso el desglose es el ancla de comparabilidad y no un
diagnóstico opcional.

## Qué NO contesta este experimento

- **Nada sobre la página entera.** La ventana es 32×32; el filtro sobre páginas se mide aparte
  con `nn/transformacion.py`, y en `esq-2d` el orden de los brazos **no se conservó** entre las
  dos escalas (`k11` era 2.º en ventana y 1/10 en página).
- **Nada sobre `tr`/`bl`**: aquí son el negativo duro.
- **Una semilla**, como en los dos anteriores. Las diferencias pequeñas entre brazos vecinos no se
  van a poder declarar.
