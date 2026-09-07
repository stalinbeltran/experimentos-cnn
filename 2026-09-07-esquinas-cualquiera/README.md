# `esq-cq` — con UN kernel, ¿qué tamaño detecta CUALQUIERA de las dos esquinas?

**Lanzado el 2026-09-07 · 7 brazos × 300 épocas · 0 máquinas · 0 $** (CPU de este droplet). El criterio está
congelado **antes** de mirar en
[`instrucciones/02-criterio.md`](instrucciones/02-criterio.md), y no declara ganador: se
reportan todos los que pasan.

## Qué pregunta

Una sola salida: **`existe`** («hay esquina») y **una posición**, sin decir si la esquina es la
superior-izquierda o la inferior-derecha. Orden del dueño (2026-09-07):

> «no se quieren detectar las 2 esquinas por separado, sino cualquiera de ellas [...] Decimos
> "hay esquina" si la hay, y la posición de ella, sin importar si es tl o br.»

Es `esq-2d` **quitándole la mitad que consistía en distinguir**. No es «`esq-2d` más fácil»:
quita la dificultad barata y conserva entera la cara (ver abajo).

## La estructura

```
entrada x (1, 32, 32) en TINTA (/255)
    │
    └── conv(x, W)  ──►  M  (m×m, m = 32 − k + 1)   ── UN mapa, sin bias, sin padding
                          │
                          └── softmax(β·M) ──► (x, y) del MÁXIMO   ── LA posición
                              existe = a · logsumexp(β·M)/β + b

β se aprende (arranca en 3,5). Cabeza de 3 parámetros: β, a, b.
```

**La cabeza vuelve a ser la de `esq-k`**, y con ella los totales: **28 · 52 · 84 · 124 · 172 ·
228 · 292** para `k` = 5·7·9·11·13·15·17. Eso vale más que la cifra — de los dos defectos que `esq-2d` declaraba al
compararse con `esq-k`, el de *«la red no es idéntica»* desaparece.

```bash
.venv/bin/python 2026-09-07-esquinas-cualquiera/nn/modelo.py   # la tabla y los dos invariantes
```

### El mínimo ya no se lee, y eso le da la vuelta a lo que se le pide al kernel

En `esq-2d`, `respuesta_tl = ⟨S,P⟩ + ⟨A,P⟩` y `respuesta_br = ⟨S,P⟩ − ⟨A,P⟩`: toda la diferencia
entre esquinas vivía en la parte **antisimétrica**. Con **una** salida leída del máximo hay que
pedir las **dos** altas, y eso exige `⟨S,P⟩ ≫ |⟨A,P⟩|`, o sea **`A → 0`**.

Es la inversa exacta. Por eso **`ant` se borra** (con esta lectura es incapaz por construcción) y
entra **`sim`**, el kernel forzado simétrico, que da equivarianza exacta bajo giro de 180° y tiene
su test. Detalle en
[`instrucciones/03-alternativas-anotadas.md`](instrucciones/03-alternativas-anotadas.md).

## ⚠⚠ Lo que se midió ANTES de diseñar nada: `br` no es una esquina de tinta

*2026-09-07, sobre las 389 ventanas del `val.npz` publicado, 0 $. Distancia del punto etiquetado
al píxel de tinta más cercano — robusta al umbral (0, 32 y 64 dan lo mismo).*

| esquina | mediana | p90 | máx |
|---|--:|--:|--:|
| `tl` | **1,00 px** | 1,00 | 1,41 |
| `bl` | 2,00 px | 3,00 | 3,16 |
| `tr` | 6,00 px | 9,22 | 10,63 |
| **`br`** | **8,06 px** | **12,23** | 17,00 |

Masa de tinta en el cuadrante propio (radio 6 px): `tl` **0,431** · `br` **0,014**, con el
cuadrante **vacío en 64 de 78** ventanas (**82 %**).

**La caja del párrafo es el rectángulo del layout y la última línea es corta**, así que su
vértice inferior-derecho cae sobre fondo. `br` es un vértice de *bounding box*, no una esquina de
tinta — y se ve a simple vista en `muestras/k13-ep000-sin-entrenar.png`, paneles 4·5·6.

✅ **Y por eso el eje llega hasta 17, que es lo que `esq-2d` no hizo.** Los radios de campo
receptivo son `(k−1)/2`:

| `k` | 5 | 7 | 9 | 11 | 13 | **15** | **17** |
|---|--:|--:|--:|--:|--:|--:|--:|
| radio | 2 px | 3 px | 4 px | 5 px | 6 px | **7 px** | **8 px** |

**Ningún brazo de `k` ≤ 13 puede VER la tinta de `br`** (8,1 px). `k17` es el primero que llega, y
además es el **techo del dataset**: su mapa 16×16 cubre exactamente [8, 23], que es donde se
sortean las esquinas; con `k`=19 habría esquinas no representables. Con el rango viejo, el
resultado en `br` estaba escrito de antemano — esto explica `esq-2d` entero (23 % de techo, 0/10
en página) sin conjeturas.

## ⚠ El desglose NO es diagnóstico opcional

**Las positivas son mitad `tl` y mitad `br`, así que ~50 % es exactamente lo que sale de "`tl`
entero, `br` nada".** Un número global sin desglose, aquí, es un autoengaño con forma de éxito.

Y es además **el ancla de comparabilidad**: la métrica titular de `esq-cq` (una tasa sobre 156) y
la de `esq-2d` (la peor de dos tasas sobre 78) **no son el mismo número**. Lo único que se puede
poner columna a columna son las tasas por esquina verdadera, sobre las mismas 78 + 78 ventanas.

Por eso el desglose sale en cada época, en `--suelos`, en el informe y en las figuras.

## Los suelos, medidos antes de entrenar

| | |
|---|--:|
| positivas | **156 / 389 = 40,1 %** (78 `tl` + 78 `br`; 78 de la otra diagonal son negativo duro) |
| acierto ≤2 px sin entrenar | **5,8 %** (5,1 `tl` · 6,4 `br`) |
| error medio | **6,08 px** |
| `f1` del «siempre sí» | **0,572** |
| **umbral del criterio** | **> 9,6 %** (suelo + 2 SE) |
| `λ` de la pérdida | **0,0293**, re-medida y congelada (el 0,038 era de otra pérdida) |

```bash
.venv/bin/python 2026-09-07-esquinas-cualquiera/nn/entrenar_local.py --suelos
```

## La predicción registrada, que se contrasta gratis

**La fracción SIMÉTRICA del kernel debe SUBIR al entrenar.** `esq-k` acabó en **74,0 %** y
`esq-2d` en **38–53 %**; se registra en cada época. Es la consecuencia directa del álgebra de
arriba, y contrastarla no cuesta una corrida nueva.

⚠ Con la salvedad de que **el supuesto de esa álgebra es falso**: `esq-2d` la construía sobre
«el parche de una `br` es el giro del de una `tl`», y con 1,0 px de tinta en una y 8,1 en la otra
**no lo es**. Simetrizar ayuda, no cierra.

## El dataset: el mismo fichero, sin regenerar nada

```
foveal-vision-data/experimentos-cnn/esquinas300-32px-r4-r20260907/
```

Se reusa **tal cual** —mismos sha, mismo split, misma semilla— porque etiqueta **las cuatro**
esquinas desde el principio, justo para que un experimento que mire otra combinación no tenga que
re-rendir nada. `esq-cq` deriva su etiqueta con `datos.objetivo()`, **la puerta única** que
importan los tres consumidores (entrenamiento, muestras e informe).

⚠ **Y ahí se arregló un aval que era vacuo.** El manifiesto decía «0 ventanas con más de una
esquina» y ese contador se calculaba sobre la etiqueta que `_ventanas` acababa de escribir —una
sola por ventana—, así que **no podía dar otra cosa que 0**: un assert incapaz de fallar. Ahora se
cuenta contra la **geometría** (`_esquinas_dentro`). El dato publicado **no se reescribe**; la
comprobación independiente sobre él da igualmente **0 de 389**, así que la conclusión se sostiene
— lo que no se sostenía era la prueba.

## Cómo se corre

```bash
cd ~/src/experimentos-cnn
E=2026-09-07-esquinas-cualquiera
.venv/bin/python $E/nn/entrenar_local.py --suelos     # sin entrenar nada
$E/nn/entrenar_local.py --init                        # los pesos de la epoca 0 (no pisa)
$E/nn/lanzar_barrido.sh                               # 7 brazos (unidad `esqcq-barrido`)
$E/nn/lanzar_barrido.sh --estado                      # ¿viva? ¿por dónde? ¿NRestarts?
```

Corre como **unidad de systemd** (padre PID 1): sobrevive al fin del turno y al reinicio del
coordinador, y se niega a lanzarse dos veces. El estado se lee del **disco**, no del log —
`python` bufferiza cuando no es un tty.

## Cuánto cuesta

**0 máquinas y 0 $.** `esq-2d` midió ~20 min para 5 brazos × 300 épocas y 0,4 MB en disco; este
montaje es el mismo con dos parámetros menos por brazo. El freno lo ve (`entrenar_local.py` está
en la lista `TRABAJOS` de `cerrable.mjs`).

## ✅ El punto de partida está medido y guardado (época 0)

Antes de entrenar se guardaron los pesos iniciales de los 7 brazos
(`nn/entrenar_local.py --init`) y se dibujó el suelo en las dos escalas. **No es adorno: la figura
de página entera necesita un `best.pt` en disco**, así que sin esto el punto de partida a esa
escala no se puede dibujar nunca más.

| | resultado en la época 0 |
|---|---|
| `muestras/k*-ep000-sin-entrenar.png` | **0/6** aciertos en las 10 ventanas congeladas, en los 7 brazos |
| `muestras/transformacion-k*-10-paginas-ep000-sin-entrenar.png` | **0/10** en los 7, mediana 42–52 px |

⚠ Las figuras llevan la **época en el nombre**. Antes se deducía de si el fichero de pesos
existía, así que en cuanto `--init` los creó, una red de época 0 pasó a llamarse `-actual`: el
nombre decía «entrenada» de algo que no había entrenado nada. La época es un dato, se lee.

## Lo que este experimento NO contesta

- **Nada sobre la página entera.** Se mide aparte con `nn/transformacion.py`, y en `esq-2d` el
  orden de los brazos **no se conservó** entre ventana y página.
- **Nada sobre `tr`/`bl`**: aquí son el negativo duro (`fp_otra_diagonal` los vigila).
- **Una semilla.** Las diferencias entre brazos vecinos no se van a poder declarar.
