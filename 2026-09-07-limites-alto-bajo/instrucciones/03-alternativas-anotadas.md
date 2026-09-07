# Las alternativas: qué se armó, qué se anotó y qué se borró

## ✅ Armada: `max` — la que se corre

Un mapa, **una** salida, la **altura** leída del máximo. Cabeza de 3 parámetros.

## 📝 Anotada y NO armada: `sim` — el kernel simétrico bajo VOLTEO VERTICAL

`W = (V + flipud(V))/2`, o sea `W[a,b] = W[k−1−a, b]`. **k(k+1)/2 grados de libertad** en el
kernel (28 a `k`=7 contra 49 de `max`).

**Qué contestaría:** separa *«el máximo no basta»* de *«el gradiente no llegó a la simetría por su
cuenta»*. Con la equivarianza forzada, la red trata el borde alto y el bajo exactamente igual.

⚠ **Es `flipud`, no `rot180`** — y ahí está la diferencia con `esq-cq`. Allí el par era (`tl`,
`br`), giro de 180°; aquí es (alto, bajo), **reflejo vertical**. Hay un test que lo fija, y un
tercer test que comprueba que las dos simetrías **no** son la misma medida (un kernel simétrico
vertical sale al 68 % bajo rot180, no al 100 %). Sin ese tercero, alguien vuelve a poner `rot180`
y todo sigue «pasando».

**Qué cuesta:** un brazo más, ~5 min, 0 $. Una línea en `BRAZOS`.

## 📝 Anotada y NO armada: agrupar el mapa sobre `x` antes del softmax

Colapsar el mapa a un **perfil 1-D por filas** (máximo o media por columnas) y leer la altura de
ahí, en vez de tomar la marginal de un softmax 2-D.

**Por qué es la hipótesis natural siguiente:** el mapa de un límite es una **cresta**, no un pico.
Un perfil por filas es literalmente la forma del objeto que se busca, y probablemente esté mejor
condicionado que la marginal.

**Por qué no se arma ahora:** es **otra estructura**, no «idéntico». Se anota — que es lo que este
repo ya hizo con `sim` en `esq-cq`.

## 📝 Anotada y NO armada: dos salidas (alto del máximo, bajo del mínimo)

Lo que `esq-2d` hacía con las esquinas, aplicado a los límites. **Contestaría** si el kernel puede
separar los dos bordes en vez de tratarlos igual.

⚠ Y aquí el álgebra de `esq-2d` **sí** encajaría, al revés que allí: con dos salidas hay que pedir
respuestas de signo opuesto, o sea un kernel **antisimétrico vertical** — y el par alto/bajo sí es
reflejo vertical uno del otro, que es justo la premisa que en `esq-2d` estaba medida falsa.

## ❌ No aplica ya: `ant` (kernel antisimétrico bajo rot180)

Se borró en `esq-cq` porque con lectura por máximo es incapaz por construcción. Aquí además la
simetría relevante es otra. No vuelve.

## ❌ Descartada por el dueño (2026-09-07): `rot` — girar la entrada

*«No queremos girar el kernel».* Sigue descartada, y sigue sin hacer falta: la equivarianza que
compraba sale gratis forzando `W` simétrico.

## 🔓 Abierto: el eje ya está en su techo

`k` ∈ {5,7,9,11,13,15,17}, y **17 es el máximo que este dataset admite** (su mapa 16×16 cubre
[8,23], justo donde se sortean los offsets). Si `k17` vuelve a ganar sin saturar, subir más **exige
regenerar el dato con otra ventana** — no es un `BRAZOS` de distancia como en `esq-2d`.
