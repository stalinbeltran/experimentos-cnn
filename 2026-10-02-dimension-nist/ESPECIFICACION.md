# `dim-nist` — el gemelo de `dim-gen` sobre los dígitos de NIST (8×8): reducir poco a poco y medir la generalización

**2026-10-02 · PREPARADO, NO LANZADO.** Lo escribió Claude a petición del dueño («Prepara otro
experimento idéntico a este con NIST como entrada…», literal en
[`instrucciones/01-encargo.md`](instrucciones/01-encargo.md)). Es historial; lo vigente manda en
[`REGLAS.md`](REGLAS.md) y [`instrucciones/02-criterio.md`](instrucciones/02-criterio.md).

## 0. Cuatro supuestos que el dueño tiene que confirmar antes de lanzar

| | supuesto elegido | la otra lectura |
|---|---|---|
| **S1** | **«NIST» = los dígitos de 8×8** de UCI (`load_digits` de scikit-learn, 1797 imágenes, 0..16, 10 clases): los bitmaps de 32×32 que extrajeron los programas de NIST, contados en bloques 4×4. «8×8 es el tamaño original» de ese dataset | **MNIST** (28×28). No preparado: cambiaría la escalera (28 → 14 → 7) |
| **S2** | **`W ∈ {8, 7, 6, 5, 4}`**: el original más cuatro reducciones por promedio de área con pesos exactos | «cuatro en total»: quitar `W = 5` |
| **S3** | **`L = 2`, `n = ⌈W/2⌉`, promedio global + `Linear(8 → 10)`**: la regla `f = 1/L` de `dim-gen` con la `L` que cabe en 8 px, y una cabeza constante aunque el mapa final sea 2×2 o 1×1 | sólo `{8, 6, 4}` con `n = W/2` y `Flatten`, idéntico a `dim-gen` pero 3 puntos |
| **S4** | **exactitud** sobre las 1617 no vistas, con el criterio de `dim-gen` y `δ = 0,01` | — |

## 1. Lo que ya está hecho (0 $, medido el 2026-10-02)

- **Dataset publicado**: `foveal-vision-data/experimentos-cnn/uci-optdigits-8px-r20261002/`
  (`datos.npz` + `manifiesto.json` + `README.md`), con el reparto 180 / 1617 dentro y las
  huellas de las cinco reducciones. 46 KB. Piso: 1/10 (azar; la clase mayoritaria de train no
  existe con 18 por clase, y la de val es 0,102). Son **13 escritores**: se mide generalizar a
  dígitos nuevos de los mismos escritores.
- **Reducciones exactas**: la rejilla de salida de `W` cae en múltiplos de `1/W`, así que el
  solape de cada píxel de salida con cada uno de entrada es un múltiplo de `1/8` y `S = A·x·Aᵀ`
  se calcula en enteros. Ejemplo, la fila 0 de `A`: `[8,0,…]` (W=8), `[7,1,…]` (7), `[6,2,…]`
  (6), `[5,3,…]` (5), `[4,4,…]` (4).
- **La red por `W`** (`C = 8`):

  | `W` | `n` | mapas | campo receptivo | parámetros |
  |---:|---:|---|---:|---:|
  | 8 | 4 | 5 → 2 | 7 | 1.258 |
  | 7 | 4 | 4 → 1 | 7 | 1.258 |
  | 6 | 3 | 4 → 2 | 5 | 754 |
  | 5 | 3 | 3 → 1 | 5 | 754 |
  | 4 | 2 | 3 → 2 | 3 | 394 |

  Entre extremos los parámetros van **×3,2** (en `dim-gen`, ×152): el confound de capacidad es
  mucho menor aquí, y el control `w8-de4` lo mide igual.
- **Por qué no `L = 4`** (corregido tras el `revisor`: `L = 4` sí cabe a `W = 8`, fue el `w008`
  de `dim-gen`): con `W ∈ {8…4}` no se pueden tener a la vez `f` constante y cabeza constante,
  que es lo que `dim-gen` tenía. `L = 2` deja el campo receptivo completo en los cinco `W` al
  precio de que `f` sea 0,50 en los pares y 0,57 / 0,60 en los impares (zigzag posible: está en
  el criterio). La alternativa exacta, `{8, 6, 4}` con `n = W/2`, es la decisión S3.
- **Código y pruebas**: `nn/` (ver `REGLAS.md` § Scripts); `nn/probar.py` y
  `nn/datos.py --comprobar` pasan; `nn/probar_lanzador.sh` pasa.
- **Ensayo de mecanismo** (§4): `lr` congelado.

## 2. Lo que se lanzará, y cuánto cuesta

30 brazos en el dev, en serie, como una unidad de systemd (`nn/lanzar.sh todo`). Cada brazo:
4000 pasos de lote 20 sobre 180 imágenes de ≤ 8×8 con redes de ≤ 1.258 parámetros — **segundos**
(medido en el ensayo, §4). Coste **0 $**. El freno lo ve (`entrenar_local.py`). Al terminar, la
unidad corre el informe, commitea el repo público, copia todo (pesos incluidos) al volumen y
avisa por Telegram.

## 3. Qué se espera, escrito antes (02-criterio.md)

Lo contrario de `dim-gen`: aquí el dato ya es muy reducido y las redes son diminutas, así que lo
más probable es que **más resolución ayude o se aplane pronto** (forma (i) o meseta corta), con
la caída fuerte en `W = 4` por falta de información. Si saliera (iii) —que 8 px estorbe—, sería
la noticia.

## 4. El ensayo de mecanismo (sólo pérdida de train, medido el 2026-10-02, 4000 pasos, semilla 1)

| `W` | `lr` | CE train inicio → final | exactitud train final | reloj |
|---:|---:|---|---:|---:|
| 8 | 1e-3 | 1,75 → 0,037 | 1,00 | 25 s |
| 8 | **3e-3** | 1,23 → 0,003 | 1,00 | 28 s |
| 6 | 1e-3 | 2,00 → 0,161 | 0,96 | 26 s |
| 4 | 1e-3 | 2,26 → 0,888 | 0,70 | 27 s |
| 4 | **3e-3** | 2,03 → 0,486 | 0,83 | 26 s |

Ninguno oscila (0 subidas >20 % entre épocas consecutivas). **`lr = 3e-3`**: con `1e-3`, `W = 4`
se queda a medio camino en 4000 pasos. **`C = 8` se queda**: la cabeza de 8 features llega a
exactitud 1,0 en train a `W = 8`. **Y ya se ve que la exactitud de train satura**, que es lo que
manda la descomposición a la entropía cruzada (criterio). Los 30 brazos: **≈13 min** en el dev.

## 5. Lo que dijo el `revisor` (2026-10-02) y qué se hizo con ello

RESERVAS, sin bloqueo: (1) «`L = 4` no cabe» era falso a `W = 8` → corregido el motivo (§1);
(2) la exactitud de train satura → descomposición en entropía cruzada; `δ` justificado con el
error binomial; piso 1/10; «atascada» definida; (3) NIST = `load_digits` se sostiene, pero 8×8 no
es el original y son 13 escritores → escrito en encargo, reglas y criterio; (4) las reducciones
fraccionarias se apartan de `dim-gen` → declarado en «Qué NO hereda», con prueba de
conservación de la media; (5) el nombre del dataset dice la fuente (`uci-optdigits-…`) y
`r<fecha>` es fecha de extracción.
