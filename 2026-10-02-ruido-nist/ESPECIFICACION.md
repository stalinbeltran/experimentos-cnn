# `ruido-nist` — qué ruido en el entrenamiento ayuda a generalizar (dígitos NIST 8×8)

**2026-10-02 · SÓLO PLAN. Nada escrito en código, nada entrenado.** Lo vigente, cuando haya
código, irá en `REGLAS.md`; el criterio, escrito antes de mirar, en `instrucciones/02-criterio.md`.

## 0. Los cinco supuestos — CONFIRMADOS el 2026-10-02 (S2 cambiado a copia fija; el resto, como se propuso)

| | supuesto elegido | por qué / la otra lectura |
|---|---|---|
| **S1** ✅ | **3 capas, no 4.** Con kernel 3×3 sin padding cada capa quita 2 px: 8 → 6 → 4 → 2 → **0**. Cuatro capas no caben en 8 px; tres dejan un mapa de 2×2 | otra salida: 4 capas con kernel 2×2 (8→7→6→5→4), pero el encargo fija 3×3 |
| **S2** | ✅ **Decidido por el dueño (2026-10-02): una COPIA ruidosa fija.** Train = las 180 originales **+ 180 copias ruidosas** (una realización por imagen, generada una vez y guardada con su huella) = 360 imágenes. Se AÑADE en vez de reemplazar para que todos los escenarios vean los mismos 180 originales y lo único distinto sea la copia. ⏳ **Pendiente para después**: ruido en línea, una realización nueva por época, para más variedad | — |
| **S3** ✅ | **El ruido se dibuja a 32×32 y se reduce a 8×8 contando bloques 4×4**, igual que NIST hizo el dataset. Así una recta es fina (1 px a 32 → ¼ de tinta a 8) y no un bloque que tapa el dígito | dibujar directamente a 8×8: cada línea ocupa 1/8 de la imagen; demasiado gruesa |
| **S4** ✅ | **Transparencia α** = mezcla con el píxel: `x' = (1−α)·x + α·v`, con `v` = tinta (1) para lo que añade y `v` = papel (0) para lo que borra. α ∈ {1,0; 0,8; 0,6; 0,4; 0,2} | — |
| **S5** ✅ | **«specular»** se lee como **ruido gaussiano** sobre cada píxel (σ en vez de α) y **sal y pimienta**; «recortes» como **cutout** (un cuadrado borrado) | si «specular» era otra cosa (brillo, reflejos), se añade como tipo |

## 1. Lo que es fijo en todos los escenarios

- **Dato**: `uci-optdigits-8px-r20261002`, ya publicado (lo creó `dim-nist`): 180 train (18 por
  dígito) / 1617 val, fracción de tinta `x/16 ∈ [0,1]`. **Validación siempre limpia.**
- **Red**: 3 × `Conv2d(3×3, C=8)` + ReLU, sin padding (8→6→4→2), promedio global, `Linear(8→10)`.
  Unos 1.250 parámetros, del orden del `w8` de `dim-nist`.
- **Protocolo**: 3996 pasos de lote 20 (con 360 imágenes de train, 18 pasos por época → 222 épocas; los mismos pasos que `dim-nist`), Adam con un `lr` congelado en un ensayo de
  sólo train, entropía cruzada, sin parada temprana, sin selección (`last.pt`).
- **Lo que pide el dueño, «pesos idénticos en todos los escenarios»**: por semilla `s ∈ {1,2,3}`
  se fijan **tres generadores separados**: (1) la inicialización —guardada como `init-s<s>.pt` y
  cargada por todos los escenarios, con huella—; (2) el orden de los lotes, idéntico en todos;
  (3) el ruido, con su propio generador, que **no toca** a los otros dos. Así lo único que cambia
  entre escenarios es el ruido. Se comprueba con un test: dos escenarios con «ruido nulo» dan
  pesos finales **bit a bit iguales**.

## 1 bis. Las semillas (decididas el 2026-10-02)

| generador | semillas | qué fija | compartido entre escenarios |
|---|---|---|---|
| **pesos iniciales** | **1, 2, 3** | `torch.manual_seed(s)` → `nn/init/init-s<s>.pt`, con huella | **sí**: todos los escenarios de la semilla `s` cargan el mismo fichero |
| **orden de los lotes** | **`100 + s`** (101, 102, 103) | `numpy.default_rng(100+s)`, una permutación de los 360 índices por época | **sí**: mismo orden en todos los escenarios (las copias ocupan los índices 180–359) |
| **ruido (la copia)** | **`1000 + 10·t + i`** con `t` = número de tipo (0–8) e `i` = nivel de intensidad (0–4) | genera la copia ruidosa de las 180 una sola vez; se guarda con su huella | **sí entre semillas de pesos**: las tres semillas ven LA MISMA copia de un escenario, así el Δ pareado mide la red y no una realización distinta |

Por qué tres generadores distintos: si compartieran uno, sacar un número para el ruido movería el
orden de los lotes y la inicialización, y dos escenarios dejarían de partir de lo mismo — justo lo
que el dueño pidió eliminar. Por qué el ruido es igual en las tres semillas: con una copia fija,
cambiar la copia entre semillas mezclaría «otra red» con «otro ruido». ⚠ El precio: el resultado
de un escenario es el de **una** realización de su ruido. Para medir cuánto pesa eso, la fase 1
repite **un** tipo con una segunda copia (semilla de ruido `+5000`); si la diferencia entre las dos
copias se acerca a los Δ entre tipos, se dice y se pasa al ruido en línea (lo pendiente de S2).

## 2. Los tipos de ruido

| tipo | qué hace (dibujado a 32×32) | parámetros |
|---|---|---|
| `borrado` | apaga una fracción `p` de los píxeles de tinta | `p` ∈ {0,1; 0,2; 0,4} |
| `externos` | enciende una fracción `p` de píxeles de fondo (motas sueltas) | `p` ∈ {0,02; 0,05; 0,1} |
| `horizontal` | 1–2 rectas horizontales de 1–2 px de grosor en posición aleatoria | α |
| `vertical` | ídem verticales | α |
| `oblicua` | ídem con ángulo uniforme en (15°, 75°) ∪ (105°, 165°) | α |
| `curva` | 1–2 arcos de Bézier cuadráticos con puntos aleatorios | α |
| `recorte` | borra un cuadrado de 8–16 px (a 32: ¼–½ del lado) | α |
| `gaussiano` | `x + N(0, σ)` recortado a [0,1], a 8×8 | σ ∈ {0,05; 0,1; 0,2} |
| `sal-pimienta` | pone a 0 o a 1 una fracción `p` de píxeles a 8×8 | `p` ∈ {0,05; 0,1} |
| **`limpio`** | **sin ruido: la línea base** | — |

## 3. Las fases

**Fase 1 — tipos** (la que pide el dueño primero): los 9 tipos a **una intensidad media**
(α = 0,6; `p` y σ del medio de su lista) + `limpio` + 1 repetición de copia, × 3 semillas = **33 corridas**. Medido en
`dim-nist`: ~27 s por corrida en el dev → **~15 min**, 0 $.

**Fase 2 — intensidades**, sólo para los tipos que la fase 1 deje como «ayuda» o «indistinguible»
(no para los que perjudican): α ∈ {1,0; 0,8; 0,6; 0,4; 0,2} (o su escala de `p`/σ), × 3 semillas.
Con 3 tipos son 45 corridas, ~20 min.

**Fase 3 (opcional, si la fase 1 no separa)**: subir a 5 semillas los tipos dudosos.

## 4. Qué sale y dónde

`nn/pesos/<escenario>-s<s>/` (métricas por época, `summary.json`, `last.pt`), `resultados/`
(informe regenerado, figura de Δ por tipo con su barra de error, una rejilla PNG con 10 ejemplos
de cada ruido para **mirar** que el ruido es lo que dice ser). Pesos al almacén, como `dim-nist`.

## 5. Riesgos

- **3 semillas dan poco poder**: el criterio compara **pareado por semilla** (mismos pesos
  iniciales), lo que quita la variación de la inicialización; aun así, un efecto por debajo de
  ~1 punto probablemente no se podrá declarar. Se dice, no se fuerza.
- **La realización del ruido es una variable** con la copia fija: se mide con un tipo repetido con
  una segunda copia (§1 bis).
- **`limpio` ve 180 imágenes y los demás 360**: para que «más imágenes» no se confunda con «ruido»,
  la base añade también 180 copias, **sin ruido** (las mismas 180 duplicadas). Así todos ven 360.
- **13 escritores** compartidos entre train y val (ver `dim-nist`).
