# `ruido-nist` — qué ruido en el entrenamiento ayuda a generalizar (dígitos NIST 8×8)

**2026-10-02 · PREPARADO y NO lanzado.** Hay código con sus pruebas (`nn/`), pesos iniciales
compartidos commiteados (`nn/init/`), `lr` congelado en un ensayo de sólo train y la rejilla para
MIRAR el ruido (`resultados/muestras-ruido.png`). **Ninguna corrida del estudio ha entrenado**:
lo que se ha corrido son el ensayo (sólo pérdida de train) y pruebas de 2 épocas en directorios
temporales. Lo vigente está en `REGLAS.md`; el criterio, escrito antes de mirar, en
`instrucciones/02-criterio.md`. Se lanza con `nn/lanzar.sh fase1` cuando el dueño lo diga.

## 0. Los cinco supuestos — CONFIRMADOS el 2026-10-02 (S2 cambiado a copia fija; el resto, como se propuso)

| | supuesto elegido | por qué / la otra lectura |
|---|---|---|
| **S1** ✅ | **3 capas, no 4.** Con kernel 3×3 sin padding cada capa quita 2 px: 8 → 6 → 4 → 2 → **0**. Cuatro capas no caben en 8 px; tres dejan un mapa de 2×2 | otra salida: 4 capas con kernel 2×2 (8→7→6→5→4), pero el encargo fija 3×3 |
| **S2** | ✅ **Decidido por el dueño (2026-10-02): una COPIA ruidosa fija.** Train = las 180 originales **+ 180 copias ruidosas** (una realización por imagen, generada una vez y guardada con su huella) = 360 imágenes. Se AÑADE en vez de reemplazar para que todos los escenarios vean los mismos 180 originales y lo único distinto sea la copia. ⏳ **Pendiente para después**: ruido en línea, una realización nueva por época, para más variedad | — |
| **S3** ✅ | **El ruido se dibuja a 32×32 y se reduce a 8×8 contando bloques 4×4**, igual que NIST hizo el dataset. Así una recta es fina (1 px a 32 → ¼ de tinta a 8) y no un bloque que tapa el dígito. ⚠ **Al implementar (§2): el dígito NO existe a 32×32** en este dato, así que lo que se dibuja a 32 es la **máscara** del trazo, se reduce a **cobertura** por bloque y se compone a 8×8 con S4. Es lo único que se puede hacer y da exactamente lo que S3 quería: trazos finos | dibujar directamente a 8×8: cada línea ocupa 1/8 de la imagen; demasiado gruesa |
| **S4** ✅ | **Transparencia α** = mezcla con el píxel: `x' = (1−α)·x + α·v`, con `v` = tinta (1) para lo que añade y `v` = papel (0) para lo que borra. α ∈ {1,0; 0,8; 0,6; 0,4; 0,2} | — |
| **S5** ✅ | **«specular»** se lee como **ruido gaussiano** sobre cada píxel (σ en vez de α) y **sal y pimienta**; «recortes» como **cutout** (un cuadrado borrado) | si «specular» era otra cosa (brillo, reflejos), se añade como tipo |

## 1. Lo que es fijo en todos los escenarios

- **Dato**: `uci-optdigits-8px-r20261002`, ya publicado (lo creó `dim-nist`): 180 train (18 por
  dígito) / 1617 val, cuentas 0..16 por píxel → fracción de tinta `x = cuenta/16 ∈ [0,1]`.
  **Validación siempre limpia**: su huella (`57a6cb6086e0f956`) está congelada en `nn/datos.py` y se
  comprueba en cada carga y en cada `summary.json`.
- **Red**: 3 × `Conv2d(3×3, C=8)` + ReLU, sin padding (8→6→4→2), promedio global, `Linear(8→10)`.
  **1.338 parámetros** (80 + 584 + 584 + 90, calculado el 2026-10-02 y fijado en `nn/probar.py`;
  el plan decía «unos 1.250»), del orden del `w8` de `dim-nist` (1.258).
- **Protocolo**: **3996 pasos de lote 20** (con 360 imágenes de train, 18 pasos por época → **222
  épocas**; los mismos pasos que `dim-nist`), Adam con **`lr = 3e-3`** congelado en el ensayo de
  sólo train (§3), entropía cruzada, sin parada temprana, sin selección (`last.pt`). Evaluación
  cada 10 épocas sobre las 1617 de val limpias y las 180 de train limpias.
- **Lo que pide el dueño, «pesos idénticos en todos los escenarios»**: por semilla `s ∈ {1,2,3}`
  hay **tres generadores separados** (§1 bis): (1) la inicialización, que es un **fichero**
  `nn/init/init-s<s>.pt` commiteado con su huella en `nn/init/huellas.json` y cargado por todos los
  escenarios (entrenar se **niega** si falta o si la huella no casa); (2) el orden de los lotes,
  idéntico en todos; (3) el ruido, con su propio generador, que **no toca** a los otros dos.
  **Comprobado** (`nn/entrenar_local.py --comprobar`, 2026-10-02): los nueve tipos con nivel 0 dan
  la copia idéntica a la original y, tras 2 épocas, **los mismos pesos que `limpio` bit a bit**;
  `horizontal@0.6` da otros; y dos procesos distintos de `limpio-s1` dan la misma huella final.

## 1 bis. Las semillas (decididas el 2026-10-02)

| generador | semillas | qué fija | compartido entre escenarios |
|---|---|---|---|
| **pesos iniciales** | **1, 2, 3** | `torch.manual_seed(s)` → `nn/init/init-s<s>.pt`, con huella | **sí**: todos los escenarios de la semilla `s` cargan el mismo fichero |
| **orden de los lotes** | **`100 + s`** (101, 102, 103) | `numpy.default_rng(100+s)`, una permutación de los 360 índices por época | **sí**: mismo orden en todos los escenarios (las copias ocupan los índices 180–359) |
| **ruido (la copia)** | **`1000 + 10·t + i`** con `t` = número de tipo (0–8, el orden de la tabla del §2) e `i` = nivel de intensidad (0–4) | genera la copia ruidosa de las 180 una sola vez; su huella va en cada `summary.json` y `nn/informe.py` comprueba que las tres semillas la comparten | **sí entre semillas de pesos**: las tres semillas ven LA MISMA copia de un escenario, así el Δ pareado mide la red y no una realización distinta |

Por qué tres generadores distintos: si compartieran uno, sacar un número para el ruido movería el
orden de los lotes y la inicialización, y dos escenarios dejarían de partir de lo mismo — justo lo
que el dueño pidió eliminar. Por qué el ruido es igual en las tres semillas: con una copia fija,
cambiar la copia entre semillas mezclaría «otra red» con «otro ruido». ⚠ El precio: el resultado
de un escenario es el de **una** realización de su ruido. Para medir cuánto pesa eso, la fase 1
repite **`oblicua@0.6`** con una segunda copia (`oblicua@0.6-r2`, semilla de ruido `+5000`); se
eligió `oblicua` porque es el tipo con más parámetros aleatorios por imagen (posición, ángulo,
número y grosor de los trazos), o sea el que más puede variar entre dos copias. Si `|Δ|` entre las
dos copias supera el umbral del criterio, se dice y se pasa al ruido en línea (lo pendiente de S2).

## 2. Los tipos de ruido

**Cómo se compone un trazo** (S3 + S4, lo que `nn/ruido.py` hace): la máscara `M` del trazo se
rasteriza a 32×32, se reduce a 8×8 contando bits por bloque 4×4 (**cobertura** `c = bits/16`) y

    x' = x + α·c·(v − x)        v = 1 para lo que añade tinta, v = 0 para lo que borra

Para `borrado` y `externos` se simula el nivel de bit **sin posiciones**: apagar cada bit de tinta
con probabilidad `p` quita `Binomial(cuenta, p)` bits; encender cada bit de fondo con probabilidad
`p` añade `Binomial(16 − cuenta, p)`. Contar bits no depende de dónde estén, así que es exactamente
«dibujado a 32 y contado». Un nivel 0 deja la imagen intacta en los nueve tipos (tiene test).

| `t` | tipo | qué hace | niveles `i = 0..4` (de menos a más) · **medio** |
|---:|---|---|---|
| 0 | `borrado` | cada bit de tinta se apaga con prob. `p` | `p` ∈ {0,05; 0,1; **0,2**; 0,3; 0,4} |
| 1 | `externos` | cada bit de fondo se enciende con prob. `p` (motas) | `p` ∈ {0,01; 0,02; **0,05**; 0,1; 0,2} |
| 2 | `horizontal` | 1–2 rectas horizontales de 1–2 px (a 32) en posición aleatoria, v = 1 | α ∈ {0,2; 0,4; **0,6**; 0,8; 1,0} |
| 3 | `vertical` | ídem verticales | α |
| 4 | `oblicua` | ídem con ángulo uniforme en (15°, 75°) ∪ (105°, 165°), por un punto aleatorio | α |
| 5 | `curva` | 1–2 arcos de Bézier cuadráticos con 3 puntos aleatorios, 1–2 px | α |
| 6 | `recorte` | borra (v = 0) un cuadrado de 8–16 px de lado (a 32: ¼–½ del lado) | α |
| 7 | `gaussiano` | `x + N(0, σ)` recortado a [0,1], a 8×8 | σ ∈ {0,02; 0,05; **0,1**; 0,2; 0,3} |
| 8 | `sal-pimienta` | pone a 0 ó a 1 (al 50 %) `round(p·64)` píxeles, a 8×8 | `p` ∈ {0,02; 0,05; **0,1**; 0,2; 0,3} |
| — | **`limpio`** | **sin ruido: la línea base** (las 180 duplicadas) | — |

⚠ **Las listas de `p` y σ tienen cinco niveles desde el 2026-10-02** (el plan traía tres, y dos para
`sal-pimienta`): la semilla de ruido `1000 + 10·t + i` que el dueño confirmó indexa `i ∈ 0..4`, y la
fase 2 recorre «α o su escala de `p`/σ» con cinco puntos. Se completaron **por los extremos sin
mover el medio** (0,2 · 0,05 · 0,1), salvo `sal-pimienta`, cuyo «medio» de {0,05; 0,1} no existía:
se fija **0,1**. Todo está en `nn/ruido.py` (`NIVELES`) y lo imprime `python nn/ruido.py`.

Lo que se ve en la rejilla (`resultados/muestras-ruido.png`, 10 ejemplos por tipo al nivel medio):
las rectas y curvas a α = 0,6 salen **tenues** (una recta de 1 px cubre ¼ del bloque → 0,15 de
tinta); es la consecuencia de S3 y se compensa en la fase 2 con α = 1 y grosor 2 (cobertura ½).

## 3. Las fases

**Ensayo de mecanismo (hecho el 2026-10-02, sólo pérdida de train, semilla 1, `limpio`, 3996
pasos):** con `lr = 3e-3` la pérdida sobre las 360 baja de 0,934 a **0,0005** con 10 subidas > 20 %
en 221 épocas (exactitud de train 1,000; CE de las 180 limpias 0,0004); con `1e-3`, de 1,461 a
0,0276 (3 subidas; 0,994). Las dos sirven; **se congela `3e-3`**, la que llega a la meseta. **18–20 s
por ensayo** medidos en el dev (2 hilos), sin evaluaciones; una corrida del estudio, con sus 22
evaluaciones, se estima en **~25 s**.

**Fase 1 — tipos** (la que pide el dueño primero): los 9 tipos a **su nivel medio** + `limpio` +
`oblicua@0.6-r2`, × 3 semillas = **33 corridas**, **~14 min estimados**, 0 $. `nn/lanzar.sh fase1`.

**Fase 2 — intensidades**, sólo para los tipos que la fase 1 deje como «ayuda» o «indistinguible
con media > 0» (`resultados/criterio-aplicado.json` → `fase2`): los 5 niveles × 3 semillas (el
medio ya está hecho y se salta). `nn/lanzar.sh fase2 <tipo> [tipo…]`; con 3 tipos, 36 corridas
nuevas, ~15 min.

**Fase 3 (opcional, si la fase 1 no separa)**: subir a 5 semillas los tipos dudosos (pide crear
`init-s4.pt` e `init-s5.pt` con `--inicializar`, y ampliar `SEMILLAS` en `nn/modelo.py`).

## 4. Qué sale y dónde

`nn/pesos/<escenario>-s<s>/` (`metrics.jsonl` por época, `summary.json`, `log.txt`, `last.pt`),
`resultados/` (`RESULTADOS.md` y `criterio-aplicado.json` regenerados por `nn/informe.py`,
`delta-por-tipo.png` con el Δ pareado por escenario y su 2·SE, `muestras-ruido.png`). Pesos y
todo lo demás al almacén (`foveal-vision-data/experimentos-cnn-resultados/ruido-nist/`), lo hace
el cierre de `nn/lanzar.sh`, como `dim-nist`.

## 5. Riesgos

- **3 semillas dan poco poder**: el criterio compara **pareado por semilla** (mismos pesos
  iniciales), lo que quita la variación de la inicialización; aun así, un efecto por debajo de
  ~1 punto probablemente no se podrá declarar. Se dice, no se fuerza.
- **La realización del ruido es una variable** con la copia fija: se mide con `oblicua@0.6-r2`
  (§1 bis).
- **`limpio` ve 180 imágenes y los demás 360**: para que «más imágenes» no se confunda con «ruido»,
  la base añade también 180 copias, **sin ruido** (las mismas 180 duplicadas). Así todos ven 360.
- **La exactitud de train satura en 1,0** (ensayo): el mecanismo (regularización contra
  facilitación) se lee en la **entropía cruzada** de las 180 de train limpias, que no satura.
  Está en el criterio como enmienda fechada, antes de entrenar nada.
- **Los trazos finos a α medio son tenues** (§2): si en la fase 1 todos los tipos de trazo salen
  «indistinguibles», la lectura es «a esta intensidad» y la fase 2 lo contesta con α = 1.
- **13 escritores** compartidos entre train y val (ver `dim-nist`).

## 6. Lo que la revisión del 2026-10-02 cambió del plan, y por qué

Al escribir el código se encontraron siete cosas que el plan dejaba abiertas o decía mal. Ninguna
cambia una decisión del dueño; todas están aplicadas y con test donde cabe:

1. **El dígito no existe a 32×32** (el dato son cuentas por bloque): «dibujar a 32 y reducir» se
   aplica a la **máscara** del trazo (cobertura) y se compone a 8 con la fórmula de S4; `borrado` y
   `externos` van a nivel de bit por Binomial. §2.
2. **Las intensidades tenían 3 ó 2 niveles** y la semilla de ruido confirmada indexa 5: ahora son
   cinco en todos los tipos, sin mover el medio. §2.
3. **`sal-pimienta` no tenía nivel medio** ({0,05; 0,1}): se fija 0,1.
4. **1.250 parámetros → 1.338** (calculados).
5. **El tipo que se repite con otra copia no estaba nombrado**: `oblicua`, con su motivo. §1 bis.
6. **Los pesos iniciales son un fichero commiteado con huella**, no una llamada a `manual_seed`
   en cada corrida: así «idénticos» no depende de que torch siembre igual dentro de un año.
7. **El mecanismo se lee en CE de train**, porque la exactitud de train llega a 1,0 (ensayo).
