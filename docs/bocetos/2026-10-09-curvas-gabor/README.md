# Boceto — un detector de CURVAS con un banco de Gabor FIJO como entrada (2026-10-09)

**No es un experimento**: no entrena nada ni gasta nada (0 $, ~25 s en el dev). Es el **plan** que pidió el dueño
—*«empleando Gabor como entrada, prepara un detector de curvas; muéstrame lo planificado con imágenes aplicadas
sobre curvas, rectas y otras formas»*— montado entero y aplicado, para decidir si merece un experimento con criterio.
Es el pendiente 3 de la línea de rectas (`CLAUDE.md` del repo): *«una curva = cadena de detecciones de rectas cortas
cuya orientación gira»* (el dueño, 2026-10-08). Todo sale de [`curvas.py`](curvas.py); los números de este
documento están en [`resultados.json`](resultados.json) y [`resultados-banco.txt`](resultados-banco.txt).

## El plan, en cuatro pasos

```
imagen 32×32 ──► 1 BANCO     12 Gabor pares 9×9 (λ = 6 px, el de rect-lin) cada 15°   r_i = ReLU(G_i ⋆ x)
              ──► 2 CAMPO     orientación LOCAL por píxel (media vectorial en ángulo doble)  θ(p), coherencia c(p)
              ──► 3 GIRO      κ(p) = [θ(p + d·u) − θ(p − d·u)] / 2d     (u = tangente en p, d = 4 px)   → R ≈ 57,3 / |κ|
              ──► 4 VEREDICTO por componente conexa, con tres números: c · giro total Δθ · κ mediana
```

| veredicto | regla | qué significa |
|---|---|---|
| **mancha** | c mediana < 0,35 | responden todas las orientaciones por igual: no hay trazo con dirección |
| **corto** | menos de 4 px donde κ se puede medir | hay trazo, pero no da para saber si gira (≈ 2·d + 4 px útiles) |
| **recta** | Δθ < 12° | la orientación no cambia a lo largo |
| **esquina** | Δθ ≥ 12° pero κ mediana < 1 °/px, o el giro está en pocos píxeles (p90 ≥ 3 × mediana) | gira **en total**, no **en los píxeles**: el giro es de golpe. Un cruce cae aquí también |
| **curva** | lo demás | gira poco a poco; su radio es 57,3 / κ |

Y cada componente trae además sus **trozos**: cuántos tramos rectos (|κ| < 1 °/px) y cuántos curvos, que es la
salida que necesitaría una segunda etapa (pendiente 2, «segmentos»).

**Qué es «Gabor como entrada» aquí:** el banco es FIJO y es el mismo kernel que en `rect-lin`/`rect-bor` (par, λ = 6,
hecho para trazo de ~3 px), sólo que con 12 orientaciones en vez de 4, porque para medir un giro hace falta resolver la
orientación mejor que a 45°. La orientación continua sale de la **media vectorial** de las 12 respuestas (ángulo doble,
porque una línea de 170° y una de 10° son casi la misma), y la **coherencia** —cuánto manda una sola orientación— es lo
que separa trazo de mancha. Lo único que se ajustó a mano son los umbrales de la tabla, **sobre estas mismas 32 figuras,
no calibrados contra nada** (regla 2 de escritura: son elegidos, no medidos).

Tres decisiones que salieron de medir, no de pensar (las tres tienen su número en `curvas.py`):

1. **El campo se suaviza (σ = 1 px) antes de leer la orientación.** Sin eso una recta a 30° es una escalera y su
   orientación local oscila a lo largo: κ mediana **1,25 °/px en una recta** (medido con σ = 0), lo mismo que un arco
   de radio 45. Con σ = 1 y d = 4, las rectas dan 0,02–0,27 (0,75 a 60°).
2. **Sólo cuentan los píxeles del LOMO del trazo y a ≥ d px de sus extremos.** El kernel 9×9 enciende una franja
   alrededor de la línea, y en los últimos píxeles de un trazo la orientación se tuerce (recta de 30°: θ = 28–30° en
   el cuerpo, 16° y 44° en las puntas). Sin esta exclusión, el 69 % de las rectas de 10 px del banco salían «curva» y
   el radio de un arco de 27 se estimaba en 48.
3. **El signo de κ no se usa.** La tangente está definida módulo 180°, así que el signo se da la vuelta donde θ cruza
   0°/180°: distinguir una S de una C pide recorrer la cadena con un sentido, y eso es la segunda etapa, no ésta.

## Lo que se ve, figura a figura

### 1 · El banco y lo que entra al campo

![banco](imagenes/1-banco-gabor.png)

Cada trozo del arco enciende el kernel de **su** tangente: la fila de abajo (respuesta > τ) va girando de 0° a 165°
a lo largo del arco. Eso es la «cadena de rectas cortas cuya orientación gira», leída directamente del banco.

### 2 · Rectas y curvas sintéticas — 12 de 14 como se esperaba

![rectas y curvas](imagenes/2-rectas-y-curvas.png)

| forma | veredicto | R estimado (real) | |
|---|---|---:|---|
| rectas de 3 px a 0°, 30°, 110° | recta | — | Δθ 0,4–2,3°, κ 0,02–0,13 |
| recta de 8 px (gruesa) | recta | — | el Gabor de λ = 6 ve sus **dos bordes** como dos rectas paralelas (2 trozos rectos), como ya pasó en `rect-bor` |
| arcos de 3 px, R = 6 · 9 · 12 · 18 | curva | 6,2 (6) · 10,9 (9) · 12,1 (12) · 18,1 (18) | |
| arco R = 27 | curva | 38,5 (27) | sobreestima: con 20 px de largo sólo quedan 12 medibles |
| arco R = 40 | **recta** ✗ | — | Δθ 3,6°: a 32×32 un arco de R = 40 y 20 px gira 28° y **no se resuelve** |
| arco R = 9 de 8 px (grueso) | curva | 12,7 (9) | |
| el mismo a escala ½ | **corto** ✗ | — | a 16×16 quedan 6 px útiles: el remedio del grosor **no cabe** en un trazo de 20 px |
| círculo R = 8 | curva | 8,0 (8) | Δθ 168°: da la vuelta entera |

### 3 · Otras formas — las esquinas son el punto débil

![otras formas](imagenes/3-otras-formas.png)

| forma | veredicto | |
|---|---|---|
| **L de brazos 22 px** | esquina ✅ | Δθ 90° y κ mediana 0,12: gira en total y no en los píxeles. 2 trozos rectos |
| **L de brazos 14 px** | curva R ≈ 31 ✗ | el 9×9 + σ = 1 redondea la esquina en un arco de ~8 px, y con brazos cortos ese arco es la mitad de lo medible |
| **V de 60°** | curva R ≈ 24 ✗ | lo mismo |
| S de dos arcos R = 6 | curva R ≈ 8,6 ✅ | 2 trozos curvos; que sean de giro contrario **no se sabe** (decisión 3) |
| cruz + | corto ✗ | el centro tiene coherencia baja y se excluye; los 4 brazos de 10 px no dan para medir |
| punteada cada 4 px | nada ✗ | Emax 1,27, justo sobre τ: el Gabor de 3 px **no une puntos** (ya lo dijo `rect-lin`), ni a escala ½ |
| mancha r = 4 | mancha ✅ | c = 0,04 |
| puntos sueltos | mancha ✗ (esperaba «nada») | un disco de 2 px responde 1,52 > τ = 1,2 y se lee como mancha pequeña |
| ruido 1 % | nada ✅ | |

**La regla que sale de aquí:** una esquina se distingue de una curva cerrada **sólo si sus brazos son largos** (≥ ~20 px
con este kernel). Con brazos de 14 px, esquina y arco de R ≈ 8 son lo mismo para un 9×9 suavizado. Si el experimento
necesita esquinas, el detector de esquina tiene que ser **otro** (la caída de coherencia en el vértice, o los
detectores de `esq-k`), no este.

### 4 · Dígitos reales (uci-optdigits, 32×32), el mismo detector sin tocar

![dígitos](imagenes/4-digitos.png)

Un dígito es **una** componente, así que un veredicto por componente dice poco (0, 2, 3, 8 → curva; 4 → esquina;
1 → curva de R ≈ 47, por el pie; 5 y 7 → curva). Lo útil está en el **mapa de κ y en los trozos**: el 0 da 1 tramo
recto y 3 curvos, el 2 da 1 recto y 3 curvos, el 3 da 3 curvos. ⚠ Con trazos de 4–6 px la mayor parte del trazo
queda **gris (no medible)**: la coherencia cae dentro del trazo grueso. Para dígitos hace falta la segunda escala, o
el Gabor más ancho —el pendiente 4a de la línea—, y eso no está probado aquí.

### 5 · ¿El giro medido sigue a 1/R?

![radio](imagenes/5-radio-estimado.png)

Arcos de 20 px y 3 px en 8 orientaciones, R = 5…40: el radio estimado sigue al real de **5 a ~30 px** (error
logarítmico mediano 0,12, o sea ±13 %) y por encima de 30 sobreestima; las rectas quedan en κ ≤ 0,27 (0,75 a 60°),
por debajo del 1 °/px que separa recta de curva. **El rango útil de radios con este kernel es 5–30 px.**

### Sobre el banco de prueba entero de `rect-lin` (2.040 figuras, sin entrenar nada)

De [`resultados-banco.txt`](resultados-banco.txt); el banco es `rect-lin-banco-r20261008` (rectas cada 3° de
grosor 2–14 y largo 10/16/22; arcos R 6–40 de grosor 2/4/8; punteadas; 900 negativos).

| | recta | curva | esquina | corto | mancha | nada |
|---|---:|---:|---:|---:|---:|---:|
| rectas finas (2–4 px), n = 1080 | **69 %** | **0 %** | 0 % | 31 % | 0 % | · |
| rectas de 6–8 px, n = 720 | 46 % | 2 % | 2 % | 20 % | 30 % | · |
| rectas de 10–14 px, n = 1080 | · | 12 % | 33 % | 49 % | 6 % | · |
| arcos R 6–27 (todos los grosores), n = 360 | 1 % | **99 %** | 0 % | 0 % | · | · |
| arcos R 40, n = 72 | 21 % | 72 % | 7 % | · | · | · |
| punteadas sep 2–3, n = 240 | 96 % | · | · | 4 % | · | · |
| punteadas sep ≥ 4, n = 600 | 1 % | · | · | 11 % | 0 % | 88 % |
| manchas, n = 300 | · | · | · | · | 99 % | 1 % |
| puntos sueltos, n = 300 | · | · | · | 9 % | 18 % | 73 % |
| ruido, n = 300 | · | · | · | · | · | 100 % |

Lo que importa de esa tabla: **ninguna recta fina se lee como curva, y el 99 % de los arcos de R ≤ 27 se leen como
curva**, sin un solo parámetro aprendido. Los dos agujeros son los de siempre en esta línea: el 31 % de rectas finas
«cortas» son las de 10 px (con 16 y 22 px, 92 % y 98 % recta), y el trazo grueso (≥ 10 px) no es un trazo para un
Gabor de λ = 6: es un rectángulo con bordes y esquinas.

## Lo que NO está hecho, y lo que sería el experimento

- **Grosor:** el Gabor de λ = 6 ve dos bordes en un trazo de 8 px y nada coherente en uno de 12. Probar la segunda
  escala (aquí sólo en dos figuras, y en el arco se quedó sin píxeles) o un Gabor más ancho, igual para todo
  (regla del dueño del 2026-10-07: un solo pre-proceso para todos).
- **Punteadas:** no las une ni a escala ½. Es la opción 3 del pendiente 2 (votación tipo Hough), no un ajuste de aquí.
- **Esquinas con brazos cortos y cruces:** necesitan otro detector; éste sólo ve que «gira de golpe» si hay brazos.
- **El signo del giro (S contra C) y unir trozos en una cadena ordenada:** la segunda etapa, pendiente 2 «segmentos».
- **Umbrales elegidos a ojo** sobre 32 figuras; nada está calibrado contra negativos como en `rect-lin` (5 % de FP).

**Si se monta el experimento** (criterio antes de mirar, R13), la pregunta que este boceto permite escribir es:
*¿un compositor que reciba, además de los 4 mapas Gabor de `rect-bor` § Tanteo, el mapa de κ (dónde gira y cuánto),
lee los dígitos mejor que 0,955?* Con la misma receta del tanteo (180/1617, 3 semillas, a ciegas 3823) y el mismo
compositor lineal, cuesta 0 $ y ~1 min en el dev. Y la predicción honesta, escrita ya: **poco**, porque en los dígitos
la mitad del trazo sale «no medible» por el grosor (figura 4); el grosor va antes.

## Cómo se repite

```bash
cd ~/src/experimentos-cnn && .venv/bin/python docs/bocetos/2026-10-09-curvas-gabor/curvas.py --banco
```

Necesita el `.venv` del repo con numpy, scipy, matplotlib y torch (cpu), y los datasets publicados
`uci-optdigits-orig-32px-r20261005` y `rect-lin-banco-r20261008` en `foveal-vision-data/experimentos-cnn/`.

## Tanteo (2026-10-09, pedido del dueño): un clasificador de dígitos con las curvas, contra el de rectas, y los dos juntos

`python digitos.py` (35 s en el dev; [`resultados-digitos.json`](resultados-digitos.json)). Mismo protocolo que el
tanteo de `rect-bor`: compositor **lineal** (Adam, 300 épocas, lr 1e-2, L2 1e-3), 180 train / 1617 val, 3 semillas
que sólo cambian la inicialización, y a ciegas los 3823 dígitos de otros escritores. Todas las características son
**fijas**; sólo se entrena el compositor. **La referencia de rectas se re-ejecutó aquí**, no se copió del documento.

| compositor | características | val (1617) | a ciegas (3823) |
|---|---:|---:|---:|
| **curvas** (recto · curvo |κ| · golpe, 8×8) | 192 | 0,867 [0,865–0,868] | 0,836 |
| **rectas** (la C de `rect-bor`: 4 Gabor, integrado, 2 escalas) | 512 | 0,955 [0,951–0,957] | 0,947 |
| **combinado** | 704 | **0,958** [0,954–0,960] | **0,954** |

La predicción escrita antes se cumplió en las tres: rectas ≈ 0,955 (sale 0,9546: **reproduce** el 0,955 del tanteo
de `rect-bor` y su 0,947 a ciegas), curvas < 0,93, y el combinado ≤ 0,96. Lo que suma el combinado es **+0,003 en val
y +0,007 a ciegas**, dentro del rango entre semillas en val y un poco por encima a ciegas. ⚠ Tanteo: 3 semillas de
compositor, un solo reparto, nada declara.

**En qué falla cada uno** (semilla 0; `imagenes/6-fallos-*.png`, los 60 primeros de cada lista):

| | fallos en val | confusiones más frecuentes |
|---|---:|---|
| [curvas](imagenes/6-fallos-curvas.png) | 215 | 7→4 (15), 8→0 (11), 3→9 (11), 9→3 (10), 8→5 (10) |
| [rectas](imagenes/6-fallos-rectas.png) | 70 | 8→9 (10), 5→9 (6), 7→9 (3), 9→3 (3), 1→6 (3) |
| [combinado](imagenes/6-fallos-combinado.png) | 68 | 8→9 (11), 5→9 (4), 1→9 (4), 7→9 (3) |

Solape (semilla 0): de los 70 fallos de rectas, **49 los falla también curvas** y sólo 21 son propios de rectas;
curvas tiene 166 fallos propios. El combinado arregla 8 fallos de rectas y estropea 6: **cambia casi nada, en los dos
sentidos**. Por dígito, lo que mueve el combinado es el 3 (0,941 → 0,958) y el 5 (0,949 → 0,963); el **8 sigue siendo el
peor en los tres** (0,857 · 0,850), y su confusión es 8→9: un 8 con el lazo de abajo cerrado a medias.

Lo que se ve en las imágenes de fallos, y es lo mismo que decía la figura 4: **las curvas solas confunden por
orientación global** (7→4, 3→9, 9→3: formas con los mismos tramos curvos en sitios parecidos, porque el mapa de κ no
lleva la orientación del trazo, sólo cuánto gira), y en trazos gruesos la mitad del trazo es «no medible». Las rectas
fallan en **lazos**: 8→9, 5→9, 8→6, 8→1, donde lo que distingue es si un lazo está cerrado, y eso no lo dice ni un Gabor
ni un κ. **Añadir κ no cierra ese hueco**, porque su información está contenida casi entera en los 4 mapas de rectas
integrados (49 de 70 fallos compartidos). El hueco es de vocabulario —lazo cerrado contra abierto, el mismo que señaló
`feat-cortas`—, no de curvatura.
