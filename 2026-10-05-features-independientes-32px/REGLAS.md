# Reglas de `feat-ind32` — PLAN (2026-10-05, nada implementado ni corrido)

**Qué pregunta:** lo mismo que `feat-ind`, pero sin reducir la imagen. Cada CNN pequeña reconoce una sola
feature fijada de antemano. Las 13 se entrenan por separado y un compositor las junta sobre dígitos
manuscritos. Allí todo vivía en 8×8 (bloques 4×4 contados); aquí el detector ve el **bitmap binario de
32×32** del que salía ese 8×8.

**Por qué puede cambiar algo** (contexto, no criterio): a 8×8, `feat-ind` dejó arcos y esquinas en «no
aprendió» (F1 0,70–0,74, se disparan entre sí: corrida 2) y atribuyó un tercio del error sobre dígitos
al grosor (1↔8). Las dos cosas son de resolución: un arco de radio 4 px ocupa 1 celda a 8×8 y 8 px a
32×32.

## Entradas

### 1. El dato: comparable, y comprobado

- **Dígitos:** UCI *optdigits-orig*, el original sin reducir (bitmaps binarios 32×32, preprocesado NIST).
  Se descargó de `archive.ics.uci.edu/static/public/80/…zip` (HTTP 200, 591 KB) el 2026-10-05. Trae 4
  ficheros: `tra` 1934 · `cv` 946 · `wdep` 943 (30 escritores) · `windep` **1797** (13 escritores
  distintos).
- ⚠ **`windep` son LOS MISMOS 1797 dígitos que `uci-optdigits-8px-r20261002`.** Medido el 2026-10-05:
  al reducir cada bitmap de `windep` contando bloques 4×4, las **1797 de 1797** filas coinciden exactas
  (los 64 valores y la etiqueta) con `optdigits.tes`, en el mismo orden. Así que **el reparto de
  `feat-ind` (180/1617, y el 900+717 de la corrida 11) se puede reproducir índice por índice**, y los
  números de los dos experimentos se pueden comparar directamente. Falta comprobar que `optdigits.tes` sigue
  el mismo orden que el `load_digits` de scikit-learn con el que se publicó el 8px (se espera que sí,
  porque los dos son la copia del test de UCI). Es la primera comprobación del §5.
- **Extra que a 8×8 no se usó:** 3823 dígitos 32×32 de **otros 30 escritores** (`tra`+`cv`+`wdep`). Sirven
  para la curva de datos (corrida 11 llegó a 0,99 con 900) sin tocar el test. Se usan sólo en un brazo
  aparte (§4, C5), para que el brazo principal siga siendo comparable.
- **Features sintéticas:** `feat-ind` ya las dibuja a 32×32 y luego las reduce (`ESPECIFICACION.md` §1).
  Aquí **no se reduce**: se guarda la máscara binaria 32×32 con el mismo ruido `p_on`/`p_off`. Vocabulario,
  rangos, segunda feature, relación `CONTIENE` y la cantidad por familia (2000+400) son los mismos, **por
  decisión, no por herencia**: si no, no se sabría si un cambio en el resultado viene de la resolución
  o del vocabulario.
- **Publicación:** dos datasets nuevos en `foveal-vision-data/experimentos-cnn/` (que vive en el
  almacén), nunca copiados aquí: `feat-ind32-sinteticas-32px-r<fecha>` (32.400 × 1 KB ≈ 33 MB sin
  comprimir, ≈ 2–4 MB empaquetados en bits: *estimado*) y `uci-optdigits-orig-32px-r<fecha>` (5620 × 128 B
  empaquetados ≈ 0,7 MB). ⚠ Antes de publicar, `/use almacen` → `estado`, porque se llenó el 2026-10-03.

## Salidas

Pesos en `nn/pesos/<feature>/` (best/last, config, metrics, summary), igual de forma que en `feat-ind` para comparar; `resultados/*.json` por corrida; los `*.npz` no se commitean. El reporte va al repo central sólo si cambia `ESTADO.md`.

### 2. La salida del detector: ¿mapa 8×8 o 32×32? — DECIDIR

| opción | qué es | a favor | en contra |
|---|---|---|---|
| **A (propuesta)** | entrada 32×32, dos convoluciones con stride 2 → **mapa 8×8**; mismo objetivo gaussiano σ 0,6 celdas | el compositor es **idéntico** (832 entradas) y toda la cadena de `feat-ind` (compositor, curva, desplazamiento) se compara uno a uno: **sólo cambia lo que ve el detector** | la posición sigue a resolución de celda |
| B | mapa 32×32, objetivo σ 2,4 px | posición 4× más fina | compositor de 13 × 1024 = 13.312 entradas con 180 dígitos de train: se compara contra otra cosa |

Propongo **A** como brazo principal. B queda escrito y sin correr.

**Red de A** *(propuesta, se fija en un ensayo sobre `arco-E` y `recta-V`, como hizo la enmienda 3 de
`feat-ind`)*: conv 3×3 16 → conv 3×3 s2 32 → conv 3×3 s2 32 → conv 3×3 32 → 1×1 → 1 canal 8×8, con
padding. Pérdida: BCE por celda (objetivo ×8) + BCE sobre el máximo, umbral por detector elegido en
train, mejor F1 de val. Las dos enmiendas de `feat-ind` (`CONTIENE` y umbral por detector) entran desde
el principio, porque allí se midió que sin ellas el criterio mide otra cosa.

## Procesos

### 3. Criterio — se escribe en `instrucciones/02-criterio.md` ANTES de generar nada

Lo que propongo escribir allí (§A y §B con los mismos umbrales que `feat-ind`, porque la pregunta es
justo «¿cambia al subir la resolución?»):

- **Por detector:** «aprendió» si F1 ≥ 0,90 y posición ≤ 1 celda ≥ 0,90. Más las tres lecturas
  obligatorias (radio, FP por familia, grosor).
- **Hipótesis H1 (detectores):** arcos y esquinas pasan de «no aprendió» a ≥ «a medias». Se dice
  confirmada si al menos 6 de los 8 suben de categoría.
- **H2 (compositor, el reparto 180/1617 de la corrida 2):** posicional ≥ 0,959 + 0,01. Con 3 semillas y
  sd ~0,001 en 8×8, 0,01 es mucho más que el ruido.
- **H3 (techo, el reparto 900/717 de la corrida 11):** mejor que 0,992 — o, si no se puede, decir que el
  techo no era de resolución.
- **H4 (1↔8):** las confusiones 1↔8 bajan de 22 (corrida 2) a ≤ 5 sin que suba otro par (el 4→1 de la
  corrida 4).
- **Lo que espero hoy:** H1 sí en arcos, esquinas a medias; H2 marginal (+0,005 a +0,015), porque a 8×8 el
  compositor ya compensa. Si sale peor que 8×8, también es un resultado: a 32×32 el manuscrito es más
  irregular que el sintético y la transferencia puede empeorar.

### 4. Las corridas, en orden (cada una con su criterio antes)

| # | qué | reproduce de `feat-ind` | coste *(estimado)* |
|---|---|---|---|
| C1 | generar y publicar los 2 datasets; rejilla de muestras; comprobar el §1 | — | minutos |
| C2 | ensayo de red sobre `arco-E` y `recta-V` (fija canales y épocas) | enmienda 3 | ~30 min |
| C3 | los 13 detectores, 1 semilla | corrida 2 | ~16× el cómputo de 8×8 por imagen → **~20 min por detector, ~4–5 h los 13** en el dev, en serie |
| C4 | aplicar a los 1797 + compositores presencia/posicional (180/1617) + atribución del error | corrida 2 y atribución | minutos |
| C5 | curva por N de train (900/717) y además con los 3823 de otros escritores | corrida 11 | minutos |
| C6 | desplazamiento (máx 3×3) y generalización | corridas 12 y 13 | minutos |

**Lo que NO se reproduce:** corrida 3 (contra-casos, no se adoptó), 4/5 (trazos gruesos: a 32×32 el grosor es
otra cosa; se decide tras C3 por la lectura de grosor), 6–8 (grupos `dig`/`cae`: no aportaron, corrida 11),
9–10 (sobreajuste y compositor combinante: no movieron nada). Si el dueño quiere alguna, se añade.

**Dónde corre:** C3 es `entrena-local` en el dev, como unidad de systemd por un `nn/lanzar.sh` commiteado,
con modo seco, `--estado` y que se niegue a lanzar dos veces (las reglas del 2026-09-07/08 de
`telegram-coordinator/CLAUDE.md`). Los 13 son independientes: si 4–5 h es demasiado, se reparten en
**una máquina de Vast con muchas CPU** (≈0,1–0,3 $ *estimado*) — eso cambia `gasta` a `alquila` y es
decisión del dueño.

### 5. Antes de escribir código

1. Comprobar que `optdigits.tes` == `load_digits` (el orden), para que el reparto por índice valga.
2. Decidir A/B (§2) y local/Vast (§4).
3. Escribir `instrucciones/02-criterio.md` con el §3.
4. Pasar `gasta` a `entrena-local` y `entrada` a `nn/entrenar_local.py` en el mismo commit que el código
   (para que el freno lo vea desde el primer entrenamiento).

## Scripts

Todavía ninguno: se copian de `feat-ind/nn/` y se adaptan a 32×32 al implementar (`features.py`, `datos.py`, `modelo.py`, `entrenar_local.py`, `lanzar.sh`, `aplicar.py`, `compositor.py`, `curva.py`, `desplazamiento.py`, `generalizacion.py`, `errores.py`).

## Qué NO hereda

- **Se copia de `feat-ind`** el vocabulario, los generadores y la estructura de `nn/`, para no
  reescribirlos. Se copian, no se importan.
- **Se cambia a propósito:** la resolución de entrada, la red (strides), los datasets, el reparto extra con
  otros escritores.
- **Se conserva a propósito**, para poder comparar: vocabulario, cantidades, umbrales del criterio y
  repartos de dígitos.
- **`feat-pos` no se reproduce:** es sólo un documento. La parte que se puede medir aquí (cuánto aporta la
  posición) sale de C4 (presencia contra posicional), igual que en `feat-ind`.
