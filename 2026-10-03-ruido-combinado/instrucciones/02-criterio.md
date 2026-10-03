# Criterio — escrito el 2026-10-03, ANTES de entrenar nada

## Qué se entrena

Cinco escenarios × 3 semillas = **15 corridas**, todo igual que `ruido-nist` (dato, red de 1.338
parámetros, `lr = 3e-3`, 3996 pasos de lote 20 sobre 360 imágenes, pesos iniciales `init-s<s>.pt`
copiados tal cual, orden de lotes `100 + s`, `last.pt`):

| escenario | la copia (en línea: nueva cada época) |
|---|---|
| `limpio` | las 180 duplicadas (la base) |
| `gaussiano@0.2-linea` | el mejor simple de `ruido-nist`, repetido aquí con su misma semilla (sale **bit a bit igual**: misma huella de copia) |
| `recorte@0.6-linea` | el segundo, ídem |
| `recorte@0.6+gaussiano@0.2-linea` | **secuencial**: recorte y, sobre esa imagen, gaussiano |
| `recorte@0.6~gaussiano@0.2-linea` | **mezcla**: cada imagen recibe UNO de los dos, al 50 % (sorteo nuevo por época) |

Los dos simples se **vuelven a correr** en vez de leerse de `ruido-nist`: cada experimento es
independiente y compara contra lo que él mismo midió. Que salgan iguales es una comprobación
gratis de que el código copiado hace lo mismo.

## Qué se mide

Por corrida, con `last.pt`: exactitud y CE sobre las 1617 de val limpias, sobre las 180 de train
limpias. **Δ pareado por semilla** contra `limpio` (el veredicto de siempre: ayuda / perjudica /
indistinguible con umbral `max(2·SE, 0,01)`) y **Δ pareado contra el mejor simple**
(`gaussiano@0.2-linea`).

## Qué se declara

- **Suman**: la combinación supera al mejor simple en más de `max(2·SE, 0,01)`.
- **Restan**: queda por debajo del mejor simple en más del umbral.
- **Indistinguible**: lo demás; se dice cuál de las dos formas (secuencial / mezcla) queda más
  cerca de sumar, sin declararla.
- Se reporta también contra `recorte@0.6-linea`, y la CE de val.

## Lo que se espera, escrito antes

Lo más plausible es **indistinguible**: en `ruido-nist` el efecto de `gaussiano@0.2` en línea ya
era de 4 puntos y el del recorte de 3, y con 3 semillas el umbral entre dos escenarios de ~0,91
ronda 0,01–0,02. Si suman, la **secuencial** (más perturbación por imagen) es la candidata; la
mezcla reparte y equivale a media dosis de cada uno. Si la secuencial **resta**, es sobredosis,
como `gaussiano@0.3` en `ruido-nist`, y es un resultado.

## Lo que NO decide

Nada sobre otras parejas, ni sobre tres ruidos, ni sobre otras redes o datos. 13 escritores
compartidos entre train y val.
