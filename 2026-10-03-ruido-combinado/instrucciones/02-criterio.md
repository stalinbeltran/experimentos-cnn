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

## Extensión a 5 semillas — enmienda del 2026-10-03, escrita DESPUÉS de ver las 3 primeras y ANTES de correr las dos nuevas

Con 3 semillas el veredicto fue «indistinguible» (secuencial +0,007 ± 0,009 contra el mejor simple,
umbral 0,018). El propio criterio decía que lo decidirían 5 semillas. Se añaden las semillas **4 y
5** (pesos iniciales nuevos `init-s4.pt`, `init-s5.pt`, con `torch.manual_seed`; orden de lotes
104 y 105; las copias de ruido son las mismas de siempre, idénticas entre semillas) a **los cinco
escenarios**, para que la tabla sea uniforme: 10 corridas nuevas.

- **La regla no cambia**: Δ pareado contra el mejor simple, umbral `max(2·SE, 0,01)`, ahora con
  `SE = sd/√5`. El veredicto final es el de las 5; el de las 3 queda en el README como lo que fue.
- ⚠ **Es una extensión decidida tras mirar**: lo que protege es que la regla es la misma y que
  no se eligen las semillas (son las dos siguientes). Si con 5 la secuencial sigue sin llegar al
  umbral, se cierra como indistinguible y **no se amplía más**.
