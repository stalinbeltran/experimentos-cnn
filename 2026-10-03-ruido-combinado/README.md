# `ruido-comb` — ¿suman los dos ruidos que más ayudaron, o el mejor solo ya lo da todo?

**Corrido el 2026-10-03 06:53 → 07:05 UTC en el dev (0 $), CERRADO.** 15 corridas: `limpio`, los
dos mejores simples de `ruido-nist` en línea (`gaussiano@0.2`, `recorte@0.6`), y sus dos
combinaciones —**secuencial** (recorte y, encima, gaussiano) y **mezcla** (cada imagen uno de los
dos al 50 %)—, × 3 semillas. Una unidad de systemd (`expc-ruidocomb`, `Result=success`,
`NRestarts=0`, 0 fallos). Mismo dato, red, `lr`, pasos, semillas y pesos iniciales que `ruido-nist`;
criterio escrito antes ([`instrucciones/02-criterio.md`](instrucciones/02-criterio.md)), aplicado
por `nn/informe.py` ([`resultados/RESULTADOS.md`](resultados/RESULTADOS.md)).

> **Veredicto: indistinguible, como se escribió antes.** La **secuencial** da la exactitud de val
> más alta vista en estos dos experimentos, **0,919** (+0,049 sobre `limpio`), pero contra el mejor
> simple (`gaussiano@0.2-linea`, 0,912) es **+0,007 ± 0,009**, por debajo del umbral 0,018: **no se
> puede declarar que sumen**. La **mezcla** (0,903) queda −0,010 ± 0,014 por debajo del simple,
> tampoco distinguible, y es la forma más lejos de sumar: repartir equivale a media dosis de cada
> uno. **Ninguna de las dos resta.**

| escenario | acc val (media ± sd) | CE val | Δ vs `limpio` ± SE | **Δ vs `gaussiano@0.2-linea`** ± SE | umbral | ¿suman? |
|---|---|---|---|---|---|---|
| `limpio` | 0,870 ± 0,032 | 1,037 | — | — | — | base |
| `gaussiano@0.2-linea` | 0,912 ± 0,021 | 0,359 | +0,042 ± 0,006 | referencia | — | — |
| `recorte@0.6-linea` | 0,899 ± 0,019 | 0,490 | +0,029 ± 0,007 | −0,013 ± 0,001 | 0,010 | (restan: es el peor de los dos simples) |
| **`recorte@0.6+gaussiano@0.2-linea`** (secuencial) | **0,919 ± 0,014** | **0,298** | **+0,049** ± 0,014 | **+0,007** ± 0,009 | 0,018 | **indistinguible** |
| `recorte@0.6~gaussiano@0.2-linea` (mezcla) | 0,903 ± 0,021 | 0,397 | +0,032 ± 0,019 | −0,010 ± 0,014 | 0,028 | indistinguible |

Exactitud de train = 1,000 en todos. Figura: [`resultados/delta.png`](resultados/delta.png); el
ruido, para mirarlo: [`resultados/muestras-ruido.png`](resultados/muestras-ruido.png).

## Lo que se lee

- **Los dos simples salieron bit a bit iguales que en `ruido-nist`** (0,9122 y 0,8994, mismas
  huellas de copia): el código copiado hace lo mismo, y la comparación es limpia.
- **La secuencial apunta a sumar y la CE de val lo acompaña** (0,298 contra 0,359 del simple, la
  más baja vista), pero las tres semillas se reparten (+0,023, +0,051, +0,072 sobre `limpio`) y con
  3 el SE contra el simple es del tamaño del efecto. Se dice «apunta», no «suma».
- **Contra `recorte@0.6-linea`, la secuencial sí supera el umbral** (+0,020 ± 0,009, umbral 0,018):
  añadir gaussiano al recorte ayuda; añadir recorte al gaussiano no se distingue.
- **Lo que decidiría**: 5 semillas (el SE bajaría a ~0,006 y el umbral a ~0,012) o 2 pasos más de
  intensidad. No se ha hecho.

## Lo que NO dice

Nada sobre otras parejas, tres ruidos, otras redes u otros datos. 13 escritores compartidos entre
train y val.

## Coste, reloj y dónde está

- **0 $**, 12 min de reloj (20–35 s por corrida; la primera tardó 177 s por coincidir con el
  checker del repo en las 2 CPU). Métricas, resúmenes, logs, informe y figuras aquí; **pesos** en el
  almacén, `foveal-vision-data/experimentos-cnn-resultados/ruido-comb/`.
- El aviso a Telegram del cierre no salió (`Falta BOT_TOKEN`: la unidad nació de una sesión de
  Claude Code). Reporte #29 en `estudios-redes-neuronales`.

## Cómo se repite

```bash
cd 2026-10-03-ruido-combinado
python nn/probar.py && python nn/entrenar_local.py --comprobar && sh nn/probar_lanzador.sh
nn/lanzar.sh todo        # 15 corridas; las que ya tienen summary.json se saltan
nn/lanzar.sh --estado
```
