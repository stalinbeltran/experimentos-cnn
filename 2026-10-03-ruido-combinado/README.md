# `ruido-comb` — ¿suman los dos ruidos que más ayudaron, o el mejor solo ya lo da todo?

**Corrido el 2026-10-03 06:53 → 07:05 UTC (3 semillas) y 07:33 → 07:44 (semillas 4 y 5) en el
dev (0 $), CERRADO.** 25 corridas: `limpio`, los
dos mejores simples de `ruido-nist` en línea (`gaussiano@0.2`, `recorte@0.6`), y sus dos
combinaciones —**secuencial** (recorte y, encima, gaussiano) y **mezcla** (cada imagen uno de los
dos al 50 %)—, × 5 semillas (las 3 del plan y 2 más, añadidas por la regla escrita en el criterio
antes de correrlas). Dos unidades de systemd (`expc-ruidocomb`, `Result=success`, `NRestarts=0`,
0 fallos). Mismo dato, red, `lr`, pasos, semillas y pesos iniciales que `ruido-nist`;
criterio escrito antes ([`instrucciones/02-criterio.md`](instrucciones/02-criterio.md)), aplicado
por `nn/informe.py` ([`resultados/RESULTADOS.md`](resultados/RESULTADOS.md)).

> **Veredicto (5 semillas): indistinguible, y se cierra.** La **secuencial** da la exactitud de val
> más alta y más estable vista en estos dos experimentos, **0,916 ± 0,012** (+0,047 sobre `limpio`),
> pero contra el mejor simple (`gaussiano@0.2-linea`, 0,902 ± 0,034) es **+0,014 ± 0,012**, por
> debajo del umbral 0,025: **no se puede declarar que sumen**. Con 3 semillas era +0,007 ± 0,009; al
> añadir dos, la secuencial subió su ventaja pero el gaussiano solo se volvió **más variable**
> (una semilla a 0,85), y el umbral creció con él. La **mezcla** (0,898) queda −0,004 bajo el
> simple: la forma más lejos de sumar. **Ninguna de las dos resta.** Lo que sí se distingue:
> **la secuencial supera al recorte solo** (+0,017 ± 0,005, umbral 0,011).

**Con 5 semillas (el veredicto):**

| escenario | acc val (media ± sd) | CE val | Δ vs `limpio` ± SE | **Δ vs `gaussiano@0.2-linea`** ± SE | umbral | ¿suman? |
|---|---|---|---|---|---|---|
| `limpio` | 0,869 ± 0,024 | 1,017 | — | — | — | base |
| `gaussiano@0.2-linea` | 0,902 ± 0,034 | 0,395 | +0,033 ± 0,011 | referencia | — | — |
| `recorte@0.6-linea` | 0,899 ± 0,015 | 0,508 | +0,030 ± 0,004 | −0,002 ± 0,012 | 0,023 | indistinguible |
| **`recorte@0.6+gaussiano@0.2-linea`** (secuencial) | **0,916 ± 0,012** | **0,297** | **+0,047** ± 0,008 | **+0,014** ± 0,012 | 0,025 | **indistinguible** |
| `recorte@0.6~gaussiano@0.2-linea` (mezcla) | 0,898 ± 0,021 | 0,403 | +0,029 ± 0,011 | −0,004 ± 0,011 | 0,021 | indistinguible |

**Con las 3 primeras (lo que se vio antes de ampliar):** `gaussiano` 0,912 ± 0,021, secuencial
0,919 ± 0,014 (+0,007 ± 0,009 contra el simple, umbral 0,018), mezcla 0,903 (−0,010 ± 0,014).

Exactitud de train = 1,000 en todos. Figura: [`resultados/delta.png`](resultados/delta.png); el
ruido, para mirarlo: [`resultados/muestras-ruido.png`](resultados/muestras-ruido.png).

## Lo que se lee

- **Los dos simples salieron bit a bit iguales que en `ruido-nist`** (0,9122 y 0,8994, mismas
  huellas de copia): el código copiado hace lo mismo, y la comparación es limpia.
- **La secuencial apunta a sumar y la CE de val lo acompaña** (0,297 contra 0,395 del simple, la
  más baja vista), y es el escenario **más estable** (sd 0,012 contra 0,034). Pero el gaussiano
  solo tiene una semilla floja (la 4, 0,85), y como el Δ es pareado, esa semilla da +0,04 para la
  secuencial y las otras cuatro entre −0,00 y +0,02: el SE crece y el umbral con él. Se dice
  «apunta», no «suma».
- **Contra `recorte@0.6-linea`, la secuencial sí supera el umbral** (+0,017 ± 0,005, umbral 0,011):
  añadir gaussiano al recorte ayuda; añadir recorte al gaussiano no se distingue.
- **Ampliar a 5 semillas no lo decidió**, y el criterio dice que no se amplía más. Lo que sí
  enseñó: el efecto del gaussiano solo es menos robusto entre semillas de lo que las 3 primeras
  sugerían (0,912 → 0,902), y el de la secuencial no (0,919 → 0,916).

## Lo que NO dice

Nada sobre otras parejas, tres ruidos, otras redes u otros datos. 13 escritores compartidos entre
train y val.

## Coste, reloj y dónde está

- **0 $**, 12 + 11 min de reloj (20–35 s por corrida; la primera tardó 177 s por coincidir con el
  checker del repo en las 2 CPU). Métricas, resúmenes, logs, informe y figuras aquí; **pesos** en el
  almacén, `foveal-vision-data/experimentos-cnn-resultados/ruido-comb/`.
- El aviso a Telegram del cierre no salió (`Falta BOT_TOKEN`: la unidad nació de una sesión de
  Claude Code). Reporte #29 en `estudios-redes-neuronales`.

## Cómo se repite

```bash
cd 2026-10-03-ruido-combinado
python nn/probar.py && python nn/entrenar_local.py --comprobar && sh nn/probar_lanzador.sh
nn/lanzar.sh todo        # 25 corridas (5 escenarios × 5 semillas); las que ya tienen summary.json se saltan
nn/lanzar.sh --estado
```
