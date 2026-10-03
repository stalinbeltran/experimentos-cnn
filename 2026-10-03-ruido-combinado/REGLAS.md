# Reglas de `ruido-comb`

**Escritas el 2026-10-03. Estado `abierto`, PREPARADO y NO lanzado.** Se copió de `ruido-nist`
(carpeta `nn/` entera, incluidos los pesos iniciales) y se releyó línea por línea: lo que cambia
está en § «Qué NO hereda».

**Qué pregunta:** ¿aplicar juntos los dos ruidos que más ayudaron en `ruido-nist` —gaussiano
σ = 0,2 y recorte α = 0,6, en línea— sube la exactitud de val por encima del mejor de los dos
solos, y cuánto?

## Entradas

- **Dataset**: `uci-optdigits-8px-r20261002`, leído con `expcnn.exigir_dataset`; 180 / 1617; val
  nunca se toca (huella congelada, igual que en `ruido-nist`).
- **Train de cada escenario** = 180 originales + 180 copias **en línea** (nueva cada época, del
  mismo generador; la época 1 es la copia fija). Escenarios: `limpio`, `gaussiano@0.2-linea`,
  `recorte@0.6-linea`, `recorte@0.6+gaussiano@0.2-linea` (secuencial), `recorte@0.6~gaussiano@0.2-linea`
  (mezcla al 50 % por imagen). Semillas: pesos 1/2/3 (`nn/init/`, copiados), lotes 100+s, ruido
  de los simples la de `ruido-nist` (1000+10·t+i), de las combinaciones **7000 + 10·t_A + t_B**
  (+50 la mezcla).
- **Combinación secuencial**: la segunda parte se aplica sobre la salida en float de la primera
  (`ruido._aplicar_sobre_float`): gaussiano suma ruido sobre la imagen ya recortada. Mezcla: por
  imagen, un sorteo al 50 % elige cuál de las dos copias se usa.

## Salidas

- `nn/pesos/<escenario>-s<s>/`: `last.pt` (no se commitea; al almacén), `metrics.jsonl`,
  `summary.json` (con `huella_copia` y `semilla_ruido`), `log.txt`.
- `resultados/RESULTADOS.md` + `criterio-aplicado.json` + `delta.png`, por `nn/informe.py`;
  `resultados/muestras-ruido.png` con las dos combinaciones.

## Procesos

1. `nn/probar.py` y `nn/entrenar_local.py --comprobar` (las de `ruido-nist` más las de las
   combinaciones: se niegan las mal formadas; la secuencial y la mezcla dan copias distintas y
   deterministas).
2. **Sin ensayo nuevo**: el `lr = 3e-3` es el de `ruido-nist`, congelado con la misma red y el
   mismo dato; se decide conservarlo.
3. `nn/lanzar.sh todo` (15 corridas, una unidad `expc-ruidocomb`). Cierre: informe, commit, almacén.
4. `README.md` con el veredicto; reporte en el central.

- **Qué se mide y umbral:** `instrucciones/02-criterio.md`.
- **Qué se llama «ganar»:** «suman» si la combinación supera al mejor simple en más del umbral.

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/ruido.py` | los 9 tipos (de `ruido-nist`) + `partes`, `aplicar_escenario`, `semilla_combo` | `python nn/ruido.py` · `--muestras` |
| `nn/datos.py` | el dataset y el train de un escenario, simple o combinado, fijo o en línea | `--comprobar` |
| `nn/modelo.py` | la red y los pesos iniciales como fichero | `--inicializar` (ya hechos) |
| `nn/entrenar_local.py` | una corrida | `--escenario <esc> --semilla <s>` · `--comprobar` |
| `nn/informe.py` | Δ contra `limpio` y contra cada simple, pareado | `python nn/informe.py` |
| `nn/lanzar.sh` | los 15 como unidad; seco; se niega a lanzar dos veces; salta lo hecho | `todo` · `corridas <id…>` · `--estado` · `SECO=1` |
| `nn/probar_lanzador.sh` | el despacho | `sh nn/probar_lanzador.sh` |

- ⚠ `entrenar_local.py` se llama así por el freno. Dependencias: el `.venv` de la raíz.

## Qué NO hereda

- **Se copió de:** `ruido-nist` (todo `nn/`, `.gitignore`, los `init-s{1,2,3}.pt`).
- **Qué se cambió a propósito:** la pregunta (combinar, que allí estaba excluida); los escenarios
  (5, no 38); dos operadores nuevos en el nombre (`+`, `~`) con semilla propia; `lanzar.sh todo`
  en vez de fases; `informe.py` compara contra el mejor simple, no por tipo; no hay ensayo de
  `lr` (se conserva el de allí, decidido).
- **Qué se conservó, decidido:** dato, red, `lr`, pasos, lote, semillas y pesos iniciales (para
  que los simples salgan **bit a bit** iguales y la comparación sea limpia), umbral `δ = 0,01`,
  en línea.
- **Contra qué se compara:** contra `limpio` y contra `gaussiano@0.2-linea` **de este
  experimento**. Los números de `ruido-nist` sólo de contexto.
- **Restricciones de otros experimentos que NO aplican:** las fases 2 y 4 de `ruido-nist`
  (intensidades, grosor), Vast, el banco.
