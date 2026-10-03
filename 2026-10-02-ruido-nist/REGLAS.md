# Reglas de `ruido-nist`

**Escritas el 2026-10-02 como plan y actualizadas ese mismo día al implementar. Estado `abierto`,
PREPARADO y NO lanzado**: hay código con pruebas, pesos iniciales compartidos, `lr` congelado y
criterio escrito; **ninguna corrida del estudio ha entrenado**. Se lanza cuando el dueño lo diga.

**Qué pregunta:** con 180 imágenes de train y 1617 de validación limpias, la misma CNN y los
mismos pesos iniciales por semilla, ¿qué tipo de ruido aplicado sólo al train mejora la
exactitud de validación, y cuánto?

## Entradas

- **Dataset**: `uci-optdigits-8px-r20261002` (publicado por `dim-nist`), leído con
  `expcnn.exigir_dataset`. Su reparto 180 / 1617 tal cual. Validación **nunca** se toca: su huella
  como float32 (`57a6cb6086e0f956`) está congelada en `nn/datos.py`, se comprueba al cargar y viaja
  en cada `summary.json`; `nn/informe.py` se queja si alguno no la lleva.
- **Train de cada escenario** = las 180 originales + 180 copias con ese ruido (una realización fija,
  generada con la semilla de ruido del escenario; su huella va en el `summary.json` y las tres
  semillas de pesos tienen que compartirla); `limpio` = las 180 duplicadas. Las copias ocupan los
  índices 180–359. Semillas: pesos 1/2/3, lotes 100+s, ruido 1000+10·t+i (+5000 por realización
  extra) (`ESPECIFICACION.md` §1 bis).
- **Transformación**: `x = cuenta/16 ∈ [0,1]`. Los trazos se dibujan como máscara a 32×32, se
  reducen a cobertura por bloque 4×4 y se componen `x' = x + α·c·(v − x)`; `borrado`/`externos`
  por Binomial a nivel de bit; `gaussiano` y `sal-pimienta` directamente a 8×8. Los nueve tipos,
  sus cinco niveles y su `t` están en `nn/ruido.py` (`TIPOS`, `NIVELES`) y en `ESPECIFICACION.md` §2.
- **Nombres**: escenario = `<tipo>@<nivel>[-r<realización>][-linea]` o `limpio`; corrida =
  `<escenario>-s<semilla>` (p. ej. `oblicua@0.6-r2-s3`, `recorte@0.6-linea-s1`). Un nivel que no
  esté en la tabla **se niega**. **`-linea`** = ruido en línea: la copia se regenera **cada época**
  del mismo generador (misma semilla de ruido; la época 1 es la copia fija), idéntica entre
  semillas de pesos; `datos.escenario` devuelve `regenerar` y el bucle lo llama al empezar cada
  época > 1.

## Salidas

- **Por corrida**, en `nn/pesos/<escenario>-s<s>/`: `last.pt` (época final; **sin `best.pt`**),
  `metrics.jsonl` (una línea por época; cada 10, exactitud y CE de val y de las 180 de train
  limpias), `summary.json` (exactitud y CE de val, de train limpio y de las 360; **por dígito**;
  huellas de init, copia y val; semillas; máquina; segundos), `log.txt`.
- **Pesos iniciales**: `nn/init/init-s{1,2,3}.pt` + `huellas.json`, **commiteados** (9 KB cada uno).
- **Pesos finales**: no se commitean aquí (`.gitignore`), **sí van al almacén** con todo lo demás
  (`foveal-vision-data/experimentos-cnn-resultados/ruido-nist/`, lo hace el cierre de
  `nn/lanzar.sh`).
- **Informe** `resultados/RESULTADOS.md` + `criterio-aplicado.json` + figura `delta-por-tipo.png`,
  regenerados por `nn/informe.py`; **rejilla** `resultados/muestras-ruido.png` (`nn/ruido.py --muestras`).
- **Qué se commitea:** métricas, resúmenes, logs, figuras, informe, pesos iniciales. Nunca el
  dataset ni las copias ruidosas (son dato derivado: se regeneran de la semilla y se reconocen por
  la huella).

## Procesos

0. **El dueño ya confirmó los supuestos** S1–S5 y las semillas (2026-10-02). Lo que falta es que
   diga **lanza**.
1. `nn/datos.py --comprobar` (huellas, 180/1617, val intacta en los 9 tipos, copia determinista),
   `nn/probar.py` (54 pruebas: nombres, semillas, qué hace cada tipo, escenarios, modelo, init),
   `nn/entrenar_local.py --comprobar` (ruido nulo = `limpio` **bit a bit** en los 9 tipos; con
   ruido, otros pesos; determinista), `sh nn/probar_lanzador.sh` (el despacho). **Todo en verde el
   2026-10-02.**
2. **Ensayo de mecanismo** (`--ensayo --lr …`, sólo pérdida de train sobre `limpio`), **hecho**:
   **`lr = 3e-3` congelado** (pérdida 0,934 → 0,0005, acc_train 1,0; `1e-3` también baja, a 0,0276).
   18–20 s por ensayo en el dev.
3. **Fase 1**: `nn/lanzar.sh fase1` (33 corridas en serie como unidad de systemd `expc-ruidonist`,
   padre PID 1; cada corrida es un proceso `nn/entrenar_local.py`). Al terminar la unidad corre
   `informe.py`, la rejilla, commitea, copia todo al almacén y avisa. `--estado` lee el disco.
4. **Fase 2**: `nn/lanzar.sh fase2 <tipos de criterio-aplicado.json → fase2>`.
4 bis. **Fase 3** (en línea): `nn/lanzar.sh fase3` — el mejor nivel de cada tipo con «ayuda» en
   `criterio-aplicado.json → por_tipo`, con `-linea`, × 3. El informe añade «en línea contra fija».
5. `README.md` con el veredicto; reporte en `estudios-redes-neuronales` (estudio con reloj, 0 $).

- **Qué se mide y con qué umbral:** `instrucciones/02-criterio.md`, escrito antes.
- **Corridas:** fase 1 = 11 escenarios × 3 = 33; fase 2 = 4 niveles nuevos × 3 por tipo (84);
  fase 3 = 5 escenarios en línea × 3 = 15.
- **Qué se llama «ganar»:** no hay ganador global; cada tipo queda en ayuda / perjudica /
  indistinguible, con su mecanismo, y los que pasan van a la fase 2.

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/ruido.py` | los 9 tipos, deterministas por generador; nombres y semillas de escenario; la rejilla | `python nn/ruido.py` (tabla) · `--muestras [png]` |
| `nn/datos.py` | lee el dataset publicado, congela huellas, arma el train de un escenario | `--comprobar` · `--huellas` |
| `nn/modelo.py` | la red (3×3×3, C=8, GAP), autocontenida; los pesos iniciales como fichero | `python nn/modelo.py` · `--inicializar` |
| `nn/probar.py` | las pruebas sin entrenar | `python nn/probar.py` |
| `nn/entrenar_local.py` | una corrida; `metrics.jsonl` según ocurre; `summary.json` al final | `--escenario <esc> --semilla <s> [--pasos N] [--hilos N] [--salida dir]` · `--ensayo --lr … [--escenario …]` · `--comprobar` · `--inicializar` |
| `nn/informe.py` | Δ pareado por escenario, veredictos, fase 2, realización, figura; del disco | `python nn/informe.py` |
| `nn/lanzar.sh` | la fase como unidad; imprime la orden; se niega a lanzar dos veces; salta lo hecho; cierre | `fase1` · `fase2 <tipo…>` · `fase3` · `corridas <id…>` · `--estado` · `SECO=1 …` |
| `nn/probar_lanzador.sh` | el despacho del lanzador, con el seco (14 casos) | `sh nn/probar_lanzador.sh` |

- ⚠ **`entrenar_local.py` se llama así por el freno** (`cerrable.mjs` → `TRABAJOS`).
- **Dependencias:** el `.venv` de la raíz del repo (torch 2.14.1+cpu, numpy 2.5.3, matplotlib 3.11,
  Pillow 12 para rasterizar las máscaras).
- **De dónde sale el código:** la forma del bucle y del lanzador se copió de `dim-nist` y se
  reescribió (escenarios, tres generadores, init como fichero, Δ pareado). No importa nada de
  ningún experimento.

## Qué NO hereda

- **Se copió de:** la forma de `nn/entrenar_local.py`, `nn/lanzar.sh` e `nn/informe.py` de
  `dim-nist` (ahorrar tecleo). Usa el dataset de `dim-nist`, y nada más de él.
- **Cambia respecto de `dim-nist`**: resolución fija (8 px), kernel fijo 3×3, **3 capas** (allí 2
  con `n = ⌈W/2⌉`); la variable es el **ruido de entrenamiento**; **3 semillas** en vez de 5 (pedido
  del dueño), compensado comparando **pareado** con pesos iniciales idénticos **desde fichero**; el
  train tiene **360** imágenes (18 pasos por época, 222 épocas) en vez de 180 (9 y 444); la base no
  es un piso sino `limpio`; el criterio es Δ contra la base, no W\*/suficiente/estorba.
- **Se conserva, decidido**: reparto 180/1617, 3996 pasos de lote 20, Adam con `lr` de ensayo de
  sólo train, sin selección, evaluación cada 10 épocas, `C = 8`, `δ = 0,01` y `δ_ce = 0,05`.
- **Contra qué se compara**: contra `limpio` de la misma semilla. Con `dim-nist` sólo de contexto
  (su `w8` da 0,851 ± 0,025 con otra red y 5 semillas: no es la base de nada aquí).
- **Restricciones de otros experimentos que NO aplican:** la reserva §3.7 del banco, Vast, el
  control de información de `dim-nist`.
