# Reglas de `feat-fallos`

**Escritas el 2026-10-06.** Es un estudio **por iteraciones**: cada una tiene su criterio escrito **antes** en
`instrucciones/02-criterio.md`, y su resultado debajo. Este fichero describe las piezas; se reescribe cuando cambian.

**Qué pregunta:** por qué los detectores de features sintéticas fallan sobre los dígitos, y qué lo arregla sin perder
acierto. Pedido por el dueño el 2026-10-06 (`instrucciones/01-encargo.md`): *«propón soluciones y ponlas a prueba … corrige,
y lo vuelves a poner a prueba»*.

## Entradas

- **Los dígitos:** `uci-optdigits-orig-32px-r20261005` (los 5620; los 1797 de `windep` con su reparto 180/1617).
- **La prueba gruesa** de `feat-bor`: `feat-bor-sinteticas-grueso-32px-r20261006` (2–12 px; sólo se evalúa).
- **Para entrenar S2:** `feat-fallos-sinteticas-grueso-32px-r20261006`, publicado desde aquí: el sorteo **grueso** de
  `feat-ind` (sus semillas, +1000) a 32×32 sin reducir; **reducido 4×4 es bit a bit** `feat-ind-sinteticas-grueso-8px-r20261003d`
  (la corrida 4 de feat-ind), comprobado antes de publicar.
- **Los bancos de detectores** se leen de su experimento **por su id**, con huella cuando el origen la guarda: `lineas`
  (feat-ind32), `contorno` y `signo` (feat-bor), las cortas (feat-cortas), y los que entrena este (`lineas-grueso`, S2).
- **Preprocesados:** `nada` · `norm3` (esqueleto Zhang–Suen + dilatar a 3 px, `nn/normalizar.py`) · `a+b` (los mapas de las
  dos vistas, juntos).

## Salidas

- `resultados/combos/<banco>-<preprocesado>.json`: las cuatro métricas de cada combinación (`nn/evaluar.py`).
- `resultados/grosor-digitos.json` y `resultados/normalizacion.png`: la causa medida y la solución vista.
- Los pesos que entrene este experimento: `nn/pesos-<repr>-<dataset>/` (sólo `best.pt`, `config.json`, `summary.json` en
  git). El libro de Vast en `resultados/vast/`.
- Las figuras del estudio, `resultados/*.png`; el README, con lo que salió y las URL.
- **Reporte en el central:** sí, al terminar (alquila).

## Procesos

1. **Medir la causa** antes de probar nada (`nn/medir_grosor.py`).
2. **Cada iteración:** criterio escrito → correr `nn/evaluar.py <banco> <preprocesado>` → resultado y diagnóstico debajo
   del criterio → la siguiente iteración sale del diagnóstico.
3. Las soluciones que entrenan van a Vast (`nn/vast.sh detectores`) con la receta de feat-ind32 sin cambios.

- **Las cuatro métricas** (siempre las mismas, `nn/evaluar.py`): compositor posicional (180/1617, 3 semillas; curva
  36/180/1080 sobre el test de 717); firma (arcos en los 1); κ de los grupos al engrosar 2 px y adelgazar 1 px (zonas, K =
  30); prueba gruesa sintética (F1, recall, FP en 2–4 y 6–12 px).
- **Qué se llama «ganar»:** lo dice cada iteración; la referencia es siempre `lineas-nada`.

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/normalizar.py` | esqueleto + dilatar: el grosor fuera antes del detector | `python nn/normalizar.py` (comprueba) |
| `nn/medir_grosor.py` | grosor de trazo de los dígitos y de las features; la figura de la normalización | → `resultados/grosor-digitos.json`, `normalizacion.png` |
| `nn/evaluar.py` | una combinación banco × preprocesado, las cuatro métricas | `<banco> <preprocesado>` · `--todas` |
| `nn/datos.py` | genera y publica el dataset grueso de S2; los conjuntos de cada detector | `--generar-grueso [--publicar]` · `--comprobar` |
| `nn/entrenar_local.py` | entrena UN detector (contrato con el freno); receta de feat-ind32 | `--repr lineas --dataset grueso --feature f` · `--comprobar` |
| `nn/vast.sh`, `nn/vast.json`, `nn/probar_vast.sh` | Vast, un modo; freno `apagar` y `/use exp-vast` | `detectores` · `--estado` · `apagar` |
| `nn/bordes.py`, `nn/modelo.py`, `nn/features.py`, `nn/metricas.py` | copias (feat-bor, feat-ind32, feat-agr) | — |

## Qué NO hereda

- **Se copió de:** feat-ind32 (receta, red, compositor, reparto, transformaciones), feat-bor (bordes, entrenar_local con
  `--repr`), feat-agr (métricas y zonas). Se copia, no se importa.
- **Qué se cambió a propósito:** cada iteración cambia UNA cosa respecto de la referencia, y la dice.
- **Contra qué se compara:** siempre con `lineas-nada`, medido aquí con el mismo código.
- **Restricciones que NO aplican:** el banco de 8×8 (feat-ind); la etiqueta como objetivo de los detectores.

## Lo que cambió al correr (2026-10-06)

- **Siete iteraciones en vez de tres.** La 4 (desinclinar, compositor tolerante), la 5 (juntar lo que funcionó), la 6
  (erosionar según el grosor) y la 7 (confirmación ciega en los 3823 dígitos que no son de `windep`) salieron de los
  diagnósticos; cada una, escrita antes y commiteada, en `instrucciones/02-criterio.md`.
- **Bancos combinables:** `nn/evaluar.py` acepta varios bancos con «+» (`lineas+lineas-grueso+cortas`) y preprocesados en
  cadena con «_» (`desinc_norm3`); lee también las cortas de `feat-cortas` por su id y su huella.
- **S2 en Vast:** 28 vCPU (Xeon E5-2680 v4, 0,0956 $/h) para 13 procesos de **1 hilo** (`--hilos 1`; se pidieron ≥ 26 vCPU,
  2 por proceso), **19 min y 0,0303 $**.
- **Scripts nuevos:** `nn/curvas_rectas.py` (la pregunta del dueño, por banco), `nn/errores.py` y `nn/diagnostico.py`
  (dónde se concentra el error), `nn/pares.py` (qué detector empuja cada confusión), `nn/tolerante.py` (S5),
  `nn/confirmar.py` (la iteración 7) y `nn/figuras.py`; y en `nn/normalizar.py`, `desinclinar` y `adelgazar_segun_grosor`.
- **Ojo con las evaluaciones largas en el dev**: con 2 vCPU, una combinación de 68 mapas tarda ~20 min si corre a la vez que
  otra. Se lanzaron con `nohup` en colas (`/tmp/cola-*.sh`): un `Bash(run_in_background)` del arnés se cortó a los 30 min a
  mitad de `--todas` (medido el 2026-10-06 a las 02:46).
