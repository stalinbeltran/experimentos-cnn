# Reglas de `delg-cnn`

**Qué pregunta:** ¿Puede una CNN pequeña, entrenada sólo con trazos sintéticos a predecir un mapa continuo de su línea
media, devolver la forma delgada de los dígitos reales de UCI mejor que las alternativas de `dig-delg`, conservando la
topología y abriendo los lazos que la tinta gruesa cierra?

## Entradas

- **Dataset:** `uci-optdigits-orig-32px-r20261005` (`expcnn.exigir_dataset`). No se regenera. **Sólo prueba**: ninguna
  red ve un dígito real al entrenar.
- **Qué se lee de él:** `imagenes` (32×32, `> 0`), `etiquetas` (para la tabla por clase y los 8/6/9 del criterio C),
  `particion == "val"` (1617) y los 64 índices de la figura 19 de `curvas-gabor` (copiados de `dig-delg`).
- **Verdad:** sintéticos con línea media conocida (`nn/sinteticos.py`, semillas 101 / 202), 6.000 + 600 + 600.
- **Condiciones que el dataset ya trae:** bitmaps binarios de NIST, trazos 3–6 px, ~2 px de margen.
- **Qué se normaliza o transforma al cargar:** nada; pre-proceso ninguno (regla del dueño del 2026-10-07).

## Salidas

- **Pesos:** `nn/pesos/<brazo>/best.pt · config.json · metrics.jsonl` (A, B, C; < 1 MB en total).
- **Métricas:** `resultados/metricas.json`: los 11 criterios con medido y pasa/no pasa por brazo, la tabla de
  agujeros por clase (tinta frente a `M > u`, tres u), parámetros y tiempos.
- **Figuras** (`resultados/`): `1-sinteticos.png` (entrada, verdad, salida de cada brazo) · `2-digitos-64.png` (los 64
  con `M` en escala continua — nunca binarizada — y su cresta) · `3-metricas.png` · `4-agujeros.png` (los 8 de val en
  los que `M` abre un lazo que la tinta tiene macizo, y los 6/9 en los que inventa uno) · `5-desplazamientos.png`.
  Se regeneran con `python nn/evaluar.py`.
- **Tablas / informe:** `README.md`, con enlaces `raw.githubusercontent.com`.
- **Qué se commitea:** todo lo de arriba, pesos incluidos. El dataset, nunca.

## Procesos

1. Commitear criterio y reglas (hecho el 2026-10-13, antes de `nn/`).
2. `nn/sinteticos.py`: trazos con verdad de línea media y grosor; 20 % de lazos macizos con su hueco en la verdad.
3. `nn/entrenar_local.py --brazo A|B|C`: 30 épocas, Adam 1e-3, lote 64, early stop por la pérdida de validación
   sintética; una semilla por brazo (es una comparación de brazos, no de semillas: si gana por poco se dice).
4. `nn/evaluar.py`: A–D del criterio, figuras, `metricas.json`; README con lo medido contra la predicción.

- **Qué se mide y con qué umbral:** los 11 de `instrucciones/02-criterio.md`.
- **Cuántos brazos y cuántas semillas:** 3 brazos × 1 semilla.
- **Qué se llama «ganar»:** el que pase más umbrales; desempate por topología (3 y 9).

## Desplazamientos

- **Qué se reconoce y a qué nivel:** la línea media (feature) y los agujeros de `M > 0,5` (objeto), en los 64 de B.
- **Rango y direcciones:** d = −4…+4 px, horizontal y vertical por separado, cada signo aparte.
- **Qué curva:** distancia mediana entre la cresta de `M(x movida d)` y la cresta de `M(x)` movida d, y % con los
  mismos agujeros, contra d; y el % que cambia respecto de d = 0.
- **Recorte:** ~2 px de margen; mover 3–4 px corta tinta; rellenar con fondo (nunca `np.roll`), % recortado en la
  figura, métricas sobre los no recortados.
- **Dónde queda:** `nn/evaluar.py --desplazamientos` → `resultados/5-desplazamientos.png`; forma esperada en
  `instrucciones/02-criterio.md` § D (plana para B; A y C pueden no serlo por la reducción).

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/sinteticos.py` | trazos sintéticos con verdad de línea media y grosor | `python nn/sinteticos.py` → `resultados/1-sinteticos.png` |
| `nn/modelo.py` | los tres brazos | importado por los otros dos |
| `nn/entrenar_local.py` | entrena un brazo con sintéticos | `python nn/entrenar_local.py --brazo A` |
| `nn/evaluar.py` | A–D, figuras, `metricas.json` | `python nn/evaluar.py [--desplazamientos]` |

- ⚠ `gasta` es `entrena-local`: el que entrena SE LLAMA `entrenar_local.py` (contrato con `cerrable.mjs`).
- **Dependencias:** el `.venv` del repo (numpy, scipy, torch, matplotlib).
- **De dónde sale el código:** copiado (no importado) de `rizo-2` (`nn/sinteticos.py`: `arco`, `recta`; `nn/modelo.py`
  como punto de partida; `rizo_huecos.huecos_separados` para el criterio 10) y de `dig-delg` (las cuatro medidas y los
  64 índices). Si `trazos` ya tiene su generador con verdad de polilínea cuando esto se corra, se copia de ahí.

## Qué NO hereda

- **Se copió de:** `rizo-2` (sintéticos, red, huecos) y `dig-delg` (medidas, 64 índices).
- **Qué se cambió a propósito:** (1) la salida es un **mapa continuo** y nada se mide con un solo umbral; (2) la verdad
  es la **línea media con campana**, no un rizo ni un borde; (3) los sintéticos incluyen **lazos macizos con hueco en
  la verdad**, que es la hipótesis del encargo; (4) se compara con `dig-delg` sobre sus mismos 64 dígitos, pero con la
  cresta de `M` en vez de un umbral.
- **Qué se conservó, y por qué:** las cuatro medidas de `dig-delg`, para que «mejor que las alternativas» sea
  comparable; el dataset sin pre-proceso.
- **Contra qué se compara:** los tres brazos entre sí y contra los números de `dig-delg`. Deja de ser comparable si
  alguna red ve dígitos reales al entrenar.
