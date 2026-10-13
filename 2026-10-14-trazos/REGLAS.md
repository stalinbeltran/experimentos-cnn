# Reglas de `trazos`

**Qué pregunta:** ¿Se pueden recuperar de un bitmap de 32 px los trazos con que se escribió el dígito (cuántos, por
dónde pasan, dónde empiezan y acaban, dónde se cruzan) con un método sin entrenar, y con qué acierto?

## Entradas

- **Dataset:** `uci-optdigits-orig-32px-r20261005` (`expcnn.exigir_dataset`). No se regenera.
- **Qué se lee de él:** `imagenes` (32×32, binarizadas con `> 0`), `etiquetas` (sólo para la tabla por clase) y
  `particion == "val"` (1617). `train` y `extra` no se usan: **no se entrena nada**.
- **La verdad:** (a) sintéticos con trazos conocidos (`nn/sinteticos_trazos.py`, semillas 101 calibrar / 202 probar);
  (b) `instrucciones/etiquetas-trazos.json`: 50 dígitos de val (5 por clase, elegidos con semilla 0), etiquetados a
  mano por Claude sobre rejilla ANTES de correr el método (número de trazos, extremos, cruces). ⚠ Etiquetas de Claude,
  ±1 px: la figura `0-etiquetas.png` está para que el dueño las corrija.
- **Condiciones que el dataset ya trae:** bitmaps binarios de NIST sin reducir, trazos de 3–6 px, dígito centrado con
  ~2 px de margen.
- **Qué se normaliza o transforma al cargar:** nada. Pre-proceso: **ninguno** (regla del dueño del 2026-10-07).

## Salidas

- **Pesos:** ninguno (no se entrena).
- **Métricas:** `resultados/metricas.json`: los 12 criterios de `instrucciones/02-criterio.md` con su medido y
  pasa/no pasa, la tabla clase × (trazos, cruces) de val, la curva de desplazamientos, los casos de grado ≥ 5 y el
  tiempo por dígito.
- **Figuras** (`resultados/`): `0-etiquetas.png` (los 50 sobre rejilla con su verdad) · `1-sinteticos.png` (ejemplos
  con verdad y recuperado) · `2-etapas.png` (un dígito etapa a etapa: tinta → línea media y grosor → grafo con poda →
  cruces resueltos → trazos, cada uno de un color) · `3-digitos.png` (los 50 con trazos de colores y la verdad) ·
  `4-por-clase.png` (la tabla clase × estructura) · `5-fallos.png` · `6-desplazamientos.png`.
  Se regeneran con `python nn/trazos.py`.
- **Tablas / informe:** `README.md` de este experimento, con enlaces `raw.githubusercontent.com` a cada figura.
- **Qué se commitea:** todo lo de arriba. El dataset, nunca.

## Procesos

1. Escribir y commitear criterio + etiquetas (este commit y el siguiente), antes de `nn/trazos.py`.
2. Sintéticos con verdad de trazos (`nn/sinteticos_trazos.py`): calibrar los dos parámetros con la semilla 101.
3. Correr el método en sintéticos de prueba (202), en los 50 etiquetados, en todo val y en los desplazamientos.
4. Figuras, `metricas.json` y README con lo medido contra la predicción.

- **Qué se mide y con qué umbral:** los 12 de `instrucciones/02-criterio.md`.
- **Cuántos brazos y cuántas semillas:** un método, dos parámetros tanteados en calibración; sin semillas (es
  determinista).
- **Qué se llama «ganar»:** no hay ganador; el método «sale» si pasa 1–3, 5–7 y 11 (criterio § final).

## Desplazamientos

- **Qué se reconoce y a qué nivel:** cada trazo (feature individual: extremos, cruces) y la cuenta por dígito (objeto
  entero), en los 50 etiquetados.
- **Rango y direcciones:** d = −4…+4 px, horizontal y vertical por separado, cada signo aparte.
- **Qué curva:** % con el mismo número de trazos que en d = 0, y error mediano de extremos respecto de etiqueta + d,
  contra d; y el % que cambia de cuenta respecto de d = 0.
- **Recorte:** ~2 px de margen; mover 3–4 px corta tinta en algunos. Se mueve rellenando con fondo (nunca `np.roll`),
  el % recortado va en la figura y las métricas se dan sobre los no recortados.
- **Dónde queda:** `nn/trazos.py --desplazamientos` → `resultados/6-desplazamientos.png`; forma esperada en
  `instrucciones/02-criterio.md` § D (plana: la línea media es local).

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/sinteticos_trazos.py` | dígitos sintéticos de 1–3 trazos con verdad (polilíneas, extremos, cruces) | `python nn/sinteticos_trazos.py` → `resultados/1-sinteticos.png` |
| `nn/trazos.py` | el método (línea media → grafo → poda → cruces → trazos), las medidas A–D, figuras y `metricas.json` | `python nn/trazos.py [--calibrar] [--desplazamientos] [--idx N]` |
| `nn/etiquetar.py` | pinta los 50 dígitos sobre rejilla con sus etiquetas | `python nn/etiquetar.py` → `resultados/0-etiquetas.png` |

- **Dependencias:** el `.venv` del repo (numpy, scipy, torch, matplotlib, scikit-image para `skeletonize` si la cresta
  de la EDT no basta — se decide en el paso 2 y se anota).
- **De dónde sale el código:** copiado de `rizo-2` (sintéticos: `arco`, `recta`; agujeros: `label` del fondo) y del
  boceto curvas-gabor (`curvas.campo`, la orientación local). Nada se importa de otro experimento.

## Qué NO hereda

- **Se copió de:** `rizo-2`: `arco` y `recta` de `nn/sinteticos.py` (para pintar trazos sintéticos con grosor) y la
  forma de agrupar por componente del fondo de `nn/rizo_huecos.py`; del boceto curvas-gabor, `curvas.campo` (la
  orientación local, si hace falta en los cruces).
- **Qué se cambió a propósito:** (1) los sintéticos llevan **verdad de trazos** (polilíneas, extremos, cruces), no de
  rizos; (2) la verdad en dígitos reales son **etiquetas de trazos** a mano, 5 por clase, no 48 doses; (3) no hay
  compositor ni entrenamiento: es un método determinista con dos parámetros; (4) el objeto de salida es el **trazo**,
  no la curva ni el hueco.
- **Qué se conservó, y por qué:** el dataset y su lectura (binarizar con `> 0`, sin pre-proceso), para que lo que
  salga de aquí se pueda combinar con los objetos de `rizo-2` sobre los mismos píxeles; y el protocolo de
  desplazamientos (regla del dueño).
- **Contra qué se compara:** no hay brazos: el método contra su verdad (sintética y a mano). Deja de ser comparable si
  los parámetros se tantean con los 50 etiquetados en vez de con los sintéticos de calibración.
