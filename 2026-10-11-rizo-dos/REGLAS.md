# Reglas de `rizo-2`

**Qué pregunta:** ¿Cuál de cuatro estrategias detecta mejor el rizo superior de los 2 (su centro, radio, grado de cierre
y abertura) en los dígitos de UCI de 32 px: sondeo por rayos, correlación con plantillas de rizo, votos por centro de
curvatura, o una CNN pequeña entrenada sólo con rizos sintéticos?

## Entradas

- **Dataset:** `uci-optdigits-orig-32px-r20261005` (`expcnn.exigir_dataset`). No se regenera.
- **Qué se lee de él:** `imagenes` (32×32, binarizadas con `> 0`), `etiquetas` (la clase, sólo para agrupar los
  resultados por clase) y `particion == "val"` (1617 dígitos). Las particiones `train` y `extra` no se usan: **ninguna
  estrategia se entrena con dígitos** — la C se entrena con sintéticos y las otras tres no se entrenan.
- **La verdad de los dígitos:** `instrucciones/etiquetas-2.json`, 48 doses de val etiquetados a mano (centro, radio,
  cierre, abertura; uno marcado `dudoso`, que no cuenta en los umbrales). Se escribieron antes de correr nada.
- **Condiciones que el dataset ya trae:** bitmaps binarios de NIST sin reducir, trazos de 3–6 px, dígito centrado en
  32×32 con margen de ~2 px.
- **Qué se normaliza o transforma al cargar:** nada. Pre-proceso: **ninguno**, el mismo (ninguno) para sintéticos y
  dígitos (regla del dueño del 2026-10-07).

## Salidas

- **Pesos:** `nn/pesos/<brazo>/best.pt · config.json · metrics.jsonl` de la estrategia C (dos brazos: `c-base` y
  `c-ancha`). Pequeños (< 1 MB en total).
- **Métricas:** `resultados/metricas.json`: por estrategia, los 15 criterios con su medido y pasa/no pasa, los
  % por clase, las curvas de desplazamiento y el tiempo por dígito.
- **Figuras** (`resultados/`), todas con el dibujo pedido por el dueño ayer: dígito en gris, cruz azul en el centro,
  arco delgado del rizo con el hueco hacia la abertura:
  `1-sinteticos.png` (ejemplos con verdad y lo que ve cada estrategia) · `2-doses-etiquetados.png` (los 48, las
  cuatro estrategias en columnas, etiqueta en verde) · `3-por-clase.png` (% con rizo superior por clase y estrategia) ·
  `4-desplazamientos.png` (las curvas) · `5-fallos.png` (los doses que la ganadora no encuentra) ·
  `0-etiquetas.png` (las 48 etiquetas sobre la rejilla, para que el dueño las corrija).
- **Tablas / informe:** `README.md` de este experimento, con los enlaces `raw.githubusercontent.com` a cada figura.
- **Qué se commitea:** todo lo de arriba. El dataset, nunca.

## Procesos

1. Etiquetar a mano los 48 doses (hecho, sobre una rejilla, antes de correr nada) y commitear criterio + etiquetas.
2. Generar sintéticos con verdad (`nn/sinteticos.py`): rizos de radio 2–7, grosor 2–6, cobertura 0,5–1, abertura
   cualquiera, con una **cola** que sale de un extremo (como el 2) y 0–2 distractores (rectas, arcos de < 0,5 vuelta);
   negativos sin rizo. Calibrar con semilla 101, probar con 202.
3. Calibrar R y P con los sintéticos de calibración (rejilla pequeña, puntuación = media de encontrados y negativos
   limpios). V no se calibra (es la referencia). Entrenar C (`nn/entrenar_local.py`, dos brazos, 1 semilla cada uno,
   local, CPU).
4. Evaluar las cuatro igual (`nn/evaluar.py`): sintéticos de prueba, los 48 doses, las clases, los desplazamientos.
   Figuras y `metricas.json`. Escribir el README con lo medido contra la predicción.

- **Qué se mide y con qué umbral:** los 15 de `instrucciones/02-criterio.md`.
- **Cuántos brazos y cuántas semillas:** C: 2 brazos × 1 semilla (es una comparación de estrategias, no de semillas:
  si C gana por poco se dice que es una semilla).
- **Qué se llama «ganar»:** la que pase más umbrales; desempates en el criterio.

## Desplazamientos

- **Qué se reconoce y a qué nivel:** una feature individual (el rizo superior) en los 47 doses etiquetados no dudosos.
- **Rango y direcciones:** d = −4…+4 px, horizontal y vertical por separado, cada signo aparte.
- **Qué curva:** % encontrados (candidato a ≤ 2,5 px de etiqueta + d) contra d, por estrategia, y el error mediano del
  centro respecto de etiqueta + d.
- **Recorte:** el dataset deja ~2 px de margen; mover 3–4 px corta tinta en algunos. Se mueve rellenando con fondo
  (nunca `np.roll`), se cuenta qué % de dígitos pierde tinta en cada d y va en la figura; el % encontrados se da sobre
  los no recortados.
- **Dónde queda:** `nn/evaluar.py` → `resultados/4-desplazamientos.png`; la forma esperada, en el criterio
  (planas para R/P/V; C baja ≤ 3 puntos).

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/sinteticos.py` | genera rizos sintéticos con verdad, colas, distractores y negativos | se importa; `python nn/sinteticos.py` dibuja 24 ejemplos en `resultados/1-sinteticos.png` |
| `nn/detector.py` | COPIA del detector de curvas delgado + lazos + rizos por rayos de `lazos` (estrategia V) | se importa |
| `nn/rayos.py` | estrategia R: `rizos(x, F, rmax, disp)` y su calibración | se importa; `python nn/rayos.py` calibra y escribe `resultados/cal-rayos.json` |
| `nn/plantillas.py` | estrategia P: banco y `rizos(x, S)` | se importa; `python nn/plantillas.py` calibra → `resultados/cal-plantillas.json` |
| `nn/modelo.py` | la CNN de la estrategia C, autocontenida | se importa |
| `nn/entrenar_local.py` | entrena un brazo de C con sintéticos | `python nn/entrenar_local.py <c-base\|c-ancha> [--epocas N]` → `nn/pesos/<brazo>/` |
| `nn/evaluar.py` | las cuatro estrategias con los mismos números: A, B, C, D del criterio; figuras y `metricas.json` | `python nn/evaluar.py` (~10 min) |
| `nn/etiquetas.py` | dibuja las 48 etiquetas sobre la rejilla | `python nn/etiquetas.py` → `resultados/0-etiquetas.png` |

- **Dependencias:** `.venv` del repo (`uv venv .venv && uv pip install torch numpy scipy matplotlib`); torch CPU.
- **De dónde sale el código:** `nn/detector.py` copiado de `lazos` (que a su vez lo copió del boceto curvas-gabor);
  el resto es de aquí. No se importa de ningún otro experimento.

## Qué NO hereda

- **Se copió de:** `lazos` (`lazos`): su `nn/detector.py` entero, la idea de los rayos de `nn/rizos.py`, el
  generador de arcos sintéticos y el dibujo del dueño (`dibujar`).
- **Qué se cambió a propósito:** (1) la verdad de los dígitos son **etiquetas a mano**, no lo esperable por clase;
  (2) los rayos admiten rizos **abiertos** (F calibrada, no 0,97) y exigen anillo (distancias parejas); (3) los
  sintéticos llevan **cola y distractores** y grosores hasta 6 px, para que lo que se calibra se parezca al dígito;
  (4) se añaden dos estrategias nuevas (plantillas, CNN); (5) la tolerancia de acierto baja de 3 a 2,5 px porque los
  radios son de 3–4 px.
- **Qué se conservó, y por qué:** el detector de lazos **sin tocar** (σ_v 1, V_min 3, TAU 0,6), porque es la referencia
  de lo medido ayer; el dibujo, porque lo pidió el dueño.
- **Contra qué se compara:** las cuatro entre sí con la misma verdad. Deja de ser comparable si una estrategia se
  entrena o calibra con los dígitos etiquetados (ninguna lo hace).
- **Restricciones de otros experimentos que NO aplican aquí:** el `TAU` y los umbrales de `lazos` sólo rigen en V.
