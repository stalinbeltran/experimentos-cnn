# Reglas de `lazos`

**Qué pregunta:** ¿puede un detector de lazos construido sobre el detector de curvas delgado (cada píxel curvo vota por
su centro de curvatura) encontrar los lazos de los dígitos con su centro, su grado de cierre (0 = media vuelta, abierto;
1 = vuelta completa) y la dirección de su abertura?

## Entradas

- **Dataset:** `uci-optdigits-orig-32px-r20261005`, leído con `expcnn.exigir_dataset(...)`; no se regenera.
- **Qué se lee de él:** las imágenes y etiquetas de la partición **val** (1617) para evaluar; las demás no se usan (no
  se entrena nada con dígitos).
- **Condiciones que el dataset ya trae:** 32×32; tinta = valor > 0 (se binariza así); sin margen vertical (los dígitos
  ocupan casi los 32 px de alto).
- **Qué se transforma al cargar:** nada más. Sin esqueleto, sin borde, sin suavizado de la imagen.
- **Sintéticos**: lazos (arcos de círculo de cobertura 0,5–1) y negativos, generados en `nn/lazos.py` con semilla fija
  (101 para calibrar, 202 para probar). No se publican: se regeneran idénticos.

## Salidas

- **Pesos:** ninguno.
- **Métricas:** `resultados/metricas.json` (calibración, sintéticos de prueba, dígitos por clase, desplazamientos).
- **Figuras:** `resultados/1-sinteticos.png` (ejemplos de prueba con verdad y detección), `resultados/2-digitos.png`
  (100 dígitos de val, 10 por clase), `resultados/3-estadisticas.png`. Dibujo pedido por el dueño: el dígito **sólo en
  gris**, una **cruz azul** en el centro de cada lazo y el lazo como **arco delgado** de su radio que cubre lo que cubre
  el trazo, con el hueco hacia su orientación.
- **Tablas / informe:** `README.md` de esta carpeta.
- **Qué se commitea:** todo lo de `resultados/`. El dato de entrada no.

## Procesos

1. Generar los sintéticos de calibrar y de probar.
2. Calibrar σ_v y V_min con los de calibrar.
3. Medir en los de probar y en los dígitos de val; desplazamientos; figuras.

- **Qué se mide y con qué umbral:** ocho umbrales, en `instrucciones/02-criterio.md` (escrito antes de mirar).
- **Cuántos brazos y semillas:** un detector, una calibración elegida; determinista.
- **Qué se llama «ganar»:** no hay ganador; se dice qué umbrales pasa.

## Desplazamientos

- **Qué se reconoce y a qué nivel:** cada **lazo** (una feature), no el dígito entero.
- **Rango y direcciones:** d = −4…+4 px en horizontal, cada signo aparte. La vertical **no se mide**: los dígitos de UCI
  ocupan los 32 px de alto y se recortan desde d = ±1 (medido en el boceto `curvas-gabor`).
- **Qué curva:** sobre los 1617 de val, % de dígitos cuyo NÚMERO de lazos cambia respecto de d = 0, y el desplazamiento
  medio del centro del lazo principal respecto de d (debería ser exactamente d).
- **Recorte:** se mide el % de dígitos recortados con cada d, en la misma figura; nada de `np.roll`.
- **Dónde queda:** `nn/lazos.py`, en `resultados/metricas.json` y en `resultados/3-estadisticas.png`. La forma esperada:
  **plana** salvo por el recorte, porque el detector es convolución + votos sin rejilla fija (regla 6).

## Qué NO hereda

- **Se copió de:** ningún experimento. Del boceto `curvas-gabor` se copia (no se importa) el detector de curvas delgado
  —kernel par, campo de orientación, giro— y su calibración, con TAU 0,6.
- **Qué NO aplica aquí:** el compositor de dígitos, las 3 semillas, train/val/a ciegas, el acierto como métrica, y el
  criterio de `dig-delg`. Aquí no se clasifican dígitos.
- **Contra qué se compara:** con lo esperable de cada clase de dígito (no hay verdad de lazos en el dataset).

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/detector.py` | el detector de curvas delgado (copiado) y el de lazos | se importa |
| `nn/lazos.py` | sintéticos, calibración, prueba, dígitos, desplazamientos, figuras | `python nn/lazos.py` |

- **Dependencias:** el `.venv` del repo con numpy, scipy, torch y matplotlib.
- **De dónde sale el código:** `nn/detector.py` copia `curvas.kernel_gabor`, `curvas.campo`, `curvas.giro` y la
  calibración de `delgadas.py` del boceto; el detector de lazos es nuevo.
