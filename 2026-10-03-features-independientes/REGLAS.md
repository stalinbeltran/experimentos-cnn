# Reglas de `feat-ind`

**Reescritas el 2026-10-03 al nacer la primera corrida** (hasta ese día el experimento era sólo
`OBSERVACIONES.md`). Vigentes para todo lo que hay en `nn/`.

**Qué pregunta:** si cada CNN pequeña reconoce una única feature fijada de antemano (reconoce, no
discrimina) y las `N` se entrenan independientes y en paralelo, ¿qué resultado neto dan juntas al
reconocer un objeto complejo compuesto por varias de esas features? ⚠ **En esta corrida NO se compara
contra una CNN monolítica** (orden del dueño del 2026-10-03: «no son necesarias, al menos en este
momento»); se mide (1) si cada detector aprende su feature y (2) qué dan dos compositores lineales
sobre dígitos reales.

## Entradas

- **Dataset:** `feat-ind-sinteticas-8px-r20261003`, publicado desde aquí (`nn/datos.py --generar
  --publicar`) y leído con `exigir_dataset`. 32.400 imágenes de 8×8 (cuentas 0..16): 13 familias ×
  (2000 train + 400 val) + `vacio` (1000 + 200). La especificación completa de cada familia está en
  `ESPECIFICACION.md` §2–§4. Para los compositores, además, **`uci-optdigits-8px-r20261002`** (los
  dígitos; reparto 180/1617 tal cual viene publicado).
- **Qué se lee de él:** `imagenes` (la entrada, /16), `principal`, `secundaria` (para armar
  positivos/negativos, con `CONTIENE` por medio), `ancla` (el objetivo), `radio`/`grosor` (desgloses). `mascara8` y `ancla32`
  **no** se usan en esta corrida: se guardaron para la alternativa «objetivo = máscara» (§5 de la
  especificación).
- **Condiciones que el dataset ya trae:** ruido leve en todas (`p_on` ≤ 0,02, `p_off` ≤ 0,10 a
  32×32); segunda feature en la mitad de las imágenes con trazo; `vacio` sin secundaria; trazos
  enteros dentro del lienzo.
- **Qué se normaliza o transforma al cargar:** cuentas / 16 → [0, 1]. Nada más.

## Salidas

- **Pesos:** `nn/pesos/<feature>/best.pt` (mejor F1 de val, desempate posición) y `last.pt`, con
  `config.json`, `metrics.jsonl` (por época) y `summary.json` (métricas de `02-criterio.md` §A,
  desgloses por radio/grosor, FP por familia, curva P/R por umbral, veredicto).
- **Métricas de los compositores:** `resultados/compositores.json`; la firma por clase de los
  detectores en `resultados/firma-por-clase.json`; los mapas en `resultados/mapas-digitos.npz`
  (**no se commitea**: `*.npz`; se regenera con `nn/aplicar.py` en segundos).
- **Figuras:** `resultados/muestras-features.png` (`nn/features.py --muestras`) y
  `resultados/mapas-digitos.png` (`nn/aplicar.py`). PNG con PIL: este venv no tiene matplotlib.
- **Qué se commitea:** todo menos `*.npz`. 13 `.pt` × 2 × ~16 KB ≈ 0,4 MB, bajo el tope de 5 MB.
- **Reporte en el central:** no en esta corrida — no cambia nada de `ESTADO.md` (ningún parámetro
  de la red de producción). `README.md` de aquí recoge qué salió.

## Procesos

1. Especificación y criterio **antes** de mirar: `ESPECIFICACION.md`, `instrucciones/02-criterio.md`.
2. Generar y publicar el dataset, una vez (`nn/datos.py --generar --publicar`); mirar la rejilla.
3. `nn/entrenar_local.py --comprobar` (mecanismo, determinismo, `--desde`, `--contra`).
4. Entrenar los 13 detectores como unidad de systemd: `nn/lanzar.sh todas` (≈2 min cada uno con los
   defectos de la corrida 2: 16/32/32, 80 épocas, pérdida con término de presencia; la corrida 1 —
   8/16/16, 40 épocas, sin ese término, con las contenedoras como negativos y umbral 0,5 fijo— está
   resumida en `resultados/corrida-1-2026-10-03.json`).
5. `nn/aplicar.py` (mapas sobre dígitos + rejilla) y `nn/compositor.py`.
6. `README.md` con los resultados, y las enmiendas al criterio si las hubo, fechadas.

- **Qué se mide y con qué umbral:** `02-criterio.md` §A (por detector: F1 ≥ 0,90 y posición ≤ 1
  celda ≥ 0,90 = «aprendió») y §B (compositores: se reportan; «ven algo» si presencia > 0,40). El
  umbral de «hallada» es por detector, elegido sobre train (enmienda 2). Los negativos de un
  detector excluyen las familias que lo **contienen** (`features.CONTIENE`, enmienda 1).
- **Cuántos brazos y cuántas semillas:** 13 detectores × 1 semilla (semilla 1; un detector tarda
  40 s y se puede repetir con `--semilla`); compositores: 3 semillas del optimizador.
- **Qué se llama «ganar»:** no se declara ganador. Cada detector recibe su veredicto; los
  compositores se reportan con su Δ (posicional − presencia).

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/features.py` | vocabulario + rasterizador 32→8; rejilla de muestras | `python nn/features.py [--muestras]` |
| `nn/datos.py` | genera/publica/comprueba el dataset; arma el conjunto de un detector; carga los dígitos | `--generar [--publicar]` · `--comprobar` · `--rederivar` |
| `nn/modelo.py` | el detector (autocontenido), el objetivo gaussiano, la lectura del mapa | `python nn/modelo.py` |
| `nn/entrenar_local.py` | entrena un detector o los 13; **re-entrena** con `--desde` y `--contra` | `--feature f [--semilla s] [--epocas n] [--desde p] [--contra a,b] [--canales 16,32,32] [--peso-objetivo 8] [--peso-presencia 1] [--lote-neg 32]` · `--todas` · `--comprobar` |
| `nn/lanzar.sh` | lo lanza como unidad de systemd; imprime la orden; modo seco; se niega a lanzar dos veces | `todas` · `una <f> [args]` · `--estado` · `SECO=1 …` |
| `nn/aplicar.py` | los 13 sobre los dígitos → mapas, firma por clase, rejilla | `python nn/aplicar.py [--n k]` |
| `nn/compositor.py` | los dos compositores lineales | `python nn/compositor.py [--sufijo -c3]` |
| `nn/combinar.py` | junta los mapas fino (c2) y grueso (c4) en un solo npz de 26 mapas con nombres | `python nn/combinar.py` → luego `compositor.py --sufijo -c24`, `errores.py --sufijo -c24` |
| `nn/obtenedor.py` | grupo `dig`: 13 kernels 5×5 por k-means de parches de 20 dígitos SIN etiqueta; `DetectorKernel` autocontenido | `--aprender` → `nn/pesos-dig/` · `--aplicar` → `mapas-digitos-dig.npz` y `-c24dig.npz` |
| `nn/obtenedor_cnn.py` | grupos `cae` (c7), `cae5` y `cae3` (c8): autocodificador convolucional disperso (WTA), codificador = 13 CNN independientes, 100 dígitos SIN etiqueta | `[--grupo cae5\|cae3] --aprender` → `nn/pesos-<grupo>/` · `--aplicar` → `mapas-digitos-<grupo>.npz` y `-c24<grupo>.npz` |
| `nn/contribucion.py` | contribución de cada detector por eliminación (re-entrena el compositor sin él) | `python nn/contribucion.py --sufijo -c24dig` |
| `nn/errores.py` | atribución del error del compositor posicional (culpable, reconocimiento/composición, difíciles en sí) | `python nn/errores.py [--sufijo -c3]` |

- ⚠ `gasta` es `entrena-local`: el que entrena **se llama `entrenar_local.py`** (contrato con el freno).
- **Dependencias:** el `.venv` de la raíz del repo (torch 2.14 CPU, numpy, Pillow). Sin matplotlib
  ni scikit-learn.
- **De dónde sale el código:** autónomo. La idea «dibujar a 32 y contar bloques 4×4» es de
  `ruido-nist`; el código está reescrito aquí, no importado.

## Qué NO hereda

- **Se copió de:** ninguno. (`OBSERVACIONES.md` nació sin copiar nada; el código tampoco.)
- **Qué se cambió a propósito:** no aplica. Decisiones propias que podrían confundirse con herencia:
  el 8×8 y la reducción por conteo son **para vivir en el espacio de los dígitos publicados**, no por
  parecerse a `dim-nist`/`ruido-nist`; la red es totalmente convolucional con padding (allí era sin
  padding + pooling global), porque aquí la salida es un mapa.
- **Qué se conservó, y por qué:** la orden literal en `instrucciones/01-encargo.md` y las
  observaciones previas, que son el contexto de la pregunta.
- **Contra qué se compara, si es que se compara:** los detectores contra su propio val sintético;
  los compositores entre sí (presencia vs posicional) y contra el azar (0,10). **No** contra
  `dim-nist`/`ruido-nist`, aunque usen los mismos dígitos: aquellos entrenan sobre los dígitos y éste
  sólo entrena el compositor sobre ellos. Si algún día se compara, se escribe aquí.
- **Restricciones de otros experimentos que NO aplican aquí:** los 3996 pasos, los pesos iniciales
  compartidos por fichero y las 3 semillas de `ruido-nist`; el `k = ⌈W/2⌉` de `dim-nist`. La
  pregunta de la **posición relativa** es de `feat-pos`; aquí la posición sólo entra como entrada del
  compositor posicional.
