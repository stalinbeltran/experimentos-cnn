# Reglas de `esq-2d`

**Escritas el 2026-09-07, con el experimento ya CERRADO**, a partir de lo que hay en su
carpeta: los scripts, el manifiesto, el criterio congelado y el README.

⚠ **Este experimento está cerrado: nada de aquí se cambia.** Sus números se reportaron con el
código que tiene (`CLAUDE.md` § «Regla 0 — cada experimento es INDEPENDIENTE de los demás»).

**Qué pregunta:** una sola convolución sin bias produce **un** mapa, y las dos esquinas en
diagonal se leen de sus dos extremos — la superior-izquierda del **máximo** y la
inferior-derecha del **mínimo**. Qué tamaño de kernel lo hace mejor, y cuánto se pierde
respecto del mismo barrido sobre **una** sola esquina.

## Entradas

- **Dataset:** `esquinas300-32px-r4-r20260907`, publicado en
  `foveal-vision-data/experimentos-cnn/`. Se lee con `expcnn.exigir_dataset(...)`, que se
  niega antes de empezar si no está publicado; **no** se re-deriva al vuelo.
- **Qué se lee de él:** las particiones `train` · `val` (2.157 · 389 ventanas) con la etiqueta
  de **6 números** — `(existe_tl, x_tl, y_tl, existe_br, x_br, y_br)` —, y
  `muestras-congeladas.npz` para las figuras. **Se usan las dos esquinas de la diagonal
  principal**; `tr` y `bl` entran como **negativo duro** (misma forma local, girada 90°).
- **Condiciones que el dataset ya trae:** semilla 1, ventana 32 px, reducción 4, 300 imágenes
  pedidas · 267 válidas; párrafo a ≥ 32 px reducidos del borde y ≥ 64 px de lado; la esquina
  cae en `[8, 23]` de la ventana; sin solape. **Las imágenes son las mismas que las de
  `esq-k`** (misma receta, misma semilla) pero **las ventanas son otras**, así que los números
  de aquel experimento sólo son comparables contra su propio dataset.
- **Un invariante que se comprueba, no se supone:** una ventana **nunca** contiene las dos
  esquinas — el párrafo mide ≥ 64 px reducidos y la ventana 32.
- **Qué se normaliza al cargar:** la entrada va en **tinta** (`/255`), 1×32×32.

## Salidas

- **Pesos:** `nn/pesos/<brazo>/best.pt` y `last.pt` (`k05 k07 k09 k11 k13`). Commiteados
  (904 KB, dentro del tope de ≈5 MB).
- **Métricas:** `nn/pesos/<brazo>/metrics.jsonl`, una línea por época. Commiteado.
- **Figuras:** `muestras/*.png` — las muestras congeladas por brazo y etiqueta de época, y
  `transformacion-<brazo>-10-paginas.png` (los kernels sobre **páginas enteras**).
- **Tabla de resultados:** la genera `nn/informe.py` leyendo la época del `best.pt`; con
  `--md` sale en markdown para pegar en el `README.md`. **No se transcribe a mano.**
- **Qué NO se commitea:** `datos/` (etapa local, `.gitignore` de la carpeta).

## Procesos

1. **Generar y publicar el dataset**: `nn/datos.py --imagenes 300`, luego `--publicar` (no
   pisa uno existente), `--comprobar` (huellas del publicado contra el manifiesto) y
   `--rederivar` (¿la receta y la semilla lo vuelven a dar?).
2. **Congelar el criterio antes de mirar** (R13): `instrucciones/02-criterio.md`, con cero
   épocas entrenadas y los suelos medidos con `--suelos`.
3. **Entrenar los 5 brazos** × 300 épocas, con `nn/lanzar_barrido.sh` — que se **commitea** a
   propósito, para que «¿cuál se usó?» tenga respuesta, y trae su `--estado` que lee del
   **disco** y no del log.
4. **Figuras, páginas enteras e informe.**

- **Qué se mide:** acierto a **≤ 2 px**, **por esquina** y sólo sobre las ventanas donde esa
  esquina existe (78 `tl` y 78 `br` en `val`).
- **El titular es la PEOR de las dos esquinas, nunca el promedio**: un promedio escondería
  justo el desenlace más probable (que resuelva `tl` y no `br`), que es la pregunta.
- **Umbral, congelado antes de mirar:** **12 % en LAS DOS esquinas** (suelo + 2·SE, con
  SE = 2,5 % en `tl` y 2,8 % en `br` sobre 78 positivas → 10,1 % y 12,0 %; se toma el mayor
  para las dos, que es la dirección estricta).
- **No se declara ganador:** se reportan **todos** los que pasan, cada uno con sus cinco
  números. El umbral sigue haciendo falta, y no es lo mismo que un ganador: separa «aprendió»
  de «se quedó en el suelo».
- **Brazos y semillas:** 5 brazos, **una** semilla (1). `epochs` 300, `lr` 0,05, lote 128,
  `lambda_coord` 0,038.
- **Alternativas anotadas y NO corridas:** `ant`, `ind-tl`, `ind-br`
  (`instrucciones/03-alternativas-anotadas.md`). **Descartada por el dueño:** `rot` (girar la
  entrada 180°).

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/datos.py` | genera, publica y verifica el dataset | `--imagenes 300` · `--publicar` · `--comprobar` · `--rederivar` |
| `nn/entrenar_local.py` | entrena **un** brazo; reanudable | `--brazo k05 --epocas 300` · `--desde-cero` · `--comprobar` · `--suelos` |
| `nn/lanzar_barrido.sh` | lanza el barrido como **unidad de systemd** y lo consulta | (sin banderas) · `--estado` |
| `nn/modelo.py` | las estructuras, autocontenidas; ejecutado comprueba sus invariantes | (sin banderas) |
| `nn/muestras.py` | la figura de las muestras congeladas | `--brazo k13` · `--etiqueta ep000` |
| `nn/transformacion.py` | cada kernel sobre páginas enteras | `--paginas 10` · `--paginas 10 --brazo k13` · `--kernel` |
| `nn/informe.py` | la tabla de resultados | (sin banderas) · `--md` |

- **`entrenar_local.py` se llama así por el contrato con el freno** (`cerrable.mjs`).
- **El barrido se lanza con `desacoplar-persistente.sh`** (padre PID 1), nunca con el
  `run_in_background` del harness. Medido el 2026-09-07 en este mismo barrido: la unidad
  sobrevivió al final de la sesión que la lanzó (`NRestarts=0`) y los dos vigilantes del
  harness no.
- **Dependencias:** `torch`, `numpy`, `Pillow`. El generador sólo para rehacer el dato.
- **De dónde sale el código:** copiado de `esq-k` y modificado. **Copiado, no importado**:
  ningún experimento importa de otro.

## Qué NO hereda

- **Se copió de:** `esq-k`.
- **Qué se cambió a propósito:**
  - **dos** esquinas en vez de una, leídas de los **dos extremos** del mismo mapa;
  - etiqueta de **6** números en vez de 3, y **dataset nuevo** por eso mismo;
  - cabeza de **5** parámetros en vez de 3;
  - `k` hasta **13** (allí, hasta 11);
  - **300** épocas (allí, 60 por defecto);
  - umbral **12 % en las dos esquinas** (allí, 13 % en una);
  - **no declara ganador** (allí, el más pequeño a igualdad);
  - añade `informe.py`, `lanzar_barrido.sh` y la vista sobre **páginas enteras**;
  - lee el dato del **dataset publicado**, no de `datos/` local.
- **Qué se conservó, y por qué:** la receta de imágenes, la semilla, la reducción y las
  condiciones de caja — **para poder comparar** contra `esq-k`, que es la pregunta («cuánto se
  pierde respecto de una sola esquina»). Conservarlo fue una decisión, no inercia.
- **Contra qué se compara:** contra `esq-k` **en lo grueso**, y el criterio ya declara por qué
  la comparación es aproximada. Dejaría de ser comparable si cambiara la receta de imágenes o
  las condiciones de caja.
- **Restricciones de otros experimentos que NO aplican aquí:** las de `esq-k` (una esquina,
  3 números, `k` ≤ 11) y las de `esq-cq` (una sola salida colapsada, lectura sólo por máximo,
  `k` hasta 17). Este experimento lee **máximo y mínimo**, que es justo lo que el siguiente
  quita.
