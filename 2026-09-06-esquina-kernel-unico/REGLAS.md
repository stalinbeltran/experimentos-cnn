# Reglas de `esq-k`

**Escritas el 2026-09-07, con el experimento ya CERRADO**, a partir de lo que hay en su
carpeta: los scripts, el manifiesto, el criterio congelado y el README. Lo que no consta en
ninguno de esos sitios se dice que no consta, en vez de reconstruirlo de memoria.

⚠ **Este experimento está cerrado: nada de aquí se cambia.** Sus números se reportaron con el
código que tiene. Si algo suyo parece «mal alineado» con los experimentos de al lado, no lo
está: cada uno decide sus propias reglas
(`CLAUDE.md` § «Regla 0 — cada experimento es INDEPENDIENTE de los demás»).

**Qué pregunta:** con UN solo kernel y una cabeza de 3 parámetros, qué tamaño de kernel produce
la transformación más efectiva para detectar la esquina superior-izquierda de un párrafo.

## Entradas

- **Dataset:** `esquina-tl300-32px-r4-r20260906`, publicado en
  `foveal-vision-data/experimentos-cnn/`.
  ⚠ **Se publicó el 2026-09-07, DESPUÉS de que el experimento terminara**, y por eso su código
  no lo lee de ahí: `nn/datos.py` y `nn/entrenar_local.py` leen de `datos/` **local** (que el
  `.gitignore` de la carpeta excluye). Es un hecho de este experimento, no un defecto que
  haya que arreglarle: los experimentos posteriores leen del publicado con
  `expcnn.exigir_dataset(...)` y hacen bien, y éste se quedó como corrió.
  ⚠ Y hasta el 2026-09-07 su `experimento.json` declaraba el nombre
  `esquina-limpia-800x600-r4`, que no corresponde a ninguna carpeta publicada; se corrigió al
  añadir la comprobación de datasets a `comprobar.py`.
- **Qué se lee de él:** las tres particiones `train` · `val` · `muestra`, con la etiqueta en
  **formato de 3 números** — `(existe, x, y)` de **una sola** esquina, la superior-izquierda.
  Los experimentos posteriores usan etiquetas de 6 números; **no son intercambiables**.
- **Condiciones que el dataset ya trae** (manifiesto: semilla 1, ventana 32 px, reducción 4,
  300 imágenes pedidas · 267 válidas; 2.159 · 389 · 120 ventanas):
  párrafo a ≥ 32 px reducidos del borde, párrafo ≥ 64 px de lado, la esquina cae dentro de la
  ventana con **≥ 5 px de margen ciego** (el de `k_max` = 11) y ≥ 8 px de contexto, sin
  solape. Los negativos duros son las otras esquinas (`tr`, `bl`, `br`).
- **Qué se normaliza al cargar:** la entrada va en **tinta** (`/255`), 1×32×32.

## Salidas

- **Pesos:** `nn/pesos/<brazo>/best.pt` y `last.pt`, un directorio por brazo
  (`k03 k05 k07 k09 k11`). **Se commitean** (696 KB en total, dentro del tope de ≈5 MB por
  experimento).
- **Métricas:** `nn/pesos/<brazo>/metrics.jsonl`, una línea por época. También commiteado.
- **Figuras:** `muestras/*.png` — las 10 muestras congeladas por red y por etiqueta de época
  (`k07-ep300.png`, `k07-ep000-sin-entrenar.png`), más `kernel-k07.png` y las dos de
  `transformacion-k07-*`. Se regeneran con `nn/muestras.py` y `nn/transformacion.py`.
- **Tabla de resultados:** en el `README.md` de este experimento, escrita a mano. **Este
  experimento no tiene `informe.py`** — los posteriores sí. Es otra diferencia deliberada.
- **Qué NO se commitea:** `datos/` (el `.gitignore` de la carpeta), porque el dato de entrada
  no entra en este repo; vive publicado en el repo de datos.

## Procesos

1. **Generar el dato** en la etapa local: `nn/datos.py --imagenes 300`, y comprobar que la
   receta lo reproduce (`--comprobar`: mismas huellas SHA-256 tras regenerar de cero,
   verificado el 2026-09-06).
2. **Congelar el criterio antes de mirar** (R13): `instrucciones/02-criterio.md`, escrito con
   cero épocas entrenadas.
3. **Entrenar los 5 brazos**, uno por tamaño de kernel.
4. **Figuras y lectura**: `nn/muestras.py` y `nn/transformacion.py`.

- **Qué se mide:** fracción de esquinas predichas a **≤ 2 px** de su sitio, **sólo** sobre las
  ventanas donde la esquina existe de verdad (156 de 389 en `val`).
- **Umbral, congelado antes de mirar:** un brazo **ha aprendido algo** si supera el **13 %**
  (= 8,3 % del techo sin entrenar + 2·SE, con SE = 2,2 % sobre 156 positivas).
- **Qué NO se usa como métrica principal:** la distancia media, porque su suelo parece bueno y
  no lo es (5,74 px salen sólo de la geometría) y porque el modo degenerado de la cabeza
  coincide con el mejor predictor constante. El porqué completo está en el criterio.
- **Brazos y semillas:** 5 brazos (`k03 k05 k07 k09 k11`), **una** semilla (1).
- **Qué se llama «ganar»:** el que pasa el umbral; a igualdad, **el kernel más pequeño**
  (`GANADOR = "k07"` en `nn/transformacion.py:37`).

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/datos.py` | genera el dataset en `datos/` y comprueba su huella | `--imagenes 300` · `--comprobar` |
| `nn/entrenar_local.py` | entrena **un** brazo; **reanudable** | `--brazo k03 --epocas 60` · `--desde-cero` · `--comprobar` |
| `nn/modelo.py` | la red, autocontenida; ejecutado imprime su tabla | (sin banderas) |
| `nn/muestras.py` | la figura de las 10 muestras congeladas | `--brazo k03` · `--etiqueta ep000` |
| `nn/transformacion.py` | aplica el kernel aprendido a entradas nuevas | `--kernel` · `--entradas 20` · `--paginas 10` |

- **`entrenar_local.py` se llama así por el contrato con el freno** (`cerrable.mjs`): con otro
  nombre, el veredicto «¿se puede apagar este server?» no ve el entrenamiento.
- **Dependencias:** `torch` y `numpy` (`uv pip install -e '.[torch]'`); `Pillow` para las
  figuras. El generador de párrafos sólo hace falta para rehacer el dato.
- **De dónde sale el código:** escrito para este experimento. `nn/modelo.py` no importa nada
  del repo, para que los pesos se puedan cargar dentro de un año.

## Qué NO hereda

- **Se copió de:** **de ninguno** — es el primer experimento del repo.
- **Qué se cambió a propósito:** nada que heredar, nada que cambiar.
- **Qué se conservó:** ídem.
- **Contra qué se compara:** contra sus propios suelos medidos sin entrenar, y contra el
  predictor constante en el centro. **Nada más.** Los experimentos posteriores se comparan
  contra éste; eso es cosa suya y no le impone nada a éste.
- **Restricciones de otros experimentos que NO aplican aquí:** todas las de los posteriores.
  En concreto: **una** esquina y no dos, etiqueta de **3** números, `k` hasta **11**, 60
  épocas por defecto (no 300), **sin** `informe.py`, **sin** `lanzar_barrido.sh`, y el dato
  leído de `datos/` local y no del dataset publicado. Ninguna de esas diferencias es un
  defecto que haya que corregir.
