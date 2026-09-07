# Reglas de `esq-cq`

**Escritas el 2026-09-07** a partir de lo que hay en esta carpeta: los scripts, el manifiesto,
el criterio congelado y el README. Su `experimento.json` declara el estado **`abierto`**.

⚠ **Estas reglas son de este experimento y de ninguno más.** Vienen de copiar `esq-2d` y
**están releídas**: lo que se cambió y lo que se conservó está abajo, con su motivo
(`CLAUDE.md` § «Regla 0 — cada experimento es INDEPENDIENTE de los demás`).

**Qué pregunta:** colapsando las dos salidas de `esq-2d` en **una** —«hay esquina» y su
posición, sin importar si es la superior-izquierda o la inferior-derecha—, qué tamaño de kernel
único lo hace mejor, y cuánto se gana o se pierde respecto de tener que **declarar cuál** es.

## Entradas

- **Dataset:** `esquinas300-32px-r4-r20260907`, publicado en
  `foveal-vision-data/experimentos-cnn/`. **Es el mismo fichero que usa `esq-2d`**, y por eso
  se comparte por **nombre** en vez de regenerarse. Se lee con `expcnn.exigir_dataset(...)`,
  que se niega antes de empezar si no está publicado.
- **Qué se lee de él, y aquí está la diferencia:** las mismas particiones y la misma etiqueta
  de 6 números, pero **colapsada a una sola salida** — `existe = tl ∨ br` y **una** posición.
  ⚠ Compartir el dataset **no** es compartir la lectura: el mismo `.npz` alimenta dos
  experimentos que miden cosas distintas.
- **Positivas:** 156 de 389 en `val` (**40,1 %**), mitad `tl` y mitad `br`; 78 negativos duros
  (`tr`/`bl`).
- **Condiciones que el dataset ya trae:** semilla 1, ventana 32 px, reducción 4; párrafo a
  ≥ 32 px reducidos del borde y ≥ 64 px de lado; la esquina en `[8, 23]` de la ventana; sin
  solape. Una ventana **nunca** contiene las dos esquinas.
- **Qué se normaliza al cargar:** entrada en **tinta** (`/255`), 1×32×32.
- ⚠⚠ **Lo que se midió ANTES de diseñar nada, y acota lo que se puede esperar:** `br` **no es
  una esquina de tinta**. La mediana de la distancia al píxel de tinta más cercano es **1,0 px
  en `tl` y 8,1 px en `br`** (p90 12,2), y el cuadrante propio de `br` está **vacío en el 82 %**
  de sus 78 ventanas — la última línea del párrafo es corta, así que el vértice
  inferior-derecho de la caja cae sobre fondo. *Medido el 2026-09-07 sobre el `val.npz`
  publicado, 0 $.*

## Salidas

- **Pesos:** `nn/pesos/<brazo>/best.pt` y `last.pt` (`k05 k07 k09 k11 k13 k15 k17`).
  Commiteados (1,3 MB, dentro del tope de ≈5 MB por experimento).
- **Métricas:** `nn/pesos/<brazo>/metrics.jsonl`, una línea por época. Commiteado.
- **Figuras:** `muestras/*.png` — muestras congeladas por brazo y etiqueta de época, y
  `transformacion-<brazo>-10-paginas-*.png`.
- **Tabla de resultados:** la genera `nn/informe.py` (`--md` para pegarla en el `README.md`).
- **Qué NO se commitea:** `datos/` (etapa local).

## Procesos

1. **El dataset ya está publicado**: no se regenera. `nn/datos.py --comprobar` casa las huellas
   del publicado contra el manifiesto, y `--rederivar` comprueba que la receta lo vuelve a dar.
2. **Criterio congelado antes de mirar** (R13): `instrucciones/02-criterio.md`, escrito con
   cero épocas y los suelos de `nn/entrenar_local.py --suelos`.
3. **Entrenar los 7 brazos** × 300 épocas con `nn/lanzar_barrido.sh`.
4. **Figuras, páginas enteras e informe.**

- **Qué se mide:** acierto a **≤ 2 px** sobre las 156 ventanas positivas, **con su desglose por
  esquina verdadera desde la primera época**.
  ⚠ El desglose no es un extra: las positivas son mitad y mitad, así que **un ~50 % global es
  exactamente lo que sale de «`tl` entero, `br` nada»**. Un número global sin desglose, en este
  experimento, no se puede leer.
- **Umbral, congelado antes de mirar:** acierto ≤ 2 px **> 9,6 %** (suelo 5,8 % + 2·SE, con
  SE = 1,9 % sobre 156 positivas). Secundarios: error medio **< 5,70 px** y `f1` de `existe`
  **> 0,572**.
- **No se declara ganador:** se reportan todos los que pasan.
- **Brazos y semillas:** 7 brazos, **una** semilla (1). `epochs` 300, `lr` 0,05, lote 128,
  `lambda_coord` **0,0293** (recalculada para esta pérdida; el 0,038 era de otra).
- **Por qué el eje llega hasta `k` = 17, y no es una preferencia:** el radio del campo receptivo
  es `(k−1)/2`, o sea 2·3·4·5·6·7·8 px para `k` = 5…17. La tinta más cercana a `br` está a
  8,1 px de mediana, así que **ningún brazo de `k` = 5…13 puede verla**; `k` = 17 es el primero
  con radio 8 y además el techo del dataset (el mapa va de 8 a 23, justo el rango donde se
  sortean las esquinas).
- **Alternativa anotada y NO corrida:** `sim` (kernel forzado simétrico, equivariante bajo giro
  de 180°) — `instrucciones/03-alternativas-anotadas.md`. **Descartada por el dueño:** `rot`.
  **Descartada por incapaz:** `ant` — con lectura por máximo responde a una esquina con el signo
  cambiado de la otra, así que sólo una puede ser el máximo.

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/datos.py` | genera, publica y verifica el dataset | `--imagenes 300` · `--publicar` · `--comprobar` · `--rederivar` |
| `nn/entrenar_local.py` | entrena **un** brazo; reanudable | `--brazo k05 --epocas 300` · `--desde-cero` · `--comprobar` · `--suelos` |
| `nn/lanzar_barrido.sh` | lanza el barrido como **unidad de systemd** y lo consulta | (sin banderas) · `--estado` |
| `nn/modelo.py` | la estructura, autocontenida; ejecutado comprueba sus invariantes | (sin banderas) |
| `nn/muestras.py` | la figura de las muestras congeladas | `--brazo k17` · `--etiqueta ep000` |
| `nn/transformacion.py` | cada kernel sobre páginas enteras | `--paginas 10` · `--paginas 10 --brazo k13` · `--kernel` |
| `nn/informe.py` | la tabla de resultados | (sin banderas) · `--md` |

- **`entrenar_local.py` se llama así por el contrato con el freno** (`cerrable.mjs`): es lo
  único que hace que un entrenamiento vivo aparezca en el veredicto «¿se puede apagar este
  server?».
- **El barrido se lanza con `desacoplar-persistente.sh`** (padre PID 1), nunca con el
  `run_in_background` del harness, que muere con la sesión.
- **Dependencias:** `torch`, `numpy`, `Pillow`.
- **De dónde sale el código:** copiado de `esq-2d` y modificado. Copiado, **no importado**.

## Qué NO hereda

- **Se copió de:** `esq-2d`, por orden del dueño: *«copia la carpeta […], deja la carpeta como
  si recién se fuera a entrenar»*.
- **Qué se cambió a propósito:**
  - **una** salida en vez de dos: `existe` + una posición, **sin decir cuál esquina es**;
  - la posición se lee **sólo del máximo**; el mínimo ya no se usa;
  - cabeza de **3** parámetros (la de `esq-k`), no de 5 — con lo que desaparece el defecto de
    `esq-2d` de que *«la red no es idéntica»* al comparar;
  - `k` hasta **17** en vez de 13, por el radio del campo receptivo (arriba);
  - `lambda_coord` **0,0293** en vez de 0,038;
  - umbral **9,6 % global** en vez de «12 % en las dos esquinas»;
  - `ant` **se borra** y entra `sim`: con lectura por máximo, la exigencia se da la vuelta
    (hace falta `A → 0`, no `A` grande).
- **Qué se conservó, y por qué se decidió conservarlo:** el **dataset** (para que la
  comparación con `esq-2d` signifique algo), la receta, la semilla, las épocas, el `lr` y el
  lote. Cada uno es una decisión de este experimento, no una herencia.
- **Contra qué se compara:** contra `esq-2d`, para medir **cuánto cuesta tener que declarar
  cuál esquina es**. Dejaría de ser comparable si cambiara el dataset o la lectura de la
  etiqueta.
- **Restricciones de otros experimentos que NO aplican aquí:**
  - de `esq-2d`: la lectura por **mínimo**, el umbral por esquina y el techo `k` = 13;
  - de `esq-k`: la etiqueta de 3 números, la esquina única y `k` ≤ 11;
  - **y una que suena a rigor y no lo es:** *«esto no es `esq-2d` más fácil»*. Quita la
    dificultad barata (decir cuál es) y **conserva entera la cara** — que en `br` no hay señal
    donde está la etiqueta. Nadie tiene que «arreglar» esa diferencia.
