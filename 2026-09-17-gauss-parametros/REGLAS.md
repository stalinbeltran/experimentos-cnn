# Reglas de `gauss-p`

**Qué pregunta:** qué `sigma` (y qué `k` para contenerlo) merece la pena meter por la
puerta de `banco-k`, visto sobre 10 muestras de `train` de su propio dataset, dado que el
único punto gaussiano ya evaluado es el defecto `k/6` y **nadie lo eligió**.

⚠ **Esto NO es un experimento de medida, y hay que leerlo sabiéndolo.** No entrena, no
mide, no declara y no tiene criterio de éxito propio. Produce **una hipótesis** —un par
`(k, sigma)`— que después juzga `banco-k` con **su** criterio, que está escrito desde
antes y es invariante (§2.1/§2.2 de su especificación). Las secciones de abajo que en
otro experimento hablarían de brazos y semillas dicen aquí «no aplica», y eso es una
decisión, no un hueco.

## Entradas

- **Dataset:** `parrafos1000-584px-r4-r20260908b`, el mismo de `banco-k`. Publicado en el
  repo de datos; se lee con `expcnn.exigir_dataset(...)` y **no** se regenera.
- **Qué se lee de él:** la partición **`train`** (100 imágenes de 146×146 `uint16`), sus
  etiquetas de 4 coordenadas y el `meta.npz` con los factores por imagen. **`monitor` y
  `eval` no se tocan.** `eval` es la partición con la que el banco declara: elegir un
  parámetro mirándola sería fuga del conjunto de prueba, un fallo más viejo y más
  conocido que la reserva del §3.7. Y `train` la red ya la ve, así que mirarla no añade
  fuga ninguna.
- **⚠ Y de `train` sólo las LIBRES DE LA RESERVA del §3.7.** `banco-k` reserva
  `LiberationMono` y el interlineado `[1.45, 1.60]` para uso exclusivo suyo, y su regla
  dice que *«los procedimientos que producen kernels no pueden usarlas»*. **Elegir
  `sigma` mirando muestras es un procedimiento que produce kernels**, aunque quien mire
  sea un ojo y no un optimizador: el §3.7 no habla de cómo se elige, habla de qué datos
  se miran. *Medido el 2026-09-17: de las 100 de `train`, **45 caen en la reserva y se
  descartan; quedan 55**.*
  La reserva se lee del **manifiesto del dataset**, no de una constante tecleada aquí, y
  si el manifiesto no la declara el experimento **se niega** en vez de suponer que no
  hay reserva.
- **Condiciones que el dataset ya trae:** marco de 584 reducido /4 → 146; etiqueta = caja
  de **tinta** en el marco de 584; colocación en `[68, 512]`.
- **Qué se transforma al cargar:** nada propio. Se aplica **el pipeline del §6 de
  `banco-k` tal cual** —normalizar el kernel en L2, convolución válida, recorte fijo a
  128, estandarizar con μ/σ— porque el objetivo es enseñar lo que la red **recibirá**, no
  una versión legible.
  ⚠ **Con una diferencia medida y declarada:** μ y σ salen de las **55 libres**, no de
  las 100 que usa el §6.5. El desvío máximo en z es **≈0,39–0,42** y es **prácticamente
  el mismo en todas las condiciones**, o sea un desplazamiento común que no altera la
  comparación entre parámetros. Se mide con `nn/muestras.py --desvio`.

## Salidas

- **Pesos:** no aplica. Aquí no se entrena nada.
- **Métricas:** **ninguna, a propósito.** La pantalla describe la **forma** del filtro
  (cuánto corta la ventana, masa central, distancia a la identidad y a la caja plana),
  y **no puntúa ningún `sigma`**. Una cifra de calidad inventada aquí sería un criterio
  escrito **después** de mirar, compitiendo con el del banco, que está escrito antes
  (R13). Hay un test que lo fija.
- **Figuras:** `nn/figura.py` produce la rejilla de muestras × `sigma`. Su destino por
  defecto es `/tmp`; una figura **elegida** puede ir a `muestras/` y commitearse, como
  hace `banco-k` con la suya.
  ⚠ **Un volcado de todas las combinaciones NO se commitea**: es republicar el dataset
  con otro nombre, y el dataset vive en un repo **privado** mientras éste es **público**.
- **El producto de verdad:** un par `(k, sigma)` y **la huella `sha256_16`** del kernel
  que se miró. No un `.npy`: el `.npy` lo escribe **el banco** con su propio código
  (`nn/kernels.py --gauss`), porque un kernel que no se regenera desde código commiteado
  es un dato huérfano.
- **Qué se commitea:** el código, este fichero, el `README.md` y como mucho una figura.
  **Nunca** el `.npz`, ni una copia suya, ni la caché de la app (vive en memoria).

## Procesos

0. **(sólo en una máquina nueva)** `nn/app.py instalar` — escribe la unidad de systemd, la
   habilita y abre el puerto en `ufw`, las tres en el mismo paso. Hace falta porque
   `experimentos-cnn` **no está** entre los repos que clona un dev nuevo.
1. **Mirar** — `/use gaussp` → `url`, y en el móvil se mueve `sigma` y se compara contra
   la identidad. O `nn/figura.py` si se prefiere una imagen fija.
2. **Elegir** — un `(k, sigma)`. La app da la huella del kernel que enseñó.
3. **Meter el kernel al banco** — `nn/kernels.py --gauss --k K --sigma S --guardar
   --esperado <huella>`, en `banco-k`. **Se niega si la huella no coincide**: eso
   significaría que la copia del §6 que hay aquí y el original han divergido, y que se
   eligió mirando una cosa y se mediría otra.
4. **Evaluar** — `nn/lanzar.sh kernel kernels/<el>.npy`, en `banco-k`. **Ese paso no es
   de este experimento**, tiene su propio lanzador con unidad, modo seco y aviso, y su
   criterio está escrito desde antes.

- **Qué se mide y con qué umbral:** aquí, nada. El criterio que declara es el §2.1/§2.2
  de `banco-k`. Lo que sí está escrito **antes de mirar** —qué esperar de esto y cuándo
  vale la pena pagar una evaluación— está en `instrucciones/02-criterio.md`.
- **Cuántos brazos y cuántas semillas:** no aplica. Las semillas las pone el banco (10).
- **Qué se llama «ganar»:** **no se declara nada aquí.** Este experimento acota dónde
  mirar; el veredicto lo da el banco o no lo da nadie.

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/banco.py` | el §6 de `banco-k` **copiado**: aplicar, normalizar, estandarizar, `gauss` | `python nn/banco.py` (corre las pruebas) |
| `nn/muestras.py` | elige las muestras y **hace cumplir la reserva del §3.7** | `python nn/muestras.py [--n 10] [--desvio]` |
| `nn/forma.py` | la forma del kernel y **cuánto lo corta la ventana `k`** | `python nn/forma.py [--k 9]` |
| `nn/app.py` | la app web, y sus mandos | `python nn/app.py [url\|estado\|instalar\|abrir\|cerrar\|forma]` |
| `nn/figura.py` | el mismo barrido en una imagen, sin depender del puerto | `python nn/figura.py --sigmas 0.4,0.8,1.5` |
| `nn/probar.py` | **las 59 comprobaciones** | `python nn/probar.py` |

- **Dependencias:** el `.venv` del repo con **numpy y pillow**. No hace falta torch: aquí
  no se entrena. El paso 4 (evaluar en el banco) **sí** lo necesita, y es de allí.
- **De dónde sale el código:** `nn/banco.py` está **copiado** de `banco-k` (`nn/pipeline.py`
  y `nn/kernels.py`), como manda la Regla 0 del repo: ningún experimento importa de otro.
  El resto es propio.
  ⚠ **Y la copia no se deja sola.** Aquí enseñar algo distinto de lo que el banco medirá
  es el único fallo que no se ve, así que `nn/probar.py` importa el original **por su
  ruta de fichero** y exige que las dos den el **mismo array bit a bit**. *Medido el
  2026-09-17: coinciden en los 9 `k` del contrato y en 54 pares (k, sigma), con
  diferencia máxima 0. Y la huella de `gauss(9, 1.5)` es `c5daea4e1e3c0099`, que es
  exactamente la del kernel que el banco **ya evaluó** (`resultados/gauss-s3/config.json`).*

## Qué NO hereda

- **Se copió de:** de ninguno. Es una carpeta nueva; de `banco-k` se copió **código**
  (el §6), no condiciones.
- **Qué se cambió a propósito:**
  - **la partición**: el `nn/muestras.py` de `banco-k` dibuja desde `eval`; aquí es
    `train`, porque aquí se **elige** y allí sólo se ilustra;
  - **`sigma` no tiene defecto**: `banco.gauss()` lo exige. El defecto `k/6` del banco es
    justo el punto ya evaluado, y dejarlo caer ahí sería contestar la pregunta antes de
    hacerla;
  - **no hay métrica ni criterio de éxito**, por lo dicho arriba.
- **Qué se conservó, y por qué se decidió conservarlo:** el pipeline del §6 **entero y sin
  tocar**, incluido el recorte fijo de 9 px y la estandarización. Es lo único que hace que
  lo que se ve sea lo que la red recibirá; cambiar cualquier pieza convertiría la vista
  previa en una ilustración bonita y falsa.
- **Contra qué se compara:** contra la **identidad** (§6.3, recorte puro), que es la línea
  base del banco, y contra el punto `k=9, sigma=1.5` que el banco ya midió. Dejaría de ser
  comparable si `banco-k` cambiara su §6 — y entonces el apretón de manos **se niega**, que
  es el punto.
- **Restricciones de otros experimentos que NO aplican aquí:** ninguna de `banco-k`
  —ni sus 10 semillas, ni sus particiones, ni su `K_MAX`— **salvo dos, que sí y a
  propósito**: la **reserva del §3.7**, que no es una condición del experimento sino una
  prohibición del sistema, y el **contrato de kernel del §5** (`k` impar, 3 ≤ k ≤ 19),
  porque un `(k, sigma)` que no lo cumpla no puede entrar por la puerta y elegirlo sería
  perder el rato.
