# Reglas de `bor-k`

**Escritas el 2026-10-01** al montar la carpeta, como parte de la fase 0d del plan
[`docs/plan-kernels-banco-2026-10-01.md`](../docs/plan-kernels-banco-2026-10-01.md) (E1).
Estado **`abierto`**: el código existe y está comprobado; **nada entrenado todavía**.

⚠ **Estas reglas son de este experimento y de ninguno más.** **No se copió de ningún
experimento**; la lectura de la cabeza viene de `esq-2d` como idea, no como código ni como
condiciones (§ «Qué NO hereda»).

**Qué pregunta:** un solo kernel de convolución de `k ≤ 19`, leído por los extremos de sus
perfiles horizontal y vertical, ¿localiza los cuatro bordes de la caja de tinta de un párrafo?
Y por `k`: ¿cuáles aprende y cuáles no?

## Entradas

- **Dataset: `parrafos1000-pagina1024-r4-r20261001`** (lo produce `bor-p4`), leído con
  `expcnn.exigir_dataset(...)` en `nn/datos.py`. No se regenera.
- **Qué se lee:** `paginas.npz` (`paginas`, `particion`), `cajas.npz` (`cajas` en px de
  render, `derivadas` en px guardados). La partición train/val/eval es **la del dataset, por
  página**: ninguna ventana de `eval` sale de una página de `train`.
- **Las ventanas** (todo en `nn/datos.py`, con el porqué de cada número):
  - lado **55**; la red **lee** sólo la región `[9, 45]` (37×37), **la misma para todo `k`**
    (la convolución `valid` se recorta después, como el §6.2 del banco);
  - **4 por párrafo**, una por borde, centradas en un punto sorteado uniforme a lo largo del
    borde y **desplazadas** `u ~ U[−12, 12]` perpendicular a él (centrada siempre, predecir
    «el centro» daría error 0 sin aprender nada); **más una al azar por párrafo** en su
    página (negativos y bordes de refilón; la que toque dos párrafos se re-sortea);
  - **limpias por construcción**: 27 + 12 = 39 ≤ la media ventana limpia que `bor-p4`
    garantiza (79 → 39). Se comprueba con un `assert` al construir, no se supone;
  - semilla de las ventanas **fija** (`SEMILLA_VENTANAS = 0`): las tres semillas de
    entrenamiento ven **las mismas** ventanas.
- **La etiqueta**, por borde `t ∈ (izq, der, sup, inf)`: `existe_t` y `coord_t`. Un borde
  existe si su línea cae en `[10, 44]` y cubre ≥ 6 px de la región leída. La coordenada va en
  el espacio de **índices de píxel** (el borde continuo menos 0,5: cae **entre** dos píxeles,
  que es donde responde una derivada; lo comprueba `nn/modelo.py`).
- **Qué se normaliza:** `x = 1 − suma/4080` (papel 0, tinta 1). Nada más.

## Salidas

- **Pesos:** `nn/pesos/kNN-sS/` → `best.pt` (mínima pérdida de `val`), `last.pt` (época 300),
  `metrics.jsonl` (una línea por época, sólo añadir). `best.pt` lleva `conv.weight` de forma
  `(1, 1, k, k)`: es lo que lee `banco-k/nn/importar_kernel.py --de bor-k`.
  27 brazos × ~20 KB por `.pt` + ~60 KB de `metrics.jsonl` ≈ **2–3 MB** *(estimado)*, dentro
  del tope del repo (≈5 MB por experimento).
- **El libro de Vast:** `resultados/vast/fase2/` (un JSON por máquina, paso a paso, con coste)
  y lo que se trajo y no cabía en su sitio, en `resultados/vast/fase2/<k>/traido/`.
- **Informe:** por borde, en `README.md`, contra el suelo trivial (§ Procesos).
- **Qué se commitea:** todo lo de arriba. **Nunca** el dataset.

## Procesos

1. **Suelos y `LAMBDA_COORD`**, antes de la primera época: `nn/entrenar_local.py --suelos`.
   Mide el **suelo trivial** de cada borde (prevalencia y coordenada mediana de `train`) y el
   `λ` que iguala `bce` y `l1` en la red sin entrenar; ese `λ` se escribe en el código y aquí
   y **se congela**.
2. **Ensayo de mecanismo** en el dev (k=9, 20 épocas): sólo comprueba que la pérdida de
   `train` baja sin oscilar con la `lr` elegida. **No se mira `val`.**
3. **Entrenar los 27 brazos en Vast**, `nn/vast.sh fase2`: una máquina por `k`, sus tres
   semillas a la vez (`--hilos` = vCPU/3), 300 épocas, Adam `lr` 0,02, lote 128, sin parada
   temprana; `best.pt` por `val`.
4. **Informe por borde** y **importar los 27 al banco** (`importar_kernel.py --de bor-k`):
   todos, hayan aprendido o no — el banco decide su utilidad, no este experimento.

- **Qué se mide y con qué umbral:** en `instrucciones/02-criterio.md`, escrito **antes**.
- **Brazos y semillas:** `k ∈ {3, 5, …, 19}` × `{1, 2, 3}` = **27**.
- **Qué se llama «ganar»:** no se declara ganador. Se dice, por `k` y por borde, si aprendió.

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/datos.py` | las ventanas y sus etiquetas, en memoria | `python nn/datos.py` (resumen + comprobación) |
| `nn/modelo.py` | la red: 1 kernel + cabeza de 9 parámetros | `python nn/modelo.py` (lee 4 bordes con un kernel puesto a mano) |
| `nn/entrenar_local.py` | entrena un brazo, reanudable | `--brazo k09-s1 [--epocas N] [--hilos N]` · `--suelos` · `--comprobar` |
| `nn/vast.json` | el descriptor de la fase 2: una máquina por `k` | lo lee `vast_instance.py trabajo` |
| `nn/vast.sh` | lanza/consulta/apaga la flota | `fase2 [k…]` · `--estado` · `apagar` · `VAST_SECO=1 …` |

- ⚠ **`entrenar_local.py` se llama así por el FRENO** (`cerrable.mjs` → `TRABAJOS`).
- **Dependencias:** el `.venv` de la raíz de `experimentos-cnn` (torch CPU, numpy). En Vast,
  el mismo torch fijado por versión (2.14.1).
- **De dónde sale el código:** escrito aquí. No importa nada de ningún experimento.

## Qué NO hereda

- **Se copió de:** ningún experimento. La idea de la cabeza (máximo y mínimo de un mapa, con
  `existe` por `logsumexp`) es la de `esq-2d`; se reescribió para dos ejes.
- **Qué se cambió a propósito respecto de `esq-2d`:** ventana de 55 (no 32), región leída fija
  para todo `k` (no `k`-dependiente), lectura por **perfiles** marginales (no por el mapa
  entero), cuatro `existe` (no dos), pérdida L1 en px (no MSE), 3 semillas (no 1).
- **Qué se conservó y por qué:** `β` aprendida con `BETA0 = 3,5` y la entrada en tinta
  `[0, 1]`: son decisiones de forma de la cabeza que aquí sirven igual.
- **Contra qué se compara:** contra **su suelo trivial** (aquí) y contra el **aleatorio de igual
  norma y `k`** (en el banco). No contra `esq-2d`: otra tarea, otro dato.
- **Restricciones de otros experimentos que NO aplican aquí:** la reserva §3.7 del banco **sí**
  aplica, y la cumple el dataset (`bor-p4`). Nada más del banco se hereda: ni su lienzo, ni su
  CNN, ni sus particiones.
