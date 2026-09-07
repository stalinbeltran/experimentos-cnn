# `esq-cq` — ⚠ PARTIDA SIN DEFINIR: copia de `esq-2d` lista para entrenar

**No se ha corrido ni una época, y la pregunta todavía no está escrita.** Esta carpeta es una
copia literal del montaje de [`esq-2d`](../2026-09-07-esquinas-diagonales/) —código, dataset,
estructura y alternativas anotadas— con **todo lo producido por su barrido borrado**. Sirve como
punto de partida para el experimento siguiente, no como un experimento en sí.

## ⚠ Lo que falta antes de lanzar nada

Tres cosas, y las tres son del dueño porque son el criterio, no la mecánica:

1. **`experimento.json` → `titulo` y `pregunta`.** Hoy dicen `POR DEFINIR`. El preflight las
   exige no vacías, así que el marcador pasa la comprobación — pero un `POR DEFINIR` en el índice
   del README raíz es justo lo que tiene que verse hasta que se decida.
2. **`instrucciones/02-criterio.md` está congelado para la pregunta de `esq-2d`**, no para ésta.
   **R13: el criterio se escribe antes de mirar** — reescribirlo *después* de la primera época
   deja de ser un criterio y pasa a ser una explicación.
3. **`instrucciones/01-encargo.md` y `03-alternativas-anotadas.md`** son igualmente los de
   `esq-2d`. Valen como herencia; hay que decir qué sigue aplicando.

Mientras `pregunta` diga `POR DEFINIR`, lo que salga de aquí no es comparable con nada, porque
nadie ha declarado contra qué se compara.

## Qué se conservó, y por qué

| Qué | Por qué se queda |
|---|---|
| `nn/*.py`, `nn/lanzar_barrido.sh` | es el montaje que se hereda — el punto de copiar |
| `nn/manifiesto.json`, `nn/receta.json` | identidad del dataset de entrada, no resultado del barrido |
| `instrucciones/` | la herencia de `esq-2d`, pendiente de revisar (ver arriba) |
| `datos/` *(ignorado por git)* | caché local del dataset publicado; no es producto de entrenar |

## Qué se borró, y por qué

Todo lo que sólo existe **porque `esq-2d` ya corrió**. Nada de esto se pierde: vive en
[`esq-2d`](../2026-09-07-esquinas-diagonales/), que está commiteado y empujado.

| Qué | Cuánto |
|---|---|
| `nn/pesos/k{05,07,09,11,13}/` (`best.pt`, `last.pt`, `metrics.jsonl`) | 892 KB, 5 brazos |
| `muestras/*.png` (`ep000-sin-entrenar`, `ep300`, `transformacion-*`) | 15 figuras |
| `nn/__pycache__/` | — |

⚠ **Las `ep000-sin-entrenar` también se borraron**, aunque suenen a «antes de entrenar»: son la
salida de `muestras.py` sobre la inicialización aleatoria concreta de `esq-2d`. Si esta copia
cambia el modelo, mienten. Se regeneran en segundos:

```bash
.venv/bin/python 2026-09-07-esquinas-cualquiera/nn/muestras.py    # sin pesos, etiqueta ep000 sola
```

## El montaje heredado

Lo que sigue describe **cómo está montado hoy**, tal como vino de `esq-2d`. Si la pregunta nueva
cambia algo de esto, hay que cambiarlo aquí también.

- **Una sola convolución `k×k`**, sin bias, sin padding, sin ReLU, y la lectura C1 por esperanza
  bajo `softmax`. Un solo mapa: `tl` del máximo, `br` del mínimo. `β` aprendida, compartida.
- **Cinco brazos**, `k` ∈ {5, 7, 9, 11, 13}, 300 épocas, `lr` 0,05, lote 128, semilla 1.
- **Cabeza de 5 parámetros** (`β` + `a,b` por esquina).
- **Alternativas implementadas y sin armar**: `ant` (kernel antisimétrico) y el control
  `ind-tl`/`ind-br`. Armarlas es añadir su línea a `BRAZOS`.
- **Descartada por el dueño**: `rot` (girar la entrada 180°), con su código borrado a propósito.

```bash
.venv/bin/python 2026-09-07-esquinas-cualquiera/nn/modelo.py   # imprime la tabla y comprueba
```

⚠ **`k` = 13 no es un tope caprichoso**: la esquina se sortea entre los píxeles 8 y 23, y el mapa
de un kernel `k` representa de `(k−1)/2` a `31−(k−1)/2`. Este dataset admite hasta **k = 17** sin
regenerarlo; a partir de 19 hay esquinas que el mapa **no puede** señalar. Lo calcula el
manifiesto, no se supone.

## El dataset: el mismo fichero, no uno equivalente

```
foveal-vision-data/experimentos-cnn/esquinas300-32px-r4-r20260907/
    train.npz · val.npz · muestra.npz · muestras-congeladas.npz · manifiesto.json · README.md
```

Se resuelve con `expcnn.exigir_dataset(...)`, que es **la única** puerta: si no está publicado, el
entrenamiento **se niega antes de empezar** en vez de generarse uno equivalente. Un dato
re-derivado es el mismo *mientras nada cambie*, y «nada cambia» no es comprobable hacia el futuro.

Trae las **cuatro** esquinas etiquetadas, no sólo la diagonal `tl`/`br` que leía `esq-2d` — a
propósito, para que un experimento que mire otra combinación no tenga que re-rendir nada.

```bash
.venv/bin/python 2026-09-07-esquinas-cualquiera/nn/datos.py --comprobar   # instantáneo
```

## Cómo se corre — ⚠ NO lanzado todavía

```bash
cd ~/src/experimentos-cnn
E=2026-09-07-esquinas-cualquiera
.venv/bin/python $E/nn/entrenar_local.py --suelos                    # sin entrenar nada
$E/nn/lanzar_barrido.sh                                              # los 5 brazos (unidad systemd)
$E/nn/lanzar_barrido.sh --estado                                     # ¿viva? ¿por dónde? ¿NRestarts?
```

⚠ **`lanzar_barrido.sh` corre como unidad de systemd** (padre PID 1): sobrevive al fin del turno
que la lanzó y al reinicio del coordinador, y **se niega a lanzarse dos veces**. El estado se lee
del **disco**, no del log — `python` bufferiza cuando no es un tty, así que un log vacío no
significa que no haya arrancado.

✅ **La unidad se renombró a `esqcq-barrido`** al hacer la copia (venía como `esq2d-barrido`).
No es cosmética: el cerrojo de «no se lanza dos veces» compara contra ese nombre, así que con el
heredado esta copia y `esq-2d` se habrían bloqueado mutuamente — y el aviso de fin habría llegado
a Telegram firmado por el experimento equivocado.

## Cuánto costaría (estimado, heredado de `esq-2d`, no medido aquí)

**0 máquinas y 0 $** — entrena en este droplet. `esq-2d` midió **~20 min de reloj** para sus 5
brazos × 300 épocas, y **0,4 MB** en disco. ⚠ Es el coste de *aquel* montaje: si la pregunta nueva
cambia brazos o épocas, cambia.

El freno lo ve (`entrenar_local.py` está en la lista `TRABAJOS` de `cerrable.mjs`), así que un
entrenamiento vivo aparece en el veredicto que se lee desde el móvil.

## Los resultados de la base están en `esq-2d`, no aquí

Lo que aquel barrido midió —que `k11` y `k13` pasan el umbral, que `br` es **0/10 sobre páginas
enteras en los cinco kernels**, y que el eje **no está acotado por arriba**— se lee en
[`../2026-09-07-esquinas-diagonales/README.md`](../2026-09-07-esquinas-diagonales/README.md).
No se copia aquí: dos copias de un veredicto es como nacen las dos mitades desfasadas.
