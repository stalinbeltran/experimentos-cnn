# Kernels de `banco-k`

Los **controles** del §10, generados por `nn/kernels.py` con `k=9` (el medio del
rango 3..19 del contrato §5).

| kernel | qué contesta | suma |
|---|---|---|
| `aleatorio-r0..r9` | ¿el mérito es de *este* kernel, o de **filtrar cualquier cosa**? (§10.2) | ~0 |
| `gauss` | referencia clásica de **suma positiva** | > 0 |
| `sobel` | referencia clásica de **suma cero** (derivada de gaussiana en x) | ~0 |

⚠ **La identidad y la caja media no tienen `.npy`**: la identidad es un **recorte
puro** (§6.3) y la caja media es un **predictor constante** que ni siquiera mira la
imagen (§10.1).

⚠ **La norma no hace falta igualarla aquí**: el banco normaliza en L2 al recibir el
kernel (§5.4), así que «igual norma» está garantizado. Lo que sí hay que construir es
«mismo `k`».

⚠ **Aquí no hay ningún kernel *evaluado* todavía**, y es correcto: el banco es
agnóstico al origen del kernel (§1) y los métodos para obtenerlos están fuera de
alcance (§15). Con estos controles el banco se calibra y se valida entero.

## ⚠ Reserva contra fuga de distribución (§3.7)

Un kernel que se obtenga con el mismo generador tiene **fuga aunque las muestras sean
distintas**. Por eso el dataset reserva **`LiberationMono`** y el interlineado
`[1.45, 1.60]` para uso **exclusivo del banco**: un procedimiento que produzca
kernels **no puede usarlos**.

## Dónde están, y cómo se sabe que son los que se evaluaron

**Los 38 `.npy` están commiteados y empujados** (21 KB en total). `*.npy` **no** está en el
`.gitignore` del repo —sí lo están `*.npz` y `*.npy.gz`—, así que viajan con el repo a
cualquier máquina nueva.

⚠ **Y no basta con que el fichero exista: tiene que ser el MISMO fichero.** Cada
`resultados/<kernel>/criterios.json` guarda el **sha256 del `.npy` antes de normalizar**
(§13.3). *Comprobado el 2026-09-08 leyendo los blobs de `origin/main` —el remoto, no el
disco— y comparando: **4/4 coinciden**.* Se rehace así:

```bash
python nn/evaluar_kernel.py --contrato kernels/<nombre>.npy   # imprime su sha256_16
# y se compara con el "contrato.sha256_16" de resultados/<nombre>/criterios.json
```

## Dos familias, y se conservan por razones distintas

| | de dónde sale | por qué se guarda |
|---|---|---|
| **controles** `aleatorio-*` · `gauss` · `sobel` · `laplaciano` | `nn/kernels.py`, con semilla fija | **se regeneran del código**. El `.npy` es una comodidad |
| **evaluados** `esqk-k03..k11` | los pesos aprendidos de `esq-k` | **NO se regeneran desde este experimento**: dependen de un entrenamiento ajeno. Por eso llevan su `<nombre>.json` con origen, brazo, norma original y hash |

⚠ **`laplaciano` estuvo en el lado equivocado hasta el 2026-09-08.** Se evaluó con un script
suelto tecleado en una sesión, así que su `.npy` estaba en git pero **la receta que lo produjo
no**: un dato huérfano, que se iba con la máquina. Ahora `nn/kernels.py` lo genera, y
*comprobado: el regenerado da el mismo `sha256_16` (`95a2da9a48f8211b`) que el evaluado.*

⚠ **Los `esqk-*` dependen de que `esq-k` siga en el repo.** Sus pesos son el único eslabón que
no se regenera. *Comprobado el 2026-09-08: los tres evaluados están en git, **cargan**, y el
`.npy` es bit a bit el `conv.weight` de su checkpoint.*
