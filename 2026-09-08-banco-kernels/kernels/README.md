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
