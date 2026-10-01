# Reglas de `bor-pca`

**Escritas el 2026-10-01** al montar la carpeta (fase 0d del plan
[`docs/plan-kernels-banco-2026-10-01.md`](../docs/plan-kernels-banco-2026-10-01.md), E3).
Estado **`abierto`**: código comprobado; nada calculado todavía.

⚠ **Estas reglas son de este experimento y de ninguno más.** No se copió de ningún experimento
(el `nn/datos.py` de aquí es propio: parches, no ventanas).

**Qué pregunta:** ¿cuál es la dirección de mayor varianza de los parches de borde de párrafo a
la escala del banco, y sirve como kernel?

## Entradas

- **Dataset: `parrafos1000-pagina1024-r4-r20261001`** (`bor-p4`), leído con
  `expcnn.exigir_dataset(...)`.
- **Qué se lee:** sólo **`train`** (por página, del dataset). Por cada párrafo y cada uno de sus
  4 bordes, **8 puntos** sorteados uniformes a lo largo del borde (semilla 0); el parche es el
  `k × k` centrado en el píxel de **tinta** junto al borde (izq/sup: el primero de tinta;
  der/inf: el último), para que los cuatro bordes queden igual de centrados.
- **Limpios por construcción**: un parche de `k ≤ 19` cabe en la ventana limpia de 79 que
  garantiza `bor-p4` (se comprueba con un `assert`).
- **Qué se normaliza:** `x = 1 − suma/4080`; se resta **la media de todos los parches** (PCA
  de libro), no la de cada parche.

## Salidas

- **Pesos:** `nn/pesos/kNN/best.pt` con el **PC1** como `conv.weight` `(1, 1, k, k)` (lo que
  importa `importar_kernel.py --de bor-pca`), el **PC2** en `pc2` y la varianza explicada.
- **`resultados/pca.json`** (varianza de PC1 y PC2, suma y huella del PC1 por `k`) y
  **`resultados/componentes.png`** (PC1 arriba, PC2 abajo, cada uno a su rango).
- **Qué se commitea:** todo lo de arriba (nueve `.pt` diminutos).

## Procesos

1. `python nn/pca.py` — los nueve `k`, en segundos.
2. **Al banco entra el PC1 de cada `k`, y sólo el PC1**, decidido antes de mirar
   (`instrucciones/02-criterio.md`).

- **Qué se mide:** la fracción de varianza de cada componente. No hay «aprendió».
- **Brazos:** `k ∈ {3, …, 19}`, **sin semillas**: es determinista.
- **«Ganar»:** no aplica. La utilidad la dice el banco.

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/datos.py` | los parches de `train`, por `k` | `python nn/datos.py` (cuántos por `k` y partición) |
| `nn/pca.py` | PC1/PC2 de los nueve `k` + figura | `python nn/pca.py` · `--comprobar` (recupera una dirección sembrada) |

- **Dependencias:** el `.venv` de la raíz (numpy; torch sólo para escribir el `.pt`).
- **El signo** de cada componente se fija (la entrada de mayor `|valor|` sale positiva): el
  banco normaliza norma y estandariza la salida, así que no cambia nada allí; sólo hace que dos
  corridas den el mismo fichero.

## Qué NO hereda

- **Se copió de:** ningún experimento.
- **Contra qué se compara:** en el banco, con los clásicos (`gauss`, `sobel`) y con el aleatorio
  de igual norma y `k`. Aquí, con nada.
- **Restricciones de otros experimentos que NO aplican aquí:** ninguna conocida. La reserva §3.7
  la cumple el dataset.
