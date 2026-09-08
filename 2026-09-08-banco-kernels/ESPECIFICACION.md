# Banco de evaluación de kernels

**Especificación técnica — v1.0**
Fecha: 2026-09-07

---

## 1. Propósito y alcance

Este documento especifica un **banco de pruebas** que mide cuánto contribuye un kernel de convolución, aplicado como preprocesamiento de las entradas, a la **generalización** de una CNN de referencia entrenada en régimen de datos escasos.

El banco es **agnóstico al origen del kernel**. El kernel entra al sistema como dato, no como parte del diseño experimental. Los procedimientos que producen kernels (autoencoders dispersos, sondas, búsquedas de arquitectura u otros) quedan **fuera del alcance** de este documento.

### 1.1 Qué es y qué no es

| Es | No es |
|---|---|
| Un instrumento de medición reutilizable | Un experimento particular |
| Evaluación de kernels ya obtenidos | Un método para obtener kernels |
| Comparación entre condiciones de preprocesamiento | Optimización de la CNN de referencia |
| Medición de generalización | Medición de eficiencia por parámetro |

### 1.2 Principio de diseño

Toda decisión de este banco sigue un criterio único: **maximizar la sensibilidad de la medición al kernel**. Cualquier elemento que dé capacidad, holgura o ayuda a la CNN de referencia reduce la evidencia que el banco puede producir, y por tanto se elimina o se minimiza deliberadamente.

---

## 2. Hipótesis y criterios de éxito

Los criterios se declaran **antes de ejecutar** cualquier corrida y no se modifican en función de los resultados obtenidos.

### 2.1 Criterio primario — utilidad

> Un kernel se considera **útil** si su IoU medio sobre `eval` en la época 200 supera al del kernel aleatorio de igual norma y mismo `k` por un margen mayor que la suma de las desviaciones estándar de ambas condiciones entre semillas.

### 2.2 Criterio secundario — generalización

> Un kernel muestra **efecto de generalización** si, además de cumplir 2.1, su brecha `IoU_train − IoU_eval` en la época 200 es menor que la de la condición identidad, bajo el mismo margen.

### 2.3 Interpretación

Un kernel puede mejorar el desempeño en `eval` por dos mecanismos distintos:

- **Facilitación:** hace el problema más fácil. Suben `train` y `eval` juntos, la brecha se mantiene.
- **Transferencia:** produce una representación más generalizable. La brecha se reduce.

Solo el segundo mecanismo corresponde al objetivo de este banco. Ambos criterios deben reportarse por separado. Un kernel que cumple 2.1 y falla 2.2 es un resultado válido y debe registrarse como tal.

---

## 3. Generación del dataset

### 3.1 Parámetros

| Parámetro | Valor |
|---|---|
| Contenido | Imágenes de párrafo, generador propio |
| Formato | Escala de grises, 1 canal |
| Fondo | Blanco uniforme |
| Resolución de generación | 584 × 584 px |
| Reducción | Promedio por área, factor /4 |
| Resolución tras reducción | 146 × 146 px |
| Muestras totales | 1000 |

### 3.2 Justificación del factor /4

La reducción evita el detalle tipográfico excesivo, pero **reducir es aplicar un kernel pasa-bajos fijo antes del kernel variable**. Un factor agresivo (/8) convierte el párrafo en un rectángulo gris casi uniforme: en esa representación ningún kernel puede ayudar ni estorbar de forma apreciable, todas las condiciones convergen al mismo resultado y el banco pierde sensibilidad.

El factor /4 conserva el texto como textura (altura de línea aproximada de 10 px en el marco reducido), que es la escala donde un detector de bordes o de trazo tiene información que explotar.

### 3.3 Restricción de colocación — OBLIGATORIA

El pipeline descarta 9 px por lado (§5). Un párrafo colocado cerca del borde puede quedar con uno de sus bordes fuera del marco final, en cuyo caso la coordenada verdadera es inalcanzable para el soft-argmax y esa muestra aporta un error irreducible que contamina la métrica.

**El generador debe garantizar que la caja del párrafo quede contenida en el rango [68, 512] en ambos ejes del marco de 584 px.** Esto deja aproximadamente 8 px de holgura en el marco final de 128.

Esta restricción debe implementarse como **aserción en el código de generación**, no como supuesto. Con 1000 muestras, 20 mal colocadas ya desplazan el IoU medio de forma perceptible.

### 3.4 Etiquetas

Cuatro coordenadas enteras en el marco de 584 px: `borde_izq`, `borde_der`, `borde_sup`, `borde_inf`.

---

## 4. Particiones

| Partición | Muestras | Función |
|---|---|---|
| `train` | 100 | Entrenamiento |
| `monitor` | 100 | Diagnóstico, sin efecto sobre el entrenamiento |
| `eval` | 800 | Métrica titular |

### 4.1 Reglas

- Las particiones son **fijas** y se generan una sola vez. Se versionan junto con el dataset.
- El muestreo es **estratificado** según las características que el generador varíe (tamaño de párrafo, densidad, posición). La estratificación evita que las 100 muestras de `train` queden sesgadas hacia una región del espacio de generación.
- `monitor` **no** se usa para parada temprana, selección de modelo ni ajuste de hiperparámetros. No hay validación en este banco: las épocas son fijas y no hay búsqueda. Su única función es diagnóstico.
- La proporción invertida (80 % en evaluación) es deliberada: reduce la varianza de la métrica titular y endurece el régimen de entrenamiento, que es donde el aporte del kernel resulta visible.

### 4.2 Aplicación del kernel

**El kernel se aplica idénticamente a las tres particiones.** Aplicarlo solo a `train` produciría desajuste de distribución entre entrenamiento y evaluación, y la medición resultante no correspondería a la hipótesis.

---

## 5. Contrato de kernel

Define qué constituye una entrada válida al banco. Cualquier kernel que cumpla el contrato es evaluable, independientemente de su procedencia.

### 5.1 Formato

| Requisito | Valor |
|---|---|
| Archivo | `.npy` |
| Forma | `(k, k)` |
| Tipo | `float32` |
| `k` | Impar, 3 ≤ k ≤ 19 |
| Canales | 1 entrada → 1 salida |

### 5.2 K_max = 19, congelado

`K_max` determina el descarte fijo y el tamaño de entrada de la CNN. **Modificarlo invalida la comparabilidad con toda medición previa.** Si en el futuro se requiere `k > 19`, no se eleva `K_max`: se construye un banco nuevo con su propia serie de resultados.

### 5.3 `k` impar, obligatorio

Con `k` par el centro del kernel cae entre píxeles y el recorte queda asimétrico por medio píxel, introduciendo un desplazamiento sistemático de ½ px en las coordenadas. El desplazamiento es pequeño pero **difiere entre condiciones**, que es precisamente el tipo de confusión que el banco debe eliminar.

### 5.4 Normalización de norma

Al recibir el kernel, el banco aplica:

```python
k_norm = k / np.sqrt(np.sum(k**2))
```

**Motivo.** Un kernel multiplicado por un escalar es el mismo detector —responde a la misma estructura— pero produce salidas de magnitud distinta, lo que altera la escala de activaciones, la magnitud de los gradientes y la velocidad efectiva de aprendizaje. Con épocas fijas, un kernel de salida grande puede parecer mejor solo por haber avanzado más rápido. Como los kernels provienen de experimentos heterogéneos, llegarán con escalas arbitrarias.

La normalización deja como única diferencia entre kernels su **forma**.

---

## 6. Pipeline de aplicación

### 6.1 Orden de operaciones

```
584×584  →  reducir /4  →  146×146  →  conv 'valid' con k  →  recorte central  →  128×128
```

**El kernel se aplica después de reducir.** Dos razones: los kernels provienen de experimentos sobre ventanas pequeñas, de modo que la escala es coherente; y aplicar el kernel a 584 px para reducir después haría que el promedio por área borrara la respuesta del kernel.

### 6.2 Recorte a tamaño común

El descarte total es **siempre 9 px por lado**, independiente de `k`. Un kernel pequeño convoluciona menos y recorta más, compensando exactamente.

| `k` | `valid` produce | Recorte por lado | conv + recorte |
|---|---|---|---|
| 3 | 144 | 8 | 1 + 8 = 9 |
| 5 | 142 | 7 | 2 + 7 = 9 |
| 7 | 140 | 6 | 3 + 6 = 9 |
| 9 | 138 | 5 | 4 + 5 = 9 |
| 11 | 136 | 4 | 5 + 4 = 9 |
| 13 | 134 | 3 | 6 + 3 = 9 |
| 15 | 132 | 2 | 7 + 2 = 9 |
| 17 | 130 | 1 | 8 + 1 = 9 |
| 19 | 128 | 0 | 9 + 0 = 9 |
| identidad | 146 | 9 | 0 + 9 = 9 |

**Motivo.** Si cada `k` determinara su propio tamaño de salida, las condiciones tendrían distinta forma de entrada y distinto campo de visión sobre el párrafo, y la diferencia de desempeño mezclaría calidad del kernel con cantidad de imagen observada. Con recorte fijo, **todas las condiciones ven exactamente los mismos píxeles del original**.

Este esquema elimina además por completo el relleno de bordes: no hay padding, no hay costura, no hay artefacto dependiente del kernel.

### 6.3 Condición identidad

Se implementa como recorte central puro de 146 → 128. **No** como convolución con delta de Dirac, para evitar cualquier diferencia numérica atribuible a la operación de convolución.

### 6.4 Transformación de etiquetas

```
coord_final = (coord_584 / 4) - 9
```

Omitir el desplazamiento de 9 px produce un sesgo constante en todas las condiciones por igual, lo que lo hace difícil de detectar: todos los kernels aparecerían uniformemente malos sin señal de la causa. Debe verificarse con una aserción sobre el dataset construido.

### 6.5 Estandarización de salida

Tras la convolución y el recorte se aplica estandarización por dataset:

```python
mu    = media(train_filtrado)
sigma = desviacion(train_filtrado)
x_std = (x - mu) / sigma
```

Los estadísticos se calculan **exclusivamente sobre `train` filtrado** y se aplican a las tres particiones.

**Motivo.** La norma L2 iguala la energía del kernel pero no la estadística de su salida. Un kernel de suma cero (tipo Sobel) produce salidas centradas en cero; uno de suma positiva (tipo Gauss) produce salidas con media alta heredada del fondo blanco. Sin estandarización, esa diferencia de distribución se contabilizaría como calidad del kernel.

Este paso es un **control**, no una variable del experimento. No se ejecutan condiciones con y sin estandarización.

---

## 7. CNN de referencia

### 7.1 Arquitectura

```
Entrada                        128×128×1
Conv 5×5, 8 canales,  stride 2, ReLU    →  64×64×8
Conv 5×5, 16 canales, stride 2, ReLU    →  32×32×16
Conv 3×3, 16 canales, stride 2, ReLU    →  16×16×16
Conv 1×1, 4 canales                     →  16×16×4
soft-argmax sobre marginales 1D         →  4 coordenadas
```

Total: **5.812 parámetros**. La convolución 1×1 final constituye la cabeza completa: **68 parámetros**.

Todas las convoluciones son `valid` con las dimensiones indicadas. La arquitectura es **fija e inmutable** para todas las condiciones.

### 7.2 Cabeza mínima, deliberada

Una cabeza densa de gran tamaño puede **compensar un kernel deficiente**, comprimiendo las diferencias entre condiciones hasta hacerlas indistinguibles del ruido entre semillas. Con 100 muestras de entrenamiento este efecto es severo.

La cabeza de 68 parámetros es deliberadamente incapaz de esa compensación, lo que traslada la carga de la representación al kernel y maximiza la sensibilidad del banco.

### 7.3 Salida — soft-argmax sobre marginales 1D

Los 4 mapas de 16×16 se procesan así:

| Canal | Reducción | Eje | Salida |
|---|---|---|---|
| 0 | suma sobre filas | columnas | borde izquierdo |
| 1 | suma sobre filas | columnas | borde derecho |
| 2 | suma sobre columnas | filas | borde superior |
| 3 | suma sobre columnas | filas | borde inferior |

Para cada marginal de 16 elementos:

```
p     = softmax(marginal / tau)
indice = sum(p * arange(16))
coord  = indice / 15          # normalizada a [0,1]
```

Temperatura `tau = 1.0`, fija e idéntica para todas las condiciones.

### 7.4 Un canal por borde — restricción crítica

**No debe usarse un único mapa de columnas para los bordes izquierdo y derecho.** La distribución resultante sería bimodal y el soft-argmax devolvería la esperanza de ambos modos, es decir el centro del párrafo, colapsando ambas predicciones al mismo valor. Un canal dedicado por borde elimina el problema.

### 7.5 Resolución de coordenada

La rejilla de 16×16 corresponde a 8 px por celda en el marco de 128. El soft-argmax interpola entre celdas y alcanza en la práctica 1–2 px.

**Diagnóstico:** si en calibración el error se estanca cerca de 8 px, eliminar el stride de la tercera convolución (pasando a 32×32) **antes** de modificar cualquier otro elemento de la arquitectura.

---

## 8. Protocolo de entrenamiento

### 8.1 Parámetros

| Parámetro | Valor | Justificación |
|---|---|---|
| Pérdida | L1 sobre coordenadas normalizadas a [0,1] | L2 castiga cuadráticamente los valores atípicos; con n=100 unas pocas muestras dominarían el gradiente y aumentarían la varianza entre semillas |
| Optimizador | Adam | |
| Learning rate | 3e-4, **constante** | Un schedule interactúa con el condicionamiento que induce cada kernel y confunde la comparación |
| Batch size | 20 | 5 pasos por época |
| Épocas | 200 (1000 pasos) | Fijas |
| Aumento de datos | **Ninguno** | §8.3 |
| Parada temprana | **Ninguna** | §8.4 |
| Semillas | ≥5, idénticas en todas las condiciones | §8.5 |
| Paradas de evaluación | 25, 50, 100, 200 | |

### 8.2 Invariancia entre condiciones

Semillas, arquitectura, hiperparámetros, particiones y orden de los lotes son **idénticos en todas las condiciones**. La única variable es el kernel aplicado a las entradas.

### 8.3 Ausencia de aumento de datos

El aumento de datos es en sí mismo un mecanismo de generalización, y uno potente. Incluirlo lo haría competir con el kernel por el mismo efecto, comprimiendo las diferencias entre condiciones.

El régimen de 100 muestras sin aumento es **artificialmente duro a propósito**: es el régimen donde el aporte del kernel resulta medible.

### 8.4 Ausencia de parada temprana

Con 100 muestras la red memorizará el conjunto de entrenamiento. Eso es lo buscado: el banco necesita **observar el régimen de sobreajuste**, no evitarlo, porque la brecha `train − eval` en ese régimen es la evidencia directa del criterio 2.2.

### 8.5 Número de semillas

Con n=100 la desviación del IoU entre semillas puede situarse en el rango 0.03–0.05, lo que vuelve exigente el margen de los criterios de éxito. El número definitivo se fija en la calibración (§11), no por defecto.

---

## 9. Métricas y registro

### 9.1 Métricas

| Métrica | Definición | Uso |
|---|---|---|
| **IoU** | Intersección sobre unión de la caja predicha contra la real | Titular |
| **MAE por borde** | Error absoluto medio en píxeles, desglosado en los 4 bordes | Diagnóstico |
| **Brecha** | `IoU_train − IoU_eval` | Criterio 2.2 |

### 9.2 Registro obligatorio

En **cada** parada de evaluación (25, 50, 100, 200) y para **cada** semilla se registra:

- IoU sobre `train`
- IoU sobre `monitor`
- IoU sobre `eval`
- Brecha `train − eval`
- MAE por borde sobre `eval`

**El registro de `train` no es opcional.** Sin él es imposible distinguir facilitación de transferencia (§2.3), que es la distinción central de este banco.

### 9.3 Reporte

Toda cifra se reporta como **media ± desviación estándar entre semillas**. Una cifra de una sola semilla no constituye resultado en este banco.

---

## 10. Condiciones de control

Los controles son lo que convierte una cifra en evidencia. Se ejecutan con el mismo protocolo que las condiciones experimentales.

| Control | Pregunta que responde | Notas |
|---|---|---|
| **Caja media** | ¿El problema es difícil? | Predictor constante: media de las 4 coordenadas de `train`, ignora la imagen |
| **Identidad** | ¿El kernel aporta algo? | Línea base real |
| **Aleatorio** | ¿El mérito es de *este* kernel o de filtrar cualquier cosa? | Igual norma y mismo `k`; múltiples semillas de aleatoriedad |
| **Gauss** | Referencia clásica | Opcional |
| **Sobel** | Referencia clásica | Opcional |

### 10.1 Caja media — se ejecuta primero

Debe ejecutarse **antes que cualquier kernel**. Si el generador coloca los párrafos con poca variabilidad, este predictor trivial puede alcanzar un IoU alto, en cuyo caso todas las condiciones quedarán comprimidas en un rango estrecho y el banco no discriminará nada.

**Si el control de caja media resulta alto, se corrige el generador. No se continúa con el experimento.**

### 10.2 Kernel aleatorio — control indispensable

Es posible que convolucionar con ruido mejore el resultado, actuando como perturbación o decorrelación de la entrada. En ese caso un kernel aprendido podría superar a la identidad por una razón ajena a lo que aprendió.

Requiere **varias semillas de aleatoriedad**, no una: su desempeño también varía.

### 10.3 Lectura conjunta

La afirmación que el banco permite sostener tiene esta forma:

```
kernel evaluado  >  aleatorio  >  identidad  >  caja media
```

Cada `>` debe superar la desviación entre semillas, o no es un `>`.

Si el kernel evaluado supera a la identidad pero **no** al aleatorio, la conclusión correcta no es «el kernel funciona» sino «filtrar funciona». Son afirmaciones distintas y solo el control permite separarlas.

---

## 11. Calibración previa del banco

Procedimiento de puesta a punto del instrumento. **No es un experimento** y sus resultados no se reportan como hallazgos.

1. Ejecutar el control de **caja media**. Verificar que su IoU es suficientemente bajo para dejar margen de discriminación. Si no lo es, corregir el generador y repetir.
2. Ejecutar **identidad** y **aleatorio** con **10 semillas**.
3. Medir la desviación estándar real del IoU en ambas.
4. Fijar el número definitivo de semillas para todas las condiciones a partir de ese dato.
5. Verificar la resolución de coordenada (§7.5). Si el MAE se estanca cerca de 8 px, aplicar el ajuste indicado y reiniciar la calibración.
6. Verificar mediante aserción que ninguna caja de párrafo queda fuera del marco final (§3.3).
7. Verificar mediante aserción la transformación de coordenadas (§6.4) sobre una muestra conocida.

Concluida la calibración, los parámetros quedan congelados.

---

## 12. Invariantes

Elementos que **no** se modifican una vez fijados. Alterar cualquiera invalida la comparabilidad con todas las mediciones previas y obliga a construir un banco nuevo.

- `K_max = 19` y el descarte de 9 px por lado
- Resolución de entrada 128 × 128 × 1
- Arquitectura de la CNN de referencia y su recuento de parámetros
- Particiones y su contenido
- Conjunto de semillas
- Hiperparámetros de entrenamiento y número de épocas
- Definición de las métricas
- Criterios de éxito (§2)

---

## 13. Artefactos

### 13.1 Entradas

```
kernels/<nombre>.npy          # contrato §5
dataset/parrafos_v1/          # imágenes 584×584 + etiquetas
dataset/parrafos_v1/splits.json
```

### 13.2 Salidas por condición

```
resultados/<condicion>/
  config.json          # kernel, k, norma original, semillas, hash del dataset
  metricas.csv         # semilla × parada × métrica
  resumen.json         # media ± desviación por parada
  criterios.json       # evaluación de §2.1 y §2.2
```

### 13.3 Trazabilidad

Cada corrida registra el hash del dataset, el hash del kernel recibido antes de normalizar, y la versión de esta especificación.

---

## 14. Trampas conocidas

Registro de errores que producen resultados silenciosamente inválidos.

| Trampa | Síntoma | Prevención |
|---|---|---|
| Olvidar el desplazamiento de 9 px | Todas las condiciones uniformemente malas, sin señal de la causa | Aserción §6.4 |
| Aplicar el kernel solo a `train` | Desajuste de distribución; resultados sin relación con la hipótesis | §4.2 |
| Párrafos fuera del marco final | Error irreducible en un subconjunto; IoU medio desplazado | Aserción §3.3 |
| Un solo mapa de columnas para izq/der | Ambas predicciones colapsan al centro | §7.4 |
| Reportar con una sola semilla | Diferencias indistinguibles del ruido | §9.3 |
| Omitir el control aleatorio | Se atribuye al kernel un efecto que produce cualquier filtrado | §10.2 |
| No registrar `train` | Facilitación indistinguible de transferencia | §9.2 |
| Cabeza grande | Diferencias entre condiciones comprimidas | §7.2 |
| Reducción /8 | Todas las condiciones convergen; banco insensible | §3.2 |

---

## 15. Fuera de alcance

- Métodos de obtención de kernels
- Optimización de la CNN de referencia
- Bancos de múltiples kernels o salidas multicanal (el contrato §5.1 admite un kernel de un canal)
- Comparación de eficiencia por parámetro entre kernels de distinto `k`. El banco mide **el kernel tal como llega**; comparar `k=3` contra `k=19` compara 9 contra 361 parámetros, lo cual es legítimo bajo el objetivo declarado pero no constituye una comparación de eficiencia.
- Aumento de datos, regularización explícita y schedules de aprendizaje

---

## Anexo A — Resumen de parámetros

| Parámetro | Valor |
|---|---|
| Resolución de generación | 584 × 584 |
| Factor de reducción | /4 (promedio por área) |
| Resolución tras reducción | 146 × 146 |
| K_max | 19 |
| Descarte por lado | 9 px |
| Entrada de la CNN | 128 × 128 × 1 |
| Rango de colocación (marco 584) | [68, 512] |
| Transformación de etiquetas | `/4` luego `−9` |
| Muestras totales | 1000 |
| Particiones | 100 / 100 / 800 |
| Parámetros de la CNN | 5.812 |
| Parámetros de la cabeza | 68 |
| Pérdida | L1 sobre [0,1] |
| Optimizador | Adam, lr 3e-4 constante |
| Batch size | 20 |
| Épocas | 200 |
| Paradas de evaluación | 25 / 50 / 100 / 200 |
| Temperatura soft-argmax | 1.0 |
| Semillas | ≥5, definitivo tras calibración |
