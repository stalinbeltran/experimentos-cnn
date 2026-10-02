# `dim-nist` — reducir los dígitos de NIST de 8×8 a 4×4, ¿qué le hace a la generalización?

**Corrido el 2026-10-02 en el dev (0 $, ~20 min), CERRADO.** 30 brazos: `W ∈ {8, 7, 6, 5, 4}` × 5
semillas + el control `w8-de4` × 5. 180 imágenes para entrenar, 1617 para validar. Criterio
escrito antes ([`instrucciones/02-criterio.md`](instrucciones/02-criterio.md)), aplicado tal cual
por `nn/informe.py` ([`resultados/RESULTADOS.md`](resultados/RESULTADOS.md)).

> **Veredicto:** aquí **la mejor resolución es la original, 8 px** (exactitud 0,851), y **5 y 6 px
> son indistinguibles de ella** (0,829 y 0,842): el **W mínimo suficiente es 5 px**. **La
> resolución NO estorba** y la brecha no crece con ella. Bajar a **4 px sí cuesta** (0,714), y es
> **falta de información**, no de generalización: el control lo confirma. Es lo contrario de
> `dim-gen`, como se escribió antes de mirar: allí 128 px sobraban, aquí 8 px ya son pocos.

| brazo | W | parámetros | exactitud val | exactitud train | CE train | CE val | clasificación |
|---|---:|---:|---|---:|---:|---:|---|
| **`w8`** | 8 | 1.258 | **0,851 ± 0,025** | 1,000 | 0,004 | 0,928 | **W\*** |
| `w7` | 7 | 1.258 | 0,765 ± 0,087 | 0,980 | 0,089 | 1,614 | (b) peor generalización |
| `w6` | 6 | 754 | 0,842 ± 0,039 | 0,986 | 0,076 | 0,727 | suficiente |
| `w5` | 5 | 754 | 0,829 ± 0,056 | 0,996 | 0,027 | 1,059 | **suficiente (el mínimo)** |
| `w4` | 4 | 394 | 0,714 ± 0,022 | 0,830 | 0,488 | 0,911 | **(a) menos información** |
| `w8-de4` (control) | 8 | 1.258 | 0,718 ± 0,110 | 0,891 | 0,299 | 1,116 | red de 8, información de 4 |

Piso 0,1 (azar). Ninguna semilla atascada. Umbral `max(2·SE_dif, 0,01)`.

## Lo que se lee, con el criterio

- **4 px**: la exactitud de train cae a 0,83 (entropía cruzada 0,49 contra 0,004 a 8 px) y la
  brecha no crece: la red ya no puede ni ajustar las 180 de train. **Es (a), información.**
- **El control** (la red de 8 px con la imagen de 4) da 0,718, igual que `w4` (0,714): **los
  parámetros no explican la caída de 8 a 4; es información.** Y con esa información la red grande
  memoriza más (brecha de entropía cruzada +0,39 sobre `w4`, umbral 0,28).
- **`w7` sale peor que 6 y 5**, con la mayor dispersión (± 0,087) y la mayor CE de validación. Es
  el **zigzag por paridad** que el criterio anotó como riesgo: a 7 px el kernel es 4 (f = 0,57) y
  el mapa final 1×1. Leído **sólo por pares** —la regla exacta de `dim-gen`— la curva es plana
  8 ≈ 6 y cae en 4. No es la resolución.

## Lo que NO dice

- Son **13 escritores** compartidos entre train y val: generalizar a dígitos nuevos de los mismos
  escritores, no a escritores nuevos.
- **8 px ya es /4 del bitmap de NIST** (32×32, no incluido en scikit-learn): no se pudo mirar por
  encima. La pregunta «¿estorba la resolución?» sólo se contesta dentro de 4–8 px.
- La exactitud de train satura en 1,0 de 5 px para arriba; la descomposición se leyó en entropía
  cruzada, como fijó el criterio.

## Coste, reloj y dónde está

- **0 $**, una unidad de systemd en el dev. El primer lanzamiento se negó en los 30 brazos porque
  4000 pasos no son épocas enteras de 9; se fijó 3996 (444 épocas, lo del ensayo) antes de que
  nada entrenara, enmienda fechada en el criterio.
- Métricas, resúmenes, logs, informe y figuras: aquí (rama `tema-2`). **Pesos y todo lo demás**:
  en el almacén, `foveal-vision-data/experimentos-cnn-resultados/dim-nist/`.
- Reporte #25 en `estudios-redes-neuronales`.
