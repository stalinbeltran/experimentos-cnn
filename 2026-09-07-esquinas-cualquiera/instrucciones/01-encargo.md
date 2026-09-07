# El encargo, tal como llegó (2026-09-07)

> «Ahora vamos a hacer otro experimento, usando los mismos valores empleados en ese, pero ahora
> queremos que detecte 2 esquinas diagonales entre sí, la sup izq y la inf derecha. No la
> entrenes aún, sólo prepara todo y dame las estructuras. Igual debe ser un único kernel por
> cnn»

Tres cosas se leen literalmente y no se negocian:

1. **Los mismos valores que `esq-k`**: misma receta,
   misma semilla, misma ventana 32×32, misma reducción por 4, sin padding, sin bias, misma
   cabeza C1 (esperanza bajo `softmax(β·M)`), mismo `lr`, mismo lote, 300 épocas. **`k` se fija
   en 7**, que es el ganador de aquel barrido por su criterio congelado.
2. **Las dos esquinas son la superior-izquierda y la inferior-derecha**, o sea las de **una
   diagonal**. Las otras dos (tr, bl) pasan a ser el negativo duro.
3. **Un único kernel por CNN.** Lo que varía entre brazos es **cómo se lee** ese único mapa.

## Por qué esto no es «`esq-k` otra vez con otra etiqueta»

Un filtro lineal que pica en una esquina superior-izquierda tiene forma de **cuadrante**
—positivo donde espera tinta, negativo donde espera fondo—, y esa forma es **orientada**. La
esquina inferior-derecha es **esa misma forma girada 180°**. Con un solo kernel, las dos
esquinas no son dos problemas: son el mismo problema visto al revés.

Eso se formaliza partiendo el kernel bajo el giro:

```
W = S + A        S = (W + rot180(W))/2        A = (W − rot180(W))/2
respuesta_tl = <S,P> + <A,P>
respuesta_br = <S,P> − <A,P>
```

Las dos respuestas son **simétricas respecto de `<S,P>`**, y **toda la diferencia entre
esquinas vive en `A`**. De ahí salen, y no de una lluvia de ideas, las tres lecturas que se
prueban: por **signo** (si manda `A`), por **signo forzado** (con `S` = 0 por construcción), o
por **vista girada** (no compartir el mapa sino la entrada).

⚠ **Y una cuarta se descartó en papel, que es donde toca descartarla.** *«Las dos esquinas son
máximos y se distinguen por el valor»* (el caso «manda `S`») no es expresable con esta cabeza:
`existe` es una función lineal de un solo escalar, o sea **monótona**, y esa hipótesis pide fondo
0 · `br` medio · `tl` alto — una **banda**, que ninguna recta separa. Para expresarla harían falta
dos escalares (el máximo y el mínimo), y eso ya es `sig`. Pagar dos minutos de máquina por
demostrar un argumento de papel es justo lo que un tanteo no debe hacer.

## La medida que se hizo ANTES de diseñar nada, y que cambió el diseño

*Medida el 2026-09-07 sobre el kernel ganador de `esq-k`, que **sólo vio esquinas tl**.*

| | |
|---|--:|
| energía en la parte **simétrica** `S` | **74,0 %** |
| energía en la parte **antisimétrica** `A` | 26,0 % |
| suma del kernel | **−73,68** |
| respuesta a un parche ideal tl / br | −0,68 / −22,35 |

O sea que aquel kernel es **sobre todo un supresor de tinta**. Consecuencia, medida sobre las 10
páginas enteras de aquel experimento: **su mínimo cae en la mancha de tinta, no en la esquina br
— 0/10, mediana 91 px.**

```bash
# ⚠ el experimento se localiza por su ID, nunca por el nombre de su carpeta (R16):
#    renombrarla tiene que seguir siendo `git mv` y nada mas.
cd "$(python3 -c "from expcnn import por_id; print(por_id('esq-k').carpeta)")"
../.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "nn")
import numpy as np, torch
from transformacion import aplicar, cargar_kernel, paginas
W = cargar_kernel("k07"); off = (W.shape[0]-1)//2
R = W[::-1, ::-1]; S, A = (W+R)/2, (W-R)/2
print("S", (S**2).sum()/(W**2).sum(), "A", (A**2).sum()/(W**2).sum())
for vista, (x, y, w, h), s in paginas(10):
    m = aplicar(255 - vista)
    r, c = np.unravel_index(int(np.argmin(m)), m.shape)
    print(s, np.hypot(c+off-(x+w), r+off-(y+h)))
PY
```

**Qué cambia eso.** La estructura obvia —«el máximo es tl, el mínimo es br»— **no sale gratis de
lo ya entrenado**: pide un kernel que gaste mucho menos en `S`, y `S` es justamente lo que apaga
el interior del párrafo. **Ésa es la tensión que este experimento mide**, y por eso el brazo
`sig` no es una apuesta segura sino una hipótesis con un dato en contra.

## Lo que se decidió al diseñarlo, y por qué

- **El eje es la ESTRUCTURA DE LECTURA, no `k`.** `k` = 7 en los cuatro brazos que compiten.
  `sig11` (k = 11) es un **punto de seguro** declarado: sólo se lee si `sig` falla, y sirve para
  distinguir «esta lectura no puede» de «con 49 pesos no cabe». No es parte del eje.
- **`ant` existe para separar «esta lectura no funciona» de «el gradiente no llega hasta ella».**
  Si `sig` falla con `S` libre, no se puede saber si la lectura por signo es mala o si el óptimo
  cómodo (el supresor de tinta, medido arriba) se la come. Forzando `A` esa duda desaparece.
- **`ind-tl` / `ind-br` son el CONTROL, no candidatos.** Son dos redes de un kernel cada una, sin
  compartir nada: el **techo** contra el que se mide lo que cuesta compartir. Sin ese techo, un
  90 % no se puede leer (¿es el límite de compartir, o el de la tarea?).
- **La ventana no cambia (32×32), así que NUNCA contiene las dos esquinas**: el párrafo mide ≥ 64
  px reducidos. Es un límite real de este experimento y está comprobado y contado en el
  manifiesto, no supuesto. La pregunta que se contesta es *«¿puede un kernel servir a las dos
  esquinas?»*, no *«¿puede señalarlas a la vez en la misma vista?»* — eso es la página entera, y
  es el experimento siguiente.
- **Las dos esquinas pesan igual en la pérdida y el titular es la PEOR de las dos.** Un promedio
  escondería justo el desenlace más probable: que una estructura resuelva tl y no br.
- **10 muestras congeladas: 3 con tl, 3 con br y 4 negativas** (tr, bl, interior, fondo). Cinco
  positivas de la misma esquina verificarían la mitad del experimento, y sin tr/bl no se ve si el
  kernel confunde diagonales.
