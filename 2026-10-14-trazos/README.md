# `trazos` — recuperar los trazos del bolígrafo de un dígito estático

**Estado: preparado el 2026-10-13, NO ejecutado.** Encargo en `instrucciones/01-encargo.md`, criterio en
`instrucciones/02-criterio.md`, reglas en `REGLAS.md`. Lo primero que hay que hacer mañana, en este orden:

1. `nn/etiquetar.py`: elegir los 50 de val (5 por clase, semilla 0), pintarlos sobre rejilla y escribir
   `instrucciones/etiquetas-trazos.json` a mano (trazos, extremos, cruces) **antes** de correr el método. Commit.
2. `nn/sinteticos_trazos.py`: copiar `arco` y `recta` de `rizo-2/nn/sinteticos.py`; cada sintético guarda sus
   polilíneas, extremos y cruces. Calibrar con la semilla 101 los dos parámetros (factor de poda, ángulo de continuación).
3. `nn/trazos.py`: las cuatro etapas, las medidas A–D, figuras y `metricas.json`. Escribir aquí lo medido contra la
   predicción de abajo.

## Predicción (escrita antes de correrlo, 2026-10-13)

- Sintéticos: número de trazos ≥ 90 % con grosor 2–4 px; con 5–6 px la poda es la que decide y cae a 80–85 %.
- Dígitos reales: la cuenta coincide en 75–85 %; el fallo típico será el cruce del 8 con trazo grueso (la línea media
  deja un rombo de 2–3 px en vez de un punto, o sea dos nodos de grado 3) y las esquinas del 4 y el 7.
- Desplazamientos: plano (≥ 97 %): el método es local.

## Para qué sirve, si sale (lo que el dueño preguntó)

Un dígito como lista de trazos es el objeto de más alto nivel de la línea «todo objetos» de `rizo-2`:
- **Como features para el compositor**: por dígito, número de trazos, número de cruces, y por trazo (ordenados por
  longitud) longitud, extremos en 3×3, giro total con signo, grosor medio; ~30 números. Es otro grupo de «responsables».
- **Como estructura, sin compositor**: cada clase tiene pocas estructuras (0: un lazo cerrado; 1: un trazo recto; 8: un
  trazo con un cruce; 4: dos trazos con una T o un cruce…). La tabla C del criterio dice cuántas hay de verdad por
  clase en esta población; si son pocas, un dígito se puede clasificar por **la estructura más cercana** (distancia de
  edición sobre el grafo de trazos), y el responsable de un fallo es un trazo concreto: «sobró un cruce», «faltó un
  extremo».
- ⚠ Lo que NO da: el orden ni el sentido de escritura, ni distingue un trazo repasado dos veces. Y la misma clase se
  escribe con distintas estructuras, así que la estructura sola no identifica: reduce las clases posibles y nombra al
  responsable; el resto sigue siendo de los detalles.
