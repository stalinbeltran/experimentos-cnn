# Criterio — escrito el 2026-10-01, ANTES de medir los suelos y ANTES de entrenar

Lo que diga aquí manda sobre lo que se vea después (R13).

## Qué se entrena

- **27 brazos**: `k ∈ {3, 5, 7, 9, 11, 13, 15, 17, 19}` × semillas `{1, 2, 3}`
  (`k09-s2`, …). Las tres semillas cambian **la inicialización y el orden de los lotes**; las
  ventanas son las mismas (`SEMILLA_VENTANAS = 0`), así que la dispersión entre semillas es
  la del **productor**, no la del dato.
- 300 épocas, Adam `lr` 0,02, lote 128, sin parada temprana ni aumento de datos.
  `best.pt` = la época de mínima pérdida de `val`; `last.pt` = la 300.
- `LAMBDA_COORD` = el que iguala `bce` y `l1` en la red **sin entrenar** (mediana de los nueve
  `k`), medido con `--suelos` y **congelado antes de la primera época**. La regla que se
  fija es «los dos términos parten iguales», no el número.

## Qué se mide, y dónde

Por **borde** —`izq`, `der`, `sup`, `inf`— y **nunca el promedio**, sobre `eval`, con
`best.pt`:

| medida | qué es |
|---|---|
| `err_t` | error medio en px de la coordenada, en las ventanas donde el borde **existe** |
| `a2_t` | fracción de esas con error ≤ 2 px |
| `f1_t` | f1 del `existe` del borde |

**El suelo** de cada borde es el **predictor trivial**: siempre la prevalencia de ese borde
en `train` y su coordenada **mediana** en `train`. Se mide con `--suelos` y se escribe en
`REGLAS.md` **antes** de la primera época.

## Qué se llama «aprendió»

Un brazo **aprendió el borde `t`** si en `eval`

    media(e_suelo − e_brazo)  >  2 · SE

sobre las ventanas de `eval` donde el borde existe, con `e` el error absoluto de cada
ventana y `SE` el error estándar de esa **diferencia pareada** (las mismas ventanas para los
dos predictores). Se dice **por borde**: un brazo puede aprender `izq`/`der` y no `sup`/`inf`.

**Por `k`**, con tres semillas: el borde `t` está aprendido en `k` si lo aprenden **las tres**.
Dos de tres se reporta como «inestable», y no se redondea a nada.

## Qué NO decide este criterio

- **Si el kernel es útil.** Eso lo dice el banco (§2.1 de su especificación: IoU en `eval`
  contra el aleatorio de igual norma y `k`). Aquí sólo se dice si el productor aprendió a
  leer bordes en **su** tarea. Los **27** kernels se importan al banco **todos**, hayan
  aprendido o no: el banco es agnóstico al origen, y filtrar aquí sería decidir por él.
- **Un ganador entre `k`.** No se declara.

## Los riesgos, escritos antes

- **Quedarse con un eje.** `esq-2d` midió que un kernel se queda con la esquina barata
  (`tl` 90 %, `br` 23 %). Aquí puede quedarse con `izq`/`der` o con `sup`/`inf`. Por eso todo
  va por borde.
- **Los bordes enfrentados.** La lectura supone que el kernel responde con signo **opuesto**
  al entrar y al salir de la tinta. Si aprende un detector simétrico (por ejemplo, de
  «cuánta tinta hay»), `izq` y `der` saldrán peores que el suelo y eso **es** un resultado.
- **`k = 3`.** Con 9 píxeles y la salida promediada en 37 filas, puede no superar el suelo.
  Se reporta igual.
