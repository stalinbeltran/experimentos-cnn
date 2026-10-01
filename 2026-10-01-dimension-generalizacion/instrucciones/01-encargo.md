# Encargo — 2026-10-01

## Lo que pidió el dueño, literal y completo

> «en el repo de experimentos especifica un experimento nuevo que se dedique exclusivamente a
> medir el efecto que tiene la reducción de las dimensiones de la imagen sobre la capacidad de
> generalizar de la cnn. Para evaluar qué tan buena es la generalización vamos a emplear sólo
> 10% del dataset como imágenes de entrenamiento y 90% como validacion. Además el número de
> capas se va a mantener constante en L (por ej 4) y cada capa va a reducir su tamaño (no
> padding al aplicar el kernel). Para cada 'reducción' se va a definir que el área de cada
> ventana será un cuadrado de nxn pixeles, donde n se calcula como n = ancho_img_entradaxf, y f
> tiene un valor aprox de 0.1 (es decir, las ventanas están diseñadas para ver un 10% del ancho
> la imagen completa) (nota que 0.1 es un ejemplo, puedes analizar si esto funciona o proponer
> otro valor más apropiado, sobretodo si encuentras problemas con ese valor). Puedes emplear un
> dataset previamente creado (en el volumen). Documenta el plan, no lo ejecutes aún.»

## Cómo se ha leído cada término

Son **decisiones de interpretación**, y las cuatro primeras son **supuestos que el dueño tiene
que confirmar** (`ESPECIFICACION.md` §0): cada una tiene otra lectura posible y el experimento
cambia con ella.

| término del encargo | cómo se lee aquí | la otra lectura, y por qué no | dónde está el porqué |
|---|---|---|---|
| **«ventana»** (S1) | el **kernel** de cada convolución (`n × n`), que es lo que «se aplica» | en este proyecto «ventana» suele ser el **recorte de entrada** (p. ej. «1000 ventanas de 146 px» en el índice de datasets). La frase «al aplicar el kernel … el área de cada ventana» se lee como la misma cosa | `ESPECIFICACION.md` §3.1 |
| **«ancho_img_entrada»** (S2) | el ancho de la imagen que **entra en la red** (`W`), el mismo `n` en las `L` capas | el ancho de la entrada de **cada capa**: encoge el kernel capa a capa y el mapa nunca colapsa (calculado y descartado) | §3.3 |
| **«la cnn» / la tarea** (S3) | el encargo no la fija: se toma la que el dataset etiqueta, **localizar la caja de tinta** de un párrafo, medida con **IoU** | cualquier otra etiqueta pediría publicar otro dato | §2.4 |
| **«reducción de las dimensiones»** (S4) | el **lado `W`** de la imagen cuadrada: 128 → 64 → 32 → 16 → 8, derivados del mismo dato por suma **exacta** de bloques | otros `W` (96, 48…) exigirían interpolar: dejarían de ser el mismo dato reducido | §2.2 |
| «f ≈ 0,1» | **no funciona con `L = 4`** (mapa final de ~0,6·W, cabeza que escala con `W²`, `n = 1` a `W = 8`). Se propone la regla **`f = 1/L`**: con `L = 4`, `f = 0,25` | — | §3.2 |
| «no padding al aplicar el kernel» | convolución `valid`, **sin stride y sin pooling**: la única reducción por capa es `n − 1` | — | §3.1 |
| «L constante (por ej 4)» | `L = 4`, y `f` sale de `L`. Si el dueño prefiere `f = 0,1`, la regla da `L = 10`; también está calculado | — | §3.4, §7 |
| «10 % entrenamiento / 90 % validación» | el reparto **que el dataset ya trae**: `train` = 100 imágenes; `monitor ∪ eval` = 900. **Ninguna decisión usa las 900**: ni parada temprana, ni selección de pesos, ni elección de `lr`. Si alguna lo hiciera, el reparto real sería 10/10/80 | — | `REGLAS.md` § Entradas |
| «capacidad de generalizar» | el **IoU sobre las 900 imágenes no vistas**, leído **junto con** `IoU_train` y la **brecha**: es lo que separa «generaliza peor» de «el dato ya no tiene la información» | — | `instrucciones/02-criterio.md` |
| «un dataset previamente creado (en el volumen)» | `parrafos1000-584px-r4-r20260908b`, publicado en el repo de datos (cuyo `origin` es el almacén) | — | §2.1 |
| «documenta el plan, no lo ejecutes aún» | esta carpeta trae identidad, reglas, especificación y criterio; **no trae código ni resultados**, y `gasta` es `no` hasta que haya código | — | `experimento.json` |
