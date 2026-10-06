# Encargo — 2026-10-06

Orden literal del dueño (mensaje a `c` por Telegram, tema principal), después de ver en `feat-agr` que el grosor del
trazo manda sobre la forma:

> «Crea un detector sintético de features idénticos a los que ya entrenamos, pero con una diferencia: no deben detectar
> lineas sino bordes. Es factible? Si los grosores son un problema, los bordes deben resolverlos, pues el borde es
> independiente del grosor (traen otros problemas, pero ya los trataremos)»

«Los que ya entrenamos» son los 13 detectores de 32×32 de `feat-ind32` (C3): el mismo vocabulario, el mismo sorteo, la
misma red y la misma receta. Lo que cambia es lo que ve el detector: el **borde** de la tinta en vez de la tinta.

La pregunta tiene dos mitades y las dos se contestan: **¿es factible?** (sí: es cambiar la imagen de entrada) y **¿el
borde resuelve el grosor?** (eso es lo que se mide). La segunda se midió primero sin entrenar (`nn/sonda_grosor.py`), y
por lo que salió se entrenan DOS bordes: el contorno, que es lo pedido literalmente, y el borde con signo (`REGLAS.md`).
