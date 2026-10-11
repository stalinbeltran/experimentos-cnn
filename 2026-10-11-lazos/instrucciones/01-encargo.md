# Encargo (el dueño, 2026-10-11, conversación de Telegram)

> «Ahora, vamos a crear otro experimento, mismas reglas, pero creamos y evaluamos con dígitos el detector de lazos,
> deben detectar también el nivel de cerrados, entre 0 (lazo abierto) y 1 (completamente cerrado), y el centro (igual
> que en las curvas). En las imágenes dibuja solo en gris el dígito, en azul una cruz en el centro del lazo, y una
> representación delgada del lazo detectado, según su orientación y grado de cierre (la orientación 'apunta' hacia la
> parte abierta del lazo).»

Lo que el encargo da por sabido, escrito aquí:

1. **«Mismas reglas»**: las del repo para un experimento (carpeta propia, `REGLAS.md`, criterio escrito y commiteado
   antes de medir, código copiado y no importado) y las que el dueño fijó para el detector el 2026-10-10: **sin
   esqueleto y sin morfología**; el detector de curvas es el delgado, sobre la imagen tal cual.
2. **«El centro (igual que en las curvas)»**: el detector de curvas ya da, por trozo curvo, el vector de curvatura
   κ·(−sin θ, cos θ), que apunta a su centro de curvatura. El centro de un lazo sale de ahí.
3. **El dibujo** está pedido literalmente: el dígito sólo en gris, una cruz azul en el centro de cada lazo y el lazo
   como un arco delgado de su radio, que cubre el ángulo que cubre el trazo y deja el hueco hacia donde apunta su
   orientación.
