# Encargo — 2026-10-07

Sale del boceto `docs/bocetos/2026-10-07-borde-de-un-lado/` (figuras y cuatro pruebas rápidas a 0 $ hechas ese día con el
dueño). Las órdenes literales, en orden (mensajes a `c` por Telegram, tema principal):

1. *«Al menos por ahora, no se aplica pre-proceso a ningún dígito antes del detector de features.»* — reescrita el mismo
   día: *«si se aplica un pre-proceso debe aplicarse a todos los dígitos antes de los detectores»* (`CLAUDE.md` de la raíz).
2. La propuesta del boceto (§ «Segunda parte»): los bordes de un lado como el ÚNICO pre-proceso, y los detectores de
   features entrenados sobre ellos — con un detector que mire **un canal cada vez**, que es lo que `feat-bor` no probó.
3. Sobre el compositor, tras la prueba rápida: *«usa un compositor sin máximo, directamente la posición, y entrena otro
   compositor donde la misma entrada esté desplazada 1, 2, 3, 4 px … Esto nos debe dar un compositor ligeramente resistente
   al desplazamiento de su entrada, que en teoría debe ser mejor»*. Medido en el boceto («Cuarta parte»): sí, desplazando
   la IMAGEN 1–2 px; desplazando celdas del 8×8, no.
4. *«Aplica todas tus sugerencias. Procede.»* y, al proponer este experimento: *«Lánzalo tal como dices. Prueba los
   entrenamientos del compositor con varios desplazamientos de la entrada del compositor para ver cómo se comporta cada
   desplazamiento por separado y luego añadiéndolos gradualmente para contar con un compositor entrenado para varios
   desplazamientos de sus entradas (en teoría debe ser más resistente a un desplazamiento ligero, y debe perder su
   capacidad cuando el dígito quede fuera de su campo de visión)»*.

Con eso el dueño aceptó también que la aumentación por desplazamiento del COMPOSITOR no es un segundo pre-proceso (la
pregunta quedó abierta en el boceto y la contestó con «tal como dices»).

⚠ «Tal como dices» se dijo con un coste de **~0,1 $ / ~1 h**. Al medir el brazo compartido salió ~10 veces más caro (8
pasadas por imagen): **se le vuelve a preguntar antes de alquilar**, con el número medido.
