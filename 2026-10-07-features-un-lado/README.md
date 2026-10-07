# `feat-1lado` — detectores de features sobre bordes de UN SOLO LADO (en curso, 2026-10-07)

Sale del boceto `docs/bocetos/2026-10-07-borde-de-un-lado/`. El único pre-proceso son **8 bordes de un solo lado**, como
primera capa FIJA del modelo. Dos brazos de 13 detectores: **control**, que mira los 8 canales a la vez, y **compartido**,
que aplica el mismo detector a cada canal por separado y se queda con el máximo. Después, un compositor de dígitos
entrenado con la entrada desplazada, y su curva de desplazamiento.

- Plan y coste: `REGLAS.md`. Criterio, escrito antes de entrenar: `instrucciones/02-criterio.md`. Encargo literal:
  `instrucciones/01-encargo.md`.
- Referencia sin entrenar nada (las líneas de `feat-ind32`, con este código): `resultados/referencia-lineas.json`, idéntica
  a la de `feat-bor`.

## Estado

| paso | estado |
|---|---|
| código y comprobaciones (`modelo.py`, `datos.py`, `entrenar_local.py --comprobar`, `probar_vast.sh`, `comprobar.py`) | ✅ 2026-10-07 |
| revisión del gasto (agente `revisor`) | ✅ RESERVAS, aplicadas: tope del compartido 6 → 8 h, vCPU por `TRABAJO_VCPU`, `progreso.json` |
| referencia del compositor y su curva (líneas) | en curso en el dev (unidad `f1l-componer-lineas`) |
| entrenar `control` en Vast | pendiente |
| entrenar `compartido` en Vast | pendiente del permiso del dueño, con el tiempo real del control |
| componer y evaluar | pendiente |
