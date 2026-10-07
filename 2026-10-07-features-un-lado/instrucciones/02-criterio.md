# Criterio — escrito el 2026-10-07, ANTES de entrenar ningún detector de este experimento

No existe ningún detector `control` ni `compartido`: nada de lo de abajo se decidió mirando uno. Lo que SÍ se miró antes, y
hay que decirlo:

1. **El boceto** (`docs/bocetos/2026-10-07-borde-de-un-lado/`): con los kernels fijos y una regresión logística, las 8 vistas
   juntas superan a la tinta (0,974 contra 0,950 con 3823 de entrenamiento) y el compositor entrenado con la entrada
   desplazada 1–2 px llega a 0,989. **No son detectores entrenados**: es la información que queda en cada vista.
2. **La referencia de las líneas** (`feat-ind32`), con el código de este experimento: §A, §B y §C en
   `resultados/referencia-lineas.json` (idénticos a los de `feat-bor`: R 0,696, F1 0,795 → 0,567, arcos en los 1 0,76 /
   0,70), y el compositor y su curva en `resultados/componer.json`.
3. **Una corrida de prueba de 5 épocas** del brazo compartido en `recta-V` (para medir tiempo y memoria): F1 0,887 en val.
   Se cita porque se vio; no cambió nada de lo de abajo.

Las cifras de este fichero y las de `criterio()` en `nn/evaluar.py` son las mismas. Si al correr hace falta cambiar algo,
va como **enmienda fechada** al final.

## A. Los detectores aprenden (val sintético, 2–4 px) — los umbrales de feat-ind32

Veredicto por detector: **aprendió** si F1 ≥ 0,90 y posición ≤ 1 celda ≥ 0,90; **a medias** si F1 ≥ 0,75; si no, **no aprendió**.

**H1** (control) y **H1** (compartido): **≥ 11 de 13** al menos «a medias», **y** el F1 medio **≥ 0,866** (el de las
líneas, 0,916, menos 0,05).

## B. El GROSOR (`feat-bor-sinteticas-grueso-32px-r20261006`) — la pregunta del experimento

La misma medida que `feat-bor` (copiada): F1, recall y FP en los grosores vistos (2–4 px) y no vistos (6–12 px).

**H2** (control) y **H2** (compartido): se confirma si se cumplen **las tres**, iguales a las de `feat-bor`:
1. los falsos positivos suben con el grosor **como mucho la mitad que en las líneas** (≤ +0,048; líneas +0,096);
2. el recall **no baja más de 0,05** (líneas: no baja; `feat-bor signo`: −0,109; contorno: −0,190);
3. el F1 en 6–12 px **no es peor que el de las líneas** (≥ 0,567).

**La hipótesis del experimento es H2 compartido.** H2 control es el control: si el compartido aguanta el grosor y el control
no, la diferencia la hace **mirar un canal cada vez**, que es lo único que distingue a los dos brazos.

## C. Los dígitos

**H3** (control) y **H3** (compartido) — el síntoma de las líneas: arco-E y arco-W encendidos en los 1 (líneas 0,76 y
0,70). Se confirma si los dos quedan **≤ 0,30** (`feat-bor signo` lo consiguió: 0,02 / 0,08).

**Diagnóstico, no hipótesis — ¿degenera el compartido?** En cada detector, qué canal gana el máximo cuando se enciende en un
dígito. Si **más de 6 de 13** detectores tienen un canal con más del **90 %** de los encendidos, el brazo aprendió «un
detector de un solo lado» y no «la forma vista desde cualquier lado»: se dice, y H2 compartido se lee con eso delante.

**H5** (control) y **H5** (compartido) — que no se pierda lo que había: el compositor posicional (180 de train, 1617 de
val, 3 semillas, sin desplazar) **≥ 0,919** (líneas 0,949 − 0,03). `feat-bor signo` dio 0,865.

**H6** (control) y **H6** (compartido) — el compositor ELEGIDO (entrenado con la entrada desplazada 0–2 px, 180 de train)
resiste un desplazamiento ligero: su acierto con el dígito desplazado **1 px y 2 px** (media de las 8 direcciones) no baja
más de **0,03** del acierto sin desplazar.

**H7** (control) y **H7** (compartido) — los gruesos REALES (el cuartil de val con más tinta de cada clase): con el
compositor elegido, **≥ el de las líneas** con el mismo compositor.

**La curva** (lo que pidió el dueño): compositores entrenados SÓLO con s px y con 0..s px, s ∈ {1, 2, 3, 4, 6, 8}, contra
el dígito desplazado d ∈ {0, 1, 2, 3, 4, 6, 8, 12, 16} px. Se **reporta entera**, sin umbral: es una descripción. Lo que
se espera, en el boceto: «solo s» tiene su pico en d ≈ s y cae en d = 0 cuando s es grande (aprende el desplazamiento que
vio, no a resistirlo); «≤ s» mantiene la meseta hasta d ≈ s y cae después; en d = 16 (medio dígito fuera del lienzo)
todos cerca del azar.

## Lo que espero hoy (antes de entrenar)

- **H1** sí en los dos; el compartido quizá algo por debajo (cada pasada ve menos), sin bajar de 0,866.
- **H2 control, no**: es `feat-bor signo` con 8 canales continuos en vez de 4 binarios, y mira los lados juntos.
- **H2 compartido, a medias**: el recall debería aguantar mucho mejor que en `feat-bor` (cada canal conserva la forma); los
  FP pueden subir igual, porque un trazo grueso tiene más borde exterior en cada canal y el máximo de 8 canales enciende
  con cualquiera. Mi apuesta: cumple el 2 y el 3, y el 1 está en duda.
- **H3** sí en los dos (como `feat-bor signo`).
- **Degeneración**: no espero que pase, pero arcos y esquinas tienen un lado preferido y podrían concentrarse en 2–3
  canales sin pasar del 90 %.
- **H5**: el control, alrededor de 0,90; el compartido, 0,90–0,94. Dudo de que ninguno llegue a 0,949.
- **H6** sí en los dos (el boceto lo da con margen); **H7** sí en el compartido, dudoso en el control.
- **Lo que más probablemente salga mal:** el ruido del dataset sintético (puntos sueltos) es justo lo que más borde de un
  lado produce, y el máximo sobre 8 canales lo multiplica por 8. Si los FP del compartido son altos ya en 2–4 px, es eso.

Si sale al revés en cualquiera de estos puntos, eso es un resultado y se escribe tal cual.

## Limitaciones que no se arreglan aquí

- **Una semilla** por detector (como `feat-ind32` y `feat-bor`).
- El control tiene 8 canales de entrada y el compartido 1: la inicialización de **todas** las capas difiere entre brazos,
  así que el efecto de «un canal cada vez» no se separa del de la inicialización.
- Los 8 canales son **continuos** (Sobel con ReLU), no binarios como el `signo` de `feat-bor`: el control no es una
  réplica exacta de `feat-bor signo`, y por eso se compara contra el control, no contra `feat-bor`.
