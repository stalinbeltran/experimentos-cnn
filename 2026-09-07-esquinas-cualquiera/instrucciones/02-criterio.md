# El criterio, congelado ANTES de la primera época

Escrito el **2026-09-07 con cero épocas entrenadas**. Los suelos están medidos sobre la partición
de validación con las redes **sin entrenar**, que es lo único que se puede medir antes de mirar.

```bash
python nn/entrenar_local.py --suelos      # reproduce la tabla de abajo
```

## La métrica principal, y su suelo

**Tasa de acierto: qué fracción de las esquinas se predice a ≤ 2 px de su sitio**, sobre las
ventanas donde hay esquina **de la que sea**. Val trae **389 ventanas y 156 positivas** (78 `tl` +
78 `br`) — exactamente las mismas 156 que `esq-k`.

| | acierto ≤2 px | ≤1 px | error medio | f1 `existe` |
|---|--:|--:|--:|--:|
| los cinco brazos **sin entrenar** | 5,1 – **5,8 %** | 1,3 – 2,6 % | 6,08 – 6,13 px | **0,572** |

⚠ Las redes sin entrenar caen en el **modo degenerado** de esta cabeza: mapa plano → softmax
uniforme → la esperanza cae en el centro del mapa, que es donde se concentran las etiquetas. La
solución vaga y la mediocre son la misma, así que hay un óptimo local cómodo justo donde arranca.

> **Un brazo ha aprendido algo** si su acierto ≤2 px pasa del **10 %**. Sale de suelo + 2·SE con
> SE = 1,9 % sobre 156 positivas: 9,5 %, redondeado hacia arriba.

> ⚠⚠ **Y hay un segundo umbral que no hace falta medir, sale de la aritmética: el 50 %.** Con 78
> `tl` y 78 `br`, un brazo que resuelva **una sola** clase topa en **50,0 %**. Así que **cualquier
> brazo por encima del 50 % ha detectado necesariamente las dos**, y cualquiera por debajo tiene
> que enseñar su desglose antes de que se diga que «detecta esquinas».

**Métrica secundaria**, siempre junto a su suelo: el **error medio**, que debe bajar de **5,71 px**
(= 6,08 − 2·SE, con SE = 0,19 px). Y `existe`, cuyo suelo es **f1 = 0,572** — el «siempre sí» con
40,1 % de positivas, y exactamente lo que dan las redes sin entrenar.

## Los desenlaces, escritos antes

**No se declara ganador** (orden del dueño del 2026-09-07: *«quiero ver todos los ganadores»*): se
reportan todos los que pasan, ordenados por `k`.

1. **Algún `k` pasa del 50 %.** Un solo kernel detecta **las dos** esquinas sin distinguirlas. Es
   el desenlace fuerte y no admite discusión sobre el desglose: la aritmética lo garantiza.
2. **Pasan del 10 % pero ninguno del 50 %, y el desglose es asimétrico.** Entonces *no* se ha
   resuelto la tarea: se ha resuelto **una** esquina, y es `esq-2d` otra vez. La lectura sería que
   el problema **nunca fue distinguirlas**, sino que un kernel no da para las dos formas.
3. **Ninguno pasa del 10 %.** Un kernel único no sirve ni para «esquina, la que sea». Sería un
   resultado sorprendente: `esq-k` llegó al 100 % con una sola, y esta tarea sólo añade la otra.
   Ahí lo primero que hay que sospechar es el montaje.
4. **El eje no es monótono, o el mejor es el `13`.** Se dice. Con una semilla no se distingue de
   una fluctuación, y el `13` es el borde del rango — el dataset admite hasta **k = 17** sin
   regenerarlo.

## La predicción registrada, para que pueda fallar

**El kernel debería volverse SIMÉTRICO bajo giro de 180°, y eso se mide.** Es la única forma de que
las dos esquinas sean el mismo extremo con un kernel libre. Se registra `simetrico` /
`antisimetrico` **en cada época**, y arrancan en 43–62 % de energía simétrica.

- **Si `libre` funciona y su fracción simétrica sube claramente**, la hipótesis de la
  descomposición queda confirmada — y entonces `sim`, con **la mitad de grados de libertad**, debería
  igualarlo. Ésa es la continuación declarada.
- **Si funciona y NO se vuelve simétrico**, hay otra solución que no habíamos previsto, y merece
  mirarse el kernel: es el caso más interesante de los tres.
- **Si no funciona**, `abs` es la otra vía: no exige simetría, sino que la cabeza ignore el signo.

⚠ **Y la comparación con `esq-k` es la que da la magnitud**: allí, una sola esquina, `k07` dio
**100 %** a ≤2 px sobre estas mismas 156 positivas. Lo que aquí falte de 100 % **es** lo que cuesta
admitir la segunda esquina, sin nada que traducir — mismas ventanas, misma red, mismo `λ`.

## Lo que se congela

dataset publicado `esquinas300-32px-r4-r20260907` · semilla 1 · ventana 32×32 · sin padding · sin
bias · cabeza C1 de 3 parámetros con β inicial 3,5 · Adam lr 0,05 · lote 128 · **300 épocas** ·
**`λ_coord` = 0,03** (medido: la regla «los dos términos parten iguales» da 0,0285–0,0303 en los
cinco, y es **el mismo 0,03 de `esq-k`**).

**Varía UNA sola cosa: el tamaño del kernel.**

## Lo que este experimento NO contesta

- **Nada sobre `tr` y `bl`**, que aquí son negativos por orden del encargo.
- **Nada sobre distinguir qué esquina es**: eso es `esq-2d`, y ya está medido.
- **Nada sobre páginas enteras.** El barrido mide ventanas de 32×32; en `esq-2d` quedó medido que
  la ventana y la página **no se ordenan igual**.
- **Una semilla.**
