# `esq-cq` — una esquina CUALQUIERA de las dos: ¿qué tamaño de kernel la detecta?

**PREPARADO el 2026-09-07, SIN ENTRENAR NI UNA ÉPOCA.** Están las estructuras, el dataset, los
suelos medidos y el criterio congelado. Lo que no hay todavía es resultado.

Es **`esq-k` con la etiqueta colapsada**: `existe` = 1 si la ventana contiene la esquina
superior-izquierda **o** la inferior-derecha, y `(x, y)` es la de la que haya. `tr` y `bl` siguen
siendo negativos, por orden del encargo (*«por ahora limitadas a las mismas tl y br»*).

## Por qué quita la tensión que hundió a `esq-2d`

Con `W = S + A` (partes simétrica y antisimétrica bajo giro de 180°), para el mismo parche:

```
respuesta_tl = <S,P> + <A,P>          respuesta_br = <S,P> − <A,P>
```

`esq-2d` necesitaba que las dos fueran **distintas y en extremos opuestos** — que mandara `A`—, y
falló: `br` se quedó en el 23 % en ventana y en **0/10 en página**. Aquí hace falta lo contrario:
que las dos sean el **mismo** extremo, y eso pasa cuando el kernel es **simétrico**. Es una
condición que un filtro lineal **sí** puede cumplir.

## La estructura: la de `esq-k`, sin tocar un parámetro

```
entrada x (1, 32, 32) en TINTA (/255)
    │
    └── conv(x, W)  ──►  M (m×m, m = 32 − k + 1)  ──►  cabeza C1 (3 parámetros)
                                                        p = softmax(β·M)
                                                        (x, y) = esperanza bajo p
                                                        existe = a · logsumexp(β·M)/β + b
```

| brazo | `k` | mapa | kernel | cabeza | **total** | (`esq-k` a ese `k`) |
|---|--:|---|--:|--:|--:|--:|
| `k05` | 5 | 28×28 | 25 | 3 | **28** | 28 |
| `k07` | 7 | 26×26 | 49 | 3 | **52** | 52 |
| `k09` | 9 | 24×24 | 81 | 3 | **84** | 84 |
| `k11` | 11 | 22×22 | 121 | 3 | **124** | 124 |
| `k13` | 13 | 20×20 | 169 | 3 | **172** | — |

**Ni un parámetro de diferencia con `esq-k`.** En `esq-2d` la cabeza tuvo que pasar a 5 porque
había dos cosas que detectar; aquí vuelve a haber una.

### Las otras dos estructuras, anotadas y sin armar

| | cómo | kernel a k=7 |
|---|---|--:|
| `sim` | fuerza `A = 0`: `W = (V + rot180(V))/2`. Las dos esquinas dan **el mismo número** por construcción | **25** (la mitad) |
| `abs` | el kernel sigue libre y la cabeza lee **`|M|`**: un pico negativo cuenta igual que uno positivo, así que un filtro **orientado** sirve para las dos | 49 |

Siguen implementadas y **comprobadas** en `nn/modelo.py` (`sim` es simétrico exacto; `abs`
encuentra un pico negativo que `libre` no ve). Armar una es añadir su línea a `BRAZOS`. Detalle en
[`instrucciones/03-alternativas-anotadas.md`](instrucciones/03-alternativas-anotadas.md).

## El dataset: el MISMO fichero, sin regenerar nada

`esquinas300-32px-r4-r20260907`, el que produjo `esq-2d`. **No se rinde ni una imagen**: aquél se
etiquetó con **las cuatro** esquinas exactamente para esto.

```bash
python nn/datos.py --comprobar    # el publicado, sus huellas, y qué sale al colapsar
```

⚠ **El colapso se comprueba, no se supone**: `cargar()` **se niega** si alguna ventana contuviera
más de una de las buscadas, en vez de quedarse con una en silencio. Hoy no pasa nunca, y el
manifiesto del dataset dice por qué (párrafo ≥ 64 px, ventana 32).

**Y por eso la comparación con `esq-k` es exacta:** mismas ventanas (2.157 / 389 / 120), **mismas
156 positivas en validación**, misma red, mismo `λ` = 0,03 medido, mismo suelo de `f1` = 0,572. Lo
único distinto es qué cuenta como positivo — así que la diferencia entre los dos barridos **es** el
precio de admitir la segunda esquina.

## El criterio, congelado antes de mirar

En [`instrucciones/02-criterio.md`](instrucciones/02-criterio.md). Suelo **5,1–5,8 %** a ≤2 px,
umbral **10 %**; error medio por debajo de **5,71 px**; `f1` por encima de **0,572**.

> ⚠⚠ **Y un segundo umbral que no hace falta medir, sale de la aritmética: el 50 %.** Con 78 `tl` y
> 78 `br`, un brazo que resuelva **una sola** clase topa en **50,0 %**. Cualquiera por encima del
> 50 % ha detectado **necesariamente las dos**; cualquiera por debajo tiene que enseñar su desglose
> por clase antes de que se diga que «detecta esquinas». Sin eso, un 50 % con `tl` al 100 % y `br`
> a cero sería `esq-2d` otra vez con otro nombre.

**La predicción registrada**: el kernel debería volverse **simétrico**, y se mide en cada época
(`simetrico` en `metrics.jsonl`, arranca en 43–62 %).

## Cómo se corre (cuando se ordene)

```bash
cd ~/src/experimentos-cnn
E=2026-09-07-esquinas-cualquiera
.venv/bin/python $E/nn/entrenar_local.py --suelos                    # sin entrenar nada
for b in k05 k07 k09 k11 k13; do
  .venv/bin/python $E/nn/entrenar_local.py --brazo $b --epocas 300   # reanudable
done
.venv/bin/python $E/nn/muestras.py --etiqueta ep300
```

**0 máquinas, 0 $**, y ~0,35 s/época medidos en el barrido gemelo → **~10 min los cinco brazos**.
El freno lo ve: `entrenar_local.py` está en su lista `TRABAJOS`.
