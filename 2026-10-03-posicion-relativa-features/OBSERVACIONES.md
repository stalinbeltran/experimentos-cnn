# `feat-pos` — Observaciones: codificar la posición relativa de las features, sin texto

**2026-10-03 · SÓLO DOCUMENTO.** Nada implementado ni medido. Igual que en `feat-ind`, cada
afirmación dice de dónde sale: **(a)** medido en este repo o en `foveal-vision`, con fecha;
**(b)** derivado de la definición, sin medir; **(c)** de memoria de la literatura, marcado. Al final,
qué esperaríamos ver, escrito hoy.

## 0. Qué hay que codificar, exactamente

El dueño lo describe con el 9: *círculo + recta vertical, unidos aproximadamente en cierta
posición*, y añade dos cosas que fijan el problema: **no debe ser texto**, y **esa es sólo una de
las codificaciones posibles de un 9**.

El ejemplo más limpio de por qué la posición importa es el par **6 / 9**: tienen **el mismo
conjunto de features** (un lazo y un trazo) y sólo cambia la **disposición relativa** (lazo abajo
con trazo subiendo a la derecha, o lazo arriba con trazo bajando). Cualquier codificación que
pierda la disposición los confunde; cualquier codificación que la conserve los separa. Es el test
mínimo de toda propuesta de esta página.

Qué tiene que **conservar** la codificación y a qué tiene que ser **indiferente**:

| conservar | ser indiferente a |
|---|---|
| qué features hay | la posición **absoluta** del dígito en la imagen |
| **dónde está cada una respecto de las otras** (vector de desplazamiento, no sólo «arriba/abajo») | la escala global, dentro de un rango |
| que una feature puede **faltar** o aparecer **dos veces** | pequeñas rotaciones y deformaciones |
| **varias** configuraciones válidas para el mismo objeto (el «sólo una de las posibles») | — |

## 1. Qué significa «representación compatible», y por qué el texto no lo es

Una representación sirve si cumple cuatro cosas, y el texto incumple las tres primeras:

1. Es un **vector o tensor** que un módulo entrenable puede consumir y derivar.
2. Tiene una **distancia**: configuraciones parecidas quedan cerca.
3. Es **componible**: varias features → una representación, con un número variable de features.
4. Es **invariante a la traslación** del objeto entero.

Un **embedding**, en el sentido que tiene en aprendizaje automático, es exactamente «un vector con
la propiedad 2». Así que la respuesta a *«¿embeddings sirven?»* es **sí, pero hay que decir
embeddings DE QUÉ**, porque hay dos cosas distintas que se llaman igual (§3).

## 2. Las codificaciones candidatas, con lo que cuesta cada una

### E1 — Mapas apilados: la posición es «en qué celda del mapa» (lo que una CNN hace por dentro)

Los `N` detectores dan `N` mapas de calor `H×W`; se apilan como `N` canales y un compositor
convolucional pequeño los lee. **La posición relativa queda codificada en los pesos del
compositor**: un kernel que cubre el tamaño del objeto aprende «lazo en el desplazamiento (−3, 0)
respecto del trazo» como patrón espacial, y un pooling global al final da la invariancia a la
traslación.

- Coste: ninguno nuevo; es la salida natural de detectores totalmente convolucionales.
- Qué pierde: la codificación **no es inspeccionable** (está en los pesos) y no es una
  representación de la configuración que se pueda comparar o guardar: es un clasificador.
- ⚠ **La variante que NO conviene es aplanar**: `flatten` + `Linear` codifica la posición por el
  índice del vector, o sea **absoluta**, y crece con `H·W`. Es lo que hace la cabeza de la red
  foveada: 12.800 features aplanadas → 12 salidas, **91,1 % de los parámetros** *(medido el
  2026-09-01, `plan-strides-rama-2026-09-01.md`)*. Ese tamaño es el precio de codificar posición
  absoluta por índice.

### E2 — Vector de pares: desplazamientos explícitos entre features

De cada mapa se extraen posiciones (picos); el objeto se representa como el **conjunto** de
ternas `(feature_i, feature_j, d_ij)` con `d_ij = p_j − p_i`. Cada terna se embebe (el
desplazamiento, con una codificación de posición: §3.1) y el conjunto se agrega con una operación
**indiferente al orden** (suma o máximo), al estilo de los *Deep Sets* *(Zaheer y otros, 2017; de
memoria)*.

- Coste: `O(N²)` pares; hay que decidir qué pasa con features ausentes (máscara de presencia) y
  repetidas (varias instancias de la misma).
- Qué da: **invariancia a la traslación por construcción** (sólo entran diferencias), y una
  codificación **legible**: «un 9 es lazo→trazo con `d ≈ (0, +3)`» se puede leer del embedding
  de la terna sin texto, porque la terna *es* la afirmación.
- Es la respuesta más directa a «codificar las posiciones».

### E3 — Posiciones relativas al centro del objeto

En vez de pares: se calcula un **centro** del objeto (centroide de la tinta, o media de las
posiciones detectadas) y cada feature se representa como `(feature_i, p_i − c)`, embebido y
sumado. `O(N)`, invariante a la traslación, y conserva la configuración entera (6 y 9 difieren en
el signo de la coordenada vertical del lazo). Pierde la explicitud par a par de E2, pero para
2–4 features por dígito la configuración respecto del centro **determina** los pares.

- Es el **equivalente directo de las codificaciones posicionales de los transformadores**
  *(sinusoidales, Vaswani y otros, 2017; de memoria)* aplicadas a coordenadas **relativas** en vez
  de a índices absolutos. Aquí es donde «embeddings» encaja literalmente.

### E4 — Vector de pose por feature (cápsulas)

Cada detector emite, además de la presencia, una **pose** (posición, escala, ángulo), y el
compositor comprueba si las poses de las partes **concuerdan** con una pose del todo *(cápsulas,
Sabour, Frosst y Hinton, 2017; de memoria)*. Es la formulación más fiel al «parte–todo», y la más
difícil de entrenar de forma estable según esa misma literatura. Se anota como referencia, no como
primer candidato.

### E5 — Grafo

Nodos = features, aristas = desplazamientos, paso de mensajes → embedding del grafo. General, pero
para 2–4 nodos es E2 con más maquinaria. Se anota para no reinventarlo si el vocabulario crece.

### Resumen

| | invariante a traslación | legible | número variable de features | coste | primero? |
|---|---|---|---|---|---|
| **E1** mapas + conv + pooling | sí (por el pooling) | no | sí | el menor | **sí, como base** |
| **E2** pares | sí (por construcción) | **sí** | sí (conjunto) | `O(N²)` | si E3 no separa casos como 6/9 |
| **E3** relativo al centro | sí | parcial | sí | `O(N)` | **sí, como codificación explícita** |
| E4 pose / cápsulas | sí | sí | sí | alto, inestable | no |
| E5 grafo | sí | sí | sí | alto | no |

## 3. «¿Embeddings sirven?» — sí, en dos sentidos que hay que separar

### 3.1 Embedding DE LA POSICIÓN (codificación posicional)

Una función fija o aprendida `R² → R^d` que convierte un desplazamiento en un vector con la
propiedad «desplazamientos parecidos → vectores parecidos» y que una capa lineal pueda leer.
Opciones, de la más simple a la más expresiva:

- **Un solo caso por celda** (*one-hot*) del desplazamiento en una rejilla **gruesa**: a 32×32
  los desplazamientos posibles son 63×63 = 3.969 celdas *(aritmética)*, demasiadas; agrupadas en
  7×7 bins quedan 49, manejable.
- **Sinusoidal / rasgos de Fourier** de `(dx, dy)`: fijo, sin parámetros, suave, y permite
  interpolar entre desplazamientos no vistos.
- **Tabla aprendida** sobre la rejilla gruesa: pocos parámetros, se ajusta al dato.

Cualquiera de las tres **sirve**; la sinusoidal es la que menos decisiones exige.

### 3.2 Embedding DE LA CONFIGURACIÓN (el objeto entero)

El vector que sale de agregar las ternas (E2) o las features relativas al centro (E3). Su utilidad
no es sólo alimentar un clasificador: **se puede medir solo**, sin clasificador, mirando si la
distancia en ese espacio separa las clases. Con features **oráculo** (configuraciones sintéticas:
lazo en `p1`, trazo en `p2` → etiqueta) y un vecino más cercano sobre el embedding, la pregunta
*«¿separa 6 de 9?»* se contesta **sin entrenar ningún detector**. Eso aísla la codificación de la
calidad de los detectores, que es exactamente lo que hace falta para no mezclar dos fallos.

### 3.3 Lo que un embedding NO hace

No sustituye a los detectores: codifica **qué y dónde** dadas las detecciones. Si el detector no
vio el lazo, ningún embedding lo repone. Y no decide el vocabulario.

## 4. La trampa de lo discreto: extraer posiciones de un mapa

Pasar de un mapa de calor a **un punto** es el paso delicado:

- `argmax` mata el gradiente: el compositor no puede enseñar al detector dónde mirar.
- `soft-argmax` (coordenada esperada bajo un softmax del mapa) mantiene el gradiente pero **falla
  con dos instancias** de la misma feature: devuelve el punto medio entre las dos, que no es
  ninguna.
- Picos con supresión de no-máximos: correcto con varias instancias, no derivable.

**Observación que simplifica todo, y que viene de la premisa de `feat-ind`**: si los detectores se
entrenan **aparte y se congelan**, la cadena **no necesita** ser derivable a través de la
extracción de posiciones. Se extraen picos como se quiera, y el compositor se entrena sobre
detecciones fijas. La independencia que allí es una hipótesis sobre el aprendizaje, aquí es una
**licencia de ingeniería**. Si en cambio se quisiera afinar todo junto (brazo C de `feat-ind`),
habría que quedarse en E1 o aceptar `soft-argmax` con sus límites.

## 5. «Sólo una de las codificaciones posibles de un 9»

El mismo dígito tiene **varias** configuraciones legítimas (lazo cerrado con cola recta, lazo
abierto, cola curva). La representación debe permitir **muchas-a-una**:

- Un clasificador **lineal** sobre el embedding de configuración exige que todas las variantes de
  una clase sean separables de las demás por un plano, lo que no está garantizado.
- Un **vecino más cercano** o un conjunto de **prototipos por clase** admite varios grupos por
  clase sin problema, y es inspeccionable: cada prototipo *es* una de las codificaciones posibles.

Observación: la elección del lector del embedding (lineal / prototipos) importa tanto como la
codificación, y los prototipos son la forma natural de hacer explícito el «varias posibles».

## 6. Resolución, otra vez

A 8×8 (`uci-optdigits-8px-r20261002`, el publicado) los desplazamientos entre features de un
dígito son de 1–3 celdas *(estimación por la geometría, NO medida)*: una codificación de posición
sobre eso tiene muy pocos valores distintos. Los bitmaps de NIST son 32×32 y no vienen con
scikit-learn (`README.md` del dataset). **Qué resolución usar lo decide el dueño**; este
documento sólo anota que a 8×8 el margen para la posición relativa es estrecho.

## 7. Qué se haría primero, y cómo se conecta con `feat-ind` sin romper la Regla 0

Primero, a 0 $ y **sin que `feat-ind` exista**:

1. Configuraciones **oráculo** sintéticas (feature, posición) → etiqueta, incluyendo pares
   difíciles como 6/9 y variantes de la misma clase.
2. **E3** con codificación sinusoidal + vecino más cercano: ¿separa? (§3.2). Si no separa casos
   que necesitan información par a par, **E2**.
3. **E1** como base de comparación, entrenado sobre los mismos mapas oráculo.
4. Ruido controlado sobre el oráculo (features perdidas, falsas, desplazadas) para medir la
   **robustez** de cada codificación antes de tener detectores reales.

Y cuando `feat-ind` tenga detectores entrenados, `feat-pos` **no los importa** (Regla 0: ningún
experimento importa de otro). Hay dos canales legales, y hay que elegir uno: **publicar las
detecciones como dataset** en `foveal-vision-data/experimentos-cnn/` y declararlo en `dataset`, o
**copiar** código y pesos a la carpeta de este experimento. Lo escrito en `REGLAS.md § Qué NO
hereda`.

## 8. Qué esperaríamos ver — escrito el 2026-10-03, antes de que exista nada

- Con oráculo, **E3 separa 6/9 y las variantes trivialmente**: está diseñada para eso, así que ese
  resultado **no** sería una prueba de nada más que de que el código funciona.
- La prueba real es la **robustez a detectores imperfectos** (§7.4): esperamos que E2/E3
  degraden **más rápido** que E1 con features perdidas, porque E1 puede compensar con el mapa
  entero y E2/E3 dependen de que el punto exista.
- La respuesta a «¿sirven los embeddings?» será: **sí para la posición** (sinusoidal o tabla, con
  poca diferencia entre ellas), y el embedding de configuración vale **como intermedio
  inspeccionable y comparable**, no como ganancia de exactitud sobre E1.
- El lector por **prototipos** hará mejor que el lineal en clases con varias codificaciones (§5).

Si sale al revés, eso es un resultado y se escribe tal cual.
