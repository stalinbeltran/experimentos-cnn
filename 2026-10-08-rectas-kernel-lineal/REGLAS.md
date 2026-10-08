# Reglas de `rect-lin`

**Qué pregunta:** ¿un detector de rectas que es un kernel lineal aprendido —unos 50–160 parámetros— aprende con POCAS
muestras, generaliza al grosor, a rectas punteadas, y responde más fuerte a una recta que a una curva? ¿Cómo queda frente
al detector CNN de `feat-ind32`?

**Pre-proceso: NINGUNO.** El detector ve la imagen 32×32 tal cual (regla del dueño del 2026-10-07). La pirámide de escalas
es parte del **detector**, no un pre-proceso: el mismo kernel aplicado a la imagen reducida ×2 y ×4.

## Entradas

- **Dataset de entrenamiento:** `rect-lin-entreno-r20261008` (lo publica `nn/datos.py --generar --publicar`). Se lee con
  `expcnn.exigir_dataset`. Lleva 1000 rectas CONTINUAS, 1000 rectas PUNTEADAS y 1000 NEGATIVOS, cada bloque barajado
  con su semilla: la curva de N toma los **N primeros** de cada uno, así que N=4 ⊂ N=8 ⊂ … (más datos, no otros datos).
- **Banco de prueba:** `rect-lin-banco-r20261008`, publicado aparte y fijo. Nunca se entrena con él.
- **Condiciones de los datos (32×32, tinta 1 sobre fondo 0, binario):**
  - recta: centro uniforme en [10, 22]², ángulo uniforme en [0°, 180°), largo 14–26 px; **entrenamiento: grosor 2–4 px**
    (el banco prueba 2–14).
  - punteada: la misma geometría hecha de discos de diámetro 2 px, separados 3–6 px entre centros (el banco prueba 2–12).
  - negativos: un tercio vacío con ruido sal (1 % de píxeles), un tercio puntos SUELTOS (4–10 discos de 2 px en posiciones
    al azar, no alineados), un tercio una mancha (disco de radio 2–5 px).
  - convención de ángulo: x a la derecha, **y hacia ABAJO** (la misma que los detectores de `feat-ind32`): 0° = —, 45° =
    \, 90° = |, 135° = /.
- **Etiqueta:** 4 salidas, una por orientación; la recta activa la del centro más cercano (0/45/90/135°, ±22,5°). Un
  negativo, ninguna.
- **Qué se transforma al cargar:** nada; `uint8` → `float32`.

## Salidas

- **Pesos:** no se guardan `.pt`; los 2 kernels aprendidos (K0, K45), `a` y `c` van en cada línea de `rejilla.jsonl`.
- **Métricas:** `resultados/rejilla.jsonl`, una línea por (k, escalas, entrenamiento, N, semilla) con todas las métricas
  del banco; `resultados/referencias.json` (Gabor y CNN).
- **Figuras:** `resultados/*.png` con `nn/evaluar.py --figuras`.
- **Informe:** `README.md`, tablas generadas por `nn/evaluar.py --tablas`.
- **Se commitea todo lo anterior.** Los datos de entrada NUNCA (van al repo de datos).

## Procesos

1. Publicar los dos datasets (una vez).
2. `nn/modelo.py --comprobar`: rot90 exacto, pirámide, forma de la salida.
3. Medir el tiempo de UN brazo; si la rejilla entera no cabe en ~1 h en el dev, se lanza con `desacoplar-persistente.sh`.
4. `nn/entrenar_local.py`: la rejilla.
5. `nn/evaluar.py`: referencias (Gabor, CNN), tablas y figuras, y la lectura contra el criterio.

- **Rejilla:** k ∈ {5, 7, 9} × escalas ∈ {1, 2, 3} × entrenamiento ∈ {continua, punteada} × N ∈ {4, 8, 16, 32, 64, 128,
  256, 1000} × 3 semillas = **432 entrenamientos**. N = número de rectas; se añaden otros N negativos.
- **Entrenamiento:** Adam, lr 0,03, 400 épocas a lote completo, entropía cruzada de **5 clases** (las 4 orientaciones +
  «nada», con el logit de «nada» fijo en 0). Semilla = inicialización.
  ⚠ Cambiado tras comprobar el MECANISMO (antes de correr la rejilla, 2026-10-08): con un umbral aprendido dentro de la
  ReLU el umbral apagaba todas las respuestas (gradiente 0), y con BCE sobre 4 salidas el modelo colapsaba a decir siempre
  «nada» (pérdida 0,3768 = la de predecir 1/8). El criterio NO se tocó. En esas pruebas se vieron algunos números (k=7,
  1 escala: recall fino 0,99 con N=128; Gabor 0,98), y van anotados en el README.
- **Qué se mide y con qué umbral:** en `instrucciones/02-criterio.md`, escrito antes de mirar.
- **Qué se llama ganar:** no se declara un ganador único; se reporta la rejilla contra las hipótesis del criterio.

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/datos.py` | genera y publica entrenamiento y banco; los carga con su huella | `--generar [--publicar]` · `--comprobar` |
| `nn/modelo.py` | el detector lineal (2 kernels + rot90, pirámide, max) y el Gabor a mano | `--comprobar` |
| `nn/referencia_cnn.py` | COPIA de la arquitectura del detector de `feat-ind32`; lee sus 4 rectas por id + huella de su firma | `--comprobar` |
| `nn/entrenar_local.py` | la rejilla; reanudable (salta lo que ya está en su fichero de salida) | `[--solo k,escalas,entreno,N,semilla]` · `[--parte i/n] [--salida f] [--hilos h]` |
| `nn/vast.sh` | la rejilla entera en UNA máquina de Vast (14 trozos × 2 hilos), libro en `resultados/vast/rejilla/`, trae `resultados/trozos/` | `rejilla` · `--estado` · `apagar` · `VAST_SECO=1 …` |
| `nn/lanzar.sh` | la rejilla en el dev como unidad `rect-lin-rejilla` | `(sin args)` · `--estado` · `SECO=1 …` |
| `nn/evaluar.py` | banco de prueba, referencias, tablas y figuras | `--referencias` · `--tablas` · `--figuras` |

- **Dependencias:** el `.venv` de la raíz del repo (torch CPU, numpy, scipy, matplotlib).
- **De dónde sale el código:** escrito aquí. Lo único copiado es la clase del detector de `feat-ind32` (en
  `referencia_cnn.py`), porque hace falta para cargar sus pesos.

- **Dos sitios a la vez (2026-10-08, pedido del dueño: medir si alquilar acelera).** El dev escribe
  `resultados/rejilla.jsonl`; Vast trae los suyos a `resultados/vast/rejilla/…/trozos/`. **Nadie fusiona escribiendo en el
  fichero del dev mientras su unidad corre** (dos escritores; aviso del revisor): la fusión la hace `evaluar.py` al LEER,
  por clave (k, escalas, entreno, N, semilla). Los brazos medidos en los dos sitios comprueban si el resultado es el mismo
  en otra CPU.

## Qué NO hereda

- **Se copió de:** ningún experimento. El generador es nuevo (rectas de ángulo CONTINUO, punteadas, sin segunda feature).
- **Qué se cambió a propósito respecto a la línea `feat-*`:** detector lineal en vez de CNN; 4 orientaciones con pesos
  compartidos por rot90 en vez de 4 detectores independientes; salida por imagen (max espacial), no mapa 8×8; sin
  compositor ni dígitos: **se evalúa el detector solo** (orden del dueño).
- **Qué se conservó:** la convención de ángulo y el tamaño 32×32, para poder pasar el mismo banco a los detectores de
  `feat-ind32`.
- **Contra qué se compara:** el Gabor a mano (N = 0) y los 4 detectores de recta de `feat-ind32` (entrenados con 2000 por
  clase y grosor 2–4: un punto fijo, no una curva). La CNN ve detección = algún mapa 8×8 sobre SU umbral de la firma.
- **Restricciones de otros experimentos que NO aplican aquí:** el ±12° de jitter del vocabulario de `feat-*` (aquí el
  ángulo es continuo y la etiqueta es el centro más cercano); la segunda feature y el ruido de `feat-ind32`; el
  compositor y los dígitos NIST.

## PENDIENTE (anotado por orden del dueño, NO se mide aquí)

- **Curvas como segunda etapa sobre rectas cortas.** Una curva es una cadena de rectas cortas cuya orientación gira poco a
  poco (co-circularidad / campo de asociación). Si este detector da varias detecciones débiles a lo largo de una curva,
  una segunda etapa podría construir el detector de curvas encima. Lo mismo para la agrupación de puntos MUY separados
  (más que el kernel), que un solo kernel no puede integrar.
- **Las rectas GRUESAS pueden quedar fuera del detector** (el dueño, 2026-10-08): «las líneas gruesas no me preocupan
  mientras el sistema detecte líneas delgadas hasta cierto punto, pues las gruesas pueden eliminarse mediante filtros de
  bordes». O sea: un trazo grueso se convierte en dos bordes finos con un filtro de bordes y es ESO lo que ve el detector.
  Se ve después. Para leer este experimento significa que el recall grueso (H4, y la mitad «grueso» de H9) pesa menos
  que el fino; el criterio no se reescribe, se lee con esto delante.

