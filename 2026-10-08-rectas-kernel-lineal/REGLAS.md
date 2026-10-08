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

- **Pesos:** no se guardan `.pt`; los kernels aprendidos van en `resultados/kernels.npz` (≈432 × 2 × k² floats, < 1 MB).
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
- **Entrenamiento:** Adam, lr 0,05, 400 épocas a lote completo, BCE con logits sobre las 4 salidas. Semilla = inicialización.
- **Qué se mide y con qué umbral:** en `instrucciones/02-criterio.md`, escrito antes de mirar.
- **Qué se llama ganar:** no se declara un ganador único; se reporta la rejilla contra las hipótesis del criterio.

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/datos.py` | genera y publica entrenamiento y banco; los carga con su huella | `--generar [--publicar]` · `--comprobar` |
| `nn/modelo.py` | el detector lineal (2 kernels + rot90, pirámide, ReLU, max) y el Gabor a mano | `--comprobar` |
| `nn/referencia_cnn.py` | COPIA de la arquitectura del detector de `feat-ind32`; lee sus 4 rectas por id + huella de su firma | `--comprobar` |
| `nn/entrenar_local.py` | la rejilla; reanudable (salta lo que ya está en `rejilla.jsonl`) | `[--solo k,escalas,entreno,N,semilla]` |
| `nn/evaluar.py` | banco de prueba, referencias, tablas y figuras | `--referencias` · `--tablas` · `--figuras` |

- **Dependencias:** el `.venv` de la raíz del repo (torch CPU, numpy, scipy, matplotlib).
- **De dónde sale el código:** escrito aquí. Lo único copiado es la clase del detector de `feat-ind32` (en
  `referencia_cnn.py`), porque hace falta para cargar sus pesos.

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
