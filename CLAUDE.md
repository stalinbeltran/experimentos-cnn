# CLAUDE.md — `experimentos-cnn`

Este repo es para **probar estructuras de red sueltas**. Cada experimento trae su propio
código, sus propias reglas y su propio criterio, y puede no parecerse en nada al de al lado.
Léete esto antes de proponer nada: aquí la libertad es real, pero tiene una frontera escrita,
y la frontera es lo que evita que un experimento cueste dinero sin que nadie lo vea.

## Qué es, y por qué está separado de `foveal-vision`

Nació el **2026-09-06** para sacar los experimentos de `foveal-vision`, con el motivo que dio
el dueño al crearlo: *«foveal tiene muchas restricciones propias que entorpecen la
especificación de las pruebas»*.

**Y separarlos no es una preferencia: ya costó una vez.**
`foveal-vision/pyproject.toml:59` lleva `norecursedirs = ["experimentos", …]` porque los
snapshots congelados de los experimentos chocaron de nombre con los tests vivos y `pytest -q`
pasó a recoger **cero** tests. Un experimento es una ráfaga —se escribe en dos días y no se
toca más—; `src/fv` cambia todos los días. Relojes opuestos, piezas distintas (R1).

⚠ **Aquí no vive código de producción.** Si un experimento necesita cambiar `fv`, se copia
lo que haga falta a su `nn/` y se deja `foveal-vision/src/fv/` intacto. Es instrucción del
dueño del 2026-09-03: *«estos son experimentos… si hay que hacer cambios al código tendremos
que copiarlo localmente (pero si vale la pena, y eso depende de nuestras pruebas)»*.

## La libertad, y su frontera exacta

Esto es lo que el dueño pidió que quedara anotado: **la estructura de este repo se puede
sugerir y re-ordenar según convenga**, y cada experimento puede tener sus propias reglas y
objetivos. Pero hay cuatro cosas que un experimento **no** puede decidir por su cuenta, y
las cuatro protegen dinero o trabajo, no gusto.

| Un experimento SÍ decide | Un experimento NO puede cambiar | Por qué |
|---|---|---|
| su arquitectura, su bucle, su métrica, su dataset, su umbral, qué llama «ganar», cuántas semillas, qué dependencias usa, si reusa `fv` o no, y **la forma interna de su carpeta** | que **el freno lo vea** si tarda o alquila | el veredicto «¿se puede apagar este server?» se lee desde el móvil y decide una factura |
| en qué lenguaje lo escribe, si publica figuras, si guarda snapshot de código | que exista un **criterio escrito antes de mirar** (R13) | escrito después no se distingue de una racionalización, y «no hubo señal» deja de ser un resultado |
| dónde deja sus artefactos **dentro** de su carpeta | que el veredicto que mueva `ESTADO.md` acabe en el **repo central** (R7) | si no, `estudios-redes-neuronales` deja de contestar qué se pagó ya |
| — | que los secretos no se commiteen · que todo vaya a `main` · que el dato de entrada **no se copie aquí** | reglas del sistema, no del experimento |

**Lo que NO está en esa lista es libre.** No hace falta pedir permiso para inventarse una
estructura de carpeta distinta, ni para no usar `expcnn`, ni para escribir el experimento
entero en un solo fichero.

## Se puede RE-ORDENAR — y por eso el nombre de la carpeta NO es la identidad

El dueño pidió explícitamente poder re-ordenar los ficheros en el futuro. Eso sólo sigue
siendo barato si se respeta **una** convención:

> **Nada de fuera de una carpeta de experimento la nombra por su ruta: se la pide al registro
> por su `id`. Y ningún experimento importa de otro experimento.**

Y una regla de forma que la sostiene: **la carpeta empieza siempre por su fecha**
(`<AAAA-MM-DD>-<nombre>`). No es estética. La comprobación de arriba busca el nombre de la
carpeta dentro de los ficheros, así que una carpeta llamada `k3s1` casaría con cualquier
docstring o tabla que hable del brazo `k3s1` — el aviso que sale siempre y se deja de leer.
Con la fecha delante el nombre es inequívoco, y renombrar sigue siendo gratis: lo que se fija
es la **forma** del nombre, no el nombre.

Con eso, re-ordenar es `git mv` y nada más: renombrar, agrupar por tema (`planas/`,
`cabezas/`), meterlas por trimestre o cambiar la forma interna de una son **gratis**.

La identidad la da un dato comprobable —el `id` de `experimento.json`— y no el nombre de la
carpeta (R16). **No es una precaución teórica**: en `foveal-vision/experimentos/`,
`comun/serie.py:19-26` y `comun/preproceso.py:69-71` cablean nombres de carpeta, así que allí
renombrar una carpeta rompe el evaluador.

**Cómo se comprueba** (las dos mitades, medidas el 2026-09-06 al escribir esto):

```bash
python3 comprobar.py                 # lista, valida, y falla si algo de fuera nombra una carpeta
git mv 2026-09-06-algo 2026-09-06-otro-nombre && python3 comprobar.py   # tiene que salir 0
```

## La forma de un experimento

Es una **sugerencia con motivo**, no un molde: lo único obligatorio es `experimento.json`.

```
<fecha>-<nombre>/
  experimento.json     LO ÚNICO OBLIGATORIO. La identidad y las obligaciones que hereda
  README.md            qué se preguntó, qué salió, y cómo repetirlo
  instrucciones/       01-encargo.md · 02-criterio.md  ← el criterio, ANTES de mirar (R13)
  nn/
    entrenar_local.py  ⚠ EL NOMBRE ES UN CONTRATO si entrena largo (ver abajo)
    modelo.py          la red AUTOCONTENIDA: no importa nada de nadie
    pesos/<brazo>/     best.pt · last.pt · config.json · metrics.jsonl · summary.json
  resultados/          métricas, figuras, kernels
  codigo/              SNAPSHOT congelado del código que lo produjo, si hizo falta
```

Las cuatro reglas se heredan de `foveal-vision/experimentos/README.md`, que ya las tiene
escritas con su porqué, y siguen valiendo aquí:

1. **`nn/modelo.py` no importa nada del repo.** Es lo único que garantiza poder cargar los
   pesos dentro de un año.
2. **`codigo/` es un snapshot, no una copia de trabajo.** No se edita nunca: dos copias vivas
   del mismo código divergen y nadie se entera.
3. **Los pesos SÍ entran en git aquí, y es una excepción con su tope.** La regla general del
   proyecto es que no se guardan (862 runs × 2,7 MB ≈ 2,3 GB, medido 2026-08-31); estas redes
   son pequeñas —de 304 a 19.656 parámetros, 12 checkpoints = 1,1 MB— así que la razón de la
   regla no aplica. **El tope es ≈5 MB por experimento**; por encima, se pregunta.
   ⚠ Se comprueba que **cargan**, no que están: un `.pt` que no abre ocupa sitio y parece un
   respaldo.
   ⚠ Esto **no** cambia la regla para la red de producción, que sigue pidiendo que el dueño lo
   ordene y va aprobada una a una en `inferencia.json`.
4. **Lo que no se puede regenerar se guarda; lo que sí, se enlaza.**

## ⚠ El contrato de nombre con el freno: `entrenar_local.py`

`telegram-coordinator/scripts/cerrable.mjs:137` **ya casa `entrenar_local.py`** en su lista
declarada `TRABAJOS` (comprobado el 2026-09-06). Ese es el mecanismo por el que un
entrenamiento de este repo aparece en el veredicto que el dueño lee desde el móvil antes de
destruir la máquina.

**Con otro nombre, el freno dice «nada corriendo» con 37 épocas vivas.** Por eso el script que
entrena se llama así, y por eso `comprobar.py` se niega si un experimento declara
`gasta: "entrena-local"` y su entrada se llama de otra forma.

`experimento.json` declara qué obligaciones hereda:

| `gasta` | Qué significa | Qué obliga |
|---|---|---|
| `"no"` | no tarda ni cuesta | nada |
| `"entrena-local"` | entrena en esta máquina, tarda | la entrada se llama `entrenar_local.py` |
| `"alquila"` | alquila máquinas: **cuesta dinero** | además, ejecutor de Telegram en el **mismo commit**, `desacoplar-persistente.sh`, y `notify.mjs … \|\| true` |

## Dónde va cada artefacto

| Qué | Dónde | Por qué |
|---|---|---|
| pesos, métricas, figuras, kernels | **aquí**, en la carpeta del experimento | vive donde su productor (R7) |
| el dataset de entrada | **se LEE de `foveal-vision-data`; NUNCA se copia aquí** | este repo es **público** y el de datos es **privado** (medido el 2026-09-06: anónimo, 200 contra 404). Git no olvida |
| el reporte de un estudio | **`estudios-redes-neuronales`**, con su fila en `reportes/README.md` | el central es quien contesta «qué se pagó ya» |
| qué se cree hoy de un parámetro | `ESTADO.md` del central (se reescribe) | estado e historial son documentos distintos (R8) |

**Cuándo un experimento merece reporte en el central — dos preguntas mecánicas**, para no
discutirlo cada vez:

1. **¿Cambia lo que `ESTADO.md` dice de algún parámetro?** Sí → reporte + fila + edición de
   `ESTADO.md`, **aunque haya costado 0 $** (hay precedente: el reporte #12 es un estudio con
   0 máquinas y 0,00 $). No → se queda en su carpeta con su `README.md`.
2. **La unidad del reporte es la PREGUNTA, no la carpeta.** Seis carpetas que contestan una
   pregunta son **un** reporte que enlaza las seis.

⚠ Esta frontera es una **propuesta pendiente de confirmar**: `reportes/README.md` del central
dice hoy *«un reporte por cada barrido, estudio o medición que se termine, venga de donde
venga»*, y la práctica ya lo contradice (de los 11 experimentos de `foveal-vision`, sólo 2
tienen reporte). Si se confirma, la enmienda se escribe **allí**, no aquí en silencio.

## Cómo se conecta con `foveal-vision` (y qué pasa si no está)

Un experimento **autónomo no necesita nada**: `comprobar.py` y `expcnn` son stdlib pura, así
que un clon limpio corre sin instalar nada.

Hay **dos** puertas, y son las dos únicas: `expcnn.exigir_fv()` para el código y
`expcnn.exigir_datos()` para el dato de entrada. Se llaman **en la primera línea** del
experimento que las necesite:

```
código:  EXPCNN_FV                    >  el hermano ../foveal-vision       >  se NIEGA
dato:    EXPCNN_DATOS > FV_DATA_ROOT  >  el hermano ../foveal-vision-data  >  se NIEGA
```

⚠ **La puerta al dato respeta `FV_DATA_ROOT`** —la variable con la que `foveal-vision`
resuelve ESE MISMO repo (`src/fv/settings.py:27`)— a propósito: dos mandos para un solo hecho
pueden discrepar sin que nada falle (R15).

⚠ **Y NO cae al repo de código como hace `fv.settings.data_root()`.** Allí ese respaldo es
correcto: quien no ha clonado el repo de datos sigue funcionando. Aquí sería un directorio sin
`windows.npz`, o sea fallar a mitad en vez de negarse al empezar (R2).

Se niega **antes de empezar**, no a la época 30 (R2). Y hay una sola indirección a propósito:
en `foveal-vision/experimentos/` hay **72 `sys.path.insert`** repartidos por 41 ficheros
(medido el 2026-09-06), cada uno deduciendo del disco dónde está el repo hermano — que es el
antipatrón de R4, y que al mudarse aquí dejaría de ser frágil para ser **falso**.

## Cómo se corre

```bash
python3 comprobar.py                  # qué hay, qué puede correr, qué está mal puesto
uv venv .venv && uv pip install -e .  # sólo si un experimento lo necesita
```

**Un entorno por experimento sale casi gratis con `uv`, así que no lo pienses dos veces:**
medido el 2026-09-06 en esta máquina, dos venvs con `numpy` ocupan 57 MB cada uno por
separado pero **58 MB los dos juntos** (comparten por enlace duro), y crear el segundo tardó
**0 s**. Lo caro es torch: el `.venv` de `foveal-vision` pesa **1,1 GB**, y ése conviene
reusarlo (`EXPCNN_FV`) en vez de copiarlo.

⚠ **Esta máquina es pequeña: 2 vCPU y 3,8 GB de RAM** (medido 2026-09-06). Aquí caben
tanteos; lo que entrene de verdad alquila máquina — y entonces entra todo lo de la fila
`alquila` de arriba.

## Desde Telegram

`/use exp` — qué experimentos hay, en qué estado y qué está mal puesto. Llega solo, sin
copiar nada ni reiniciar el bot: el coordinador escanea `~/src/*/telegram`.

⚠ Un ejecutor nuevo de este repo se llama **`exp-<algo>`**, para no colisionar con los de
`foveal-vision` (que ya tiene `estudio`, `entrenar`, `vigilante`…). Las colisiones se avisan
pero las gana la primera fuente, y eso no es un criterio que quieras (R15).

## Los huecos conocidos, sin adornar

- **No hay CI, y por tanto `comprobar.py` no lo corre nadie solo** (R17). Ninguno de los siete
  repos lo tiene. Hoy se corre a mano o por `/use exp`.
- **Este repo no tiene `.claude/`**, así que una sesión abierta con el cwd aquí **no dispara**
  el triage, ni el registro de sesión, ni el archivado de conversación — esos hooks viven
  sólo en `telegram-coordinator`. No es un freno que se pueda dar por supuesto.
- **`comun/` no existe, y es a propósito.** Se crea cuando haya **dos** experimentos que
  tengan que medir con la misma regla, nunca con el primero: un `comun/` creado el primer día
  es un `utils/` con mejor nombre (antipatrón de R1). En `foveal-vision` se ganó su sitio con
  seis gemelos.
- **⏳ ABIERTO: los 11 experimentos de `foveal-vision/experimentos/` siguen allí.** Mientras
  convivan los dos sitios, *«¿dónde está un experimento?»* tiene dos respuestas, que es la
  definición de migración a medias (R19). Está sin decidir a propósito: **el dueño no ha
  pedido moverlos**. Si se mueven, lo caro no son los ficheros sino los 72 `sys.path.insert`
  y los nombres de carpeta cableados en `comun/`.

## Las reglas de escritura y de diseño se heredan enteras

Las cinco reglas de redacción (todo número con su procedencia, «sobrevive» con complemento,
caducidad de los cerrojos, terminado = invocable desde Telegram, el preflight crece con cada
fallo) y las 19 reglas de diseño viven en `telegram-coordinator`
(`CLAUDE.md` y `docs/reglas-de-diseno.md`) y **valen aquí igual**. Se entra por la tabla de
disparadores del § 0, no se lee entero.

Y las dos operativas que más muerden:

- **Lo que no está empujado, no existe.** Estos servidores se rehacen sin aviso: cambio o
  documentación terminada → commit → push, el mismo día. Todo a `main`.
- **Cada respuesta al usuario termina con la línea del freno**:
  `node ~/src/telegram-coordinator/scripts/cerrable.mjs --breve`.

Comentarios y mensajes al usuario: en **español**, como el resto del proyecto.
