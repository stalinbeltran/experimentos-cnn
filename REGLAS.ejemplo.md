# Reglas de `<id>`

<!--
  PLANTILLA. Se copia a la carpeta del experimento como `REGLAS.md` y se rellena.
  Cada línea que siga diciendo RELLENAR hace fallar a `comprobar.py`, a propósito:
  un hueco se lee como «aquí no aplica» y en realidad es «todavía no lo he pensado».

  ⚠ Estas reglas son de ESTE experimento y de ninguno más. Si vienes de copiar otro,
  léelas línea por línea y cambia lo que no aplique. Cambiar cualquier cosa es gratis
  y no hay que justificarlo; CONSERVARLA sin decidirlo es lo que sí es un fallo.
  (`CLAUDE.md` § «Regla 0 — cada experimento es INDEPENDIENTE de los demás».)
-->

**Qué pregunta:** RELLENAR — una frase, la misma que el `pregunta` de `experimento.json`.

## Entradas

- **Dataset:** RELLENAR — el nombre publicado en `foveal-vision-data/experimentos-cnn/`, el
  mismo que declara `experimento.json`. Se lee con `expcnn.exigir_dataset(...)`; no se
  regenera. Si este experimento no consume un dataset publicado, dilo y di qué consume.
- **Qué se lee de él:** RELLENAR — qué particiones, qué columnas de la etiqueta, qué se
  ignora. Dos experimentos sobre el mismo `.npz` pueden leer cosas distintas.
- **Condiciones que el dataset ya trae:** RELLENAR — margen, reducción, descartes, qué cuenta
  como positivo. Van aquí aunque estén en el manifiesto del dataset: aquí es donde se lee
  antes de tocar nada.
- **Qué se normaliza o transforma al cargar:** RELLENAR.

## Salidas

- **Pesos:** RELLENAR — dónde caen y con qué nombre por brazo.
- **Métricas:** RELLENAR — qué fichero, qué columnas, por época o al final.
- **Figuras:** RELLENAR — cuáles, dónde, y con qué comando se regeneran.
- **Tablas / informe:** RELLENAR — quién las genera y dónde se pegan.
- **Qué de esto se commitea y qué no:** RELLENAR (el tope de pesos del repo es ≈5 MB por
  experimento; el dato de entrada NUNCA se commitea aquí).

## Procesos

Los pasos, en orden, y qué se decide en cada uno:

1. RELLENAR — p. ej. generar y publicar el dataset (una vez en la vida).
2. RELLENAR — p. ej. comprobar el mecanismo sin entrenar.
3. RELLENAR — p. ej. entrenar los brazos.
4. RELLENAR — p. ej. figuras e informe.

- **Qué se mide y con qué umbral:** RELLENAR — y el criterio completo, escrito **antes de
  mirar**, va en `instrucciones/02-criterio.md` (R13).
- **Cuántos brazos y cuántas semillas:** RELLENAR.
- **Qué se llama «ganar»:** RELLENAR — o «no se declara ganador: se reportan todos los que
  pasan», si es el caso.

## Scripts

Los de este experimento, con su interfaz exacta. **Los nombres y las banderas son de aquí**:
otro experimento puede llamar a lo mismo de otra forma y estar bien.

| script | qué hace | cómo se llama |
|---|---|---|
| RELLENAR | RELLENAR | RELLENAR |

- ⚠ **Si `gasta` es `entrena-local`, el script que entrena SE LLAMA `entrenar_local.py`.** Es
  el único nombre que este repo impone, y no es estética: es el contrato con el freno
  (`cerrable.mjs`), que es quien decide si se puede apagar el server.
- **Dependencias:** RELLENAR — qué venv, qué paquetes.
- **De dónde sale el código:** RELLENAR — autónomo, o copiado de dónde. Ningún experimento
  **importa** de otro: se copia.

## Qué NO hereda

- **Se copió de:** RELLENAR — el `id` del experimento de origen, o «no se copió de ninguno».
- **Qué se cambió a propósito:** RELLENAR — la lista, con el motivo de cada cambio. Es la
  parte que evita que el siguiente que llegue «arregle» una diferencia deliberada.
- **Qué se conservó, y por qué se decidió conservarlo:** RELLENAR — conservar también es una
  decisión; lo que no vale es conservar por inercia.
- **Contra qué se compara, si es que se compara:** RELLENAR — y qué haría que dejara de ser
  comparable. Dejar de serlo es legítimo si se dice; lo que no vale es descubrirlo después.
- **Restricciones de otros experimentos que NO aplican aquí:** RELLENAR — nómbralas si son de
  las que se cuelan solas, o pon «ninguna conocida».
