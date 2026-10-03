# Reglas de `feat-pos`

**Escritas el 2026-10-03. Estado `abierto`, SÓLO DOCUMENTO.** Hoy este experimento es un
documento de observaciones (`OBSERVACIONES.md`), por orden del dueño: no hay código, ni dataset,
ni corrida. Estas reglas describen eso, y se reescriben **en el mismo commit** en que nazca la
primera corrida.

**Qué pregunta:** ¿cómo se codifica la posición relativa entre las features de un objeto —p. ej. un
9 = círculo + recta vertical unidos en cierta posición, y ésa es sólo una de sus codificaciones
posibles— de forma que la información posicional entre en el resultado del reconocimiento sin
pasar por texto, y sirven los embeddings (de posición, o de la configuración entera) para eso?

## Entradas

- **Dataset:** ninguno todavía (`dataset: null`, a propósito: no se mide nada). ⚠ A 8×8 (el
  publicado `uci-optdigits-8px-r20261002`) el margen para posiciones relativas es estrecho
  (`OBSERVACIONES.md` § 6); la resolución la decide el dueño.
- **Qué se lee de él:** nada todavía. Lo primero que se consumiría son **configuraciones
  oráculo** sintéticas (feature, posición → etiqueta), que no dependen de ningún detector.
- **Condiciones que el dataset ya trae:** no aplica todavía.
- **Qué se normaliza o transforma al cargar:** no aplica todavía.

## Salidas

- **Pesos:** ninguno.
- **Métricas:** ninguna.
- **Figuras:** ninguna.
- **Tablas / informe:** `OBSERVACIONES.md`, con las cinco codificaciones candidatas (E1–E5), los dos
  sentidos de «embedding» y la sección «qué esperaríamos ver», fechada.
- **Qué de esto se commitea:** todo lo que hay (texto). Reporte en el central: **no**;
  `reporte: null`.

## Procesos

1. Escribir las observaciones (hecho el 2026-10-03). **No hay corrida.**
2. Cuando el dueño decida implementar: fijar resolución y vocabulario, escribir
   `instrucciones/02-criterio.md` **antes de mirar**, declarar `gasta`, `entrada` y `dataset`, y
   reescribir estas reglas.

- **Qué se mide y con qué umbral:** nada todavía. `OBSERVACIONES.md` § 7 propone el orden (oráculo
  → E3 → E2 si hace falta → E1 como base → robustez a detectores imperfectos).
- **Cuántos brazos y cuántas semillas:** sin decidir.
- **Qué se llama «ganar»:** sin decidir.

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| ninguno | — | — |

- Si algún día `gasta` pasa a `entrena-local`, el script que entrene **se llama
  `entrenar_local.py`** (contrato con el freno `cerrable.mjs`).
- **Dependencias:** ninguna.
- **De dónde sale el código:** no hay.

## Qué NO hereda

- **Se copió de:** no se copió de ningún experimento.
- **Qué se cambió a propósito:** nada que cambiar; la forma de la carpeta (sin `README.md`, sin
  `02-criterio.md`, con `OBSERVACIONES.md`) se decidió para este caso.
- **Qué se conservó, y por qué:** los dos ficheros obligatorios del repo y la orden literal en
  `instrucciones/01-encargo.md`.
- **Contra qué se compara, si es que se compara:** contra nada todavía.
- **Restricciones de otros experimentos que NO aplican aquí:** ninguna conocida, salvo el dataset
  de 8×8, que no es decisión tomada.

### ⚠ La relación con `feat-ind`, escrita para que el siguiente no haga el `import`

`feat-pos` **nace de la pregunta** de `feat-ind` (las posiciones de las features que aquél
reconoce), y eso es **linaje** (`experimento.json` → `linaje`, por `id`), no herencia ni
dependencia:

1. **Se puede estudiar sin que `feat-ind` exista**, con features **oráculo** (dibujadas o
   anotadas). Es lo primero que se haría (`OBSERVACIONES.md` § 7).
2. **Ningún experimento importa de otro** (Regla 0 del repo). Si algún día se quieren usar los
   detectores de `feat-ind`, hay **dos** canales legales y hay que elegir uno: **publicar sus
   detecciones como dataset** en `foveal-vision-data/experimentos-cnn/` y declararlo aquí en
   `dataset`, o **copiar** su código y sus pesos a la carpeta de este experimento.
3. **No se compara contra los números de `feat-ind`**: contestan preguntas distintas.
