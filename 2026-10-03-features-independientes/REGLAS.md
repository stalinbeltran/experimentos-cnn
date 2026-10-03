# Reglas de `feat-ind`

**Escritas el 2026-10-03. Estado `abierto`, SÓLO DOCUMENTO.** Hoy este experimento es un
documento de observaciones (`OBSERVACIONES.md`), por orden del dueño: no hay código, ni dataset,
ni corrida. Estas reglas describen eso, y se reescriben **en el mismo commit** en que nazca la
primera corrida.

**Qué pregunta:** si cada CNN pequeña reconoce una única feature fijada de antemano (reconoce, no
discrimina) y las `N` se entrenan independientes y en paralelo, ¿qué resultado neto dan juntas al
reconocer un objeto complejo compuesto por varias de esas features, comparado con una sola CNN que
lo clasifica directamente?

## Entradas

- **Dataset:** ninguno todavía. `experimento.json` declara `dataset: null` a propósito: declarar
  uno afirmaría «sobre esto se mide», y no se mide nada. ⚠ El publicado `uci-optdigits-8px-r20261002`
  (8×8) es un candidato con una pega escrita en `OBSERVACIONES.md` § 2.5 (resolución); la decisión
  es del dueño.
- **Qué se lee de él:** nada todavía.
- **Condiciones que el dataset ya trae:** no aplica todavía.
- **Qué se normaliza o transforma al cargar:** no aplica todavía.

## Salidas

- **Pesos:** ninguno.
- **Métricas:** ninguna.
- **Figuras:** ninguna.
- **Tablas / informe:** `OBSERVACIONES.md`, con sus siete secciones y la de «qué esperaríamos ver»
  fechada.
- **Qué de esto se commitea:** todo lo que hay (texto). Reporte en el central: **no**, no cambia
  nada de `ESTADO.md` porque no se midió nada; `reporte: null`.

## Procesos

1. Escribir las observaciones (hecho el 2026-10-03). **No hay corrida.**
2. Cuando el dueño decida implementar: fijar vocabulario y resolución, escribir
   `instrucciones/02-criterio.md` **antes de mirar**, declarar `gasta`, `entrada` y `dataset`, y
   reescribir estas reglas. Hasta entonces, nada de lo de abajo aplica.

- **Qué se mide y con qué umbral:** nada todavía. Lo que se esperaría ver está en
  `OBSERVACIONES.md` § 7 y se formaliza en el criterio cuando haya corrida.
- **Cuántos brazos y cuántas semillas:** sin decidir; `OBSERVACIONES.md` § 3 propone cuatro brazos
  (A monolítico · B detectores congelados + compositor · C afinado junto · D oráculo).
- **Qué se llama «ganar»:** sin decidir.

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| ninguno | — | — |

- Si algún día `gasta` pasa a `entrena-local`, el script que entrene **se llama
  `entrenar_local.py`** (contrato con el freno `cerrable.mjs`).
- **Dependencias:** ninguna.
- **De dónde sale el código:** no hay. Si se implementa, el rasterizador de trazos se **copia** de la
  idea de `ruido-nist` (no se importa).

## Qué NO hereda

- **Se copió de:** no se copió de ningún experimento.
- **Qué se cambió a propósito:** nada que cambiar; la forma de la carpeta (sin `README.md`, sin
  `02-criterio.md`, con `OBSERVACIONES.md`) se decidió para este caso: no hay resultados que
  contar ni corrida a la que fijarle criterio.
- **Qué se conservó, y por qué:** los dos ficheros obligatorios del repo y la orden literal en
  `instrucciones/01-encargo.md`, que es lo único que no se puede reconstruir después.
- **Contra qué se compara, si es que se compara:** contra nada todavía. Los números de `dim-nist`
  y `ruido-nist` que cita `OBSERVACIONES.md` son **contexto**, no comparación.
- **Restricciones de otros experimentos que NO aplican aquí:** el dataset de 8×8 de `dim-nist` y
  `ruido-nist` no es una decisión tomada; sus protocolos (3996 pasos, pesos iniciales
  compartidos, 3 semillas) tampoco. La pregunta hermana —codificar la **posición** relativa de las
  features— es de `feat-pos`, y no se contesta aquí.
