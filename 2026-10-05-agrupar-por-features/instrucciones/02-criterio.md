# Criterio — escrito el 2026-10-05, ANTES de calcular ningún grupo

No se ha corrido ningún agrupamiento: nada de lo de abajo se decidió mirando uno. Lo único mirado antes de escribirlo:
la **firma por clase** que `feat-ind32` y `feat-ind` ya guardaron (qué fracción de cada dígito enciende cada detector), el
orden de las etiquetas en el fichero (para saber si había escritores: no los hay) y que los índices de
`errores-c.json` son los de este dataset (67 de 67 casan en etiqueta y origen). Si al correr hace falta cambiar algo, va
como **enmienda fechada** al final.

Todo sobre la partición principal —**Z32, K = 30**, la de menor inercia de las 5 semillas— salvo donde se diga otra cosa.
Las definiciones (pureza, variante, grupo mezclado, consistencia, los índices de L2b) son las de `REGLAS.md` §5.

## H0 · Los grupos siguen a los dígitos

**Pureza(Z32, K = 30) ≥ 0,70**, con el azar (etiquetas permutadas) al lado. Si no llega, los grupos se forman por otra
cosa —grosor, estilo— y todo lo demás se lee con esa sospecha.

## H1 · El ejemplo del dueño: el 1 se parte por la bandera

Se **confirma** si (a) el 1 tiene **≥ 2 variantes estables**, y (b) en al menos un par de ellas la **bandera** *b* las separa con
**AUC ≥ 0,80, y mejor que el grosor y que la inclinación**. Se **refuta** si el 1 no se parte, o si se parte pero por
grosor o por inclinación — y entonces se dice por cuál. Lo mismo se reporta para Z8 y X8, sin umbral.

⚠ *b* se mide en los píxeles, **sin detectores**: es el testigo independiente. Si se midiera con los mapas, H1 sólo diría
que los grupos se parecen a la descripción con la que se formaron.

## H2 · Hay variantes más allá del 1, y no son un efecto de K

**≥ 4 de las 10 clases con ≥ 2 variantes estables** (las que reaparecen en ≥ 4 de las 5 semillas). Con 30 grupos para 10
dígitos alguna clase se parte por fuerza; por eso no cuenta cualquier corte, sólo el que se repite.

Y se reporta, sin umbral, **K = 10**: con tantos grupos como etiquetas, si una clase se parte, otras dos tienen que
compartir grupo. Ahí se ve dónde una forma pesa más que una etiqueta.

## H3 · Hay formas compartidas, y explican los errores de hoy

(a) Hay **al menos un grupo mezclado** (mayoría < 70 %, ≥ 20 miembros); se listan sus pares. (b) De los 54 dígitos que
falla C, los que caen en grupos mezclados son **≥ 5** y su proporción es **≥ 2 veces** la de todos los dígitos.

## H4 · Agrupa por forma, no por apariencia — sin etiquetas

Consistencia corregida por azar (κ) bajo **desplazar 4 px**: **κ(Z32) ≥ κ(X8) + 0,10 y κ(Z32) ≥ κ(M32) + 0,10**.
Engrosar y adelgazar se reportan sin umbral, y la consistencia sin corregir, al lado.

⚠ P32 no entra en esta comparación: es invariante al desplazamiento **por construcción** (no mira el dónde), así que su
consistencia alta no dice nada de ella.

## H5 · Son de los dígitos: no de la semilla, ni de los escritores, ni del banco

- **Semilla:** ARI medio entre las 5 semillas (10 pares) **≥ 0,60**.
- **Escritores nuevos:** ARI(los 1797 asignados desde los centroides de los 3823, los 1797 agrupados aparte) **≥ 0,8 ×**
  el ARI entre semillas; y **≥ 24 de los 30** grupos de los 3823 reciben **≥ 1 %** de los 1797.
- **Banco:** ARI(Z32, Z8) **≥ 0,5 ×** el ARI entre semillas. Si no, los grupos dependen del banco y se dice así.

Los umbrales de escritores y banco van **relativos** al ARI entre semillas a propósito: dos k-means sobre los mismos datos
ya discrepan, y ese desacuerdo es el suelo contra el que se mide cualquier otro.

## H6 · Sirven para elegir qué etiquetar

Con los K medoides etiquetados, el compositor lineal sobre M32 acierta en los 1797 **al menos 3 puntos más que con K al
azar** (media de 5 sorteos), en **al menos 2 de K ∈ {20, 30, 50}**. Se reportan además K = 10 y 100, el azar
estratificado y la propagación.

⚠ La comparación justa es contra el azar **simple**: ni los medoides ni él miran etiquetas para elegir. El estratificado
sí las mira (garantiza K/10 por clase, como los datasets balanceados de las curvas de hoy), y por eso sale como
referencia y no como rival.

## Lo que espero hoy (antes de mirar)

- **H0** sí, pureza 0,80–0,92, y X8 parecida (± 0,05): la pureza no es lo que separa las dos descripciones; lo que las
  separa es **por qué** se parte cada clase.
- **H1** es la que más dudo. A favor: el detector de recta-S (`/`) de 32×32 es de los buenos (F1 0,971), y la firma dice
  que se enciende en el **28 %** de los 1 —una cota del tamaño de la variante con bandera, que también puede ser
  inclinación—. En contra: el grosor ya engañó a los detectores de 8×8 (el 1↔8 de `feat-ind`). Espero **sí con Z32 y no
  con Z8**, que partirá los 1 por grosor.
- **K = 10**: el 1 se parte (recto / con bandera, y el de bandera se va con el 7) y el 4 y el 9 comparten grupo.
- **H2** sí: el 1, el 7 (con y sin travesaño), el 4 (abierto y cerrado) y el 2 (con y sin lazo). El dataset se hizo en la
  Universidad del Bósforo (Estambul; `optdigits-orig.names`) —que los 43 escritores fueran de allí **no lo dice**, es un
  supuesto—, y si lo eran, el 7 cruzado y el 1 de bandera larga, que son la forma continental, tendrán peso.
- **H3**: grupos mezclados 1–7 (el 1 de bandera larga junto al 7), 4–9 (el 4 cerrado) y 3–5; y sí, los fallos de C se
  concentran ahí — sus pares más repetidos fueron 8→1, 3→5, 9→4, 0→8, 3→8, 4→9 (README de `feat-ind32`).
- **H4** sí en desplazar: Z tolera ±1 celda por construcción, y X8 y M32 no. Al **engrosar no sé**: con el compositor,
  los detectores de 32 aguantaron algo mejor que los píxeles crudos (0,615 contra 0,578, C6 de `feat-ind32`), pero sólo
  vieron grosores de 2–4 px, y engrosar 2 px los saca de ese rango.
- **H5**: ARI entre semillas 0,60–0,75; los escritores nuevos, sí; el banco, por poco (0,5–0,6 del ARI entre semillas).
- **H6** sí con K = 20–50 (+3 a +8 puntos); **no con K = 10**, porque los medoides no garantizan uno por clase; con K = 100
  la ventaja se cierra.
- **Lo que más probablemente salga mal:** que el **grosor mande sobre la forma** —grupos partidos en «trazo gordo /
  trazo fino» en vez de «con bandera / sin bandera»—. H1 está escrita para detectarlo, con el grosor como rival explícito.

Si sale al revés en cualquiera de estos puntos, eso es un resultado y se escribe tal cual.
