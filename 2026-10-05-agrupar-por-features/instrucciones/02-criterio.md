# Criterio — escrito el 2026-10-05, ANTES de calcular ningún grupo

No se ha corrido ningún agrupamiento: nada de lo de abajo se decidió mirando uno. **Segunda versión del mismo día**,
reescrita a raíz del `revisor` y del `verificador` antes de calcular nada (la primera está en `6d42bda`): la primera
podía dar por refutado el ejemplo del dueño justo cuando se cumplía, y tres de sus hipótesis salían por construcción.

Lo único mirado antes de escribirlo: la **firma por clase** que `feat-ind32` y `feat-ind` ya guardaron (qué fracción de
cada dígito enciende cada detector; está en `REGLAS.md` §2), el orden de las etiquetas en el fichero (14 bloques en
`windep`), que los índices de `errores-c.json` son los de este dataset (67 de 67 casan en etiqueta y origen) y la
`particion()` con que C eligió sus dígitos de test. Si al correr hace falta cambiar algo, va como **enmienda fechada** al
final.

**Todo se evalúa en los dos bancos principales, Z32 y Z8** (D1 de `REGLAS.md`), con K = 30 y la partición de menor
inercia de las 5 semillas, salvo donde se diga. **X8 (píxeles) es el control de todas.** Las definiciones (grupos de
*c*, corte, persistir, corte de forma, grupo mezclado, enriquecimiento, κ, los índices de L2b) son las de `REGLAS.md` §5.

## Comprobaciones de cordura — no son hipótesis: si fallan, algo está roto y no se lee nada más

- **C1** · Las copias de los detectores reproducen la firma por clase de origen, exacta, y la `particion()` copiada da
  la `huella_particiones` de `feat-ind32`.
- **C2** · NMI(Z, K = 30) ≥ 10 veces la NMI con las etiquetas permutadas: los grupos no son ruido.

⚠ El desplazamiento **no** es una comprobación de cordura, aunque lo parezca: Z lo **tolera, no lo garantiza**. Un rasgo
de una celda que cruza de la columna 3 a la 4 cambia hasta 4 coordenadas de Z y sólo 2 de M32, aunque 4 de los 7 pasos
posibles entre columnas vecinas no le cambian a Z ninguna. Por eso va como lectura en H4, sin umbral.

## H0 · ¿Más cerca o más lejos de las etiquetas que los píxeles?

Con el mismo K —la pureza y la NMI crecen con K por sí solas—: **NMI(Z) − NMI(X8) ≥ +0,03 en K = 10 y en K = 30 → «más
cerca»; ≤ −0,03 en los dos → «más lejos»; si no, empate.** No hay resultado bueno ni malo: se pidió saber cuánto
coinciden, y esto lo dice contra la referencia sin features.

## H1 · El ejemplo del dueño: el 1 se parte por la bandera

Sobre **todos** los 1, caigan en el grupo que caigan: si los 1 con bandera se van a un grupo de mayoría 7, eso es el
ejemplo en su forma más fuerte, no un fallo.

Se **confirma** si hay al menos un **corte del 1 que persiste** en K′ = 20 y 50 y en el que la **bandera** *b* separa a
los 1 de un lado y del otro con **AUC ≥ 0,80, mejor que el grosor y que la inclinación**. Se **refuta** si el 1 no tiene
ningún corte que persista, o si los que persisten los separa mejor el grosor o la inclinación; entonces se dice cuál.
Para X8 se reporta lo mismo, sin umbral.

⚠ *b* se mide en los píxeles, **sin detectores**: es el testigo independiente. Medido con los mapas, H1 sólo diría que
los grupos se parecen a la descripción con la que se formaron.

## H2 · Hay cortes de forma más allá del 1

**≥ 4 de las 10 clases con al menos un corte de forma** (persiste en K′ = 20 y 50, y ni el grosor ni la inclinación lo
explican). Con 30 grupos para 10 dígitos alguna clase se parte por fuerza, y una nube alargada se parte igual en cada
semilla; lo que no es por fuerza es que el corte sobreviva a cambiar K y que no sea de grosor.

⚠ Fuera del 1 no hay testigo **positivo** de forma: el grosor y la inclinación sólo descartan. Lo demás lo dicen las
rejillas, y hay que mirarlas.

Y se reporta, sin umbral, **K = 10**: con tantos grupos como etiquetas, si una clase se parte, otras tienen que
compartir grupo. Ahí se ve dónde una forma pesa más que una etiqueta.

## H3 · Los grupos mezclados explican los errores de C, y más que los de píxeles

**Enriquecimiento(Z) ≥ 2, con ≥ 5 de las 67 evaluaciones falladas en grupos mezclados, y enriquecimiento(Z) >
enriquecimiento(X8).** Se cuenta por evaluación: el denominador son las 6000 evaluaciones de C (cada dígito, tantas veces
como se evaluó), no todo el pool. La lista de pares que comparten grupo se reporta.

⚠ La comparación con X8 es la que da sentido a esto: en `feat-ind`, el 68 % de los fallos del compositor también los
fallaba un lineal sobre píxeles. Si los grupos de píxeles explican los errores igual, lo que se ve es que esos dígitos son
difíciles, no que las features expliquen nada.

## H4 · Agrupa por forma, no por grosor — sin etiquetas

Con κ (consistencia corregida por azar) al **engrosar 2 px** y al **adelgazar 1 px**: **κ(Z) − κ(X8) ≥ +0,05 en los dos
→ confirmada («la forma aguanta el grosor mejor que los píxeles»); ≤ −0,05 en alguno → refutada; si no, empate.** El
desplazamiento de 4 px se reporta (Z, M32 y X8), sin umbral: ver la nota de las comprobaciones de cordura.

## H5 · Son de los dígitos: no de la semilla ni de los escritores

- **Semilla:** ARI medio entre las 5 semillas (10 pares) **≥ 0,60**.
- **Escritores nuevos:** ARI(los 1797 asignados desde los centroides de los 3823, los 1797 agrupados aparte) **≥ 0,8 ×**
  el ARI entre semillas; y **≥ 24 de los 30** grupos de los 3823 reciben **≥ 1 %** de los 1797.

El umbral de escritores va **relativo** al de semillas a propósito: dos k-means sobre los mismos datos ya discrepan, y ese
desacuerdo es el suelo contra el que se mide cualquier otro. Se reportan, sin umbral, el ARI entre brazos (Z32–Z8 mide
la resolución, no independencia) y los grupos concentrados en un bloque de `windep` (posible estilo de una persona).

## H6 · *(opcional, D3)* Sirven para elegir qué etiquetar, y mejor que los de píxeles

Con K medoides etiquetados, el compositor lineal sobre M32 acierta en los 1797 **al menos 2 puntos más con los medoides
de Z que con los de X8, y al menos 3 más que con K al azar** (media de 5 sorteos), en **al menos 2 de K ∈ {20, 30, 50}**.
Se reportan además K = 10 y 100, el azar estratificado y la propagación.

⚠ Elegir medoides le gana al azar con casi cualquier agrupamiento, porque cubre mejor las clases. Por eso el rival que
cuenta es X8: sin él, la mejora no se le podría atribuir a las features. El estratificado mira las etiquetas para elegir
(garantiza K/10 por clase, como los datasets balanceados de las curvas de hoy), y por eso sale como referencia y no como
rival.

## L7 · Qué manda — sin umbral, salvo el disparador de Z−a

Se reporta qué fracción de la separación entre grupos aporta cada feature y cada zona. **Si los 4 arcos aportan más de
la mitad en un banco, se corre Z−a** (ese banco sin arcos) y H1, H2 y H4 se reportan también con él (`REGLAS.md` §5).

## Lo que espero hoy (antes de mirar)

- **C1 y C2** salen.
- **H0** empate: la coincidencia con las etiquetas no es lo que separa las dos descripciones; lo que las separa es **por
  qué** se parte cada clase.
- **H1** es la que más dudo, y ahora más que en la primera versión. En contra: los arcos se encienden en el 70–77 % de los
  1 **en los dos bancos**, así que si los 1 se parten, el candidato obvio es el grosor (el 1↔8 de `feat-ind`). A favor
  sólo en el de 32: su «/» se enciende en el 28 % de los 1, pero **no sé qué ve**, porque en los 7, cuyo trazo es un «/»,
  sólo en el 12 %. Espero **refutada en Z8, por grosor**, y una moneda al aire en Z32.
- **H2** al borde: el 7 (con y sin travesaño) y el 2 (con y sin lazo) como cortes de forma; el 4 abierto/cerrado,
  probablemente descartado por grosor (los 4 de `windep`, tal como los ve el banco de 8×8, son «un triángulo relleno con
  un palo debajo»: corrida 4 de `feat-ind`). Espero 3–5 clases. El dataset se hizo en la Universidad del Bósforo (Estambul; `optdigits-orig.names`);
  que los 43 escritores fueran de allí **no lo dice**, es un supuesto, y si lo eran, el 7 cruzado y el 1 de bandera
  larga, que son la forma continental, tendrán peso.
- **H3** sí en Z, y por encima de X8: C se construyó sobre estos mismos detectores, así que sus confusiones deberían
  vivir donde las features confunden. Sus pares más repetidos, contando cada dígito una vez: 8→1 ×7, 3→5 ×5, 9→4 ×4 y,
  empatados a ×3, 0→8, 3→8, 4→9, 9→1 y 9→3 (README de `feat-ind32`). Espero grupos mezclados 4–9 y 3–5, que están
  entre esos pares, y 1–7 por la bandera larga — éste **no** lo apoyan los pares de C (7→1 fue sólo ×2): es la apuesta
  de H1, no un dato.
- **H4** refutada, por el lado de **adelgazar**: con el compositor, al adelgazar 1 px los detectores de 32 dieron 0,720 y
  los píxeles crudos 0,875; al engrosar 2 px, 0,615 contra 0,578 (C6 de `feat-ind32`). Los detectores sólo vieron
  grosores de 2–4 px, y adelgazar o engrosar los saca de ese rango; los píxeles no tienen rango que perder.
- **H5**: ARI entre semillas 0,60–0,75; los escritores nuevos, sí.
- **H6** contra el azar, sí; contra los medoides de X8, **empate**: lo que más gana al elegir es cubrir las clases, y eso
  lo hace cualquier agrupamiento razonable.
- **L7**: los arcos y la recta vertical, arriba del todo en los dos bancos. No sé si pasarán de la mitad.
- **Lo que más probablemente salga mal:** que el **grosor mande sobre la forma** —cortes de «trazo gordo / trazo fino»
  en vez de «con bandera / sin bandera»—. H1, H2 y H4 lo tienen como rival explícito, y Z−a es la salida si son los arcos.

Si sale al revés en cualquiera de estos puntos, eso es un resultado y se escribe tal cual.
