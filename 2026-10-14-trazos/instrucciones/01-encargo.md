# Encargo (el dueño, 2026-10-13, conversación de Telegram)

> «Tú sabes que estos trazos a mano son hechos con un bolígrafo. ¿Hay algún algoritmo que nos permita obtener los
> trazos empleados? […] Prepara y documenta este experimento para mañana. Luego dime cómo podría emplearlo para
> facilitar la detección. Si conocemos los trazos podemos conocer el dígito, pero veo que obtenerlo puede ser un
> problema más complicado.»

Lo que el encargo da por sabido, escrito aquí:

1. **«Los trazos del bolígrafo»** son las curvas que recorrió la punta: cada una empieza cuando el bolígrafo toca el
   papel y acaba cuando se levanta. En la imagen se ven como una mancha de grosor casi constante (3–6 px en UCI a
   32 px); donde dos trazos se cruzan o un trazo se cruza a sí mismo (el 8), la mancha se funde. Lo que se pierde al
   escanear es el **orden** y el **sentido**: de una imagen estática no se puede saber si el 1 se escribió de arriba
   abajo. Este experimento recupera **forma, extremos, cruces y cuenta**, no el orden.
2. **«Más complicado»**: es un problema con nombre (recuperación de la trayectoria a partir de escritura *offline*;
   Doermann y Rosenfeld 1995, Kato y Yasuhara 2000, Lau, Yuen y Tang 2005 — citados de memoria, NO comprobados). La
   familia que no entrena es: línea media → grafo de extremos y cruces → en cada cruce, continúa la pareja de ramas
   con menor cambio de dirección. Se elige ésa porque no necesita verdad *online* (UCI no la tiene) y porque reutiliza
   lo que ya hay: la transformada de distancia (huecos) y el campo de orientación del gris.
3. **Para qué** (la pregunta de después del encargo): un dígito descrito por sus trazos es la lista de «responsables»
   más corta posible —un 8 es *un* trazo con *un* cruce; un 4, dos o tres trazos; un 1, uno sin cruces—. Es la
   continuación de la línea «todo objetos» de `rizo-2` con un objeto de más alto nivel. ⚠ Y lo que el dueño ya ve: la
   misma clase se escribe con distintos trazos (un 4 abierto o cerrado, un 7 con o sin barra), así que la cuenta sola
   no identifica; identifica la **estructura** (cuántos, cómo se tocan, cómo giran).
4. **Verdad**: no hay. Se construye en dos capas, las dos antes de correr nada: (a) sintéticos con trazos conocidos
   (polilíneas y arcos pintados con grosor), que miden si el método hace lo que dice; (b) 50 dígitos de val (5 por
   clase) etiquetados a mano por Claude sobre una rejilla —número de trazos, extremos de cada uno y cruces— para medir
   en dígitos reales. ⚠ Son etiquetas de Claude, con el error de leer un bitmap: la figura que las enseña está para
   que el dueño las corrija.
5. **Gasto previsto: 0 $.** No se entrena nada; todo corre en este dev en segundos por dígito.
