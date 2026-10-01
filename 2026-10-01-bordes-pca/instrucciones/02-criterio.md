# Criterio — escrito el 2026-10-01, ANTES de calcular ninguna componente

Lo que diga aquí manda sobre lo que se vea después (R13).

- **Qué se calcula:** por cada `k ∈ {3, 5, …, 19}`, las dos primeras componentes principales
  de los parches `k × k` de **`train`** centrados en puntos de borde (8 por borde y párrafo,
  los cuatro bordes mezclados), con la media del conjunto restada.
- **Qué va al banco: el PC1 de cada `k`, y sólo el PC1.** Los nueve, sin filtrar. El PC2 se
  guarda en el mismo checkpoint para poder **verlo**, y **no** se importa: decidirlo después
  de mirarlo sería elegir el kernel por su resultado.
- **Qué se reporta:** la fracción de varianza que explica cada componente y su forma
  (`resultados/componentes.png`). No hay métrica de «aprendió»: no se entrena nada.
- **No se declara ganador.** El veredicto de utilidad es el del banco (§2.1 de su
  especificación).

**El riesgo, escrito antes.** Con los cuatro bordes mezclados, la dirección de mayor varianza
puede no ser un escalón sino **«cuánta tinta hay en el parche»** —un pasa-bajos—, porque es lo
que más cambia de un parche a otro; los escalones (horizontal y vertical) quedarían en el PC2
y el PC3. Si sale eso, es el resultado: al banco entra el PC1 igual, y la figura enseña dónde
quedaron los escalones.
