# Encargo — 2026-10-02

> «en el repo de experimentos genera un experimento donde vamos a crear cnn para reconocimiento
> de dígitos NIST 8x8. Este dataset lo vamos a subdividir en un 10% para training y un 90% para
> validación, y vamos a probar la capacidad de las cnn para generalizar (por eso hay mucho menos
> training que validación). Las capas van a ser siempre las mismas (por ej 4), sin padding,
> siempre con el mismo kernel 3x3, pero vamos a hacer evaluaciones paralelas del 'mejoramiento'
> del dataset. Vamos a aplicar distintos tipos de 'ruido' para modificar el dataset de training
> (nunca validación) para evaluar cuál tipo de 'ruido' colabora mejor con el aprendizaje. Algunos
> ruidos van a eliminar pixeles aleatorios en distintas proporciones, otros van a introducir
> pixeles externos, otros van a introducir lineas horizontales, otros verticales, otros oblicuas,
> otros curvas, cada uno con distintas transparencias (entre 100% y digamos un 20%). La idea es,
> para cada ruido por separado obtener una estimación de su contribución a la mejora de la
> capacidad de generalización de la nn. Podemos evaluar varias semillas (tal vez 3) para generar
> los pesos de las cnn, pero una vez generados los pesos deben ser idénticos para todos los
> escenarios (de modo que eliminamos esta variable, y sólo el ruido sea evaluado).
> Inicialmente podemos evaluar primero 'tipos' de ruido (por ej rectas, curvas, recortes, ruido
> specular, etc) para tener una aproximación de cuál de ellos genera el efecto más significativo.
> Documenta el plan. No ejecutes aún»

Las lecturas que el dueño tiene que confirmar están en `ESPECIFICACION.md` §0.
