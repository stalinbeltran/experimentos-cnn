# Encargo (el dueño, 2026-10-10, conversación de Telegram)

> «Crea otro experimento, que use las mismas entradas que en
> https://raw.githubusercontent.com/stalinbeltran/experimentos-cnn/9f0becc/docs/bocetos/2026-10-09-curvas-gabor/imagenes/19-fallos-gris.png
> (pues se ven bien), y cuyo objetivo sea aprovechar estas formas para obtener el mismo dígito, pero más delgado. Usa
> este mismo set de dígitos del url. Dame alternativas para obtener el dígito delgado.»

Contexto que el encargo da por sabido, y que se escribe aquí para no tener que ir a buscarlo:

1. **«Las mismas entradas»** son las de la figura 19 del boceto: el dígito pasado por un banco de **Gabor impar**
   (λ_b 10, kernel 9×9, 8 orientaciones), y su borde = max |respuesta| normalizado por la respuesta a un escalón ideal,
   **sin binarizar**. En esa figura los trazos gruesos salen como una banda oscura con una **línea clara en el medio**:
   los dos bordes del trazo responden y en el centro la respuesta se cancela. Esa línea clara es justo el «dígito
   delgado» que se busca, y es lo que «aprovechar estas formas» sugiere usar.
2. **«Este mismo set de dígitos»**: los 64 dígitos de val que falla aquel compositor (semilla 0). La lista se copia en
   `entradas/digitos-64.json`; el dato se lee del dataset publicado.
3. **La regla del dueño del mismo día** (`CLAUDE.md` del repo): el borde se saca con Gabor, **nunca con morfología**.
   No habla del adelgazamiento, pero su espíritu es el mismo: por eso el esqueleto morfológico clásico entra aquí **sólo
   como referencia** para comparar, nunca como candidato, salvo que el dueño diga otra cosa.
4. **«Dame alternativas»**: el objetivo de esta primera vuelta es enseñar varias y medirlas con un criterio escrito
   antes, para que el dueño elija; no declarar una ganadora.
