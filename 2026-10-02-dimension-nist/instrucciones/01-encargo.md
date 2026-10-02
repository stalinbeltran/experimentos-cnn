# Encargo — 2026-10-02

## Lo que pidió el dueño, literal

> «Prepara otro experimento identico a este con NIST como entrada. Las reducciones aquí deben
> ser menos dramáticas, ej de 8x8 (creo q es el tamaño original) pasamos hasta 4x4, lo q nos
> deja máximo de 4 medidas. Mismos criterios, queremos hallar cual resolucion nos da la mejor
> generalización»

«Este» es `dim-gen` (el experimento de párrafos de 128 px), cerrado ese mismo día.

## Cómo se ha leído, y qué tiene que confirmar el dueño antes de lanzar

| término | cómo se lee aquí | la otra lectura |
|---|---|---|
| **«NIST»** (S1) | los **dígitos manuscritos de 8×8** de UCI (*Optical Recognition of Handwritten Digits*), la copia que trae scikit-learn (`load_digits`, 1797 imágenes, valores 0..16, 10 clases). Vienen de los bitmaps de 32×32 que extrajeron los programas de preprocesado de **NIST**, contados en bloques 4×4 — por eso «8×8 es el tamaño original» de ese dataset | **MNIST** (28×28, 70.000 imágenes, también de NIST). Si era eso, el plan cambia: 28 → 14 → 7 y otra escalera. **No se ha preparado** |
| **«de 8×8 hasta 4×4, máximo de 4 medidas»** (S2) | el original **más cuatro reducciones**: `W ∈ {8, 7, 6, 5, 4}`, por promedio de área con pesos exactos (8→4 es la media de bloques 2×2; 7, 6 y 5 son fraccionarias pero exactas) | «cuatro medidas en total»: quitar una (p. ej. 5) |
| **«idéntico a este»** (S3) | mismo diseño: kernel = fracción fija del ancho (`f = 1/L`), sin padding, 10 % / 90 %, 5 semillas, 4000 pasos, un `lr`, criterio y control. **Pero `L = 4` no cabe en 8 px** (cuatro capas sin padding quitan ≥ 4 px): `L = 2`, `n = ⌈W/2⌉`, y como el mapa final es 2×2 (W par) o 1×1 (W impar) la cabeza es un **promedio global + Linear(C → 10)**, constante entre W | sólo `W ∈ {8, 6, 4}` con `n = W/2` exacto y `Flatten` como `dim-gen` (3 puntos, sin promedio global) |
| **«mismos criterios»** (S4) | el criterio de `dim-gen` con **exactitud** en vez de IoU (es clasificación): piso = clase mayoritaria, umbral `max(2·SE_dif, 0,01)`, W\*, W mínimo suficiente, «estorba», brecha, la descomposición en información / generalización / parámetros y el control | — |
| «generalización» | exactitud sobre las **1617** imágenes no vistas, leída junto con la de train y la brecha. ⚠ Las 1797 son de **13 escritores**: se mide generalizar a dígitos nuevos de los mismos escritores, no a escritores nuevos | — |
