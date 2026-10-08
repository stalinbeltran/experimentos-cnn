# Criterio de `rect-bor` — escrito ANTES de medir (2026-10-08)

Definiciones: las de `rect-lin` (detecta = logit > 0; recall = detecta y orientación correcta; FP por tipo; medias de 3
semillas), más **grueso-largo** = grosor 10–14 con largo 16–22. «Control» = mismo k, sin filtro, de `rect-lin`.
«La mejor» = el pre-proceso y k con mayor recall grueso-largo a N = 1000 entre los que cumplen G2.

| # | hipótesis | se cumple si |
|---|---|---|
| **G1** el filtro trae las gruesas | con algún filtro y algún k, N = 1000 | recall grueso-largo ≥ 0,80 (el control de `rect-lin` está en ~0,10) |
| **G2** sin perder las finas | ese mismo brazo | recall fino ≥ control − 0,03 |
| **G3** sin subir los FP | ese mismo brazo | FP total ≤ 0,05. ⚠ Espero que **suba el de manchas**: un disco pasa a ser un ANILLO, o sea una curva |
| **G4** sin más muestras | ese mismo brazo | N90 ≤ el N90 del control con el mismo k |
| **G5** el grosor medio | ese mismo brazo | recall medio (6–8 px) ≥ 0,85 |
| **G6** contorno ≥ sobel | mismo k, N = 1000 | recall grueso-largo de contorno ≥ el de sobel. Predicción: el contorno da 1 px limpio; Sobel, bordes de ~2 px a los dos lados |
| **G7** no hace falta aprender | el Gabor (sin entrenar) con el filtro de la mejor | recall grueso-largo ≥ 0,70 |

**Si G1 falla**, la idea «el filtro de bordes resuelve el grosor» no se sostiene con este detector, y es un resultado.
**Si G1 y G2 se cumplen pero G3 no**, el filtro sirve y el coste es un tipo de falso positivo concreto, que se nombra.

**Lo que NO contesta:** si con esto se leen mejor los dígitos; el trazo grueso en curvas y punteadas sólo se reporta.
