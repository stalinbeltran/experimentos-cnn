# Encargo — 2026-10-01

## Lo que pidió el dueño, literal

Primero el plan:

> «Ahí tenemos un banco de kernels. Ahora necesito que crees varios experimentos para
> generar los kernels de los distintos tamaños esperados por el banco. Detalla el plan de
> implementación. Debes usar servers contratados para poder correr esos experimentos en
> paralelo. Aún no ejecutes, solo arma el plan.»

y, sobre el plan ([`docs/plan-kernels-banco-2026-10-01.md`](../../docs/plan-kernels-banco-2026-10-01.md)):

> «En el repo de experimentos, doc 'plan kernels banco 2026...' está el plan para ejecutar
> un estudio de kernels. Ejecutalo. Decisiones: 1, recomendado. 2, un kernel. 3, 3
> semillas. 4, Todos. 5, recomendado.»

> «Usa tus sugerencias. Arranca»

## Qué le toca a este experimento

La **fase 0a** del plan y nada más: el dato de entrada de `bor-k`, `bor-ae` y `bor-pca`.

La **decisión 1** (escala) quedó en la recomendada: *dataset nuevo a /4 con la geometría ×4,
porque es la escala del banco y de toda la familia `esq-*`*. El motivo, del §1 del plan: el
banco aplica el kernel **después** de reducir /4, así que la tinta que ve tiene 2,75–7,5 px de
cuerpo; `bor-p` está a 1024 sin reducir (11–30 px), y reducirlo tal cual deja párrafos a 10,5
px unos de otros, por debajo de lo que pide un kernel de 19.

## Cómo se leyó, y en qué se convirtió

| lo que dice el plan | en qué se convirtió | dónde vive |
|---|---|---|
| «geometría ×4 en render» | separación 160 y margen 128 en px de render (×4 de los 40 y 32 de `bor-p`) | `nn/geometria.py` |
| «guardado a /4 por promedio por área (el mismo método que el banco)» | `uint16` con la **suma** del bloque 4×4, que es exactamente el §3.8 del banco | `reducir()` |
| «las cajas se transforman con la misma aritmética» | se guardan en px de render; en la guardada valen `coord / 4` | manifiesto |
| «caben menos párrafos por página: se ajusta el lienzo o se acepta, y se dice» | **se acepta**, con 2–4 por página en vez de 2–5, y se dice en `REGLAS.md` y en el manifiesto | `PARRAFOS_POR_PAGINA` |
| «la reserva §3.7 se sigue comprobando antes de rendir» | `comprobar_reserva()` sin cambios | `nn/generar_paginas.py` |
| «nombre nuevo» | `parrafos1000-pagina1024-r4-r20261001` | `NOMBRE` |
