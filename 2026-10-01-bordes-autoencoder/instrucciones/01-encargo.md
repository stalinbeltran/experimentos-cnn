# Encargo — 2026-10-01

## Lo que pidió el dueño, literal

> «Ahí tenemos un banco de kernels. Ahora necesito que crees varios experimentos para
> generar los kernels de los distintos tamaños esperados por el banco. Detalla el plan de
> implementación. Debes usar servers contratados para poder correr esos experimentos en
> paralelo. Aún no ejecutes, solo arma el plan.»

y, sobre el plan ([`docs/plan-kernels-banco-2026-10-01.md`](../../docs/plan-kernels-banco-2026-10-01.md)):

> «En el repo de experimentos, doc 'plan kernels banco 2026...' está el plan para ejecutar
> un estudio de kernels. Ejecutalo. Decisiones: 1, recomendado. 2, un kernel. 3, 3
> semillas. 4, Todos. 5, recomendado.»

> «Usa tus sugerencias. Arranca»

## Qué le toca a `bor-ae`

El experimento **E2** del §3 del plan. Las decisiones que le afectan:

- **1, recomendado** — el dato es `parrafos1000-pagina1024-r4-r20261001` (`bor-p4`): la
  escala del banco (/4), con la geometría ×4 para que un kernel de 19 quepa.
- «4, Todos» — los tres experimentos, éste incluido; y «Usa tus sugerencias»: `λ` se calibra ANTES con un tanteo barato en el dev, con la regla escrita antes y capaz de detectar un kernel delta.
- **3, 3 semillas** — donde hay entrenamiento, tres semillas por `k`, para tener la
  dispersión del **productor** y no sólo la del banco.
- **5, recomendado** — los kernels ya entrenados de `esq-k`/`esq-cq` **no** se importan
  (todos tienen fuga §3.7).

Y de las sugerencias aceptadas el mismo día: las máquinas viven como mucho **3 h**
(`--horas-max`), una por `k`: nueve por experimento.
