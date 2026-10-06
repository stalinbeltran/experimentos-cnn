# Reglas de `feat-cortas`

**Escritas el 2026-10-06, antes de entrenar.** Criterio en `instrucciones/02-criterio.md`.

**Qué pregunta:** si los detectores aprenden primitivas CORTAS —rectas y curvas— en vez de las 13 features largas de
`feat-ind32`, ¿las aprenden, distinguen curva de recta, y predicen los dígitos al menos tan bien?

## Entradas

- **Dataset:** `feat-cortas-sinteticas-32px-r20261006`, publicado desde aquí (`nn/datos.py --generar --publicar`), leído
  con `exigir_dataset`. 8 familias + `vacio` a 32×32 binario:
  - **rectas cortas** `corta-V`, `corta-H`, `corta-S` (/), `corta-B` (\): largo **6–12 px** *(decido: las largas de
    feat-ind32 iban de 10 a 28)*, ángulo nominal ± 12°;
  - **curvas cortas** `curva-E/W/N/S`: arcos de radio **5–10 px** y apertura **60–100°** *(decido: 5 a 17 px de largo,
    del orden de las rectas cortas)*, con el centro a E/W/N/S ± 40°.

  Grosor 2–4 px, ruido (`p_on` ≤ 0,02, `p_off` ≤ 0,10) y una segunda feature en la mitad: lo de feat-ind32. 2000 + 400 por
  familia; `vacio`, 1000 + 200. Semillas 20261006 + 10·familia (+1 en val). Ninguna familia contiene a otra.
- **Los dígitos:** `uci-optdigits-orig-32px-r20261005` (los 1797 de `windep` con su reparto 180/1617).
- **Al cargar:** 0/1 a float32. Para predecir, el dígito pasa por `norm3` (esqueleto + 3 px, copiado de `feat-fallos`),
  y se reportan también crudo y las dos vistas.

## Salidas

- **Pesos:** `nn/pesos/<f>/` (`best.pt`, `config.json`, `summary.json`; `last.pt`, `metrics.jsonl` y `log.txt` fuera de
  git). 8 × ~170 KB ≈ 1,4 MB.
- **Evaluación:** `resultados/evaluacion.json` y `resultados/cortas.png`.
- **El libro de Vast:** `resultados/vast/detectores/` (lo commitea `nn/vast.sh` al alquilar).
- **Reporte en el central:** sí, al terminar (alquila).

## Procesos

1. Generar y publicar el dataset; mirar la rejilla (`nn/features.py --muestras`).
2. `nn/entrenar_local.py --comprobar`, `nn/probar_vast.sh`, `VAST_SECO=1 nn/vast.sh detectores`.
3. **Entrenar en Vast** (`nn/vast.sh detectores`): los 8 a la vez en una máquina de 8–24 vCPU, receta de feat-ind32 sin
   cambios (80 épocas, lotes 32 + 32, lr 2·10⁻³, umbral en train, `best` por F1 de val, semilla 1).
4. **Evaluar** (`nn/evaluar.py`): §A y H1-cv de los `summary.json`; §B en el dev.
5. README, reporte en el central, commit y push.

- **Brazos y semillas:** 8 detectores × 1 semilla; el compositor, 3.
- **Qué se llama «ganar»:** H2 contra la referencia de las 13 largas.

## Scripts

| script | qué hace | cómo se llama |
|---|---|---|
| `nn/features.py` | el vocabulario corto y el rasterizador (copia del de feat-ind32 con otras familias y medidas) | `python nn/features.py [--muestras]` |
| `nn/datos.py` | genera, publica y carga el dataset; los conjuntos de cada detector | `--generar [--publicar]` · `--comprobar` |
| `nn/modelo.py` | el detector de feat-ind32, sin cambios | — |
| `nn/entrenar_local.py` | entrena UN detector (el nombre es el contrato con el freno); receta de feat-ind32 | `--feature f [--hilos 1]` · `--comprobar` |
| `nn/normalizar.py` | esqueleto + 3 px (copiado de feat-fallos) | `python nn/normalizar.py` |
| `nn/evaluar.py` | §A, H1-cv y §B | → `resultados/evaluacion.json`, `resultados/cortas.png` |
| `nn/vast.sh`, `nn/vast.json`, `nn/probar_vast.sh` | Vast, un modo; freno `apagar` y `/use exp-vast` | `detectores` · `--estado` · `apagar` |

- **Dependencias:** el `.venv` de la raíz (numpy 2.5.3, torch 2.14.1 CPU, Pillow 12.3.0, matplotlib 3.10.9).

## Qué NO hereda

- **Se copió de:** `feat-ind32` (features.py con otro vocabulario, modelo.py, entrenar_local.py, el compositor y el reparto
  de la curva) y `feat-fallos` (normalizar.py, vast.sh). Se copia, no se importa.
- **Qué se cambió a propósito:** el vocabulario (8 primitivas cortas, sin `CONTIENE`), y la vista principal al predecir
  (`norm3`, porque los detectores son finos y los dígitos gruesos: lo midió feat-fallos).
- **Qué se conservó, y por qué:** la red, la receta, el ruido y el compositor, para que la única diferencia con las 13 largas
  sea el vocabulario.
- **Contra qué se compara:** las 13 features largas de feat-ind32 (por id y huella).
- **Restricciones de otros experimentos que NO aplican aquí:** la relación `CONTIENE` y los tramos de radio de feat-ind32.
