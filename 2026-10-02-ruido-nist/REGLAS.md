# Reglas de `ruido-nist`

**2026-10-02 · SÓLO PLAN**: no hay código. Lo que aquí se fija es el contrato con el que se
escribirá; si cambia al implementar, se cambia aquí en el mismo commit.

**Qué pregunta:** con 180 imágenes de train y 1617 de validación limpias, la misma CNN y los
mismos pesos iniciales por semilla, ¿qué tipo de ruido aplicado sólo al train mejora la
exactitud de validación, y cuánto?

## Entradas

- **Dataset**: `uci-optdigits-8px-r20261002` (publicado por `dim-nist`), leído con
  `expcnn.exigir_dataset`. Su reparto 180 / 1617 tal cual. Validación **nunca** se toca.
- **Train de cada escenario** = las 180 originales + 180 copias con ese ruido (una realización fija,
  generada con la semilla de ruido del escenario y guardada con huella); `limpio` = las 180
  duplicadas. Semillas: pesos 1/2/3, lotes 100+s, ruido 1000+10·t+i (`ESPECIFICACION.md` §1 bis).
- **Transformación**: `x/16 ∈ [0,1]`. El ruido se dibuja a 32×32 y se reduce a 8×8 por conteo de
  bloques 4×4 (gaussiano y sal-pimienta, directamente a 8×8). Ver `ESPECIFICACION.md` §2.

## Salidas

- `nn/pesos/<escenario>-s<s>/`: `metrics.jsonl`, `summary.json`, `log.txt`, `last.pt` (los `.pt`
  no se commitean aquí; van al almacén). `nn/init/init-s<s>.pt` con su huella: los pesos iniciales
  que comparten todos los escenarios.
- `resultados/`: informe y figuras regenerados por `nn/informe.py`; `muestras-ruido.png`.

## Procesos

1. Código + pruebas: que el ruido nulo dé pesos finales idénticos bit a bit a `limpio`; que la
   validación no pase nunca por el ruido; que cada tipo produzca lo que dice (rejilla PNG).
2. Ensayo de mecanismo: sólo pérdida de train en `limpio` → congela `lr`.
3. Fase 1 (tipos), fase 2 (intensidades), como unidad de systemd en el dev.
4. Informe, README con veredicto, reporte en el central.

## Scripts (previstos)

`nn/datos.py` (carga), `nn/ruido.py` (los 9 tipos, deterministas por generador), `nn/modelo.py`
(3 capas 3×3 valid + GAP), `nn/entrenar_local.py --escenario <tipo>@<intensidad> --semilla s`
(el nombre es el contrato con el freno), `nn/informe.py`, `nn/lanzar.sh fase1|fase2 · --estado ·
SECO=1`, `nn/probar.py`.

## Qué NO hereda

- **Se copió de:** ninguno. Usa el dataset de `dim-nist`, y nada más de él.
- **Cambia respecto de `dim-nist`**: resolución fija (8 px), kernel fijo 3×3, 3 capas; la
  variable es el ruido de entrenamiento; 3 semillas en vez de 5 (pedido del dueño), compensado
  comparando **pareado** con pesos iniciales idénticos.
- **Se conserva, decidido**: reparto, 3996 pasos de lote 20, Adam con `lr` de ensayo, sin
  selección, `δ = 0,01`.
- **Contra qué se compara**: contra `limpio` de la misma semilla. Con `dim-nist` sólo de contexto.
