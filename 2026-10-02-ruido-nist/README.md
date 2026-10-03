# `ruido-nist` — ¿qué ruido de entrenamiento ayuda a generalizar en dígitos de 8×8?

**Fase 1 corrida el 2026-10-03 00:59 → 01:18 UTC en el dev (0 $, unidad `expc-ruidonist`,
`Result=success`, `NRestarts=0`), 33 corridas: `limpio` + 9 tipos a su nivel medio +
`oblicua@0.6-r2`, × 3 semillas.** 180 imágenes para entrenar (+180 copias), 1617 limpias para
validar, pesos iniciales idénticos por semilla. Criterio escrito antes
([`instrucciones/02-criterio.md`](instrucciones/02-criterio.md)), aplicado tal cual por
`nn/informe.py` ([`resultados/RESULTADOS.md`](resultados/RESULTADOS.md)). **Estado: abierto;
la fase 2 (intensidades) no se ha lanzado.**

> **Veredicto de la fase 1:** de los nueve ruidos a intensidad media, **sólo `recorte` (cutout de
> ¼–½ del lado, α = 0,6) ayuda**: +0,029 de exactitud de val sobre `limpio` (0,870 → 0,899),
> pareado por semilla, con umbral 0,021. **Ninguno perjudica.** Los otros ocho quedan
> **indistinguibles** (|Δ| ≤ 0,010 todos). Y un hallazgo que el criterio no pedía: **casi todos
> los ruidos bajan la entropía cruzada de val** (`recorte` −0,39, `sal-pimienta` −0,19,
> `gaussiano` −0,14, `vertical` −0,12, `oblicua` −0,11 nats) sin mover la exactitud: la red
> acierta igual pero **se equivoca con menos seguridad**.

| escenario | acc val (media ± sd) | **Δ acc val** ± SE | umbral | Δ CE val | veredicto |
|---|---|---|---|---|---|
| `limpio` (base) | 0,870 ± 0,032 | — | — | — | CE val 1,037 |
| **`recorte@0.6`** | **0,899 ± 0,025** | **+0,029** ± 0,011 | 0,021 | **−0,387** | **ayuda** |
| `vertical@0.6` | 0,880 ± 0,030 | +0,010 ± 0,005 | 0,010 | −0,120 | indistinguible |
| `borrado@0.2` | 0,879 ± 0,027 | +0,008 ± 0,006 | 0,011 | +0,016 | indistinguible |
| `sal-pimienta@0.1` | 0,878 ± 0,031 | +0,007 ± 0,011 | 0,021 | −0,190 | indistinguible |
| `oblicua@0.6` | 0,877 ± 0,027 | +0,006 ± 0,004 | 0,010 | −0,109 | indistinguible |
| `gaussiano@0.1` | 0,874 ± 0,047 | +0,003 ± 0,010 | 0,020 | −0,136 | indistinguible |
| `curva@0.6` | 0,873 ± 0,041 | +0,003 ± 0,005 | 0,011 | −0,062 | indistinguible |
| `externos@0.05` | 0,870 ± 0,035 | −0,000 ± 0,003 | 0,010 | −0,038 | indistinguible |
| `horizontal@0.6` | 0,860 ± 0,044 | −0,010 ± 0,007 | 0,014 | +0,003 | indistinguible |
| `oblicua@0.6-r2` (2ª copia) | 0,865 ± 0,063 | −0,005 ± 0,018 | 0,037 | −0,060 | indistinguible |

Exactitud de train (180 limpias) = 1,000 y CE de train ≈ 0,001 en **todos** los escenarios: el
mecanismo no se pudo leer ni en CE (ningún `Δce_train` llega a 0,05). Figura:
[`resultados/delta-por-tipo.png`](resultados/delta-por-tipo.png); el ruido, para mirarlo:
[`resultados/muestras-ruido.png`](resultados/muestras-ruido.png).

## Lo que se lee, con el criterio

- **`recorte` es el único que supera el umbral**, y lo hace con margen (+0,029 contra 0,021) y en
  las tres semillas (+0,009, +0,034, +0,044). Es el ruido que **quita** más información por imagen
  (un cuadrado de ¼–½ del lado): obliga a reconocer el dígito por partes. Y es el que más baja la
  CE de val.
- **`vertical` y `borrado` rozan el umbral** (+0,010 y +0,008 contra 0,010 y 0,011): con 3 semillas
  no se distinguen; con 5 podrían. `horizontal` apunta al otro lado (−0,010).
- **La realización**: `oblicua@0.6-r2` contra `oblicua@0.6` da Δ = −0,011 ± 0,022. El criterio dice
  «no se distingue», pero ⚠ **el SE de esa comparación (0,022) es del tamaño de la amplitud entre
  tipos (0,039)**: la prueba de la realización está tan corta de poder como el resto. Una copia fija
  sirve para `recorte` (su efecto es 3× ese ruido); para los que rozan el umbral, no se sabe.
- **Pasan a la fase 2** por la regla escrita (ayuda, o indistinguible con media > 0): `borrado`,
  `vertical`, `oblicua`, `curva`, `recorte`, `gaussiano`, `sal-pimienta` — 7 tipos, 84 corridas
  nuevas (~30 min, 0 $). ⚠ La regla es permisiva a propósito (no descartar por falta de poder); el
  que de verdad tiene señal es `recorte`.

## Lo que NO dice

- Nada sobre combinar ruidos, ni sobre otras intensidades (eso es la fase 2), ni sobre ruido en
  línea (una realización por época; pendiente de S2).
- Con 3 semillas, un efecto por debajo de ~0,01 no se puede declarar; se dijo antes de mirar.
- 13 escritores compartidos entre train y val (ver `dim-nist`).

## Coste, reloj y dónde está

- **0 $**, 19 min de reloj (20–22 s por corrida, 2 hilos). `NRestarts=0`.
- Métricas, resúmenes, logs, informe y figuras: aquí (rama `tema-2`, `4b17d5f`). **Pesos (`last.pt`),
  `init/` y todo lo demás**: en el almacén, `foveal-vision-data/experimentos-cnn-resultados/ruido-nist/`
  (`b888e6c5`).
- El aviso a Telegram del cierre **no salió** (`Falta BOT_TOKEN`): la unidad se lanzó desde una
  sesión de Claude Code, no desde el bot. El `|| true` hizo su trabajo: el cierre terminó entero.

## Cómo se repite

```bash
cd 2026-10-02-ruido-nist
python nn/probar.py && python nn/entrenar_local.py --comprobar && sh nn/probar_lanzador.sh
nn/lanzar.sh fase1                 # 33 corridas; las que ya tienen summary.json se saltan
nn/lanzar.sh fase2 recorte …       # los tipos de resultados/criterio-aplicado.json → fase2
nn/lanzar.sh --estado
```
