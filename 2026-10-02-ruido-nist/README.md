# `ruido-nist` — ¿qué ruido de entrenamiento ayuda a generalizar en dígitos de 8×8?

**Corrido el 2026-10-03 en el dev (0 $), CERRADO en sus dos fases.** Fase 1 (00:59 → 01:18 UTC,
33 corridas): `limpio` + 9 tipos a su nivel medio + `oblicua@0.6-r2`, × 3 semillas. Fase 2
(03:56 → 04:29 UTC, 84 corridas): los 5 niveles de los 7 tipos que pasaron, × 3 semillas. **117
corridas** en total, una unidad de systemd por fase (`expc-ruidonist`, `Result=success`,
`NRestarts=0`, 0 fallos). 180 imágenes para entrenar (+180 copias), 1617 limpias para validar,
pesos iniciales idénticos por semilla. Criterio escrito antes
([`instrucciones/02-criterio.md`](instrucciones/02-criterio.md)), aplicado tal cual por
`nn/informe.py` ([`resultados/RESULTADOS.md`](resultados/RESULTADOS.md)).

> **Veredicto.** Contra `limpio` (exactitud de val 0,870 ± 0,032), **ningún ruido perjudica a
> ningún nivel**, y **cinco de nueve tipos tienen un nivel que ayuda**: **`recorte@0.6`** (cutout
> de ¼–½ del lado, α = 0,6) es el mejor con **+0,029** (0,899), seguido de `curva@0.8` (+0,021),
> `gaussiano@0.2` (+0,017), `vertical@1` (+0,014) y `oblicua@1` (+0,012). `borrado@0.4` (+0,019) y
> `sal-pimienta@0.2` (+0,018) son del mismo tamaño pero no se distinguen con 3 semillas.
> `externos` y `horizontal` no pasaron de la fase 1. **La forma general es «más ruido, mejor»**
> dentro del rango probado — los trazos suben hasta α = 1 (el eje no está acotado), `recorte` y
> `curva` tienen pico interior — y **`gaussiano` es el único con sobredosis** (σ = 0,3 vuelve a
> 0,000). Y lo que no pedía el criterio: casi todos bajan la **entropía cruzada de val** (hasta
> −0,39 nats con `recorte@0.6`) aunque la exactitud no se mueva: la red se equivoca con menos
> seguridad.

## Fase 1 — tipos, al nivel medio

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

Sólo `recorte` supera el umbral. Pasan a la fase 2, por la regla escrita (ayuda o indistinguible
con media > 0), siete tipos; quedan fuera `externos` y `horizontal`.

## Fase 2 — la intensidad dentro de cada tipo (Δ exactitud de val, pareado; \* = ayuda)

| tipo | nivel más suave → más fuerte | mejor | forma |
|---|---|---|---|
| `borrado` (p) | −0,003 · +0,010 · +0,008 · +0,009 · **+0,019** | `@0.4` (indist.) | sube; no acotado |
| `vertical` (α) | +0,003 · +0,005 · +0,010 · +0,007 · **+0,014\*** | `@1` | sube; no acotado |
| `oblicua` (α) | +0,004 · +0,001 · +0,006 · +0,006 · **+0,012\*** | `@1` | sube; no acotado |
| `curva` (α) | +0,005 · +0,004 · +0,003 · **+0,021\*** · +0,012\* | `@0.8` | pico interior |
| `recorte` (α) | +0,012\* · +0,011 · **+0,029\*** · +0,007 · +0,023 | `@0.6` | pico interior |
| `gaussiano` (σ) | −0,005 · +0,006 · +0,003 · **+0,017\*** · −0,000 | `@0.2` | pico, y cae a 0 en σ = 0,3 |
| `sal-pimienta` (p) | +0,001 · +0,011 · +0,007 · **+0,018** · +0,017 | `@0.2` (indist.) | pico interior |

De 38 escenarios, 7 dan «ayuda» y 0 «perjudica»; el peor Δ de todo el estudio es −0,010
(`horizontal@0.6`). Exactitud de train (180 limpias) = 1,000 y CE de train ≈ 0,001 en **todos**:
el mecanismo (regularización contra facilitación) **no se pudo leer** ni en CE.

Figuras: [`resultados/delta-por-tipo.png`](resultados/delta-por-tipo.png) (los 38 escenarios con
sus semillas); el ruido, para mirarlo: [`resultados/muestras-ruido.png`](resultados/muestras-ruido.png).

## Lo que se lee, y con qué cautela

- **`recorte` es el ruido que mejor colabora**, y lo es con margen (+0,029 contra umbral 0,021, las
  tres semillas positivas) y en la lectura de CE (−0,39, la mayor). Quita un cuadrado entero del
  dígito: obliga a reconocerlo por partes.
- **Los efectos son pequeños**: entre 1 y 3 puntos sobre 0,870, con 3 semillas. ⚠ Son **38
  comparaciones** a 2·SE sin corrección (el criterio no la fijó): con umbral `max(2·SE, 0,01)`,
  los «ayuda» de `vertical@1` y `oblicua@1` entran porque su SE es diminuto (0,002–0,004) y el
  umbral cae a δ; cabría esperar 1–2 falsos positivos por azar entre 38. **Los que no dependen de
  eso** son `recorte@0.6`, `curva@0.8` y `gaussiano@0.2` (Δ ≥ 0,017, 1,5–2× su umbral).
- **La realización** (`oblicua@0.6` con dos copias): Δ −0,011 ± 0,022, «no se distingue», pero ese
  SE es del tamaño de la amplitud entre tipos en la fase 1 (0,039): la prueba está tan corta de
  poder como el resto. Para `recorte@0.6` no importa (su efecto la triplica).
- **El eje de los trazos no está acotado**: `vertical` y `oblicua` suben hasta α = 1, que es el
  máximo de opacidad; lo siguiente sería **grosor** (hoy 1–2 px a 32) o **más trazos**, no más α.
  `borrado` sube hasta p = 0,4, también el borde.

## Lo que NO dice

- Nada sobre **combinar** ruidos, ni sobre ruido **en línea** (una realización nueva por época;
  pendiente de S2). Con lo medido, en línea es lo natural para `recorte`, `curva` y `gaussiano`.
- Nada sobre otras redes ni otros datos; con 13 escritores compartidos entre train y val, es
  generalizar a dígitos nuevos de los mismos escritores.

## Coste, reloj y dónde está

- **0 $**, 19 + 33 min de reloj (20–24 s por corrida, 2 hilos), dos unidades de systemd, 0 fallos.
- Métricas, resúmenes, logs, informe y figuras: aquí. **Pesos (`last.pt`), `init/` y todo lo
  demás**: en el almacén, `foveal-vision-data/experimentos-cnn-resultados/ruido-nist/`.
- Los avisos a Telegram de los dos cierres **no salieron** (`Falta BOT_TOKEN`: las unidades se
  lanzaron desde una sesión de Claude Code, no desde el bot). El `|| true` hizo su trabajo.
- Reporte #26 en `estudios-redes-neuronales`.

## Cómo se repite

```bash
cd 2026-10-02-ruido-nist
python nn/probar.py && python nn/entrenar_local.py --comprobar && sh nn/probar_lanzador.sh
nn/lanzar.sh fase1                 # 33 corridas; las que ya tienen summary.json se saltan
nn/lanzar.sh fase2 <tipos>         # los de resultados/criterio-aplicado.json → fase2
nn/lanzar.sh --estado
```
