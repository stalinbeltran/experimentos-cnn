# Calibración del banco `banco-k`

**No es un experimento y estas cifras no se reportan como hallazgos** (§11). Fijan el instrumento y contestan una sola pregunta: **¿puede este banco producir evidencia?**

Generado por `python nn/calibrar.py --informe` leyendo `resultados/calibracion.json`. No se transcribe nada a mano.

## Veredicto

| paso | resultado |
|---|---|
| 1. caja media ≤ 0,40 (§10.1) | ❌ NO pasa |
| 2. identidad bajo el techo (§10.1.1) | — no corrido |
| 3. rango útil ≥ 3 desviaciones | — no corrido |
| 5. resolución bajo la celda (§7.5) | — no corrido |
| 6-9. aserciones (§3.3 · §6.4 · §7.1 · §7.5) | — no corrido |
| 10. balance marginal (§3.6) | ✅ pasa |

---

⚠ **Concluida la calibración, los parámetros quedan congelados** (§12). Cambiar cualquier invariante obliga a un banco nuevo con su propia serie, no a re-etiquetar éste.
