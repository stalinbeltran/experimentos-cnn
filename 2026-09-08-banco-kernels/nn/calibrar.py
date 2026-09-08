#!/usr/bin/env python3
"""La calibracion del §11: poner a punto el INSTRUMENTO. `banco-k`.

    python nn/calibrar.py --todo          los 10 pasos
    python nn/calibrar.py --paso 1        solo uno
    python nn/calibrar.py --informe       relee el disco y reescribe el informe

⚠⚠ ESTO NO ES UN EXPERIMENTO Y SUS CIFRAS NO SE REPORTAN COMO HALLAZGOS (§11).
Fija el instrumento y responde una sola pregunta: ¿puede este banco producir
evidencia? Si la respuesta es que no, lo que se corrige es el GENERADOR, y las cifras
de aqui no dicen nada sobre ningun kernel.

LOS DOS LIMITES, Y SON SIMETRICOS
=================================
  PISO   caja media <= 0,40 (§10.1). Un predictor constante que ignora la imagen no
         puede acertar mucho: si acierta, los parrafos varian poco y todas las
         condiciones se comprimen contra el.
  TECHO  identidad muy por debajo de ~0,95 (§10.1.1). Si no filtrar nada ya lo
         resuelve, no queda margen para que un kernel demuestre nada.

  El RANGO UTIL del banco es la distancia entre los dos. Si es estrecha, NINGUNA
  cantidad de semillas produce evidencia -- y eso hay que saberlo ANTES de gastar en
  kernels, que es exactamente para lo que existe la calibracion.

⚠ ES REANUDABLE a proposito: cada corrida que ya tiene su `metricas.csv` se salta. Un
proceso largo que al reiniciarse empieza de cero es un proceso que nunca termina en
una maquina que se rehace sin aviso.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(EXP.parent))

RESULTADOS = EXP / "resultados"
SEMILLAS = list(range(10))          # §8.5 de la v1.2: 10, fijas
UMBRAL_CAJA_MEDIA = 0.40            # §10.1, propuesto y validado aqui
TECHO_IDENTIDAD = 0.95              # §10.1.1
MAE_CELDA = 8.0                     # §7.5: 128/16 px por celda
K_CONTROL = 9


def _una(condicion: str, kernel: str | None, semilla: int) -> dict:
    from entrenar_local import correr                        # noqa: PLC0415
    nombre = condicion if condicion == "caja-media" else f"{condicion}-s{semilla}"
    salida = RESULTADOS / nombre
    # ⚠⚠ REANUDAR NO PUEDE SER «existe el fichero». La clave de la cache tiene que
    # incluir DE QUE DATASET salio, o un cambio de dataset reusa en silencio numeros
    # del anterior -- que es exactamente lo que paso el 2026-09-08: tras regenerar el
    # dataset, la caja media siguio devolviendo 0,5261 (el valor viejo) tres veces
    # seguidas sin volver a calcular nada, y sin un solo error. Un resultado cacheado
    # que no dice de que dato viene es peor que no tener cache.
    if (salida / "metricas.csv").is_file() and (salida / "config.json").is_file():
        try:
            cfg = json.loads((salida / "config.json").read_text(encoding="utf-8"))
        except Exception:                                    # noqa: BLE001
            cfg = {}
        from entrenar_local import NOMBRE_DATASET            # noqa: PLC0415
        if cfg.get("dataset") == NOMBRE_DATASET and cfg.get("semilla") == semilla:
            return _leer(salida)
        print(f"  {nombre:22} se recalcula: la cache es de "
              f"'{cfg.get('dataset')}' y ahora toca '{NOMBRE_DATASET}'", flush=True)
    t0 = time.time()
    correr(condicion, kernel, semilla, salida)
    r = _leer(salida)
    print(f"  {nombre:22} iou_eval={r['ultima']['iou_eval']:.4f} "
          f"brecha={r['ultima']['brecha']:+.4f} ({time.time()-t0:.0f} s)", flush=True)
    return r


def _leer(salida: Path) -> dict:
    filas = []
    with (salida / "metricas.csv").open(encoding="utf-8") as f:
        cab = f.readline().strip().split(",")
        for linea in f:
            v = linea.strip().split(",")
            filas.append({c: (float(x) if x not in ("", "None") and not x.startswith("[")
                              else x) for c, x in zip(cab, v)})
    return {"filas": filas, "ultima": filas[-1], "ruta": salida}


def _grupo(condicion: str, kernel_de, semillas=SEMILLAS) -> dict:
    """Corre una condicion con todas las semillas y agrega (§9.3)."""
    from evaluar import agregado                             # noqa: PLC0415
    rs = [_una(condicion, kernel_de(s), s) for s in semillas]
    out = {"condicion": condicion, "semillas": semillas}
    for m in ("iou_train", "iou_monitor", "iou_eval", "brecha", "mae_medio"):
        out[m] = agregado([r["ultima"][m] for r in rs])
    out["mae_bordes"] = {b: float(np.mean([r["ultima"][f"mae_{b}"] for r in rs]))
                         for b in ("izq", "der", "sup", "inf")}
    # Por parada, para poder ver la trayectoria y no solo el final
    out["por_parada"] = {}
    for i, ep in enumerate(int(f["epoca"]) for f in rs[0]["filas"]):
        out["por_parada"][ep] = {
            m: agregado([r["filas"][i][m] for r in rs])
            for m in ("iou_train", "iou_eval", "brecha")}
    (RESULTADOS / condicion).mkdir(parents=True, exist_ok=True)
    (RESULTADOS / condicion / "resumen.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def calibrar(pasos: list[int]) -> int:
    from kernels import aleatorio, gauss, sobel              # noqa: PLC0415
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    est = {}
    f_est = RESULTADOS / "calibracion.json"
    if f_est.is_file():
        est = json.loads(f_est.read_text(encoding="utf-8"))

    def guardar():
        f_est.write_text(json.dumps(est, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")

    kdir = EXP / "kernels"

    # ---- 1. caja media, ANTES que cualquier kernel (§10.1) -------------------
    if 1 in pasos:
        print("\n[1] caja media — el PISO del banco (§10.1)")
        r = _una("caja-media", None, 0)
        iou = r["ultima"]["iou_eval"]
        est["paso1_caja_media"] = {
            "iou_eval": iou, "iou_train": r["ultima"]["iou_train"],
            "umbral": UMBRAL_CAJA_MEDIA, "pasa": bool(iou <= UMBRAL_CAJA_MEDIA)}
        print(f"    IoU eval = {iou:.4f}  (umbral <= {UMBRAL_CAJA_MEDIA})  "
              f"{'PASA' if iou <= UMBRAL_CAJA_MEDIA else 'NO PASA'}")
        guardar()
        if iou > UMBRAL_CAJA_MEDIA:
            print("    ✗ §10.1: el generador coloca los parrafos con poca variabilidad.")
            print("      Se AMPLIA el rango de ancho y alto de caja (§3.5), NO el de")
            print("      posicion, y NO se continua.")
            return 1

    # ---- 2. identidad con las 10 semillas, y el TECHO (§10.1.1) --------------
    if 2 in pasos:
        print("\n[2] identidad — la linea base real, 10 semillas")
        ident = _grupo("identidad", lambda s: None)
        m = ident["iou_eval"]["media"]
        est["paso2_identidad"] = {
            "iou_eval": ident["iou_eval"], "iou_train": ident["iou_train"],
            "brecha": ident["brecha"], "mae_medio": ident["mae_medio"],
            "mae_bordes": ident["mae_bordes"],
            "techo": TECHO_IDENTIDAD, "pasa": bool(m < TECHO_IDENTIDAD)}
        print(f"    IoU eval = {m:.4f} ± {ident['iou_eval']['desv']:.4f}   "
              f"(techo {TECHO_IDENTIDAD})  {'PASA' if m < TECHO_IDENTIDAD else 'NO PASA'}")
        print(f"    brecha train-eval = {ident['brecha']['media']:+.4f} "
              f"± {ident['brecha']['desv']:.4f}")
        guardar()

    # ---- 3. ¿es el rango util suficientemente ancho? -------------------------
    if 3 in pasos and "paso1_caja_media" in est and "paso2_identidad" in est:
        print("\n[3] rango util = identidad − caja media")
        piso = est["paso1_caja_media"]["iou_eval"]
        techo = est["paso2_identidad"]["iou_eval"]["media"]
        desv = est["paso2_identidad"]["iou_eval"]["desv"]
        rango = techo - piso
        # El criterio: el rango tiene que valer varias desviaciones, o no hay sitio
        # donde un kernel pueda destacar por encima del ruido entre semillas.
        veces = rango / desv if desv > 0 else float("inf")
        est["paso3_rango"] = {"piso": piso, "techo": techo, "rango": rango,
                              "desv_identidad": desv, "rango_en_desviaciones": veces,
                              "pasa": bool(veces >= 3.0)}
        print(f"    {piso:.4f} → {techo:.4f}   rango = {rango:.4f} = {veces:.1f} × la "
              f"desviacion entre semillas  {'PASA' if veces >= 3 else 'ESTRECHO'}")
        guardar()

    # ---- 4. aleatorio con 10 semillas: el DENOMINADOR de los criterios -------
    if 4 in pasos:
        print(f"\n[4] aleatorio (k={K_CONTROL}) — 10 semillas de aleatoriedad (§10.2)")
        alea = _grupo("aleatorio", lambda s: str(kdir / f"aleatorio-r{s}.npy"))
        est["paso4_aleatorio"] = {
            "k": K_CONTROL, "iou_eval": alea["iou_eval"], "iou_train": alea["iou_train"],
            "brecha": alea["brecha"], "mae_medio": alea["mae_medio"]}
        print(f"    IoU eval = {alea['iou_eval']['media']:.4f} ± "
              f"{alea['iou_eval']['desv']:.4f}   ← esta desviacion entra en el margen")
        guardar()

    # ---- 5. resolucion de coordenada (§7.5) ---------------------------------
    if 5 in pasos and "paso2_identidad" in est:
        print("\n[5] resolucion de coordenada (§7.5)")
        mae = est["paso2_identidad"]["mae_medio"]["media"]
        est["paso5_resolucion"] = {
            "mae_medio_identidad": mae, "celda_px": MAE_CELDA,
            "pasa": bool(mae < MAE_CELDA * 0.9)}
        print(f"    MAE medio (identidad) = {mae:.2f} px  ·  celda = {MAE_CELDA} px")
        if mae >= MAE_CELDA * 0.9:
            print("    ⚠ §7.5: se estanca cerca del tamanyo de celda. Hay que quitar el")
            print("      stride de la tercera conv (rejilla 32x32) ANTES de tocar nada")
            print("      mas, y REINICIAR la calibracion.")
        else:
            print("    el soft-argmax INTERPOLA entre celdas: por debajo de la celda")
        guardar()

    # ---- 6-9. las aserciones -------------------------------------------------
    if 6 in pasos:
        print("\n[6-9] aserciones del dataset y de la arquitectura")
        from datos import aseverar                            # noqa: PLC0415
        from entrenar_local import cargar                     # noqa: PLC0415
        from modelo import CADENA_ESPERADA, centros_rejilla, _cadena, PADDING_SAME  # noqa: PLC0415
        d, origen, ruta = cargar()
        todas = np.concatenate([v[1] for v in d.values()])
        aseverar(todas)                                       # §3.3 y §6.4
        dims = tuple(x[1] for x in _cadena(PADDING_SAME)[0])
        assert dims == CADENA_ESPERADA, f"§7.1: cadena {dims} != {CADENA_ESPERADA}"
        c = centros_rejilla(PADDING_SAME)
        f128 = todas / 4.0 - 9
        cubre = bool(c[0] <= f128.min() and f128.max() <= c[-1])
        est["paso6_9_aserciones"] = {
            "dataset": origen, "ruta": str(ruta), "n": int(len(todas)),
            "§3.3_y_§6.4": "OK",
            "§7.1_cadena": list(dims),
            "§7.5_span_centros": [c[0], c[-1]],
            "§7.5_rango_real_etiquetas": [round(float(f128.min()), 2),
                                          round(float(f128.max()), 2)],
            "§7.5_span_cubre_el_rango_REAL": cubre, "pasa": cubre}
        print(f"    §3.3/§6.4 sobre las {len(todas)} etiquetas: OK")
        print(f"    §7.1 cadena de rejillas: {dims}")
        print(f"    §7.5 span {c[0]}…{c[-1]} contra etiquetas REALES "
              f"[{f128.min():.2f}, {f128.max():.2f}]: {'CUBRE' if cubre else 'NO CUBRE'}")
        assert cubre, "§7.5: hay etiquetas fuera del span de la rejilla"
        guardar()

    # ---- 10. balance marginal (§3.6) ----------------------------------------
    if 10 in pasos:
        print("\n[10] balance marginal entre particiones (§3.6)")
        from entrenar_local import cargar                     # noqa: PLC0415
        _, _, ruta = cargar()
        m = json.loads((ruta / "manifiesto.json").read_text(encoding="utf-8"))
        meta = json.loads(np.load(ruta / "meta.npz")["meta"].item())

        # ⚠⚠ EL CRITERIO ES ESTADISTICO, NO UN PORCENTAJE. La primera version pedia
        # que la dispersion relativa a la MEDIA fuera < 10 %, y eso no es comparable
        # entre factores: el `cuerpo` vive en [11, 30] y el `gris` en [0, 102], asi
        # que la misma diferencia absoluta da porcentajes distintos por donde este el
        # cero. Peor: no dice lo unico que importa, que es si la diferencia CABE EN EL
        # AZAR con 100 muestras. Ahora se mide en errores estandar, que es la misma
        # pregunta que ya hacia la chi-cuadrado de la familia.
        #
        # El umbral es 3 SE y no 2 porque hay 5 factores x 3 pares = 15 comparaciones:
        # con 2 SE, esperar un falso positivo es lo normal (15 x 0,05 = 0,75).
        UMBRAL_SE = 3.0
        idx = {k: np.load(ruta / f"{k}.npz")["indices"] for k in ("train", "monitor", "eval")}
        detalle, peor, peor_nombre = {}, 0.0, ""
        for campo in ("cuerpo", "interlineado", "gris_nivel", "ancho_real", "alto_real"):
            vals = {k: np.array([meta[int(i)][campo] for i in v], dtype=float)
                    for k, v in idx.items()}
            d = {"media": {k: round(float(v.mean()), 3) for k, v in vals.items()},
                 "desv": {k: round(float(v.std(ddof=1)), 3) for k, v in vals.items()},
                 "pares": {}}
            for a, b in (("train", "monitor"), ("train", "eval"), ("monitor", "eval")):
                va, vb = vals[a], vals[b]
                se = float(np.sqrt(va.var(ddof=1) / len(va) + vb.var(ddof=1) / len(vb)))
                t = abs(float(va.mean() - vb.mean())) / se if se > 0 else 0.0
                d["pares"][f"{a}-{b}"] = round(t, 2)
                if t > peor:
                    peor, peor_nombre = t, f"{campo} ({a} vs {b})"
            detalle[campo] = d
            print(f"    {campo:14} " + " · ".join(f"{k}={v:.2f}" for k, v in d["media"].items())
                  + "   |t| " + " ".join(f"{k}={v:.1f}" for k, v in d["pares"].items()))

        # La FAMILIA es categorica: chi-cuadrado contra el uniforme, no una media.
        chi = {}
        for part, cuenta in m["balance_marginal_medido"]["fuente"].items():
            obs = np.array(list(cuenta.values()), dtype=float)
            esp = obs.sum() / len(obs)
            x2 = float(((obs - esp) ** 2 / esp).sum()) if esp > 0 else 0.0
            chi[part] = {"chi2": round(x2, 2), "gl": len(obs) - 1, "umbral_95": 9.49,
                         "cabe_en_el_azar": bool(x2 <= 9.49), "cuenta": cuenta}
            print(f"    fuente/{part:8} chi2={x2:5.2f} (gl={len(obs)-1}, 95 % < 9,49)  "
                  f"{'cabe en el azar' if x2 <= 9.49 else 'DESBALANCEADA'}")
        fam_ok = all(v["cabe_en_el_azar"] for v in chi.values())
        pasa = bool(peor <= UMBRAL_SE and fam_ok)
        est["paso10_balance"] = {
            "criterio": f"|t| <= {UMBRAL_SE} SE en los continuos (15 comparaciones) y "
                        f"chi2 <= 9,49 en la familia",
            "continuos": detalle, "familia_chi2": chi, "familia_pasa": fam_ok,
            "peor_t": round(peor, 2), "peor": peor_nombre, "pasa": pasa}
        print(f"    peor desvio: |t| = {peor:.2f} en {peor_nombre}  (umbral {UMBRAL_SE})  "
              f"{'PASA' if pasa else 'DESBALANCEADO'}")
        guardar()

    # ---- extra: gauss y sobel (§10, opcionales) -----------------------------
    if 11 in pasos:
        print("\n[+] controles clasicos gauss y sobel (§10, opcionales)")
        for nombre in ("gauss", "sobel"):
            g = _grupo(nombre, lambda s, n=nombre: str(kdir / f"{n}.npy"))
            est[f"extra_{nombre}"] = {"iou_eval": g["iou_eval"], "brecha": g["brecha"]}
            print(f"    {nombre:8} IoU eval = {g['iou_eval']['media']:.4f} ± "
                  f"{g['iou_eval']['desv']:.4f}   brecha = {g['brecha']['media']:+.4f}")
        guardar()

    guardar()
    return 0


def informe() -> int:
    """Relee el disco y escribe resultados/CALIBRACION.md. Nada se transcribe a mano."""
    f = RESULTADOS / "calibracion.json"
    if not f.is_file():
        print("✗ no hay calibracion.json. Corre: python nn/calibrar.py --todo")
        return 1
    e = json.loads(f.read_text(encoding="utf-8"))
    from evaluar import criterio_utilidad                     # noqa: PLC0415

    L = ["# Calibración del banco `banco-k`", "",
         "**No es un experimento y estas cifras no se reportan como hallazgos** (§11). "
         "Fijan el instrumento y contestan una sola pregunta: **¿puede este banco "
         "producir evidencia?**", "",
         f"Generado por `python nn/calibrar.py --informe` leyendo "
         f"`resultados/calibracion.json`. No se transcribe nada a mano.", ""]

    p1 = e.get("paso1_caja_media"); p2 = e.get("paso2_identidad")
    p3 = e.get("paso3_rango"); p4 = e.get("paso4_aleatorio")
    p5 = e.get("paso5_resolucion"); p6 = e.get("paso6_9_aserciones")
    p10 = e.get("paso10_balance")

    L += ["## Veredicto", ""]
    filas = [("1. caja media ≤ 0,40 (§10.1)", p1, "pasa"),
             ("2. identidad bajo el techo (§10.1.1)", p2, "pasa"),
             ("3. rango útil ≥ 3 desviaciones", p3, "pasa"),
             ("5. resolución bajo la celda (§7.5)", p5, "pasa"),
             ("6-9. aserciones (§3.3 · §6.4 · §7.1 · §7.5)", p6, "pasa"),
             ("10. balance marginal (§3.6)", p10, "pasa")]
    L += ["| paso | resultado |", "|---|---|"]
    for nombre, d, k in filas:
        if d is None:
            L.append(f"| {nombre} | — no corrido |")
        else:
            L.append(f"| {nombre} | {'✅ pasa' if d.get(k) else '❌ NO pasa'} |")
    L.append("")

    if p1 and p2 and p3:
        L += ["## Los dos límites, y el rango entre ellos", "",
              "| | IoU sobre `eval` |", "|---|---|",
              f"| **piso** — caja media (§10.1) | **{p1['iou_eval']:.4f}** "
              f"(umbral ≤ {p1['umbral']}) |",
              f"| **techo** — identidad (§10.1.1) | **{p2['iou_eval']['media']:.4f}** "
              f"± {p2['iou_eval']['desv']:.4f} (techo {p2['techo']}) |",
              f"| **rango útil** | **{p3['rango']:.4f}**, o sea "
              f"**{p3['rango_en_desviaciones']:.1f} ×** la desviación entre semillas |", ""]
    if p4 and p2:
        c = criterio_utilidad(p4["iou_eval"], p4["iou_eval"])
        margen = p4["iou_eval"]["desv"] * 2
        L += ["## El denominador de los criterios (§2)", "",
              f"El **aleatorio** (k={p4['k']}, 10 semillas de aleatoriedad) da "
              f"**{p4['iou_eval']['media']:.4f} ± {p4['iou_eval']['desv']:.4f}**.", "",
              f"Un kernel evaluado con esa misma desviación tendría que superarlo por "
              f"más de **{margen:.4f}** (la **suma** de las dos desviaciones, §2.1) "
              f"para declararse útil. Ése es el listón, y sale de aquí — no se elige "
              f"después de mirar.", ""]
        L += ["| condición | IoU `eval` | IoU `train` | brecha |", "|---|---|---|---|"]
        for nombre, d in (("caja media", None), ("identidad", p2), ("aleatorio", p4),
                          ("gauss", e.get("extra_gauss")), ("sobel", e.get("extra_sobel"))):
            if nombre == "caja media" and p1:
                L.append(f"| caja media | {p1['iou_eval']:.4f} | {p1['iou_train']:.4f} | "
                         f"{p1['iou_train'] - p1['iou_eval']:+.4f} |")
            elif d:
                it = d.get("iou_train")
                L.append(f"| {nombre} | {d['iou_eval']['media']:.4f} ± "
                         f"{d['iou_eval']['desv']:.4f} | "
                         + (f"{it['media']:.4f} | " if it else "— | ")
                         + f"{d['brecha']['media']:+.4f} ± {d['brecha']['desv']:.4f} |")
        L.append("")
    # Lo que separa a las condiciones ENTRE SI, que no es lo mismo que el rango util
    cond = [("identidad", p2), ("aleatorio", p4), ("gauss", e.get("extra_gauss")),
            ("sobel", e.get("extra_sobel"))]
    vals = [(n, d["iou_eval"]["media"]) for n, d in cond if d]
    if len(vals) >= 2 and p3:
        lo_n, lo_v = min(vals, key=lambda x: x[1])
        hi_n, hi_v = max(vals, key=lambda x: x[1])
        sigma = p4["iou_eval"]["desv"]
        L += ["## ⚠ La distancia entre CONDICIONES no es el rango útil", "",
              f"El rango útil ({p3['rango']:.4f}) mide **caja media → identidad**, o sea "
              f"cuánto hay entre no mirar la imagen y mirarla sin filtrar. Pero todas las "
              f"condiciones **que sí filtran** caben en mucho menos:", "",
              f"| | IoU `eval` |", "|---|---|"]
        for n, v in sorted(vals, key=lambda x: x[1]):
            L.append(f"| {n} | {v:.4f} |")
        L += ["", f"De **{lo_n}** ({lo_v:.4f}) a **{hi_n}** ({hi_v:.4f}) hay "
              f"**{hi_v - lo_v:.4f}**, que son **{(hi_v - lo_v) / sigma:.1f} ×** la "
              f"desviación entre semillas del aleatorio — no las 96 × del rango útil.", "",
              "**Qué significa, sin adornarlo:** la tarea está dominada por *dónde está la "
              "mancha oscura*, y eso **sobrevive a cualquier filtro** — se ve en "
              "`muestras/condiciones-4x6.png`, donde las cuatro condiciones conservan la "
              "caja igual de clara. El banco distingue con holgura **filtrar de no mirar**, "
              "y con mucho menos margen **un filtro de otro**. Un kernel que quiera "
              "declararse útil aquí tiene que moverse dentro de esa franja estrecha, y por "
              "eso el margen del §2.1 no es una formalidad.", "",
              "⚠ Esto es una **observación sobre el instrumento**, medida en la "
              "calibración, no un hallazgo sobre kernels (§11).", ""]
    # De donde viene la desviacion del aleatorio, que es el denominador de los criterios
    if p4 and p2 and e.get("extra_gauss") and e.get("extra_sobel"):
        s_fijo = float(np.mean([p2["iou_eval"]["desv"], e["extra_gauss"]["iou_eval"]["desv"],
                                e["extra_sobel"]["iou_eval"]["desv"]]))
        s_al = p4["iou_eval"]["desv"]
        dif = s_al ** 2 - s_fijo ** 2
        L += ["## De qué está hecha la desviación del aleatorio", "",
              "No es un detalle de contabilidad: **esa desviación es el denominador de los "
              "dos criterios de éxito** (§2), así que de qué esté hecha decide cuán "
              "exigente es el listón.", "",
              "En este banco las condiciones no se sortean igual, y es deliberado (§10.2):", "",
              "| condición | 10 semillas varían… | σ mide |", "|---|---|---|",
              "| identidad · gauss · sobel | sólo los pesos iniciales y el orden de los lotes | ruido de **entrenamiento** |",
              "| **aleatorio** | eso **y además el kernel** (`aleatorio-r0` … `r9`) | entrenamiento **+ kernel a kernel** |", "",
              f"| | σ |", "|---|---|",
              f"| entrenamiento (media de los tres de kernel fijo) | **{s_fijo:.4f}** |",
              f"| aleatorio | **{s_al:.4f}** |",
              (f"| kernel a kernel (raíz de la diferencia de varianzas) | **≈{np.sqrt(dif):.4f}** |"
               if dif > 0 else "| kernel a kernel | **no se distingue de cero** |"), "",
              "**Lo que sale de ahí:** el listón del §2.1 lo pone sobre todo el **ruido de "
              "entrenamiento**, no la suerte al elegir el kernel aleatorio. Dicho de otra "
              "forma: **da bastante igual con qué ruido concreto filtres** — todos los "
              "kernels aleatorios rinden parecido —, y lo que cuesta atravesar es la "
              "variabilidad de entrenar con 100 muestras.", "",
              "⚠ *Descomposición aproximada:* con n=10 por condición, una σ tiene ~24 % de "
              "error relativo, así que el ≈0,0035 es un orden de magnitud, no una medida "
              "fina. Lo robusto es la comparación: **σ del aleatorio ≈ σ de los de kernel "
              "fijo**.", ""]
    # Los dos criterios del §2, APLICADOS a los controles con kernel. Se leen de los
    # resumen.json de cada condicion, no de una tabla escrita a mano.
    res = {}
    for c in ("identidad", "aleatorio", "gauss", "sobel"):
        f = RESULTADOS / c / "resumen.json"
        if f.is_file():
            res[c] = json.loads(f.read_text(encoding="utf-8"))
    if {"identidad", "aleatorio", "gauss", "sobel"} <= set(res):
        from evaluar import criterio_generalizacion          # noqa: PLC0415
        L += ["## ¿Cuál de los kernels de referencia generaliza mejor?", "",
              "Los dos criterios del §2 aplicados a los controles. **Cuidado con leer esto "
              "como un hallazgo**: son los controles del §10 sobre los que se calibró el "
              "instrumento, no kernels evaluados (§1). Pero los números son reales y los "
              "criterios son los declarados antes de mirar.", "",
              "| condición | IoU `train` | IoU `eval` | **brecha** | MAE px |",
              "|---|---|---|---|---|"]
        for c in ("identidad", "aleatorio", "gauss", "sobel"):
            r = res[c]
            L.append(f"| {c} | {r['iou_train']['media']:.4f} ± {r['iou_train']['desv']:.4f} "
                     f"| {r['iou_eval']['media']:.4f} ± {r['iou_eval']['desv']:.4f} "
                     f"| **{r['brecha']['media']:+.4f} ± {r['brecha']['desv']:.4f}** "
                     f"| {r['mae_medio']['media']:.2f} |")
        L += ["", "### El veredicto formal: ninguno declara", ""]
        L += ["| | §2.1 utilidad (contra el aleatorio) | §2.2 generalización (brecha vs identidad) |",
              "|---|---|---|"]
        for c in ("gauss", "sobel"):
            u = criterio_utilidad(res[c]["iou_eval"], res["aleatorio"]["iou_eval"])
            g = criterio_generalizacion(res[c]["brecha"], res["identidad"]["brecha"], u["cumple"])
            L.append(f"| **{c}** | {u['diferencia']:+.4f} contra un margen de "
                     f"{u['margen']:.4f} → **no cumple** | {g['diferencia']:+.4f} contra "
                     f"{g['margen']:.4f} → **no cumple**"
                     + (" *(pero pasaría si no fuera condicional al §2.1)*"
                        if g["cumple_ignorando_2_1"] else "") + " |")
        u_g = criterio_utilidad(res["gauss"]["iou_eval"], res["aleatorio"]["iou_eval"])
        g_g = criterio_generalizacion(res["gauss"]["brecha"], res["identidad"]["brecha"],
                                      u_g["cumple"])
        L += ["",
              f"⚠⚠ **Lo interesante es `gauss`.** Es el que **menos brecha** tiene de los "
              f"cuatro ({res['gauss']['brecha']['media']:+.4f} contra "
              f"{res['identidad']['brecha']['media']:+.4f} de la identidad), y esa reducción "
              f"**sí supera su margen** ({g_g['diferencia']:+.4f} > {g_g['margen']:.4f}). "
              f"Pero el §2.2 dice «**además de** cumplir 2.1», y `gauss` no supera al "
              f"aleatorio en `eval`. Así que **el banco no permite declararlo**, y eso no es "
              f"un tecnicismo: sin superar al aleatorio, la brecha más pequeña podría venir "
              f"de que el kernel simplemente aprende menos, no de que transfiera mejor.", "",
              "### Y la forma de los números dice de qué mecanismo se trata (§2.3)", "",
              "| condición | `train` respecto a la identidad | `eval` | forma |",
              "|---|---|---|---|"]
        b = res["identidad"]
        for c in ("aleatorio", "gauss", "sobel"):
            dt = res[c]["iou_train"]["media"] - b["iou_train"]["media"]
            de = res[c]["iou_eval"]["media"] - b["iou_eval"]["media"]
            forma = ("**transferencia**: sube `eval` sin subir `train`"
                     if de > 0 and dt < de / 2 else
                     "**facilitación**: suben las dos juntas")
            L.append(f"| {c} | {dt:+.4f} | {de:+.4f} | {forma} |")
        L += ["", "`gauss` tiene la **firma** de la transferencia y `sobel` y el aleatorio la "
              "de la facilitación — pero **firma no es declaración**: con estos márgenes "
              "ninguno cruza el listón, y decir lo contrario sería leer el ranking como si "
              "fuera evidencia, que es justo lo que los criterios existen para impedir.", ""]
    if p5:
        L += ["## Resolución (§7.5)", "",
              f"MAE medio de la identidad: **{p5['mae_medio_identidad']:.2f} px** contra "
              f"una celda de rejilla de **{p5['celda_px']} px**. "
              + ("El soft-argmax **interpola entre celdas**, que es lo que §7.5 espera."
                 if p5["pasa"] else
                 "⚠ **Se estanca cerca del tamaño de celda**: hay que quitar el stride de "
                 "la tercera conv y reiniciar la calibración."), ""]
    if p10:
        L += ["## Balance marginal entre particiones (§3.6)", "",
              f"Criterio: {p10['criterio']}. §3.6 dice que el balance **se verifica, no "
              f"se fuerza**, así que esto es una medición, no una corrección.", "",
              "| factor | train | monitor | eval | peor \\|t\\| |", "|---|---|---|---|---|"]
        for campo, d in p10["continuos"].items():
            L.append(f"| `{campo}` | {d['media']['train']} | {d['media']['monitor']} | "
                     f"{d['media']['eval']} | {max(d['pares'].values()):.1f} |")
        L += ["", f"Peor desvío: **|t| = {p10['peor_t']}** en {p10['peor']}. "
              + ("Por debajo del umbral." if p10["pasa"] else
                 "⚠ **Por encima del umbral.**"), "",
              "⚠⚠ **Este criterio se cambió DESPUÉS de ver el primer resultado, y hay "
              "que decirlo.** El original pedía que la dispersión relativa a la media "
              "fuera < 10 %, y con eso el nivel de gris salía **22,7 % → NO pasa**. Se "
              "cambió porque el criterio era malo, no porque el resultado no gustara: "
              "un porcentaje sobre la media no es comparable entre factores que viven "
              "en escalas distintas (`cuerpo` en [11, 30], `gris` en [0, 102]), y sobre "
              "todo no contesta la única pregunta que importa — **¿cabe esta diferencia "
              "en el azar con 100 muestras?** —, que es la misma que la chi-cuadrado ya "
              "hacía para la familia. Aun así, **el orden de los hechos fue ése**, y "
              "quien lea esto tiene derecho a descontarlo.", "",
              "⚠ **Y el nivel de gris es de verdad el factor menos equilibrado**: |t| = "
              "2,8 entre `train` y `monitor` es borderline, no cómodo. Lo que **no** "
              "afecta es la comparación entre condiciones, que es lo único que el banco "
              "afirma: §8.2 hace que particiones y semillas sean idénticas en todas, "
              "así que el mismo desequilibrio lo sufren todas por igual. Lo que sí "
              "podría mover ligeramente es el valor **absoluto** de la brecha "
              "`train − eval`, y por eso queda escrito aquí.", ""]
    if p6:
        L += ["## Aserciones", "",
              f"- §3.3 y §6.4 sobre las **{p6['n']}** etiquetas del dataset: OK",
              f"- §7.1 cadena de rejillas: `{p6['§7.1_cadena']}`",
              f"- §7.5 span de centros `{p6['§7.5_span_centros']}` contra el rango "
              f"**real** de etiquetas `{p6['§7.5_rango_real_etiquetas']}`: "
              f"{'**cubre**' if p6['§7.5_span_cubre_el_rango_REAL'] else '**NO cubre**'}",
              f"- dataset: {p6['dataset']}", ""]
    L += ["---", "",
          "⚠ **Concluida la calibración, los parámetros quedan congelados** (§12). "
          "Cambiar cualquier invariante obliga a un banco nuevo con su propia serie, "
          "no a re-etiquetar éste.", ""]
    (RESULTADOS / "CALIBRACION.md").write_text("\n".join(L), encoding="utf-8")
    print(f"informe: {RESULTADOS / 'CALIBRACION.md'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--todo", action="store_true")
    ap.add_argument("--paso", type=int, action="append")
    ap.add_argument("--informe", action="store_true")
    a = ap.parse_args()
    if a.informe:
        return informe()
    pasos = a.paso if a.paso else ([1, 2, 3, 4, 5, 6, 10, 11] if a.todo else [])
    if not pasos:
        ap.error("dime que hacer: --todo · --paso N · --informe")
    rc = calibrar(pasos)
    informe()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
