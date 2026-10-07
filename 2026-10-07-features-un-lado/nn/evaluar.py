#!/usr/bin/env python3
"""La EVALUACIÓN de los DETECTORES de `feat-1lado` (instrucciones/02-criterio.md), escrita ANTES de entrenar. Tres bancos
de 13: `lineas` (feat-ind32, la tinta: la referencia, por su id y su huella), `control` y `compartido`. El compositor y la
curva de desplazamiento son otro paso (nn/componer.py); el criterio de aquí los lee de resultados/componer.json.

  §A  val sintético (2–4 px, el de entrenar): F1, P, R, pos ≤ 1 y veredicto, de cada summary.json
  §B  la prueba de GROSOR (`feat-bor-sinteticas-grueso-32px-r20261006`), con la MISMA medida que feat-bor (copiada): F1,
      recall y FP por tramo (2–4 px vistos, 6–12 px no vistos) y R = media de F1(no vistos)/F1(vistos)
  §C  los dígitos (windep): con qué frecuencia se enciende cada detector en cada clase — el síntoma de feat-bor: los
      arcos en los 1 —, y en `compartido`, QUÉ CANAL gana el máximo cuando el detector se enciende (si casi siempre es
      el mismo, el brazo degeneró en un detector de un solo lado)

    python nn/evaluar.py                 → resultados/evaluacion.json (y el criterio, si ya existe componer.json)
    python nn/evaluar.py --solo-lineas   → resultados/referencia-lineas.json: la referencia, ANTES de entrenar nada
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
sys.path.insert(0, str(EXP.parent))
sys.path.insert(0, str(AQUI))
from expcnn import por_id                       # noqa: E402
import datos                                    # noqa: E402
import features as F                            # noqa: E402
import modelo                                   # noqa: E402

RES = EXP / "resultados"
VISTOS, NO_VISTOS = (2, 3, 4), (6, 8, 10, 12)


def r4(v) -> float:
    return round(float(v), 4)


def carpeta(rep: str) -> Path:
    return (por_id("feat-ind32").carpeta / "nn" / "pesos") if rep == "lineas" else AQUI / f"pesos-{rep}"


def banco(rep: str) -> tuple[list, np.ndarray, dict]:
    reds, um, origen = [], np.zeros(len(F.CON_TRAZO), np.float32), {}
    if rep == "lineas":
        firma = json.loads((por_id("feat-ind32").carpeta / "resultados" / "firma-por-clase.json").read_text(encoding="utf-8"))["huellas_best"]
    for j, f in enumerate(F.CON_TRAZO):
        ruta = carpeta(rep) / f / "best.pt"
        if not ruta.is_file():
            raise SystemExit(f"✗ falta {ruta}: el brazo '{rep}' no está entrenado (nn/vast.sh {rep})")
        red, est = modelo.cargar_tinta(ruta) if rep == "lineas" else modelo.cargar(ruta)
        h = modelo.huella_pesos(red)
        if rep == "lineas" and h != firma[f]:
            raise SystemExit(f"✗ feat-ind32/{f}: huella {h} ≠ {firma[f]} de su firma: sus pesos cambiaron. Me niego.")
        reds.append(red); um[j] = est["umbral"]; origen[f] = h
    return reds, um, origen


def seccion_a(reps) -> dict:
    out = {}
    for rep in reps:
        filas = {}
        for f in F.CON_TRAZO:
            s = json.loads((carpeta(rep) / f / "summary.json").read_text(encoding="utf-8"))
            b = s["best"]
            filas[f] = {k: b[k] for k in ("f1", "precision", "recall", "pos_ok", "umbral")} | {"veredicto": s["veredicto"],
                                                                                              "epoca": b["epoca"]}
        out[rep] = {"detectores": filas, "f1_medio": r4(np.mean([v["f1"] for v in filas.values()])),
                    "al_menos_a_medias": sum(v["veredicto"] in ("aprendió", "a medias") for v in filas.values()),
                    "aprendio": sum(v["veredicto"] == "aprendió" for v in filas.values())}
        print(f"  §A {rep:<10} F1 medio {out[rep]['f1_medio']:.3f} · a medias o más {out[rep]['al_menos_a_medias']}/13", flush=True)
    return out


@torch.no_grad()
def seccion_b(bancos: dict) -> dict:
    """COPIA de §B de feat-bor (nn/evaluar.py), sin la representación: la tinta entra tal cual."""
    g = datos.cargar(datos.GRUESO)
    out = {}
    for rep, (reds, um, _) in bancos.items():
        por_f = {}
        for j, f in enumerate(F.CON_TRAZO):
            c = datos.conjunto(g, f, None)
            sp = torch.sigmoid(modelo.logits_por_lotes(reds[j], c["x_pos"])).flatten(1)
            sn = torch.sigmoid(modelo.logits_por_lotes(reds[j], c["x_neg"])).flatten(1)
            hp = (sp.max(1).values >= um[j]).numpy(); hn = (sn.max(1).values >= um[j]).numpy()
            idx = sp.argmax(1).numpy(); pos = np.stack([idx // 8, idx % 8], 1)
            ok_pos = np.abs(pos - c["ancla_pos"]).max(1) <= 1
            fila = {}
            for w in sorted(set(c["grosor_pos"].tolist())):
                m = c["grosor_pos"] == w
                fila[str(w)] = {"n": int(m.sum()), "recall": r4(hp[m].mean()),
                                "pos_ok": r4(ok_pos[m & hp].mean()) if (m & hp).any() else None}
            tramos = {}
            for nombre, gs in (("vistos", VISTOS), ("no_vistos", NO_VISTOS)):
                mp, mn = np.isin(c["grosor_pos"], gs), np.isin(c["grosor_neg"], gs)
                tp, fn_, fp = int(hp[mp].sum()), int((~hp[mp]).sum()), int(hn[mn].sum())
                pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn_)
                tramos[nombre] = {"n_pos": int(mp.sum()), "n_neg": int(mn.sum()), "recall": r4(rc), "precision": r4(pr),
                                  "fp_tasa": r4(hn[mn].mean()) if mn.any() else None,
                                  "f1": r4(2 * pr * rc / max(1e-9, pr + rc))}
            valido = tramos["vistos"]["n_pos"] >= 20 and tramos["no_vistos"]["n_pos"] >= 20 and tramos["vistos"]["f1"] > 0
            por_f[f] = {"por_grosor": fila, **tramos,
                        "r": r4(tramos["no_vistos"]["f1"] / tramos["vistos"]["f1"]) if valido else None}
        val = {f: v for f, v in por_f.items() if v["r"] is not None}

        def media(tramo, k):
            return r4(np.mean([v[tramo][k] for v in val.values()]))
        out[rep] = {"detectores": por_f, "R": r4(np.mean([v["r"] for v in val.values()])), "detectores_con_R": list(val),
                    **{f"{k}_{tramo}": media(tramo, k) for tramo in ("vistos", "no_vistos") for k in ("f1", "recall", "fp_tasa")}}
        o = out[rep]
        print(f"  §B {rep:<10} R {o['R']:.3f} · F1 2–4 px {o['f1_vistos']:.3f} → 6–12 px {o['f1_no_vistos']:.3f}"
              f" · recall {o['recall_vistos']:.3f} → {o['recall_no_vistos']:.3f} · FP {o['fp_tasa_vistos']:.3f} → "
              f"{o['fp_tasa_no_vistos']:.3f}", flush=True)
    return out


@torch.no_grad()
def seccion_c(bancos: dict) -> dict:
    dg = datos.digitos(); y = dg["y"]; w = dg["origen"] == "windep"
    x = dg["x"][w]; yw = y[w]
    out = {"firma": {}, "canal_ganador": {}}
    for rep, (reds, um, _) in bancos.items():
        conf = np.stack([torch.sigmoid(modelo.logits_por_lotes(r, x)).flatten(1).max(1).values.numpy() for r in reds], 1)
        pres = conf >= um[None]
        out["firma"][rep] = {str(c): {f: r4(pres[yw == c, j].mean()) for j, f in enumerate(F.CON_TRAZO)} for c in range(10)}
        if rep == "compartido":
            for j, f in enumerate(F.CON_TRAZO):
                gan = np.concatenate([reds[j].por_canal(torch.from_numpy(x[i:i + 512]).float()).flatten(2).max(2).values
                                      .argmax(1).numpy() for i in range(0, len(x), 512)])
                h = pres[:, j]
                cuenta = np.bincount(gan[h], minlength=8) if h.any() else np.zeros(8, int)
                out["canal_ganador"][f] = {"n_encendido": int(h.sum()),
                                           "reparto": {modelo.FLECHAS[k]: r4(cuenta[k] / max(1, h.sum())) for k in range(8)},
                                           "maximo": r4(cuenta.max() / max(1, h.sum()))}
        f1 = out["firma"][rep]["1"]
        print(f"  §C {rep:<10} en los 1: arco-E {f1['arco-E']:.2f} · arco-W {f1['arco-W']:.2f} · lazo {f1['lazo']:.2f}", flush=True)
    return out


def criterio(a: dict, b: dict, c: dict, comp: dict | None) -> dict:
    """instrucciones/02-criterio.md, en código. Mismas cifras en los dos sitios."""
    cri = {}
    l_a, l_b = a["lineas"], b["lineas"]
    for rep in ("control", "compartido"):
        ok = a[rep]["al_menos_a_medias"] >= 11 and a[rep]["f1_medio"] >= l_a["f1_medio"] - 0.05
        cri[f"H1 {rep}"] = {"veredicto": "confirmada" if ok else "refutada", "al_menos_a_medias": a[rep]["al_menos_a_medias"],
                            "f1_medio": a[rep]["f1_medio"], "f1_medio_lineas": l_a["f1_medio"]}
        o = b[rep]
        sube_fp, sube_fp_l = o["fp_tasa_no_vistos"] - o["fp_tasa_vistos"], l_b["fp_tasa_no_vistos"] - l_b["fp_tasa_vistos"]
        baja_rc = o["recall_vistos"] - o["recall_no_vistos"]
        partes = {"fp_sube_como_mucho_la_mitad_que_lineas": sube_fp <= sube_fp_l / 2,
                  "recall_no_baja_mas_de_0,05": baja_rc <= 0.05,
                  "f1_grueso_no_peor_que_lineas": o["f1_no_vistos"] >= l_b["f1_no_vistos"]}
        cri[f"H2 {rep}"] = {"veredicto": "confirmada" if all(partes.values()) else "refutada", "partes": partes,
                            "fp_sube": r4(sube_fp), "fp_sube_lineas": r4(sube_fp_l), "recall_baja": r4(baja_rc),
                            "f1_no_vistos": o["f1_no_vistos"], "f1_no_vistos_lineas": l_b["f1_no_vistos"], "R": o["R"]}
        f1 = c["firma"][rep]["1"]
        cri[f"H3 {rep}"] = {"veredicto": "confirmada" if f1["arco-E"] <= 0.30 and f1["arco-W"] <= 0.30 else "refutada",
                            "arco-E_en_los_1": f1["arco-E"], "arco-W_en_los_1": f1["arco-W"]}
    deg = [f for f, v in c.get("canal_ganador", {}).items() if v["n_encendido"] >= 20 and v["maximo"] > 0.90]
    cri["degenera compartido"] = {"veredicto": "sí" if len(deg) > 6 else "no", "detectores_con_un_canal_>90%": deg}
    if comp:
        B = comp["bancos"]
        for rep in ("control", "compartido"):
            if rep not in B or "lineas" not in B:
                continue
            r, l = B[rep], B["lineas"]
            cri[f"H5 {rep}"] = {"veredicto": "confirmada" if r["C1"]["normal"] >= 0.919 else "refutada",
                                "acc": r["C1"]["normal"], "umbral": 0.919, "lineas_medido_aqui": l["C1"]["normal"],
                                "feat_bor_signo": 0.865}
            e = r["C2_180"]
            ok = e["media"]["1"] >= e["normal"] - 0.03 and e["media"]["2"] >= e["normal"] - 0.03
            cri[f"H6 {rep}"] = {"veredicto": "confirmada" if ok else "refutada", "d0": e["normal"],
                                "d1": e["media"]["1"], "d2": e["media"]["2"],
                                "sin_aumentar_d2": r["C1"]["media"]["2"]}
            g, gl = r["C2_180"]["gruesos_reales"], l["C2_180"]["gruesos_reales"]
            cri[f"H7 {rep}"] = {"veredicto": "confirmada" if g >= gl else "refutada", "gruesos_reales": g, "lineas": gl}
    return cri


def main() -> int:
    t0 = time.time(); torch.set_num_threads(2)
    reps = ("lineas",) if "--solo-lineas" in sys.argv else ("lineas", "control", "compartido")
    bancos = {rep: banco(rep) for rep in reps}
    a = seccion_a(reps); b = seccion_b(bancos); c = seccion_c(bancos)
    RES.mkdir(parents=True, exist_ok=True)
    out = {"A": a, "B": b, "C": c, "huellas": {rep: bancos[rep][2] for rep in reps}, "kernels": modelo.HUELLA_KERNELS}
    if reps == ("lineas",):
        out |= {"segundos": round(time.time() - t0, 1),
                "nota": "la referencia de las líneas de feat-ind32, con este mismo código, ANTES de entrenar ningún detector"}
        (RES / "referencia-lineas.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"→ resultados/referencia-lineas.json en {out['segundos']} s")
        return 0
    comp_p = RES / "componer.json"
    comp = json.loads(comp_p.read_text(encoding="utf-8")) if comp_p.is_file() else None
    out["criterio"] = criterio(a, b, c, comp)
    out["segundos"] = round(time.time() - t0, 1)
    (RES / "evaluacion.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n→ resultados/evaluacion.json en {out['segundos']} s")
    for h, v in out["criterio"].items():
        print(f"  {h:<22} {v['veredicto']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
