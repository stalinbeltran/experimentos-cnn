#!/usr/bin/env python3
"""La EVALUACIÓN de `feat-bor` (instrucciones/02-criterio.md), escrita ANTES de entrenar. Tres bancos de 13 detectores:

  lineas    los de `feat-ind32` (C3), la referencia: se leen de su carpeta POR SU ID y se comprueban por su huella (no se
            copian aquí, por el tope de tamaño del repo: los 26 nuevos ya ocupan ~4,5 MB)
  contorno  nn/pesos-contorno/<f>/best.pt  (borde sin signo, 1 canal)
  signo     nn/pesos-signo/<f>/best.pt     (borde con signo, 4 canales)

  §A  val sintético (2–4 px, el de entrenar): F1, P, R, pos ≤ 1 y veredicto, de cada summary.json
  §B  la prueba de GROSOR (`feat-bor-sinteticas-grueso-32px-r20261006`): recall por grosor de la principal, FP por grosor
      del negativo, y r = F1(6–12 px) ÷ F1(2–4 px) de cada detector —positivos y negativos del mismo tramo de grosor—;
      R = la media de los detectores con ≥ 20 positivos en los dos tramos. F1 y no recall: medido con las líneas antes de
      entrenar nada, el grosor no les quita recall, les da FALSOS POSITIVOS (criterio, enmienda previa)
  §C  los dígitos (los 5620): la firma por clase (windep), la consistencia de los grupos (zonas, K = 30, κ) al engrosar 2 px
      y adelgazar 1 px —la medida que refutó H4 en feat-agr—, y el compositor posicional 180/1617

    python nn/evaluar.py                 → resultados/evaluacion.json, resultados/grosor.png, resultados/consistencia.png
    python nn/evaluar.py --solo-lineas   → resultados/referencia-lineas.json: la referencia, ANTES de entrenar nada
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
sys.path.insert(0, str(EXP.parent))
sys.path.insert(0, str(AQUI))
from expcnn import por_id                       # noqa: E402
import bordes as B                              # noqa: E402
import datos                                    # noqa: E402
import entrenar_local as T                      # noqa: E402
import features as F                            # noqa: E402
import metricas as M                            # noqa: E402
import modelo                                   # noqa: E402

RES = EXP / "resultados"
REPS = ("lineas", "contorno", "signo")
VISTOS, NO_VISTOS = (2, 3, 4), (6, 8, 10, 12)
K0, SEMILLAS = 30, (1, 2, 3, 4, 5)
INICIOS = (0, 2, 4)                               # zonas: ventanas 4×4 celdas con paso 2 (las de feat-agr)
# COPIADO de `feat-ind32` (nn/componer.py): las transformaciones de C6 y el compositor lineal de C4
TRANSFORMACIONES = ("desplazar", "ruido", "engrosar", "adelgazar", "ocluir")
SEMILLA_T = 2027
EPOCAS, LR, L2 = 300, 1e-2, 1e-3


def r4(v) -> float:
    return round(float(v), 4)


def transformar(x: np.ndarray, que: str) -> np.ndarray:
    """COPIA de `feat-ind32` (C6): engrosar = dilatar 2 px · adelgazar = erosionar 1 px. x (N,1,32,32) float 0/1."""
    t = torch.from_numpy(x)
    if que == "engrosar":
        return Fn.max_pool2d(t, 5, 1, 2).numpy()
    if que == "adelgazar":
        return (-Fn.max_pool2d(-t, 3, 1, 1)).numpy()
    raise ValueError(que)


def banco(rep: str) -> tuple[list, np.ndarray, dict]:
    """Los 13 detectores de una representación, sus umbrales y de dónde salen. `lineas` se comprueba por su huella."""
    reds, um, origen = [], np.zeros(len(F.CON_TRAZO), np.float32), {}
    if rep == "lineas":
        org = por_id("feat-ind32").carpeta
        huellas = json.loads((org / "resultados" / "firma-por-clase.json").read_text(encoding="utf-8"))["huellas_best"]
    for j, f in enumerate(F.CON_TRAZO):
        ruta = (por_id("feat-ind32").carpeta / "nn" / "pesos" / f / "best.pt") if rep == "lineas" \
            else T.carpeta_pesos(rep) / f / "best.pt"
        if not ruta.is_file():
            raise SystemExit(f"✗ falta {ruta}: los detectores de '{rep}' no están entrenados (nn/vast.sh detectores)")
        red, est = modelo.cargar(ruta)
        h = modelo.huella_pesos(red)
        if rep == "lineas" and h != huellas[f]:
            raise SystemExit(f"✗ feat-ind32/{f}: la huella {h} no es la de su firma ({huellas[f]}): sus pesos cambiaron. Me niego.")
        reds.append(red); um[j] = est["umbral"]; origen[f] = h
    return reds, um, origen


@torch.no_grad()
def mapas(reds: list, x: np.ndarray, lote: int = 1024) -> np.ndarray:
    out = np.zeros((len(x), len(reds), 8, 8), np.float32)
    for i in range(0, len(x), lote):
        xt = torch.from_numpy(np.ascontiguousarray(x[i:i + lote], dtype=np.float32))
        for j, red in enumerate(reds):
            out[i:i + lote, j] = torch.sigmoid(red(xt))[:, 0].numpy()
    return out


def zonas(s: np.ndarray) -> np.ndarray:
    out = np.empty((len(s), s.shape[1], 3, 3), np.float32)
    for i, r in enumerate(INICIOS):
        for j, c in enumerate(INICIOS):
            out[:, :, i, j] = s[:, :, r:r + 4, c:c + 4].max((2, 3))
    return out.reshape(len(s), -1)


# ------------------------------------------------------------------------------------------------ §A
def seccion_a() -> dict:
    out = {}
    for rep in REPS:
        base = (por_id("feat-ind32").carpeta / "nn" / "pesos") if rep == "lineas" else T.carpeta_pesos(rep)
        filas = {}
        for f in F.CON_TRAZO:
            s = json.loads((base / f / "summary.json").read_text(encoding="utf-8"))
            b = s["best"]
            filas[f] = {k: b[k] for k in ("f1", "precision", "recall", "pos_ok", "umbral")} | {"veredicto": s["veredicto"],
                                                                                              "epoca": b["epoca"]}
        n_ok = sum(v["veredicto"] in ("aprendió", "a medias") for v in filas.values())
        out[rep] = {"detectores": filas, "al_menos_a_medias": n_ok,
                    "aprendio": sum(v["veredicto"] == "aprendió" for v in filas.values())}
    return out


# ------------------------------------------------------------------------------------------------ §B
@torch.no_grad()
def seccion_b(bancos: dict) -> dict:
    g = datos.cargar(datos.GRUESO)
    out = {}
    for rep in REPS:
        reds, um, _ = bancos[rep]
        por_f = {}
        for j, f in enumerate(F.CON_TRAZO):
            c = datos.conjunto(g, f, None, rep)
            sp = torch.sigmoid(reds[j](torch.from_numpy(c["x_pos"]).float())).flatten(1)
            sn = torch.sigmoid(reds[j](torch.from_numpy(c["x_neg"]).float())).flatten(1)
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
        print(f"  §B {rep:<9} R {o['R']:.3f} ({len(val)} detectores) · F1 2–4 px {o['f1_vistos']:.3f} → 6–12 px {o['f1_no_vistos']:.3f}"
              f" · recall {o['recall_vistos']:.3f} → {o['recall_no_vistos']:.3f} · FP {o['fp_tasa_vistos']:.3f} → "
              f"{o['fp_tasa_no_vistos']:.3f}", flush=True)
    return out


# ------------------------------------------------------------------------------------------------ §C
def ajustar(x: np.ndarray, y: np.ndarray, sem: int) -> torch.nn.Linear:
    """COPIA de `feat-ind32` (nn/componer.py): Linear, Adam, 300 épocas a lote completo, lr 1e-2, L2 1e-3."""
    torch.manual_seed(sem)
    W = torch.nn.Linear(x.shape[1], 10); opt = torch.optim.Adam(W.parameters(), lr=LR, weight_decay=L2)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(EPOCAS):
        opt.zero_grad(); Fn.cross_entropy(W(xt), yt).backward(); opt.step()
    return W


def seccion_c(bancos: dict) -> dict:
    dg = datos.digitos(); y = dg["y"]; w = dg["origen"] == "windep"
    x0 = dg["x"].astype(np.float32)
    versiones = {"limpio": x0, "engrosar": transformar(x0, "engrosar"), "adelgazar": transformar(x0, "adelgazar")}
    out = {"firma": {}, "consistencia": {}, "compositor": {}}
    descr = {}
    for rep in REPS:
        reds, um, _ = bancos[rep]
        s = {v: mapas(reds, B.bordes(xv, rep)) for v, xv in versiones.items()}
        pres = s["limpio"].reshape(len(y), 13, -1).max(2) >= um[None]
        out["firma"][rep] = {str(c): {f: r4(pres[w & (y == c), j].mean()) for j, f in enumerate(F.CON_TRAZO)} for c in range(10)}
        descr[rep] = {v: zonas(sv) for v, sv in s.items()}
        tr, va = dg["train"], dg["val"]
        xm = s["limpio"].reshape(len(y), -1)
        accs = [float((ajustar(xm[tr], y[tr], sem)(torch.from_numpy(xm[va])).argmax(1).numpy() == y[va]).mean()) for sem in (1, 2, 3)]
        out["compositor"][rep] = {"acc_val_media": r4(np.mean(accs)), "acc_val_por_semilla": [r4(a) for a in accs]}
        print(f"  §C {rep:<9} compositor posicional 180/1617: {np.mean(accs):.4f} · arco-E/arco-W en los 1: "
              f"{out['firma'][rep]['1']['arco-E']:.2f} / {out['firma'][rep]['1']['arco-W']:.2f} · lazo en los 1: "
              f"{out['firma'][rep]['1']['lazo']:.2f}", flush=True)
    descr["X8 (píxeles)"] = {v: xv.reshape(-1, 1, 8, 4, 8, 4).sum((3, 5)).reshape(len(y), -1) / 16 for v, xv in versiones.items()}
    for rep, d in descr.items():
        tod = [M.kmeans(d["limpio"], K0, s) for s in SEMILLAS]
        i = int(np.argmin([q["inercia"] for q in tod])); C = tod[i]["centroides"]
        lab0 = M.asignar(d["limpio"], C)
        fila = {"semilla_principal": SEMILLAS[i]}
        for t in ("engrosar", "adelgazar"):
            c = float((M.asignar(d[t], C) == lab0).mean())
            fila[t] = {"c": r4(c), "kappa": r4(M.kappa(c, lab0, K0))}
        out["consistencia"][rep] = fila
        print(f"  §C {rep:<13} κ engrosar {fila['engrosar']['kappa']:.3f} · adelgazar {fila['adelgazar']['kappa']:.3f}", flush=True)
    return out


# ------------------------------------------------------------------------------------------------ criterio
def criterio(a: dict, b: dict, c: dict) -> dict:
    cri = {}
    for rep in ("contorno", "signo"):
        f1m = float(np.mean([v["f1"] for v in a[rep]["detectores"].values()]))
        f1l = float(np.mean([v["f1"] for v in a["lineas"]["detectores"].values()]))
        ok = a[rep]["al_menos_a_medias"] >= 11 and f1m >= f1l - 0.05
        cri[f"H1 {rep}"] = {"veredicto": "confirmada" if ok else "refutada", "al_menos_a_medias": a[rep]["al_menos_a_medias"],
                            "f1_medio": r4(f1m), "f1_medio_lineas": r4(f1l)}
        o, l = b[rep], b["lineas"]
        sube_fp, sube_fp_l = o["fp_tasa_no_vistos"] - o["fp_tasa_vistos"], l["fp_tasa_no_vistos"] - l["fp_tasa_vistos"]
        baja_rc = o["recall_vistos"] - o["recall_no_vistos"]
        partes = {"fp_sube_como_mucho_la_mitad_que_lineas": sube_fp <= sube_fp_l / 2,
                  "recall_no_baja_mas_de_0,05": baja_rc <= 0.05,
                  "f1_grueso_no_peor_que_lineas": o["f1_no_vistos"] >= l["f1_no_vistos"]}
        cri[f"H2 {rep}"] = {"veredicto": "confirmada" if all(partes.values()) else "refutada", "partes": partes,
                            "fp_sube": r4(sube_fp), "fp_sube_lineas": r4(sube_fp_l), "recall_baja": r4(baja_rc),
                            "f1_no_vistos": o["f1_no_vistos"], "f1_no_vistos_lineas": l["f1_no_vistos"],
                            "R": o["R"], "R_lineas": l["R"]}
        k, kl = c["consistencia"][rep], c["consistencia"]["lineas"]
        ok = all(k[t]["kappa"] >= kl[t]["kappa"] + 0.20 for t in ("engrosar", "adelgazar"))
        cri[f"H4 {rep}"] = {"veredicto": "confirmada" if ok else "refutada",
                            "kappa": {t: k[t]["kappa"] for t in ("engrosar", "adelgazar")},
                            "kappa_lineas": {t: kl[t]["kappa"] for t in ("engrosar", "adelgazar")}}
        acc, accl = c["compositor"][rep]["acc_val_media"], c["compositor"]["lineas"]["acc_val_media"]
        cri[f"H5 {rep}"] = {"veredicto": "confirmada" if acc >= accl - 0.03 else "refutada", "acc": acc, "acc_lineas": accl}
    f1 = c["firma"]["signo"]["1"]
    cri["H3 signo"] = {"veredicto": "confirmada" if f1["arco-E"] <= 0.30 and f1["arco-W"] <= 0.30 else "refutada",
                       "arco-E_en_los_1": f1["arco-E"], "arco-W_en_los_1": f1["arco-W"],
                       "lineas": {k: c["firma"]["lineas"]["1"][k] for k in ("arco-E", "arco-W")}}
    return cri


def figuras(b: dict, c: dict) -> None:
    import matplotlib                                                # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                                  # noqa: PLC0415
    SUP, T1, T2, REJ = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    COL = {"lineas": "#2a78d6", "contorno": "#eb6834", "signo": "#1baf7a"}
    gros = (2, 3, 4, 6, 8, 10, 12)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=100, facecolor=SUP); ax.set_facecolor(SUP)
    for rep in REPS:
        ys = [np.mean([v["por_grosor"][str(g)]["recall"] for v in b[rep]["detectores"].values() if str(g) in v["por_grosor"]])
              for g in gros]
        ax.plot(gros, ys, color=COL[rep], lw=2, marker="o", ms=5, label=rep, solid_capstyle="round")
    ax.axvspan(4.5, 12.5, color=REJ, alpha=0.6, lw=0, zorder=0)
    ax.text(8.5, 0.04, "grosores NO vistos al entrenar", ha="center", color=T2, fontsize=9)
    ax.set_xticks(gros); ax.set_ylim(0, 1); ax.set_xlabel("grosor del trazo (px de 32)", color=T2, fontsize=9)
    ax.set_ylabel("recall medio de los 13 detectores", color=T2, fontsize=9)
    ax.tick_params(colors=T2, labelsize=8); ax.grid(axis="y", color=REJ, lw=1)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(REJ)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.13), labelcolor=T1)
    ax.set_title("¿Reconoce trazos más gruesos que los que vio? (prueba sintética)", color=T1, fontsize=11, loc="left", pad=28)
    fig.tight_layout(); fig.savefig(RES / "grosor.png", facecolor=SUP); plt.close(fig)
    reps = list(c["consistencia"]); trans = ("engrosar", "adelgazar")
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=100, facecolor=SUP); ax.set_facecolor(SUP)
    for j, t in enumerate(trans):
        ax.bar(np.arange(len(reps)) + (j - 0.5) * 0.3, [c["consistencia"][r][t]["kappa"] for r in reps], 0.28,
               color=("#eb6834", "#1baf7a")[j], label={"engrosar": "engrosar 2 px", "adelgazar": "adelgazar 1 px"}[t], zorder=3)
    ax.set_xticks(range(len(reps))); ax.set_xticklabels(reps, color=T1, fontsize=10); ax.set_ylim(0, 1)
    ax.set_ylabel("κ (consistencia corregida por azar)", color=T2, fontsize=9)
    ax.tick_params(axis="y", colors=T2, labelsize=8); ax.tick_params(axis="x", length=0); ax.grid(axis="y", color=REJ, lw=1, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(REJ)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.13), labelcolor=T1)
    ax.set_title("Dígitos: ¿el mismo dígito, más grueso o más fino, cae en su grupo? (K = 30)", color=T1, fontsize=11,
                 loc="left", pad=28)
    fig.tight_layout(); fig.savefig(RES / "consistencia.png", facecolor=SUP); plt.close(fig)


def main() -> int:
    global REPS
    t0 = time.time(); torch.set_num_threads(2)
    if "--solo-lineas" in sys.argv:
        REPS = ("lineas",)
        bancos = {"lineas": banco("lineas")}
        a = seccion_a(); b = seccion_b(bancos); c = seccion_c(bancos)
        out = {"A": a, "B": b, "C": c, "huellas": {"lineas": bancos["lineas"][2]}, "segundos": round(time.time() - t0, 1),
               "nota": "la referencia de las líneas de feat-ind32, calculada con este mismo código ANTES de entrenar ningún detector de borde"}
        RES.mkdir(parents=True, exist_ok=True)
        (RES / "referencia-lineas.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"→ resultados/referencia-lineas.json en {out['segundos']} s")
        return 0
    bancos = {rep: banco(rep) for rep in REPS}
    a = seccion_a(); b = seccion_b(bancos); c = seccion_c(bancos)
    out = {"A": a, "B": b, "C": c, "criterio": criterio(a, b, c),
           "huellas": {rep: bancos[rep][2] for rep in REPS}, "segundos": round(time.time() - t0, 1)}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "evaluacion.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    figuras(b, c)
    print(f"\n→ resultados/evaluacion.json en {out['segundos']} s")
    for h, v in out["criterio"].items():
        print(f"  {h:<12} {v['veredicto']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
