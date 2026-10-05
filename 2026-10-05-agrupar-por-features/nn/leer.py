#!/usr/bin/env python3
"""La LECTURA de los grupos (L1–L7, REGLAS.md §5) y la evaluación del criterio (instrucciones/02-criterio.md). Aquí, y sólo
aquí, entra la etiqueta: los grupos ya están hechos (nn/entrenar_local.py) y no se tocan.

    python nn/leer.py      → resultados/lecturas.json, y el veredicto por pantalla

Detalles que el criterio no fijaba, decididos ANTES de leer nada con etiquetas (y anotados en REGLAS.md §5):
  · el azar de L1: 20 permutaciones de las etiquetas (semilla 0);
  · «persiste en K′»: el grupo de K′ donde cae la MAYORÍA RELATIVA de los c de cada lado (empate: el de menor número);
  · el bloque de windep de un grupo: sólo si el grupo tiene ≥ 10 miembros de windep;
  · L6: azar con semillas 1001–1005, estratificado con 2001–2005, compositor con semillas 1–3 (el de feat-ind32, copiado).
"""

from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import datos                                    # noqa: E402
import detectores as D                          # noqa: E402
import entrenar_local as E                      # noqa: E402
import metricas as M                            # noqa: E402
import representar as R                         # noqa: E402

RES = AQUI.parent / "resultados"
PRINCIPALES = ("Z32", "Z8")
CONTROL = "X8"
TODOS = ("Z32", "Z8", "P32", "M32", "X8")
K0, KS, K_PERSISTE, K_L6 = 30, (10, 20, 30, 50), (20, 50), (10, 20, 30, 50, 100)
INDICES = ("bandera", "grosor", "inclinacion")
TRANSF = ("desplazar", "engrosar", "adelgazar")
UMBRAL_C, UMBRAL_AUC, MEZCLA, MEZCLA_MIN = 0.10, 0.80, 0.70, 20
# COPIADO de `feat-ind32` (nn/componer.py): el compositor lineal de hoy
EPOCAS, LR, L2 = 300, 1e-2, 1e-3


def r4(v) -> float:
    return round(float(v), 4)


def composicion(lab: np.ndarray, y: np.ndarray, K: int) -> np.ndarray:
    t = np.zeros((K, 10), np.int64)
    np.add.at(t, (lab, y), 1)
    return t


# ---------------------------------------------------------------- L1
def l1(y: np.ndarray, brazos: tuple) -> dict:
    rng = np.random.default_rng(0); perms = [rng.permutation(y) for _ in range(20)]
    out = {}
    for b in brazos:
        out[b] = {}
        for K in KS:
            tod = E.todas("a", b, K); p = min(tod, key=lambda q: q["inercia"])
            m = np.array([[M.pureza(q["grupos"], y), M.nmi(q["grupos"], y), M.ari(q["grupos"], y)] for q in tod])
            out[b][K] = {"semilla_principal": p["semilla"],
                         "principal": {"pureza": r4(M.pureza(p["grupos"], y)), "nmi": r4(M.nmi(p["grupos"], y)),
                                       "ari": r4(M.ari(p["grupos"], y))},
                         "media_5": dict(zip(("pureza", "nmi", "ari"), map(r4, m.mean(0)))),
                         "sd_5": dict(zip(("pureza", "nmi", "ari"), map(r4, m.std(0, ddof=1)))),
                         "azar": {"pureza": r4(np.mean([M.pureza(p["grupos"], q) for q in perms])),
                                  "nmi": r4(np.mean([M.nmi(p["grupos"], q) for q in perms]))}}
    return out


def k10(y: np.ndarray, brazo: str) -> dict:
    """Con tantos grupos como etiquetas: qué clase se parte y qué clases comparten grupo."""
    lab = E.principal("a", brazo, 10)["grupos"]; t = composicion(lab, y, 10)
    clases = {}
    for c in range(10):
        n = t[:, c].sum()
        clases[c] = [{"grupo": int(g), "frac_de_c": r4(t[g, c] / n), "mayoria_del_grupo": int(t[g].argmax()),
                      "frac_mayoria": r4(t[g].max() / t[g].sum())} for g in np.argsort(-t[:, c]) if t[g, c] >= UMBRAL_C * n]
    grupos = [{"grupo": g, "n": int(t[g].sum()),
               "etiquetas": {int(c): r4(t[g, c] / t[g].sum()) for c in np.argsort(-t[g])[:3] if t[g, c] > 0}} for g in range(10)]
    return {"partida": {c: v for c, v in clases.items() if len(v) >= 2}, "clases": clases, "grupos": grupos}


# ---------------------------------------------------------------- L2
def cortes(brazo: str, y: np.ndarray, ind: dict) -> dict:
    p30 = E.principal("a", brazo, K0)["grupos"]
    pk = {k: E.principal("a", brazo, k)["grupos"] for k in K_PERSISTE}
    out = {}
    for c in range(10):
        mc = y == c; nc = int(mc.sum())
        gs = [g for g in range(K0) if ((p30 == g) & mc).sum() >= UMBRAL_C * nc]
        t = composicion(p30, y, K0)
        grupos = [{"grupo": g, "n_c": int(t[g, c]), "frac_de_c": r4(t[g, c] / nc), "n_grupo": int(t[g].sum()),
                   "mayoria": int(t[g].argmax()), "frac_mayoria": r4(t[g].max() / t[g].sum()),
                   **{f"{n}_medio": r4(ind[n][mc & (p30 == g)].mean()) for n in INDICES}} for g in gs]
        pares = []
        for g1, g2 in combinations(gs, 2):
            a1, a2 = mc & (p30 == g1), mc & (p30 == g2)
            pers = {k: int(np.bincount(pk[k][a1]).argmax()) != int(np.bincount(pk[k][a2]).argmax()) for k in K_PERSISTE}
            auc = {n: r4(M.auc_sd(ind[n][a1], ind[n][a2])) for n in INDICES}
            pares.append({"grupos": [g1, g2], "persiste": {str(k): v for k, v in pers.items()},
                          "persiste_en_los_dos": all(pers.values()), "auc": auc,
                          "de_forma": all(pers.values()) and max(auc["grosor"], auc["inclinacion"]) < UMBRAL_AUC,
                          "lo_explica_mejor": max(auc, key=auc.get)})
        out[c] = {"n": nc, "grupos_de_c": grupos, "cortes": pares}
    return out


def h1(cort: dict) -> dict:
    pares = cort[1]["cortes"]; pers = [p for p in pares if p["persiste_en_los_dos"]]
    buenos = [p for p in pers if p["auc"]["bandera"] >= UMBRAL_AUC and p["auc"]["bandera"] > p["auc"]["grosor"]
              and p["auc"]["bandera"] > p["auc"]["inclinacion"]]
    if buenos:
        v, por = "confirmada", "la bandera separa a los 1 de un corte que persiste"
    elif not pers:
        v, por = "refutada", "el 1 no tiene ningún corte que persista en K = 20 y 50"
    else:
        mejor = max(pers, key=lambda p: max(p["auc"].values()))
        v, por = "refutada", f"los cortes del 1 que persisten los explica mejor: {mejor['lo_explica_mejor']}"
    return {"veredicto": v, "por_que": por, "cortes_del_1": len(pares), "que_persisten": len(pers), "con_bandera": buenos}


def h2(cort: dict) -> dict:
    con = {c: [p["grupos"] for p in v["cortes"] if p["de_forma"]] for c, v in cort.items()}
    clases = [c for c, v in con.items() if v]
    return {"veredicto": "confirmada" if len(clases) >= 4 else "refutada", "clases_con_corte_de_forma": clases,
            "n_clases": len(clases), "umbral": 4, "cortes_de_forma": {c: v for c, v in con.items() if v}}


def bandera_corregida(x32: np.ndarray) -> np.ndarray:
    """⚠ A POSTERIORI (escrito tras ver que el testigo `bandera` de L2b estaba sesgado por la inclinación). La bandera medida
    contra el BORDE IZQUIERDO DEL TALLO extrapolado, no contra una columna fija: recta ajustada al borde izquierdo en las
    filas del tallo (de r0 + h/3 a r1 − h/6, sin el sexto inferior, donde puede haber pie), y b′ = el máximo, en el cuarto
    superior, de cuánto se adelanta la tinta a la izquierda de esa recta. Un 1 inclinado sin bandera da b′ ≈ 0 sea cual
    sea su inclinación; el testigo original le restaba (con «/») o le sumaba (con «\\») bandera."""
    out = np.zeros(len(x32))
    for i in range(len(x32)):
        im = x32[i, 0] > 0.5; filas = np.flatnonzero(im.any(1))
        if len(filas) < 8:
            continue
        r0, r1 = filas[0], filas[-1]; h = r1 - r0 + 1
        izq = {int(r): int(np.flatnonzero(im[r])[0]) for r in filas}
        tallo = [r for r in izq if r0 + h / 3 <= r <= r1 - h / 6]
        arriba = [r for r in izq if r <= r0 + h / 4]
        if len(tallo) < 3 or not arriba:
            continue
        pend, orde = np.polyfit(np.array(tallo, float), np.array([izq[r] for r in tallo], float), 1)
        out[i] = max(orde + pend * r - izq[r] for r in arriba)
    return out


def a_posteriori_bandera(y: np.ndarray, ind: dict, brazos: tuple, x32: np.ndarray) -> dict:
    """⚠ A POSTERIORI: NO es del criterio; se escribió DESPUÉS de ver H1 refutada y no cambia su veredicto. (1) Cuántos 1
    tienen bandera según el testigo original y en qué grupos caen los 40 con más bandera (aunque sean grupos con < 10 % de
    los 1, que H1 no mira). (2) Lo mismo con la bandera corregida por la inclinación, y los AUC de H1 rehechos con ella."""
    m1 = np.flatnonzero(y == 1)
    bc = np.zeros(len(y)); bc[m1] = bandera_corregida(x32[m1])
    out = {"n_unos": int(len(m1))}
    for nombre, b in (("bandera", ind["bandera"]), ("bandera_corregida", bc)):
        top = m1[np.argsort(-b[m1], kind="stable")[:40]]
        o = {"percentiles": {str(q): r4(np.percentile(b[m1], q)) for q in (5, 25, 50, 75, 90, 95, 99)},
             "frac_al_menos_4": r4((b[m1] >= 4).mean()), "frac_al_menos_6": r4((b[m1] >= 6).mean()),
             "rango_de_los_40": [r4(b[top].min()), r4(b[top].max())], "donde_caen_los_40": {}}
        for brazo in brazos:
            lab = E.principal("a", brazo, K0)["grupos"]; t = composicion(lab, y, K0)
            cnt = np.bincount(lab[top], minlength=K0)
            o["donde_caen_los_40"][brazo] = [{"grupo": int(g), "de_los_40": int(cnt[g]), "unos_en_el_grupo": int(t[g, 1]),
                                              "frac_de_los_1": r4(t[g, 1] / len(m1)), "mayoria": int(t[g].argmax()),
                                              "frac_mayoria": r4(t[g].max() / t[g].sum())}
                                             for g in np.argsort(-cnt, kind="stable") if cnt[g] > 0]
        out[nombre] = o
    out["H1_rehecha_con_la_corregida"] = {}
    for brazo in brazos:
        p30 = E.principal("a", brazo, K0)["grupos"]
        pk = {k: E.principal("a", brazo, k)["grupos"] for k in K_PERSISTE}
        gs = [g for g in range(K0) if ((p30 == g) & (y == 1)).sum() >= UMBRAL_C * len(m1)]
        filas = []
        for g1, g2 in combinations(gs, 2):
            a1, a2 = (y == 1) & (p30 == g1), (y == 1) & (p30 == g2)
            pers = all(int(np.bincount(pk[k][a1]).argmax()) != int(np.bincount(pk[k][a2]).argmax()) for k in K_PERSISTE)
            auc = {"bandera_corregida": r4(M.auc_sd(bc[a1], bc[a2])), "grosor": r4(M.auc_sd(ind["grosor"][a1], ind["grosor"][a2])),
                   "inclinacion": r4(M.auc_sd(ind["inclinacion"][a1], ind["inclinacion"][a2]))}
            filas.append({"grupos": [g1, g2], "persiste_en_los_dos": pers, "auc": auc,
                          "medias_bandera_corregida": [r4(bc[a1].mean()), r4(bc[a2].mean())],
                          "la_bandera_es_la_mejor_y_>=0,80": auc["bandera_corregida"] >= UMBRAL_AUC
                          and auc["bandera_corregida"] > auc["grosor"] and auc["bandera_corregida"] > auc["inclinacion"]})
        out["H1_rehecha_con_la_corregida"][brazo] = filas
    return out


# ---------------------------------------------------------------- L3
def mezclados(lab: np.ndarray, y: np.ndarray) -> list:
    t = composicion(lab, y, K0)
    return [g for g in range(K0) if t[g].sum() >= MEZCLA_MIN and t[g].max() / t[g].sum() < MEZCLA]


def l3(brazo: str, y: np.ndarray, X: np.ndarray, evals: np.ndarray, fallos: np.ndarray) -> dict:
    p = E.principal("a", brazo, K0); lab = p["grupos"]; t = composicion(lab, y, K0)
    mez = mezclados(lab, y)
    mm = np.isin(lab, mez)
    rf, re = float(mm[fallos].mean()), float(mm[evals].mean())
    d = M.dist2(np.ascontiguousarray(X, np.float32), p["centroides"])[np.arange(len(X)), lab]
    lejos = np.argsort(-d)[:30]
    unicos = set(fallos.tolist())
    return {"grupos_mezclados": [{"grupo": g, "n": int(t[g].sum()),
                                  "etiquetas": {int(c): r4(t[g, c] / t[g].sum()) for c in np.argsort(-t[g])[:3] if t[g, c] > 0}}
                                 for g in mez],
            "frac_digitos_en_mezclados": r4(mm.mean()),
            "fallos_c_en_mezclados": int(mm[fallos].sum()), "fallos_c": int(len(fallos)),
            "frac_fallos_en_mezclados": r4(rf), "frac_evaluaciones_en_mezclados": r4(re),
            "enriquecimiento": r4(rf / re) if re > 0 else None,
            "los_30_mas_lejanos": {"indices": lejos.tolist(), "son_fallos_de_c": int(sum(i in unicos for i in lejos.tolist()))}}


# ---------------------------------------------------------------- L4
def l4(r: dict, brazos: tuple) -> dict:
    out = {}
    for b in brazos:
        tod = E.todas("a", b, K0); prim = min(tod, key=lambda q: q["inercia"])["semilla"]
        filas = {}
        for q in tod:
            lab0 = M.asignar(R.rep(r, b, "limpio"), q["centroides"])
            filas[q["semilla"]] = {}
            for t in TRANSF:
                c = float((M.asignar(R.rep(r, b, t), q["centroides"]) == lab0).mean())
                filas[q["semilla"]][t] = {"c": r4(c), "kappa": r4(M.kappa(c, lab0, K0))}
        out[b] = {"principal": filas[prim], "semilla_principal": prim,
                  "media_5": {t: {k: r4(np.mean([filas[s][t][k] for s in filas])) for k in ("c", "kappa")} for t in TRANSF}}
    return out


def h4(cons: dict, z: str) -> dict:
    d = {t: r4(cons[z]["principal"][t]["kappa"] - cons[CONTROL]["principal"][t]["kappa"]) for t in ("engrosar", "adelgazar")}
    if d["engrosar"] >= 0.05 and d["adelgazar"] >= 0.05:
        v = "confirmada"
    elif d["engrosar"] <= -0.05 or d["adelgazar"] <= -0.05:
        v = "refutada"
    else:
        v = "empate"
    return {"veredicto": v, "kappa_Z_menos_X8": d,
            "desplazar_kappa": {b: cons[b]["principal"]["desplazar"]["kappa"] for b in (z, "M32", CONTROL)}}


# ---------------------------------------------------------------- L5
def ari_semillas(b: str, K: int = K0, ajuste: str = "a") -> list:
    tod = E.todas(ajuste, b, K)
    return [M.ari(p["grupos"], q["grupos"]) for p, q in combinations(tod, 2)]


def l5(r: dict, y: np.ndarray, origen: np.ndarray, brazos: tuple) -> dict:
    out = {"semillas": {b: {"media": r4(np.mean(ari_semillas(b))), "min": r4(np.min(ari_semillas(b)))} for b in brazos}}
    w = origen == "windep"
    out["escritores_nuevos"] = {}
    for b in ("Z32", "Z8", "X8"):
        pb, pc = E.principal("b", b, K0), E.principal("c", b, K0)
        lab_b = M.asignar(R.rep(r, b)[w], pb["centroides"])
        a = M.ari(lab_b, pc["grupos"]); ref = float(np.mean(ari_semillas(b)))
        cnt = np.bincount(lab_b, minlength=K0)
        out["escritores_nuevos"][b] = {"ari_asignados_vs_aparte": r4(a), "ari_semillas_5620": r4(ref), "cociente": r4(a / ref),
                                       "ari_semillas_1797_aparte": r4(np.mean(ari_semillas(b, K0, "c"))),
                                       "grupos_con_1pct_de_los_1797": int((cnt >= 0.01 * w.sum()).sum()),
                                       "grupos_casi_vacios_en_1797": [int(g) for g in np.flatnonzero(cnt < 0.01 * w.sum())]}
    prims = {b: E.principal("a", b, K0)["grupos"] for b in brazos}
    out["ari_entre_brazos"] = {f"{a}~{b}": r4(M.ari(prims[a], prims[b])) for a, b in combinations(brazos, 2)}
    bl = datos.bloques(origen); out["bloques_windep"] = {}
    for b in ("Z32", "Z8", "X8"):
        lab = prims[b]; marcados = []
        for g in range(K0):
            m = (lab == g) & (bl >= 0)
            if m.sum() >= 10:
                cuenta = np.bincount(bl[m], minlength=len(datos.BLOQUES_WINDEP))
                if cuenta.max() / m.sum() >= 0.5:
                    marcados.append({"grupo": g, "n_windep": int(m.sum()), "bloque": int(cuenta.argmax()),
                                     "frac": r4(cuenta.max() / m.sum()), "mayoria": int(np.bincount(y[lab == g]).argmax())})
        out["bloques_windep"][b] = marcados
    return out


# ---------------------------------------------------------------- L6
def ajustar(x: np.ndarray, y: np.ndarray, sem: int) -> torch.nn.Linear:
    """COPIA de `feat-ind32` (nn/componer.py): Linear, Adam, 300 épocas a lote completo, lr 1e-2, L2 1e-3."""
    torch.manual_seed(sem)
    W = torch.nn.Linear(x.shape[1], 10); opt = torch.optim.Adam(W.parameters(), lr=LR, weight_decay=L2)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(EPOCAS):
        opt.zero_grad(); Fn.cross_entropy(W(xt), yt).backward(); opt.step()
    return W


def predecir(W, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return W(torch.from_numpy(x)).argmax(1).numpy()


def l6(r: dict, y: np.ndarray, origen: np.ndarray) -> dict:
    w, bmask = origen == "windep", origen != "windep"
    bidx = np.flatnonzero(bmask)
    mapa = np.ascontiguousarray(R.rep(r, "M32"), np.float32)
    xw, yw = mapa[w], y[w]

    def acierto(idx: np.ndarray) -> float:
        return float(np.mean([(predecir(ajustar(mapa[idx], y[idx], s), xw) == yw).mean() for s in (1, 2, 3)]))

    out = {}
    for K in K_L6:
        fila = {}
        for b in ("Z32", "Z8", "X8"):
            pb = E.principal("b", b, K)
            med = bidx[M.medoides(R.rep(r, b)[bmask], pb["centroides"], pb["grupos"])]
            lab_w = M.asignar(R.rep(r, b)[w], pb["centroides"])
            fila[f"medoides_{b}"] = {"acierto": r4(acierto(med)), "clases_cubiertas": int(len(set(y[med].tolist()))),
                                     "propagacion": r4((y[med][lab_w] == yw).mean())}
        az = [acierto(np.random.default_rng(1000 + d).choice(bidx, K, replace=False)) for d in range(1, 6)]
        es = []
        for d in range(1, 6):
            rng = np.random.default_rng(2000 + d)
            es.append(acierto(np.concatenate([rng.choice(bidx[y[bidx] == c], K // 10, replace=False) for c in range(10)])))
        fila["azar"] = {"media": r4(np.mean(az)), "sd": r4(np.std(az, ddof=1))}
        fila["estratificado"] = {"media": r4(np.mean(es)), "sd": r4(np.std(es, ddof=1))}
        out[K] = fila
        print(f"  L6 K={K:<3} medoides Z32 {fila['medoides_Z32']['acierto']:.3f} · Z8 {fila['medoides_Z8']['acierto']:.3f} · "
              f"X8 {fila['medoides_X8']['acierto']:.3f} · azar {fila['azar']['media']:.3f} · "
              f"estratificado {fila['estratificado']['media']:.3f}", flush=True)
    return out


def h6(l6r: dict, z: str) -> dict:
    ks = [K for K in (20, 30, 50) if l6r[K][f"medoides_{z}"]["acierto"] - l6r[K]["medoides_X8"]["acierto"] >= 0.02
          and l6r[K][f"medoides_{z}"]["acierto"] - l6r[K]["azar"]["media"] >= 0.03]
    return {"veredicto": "confirmada" if len(ks) >= 2 else "refutada", "K_que_cumplen": ks,
            "por_K": {K: {"Z_menos_X8": r4(l6r[K][f"medoides_{z}"]["acierto"] - l6r[K]["medoides_X8"]["acierto"]),
                          "Z_menos_azar": r4(l6r[K][f"medoides_{z}"]["acierto"] - l6r[K]["azar"]["media"])} for K in (20, 30, 50)}}


# ---------------------------------------------------------------- todo
def main() -> int:
    t0 = time.time()
    r = R.cargar(); origen = r["origen"].astype(str)
    y = datos.etiquetas()                                            # ← AQUÍ entra la etiqueta, y sólo aquí
    ind = {n: r[n] for n in INDICES}
    regla = json.loads((RES / "regla-sin-arcos.json").read_text(encoding="utf-8"))
    extra = tuple(f"{b}-a" for b, v in regla.items() if v["corre_sin_arcos"])
    print("C1 · las copias y la particion():", flush=True)
    c1 = D.comprobar() == 0 and datos.huella_particiones(y) == json.loads(
        (datos.por_id("feat-ind32").carpeta / "resultados" / "ganancia.json").read_text(encoding="utf-8"))["huella_particiones"]
    out = {"brazos_extra_sin_arcos": list(extra)}
    out["L1"] = l1(y, TODOS + extra)
    out["L1_K10"] = {b: k10(y, b) for b in ("Z32", "Z8", "X8")}
    out["L2"] = {b: cortes(b, y, ind) for b in ("Z32", "Z8", "X8") + extra}
    out["a_posteriori_bandera"] = a_posteriori_bandera(y, ind, ("Z32", "Z8", "X8") + extra, datos.cargar()["x32"])
    evals, fallos = datos.evaluaciones_c(y)
    out["L3"] = {b: l3(b, y, R.rep(r, b), evals, fallos) for b in ("Z32", "Z8", "X8")}
    out["L4"] = l4(r, TODOS + extra)
    out["L5"] = l5(r, y, origen, TODOS)
    out["L6"] = l6(r, y, origen)
    out["L7"] = {b: json.loads((RES / f"grupos-{b}-K30.json").read_text(encoding="utf-8"))["que_manda"]
                 for b in ("Z32", "Z8") + extra}
    out["L7_regla_sin_arcos"] = regla
    out["composicion_K30"] = {b: composicion(E.principal("a", b, K0)["grupos"], y, K0).tolist() for b in ("Z32", "Z8", "X8") + extra}
    cri = {"C1": {"cumple": bool(c1)}}
    cri["C2"] = {z: {"nmi": out["L1"][z][K0]["principal"]["nmi"], "nmi_azar": out["L1"][z][K0]["azar"]["nmi"],
                     "cumple": out["L1"][z][K0]["principal"]["nmi"] >= 10 * out["L1"][z][K0]["azar"]["nmi"]} for z in PRINCIPALES}
    cri["H0"] = {}
    for z in PRINCIPALES:
        dif = {K: r4(out["L1"][z][K]["principal"]["nmi"] - out["L1"][CONTROL][K]["principal"]["nmi"]) for K in (10, 30)}
        v = "más cerca" if all(d >= 0.03 for d in dif.values()) else ("más lejos" if all(d <= -0.03 for d in dif.values()) else "empate")
        cri["H0"][z] = {"veredicto": v, "nmi_Z_menos_X8": dif}
    cri["H1"] = {b: h1(out["L2"][b]) for b in PRINCIPALES + extra}
    cri["H1_X8_sin_umbral"] = h1(out["L2"][CONTROL])
    cri["H2"] = {b: h2(out["L2"][b]) for b in PRINCIPALES + extra}
    cri["H2_X8_sin_umbral"] = h2(out["L2"][CONTROL])
    cri["H3"] = {}
    for z in PRINCIPALES:
        a, x = out["L3"][z], out["L3"][CONTROL]
        ok = (a["enriquecimiento"] or 0) >= 2 and a["fallos_c_en_mezclados"] >= 5 and (a["enriquecimiento"] or 0) > (x["enriquecimiento"] or 0)
        cri["H3"][z] = {"veredicto": "confirmada" if ok else "refutada", "enriquecimiento": a["enriquecimiento"],
                        "fallos_en_mezclados": a["fallos_c_en_mezclados"], "enriquecimiento_X8": x["enriquecimiento"]}
    cri["H4"] = {b: h4(out["L4"], b) for b in PRINCIPALES + extra}
    cri["H5"] = {}
    for z in PRINCIPALES:
        s = out["L5"]["semillas"][z]["media"]; en = out["L5"]["escritores_nuevos"][z]
        partes = {"semillas": s >= 0.60, "escritores_ari": en["cociente"] >= 0.8, "escritores_grupos": en["grupos_con_1pct_de_los_1797"] >= 24}
        cri["H5"][z] = {"veredicto": "confirmada" if all(partes.values()) else "refutada", "partes": partes,
                        "ari_semillas": s, "cociente_escritores": en["cociente"], "grupos_con_1pct": en["grupos_con_1pct_de_los_1797"]}
    cri["H6"] = {z: h6(out["L6"], z) for z in PRINCIPALES}
    out["criterio"] = cri
    out["segundos"] = round(time.time() - t0, 1)
    (RES / "lecturas.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n→ resultados/lecturas.json en {out['segundos']} s\n")
    print(f"C1 {'cumple' if c1 else 'FALLA'} · C2 " + " · ".join(f"{z} NMI {cri['C2'][z]['nmi']:.3f} (azar {cri['C2'][z]['nmi_azar']:.3f})"
                                                              for z in PRINCIPALES))
    for h in ("H0", "H1", "H2", "H3", "H4", "H5", "H6"):
        print(f"{h}: " + " · ".join(f"{b} {v['veredicto']}" for b, v in cri[h].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
