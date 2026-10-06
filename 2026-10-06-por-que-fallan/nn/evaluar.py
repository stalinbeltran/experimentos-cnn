#!/usr/bin/env python3
"""La EVALUACIÓN de `feat-fallos`: una COMBINACIÓN = un banco de detectores × un preprocesado de la imagen, medida siempre
con las mismas cuatro cosas (instrucciones/02-criterio.md):

  compositor   el posicional de feat-ind32 (13 mapas 8×8 → 10) con 180 de train y 1617 de val, 3 semillas; y la curva
               con 36 / 180 / 1080 de train sobre el test fijo de 717 (el reparto de la corrida 11 de feat-ind)
  firma        en qué fracción de cada dígito se enciende cada detector (los 1797 windep); el síntoma: arcos en los 1
  κ            la consistencia de los grupos (zonas 13×9, K = 30, la de menor inercia de 5 semillas) al engrosar 2 px y
               al adelgazar 1 px el dígito ANTES del preprocesado — la medida que refutó H4 en feat-agr
  grueso       la prueba de grosor de feat-bor (2–12 px): F1, recall y falsos positivos en 2–4 px y en 6–12 px

Bancos (se leen de su experimento POR SU ID y se comprueban por huella cuando el origen la guarda):
  lineas     feat-ind32 (C3)        · entrada: la tinta
  contorno   feat-bor               · entrada: el contorno de la tinta
  signo      feat-bor               · entrada: el borde con signo (4 canales)
  (y los que entrene este experimento o feat-cortas, cuando existan)
Preprocesados: `nada` · `norm3` (esqueleto + 3 px, nn/normalizar.py)

    python nn/evaluar.py lineas nada            → resultados/combos/lineas-nada.json
    python nn/evaluar.py --todas                → las que falten
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
sys.path.insert(0, str(EXP.parent)); sys.path.insert(0, str(AQUI))
from expcnn import exigir_dataset, por_id       # noqa: E402
import bordes as B                              # noqa: E402
import features as F                            # noqa: E402
import metricas as M                            # noqa: E402
import modelo                                   # noqa: E402
import normalizar as N                          # noqa: E402

RES = EXP / "resultados"; COMBOS = RES / "combos"
DIGITOS, GRUESO = "uci-optdigits-orig-32px-r20261005", "feat-bor-sinteticas-grueso-32px-r20261006"
VISTOS, NO_VISTOS = (2, 3, 4), (6, 8, 10, 12)
K0, SEMILLAS = 30, (1, 2, 3, 4, 5)
EPOCAS, LR, L2 = 300, 1e-2, 1e-3                 # el compositor de feat-ind32 (C4), copiado
SEMILLA_REPARTO, RESERVA_POR_CLASE, TAMANOS = 2026, 90, (36, 180, 1080)   # el reparto de feat-ind (corrida 11), copiado
PREPROCESADOS = ("nada", "norm3", "nada+norm3")     # «a+b»: los mapas de las dos vistas, juntos (26)


def r4(v) -> float:
    return round(float(v), 4)


# ------------------------------------------------------------------------------------------------ los bancos
def _origen_banco(nombre: str) -> tuple[Path, str, dict | None]:
    """(carpeta de pesos, representación de entrada, huellas esperadas o None)."""
    if nombre == "lineas":
        org = por_id("feat-ind32").carpeta
        h = json.loads((org / "resultados" / "firma-por-clase.json").read_text(encoding="utf-8"))["huellas_best"]
        return org / "nn" / "pesos", "lineas", h
    if nombre in ("contorno", "signo"):
        org = por_id("feat-bor").carpeta
        ev = org / "resultados" / "evaluacion.json"
        h = json.loads(ev.read_text(encoding="utf-8"))["huellas"][nombre] if ev.is_file() else None
        return org / "nn" / f"pesos-{nombre}", nombre, h
    propio = AQUI / f"pesos-{nombre}"
    if propio.is_dir():
        rep = json.loads((propio / "arco-E" / "config.json").read_text(encoding="utf-8")).get("representacion", "lineas")
        return propio, rep, None
    raise SystemExit(f"✗ banco '{nombre}' desconocido")


def banco(nombre: str, familias=F.CON_TRAZO) -> dict:
    carpeta, rep, huellas = _origen_banco(nombre)
    reds, um, hs = [], np.zeros(len(familias), np.float32), {}
    for j, f in enumerate(familias):
        red, est = modelo.cargar(carpeta / f / "best.pt")
        h = modelo.huella_pesos(red)
        if huellas is not None and h != huellas[f]:
            raise SystemExit(f"✗ {nombre}/{f}: la huella {h} no es la que guarda su origen ({huellas[f]}). Me niego.")
        reds.append(red); um[j] = est["umbral"]; hs[f] = h
    return {"nombre": nombre, "rep": rep, "reds": reds, "umbrales": um, "huellas": hs, "familias": list(familias)}


def preparar(x: np.ndarray, rep: str, prepro: str) -> np.ndarray:
    """x (N,1,32,32) 0/1 → la entrada del banco: primero el preprocesado, después la representación. Un preprocesado es una
    CADENA de pasos unidos por «_», que se aplican en orden: `nada` · `norm3` · `desinc` · `desinc_norm3` (la iteración 4)."""
    for paso in prepro.split("_"):
        if paso == "norm3":
            x = N.normalizar(x, 3)
        elif paso == "desinc":
            x = N.desinclinar(x)
        elif paso != "nada":
            raise ValueError(prepro)
    return B.bordes(x, rep).astype(np.float32)


@torch.no_grad()
def mapas(b: dict, x: np.ndarray, lote: int = 1024) -> np.ndarray:
    out = np.zeros((len(x), len(b["reds"]), 8, 8), np.float32)
    for i in range(0, len(x), lote):
        xt = torch.from_numpy(np.ascontiguousarray(x[i:i + lote]))
        for j, red in enumerate(b["reds"]):
            out[i:i + lote, j] = torch.sigmoid(red(xt))[:, 0].numpy()
    return out


def zonas(s: np.ndarray) -> np.ndarray:
    out = np.empty((len(s), s.shape[1], 3, 3), np.float32)
    for i, r in enumerate((0, 2, 4)):
        for j, c in enumerate((0, 2, 4)):
            out[:, :, i, j] = s[:, :, r:r + 4, c:c + 4].max((2, 3))
    return out.reshape(len(s), -1)


def transformar(x: np.ndarray, que: str) -> np.ndarray:
    """COPIA de feat-ind32 (C6): engrosar = dilatar 2 px · adelgazar = erosionar 1 px."""
    t = torch.from_numpy(x)
    return Fn.max_pool2d(t, 5, 1, 2).numpy() if que == "engrosar" else (-Fn.max_pool2d(-t, 3, 1, 1)).numpy()


def ajustar(x: np.ndarray, y: np.ndarray, sem: int) -> torch.nn.Linear:
    torch.manual_seed(sem)
    W = torch.nn.Linear(x.shape[1], 10); opt = torch.optim.Adam(W.parameters(), lr=LR, weight_decay=L2)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(EPOCAS):
        opt.zero_grad(); Fn.cross_entropy(W(xt), yt).backward(); opt.step()
    return W


def acierto(xtr, ytr, xte, yte) -> list:
    return [r4((ajustar(xtr, ytr, s)(torch.from_numpy(xte)).argmax(1).numpy() == yte).mean()) for s in (1, 2, 3)]


def reparto(y: np.ndarray, tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """COPIA de feat-ind (nn/curva.py, vía feat-ind32): (orden de train ampliable, máscara de test de 717)."""
    rng = np.random.default_rng(SEMILLA_REPARTO)
    val = np.flatnonzero(~tr); reserva = []
    for c in range(10):
        reserva += list(rng.choice(val[y[val] == c], RESERVA_POR_CLASE, replace=False))
    reserva = np.array(reserva); test = ~tr.copy(); test[reserva] = False

    def intercalar(idx):
        por = [list(rng.permutation(idx[y[idx] == c])) for c in range(10)]
        out = []
        while any(por):
            for p in por:
                if p:
                    out.append(p.pop())
        return np.array(out)
    return np.concatenate([intercalar(np.flatnonzero(tr)), intercalar(reserva)]), test


def _digitos() -> dict:
    d = np.load(exigir_dataset(DIGITOS) / "datos.npz")
    w = d["origen"] == "windep"
    return {"x": d["imagenes"][:, None].astype(np.float32), "y": d["etiquetas"].astype(np.int64), "w": w,
            "train": d["particion"] == "train", "val": d["particion"] == "val"}


# ------------------------------------------------------------------------------------------------ una combinación
def evaluar(nombre: str, prepro: str, dg: dict | None = None) -> dict:
    """`nombre` puede juntar BANCOS con «+» (p. ej. `lineas+lineas-grueso`): sus mapas van juntos al compositor; la firma y
    la prueba gruesa son del primero."""
    t0 = time.time(); torch.set_num_threads(2)
    bancos = [banco(n) for n in nombre.split("+")]; b = bancos[0]; dg = dg or _digitos(); y, w = dg["y"], dg["w"]
    vers = {"limpio": dg["x"], "engrosar": transformar(dg["x"], "engrosar"), "adelgazar": transformar(dg["x"], "adelgazar")}
    vistas = prepro.split("+")
    s = {v: np.concatenate([mapas(bb, preparar(xv, bb["rep"], vi)) for bb in bancos for vi in vistas], 1) for v, xv in vers.items()}
    out = {"banco": nombre, "representacion": b["rep"], "preprocesado": prepro, "huellas": {bb["nombre"]: bb["huellas"] for bb in bancos}}
    # firma (de la PRIMERA vista)
    pres = s["limpio"][:, :13].reshape(len(y), 13, -1).max(2) >= b["umbrales"][None]
    out["firma"] = {str(c): {f: r4(pres[w & (y == c), j].mean()) for j, f in enumerate(b["familias"])} for c in range(10)}
    # compositor
    xm = s["limpio"].reshape(len(y), -1)
    yw, xw, trw = y[w], xm[w], dg["train"][w]
    out["compositor_180"] = {"por_semilla": acierto(xw[trw], yw[trw], xw[~trw], yw[~trw])}
    out["compositor_180"]["media"] = r4(np.mean(out["compositor_180"]["por_semilla"]))
    orden, test = reparto(yw, trw)
    out["curva_717"] = {str(n): r4(np.mean(acierto(xw[orden[:n]], yw[orden[:n]], xw[test], yw[test]))) for n in TAMANOS}
    # κ
    z = {v: zonas(sv) for v, sv in s.items()}
    tod = [M.kmeans(z["limpio"], K0, sem) for sem in SEMILLAS]
    i = int(np.argmin([q["inercia"] for q in tod])); C = tod[i]["centroides"]; lab0 = M.asignar(z["limpio"], C)
    out["kappa"] = {"semilla": SEMILLAS[i]}
    for t in ("engrosar", "adelgazar"):
        c = float((M.asignar(z[t], C) == lab0).mean())
        out["kappa"][t] = {"c": r4(c), "kappa": r4(M.kappa(c, lab0, K0))}
    # la prueba gruesa (sintética), con el MISMO preprocesado (no aplica a dos vistas: es por detector)
    # (ni a dos bancos, ni a `desinc`: desinclinar una feature sintética suelta convierte una recta-S en una recta-V)
    out["grueso"] = grueso(b, prepro) if len(vistas) == 1 and len(bancos) == 1 and "desinc" not in prepro else None
    out["segundos"] = round(time.time() - t0, 1)
    return out


@torch.no_grad()
def grueso(b: dict, prepro: str) -> dict:
    g = np.load(exigir_dataset(GRUESO) / "datos.npz")
    x = preparar(g["imagenes"][:, None].astype(np.float32), b["rep"], prepro)
    prin, sec, gr = g["principal"], g["secundaria"], g["grosor"]
    s = mapas(b, x).reshape(len(x), 13, -1).max(2)
    tramos = {"vistos": np.isin(gr, VISTOS), "no_vistos": np.isin(gr, NO_VISTOS)}
    por_f = {}
    for j, f in enumerate(b["familias"]):
        i = F.FAMILIAS.index(f)
        pos = prin == i
        cont = [F.FAMILIAS.index(c) for c in F.contenedoras(f)]
        neg = (prin != i) & (sec != i) & ~np.isin(prin, cont) & ~np.isin(sec, cont)
        h = s[:, j] >= b["umbrales"][j]
        fila = {}
        for tn, mt in tramos.items():
            tp, fn_, fp = int((h & pos & mt).sum()), int((~h & pos & mt).sum()), int((h & neg & mt).sum())
            pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn_)
            fila[tn] = {"n_pos": int((pos & mt).sum()), "recall": r4(rc), "fp": r4(h[neg & mt].mean()),
                        "f1": r4(2 * pr * rc / max(1e-9, pr + rc))}
        por_f[f] = fila
    val = [f for f, v in por_f.items() if v["vistos"]["n_pos"] >= 20 and v["no_vistos"]["n_pos"] >= 20]
    res = {"detectores": por_f, "con_los_dos_tramos": val}
    for tn in tramos:
        for k in ("f1", "recall", "fp"):
            res[f"{k}_{tn}"] = r4(np.mean([por_f[f][tn][k] for f in val]))
    return res


def guardar(o: dict) -> Path:
    COMBOS.mkdir(parents=True, exist_ok=True)
    p = COMBOS / f"{o['banco']}-{o['preprocesado']}.json"
    p.write_text(json.dumps(o, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def linea(o: dict) -> str:
    g = o["grueso"] or {k: float("nan") for k in ("f1_vistos", "f1_no_vistos", "fp_vistos", "fp_no_vistos")}
    f1 = o["firma"]["1"]
    return (f"{o['banco']:<10} {o['preprocesado']:<6} compositor {o['compositor_180']['media']:.4f} · curva "
            + "/".join(f"{o['curva_717'][str(n)]:.3f}" for n in TAMANOS)
            + f" · κ eng {o['kappa']['engrosar']['kappa']:.3f} adel {o['kappa']['adelgazar']['kappa']:.3f}"
            + f" · arcos en 1 {f1['arco-E']:.2f}/{f1['arco-W']:.2f} · grueso F1 {g['f1_vistos']:.3f}→{g['f1_no_vistos']:.3f}"
            + f" FP {g['fp_vistos']:.3f}→{g['fp_no_vistos']:.3f} [{o['segundos']} s]")


def main(argv: list[str]) -> int:
    if argv and argv[0] == "--todas":
        hechas = {p.stem for p in COMBOS.glob("*.json")}
        for nombre in ("lineas", "contorno", "signo", "lineas-grueso", "lineas+lineas-grueso"):
            for prepro in PREPROCESADOS:
                if f"{nombre}-{prepro}" not in hechas:
                    try:
                        o = evaluar(nombre, prepro); guardar(o); print(linea(o), flush=True)
                    except (SystemExit, FileNotFoundError) as e:
                        print(f"  {nombre}-{prepro}: no se puede todavía ({e})", flush=True)
        return 0
    o = evaluar(argv[0], argv[1]); guardar(o); print(linea(o))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
