#!/usr/bin/env python3
"""La EVALUACIÓN de `feat-cortas` (instrucciones/02-criterio.md), escrita antes de entrenar.

  §A    por detector, de su summary.json: F1, P, R, posición y veredicto
  H1-cv curva corta contra recta corta: falsos positivos de cada detector de curva sobre las rectas cortas y al revés (val)
  §B    predecir los dígitos: compositor posicional (8 mapas → 512) con 180/1617 y la curva 36/180/1080 sobre el test de 717,
        en `nada`, `norm3` y `nada+norm3`; y cortas (`norm3`) + las 13 largas de feat-ind32 (crudas) en un compositor

    python nn/evaluar.py      → resultados/evaluacion.json, resultados/cortas.png
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
import features as F                            # noqa: E402
import modelo                                   # noqa: E402
import normalizar as N                          # noqa: E402

RES = EXP / "resultados"
PESOS = AQUI / "pesos"
DIGITOS = "uci-optdigits-orig-32px-r20261005"
LARGAS = ("arco-E", "arco-W", "arco-N", "arco-S", "recta-V", "recta-H", "recta-S", "recta-B", "lazo",
          "esquina-NE", "esquina-NW", "esquina-SE", "esquina-SW")
EPOCAS, LR, L2 = 300, 1e-2, 1e-3                 # el compositor de feat-ind32 (C4), copiado
SEMILLA_REPARTO, RESERVA_POR_CLASE, TAMANOS = 2026, 90, (36, 180, 1080)


def r4(v) -> float:
    return round(float(v), 4)


def cargar_banco(carpeta: Path, familias, huellas: dict | None = None) -> list:
    reds = []
    for f in familias:
        red, _ = modelo.cargar(carpeta / f / "best.pt")
        if huellas is not None and modelo.huella_pesos(red) != huellas[f]:
            raise SystemExit(f"✗ {f}: la huella no es la de su origen. Me niego.")
        reds.append(red)
    return reds


@torch.no_grad()
def mapas(reds: list, x: np.ndarray, lote: int = 1024) -> np.ndarray:
    out = np.zeros((len(x), len(reds), 8, 8), np.float32)
    for i in range(0, len(x), lote):
        xt = torch.from_numpy(np.ascontiguousarray(x[i:i + lote], dtype=np.float32))
        for j, red in enumerate(reds):
            out[i:i + lote, j] = torch.sigmoid(red(xt))[:, 0].numpy()
    return out


def ajustar(x, y, sem):
    torch.manual_seed(sem)
    W = torch.nn.Linear(x.shape[1], 10); opt = torch.optim.Adam(W.parameters(), lr=LR, weight_decay=L2)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    for _ in range(EPOCAS):
        opt.zero_grad(); Fn.cross_entropy(W(xt), yt).backward(); opt.step()
    return W


def acierto(xtr, ytr, xte, yte) -> float:
    return r4(np.mean([(ajustar(xtr, ytr, s)(torch.from_numpy(xte)).argmax(1).numpy() == yte).mean() for s in (1, 2, 3)]))


def reparto(y, tr):
    """COPIA de feat-ind (nn/curva.py, vía feat-ind32)."""
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


def medir(x: np.ndarray, y: np.ndarray, tr: np.ndarray) -> dict:
    orden, test = reparto(y, tr)
    return {"compositor_180": acierto(x[tr], y[tr], x[~tr], y[~tr]),
            "curva_717": {str(n): acierto(x[orden[:n]], y[orden[:n]], x[test], y[test]) for n in TAMANOS}}


def main() -> int:
    t0 = time.time(); torch.set_num_threads(2)
    fams = list(F.CON_TRAZO)
    out = {"A": {}, "H1_cv": {}, "B": {}}
    # §A y H1-cv
    rectas, curvas = F.GRUPOS["rectas cortas"], F.GRUPOS["curvas cortas"]
    for f in fams:
        s = json.loads((PESOS / f / "summary.json").read_text(encoding="utf-8"))
        b = s["best"]
        out["A"][f] = {k: b[k] for k in ("f1", "precision", "recall", "pos_ok", "umbral")} | {"veredicto": s["veredicto"]}
        fp = b["fp_por_familia"]
        otras = rectas if f in curvas else curvas
        out["H1_cv"][f] = {"fp_sobre_el_otro_grupo": r4(np.mean([fp[o]["tasa"] for o in otras if o in fp])),
                           "por_familia": {o: fp[o]["tasa"] for o in otras if o in fp}}
    cv = {"curva_sobre_rectas": r4(np.mean([out["H1_cv"][f]["fp_sobre_el_otro_grupo"] for f in curvas])),
          "recta_sobre_curvas": r4(np.mean([out["H1_cv"][f]["fp_sobre_el_otro_grupo"] for f in rectas]))}
    out["H1_cv_medias"] = cv
    # §B
    d = np.load(exigir_dataset(DIGITOS) / "datos.npz")
    w = d["origen"] == "windep"
    x = d["imagenes"][w][:, None].astype(np.float32); y = d["etiquetas"][w].astype(np.int64); tr = d["particion"][w] == "train"
    cortas = cargar_banco(PESOS, fams)
    vistas = {"nada": x, "norm3": N.normalizar(x, 3).astype(np.float32)}
    m = {v: mapas(cortas, xv).reshape(len(x), -1) for v, xv in vistas.items()}
    for nombre, xm in {"cortas nada": m["nada"], "cortas norm3": m["norm3"],
                       "cortas nada+norm3": np.concatenate([m["nada"], m["norm3"]], 1)}.items():
        out["B"][nombre] = medir(xm, y, tr)
        print(f"  {nombre:<22} compositor {out['B'][nombre]['compositor_180']:.4f} · curva "
              + "/".join(f"{out['B'][nombre]['curva_717'][str(n)]:.3f}" for n in TAMANOS), flush=True)
    org = por_id("feat-ind32").carpeta
    huellas = json.loads((org / "resultados" / "firma-por-clase.json").read_text(encoding="utf-8"))["huellas_best"]
    largas = cargar_banco(org / "nn" / "pesos", LARGAS, huellas)
    ml = mapas(largas, x).reshape(len(x), -1)
    mln = mapas(largas, vistas["norm3"]).reshape(len(x), -1)
    for nombre, xm in {"13 largas nada (referencia)": ml, "13 largas norm3": mln, "13 largas nada+norm3": np.concatenate([ml, mln], 1),
                       "cortas norm3 + 13 largas nada": np.concatenate([m["norm3"], ml], 1)}.items():
        out["B"][nombre] = medir(xm, y, tr)
        print(f"  {nombre:<22} compositor {out['B'][nombre]['compositor_180']:.4f} · curva "
              + "/".join(f"{out['B'][nombre]['curva_717'][str(n)]:.3f}" for n in TAMANOS), flush=True)
    n_ok = sum(v["veredicto"] in ("aprendió", "a medias") for v in out["A"].values())
    B_ = {k: v["compositor_180"] for k, v in out["B"].items()}
    h2 = {v: {"cortas": B_[f"cortas {v}"], "largas": B_["13 largas nada (referencia)" if v == "nada" else f"13 largas {v}"]}
          for v in ("nada", "norm3", "nada+norm3")}
    for v in h2.values():
        v["veredicto"] = "gana" if v["cortas"] >= v["largas"] + 0.01 else ("empata" if v["cortas"] >= v["largas"] - 0.01 else "pierde")
    umbral_h3 = r4(B_["13 largas nada+norm3"] + 0.01)
    cri = {"H1": {"veredicto": "confirmada" if n_ok >= 6 else "refutada", "al_menos_a_medias": n_ok},
           "H1-cv": {"veredicto": "confirmada" if max(cv.values()) <= 0.10 else "refutada", **cv},
           "H2": {"veredicto": h2["norm3"]["veredicto"], "por_vista": h2},
           "H3": {"veredicto": "confirmada" if B_["cortas norm3 + 13 largas nada"] >= umbral_h3 else "refutada",
                  "acc": B_["cortas norm3 + 13 largas nada"], "umbral": umbral_h3}}
    out["criterio"] = cri; out["segundos"] = round(time.time() - t0, 1)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "evaluacion.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    figura(out)
    for h, v in cri.items():
        print(f"  {h:<6} {v['veredicto']}")
    return 0


ETIQUETAS = {"cortas nada": "8 cortas\ncrudas", "cortas norm3": "8 cortas\na 3 px", "cortas nada+norm3": "8 cortas\ncrudas + 3 px",
             "13 largas nada (referencia)": "13 largas\ncrudas (ref.)", "13 largas norm3": "13 largas\na 3 px",
             "13 largas nada+norm3": "13 largas\ncrudas + 3 px", "cortas norm3 + 13 largas nada": "8 cortas a 3 px\n+ 13 largas crudas"}


def figura(out: dict) -> None:
    import matplotlib                                                # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt                                  # noqa: PLC0415
    SUP, T1, T2, REJ = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    COL = ("#2a78d6", "#eb6834", "#1baf7a")
    nombres = list(out["B"]); tam = [str(n) for n in TAMANOS]
    fig, ax = plt.subplots(figsize=(9.6, 4.4), dpi=100, facecolor=SUP); ax.set_facecolor(SUP)
    ancho = 0.26
    for j, n in enumerate(tam):
        ax.bar(np.arange(len(nombres)) + (j - 1) * (ancho + 0.02), [out["B"][k]["curva_717"][n] for k in nombres], ancho,
               color=COL[j], label=f"{n} dígitos de train", zorder=3)
    ax.set_xticks(range(len(nombres))); ax.set_xticklabels([ETIQUETAS.get(k, k) for k in nombres], fontsize=8, color=T1)
    ax.set_ylim(0.6, 1.0); ax.set_ylabel("acierto en 717 dígitos nuevos", color=T2, fontsize=9)
    ax.tick_params(axis="y", colors=T2, labelsize=8); ax.tick_params(axis="x", length=0); ax.grid(axis="y", color=REJ, lw=1, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(REJ)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.13), labelcolor=T1)
    ax.set_title("Rectas y curvas cortas para predecir dígitos", color=T1, fontsize=11, loc="left", pad=28)
    fig.tight_layout(); fig.savefig(RES / "cortas.png", facecolor=SUP); plt.close(fig)


if __name__ == "__main__":
    if sys.argv[1:] == ["--figura"]:                                 # sólo redibuja, de resultados/evaluacion.json
        figura(json.loads((RES / "evaluacion.json").read_text(encoding="utf-8"))); raise SystemExit(0)
    raise SystemExit(main())
