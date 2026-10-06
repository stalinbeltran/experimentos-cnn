#!/usr/bin/env python3
"""Entrena UN detector de `feat-fallos` (entrada 32×32, mapa 8×8): una feature contra todas las demás, sobre el dataset
GRUESO (2–12 px) por defecto. COPIA del de `feat-bor` (2026-10-06), que lo es del de `feat-ind32`: la receta no cambia; aquí
cambian el dataset (`--dataset`) y, como allí, la representación (`--repr`). Los pesos van a nn/pesos-<repr>-<dataset>/. La
receta —lotes, épocas, lr, pérdida, umbral, best— es la de feat-ind32 sin tocar; los datos se guardan en uint8 y se pasan
a float en cada lote (los mismos valores 0/1: para no tener 4 copias float32 de 32.400 imágenes en memoria).

    python nn/entrenar_local.py --repr lineas --dataset grueso --feature arco-E [--semilla 1] [--epocas 80] [--hilos N]
    python nn/entrenar_local.py --comprobar      el mecanismo, sin entrenar de verdad

⚠ EL NOMBRE DEL FICHERO ES UN CONTRATO con el freno (`cerrable.mjs` -> TRABAJOS).

Lotes balanceados 32 + 32; una época = una pasada por las 2000 positivas (63 pasos). Pérdida: BCE por celda contra la
gaussiana del ancla (×8 en el objetivo) + BCE sobre el máximo del mapa. Umbral por detector elegido sobre TRAIN.
`best.pt` = mejor F1 de val (desempate: posición). Todo esto es lo de la corrida 2 de feat-ind, sin cambios.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bordes as B                             # noqa: E402
import datos                                   # noqa: E402
import features as F                           # noqa: E402
import modelo                                  # noqa: E402

AQUI = Path(__file__).resolve().parent
DATASETS = {"grueso": datos.GRUESO_ENTRENAR, "fino": datos.FINO}


def carpeta_pesos(representacion: str, dataset: str = "grueso") -> Path:
    return AQUI / f"pesos-{representacion}-{dataset}"
LOTE_POS = LOTE_NEG = 32
EPOCAS_DEF = 80          # corrida 1: 40; ENMIENDA corrida 2: el best de arco-E caía en la 65-70
EVAL_CADA = 5
LR = 2e-3
UMBRALES = tuple(round(u, 2) for u in np.arange(0.1, 0.95, 0.05))


def maquina() -> dict:
    cpu = ""
    try:
        for linea in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
            if linea.startswith("model name"):
                cpu = linea.split(":", 1)[1].strip(); break
    except OSError:
        pass
    return {"cpu": cpu, "nproc": os.cpu_count(), "hilos_torch": torch.get_num_threads(), "torch": torch.__version__,
            "numpy": np.__version__, "python": platform.python_version(), "host": platform.node()}


PESO_PRESENCIA = 1.0   # ENMIENDA corrida 2: BCE sobre el MÁXIMO del mapa (lo que se lee) + la BCE por celda


def perdida(logits, objetivo, peso_objetivo=modelo.PESO_OBJETIVO, peso_presencia=PESO_PRESENCIA):
    peso = 1.0 + (peso_objetivo - 1.0) * objetivo
    por_celda = Fn.binary_cross_entropy_with_logits(logits, objetivo, weight=peso)
    if peso_presencia <= 0:
        return por_celda
    esta = (objetivo.flatten(1).max(1).values > 0.5).float()          # ¿hay ancla en esta imagen?
    maximo = logits.flatten(1).max(1).values
    return por_celda + peso_presencia * Fn.binary_cross_entropy_with_logits(maximo, esta)


@torch.no_grad()
def evaluar(red, c: dict, umbral: float = modelo.UMBRAL) -> dict:
    """Precisión / recall / F1 de «hallada», posición ≤1 celda entre las halladas, y desgloses."""
    red.eval()
    xp, xn = torch.from_numpy(c["x_pos"]).float(), torch.from_numpy(c["x_neg"]).float()
    hp, pp, cp = modelo.leer(red(xp)); hn, _, cn = modelo.leer(red(xn))
    hp, hn = cp >= umbral, cn >= umbral
    tp, fn_, fp = int(hp.sum()), int((~hp).sum()), int(hn.sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn_)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    anc = torch.from_numpy(c["ancla_pos"])
    dist = (pp - anc).abs().max(1).values
    pos_ok = float((dist[hp] <= 1).float().mean()) if tp else 0.0
    pos_exacta = float((dist[hp] == 0).float().mean()) if tp else 0.0
    hp_np = hp.numpy()
    por_radio = {}
    for nombre, lo, hi in F.RADIO_TRAMOS:
        m = (c["radio_pos"] >= lo) & (c["radio_pos"] < hi)
        if m.any():
            por_radio[nombre] = {"n": int(m.sum()), "recall": round(float(hp_np[m].mean()), 4)}
    por_grosor = {str(g): {"n": int((c["grosor_pos"] == g).sum()), "recall": round(float(hp_np[c["grosor_pos"] == g].mean()), 4)}
                  for g in sorted(set(int(v) for v in c["grosor_pos"]))}
    fam_neg = c["familia_neg"]; hn_np = hn.numpy()
    fp_por_familia = {F.FAMILIAS[i]: {"n": int((fam_neg == i).sum()), "fp": int(hn_np[fam_neg == i].sum()),
                                      "tasa": round(float(hn_np[fam_neg == i].mean()), 4)}
                      for i in np.unique(fam_neg)}
    con_sec = c["secundaria_pos"] >= 0
    recall_con_secundaria = round(float(hp_np[con_sec].mean()), 4) if con_sec.any() else None
    recall_sola = round(float(hp_np[~con_sec].mean()), 4) if (~con_sec).any() else None
    return {"n_pos": len(xp), "n_neg": len(xn), "tp": tp, "fp": fp, "fn": fn_, "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "pos_ok": round(pos_ok, 4), "pos_exacta": round(pos_exacta, 4),
            "recall_por_radio": por_radio, "recall_por_grosor": por_grosor, "fp_por_familia": fp_por_familia,
            "recall_con_secundaria": recall_con_secundaria, "recall_sola": recall_sola,
            "conf_pos_media": round(float(cp.mean()), 4), "conf_neg_media": round(float(cn.mean()), 4)}


@torch.no_grad()
def curva_pr(red, c: dict) -> list[dict]:
    red.eval()
    cp = modelo.leer(red(torch.from_numpy(c["x_pos"]).float()))[2].numpy()
    cn = modelo.leer(red(torch.from_numpy(c["x_neg"]).float()))[2].numpy()
    out = []
    for u in UMBRALES:
        tp, fp, fn_ = int((cp >= u).sum()), int((cn >= u).sum()), int((cp < u).sum())
        p, r = tp / max(1, tp + fp), tp / max(1, tp + fn_)
        out.append({"umbral": u, "precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(1e-9, p + r), 4)})
    return out


@torch.no_grad()
def umbral_en_train(red, c: dict, rng: np.random.Generator, n_neg: int = 6000) -> float:
    """ENMIENDA corrida 2: el umbral que maximiza F1 sobre TRAIN (todas las positivas + n_neg negativas
    fijas). Val no interviene en elegirlo."""
    red.eval()
    cp = modelo.leer(red(torch.from_numpy(c["x_pos"]).float()))[2].numpy()
    idx = rng.choice(len(c["x_neg"]), size=min(n_neg, len(c["x_neg"])), replace=False)
    cn = modelo.leer(red(torch.from_numpy(c["x_neg"][idx]).float()))[2].numpy()
    mejor, mu = -1.0, modelo.UMBRAL
    for u in UMBRALES:
        tp, fp, fn_ = int((cp >= u).sum()), int((cn >= u).sum()), int((cp < u).sum())
        pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn_)
        f1 = 2 * pr * rc / max(1e-9, pr + rc)
        if f1 > mejor:
            mejor, mu = f1, float(u)
    return mu


def _guardar(destino: Path, red, config: dict, epoca: int, metricas: dict | None = None, umbral: float = modelo.UMBRAL) -> None:
    tmp = destino.with_suffix(".tmp")
    torch.save({"modelo": red.state_dict(), "config": config, "epoca": epoca, "val": metricas, "umbral": umbral}, tmp)
    tmp.replace(destino)


def _mejor(a: dict | None, b: dict) -> bool:
    return a is None or (b["f1"], b["pos_ok"]) > (a["f1"], a["pos_ok"])


def entrenar(d: dict, feature: str, semilla: int, epocas: int, representacion: str, raiz: Path | None = None,
             dataset: str = "grueso",
             registro_on: bool = True, lr: float = LR, canales=modelo.CANALES, peso_objetivo: float = modelo.PESO_OBJETIVO,
             lote_neg: int = LOTE_NEG, peso_presencia: float = PESO_PRESENCIA) -> dict:
    raiz = raiz or carpeta_pesos(representacion, dataset)
    tr, va = datos.conjunto(d, feature, "train", representacion), datos.conjunto(d, feature, "val", representacion)
    torch.manual_seed(semilla)
    red = modelo.Detector(canales, entrada=B.CANALES[representacion])
    huella_init = modelo.huella_pesos(red)
    opt = torch.optim.Adam(red.parameters(), lr=lr)
    rng = np.random.default_rng(100 + semilla)
    xp, xn = torch.from_numpy(tr["x_pos"]), torch.from_numpy(tr["x_neg"])
    tp_obj = modelo.objetivo(torch.from_numpy(tr["ancla_pos"]))
    tn_obj = torch.zeros(lote_neg, 1, modelo.LADO, modelo.LADO)
    dir_b = raiz / feature; dir_b.mkdir(parents=True, exist_ok=True)
    registro = dir_b / "metrics.jsonl"
    if registro_on:
        registro.write_text("", encoding="utf-8")
    config = {"feature": feature, "semilla": semilla, "epocas": epocas, "lote": [LOTE_POS, lote_neg], "lr": lr,
              "huella_init": huella_init, "n_train_pos": len(xp), "n_train_neg": len(xn), "n_val_pos": len(va["x_pos"]),
              "n_val_neg": len(va["x_neg"]), "canales": list(red.canales), "strides": list(red.strides),
              "representacion": representacion, "entrada": red.entrada, "dataset_corto": dataset,
              "parametros": red.n_parametros(), "sigma": modelo.SIGMA, "peso_objetivo": peso_objetivo,
              "peso_presencia": peso_presencia, "umbral": "elegido sobre train por F1; ver best.umbral",
              "dataset": d.get("nombre", datos.DATASET), "negativos_excluidos_por_contener": list(F.contenedoras(feature)),
              "huella_imagenes": d["manifiesto"]["huellas"]["imagenes"]}
    print(f"empiezo {representacion}/{feature}: {len(xp)} pos / {len(xn)} neg, {epocas} épocas, init {huella_init}", flush=True)
    t0 = time.time(); mejor = None; pasos = 0; umbral = modelo.UMBRAL
    for ep in range(1, epocas + 1):
        red.train(); te = time.time()
        perm = rng.permutation(len(xp)); suma, n = 0.0, 0
        for i in range(0, len(perm), LOTE_POS):
            ip = torch.from_numpy(perm[i:i + LOTE_POS])
            in_ = torch.from_numpy(rng.integers(0, len(xn), size=lote_neg))
            x = torch.cat([xp[ip], xn[in_]]).float(); obj = torch.cat([tp_obj[ip], tn_obj[:len(in_)]])
            opt.zero_grad(); l = perdida(red(x), obj, peso_objetivo, peso_presencia); l.backward(); opt.step()
            suma += l.item() * len(x); n += len(x); pasos += 1
        fila = {"epoca": ep, "paso": pasos, "perdida": round(suma / n, 6)}
        if ep % EVAL_CADA == 0 or ep == epocas:
            umbral = umbral_en_train(red, tr, np.random.default_rng(999 + semilla))
            m = evaluar(red, va, umbral); m["umbral"] = umbral
            fila.update({k: m[k] for k in ("precision", "recall", "f1", "pos_ok", "umbral")})
            if _mejor(mejor, m):
                mejor = {**m, "epoca": ep}; _guardar(dir_b / "best.pt", red, config, ep, m, umbral)
            print(f"  {feature} ep {ep:>3}/{epocas} perdida {fila['perdida']:.4f} · umbral {umbral:.2f} P {m['precision']:.3f} R {m['recall']:.3f} "
                  f"F1 {m['f1']:.3f} pos≤1 {m['pos_ok']:.3f}  [{time.time() - t0:.0f} s]", flush=True)
        fila["s"] = round(time.time() - te, 2)
        if registro_on:
            with registro.open("a", encoding="utf-8") as f:
                f.write(json.dumps(fila) + "\n")
    final = evaluar(red, va, umbral); final["umbral"] = umbral
    _guardar(dir_b / "last.pt", red, config, epocas, final, umbral)
    red_b, _ = modelo.cargar(dir_b / "best.pt")
    resumen = {**config, "segundos": round(time.time() - t0, 1), "last": final, "best": mejor,
               "curva_pr_best": curva_pr(red_b, va), "huella_final": modelo.huella_pesos(red),
               "veredicto": veredicto(mejor), "maquina": maquina(),
               "terminado": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}
    (dir_b / "config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (dir_b / "summary.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"listo {feature}: best ep {mejor['epoca']} umbral {mejor['umbral']:.2f} F1 {mejor['f1']:.4f} P {mejor['precision']:.4f} R {mejor['recall']:.4f} "
          f"pos≤1 {mejor['pos_ok']:.4f} → {resumen['veredicto']} · {resumen['segundos']} s", flush=True)
    return resumen


def veredicto(m: dict) -> str:
    """El de instrucciones/02-criterio.md §A."""
    if m["f1"] >= 0.90 and m["pos_ok"] >= 0.90:
        return "aprendió"
    if m["f1"] >= 0.75:
        return "a medias"
    return "no aprendió"


def comprobar() -> int:
    import tempfile
    ok = True

    def prueba(que, bien, det=""):
        nonlocal ok
        ok &= bool(bien); print(f"  [{'ok' if bien else 'FALLA':>5}] {que}" + (f"  {det}" if det else ""))

    d = datos.cargar()
    for rep in B.REPRESENTACIONES:
        c = datos.conjunto(d, "arco-E", "train", rep)
        prueba(f"{rep}: arco-E train = 2000 positivas de ({B.CANALES[rep]}, 32, 32)",
               len(c["x_pos"]) == datos.N_TRAIN and c["x_pos"].shape[1:] == (B.CANALES[rep], 32, 32), str(c["x_pos"].shape))
    prueba("negativos de arco-E: sin lazo (lo contiene)", F.FAMILIAS.index("lazo") not in set(np.unique(c["familia_neg"])))
    cv_ = datos.conjunto(d, "recta-V", "train", "lineas")
    prueba("negativos de recta-V: sin esquinas", not any(F.FAMILIAS.index(e) in set(np.unique(cv_["familia_neg"])) for e in F.ESQ_BRAZOS))
    prueba("el ancla 8×8 es la celda 4×4 del ancla en px", bool((d["ancla"][d["ancla"][:, 0] >= 0] == (d["ancla32"][d["ancla"][:, 0] >= 0] // 4).astype(np.int8)).all()))
    with tempfile.TemporaryDirectory() as tmp:
        for rep in ("lineas",):
            r1 = entrenar(d, "recta-V", 1, 1, rep, Path(tmp) / rep, registro_on=False)
            r2 = entrenar(d, "recta-V", 1, 1, rep, Path(tmp) / f"{rep}-b", registro_on=False)
            prueba(f"{rep}: misma semilla, 1 época: pesos bit a bit iguales", r1["huella_final"] == r2["huella_final"])
            red, est = modelo.cargar(Path(tmp) / rep / "recta-V" / "best.pt")
            prueba(f"{rep}: best.pt carga con su entrada ({B.CANALES[rep]}) y trae config, val y umbral",
                   red.entrada == B.CANALES[rep] and est["config"]["representacion"] == rep and "f1" in est["val"]
                   and 0 < est["umbral"] < 1)
            prueba(f"{rep}: 1 época tarda (s)", True, f"{r1['segundos']} s con {torch.get_num_threads()} hilo(s)")
    for m, v in (({"f1": 0.95, "pos_ok": 0.95}, "aprendió"), ({"f1": 0.95, "pos_ok": 0.5}, "a medias"), ({"f1": 0.5, "pos_ok": 1}, "no aprendió")):
        prueba(f"veredicto {m} = {v}", veredicto(m) == v)
    print("el mecanismo funciona." if ok else "✗ algo no funciona")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature", choices=F.CON_TRAZO)
    p.add_argument("--repr", choices=("lineas", "contorno", "signo"), default="lineas", help="qué ve el detector (nn/bordes.py)")
    p.add_argument("--dataset", choices=tuple(DATASETS), default="grueso")
    p.add_argument("--semilla", type=int, default=1)
    p.add_argument("--epocas", type=int, default=EPOCAS_DEF)
    p.add_argument("--hilos", type=int)
    p.add_argument("--salida", type=Path, help="por defecto nn/pesos-<repr>")
    p.add_argument("--comprobar", action="store_true")
    a = p.parse_args()
    if a.hilos:
        torch.set_num_threads(a.hilos)
    if a.comprobar:
        return comprobar()
    if not a.feature:
        p.error("hace falta --feature <familia>, o --comprobar")
    entrenar(datos.cargar(DATASETS[a.dataset]), a.feature, a.semilla, a.epocas, a.repr, a.salida, dataset=a.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
