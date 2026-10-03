#!/usr/bin/env python3
"""Entrena UN detector de `feat-ind`: una feature, contra todas las demás (o contra `--contra`).

    python nn/entrenar_local.py --feature arco-E [--semilla 1] [--epocas 40] [--hilos N]
    python nn/entrenar_local.py --feature arco-E --desde nn/pesos/arco-E/last.pt --contra esquina-SE,esquina-NE
                                                 RE-ENTRENA desde pesos guardados contra una lista de contra-casos
    python nn/entrenar_local.py --todas          los 13 detectores, en serie, con los defectos
    python nn/entrenar_local.py --comprobar      el mecanismo, sin entrenar de verdad

⚠ EL NOMBRE DEL FICHERO ES UN CONTRATO con el freno (`cerrable.mjs` -> TRABAJOS).

Lotes balanceados: 32 positivas + 32 negativas; una época = una pasada por las positivas (2000 →
62 pasos). Pérdida: BCE por celda contra la gaussiana del ancla, celdas del objetivo pesadas ×8
(modelo.PESO_OBJETIVO). `best.pt` = mejor F1 de val (desempate: posición). Se guarda también `last.pt`.
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
import datos                                   # noqa: E402
import features as F                           # noqa: E402
import modelo                                  # noqa: E402

AQUI = Path(__file__).resolve().parent
PESOS = AQUI / "pesos"
LOTE_POS = LOTE_NEG = 32
EPOCAS_DEF = 80          # corrida 1: 40; ENMIENDA corrida 2: el best de arco-E caía en la 65-70
EVAL_CADA = 5
LR = 2e-3
FRAC_ENFASIS = 0.5
LR_REENTRENO = 1e-3
EPOCAS_REENTRENO = 40
TOP_CONTRA = 3
PESOS_C3 = AQUI / "pesos-c3"
PESOS_C4 = AQUI / "pesos-c4"
REENTRENAR = ("arco-E", "arco-W", "arco-N", "arco-S", "esquina-NE", "esquina-NW", "esquina-SE", "esquina-SW")
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
    xp, xn = torch.from_numpy(c["x_pos"]), torch.from_numpy(c["x_neg"])
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
    cp = modelo.leer(red(torch.from_numpy(c["x_pos"])))[2].numpy(); cn = modelo.leer(red(torch.from_numpy(c["x_neg"])))[2].numpy()
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
    cp = modelo.leer(red(torch.from_numpy(c["x_pos"])))[2].numpy()
    idx = rng.choice(len(c["x_neg"]), size=min(n_neg, len(c["x_neg"])), replace=False)
    cn = modelo.leer(red(torch.from_numpy(c["x_neg"][idx])))[2].numpy()
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


def entrenar(d: dict, feature: str, semilla: int, epocas: int, desde: Path | None, contra: list[str] | None,
             raiz: Path = PESOS, registro_on: bool = True, lr: float = LR, canales=modelo.CANALES,
             peso_objetivo: float = modelo.PESO_OBJETIVO, lote_neg: int = LOTE_NEG, peso_presencia: float = PESO_PRESENCIA,
             enfatizar: list[str] | None = None, frac_enfasis: float = FRAC_ENFASIS) -> dict:
    tr, va = datos.conjunto(d, feature, "train", contra), datos.conjunto(d, feature, "val", contra)
    torch.manual_seed(semilla)
    red = modelo.Detector(canales)
    huella_desde = None
    if desde is not None:
        red, est = modelo.cargar(desde); huella_desde = modelo.huella_pesos(red)
    huella_init = modelo.huella_pesos(red)
    opt = torch.optim.Adam(red.parameters(), lr=lr)
    rng = np.random.default_rng(100 + semilla)
    xp, xn = torch.from_numpy(tr["x_pos"]), torch.from_numpy(tr["x_neg"])
    tp_obj = modelo.objetivo(torch.from_numpy(tr["ancla_pos"]))
    tn_obj = torch.zeros(lote_neg, 1, modelo.LADO, modelo.LADO)
    # --enfatizar: una fracción de las negativas de cada lote sale de los contra-casos; el resto, de TODAS
    # (a diferencia de --contra, no quita ningún negativo: re-entrenar no puede olvidar lo que ya sabía)
    idx_enf = np.flatnonzero(np.isin(tr["familia_neg"], [F.FAMILIAS.index(e) for e in enfatizar])) if enfatizar else np.array([], int)
    if enfatizar and len(idx_enf) == 0:
        raise SystemExit(f"✗ --enfatizar {enfatizar}: ninguna negativa de train es de esas familias")
    n_enf = int(round(frac_enfasis * lote_neg)) if len(idx_enf) else 0
    dir_b = raiz / feature; dir_b.mkdir(parents=True, exist_ok=True)
    registro = dir_b / "metrics.jsonl"
    if registro_on:
        registro.write_text("", encoding="utf-8")
    config = {"feature": feature, "semilla": semilla, "epocas": epocas, "lote": [LOTE_POS, lote_neg], "lr": lr,
              "contra": contra or "todas las demás + vacio", "enfatizar": enfatizar, "frac_enfasis": frac_enfasis if enfatizar else None, "desde": str(desde) if desde else None, "huella_desde": huella_desde,
              "huella_init": huella_init, "n_train_pos": len(xp), "n_train_neg": len(xn), "n_val_pos": len(va["x_pos"]),
              "n_val_neg": len(va["x_neg"]), "canales": list(red.canales), "parametros": red.n_parametros(), "sigma": modelo.SIGMA,
              "peso_objetivo": peso_objetivo, "peso_presencia": peso_presencia, "umbral": "elegido sobre train por F1 (enmienda corrida 2); ver best.umbral", "dataset": d.get("nombre", datos.DATASET),
              "negativos_excluidos_por_contener": list(F.contenedoras(feature)) if not contra else [],
              "huella_imagenes": d["manifiesto"]["huellas"]["imagenes"]}
    print(f"empiezo {feature}: {len(xp)} pos / {len(xn)} neg, {epocas} épocas, init {huella_init}"
          + (f" (desde {desde})" if desde else "") + f", contra {config['contra']}", flush=True)
    t0 = time.time(); mejor = None; pasos = 0; umbral = modelo.UMBRAL
    rng_umbral = np.random.default_rng(999 + semilla)
    for ep in range(1, epocas + 1):
        red.train(); te = time.time()
        perm = rng.permutation(len(xp)); suma, n = 0.0, 0
        for i in range(0, len(perm), LOTE_POS):
            ip = torch.from_numpy(perm[i:i + LOTE_POS])
            if n_enf:
                in_ = torch.from_numpy(np.concatenate([rng.choice(idx_enf, size=n_enf),
                                                       rng.integers(0, len(xn), size=lote_neg - n_enf)]))
            else:
                in_ = torch.from_numpy(rng.integers(0, len(xn), size=lote_neg))
            x = torch.cat([xp[ip], xn[in_]]); obj = torch.cat([tp_obj[ip], tn_obj[:len(in_)]])
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
                  f"F1 {m['f1']:.3f} pos≤1 {m['pos_ok']:.3f}", flush=True)
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


def contra_casos(feature: str, origen: Path = PESOS, n: int = TOP_CONTRA) -> list[str]:
    """Las n familias con MÁS FALSOS POSITIVOS (en número) del best.pt de `origen` — leídas de su summary.json,
    o sea del val de la corrida anterior. Es la regla declarada en 02-criterio.md (corrida 3)."""
    fp = json.loads((origen / feature / "summary.json").read_text(encoding="utf-8"))["best"]["fp_por_familia"]
    return [k for k, _ in sorted(fp.items(), key=lambda kv: (-kv[1]["fp"], kv[0]))[:n]]


def reentrenar_todos(d: dict, semilla: int, destino: Path = PESOS_C3) -> int:
    for f in REENTRENAR:
        entrenar(d, f, semilla, EPOCAS_REENTRENO, PESOS / f / "last.pt", None, destino, lr=LR_REENTRENO,
                 canales=modelo.CANALES, enfatizar=contra_casos(f))
    return 0


def reentrenar_grueso(semilla: int, destino: Path = PESOS_C4) -> int:
    """Corrida 4: los 13, desde nn/pesos/<f>/last.pt (corrida 2), sobre el dataset de trazos GRUESOS."""
    d = datos.cargar(datos.DATASET_GRUESO)
    for f in F.CON_TRAZO:
        entrenar(d, f, semilla, EPOCAS_REENTRENO, PESOS / f / "last.pt", None, destino, lr=LR_REENTRENO, canales=modelo.CANALES)
    return 0


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
    c = datos.conjunto(d, "arco-E", "train")
    prueba("arco-E train: 2000 positivas", len(c["x_pos"]) == datos.N_TRAIN, str(len(c["x_pos"])))
    prueba("negativos de arco-E: sin lazo (lo contiene) ni como principal ni como secundaria",
           F.FAMILIAS.index("lazo") not in set(np.unique(c["familia_neg"])))
    cl = datos.conjunto(d, "lazo", "train")
    prueba("negativos de lazo: SÍ llevan arcos (un arco no contiene un lazo)", F.FAMILIAS.index("arco-E") in set(np.unique(cl["familia_neg"])))
    cv_ = datos.conjunto(d, "recta-V", "train")
    prueba("negativos de recta-V: sin esquinas", not any(F.FAMILIAS.index(e) in set(np.unique(cv_["familia_neg"])) for e in F.ESQ_BRAZOS))
    cx = datos.conjunto(d, "arco-E", "train", ["lazo"])
    prueba("--contra lazo manda sobre CONTIENE (entra como negativo si se pide)", set(np.unique(cx["familia_neg"])) == {F.FAMILIAS.index("lazo")})
    cc = datos.conjunto(d, "arco-E", "train", ["lazo", "vacio"])
    prueba("--contra lazo,vacio restringe los negativos", set(np.unique(cc["familia_neg"])) == {F.FAMILIAS.index("lazo"), F.FAMILIAS.index("vacio")})
    cv = datos.conjunto(d, "vacio", "val")
    prueba("vacio no tiene secundaria", (d["secundaria"][d["principal"] == F.FAMILIAS.index("vacio")] == -1).all())
    t = modelo.objetivo(torch.tensor([[0, 0], [7, 7], [-1, -1]]))
    prueba("objetivo: pico 1 en el ancla, 0 en negativos", abs(float(t[0, 0, 0, 0]) - 1) < 1e-6 and float(t[2].sum()) == 0)
    h, pos, conf = modelo.leer(torch.full((1, 1, 8, 8), -5.0).index_put_((torch.tensor([0]), torch.tensor([0]), torch.tensor([2]), torch.tensor([5])), torch.tensor([3.0])))
    prueba("leer: halla el máximo y su celda", bool(h[0]) and pos[0].tolist() == [2, 5])
    with tempfile.TemporaryDirectory() as tmp:
        r1 = entrenar(d, "recta-V", 1, 1, None, None, Path(tmp), registro_on=False)
        r2 = entrenar(d, "recta-V", 1, 1, None, None, Path(tmp) / "b", registro_on=False)
        prueba("misma semilla, 1 época: pesos bit a bit iguales", r1["huella_final"] == r2["huella_final"])
        r3 = entrenar(d, "recta-V", 1, 1, Path(tmp) / "recta-V" / "last.pt", None, Path(tmp) / "c", registro_on=False)
        prueba("--desde arranca de los pesos guardados", r3["huella_desde"] == r1["huella_final"] and r3["huella_final"] != r1["huella_final"])
        red, est = modelo.cargar(Path(tmp) / "recta-V" / "best.pt")
        prueba("best.pt carga y trae config, val y umbral", est["config"]["feature"] == "recta-V" and "f1" in est["val"] and 0 < est["umbral"] < 1)
        r4 = entrenar(d, "recta-V", 1, 1, None, ["lazo"], Path(tmp) / "d", registro_on=False)
        prueba("--contra queda en config", r4["contra"] == ["lazo"])
        r5 = entrenar(d, "recta-V", 1, 1, None, None, Path(tmp) / "e", registro_on=False, enfatizar=["lazo"])
        prueba("--enfatizar: no quita negativos y queda en config", r5["n_train_neg"] == r1["n_train_neg"] and r5["enfatizar"] == ["lazo"]
               and r5["huella_final"] != r1["huella_final"])
    for m, v in (({"f1": 0.95, "pos_ok": 0.95}, "aprendió"), ({"f1": 0.95, "pos_ok": 0.5}, "a medias"), ({"f1": 0.5, "pos_ok": 1}, "no aprendió")):
        prueba(f"veredicto {m} = {v}", veredicto(m) == v)
    print("el mecanismo funciona." if ok else "✗ algo no funciona")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature", choices=F.CON_TRAZO)
    p.add_argument("--todas", action="store_true")
    p.add_argument("--semilla", type=int, default=1)
    p.add_argument("--epocas", type=int, default=EPOCAS_DEF)
    p.add_argument("--desde", type=Path, help="pesos de partida (re-entrenar)")
    p.add_argument("--enfatizar", help="familias negativas a REFORZAR (la mitad de cada lote), sin quitar las demás")
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--reentrenar-grueso", action="store_true",
                   help="corrida 4: los 13 desde nn/pesos/<f>/last.pt sobre el dataset grueso → nn/pesos-c4/")
    p.add_argument("--reentrenar-todos", action="store_true",
                   help=f"corrida 3: {len(REENTRENAR)} detectores desde nn/pesos/<f>/last.pt, enfatizando sus {TOP_CONTRA} familias con más FP → nn/pesos-c3/")
    p.add_argument("--contra", help="familias negativas, separadas por comas (por defecto todas las demás + vacio)")
    p.add_argument("--hilos", type=int)
    p.add_argument("--salida", type=Path, default=PESOS)
    p.add_argument("--peso-objetivo", type=float, default=modelo.PESO_OBJETIVO, help="peso de las celdas del objetivo en la BCE")
    p.add_argument("--lote-neg", type=int, default=LOTE_NEG, help="negativas por lote (positivas: 32)")
    p.add_argument("--peso-presencia", type=float, default=PESO_PRESENCIA, help="peso de la BCE sobre el máximo del mapa; 0 la quita")
    p.add_argument("--canales", help="p. ej. 16,32,32 (por defecto los de modelo.CANALES); queda en config.json")
    p.add_argument("--comprobar", action="store_true")
    a = p.parse_args()
    if a.hilos:
        torch.set_num_threads(a.hilos)
    if a.comprobar:
        return comprobar()
    contra = a.contra.split(",") if a.contra else None
    if contra:
        malas = [c for c in contra if c not in F.FAMILIAS]
        if malas:
            p.error(f"familias desconocidas en --contra: {malas}; las de aquí son {F.FAMILIAS}")
    canales = tuple(int(c) for c in a.canales.split(",")) if a.canales else modelo.CANALES
    enf = a.enfatizar.split(",") if a.enfatizar else None
    for lista in (enf or []):
        if lista not in F.FAMILIAS:
            p.error(f"familia desconocida en --enfatizar: {lista}")
    if a.reentrenar_grueso:
        return reentrenar_grueso(a.semilla)
    d = datos.cargar()
    if a.reentrenar_todos:
        return reentrenar_todos(d, a.semilla)
    if a.todas:
        if a.desde:
            p.error("--todas no admite --desde: cada detector parte de sus propios pesos")
        for f in F.CON_TRAZO:
            entrenar(d, f, a.semilla, a.epocas, None, contra, a.salida, canales=canales, peso_objetivo=a.peso_objetivo, lote_neg=a.lote_neg, peso_presencia=a.peso_presencia)
        return 0
    if not a.feature:
        p.error("hace falta --feature <familia>, o --todas, o --comprobar")
    entrenar(d, a.feature, a.semilla, a.epocas, a.desde, contra, a.salida, lr=a.lr, canales=canales, peso_objetivo=a.peso_objetivo,
             lote_neg=a.lote_neg, peso_presencia=a.peso_presencia, enfatizar=enf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
