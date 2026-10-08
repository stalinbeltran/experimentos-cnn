#!/usr/bin/env python3
"""Entrena UN detector de `feat-1lado` (entra la TINTA 32×32, la primera capa la pasa a 8 bordes de un lado; mapa 8×8):
una feature contra todas las demás. COPIA del de `feat-bor` (2026-10-07) con estos cambios, y sólo éstos:

  - `--brazo control|compartido` en vez de `--repr`: los datos entran tal cual y el pre-proceso es la capa FIJA del
    modelo (nn/modelo.py), que guarda su huella en `config.json`;
  - los forwards de evaluación y del umbral van A TROZOS (`modelo.logits_por_lotes`): con `compartido` el de 6000
    negativos de golpe serían 48.000 pasadas, ~10 GB por proceso (estimado por el arquitecto).

La RECETA —lotes 32 + 32, 80 épocas, lr 2·10⁻³, BCE por celda (objetivo ×8) + BCE del máximo, umbral elegido en train,
`best` por F1 de val, semilla 1— se conserva por DECISIÓN de este experimento (REGLAS.md § «Qué NO hereda»), no porque
venga en la copia: así la única diferencia con feat-bor y feat-ind32 es lo que ve el detector.

    python nn/entrenar_local.py --brazo compartido --feature arco-E [--semilla 1] [--epocas 80] [--hilos N]
    python nn/entrenar_local.py --comprobar [--brazo B --hilos N]   el mecanismo, y cuánto tarda UNA época
    python nn/entrenar_local.py --componer                          el compositor y la curva de desplazamiento (nn/componer.py)
    python nn/entrenar_local.py --lados                             compositores por lado, solos y gradual (nn/lados.py)

⚠ EL NOMBRE DEL FICHERO ES UN CONTRATO con el freno (`cerrable.mjs` -> TRABAJOS): por eso `componer`, que tarda, se
lanza a través de él.
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
def carpeta_pesos(brazo: str) -> Path:
    return AQUI / f"pesos-{brazo}"
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
    xp, xn = c["x_pos"], c["x_neg"]
    hp, pp, cp = modelo.leer(modelo.logits_por_lotes(red, xp)); hn, _, cn = modelo.leer(modelo.logits_por_lotes(red, xn))
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
    cp = modelo.leer(modelo.logits_por_lotes(red, c["x_pos"]))[2].numpy()
    cn = modelo.leer(modelo.logits_por_lotes(red, c["x_neg"]))[2].numpy()
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
    cp = modelo.leer(modelo.logits_por_lotes(red, c["x_pos"]))[2].numpy()
    idx = rng.choice(len(c["x_neg"]), size=min(n_neg, len(c["x_neg"])), replace=False)
    cn = modelo.leer(modelo.logits_por_lotes(red, c["x_neg"][idx]))[2].numpy()
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


def entrenar(d: dict, feature: str, semilla: int, epocas: int, brazo: str, raiz: Path | None = None,
             registro_on: bool = True, lr: float = LR, canales=modelo.CANALES, peso_objetivo: float = modelo.PESO_OBJETIVO,
             lote_neg: int = LOTE_NEG, peso_presencia: float = PESO_PRESENCIA) -> dict:
    raiz = raiz or carpeta_pesos(brazo)
    tr, va = datos.conjunto(d, feature, "train"), datos.conjunto(d, feature, "val")
    torch.manual_seed(semilla)
    red = modelo.Detector(brazo, canales)
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
              "brazo": brazo, "huella_kernels": modelo.HUELLA_KERNELS, "angulos": list(modelo.ANGULOS),
              "parametros": red.n_parametros(), "sigma": modelo.SIGMA, "peso_objetivo": peso_objetivo,
              "peso_presencia": peso_presencia, "umbral": "elegido sobre train por F1; ver best.umbral",
              "dataset": d.get("nombre", datos.DATASET), "negativos_excluidos_por_contener": list(F.contenedoras(feature)),
              "huella_imagenes": d["manifiesto"]["huellas"]["imagenes"]}
    print(f"empiezo {brazo}/{feature}: {len(xp)} pos / {len(xn)} neg, {epocas} épocas, init {huella_init}", flush=True)
    t0 = time.time(); mejor = None; pasos = 0; umbral = modelo.UMBRAL; s_epocas = []
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
        s_epocas.append(round(time.time() - te, 2))                       # sólo el entrenamiento de la época
        if ep % EVAL_CADA == 0 or ep == epocas:
            umbral = umbral_en_train(red, tr, np.random.default_rng(999 + semilla))
            m = evaluar(red, va, umbral); m["umbral"] = umbral
            fila.update({k: m[k] for k in ("precision", "recall", "f1", "pos_ok", "umbral")})
            if _mejor(mejor, m):
                mejor = {**m, "epoca": ep}; _guardar(dir_b / "best.pt", red, config, ep, m, umbral)
            # Lo que haya hasta aquí, por si el tope de horas corta antes del final: summary.json sólo existe al acabar.
            (dir_b / "progreso.json").write_text(json.dumps({"epoca": ep, "de": epocas, "best": mejor,
                                                              "segundos": round(time.time() - t0, 1)}, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
            print(f"  {feature} ep {ep:>3}/{epocas} perdida {fila['perdida']:.4f} · umbral {umbral:.2f} P {m['precision']:.3f} R {m['recall']:.3f} "
                  f"F1 {m['f1']:.3f} pos≤1 {m['pos_ok']:.3f}  [{time.time() - t0:.0f} s]", flush=True)
        fila["s"] = round(time.time() - te, 2)
        if registro_on:
            with registro.open("a", encoding="utf-8") as f:
                f.write(json.dumps(fila) + "\n")
    final = evaluar(red, va, umbral); final["umbral"] = umbral
    _guardar(dir_b / "last.pt", red, config, epocas, final, umbral)
    red_b, _ = modelo.cargar(dir_b / "best.pt")
    resumen = {**config, "segundos": round(time.time() - t0, 1), "s_epoca_mediana": float(np.median(s_epocas)),
               "last": final, "best": mejor,
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


def comprobar(brazos=modelo.BRAZOS) -> int:
    import tempfile
    ok = True

    def prueba(que, bien, det=""):
        nonlocal ok
        ok &= bool(bien); print(f"  [{'ok' if bien else 'FALLA':>5}] {que}" + (f"  {det}" if det else ""), flush=True)

    d = datos.cargar()
    c = datos.conjunto(d, "arco-E", "train")
    prueba("arco-E train = 2000 positivas de (1, 32, 32): la tinta, sin transformar",
           len(c["x_pos"]) == datos.N_TRAIN and c["x_pos"].shape[1:] == (1, 32, 32), str(c["x_pos"].shape))
    prueba("negativos de arco-E: sin lazo (lo contiene)", F.FAMILIAS.index("lazo") not in set(np.unique(c["familia_neg"])))
    cv_ = datos.conjunto(d, "recta-V", "train")
    prueba("negativos de recta-V: sin esquinas", not any(F.FAMILIAS.index(e) in set(np.unique(cv_["familia_neg"])) for e in F.ESQ_BRAZOS))
    prueba("el ancla 8×8 es la celda 4×4 del ancla en px", bool((d["ancla"][d["ancla"][:, 0] >= 0] == (d["ancla32"][d["ancla"][:, 0] >= 0] // 4).astype(np.int8)).all()))
    with tempfile.TemporaryDirectory() as tmp:
        for brazo in brazos:
            r1 = entrenar(d, "recta-V", 1, 1, brazo, Path(tmp) / brazo, registro_on=False)
            r2 = entrenar(d, "recta-V", 1, 1, brazo, Path(tmp) / f"{brazo}-b", registro_on=False)
            prueba(f"{brazo}: misma semilla, 1 época: pesos bit a bit iguales", r1["huella_final"] == r2["huella_final"])
            red, est = modelo.cargar(Path(tmp) / brazo / "recta-V" / "best.pt")
            prueba(f"{brazo}: best.pt carga con su brazo y sus kernels, y trae config, val y umbral",
                   red.brazo == brazo and est["config"]["huella_kernels"] == modelo.HUELLA_KERNELS and "f1" in est["val"]
                   and 0 < est["umbral"] < 1)
            prueba(f"{brazo}: 1 época tarda", True, f"{r1['s_epoca_mediana']} s de entrenamiento ({r1['segundos']} s con la "
                   f"evaluación) con {torch.get_num_threads()} hilo(s)")
    for m, v in (({"f1": 0.95, "pos_ok": 0.95}, "aprendió"), ({"f1": 0.95, "pos_ok": 0.5}, "a medias"), ({"f1": 0.5, "pos_ok": 1}, "no aprendió")):
        prueba(f"veredicto {m} = {v}", veredicto(m) == v)
    print("el mecanismo funciona." if ok else "✗ algo no funciona")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--feature", choices=F.CON_TRAZO)
    p.add_argument("--brazo", choices=modelo.BRAZOS, help="control: los 8 canales a la vez · compartido: uno cada vez, y máximo")
    p.add_argument("--semilla", type=int, default=1)
    p.add_argument("--epocas", type=int, default=EPOCAS_DEF)
    p.add_argument("--hilos", type=int)
    p.add_argument("--salida", type=Path, help="por defecto nn/pesos-<brazo>")
    p.add_argument("--comprobar", action="store_true")
    p.add_argument("--componer", action="store_true", help="nn/componer.py: el compositor y la curva de desplazamiento")
    p.add_argument("--lados", action="store_true", help="nn/lados.py: compositores por lado, solos y gradual (2026-10-08)")
    a, resto = p.parse_known_args()
    if a.hilos:
        torch.set_num_threads(a.hilos)
    if a.lados:
        import lados                                                    # noqa: PLC0415
        return lados.main()
    if a.componer:
        import componer                                                 # noqa: PLC0415
        return componer.main(resto)
    if resto:
        p.error(f"argumentos desconocidos: {resto}")
    if a.comprobar:
        return comprobar((a.brazo,) if a.brazo else modelo.BRAZOS)
    if not a.feature or not a.brazo:
        p.error("hace falta --brazo <control|compartido> y --feature <familia>, o --comprobar, o --componer")
    entrenar(datos.cargar(), a.feature, a.semilla, a.epocas, a.brazo, a.salida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
