#!/usr/bin/env python3
"""Entrena UNA corrida de `ruido-nist`: un escenario × una semilla. 3996 pasos de lote 20 sobre las
360 imágenes de train del escenario (180 originales + 180 copias; 18 pasos por época, 222 épocas), y
mide cada 10 épocas y al final la exactitud y la entropía cruzada sobre las 1617 de val LIMPIAS y
sobre las 180 de train LIMPIAS. Sin selección: cuenta `last.pt`.

    python nn/entrenar_local.py --escenario horizontal@0.6 --semilla 2 [--pasos N] [--hilos N]
    python nn/entrenar_local.py --ensayo --lr 3e-3 [--escenario limpio] [--pasos N]   solo perdida de train
    python nn/entrenar_local.py --comprobar      el mecanismo, incluido «ruido nulo == limpio bit a bit»
    python nn/entrenar_local.py --inicializar    los pesos iniciales compartidos (nn/init/)

⚠ EL NOMBRE DEL FICHERO ES UN CONTRATO con el freno (`cerrable.mjs` -> TRABAJOS): cada corrida es UN
PROCESO con este nombre en la línea de comando.

Tres generadores, separados a propósito (ESPECIFICACION.md §1 bis): los pesos iniciales vienen del
FICHERO nn/init/init-s<s>.pt (compartido por todos los escenarios de la semilla); el orden de los
lotes de `numpy.default_rng(100 + s)` (idéntico en todos los escenarios); la copia ruidosa de su
propia semilla 1000 + 10·t + i (idéntica en las tres semillas de pesos). Lo ÚNICO que cambia entre
dos escenarios de la misma semilla es la copia.
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
import torch.nn.functional as F_

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos                                   # noqa: E402
import modelo                                  # noqa: E402
import ruido                                   # noqa: E402

AQUI = Path(__file__).resolve().parent
PESOS = AQUI / "pesos"
ENSAYO = AQUI / "ensayo"

N_TRAIN = 2 * datos.N_TRAIN       # 360: originales + copia
LOTE = 20                         # 18 pasos por época
PASOS_POR_EPOCA = N_TRAIN // LOTE
PASOS_DEF = 3996                  # 222 épocas × 18 pasos: los mismos pasos que dim-nist, sobre el doble de imágenes
EVAL_CADA = 10
SEMILLA_LOTES = 100               # + s
# CONGELADO el 2026-10-02 tras el ensayo sobre `limpio` (sólo pérdida de train, 3996 pasos, semilla 1): con 3e-3 la
# pérdida (360) va de 0,934 a 0,0005 con 10 subidas >20 % en 221 épocas (acc_train 1,0; CE de las 180 limpias
# 0,0004); con 1e-3 de 1,461 a 0,0276 (3 subidas; acc 0,994). Las dos sirven; se elige 3e-3, la que llega a la meseta.
LR = 3e-3


def ident(escenario: str, semilla: int) -> str:
    return f"{escenario}-s{semilla}"


def parsear_ident(nombre: str) -> tuple[str, int]:
    """'oblicua@0.6-r2-s3' -> ('oblicua@0.6-r2', 3)."""
    base, sep, s = nombre.rpartition("-s")
    if not sep or not s.isdigit() or int(s) not in modelo.SEMILLAS:
        raise ValueError(f"'{nombre}': la forma es <escenario>-s<semilla>, semilla en {modelo.SEMILLAS}")
    ruido.parsear(base)
    return base, int(s)


@torch.no_grad()
def evaluar(red, x, y) -> tuple[float, np.ndarray, float]:
    """(exactitud, aciertos por imagen, entropía cruzada media)."""
    red.eval()
    out = red(x)
    aciertos = (out.argmax(1) == y).numpy()
    return float(aciertos.mean()), aciertos, float(F_.cross_entropy(out, y))


def maquina() -> dict:
    cpu = ""
    try:
        for linea in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
            if linea.startswith("model name"):
                cpu = linea.split(":", 1)[1].strip(); break
    except OSError:
        pass
    return {"cpu": cpu, "nproc": os.cpu_count(), "hilos_torch": torch.get_num_threads(),
            "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
            "host": platform.node()}


def por_clase(aciertos: np.ndarray, y: np.ndarray) -> dict:
    return {str(c): {"n": int((y == c).sum()), "acc": round(float(aciertos[y == c].mean()), 5)} for c in range(10)}


def _guardar(destino: Path, red, config: dict, epoca: int) -> None:
    tmp = destino.with_suffix(".tmp")
    torch.save({"modelo": red.state_dict(), "config": config, "epoca": epoca}, tmp)
    tmp.replace(destino)


def _bucle(red, opt, xtr, ytr, rng, epocas, registro, nombre, xva=None, yva=None, xli=None, yli=None,
           guardar_en=None, config=None, regenerar=None) -> dict:
    paso, ultimo = 0, {}
    for ep in range(1, epocas + 1):
        t0 = time.time()
        if regenerar is not None and ep > 1:
            # ruido EN LÍNEA: la copia (índices 180–359) es nueva en cada época; la época 1 es la fija
            xtr[datos.N_TRAIN:] = torch.from_numpy(regenerar())
        red.train()
        perm = rng.permutation(len(xtr))
        suma, n, ok = 0.0, 0, 0
        for i in range(0, len(perm), LOTE):
            idx = torch.from_numpy(perm[i:i + LOTE])
            opt.zero_grad()
            out = red(xtr[idx])
            perdida = F_.cross_entropy(out, ytr[idx])
            perdida.backward()
            opt.step()
            paso += 1
            suma += perdida.item() * len(idx); n += len(idx)
            ok += int((out.detach().argmax(1) == ytr[idx]).sum())
        fila = {"epoca": ep, "paso": paso, "perdida_train360": round(suma / n, 6), "acc_train360_paso": round(ok / n, 5)}
        evaluo = xva is not None and (ep % EVAL_CADA == 0 or ep == epocas)
        if evaluo:
            acc_va, _, ce_va = evaluar(red, xva, yva)
            acc_li, _, ce_li = evaluar(red, xli, yli)
            fila.update({"acc_train": round(acc_li, 5), "acc_val": round(acc_va, 5), "brecha": round(acc_li - acc_va, 5),
                         "ce_train": round(ce_li, 5), "ce_val": round(ce_va, 5), "brecha_ce": round(ce_va - ce_li, 5)})
            if guardar_en is not None:
                _guardar(guardar_en, red, config or {}, ep)
        fila["s"] = round(time.time() - t0, 3)
        if registro is not None:
            with registro.open("a", encoding="utf-8") as f:
                f.write(json.dumps(fila) + "\n")
        if ep % 50 == 0 or ep == 1 or ep == epocas:
            extra = f" · acc_val {fila['acc_val']:.4f} brecha {fila['brecha']:+.4f}" if evaluo else ""
            print(f"  {nombre} ep {ep:>3}/{epocas} perdida {fila['perdida_train360']:.4f} acc~{fila['acc_train360_paso']:.3f}{extra}", flush=True)
        ultimo = fila
    return ultimo


def _preparar(escenario: str, semilla: int):
    d = datos.escenario(escenario)
    t = {k: torch.from_numpy(d[k]) for k in ("x_train", "y_train", "x_val", "y_val", "x_train_limpio", "y_train_limpio")}
    red = modelo.cargar_inicial(semilla)
    return d, t, red


def entrenar(escenario: str, semilla: int, pasos: int, lr, raiz: Path = PESOS) -> int:
    if lr is None:
        raise SystemExit("✗ LR no está congelado en nn/entrenar_local.py. Corre `--ensayo --lr …`, escribe el número aquí y en REGLAS.md, y entrena.")
    if pasos % PASOS_POR_EPOCA:
        raise SystemExit(f"✗ {pasos} pasos no son épocas enteras de {PASOS_POR_EPOCA} pasos ({N_TRAIN} imágenes / lote {LOTE})")
    nombre = ident(escenario, semilla)
    epocas = pasos // PASOS_POR_EPOCA
    d, t, red = _preparar(escenario, semilla)
    huella_init = modelo.huella_pesos(red)
    opt = torch.optim.Adam(red.parameters(), lr=lr)
    rng = np.random.default_rng(SEMILLA_LOTES + semilla)
    dir_b = raiz / nombre
    dir_b.mkdir(parents=True, exist_ok=True)
    registro = dir_b / "metrics.jsonl"
    registro.write_text("", encoding="utf-8")
    config = {"id": nombre, "escenario": escenario, "tipo": d["tipo"], "nivel": d["nivel"], "linea": d["linea"], "variante": d["variante"],
              "indice_nivel": None if d["tipo"] == ruido.LIMPIO else ruido.indice_nivel(d["tipo"], d["nivel"]),
              "realizacion": d["realizacion"], "semilla": semilla, "semilla_lotes": SEMILLA_LOTES + semilla,
              "semilla_ruido": d["semilla_ruido"], "pasos": pasos, "epocas": epocas, "lote": LOTE, "lr": lr,
              "L": modelo.L, "k": modelo.K, "C": modelo.C, "mapas": red.lados, "parametros": red.n_parametros(),
              "dataset": d["dataset"], "huella_init": huella_init, "huella_copia": d["huella_copia"],
              "huella_x_val": d["huella_x_val"], "huella_x_train_limpio": d["huella_x_train_limpio"],
              "tinta_media_original": round(d["tinta_media_original"], 5), "tinta_media_copia": round(d["tinta_media_copia"], 5)}
    print(f"empiezo {nombre}: {'EN LÍNEA (una copia nueva por época), época 1 = ' if d['linea'] else 'copia '}{d['huella_copia']} (semilla de ruido {d['semilla_ruido']}), init {huella_init}, "
          f"{red.n_parametros()} parámetros, {epocas} épocas, lr {lr}, hilos {torch.get_num_threads()}", flush=True)
    t0 = time.time()
    _bucle(red, opt, t["x_train"], t["y_train"], rng, epocas, registro, nombre, t["x_val"], t["y_val"],
           t["x_train_limpio"], t["y_train_limpio"], dir_b / "last.pt", config, d["regenerar"])
    acc_li, _, ce_li = evaluar(red, t["x_train_limpio"], t["y_train_limpio"])
    acc_360, _, ce_360 = evaluar(red, t["x_train"], t["y_train"])
    acc_va, ac_va, ce_va = evaluar(red, t["x_val"], t["y_val"])
    _guardar(dir_b / "last.pt", red, config, epocas)
    resumen = {**config, "acc_train": round(acc_li, 5), "acc_val": round(acc_va, 5), "brecha": round(acc_li - acc_va, 5),
               "ce_train": round(ce_li, 5), "ce_val": round(ce_va, 5), "brecha_ce": round(ce_va - ce_li, 5),
               "acc_train360": round(acc_360, 5), "ce_train360": round(ce_360, 5),
               "huella_final": modelo.huella_pesos(red), "piso": datos.piso(),
               "acc_val_por_clase": por_clase(ac_va, d["y_val"]),
               "segundos": round(time.time() - t0, 1), "maquina": maquina(),
               "terminado": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}
    (dir_b / "summary.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"listo {nombre}: acc_train {acc_li:.4f} acc_val {acc_va:.4f} brecha {acc_li - acc_va:+.4f} · "
          f"ce_train {ce_li:.4f} ce_val {ce_va:.4f} en {resumen['segundos']} s", flush=True)
    return 0


def ensayo(escenario: str, lr: float, pasos: int) -> int:
    """SÓLO la pérdida de train, para elegir el lr. No mira val."""
    epocas = max(1, pasos // PASOS_POR_EPOCA)
    d, t, red = _preparar(escenario, 1)
    opt = torch.optim.Adam(red.parameters(), lr=lr)
    ENSAYO.mkdir(parents=True, exist_ok=True)
    registro = ENSAYO / f"{escenario}-lr{lr:g}.jsonl"
    registro.write_text("", encoding="utf-8")
    print(f"ensayo {escenario} lr={lr:g}: {epocas} épocas ({red.n_parametros()} parámetros), SIN mirar val")
    _bucle(red, opt, t["x_train"], t["y_train"], np.random.default_rng(SEMILLA_LOTES + 1), epocas, registro, f"ensayo-lr{lr:g}")
    filas = [json.loads(l) for l in registro.read_text(encoding="utf-8").splitlines()]
    p = np.array([f["perdida_train360"] for f in filas]); acc = np.array([f["acc_train360_paso"] for f in filas])
    subidas = int((p[1:] > 1.2 * p[:-1]).sum())
    tramo = max(1, len(p) // 10)
    ini, fin = float(p[:tramo].mean()), float(p[-tramo:].mean())
    veredicto = ("baja sin oscilar" if fin < 0.5 * ini and subidas <= max(1, len(p) // 10)
                 else "NO sirve: " + ("no baja a la mitad" if fin >= 0.5 * ini else "oscila"))
    acc_li, _, ce_li = evaluar(red, t["x_train_limpio"], t["y_train_limpio"])
    print(f"  pérdida (360): inicio {ini:.4f} -> final {fin:.4f} (mín {p.min():.4f}); acc_train360 final {acc[-tramo:].mean():.3f}; "
          f"180 limpias: acc {acc_li:.3f} ce {ce_li:.4f}; subidas >20 %: {subidas} de {len(p) - 1} épocas  →  {veredicto}")
    return 0


def comprobar() -> int:
    """El mecanismo, sin entrenar de verdad: 2 épocas. Y la prueba de la ESPECIFICACION (§1): un escenario
    con ruido NULO da los mismos pesos finales que `limpio`, BIT A BIT; y uno con ruido, otros."""
    ok = True

    def prueba(que, bien, det=""):
        nonlocal ok
        ok &= bool(bien)
        print(f"  [{'ok' if bien else 'FALLA':>5}] {que}" + (f"  {det}" if det else ""))

    def corta(x, y, semilla, epocas=2, regenerar=None):
        red = modelo.cargar_inicial(semilla)
        opt = torch.optim.Adam(red.parameters(), lr=1e-3)
        _bucle(red, opt, torch.from_numpy(x), torch.from_numpy(y), np.random.default_rng(SEMILLA_LOTES + semilla), epocas, None, "corta", regenerar=regenerar)
        return red

    l = datos.escenario("limpio")
    red_a = corta(l["x_train"], l["y_train"], 1)
    red_b = corta(l["x_train"], l["y_train"], 1)
    prueba("limpio dos veces: pesos finales bit a bit iguales (determinista)", modelo.huella_pesos(red_a) == modelo.huella_pesos(red_b))
    for tipo in ruido.TIPOS:
        c = ruido.aplicar(tipo, 0.0, datos.limpio()["cuentas_train"], np.random.default_rng(ruido.semilla(tipo, ruido.NIVELES[tipo][0])))
        x = np.concatenate([l["x_train_limpio"], c[:, None]], axis=0)
        igual_dato = np.array_equal(x, l["x_train"])
        red_n = corta(x, l["y_train"], 1)
        prueba(f"{tipo:<13} con nivel 0: copia == original y pesos == limpio bit a bit",
               igual_dato and modelo.huella_pesos(red_n) == modelo.huella_pesos(red_a))
    e = datos.escenario("horizontal@0.6")
    red_h = corta(e["x_train"], e["y_train"], 1)
    prueba("horizontal@0.6: pesos finales DISTINTOS de limpio (el ruido llega a la red)", modelo.huella_pesos(red_h) != modelo.huella_pesos(red_a))
    el = datos.escenario("horizontal@0.6-linea")
    prueba("en línea: la época 1 es la copia fija (misma huella) y trae `regenerar`", el["huella_copia"] == e["huella_copia"] and el["regenerar"] is not None and el["linea"])
    red_l = corta(el["x_train"], el["y_train"], 1, regenerar=el["regenerar"])
    prueba("en línea 2 épocas: pesos DISTINTOS de la copia fija (la época 2 ya es otra copia)", modelo.huella_pesos(red_l) != modelo.huella_pesos(red_h))
    el1 = datos.escenario("horizontal@0.6-linea")
    red_l1 = corta(el1["x_train"], el1["y_train"], 1, epocas=1, regenerar=el1["regenerar"])
    red_h1 = corta(e["x_train"], e["y_train"], 1, epocas=1)
    prueba("en línea 1 época == copia fija 1 época, bit a bit", modelo.huella_pesos(red_l1) == modelo.huella_pesos(red_h1))
    cuentas = datos.limpio()["cuentas_train"]
    rng0 = np.random.default_rng(7)
    def regen0():
        return ruido.aplicar("recorte", 0.0, cuentas, rng0)[:, None]
    red_l0 = corta(l["x_train"], l["y_train"], 1, regenerar=regen0)
    prueba("en línea con nivel 0 == limpio bit a bit (regenerar no cambia nada)", modelo.huella_pesos(red_l0) == modelo.huella_pesos(red_a))
    red_2 = corta(l["x_train"], l["y_train"], 2)
    prueba("otra semilla: otros pesos", modelo.huella_pesos(red_2) != modelo.huella_pesos(red_a))
    prueba("val del escenario con ruido == val congelada", e["huella_x_val"] == datos.HUELLA_X_VAL)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "last.pt"
        _guardar(f, red_h, {"id": "prueba"}, 2)
        est = torch.load(f, map_location="cpu", weights_only=False)
        otra = modelo.Red(); otra.load_state_dict(est["modelo"])
        prueba("guarda y recupera last.pt", modelo.huella_pesos(otra) == modelo.huella_pesos(red_h) and est["epoca"] == 2)
    try:
        parsear_ident("oblicua@0.6-r2-s3"); bien = parsear_ident("oblicua@0.6-r2-s3") == ("oblicua@0.6-r2", 3)
    except ValueError:
        bien = False
    prueba("parsear_ident('oblicua@0.6-r2-s3')", bien)
    prueba("parsear_ident('vertical@1-grueso-linea-s2')", parsear_ident("vertical@1-grueso-linea-s2") == ("vertical@1-grueso-linea", 2))
    prueba("parsear_ident('recorte@0.6-linea-s2')", parsear_ident("recorte@0.6-linea-s2") == ("recorte@0.6-linea", 2))
    for malo in ("limpio", "limpio-s9", "horizontal@0.7-s1", "inventado@0.6-s1", "horizontal-s1", "limpio-linea-s1", "recorte@0.6-grueso-s1"):
        try:
            parsear_ident(malo); bien = False
        except ValueError:
            bien = True
        prueba(f"parsear_ident se niega con '{malo}'", bien)
    prueba(f"{PASOS_DEF} pasos son épocas enteras ({PASOS_DEF // PASOS_POR_EPOCA} de {PASOS_POR_EPOCA})", PASOS_DEF % PASOS_POR_EPOCA == 0)
    print("el mecanismo funciona." if ok else "✗ algo no funciona")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--escenario", help="p. ej. limpio, horizontal@0.6, oblicua@0.6-r2")
    p.add_argument("--semilla", type=int, choices=modelo.SEMILLAS)
    p.add_argument("--pasos", type=int, default=PASOS_DEF)
    p.add_argument("--hilos", type=int)
    p.add_argument("--salida", type=Path, default=PESOS, help="raíz de las corridas (por defecto nn/pesos)")
    p.add_argument("--ensayo", action="store_true")
    p.add_argument("--lr", type=float, default=None, help="SÓLO para el ensayo")
    p.add_argument("--comprobar", action="store_true")
    p.add_argument("--inicializar", action="store_true")
    a = p.parse_args()
    if a.hilos:
        torch.set_num_threads(a.hilos)
    if a.inicializar:
        return modelo.inicializar()
    if a.comprobar:
        return comprobar()
    if a.ensayo:
        if a.lr is None:
            p.error("el ensayo necesita --lr")
        return ensayo(a.escenario or ruido.LIMPIO, a.lr, a.pasos)
    if not a.escenario or a.semilla is None:
        p.error("hacen falta --escenario y --semilla (o --ensayo / --comprobar / --inicializar)")
    if a.lr is not None:
        p.error("--lr es sólo del ensayo: entrenar usa el LR congelado en el código")
    return entrenar(a.escenario, a.semilla, a.pasos, LR, a.salida)


if __name__ == "__main__":
    raise SystemExit(main())
