#!/usr/bin/env python3
"""Entrena UN brazo de `dim-nist`: 4000 pasos de lote 20 sobre las 180 imagenes de train (9 pasos
por epoca, 444 epocas) y mide la EXACTITUD sobre las 1617 de val cada 10 epocas y al final.
Sin seleccion: cuenta `last.pt`.

    python nn/entrenar_local.py --brazo w6-s3 [--pasos 4000] [--hilos N]
    python nn/entrenar_local.py --ensayo --w 8 --lr 1e-3 [--pasos 4000]   solo perdida de train
    python nn/entrenar_local.py --comprobar

⚠ EL NOMBRE DEL FICHERO ES UN CONTRATO con el freno (`cerrable.mjs` -> TRABAJOS), y cada brazo
es UN PROCESO con este nombre en la linea de comando.

La semilla fija la inicializacion y el ORDEN DE LOS LOTES (generador propio sembrado con ella),
el mismo para todos los brazos de una semilla. Las 1617 de val NO deciden nada.
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

AQUI = Path(__file__).resolve().parent
PESOS = AQUI / "pesos"
ENSAYO = AQUI / "ensayo"

LOTE = 20                   # 9 pasos por epoca sobre 180 imagenes
PASOS_DEF = 4000            # como dim-gen: 444 epocas aqui
EVAL_CADA = 10
# CONGELADO el 2026-10-02 tras el ensayo (solo perdida de train, 4000 pasos): con 3e-3 baja sin
# ninguna subida >20 % en W=8 (0,0031 final, acc_train 1,0) y W=4 (0,486, acc_train 0,83); con 1e-3
# tambien baja pero W=4 se queda en 0,89 (acc_train 0,70): aun lejos de la meseta. Se elige 3e-3.
LR = 3e-3


@torch.no_grad()
def evaluar(red, x, y) -> tuple[float, np.ndarray, float]:
    """(exactitud, aciertos por imagen, entropia cruzada media). La entropia cruzada NO satura
    cuando la exactitud de train llega a 1,0: es la que usa la descomposicion del criterio."""
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


def _bucle(red, opt, xtr, ytr, rng, epocas, registro, evaluar_en, ident, xva=None, yva=None,
           guardar_en=None, config=None) -> dict:
    paso, ultimo = 0, {}
    for ep in range(1, epocas + 1):
        t0 = time.time()
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
        fila = {"epoca": ep, "paso": paso, "perdida_train": round(suma / n, 6), "acc_train_paso": round(ok / n, 5)}
        evaluo = evaluar_en is not None and (ep % EVAL_CADA == 0 or ep == epocas)
        if evaluo:
            acc_va, _, ce_va = evaluar(red, xva, yva)
            acc_tr, _, ce_tr = evaluar(red, xtr, ytr)
            fila.update({"acc_train": round(acc_tr, 5), "acc_val": round(acc_va, 5), "brecha": round(acc_tr - acc_va, 5),
                         "ce_train": round(ce_tr, 5), "ce_val": round(ce_va, 5), "brecha_ce": round(ce_va - ce_tr, 5)})
            if guardar_en is not None:
                _guardar(guardar_en, red, config or {}, ep)
        fila["s"] = round(time.time() - t0, 3)
        with registro.open("a", encoding="utf-8") as f:
            f.write(json.dumps(fila) + "\n")
        if ep % 50 == 0 or ep == 1 or ep == epocas:
            extra = f" · acc_val {fila['acc_val']:.4f} brecha {fila['brecha']:+.4f}" if evaluo else ""
            print(f"  {ident} ep {ep:>3}/{epocas} perdida {fila['perdida_train']:.4f} acc_train~{fila['acc_train_paso']:.3f}{extra}", flush=True)
        ultimo = fila
    return ultimo


def entrenar(ident: str, pasos: int, lr) -> int:
    if lr is None:
        raise SystemExit("✗ LR no esta congelado en nn/entrenar_local.py. Corre `--ensayo`, escribe el numero aqui y en REGLAS.md, y entrena.")
    nombre, W, control, s = modelo.parsear(ident)
    if pasos % (180 // LOTE):
        raise SystemExit(f"✗ {pasos} pasos no son epocas enteras de {180 // LOTE} pasos")
    epocas = pasos // (180 // LOTE)
    d = datos.cargar(W, control)
    xtr, ytr = torch.from_numpy(d["x_train"]), torch.from_numpy(d["y_train"])
    xva, yva = torch.from_numpy(d["x_val"]), torch.from_numpy(d["y_val"])
    red = modelo.construir(W, s)
    opt = torch.optim.Adam(red.parameters(), lr=lr)
    rng = np.random.default_rng(s)
    dir_b = PESOS / ident
    dir_b.mkdir(parents=True, exist_ok=True)
    registro = dir_b / "metrics.jsonl"
    registro.write_text("", encoding="utf-8")
    config = {"id": ident, "brazo": nombre, "W": W, "control": control, "semilla": s, "pasos": pasos,
              "epocas": epocas, "lote": LOTE, "lr": lr, "L": modelo.L, "f": modelo.F, "C": modelo.C,
              "n": red.n, "mapas": red.lados, "parametros": red.n_parametros(), "dataset": d["dataset"],
              "huella_x": d["huella_x"], "huella_y": d["huella_y"]}
    print(f"empiezo {ident}: W={W}{' (control de4)' if control else ''} n={red.n} {red.n_parametros()} parametros, "
          f"{epocas} epocas, lr {lr}, hilos {torch.get_num_threads()}", flush=True)
    t0 = time.time()
    _bucle(red, opt, xtr, ytr, rng, epocas, registro, {}, ident, xva, yva, dir_b / "last.pt", config)
    acc_tr, _, ce_tr = evaluar(red, xtr, ytr)
    acc_va, ac_va, ce_va = evaluar(red, xva, yva)
    _guardar(dir_b / "last.pt", red, config, epocas)
    resumen = {**config, "acc_train": round(acc_tr, 5), "acc_val": round(acc_va, 5), "brecha": round(acc_tr - acc_va, 5),
               "ce_train": round(ce_tr, 5), "ce_val": round(ce_va, 5), "brecha_ce": round(ce_va - ce_tr, 5),
               "piso": datos.piso(d["y_val"]),
               "acc_val_por_clase": por_clase(ac_va, d["y_val"]),
               "segundos": round(time.time() - t0, 1), "maquina": maquina(),
               "terminado": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}
    (dir_b / "summary.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"listo {ident}: acc_train {acc_tr:.4f} acc_val {acc_va:.4f} brecha {acc_tr - acc_va:+.4f} · "
          f"ce_train {ce_tr:.4f} ce_val {ce_va:.4f} en {resumen['segundos']} s", flush=True)
    return 0


def ensayo(W: int, lr: float, pasos: int, control: bool = False) -> int:
    """SOLO la perdida de train, para elegir el lr. No mira val."""
    epocas = max(1, pasos // (180 // LOTE))
    d = datos.cargar(W, control)
    xtr, ytr = torch.from_numpy(d["x_train"]), torch.from_numpy(d["y_train"])
    red = modelo.construir(W, 1)
    opt = torch.optim.Adam(red.parameters(), lr=lr)
    ENSAYO.mkdir(parents=True, exist_ok=True)
    registro = ENSAYO / f"w{W}{'-de4' if control else ''}-lr{lr:g}.jsonl"
    registro.write_text("", encoding="utf-8")
    print(f"ensayo W={W} lr={lr:g}: {epocas} epocas ({red.n_parametros()} parametros), SIN mirar val")
    _bucle(red, opt, xtr, ytr, np.random.default_rng(1), epocas, registro, None, f"ensayo-w{W}-lr{lr:g}")
    filas = [json.loads(l) for l in registro.read_text(encoding="utf-8").splitlines()]
    p = np.array([f["perdida_train"] for f in filas]); acc = np.array([f["acc_train_paso"] for f in filas])
    subidas = int((p[1:] > 1.2 * p[:-1]).sum())
    tramo = max(1, len(p) // 10)
    ini, fin = float(p[:tramo].mean()), float(p[-tramo:].mean())
    veredicto = ("baja sin oscilar" if fin < 0.5 * ini and subidas <= max(1, len(p) // 10)
                 else "NO sirve: " + ("no baja a la mitad" if fin >= 0.5 * ini else "oscila"))
    print(f"  perdida: inicio {ini:.4f} -> final {fin:.4f} (min {p.min():.4f}); acc_train final {acc[-tramo:].mean():.3f}; "
          f"subidas >20 %: {subidas} de {len(p) - 1} epocas  →  {veredicto}")
    return 0


def comprobar() -> int:
    import tempfile
    ok = True
    for ident in ("w4-s1", "w5-s1", "w6-s1", "w7-s1", "w8-s1", "w8-de4-s1"):
        nombre, W, control, s = modelo.parsear(ident)
        red = modelo.construir(W, s)
        x = torch.rand(LOTE, 1, W, W); y = torch.randint(0, 10, (LOTE,))
        opt = torch.optim.Adam(red.parameters(), lr=1e-3)
        antes = F_.cross_entropy(red(x), y).item()
        for _ in range(2):
            opt.zero_grad(); F_.cross_entropy(red(x), y).backward(); opt.step()
        despues = F_.cross_entropy(red(x), y).item()
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "last.pt"
            _guardar(f, red, {"id": ident}, 2)
            est = torch.load(f, map_location="cpu", weights_only=False)
            otra = modelo.construir(W, 99); otra.load_state_dict(est["modelo"])
            igual = all(torch.equal(p, q) for p, q in zip(otra.parameters(), red.parameters()))
        bien = igual and est["epoca"] == 2 and np.isfinite(despues)
        ok &= bien
        print(f"  {ident:<10} 2 pasos ({antes:.3f} -> {despues:.3f}), guarda y recupera ... {'ok' if bien else 'FALLA'}")
    print("el mecanismo funciona." if ok else "✗ algo no funciona")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--brazo")
    p.add_argument("--pasos", type=int, default=PASOS_DEF)
    p.add_argument("--hilos", type=int)
    p.add_argument("--ensayo", action="store_true")
    p.add_argument("--w", type=int, default=8)
    p.add_argument("--control", action="store_true")
    p.add_argument("--lr", type=float, default=None, help="SOLO para el ensayo")
    p.add_argument("--comprobar", action="store_true")
    a = p.parse_args()
    if a.hilos:
        torch.set_num_threads(a.hilos)
    if a.comprobar:
        return comprobar()
    if a.ensayo:
        if a.lr is None:
            p.error("el ensayo necesita --lr")
        return ensayo(a.w, a.lr, a.pasos, a.control)
    if not a.brazo:
        p.error("hace falta --brazo (o --ensayo / --comprobar)")
    if a.lr is not None:
        p.error("--lr es solo del ensayo: entrenar usa el LR congelado en el codigo")
    return entrenar(a.brazo, a.pasos, LR)


if __name__ == "__main__":
    raise SystemExit(main())
