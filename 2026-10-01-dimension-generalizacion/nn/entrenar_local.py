#!/usr/bin/env python3
"""Entrena UN brazo de `dim-gen`: 4000 pasos de lote 20 sobre las 100 imagenes de train, y mide
el IoU sobre las 900 restantes cada 10 epocas y al final. Sin seleccion: cuenta `last.pt`.

    python nn/entrenar_local.py --brazo w064-s3 [--pasos 1000] [--hilos N]
    python nn/entrenar_local.py --ensayo --w 32 --lr 1e-3 [--pasos 1000]   solo perdida de train
    python nn/entrenar_local.py --comprobar                                  2 pasos por W + guardar/cargar

⚠ EL NOMBRE DEL FICHERO ES UN CONTRATO, no una preferencia: `telegram-coordinator/scripts/
cerrable.mjs` casa `entrenar_local\\.py` en su lista TRABAJOS. Con otro nombre, el freno diria
«nada corriendo» con 200 epocas vivas. Y cada brazo es UN PROCESO con este nombre en la linea
de comando (importarlo desde otro script lo haria invisible).

LA SEMILLA fija la inicializacion (en `modelo.construir`) y el ORDEN DE LOS LOTES (un
generador propio sembrado con ella), que por eso es el mismo para todos los brazos de una
semilla. Las 900 de validacion NO deciden nada: ni parada, ni seleccion, ni lr.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos                                   # noqa: E402
import modelo                                  # noqa: E402

AQUI = Path(__file__).resolve().parent
PESOS = AQUI / "pesos"
ENSAYO = AQUI / "ensayo"

LOTE = 20                   # 5 pasos por epoca sobre 100 imagenes
PASOS_DEF = 4000            # 800 epocas. El plan decia 1000; el ensayo (SOLO perdida de train, antes
                            # de mirar nada de val) los dejo cortos: a W=32 la perdida a las 200 epocas
                            # era 0,033 y seguia cayendo, y se aplana hacia las 800 (0,0077); a W=8 es
                            # plana desde las 400; a W=16 aun baja un 7 % por cada 200 a las 1000.
                            # Medido el 2026-10-02 (nn/ensayo/). Se fijo ANTES de la primera corrida.
EVAL_CADA = 10              # epocas entre evaluaciones de las 900
LOTE_EVAL = 100             # la maquina del dev tiene 3,8 GB y no tiene swap
# CONGELADO el 2026-10-02 tras el ensayo de mecanismo (REGLAS.md § Procesos 3), uno para TODOS los
# brazos: con 1e-3 la perdida de train baja sin ninguna subida >20 % entre epocas consecutivas en
# W=8, 16 y 32 (5000 pasos), W=64 (250 pasos) y W=128 (50 pasos, 0,595 -> 0,133). Con 3e-4 tambien
# baja (W=32) pero mas despacio (0,058 contra 0,046 a los 1000 pasos). Elegido por ESTABILIDAD.
LR = 1e-3


def iou_t(pred: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    """La misma regla que `datos.iou`: prediccion recortada a [0,1], invertida = 0."""
    p = pred.clamp(0.0, 1.0)
    r = real
    valida = (p[:, 1] > p[:, 0]) & (p[:, 3] > p[:, 2])
    ix = (torch.minimum(p[:, 1], r[:, 1]) - torch.maximum(p[:, 0], r[:, 0])).clamp(min=0)
    iy = (torch.minimum(p[:, 3], r[:, 3]) - torch.maximum(p[:, 2], r[:, 2])).clamp(min=0)
    inter = ix * iy
    ap = (p[:, 1] - p[:, 0]) * (p[:, 3] - p[:, 2])
    ar = (r[:, 1] - r[:, 0]) * (r[:, 3] - r[:, 2])
    out = inter / (ap + ar - inter).clamp(min=1e-12)
    return torch.where(valida, out, torch.zeros_like(out))


@torch.no_grad()
def evaluar(red: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, np.ndarray]:
    red.eval()
    ious = []
    for i in range(0, len(x), LOTE_EVAL):
        ious.append(iou_t(red(x[i:i + LOTE_EVAL]), y[i:i + LOTE_EVAL]))
    todos = torch.cat(ious).numpy()
    return float(todos.mean()), todos


def maquina() -> dict:
    cpu = ""
    try:
        for linea in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
            if linea.startswith("model name"):
                cpu = linea.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"cpu": cpu, "nproc": os.cpu_count(), "hilos_torch": torch.get_num_threads(),
            "torch": torch.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "host": platform.node()}


def por_factor(ious: np.ndarray, f: dict) -> dict:
    """El IoU de val desglosado por los factores del generador. No decide nada: evita tener que
    conservar pesos para leerlo despues."""
    out: dict = {"fuente": {}}
    for fu in sorted(set(f["fuente"].tolist())):
        m = f["fuente"] == fu
        out["fuente"][fu] = {"n": int(m.sum()), "iou": round(float(ious[m].mean()), 5)}
    for k in ("cuerpo", "gris_nivel", "area"):
        v = f[k]
        cortes = np.quantile(v, [0.25, 0.5, 0.75])
        bins = np.digitize(v, cortes)
        out[k] = {}
        for b in range(4):
            m = bins == b
            if m.any():
                out[k][f"q{b + 1}"] = {"n": int(m.sum()), "iou": round(float(ious[m].mean()), 5),
                                       "rango": [round(float(v[m].min()), 4), round(float(v[m].max()), 4)]}
    return out


def _guardar(destino: Path, red: torch.nn.Module, config: dict, epoca: int) -> None:
    tmp = destino.with_suffix(".tmp")
    torch.save({"modelo": red.state_dict(), "config": config, "epoca": epoca}, tmp)
    tmp.replace(destino)


def _bucle(red, opt, xtr, ytr, rng, epocas: int, registro, evaluar_en: dict | None, ident: str,
           xva=None, yva=None, guardar_en: Path | None = None, config: dict | None = None) -> dict:
    """El bucle comun al entrenamiento y al ensayo. `evaluar_en` = None: nunca mira val."""
    paso = 0
    ultimo = {}
    for ep in range(1, epocas + 1):
        t0 = time.time()
        red.train()
        perm = rng.permutation(len(xtr))
        suma, n, ious_paso = 0.0, 0, []
        for i in range(0, len(perm), LOTE):
            idx = torch.from_numpy(perm[i:i + LOTE])
            opt.zero_grad()
            out = red(xtr[idx])
            perdida = (out - ytr[idx]).abs().mean()
            perdida.backward()
            opt.step()
            paso += 1
            suma += perdida.item() * len(idx)
            n += len(idx)
            ious_paso.append(iou_t(out.detach(), ytr[idx]))
        fila = {"epoca": ep, "paso": paso, "perdida_train": round(suma / n, 6),
                "iou_train_paso": round(float(torch.cat(ious_paso).mean()), 5)}
        evaluo = evaluar_en is not None and (ep % EVAL_CADA == 0 or ep == epocas)
        if evaluo:
            iou_va, _ = evaluar(red, xva, yva)
            iou_tr, _ = evaluar(red, xtr, ytr)
            fila.update({"iou_train": round(iou_tr, 5), "iou_val": round(iou_va, 5),
                         "brecha": round(iou_tr - iou_va, 5)})
            if guardar_en is not None:
                _guardar(guardar_en, red, config or {}, ep)
        fila["s"] = round(time.time() - t0, 2)
        with registro.open("a", encoding="utf-8") as f:
            f.write(json.dumps(fila) + "\n")
        if ep % 10 == 0 or ep == 1 or ep == epocas:
            extra = (f" · iou_val {fila['iou_val']:.4f} brecha {fila['brecha']:+.4f}" if evaluo else "")
            print(f"  {ident} ep {ep:>3}/{epocas} perdida {fila['perdida_train']:.4f} "
                  f"iou_train~{fila['iou_train_paso']:.3f}{extra} · {fila['s']}s", flush=True)
        ultimo = fila
    return ultimo


def entrenar(ident: str, pasos: int, lr: float | None) -> int:
    if lr is None:
        raise SystemExit("✗ LR no esta congelado en nn/entrenar_local.py. Corre el ensayo "
                         "(`--ensayo`), escribe el numero en el codigo y en REGLAS.md, y entrena.")
    nombre, W, control, s = modelo.parsear(ident)
    if pasos % (100 // LOTE):
        raise SystemExit(f"✗ {pasos} pasos no son epocas enteras de {100 // LOTE} pasos")
    epocas = pasos // (100 // LOTE)
    d = datos.cargar(W, control)
    xtr, ytr = torch.from_numpy(d["x_train"]), torch.from_numpy(d["y_train"])
    xva, yva = torch.from_numpy(d["x_val"]), torch.from_numpy(d["y_val"])
    red = modelo.construir(W, s)
    opt = torch.optim.Adam(red.parameters(), lr=lr)
    rng = np.random.default_rng(s)
    dir_b = PESOS / ident
    dir_b.mkdir(parents=True, exist_ok=True)
    registro = dir_b / "metrics.jsonl"
    registro.write_text("", encoding="utf-8")            # un brazo se entrena de cero, entero
    config = {"id": ident, "brazo": nombre, "W": W, "control": control, "semilla": s,
              "pasos": pasos, "epocas": epocas, "lote": LOTE, "lr": lr, "L": modelo.L,
              "f": modelo.F, "C": modelo.C, "n": red.n, "mapas": red.lados,
              "parametros": red.n_parametros(), "dataset": d["dataset"],
              "huella_x": d["huella_x"], "huella_y": d["huella_y"]}
    print(f"empiezo {ident}: W={W}{' (control de16)' if control else ''} n={red.n} "
          f"{red.n_parametros()} parametros, {epocas} epocas, lr {lr}, hilos {torch.get_num_threads()}",
          flush=True)
    t0 = time.time()
    _bucle(red, opt, xtr, ytr, rng, epocas, registro, {}, ident, xva, yva, dir_b / "last.pt", config)
    iou_tr, _ = evaluar(red, xtr, ytr)
    iou_va, ious_va = evaluar(red, xva, yva)
    _guardar(dir_b / "last.pt", red, config, epocas)
    resumen = {**config,
               "iou_train": round(iou_tr, 5), "iou_val": round(iou_va, 5),
               "brecha": round(iou_tr - iou_va, 5),
               "piso_caja_media": round(datos.piso_caja_media(d["y_train"], d["y_val"]), 5),
               "iou_val_por_factor": por_factor(ious_va, d["factores_val"]),
               "segundos": round(time.time() - t0, 1), "maquina": maquina(),
               "terminado": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}
    (dir_b / "summary.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    print(f"listo {ident}: iou_train {iou_tr:.4f} iou_val {iou_va:.4f} brecha {iou_tr - iou_va:+.4f} "
          f"en {resumen['segundos']} s", flush=True)
    return 0


def ensayo(W: int, lr: float, pasos: int, control: bool = False) -> int:
    """SOLO la perdida de train, para elegir el lr (REGLAS.md § Procesos 3). No mira val."""
    epocas = max(1, pasos // (100 // LOTE))
    d = datos.cargar(W, control)
    xtr, ytr = torch.from_numpy(d["x_train"]), torch.from_numpy(d["y_train"])
    red = modelo.construir(W, 1)
    opt = torch.optim.Adam(red.parameters(), lr=lr)
    ENSAYO.mkdir(parents=True, exist_ok=True)
    registro = ENSAYO / f"w{W}{'-de16' if control else ''}-lr{lr:g}.jsonl"
    registro.write_text("", encoding="utf-8")
    ident = f"ensayo-w{W}-lr{lr:g}"
    print(f"ensayo W={W} lr={lr:g}: {epocas} epocas ({red.n_parametros()} parametros), SIN mirar val")
    _bucle(red, opt, xtr, ytr, np.random.default_rng(1), epocas, registro, None, ident)
    filas = [json.loads(l) for l in registro.read_text(encoding="utf-8").splitlines()]
    p = np.array([f["perdida_train"] for f in filas])
    subidas = int((p[1:] > 1.2 * p[:-1]).sum())
    tramo = max(1, len(p) // 10)
    ini, fin = float(p[:tramo].mean()), float(p[-tramo:].mean())
    veredicto = ("baja sin oscilar" if fin < 0.5 * ini and subidas <= max(1, len(p) // 10)
                 else "NO sirve: " + ("no baja a la mitad" if fin >= 0.5 * ini else "oscila"))
    print(f"  perdida: inicio {ini:.4f} -> final {fin:.4f} (min {p.min():.4f}); subidas >20 %: "
          f"{subidas} de {len(p) - 1} epocas  →  {veredicto}")
    return 0


def comprobar() -> int:
    import tempfile
    ok = True
    for ident in ("w008-s1", "w016-s1", "w032-s1", "w064-s1", "w128-s1", "w128-de16-s1"):
        nombre, W, control, s = modelo.parsear(ident)
        red = modelo.construir(W, s)
        x = torch.rand(LOTE, 1, W, W); y = torch.rand(LOTE, 4)
        opt = torch.optim.Adam(red.parameters(), lr=1e-3)
        antes = (red(x) - y).abs().mean().item()
        for _ in range(2):
            opt.zero_grad(); (red(x) - y).abs().mean().backward(); opt.step()
        despues = (red(x) - y).abs().mean().item()
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "last.pt"
            _guardar(f, red, {"id": ident}, 2)
            est = torch.load(f, map_location="cpu", weights_only=False)
            otra = modelo.construir(W, 99)
            otra.load_state_dict(est["modelo"])
            igual = all(torch.equal(p, q) for p, q in zip(otra.parameters(), red.parameters()))
        bien = igual and est["epoca"] == 2 and np.isfinite(despues)
        ok &= bien
        print(f"  {ident:<13} 2 pasos ({antes:.3f} -> {despues:.3f}), guarda y recupera ... "
              f"{'ok' if bien else 'FALLA'}")
    print("el mecanismo funciona." if ok else "✗ algo no funciona")
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--brazo", help="p. ej. w064-s3 o w128-de16-s1")
    p.add_argument("--pasos", type=int, default=PASOS_DEF)
    p.add_argument("--hilos", type=int, help="torch.set_num_threads: para correr varios a la vez")
    p.add_argument("--ensayo", action="store_true", help="solo la perdida de train, a nn/ensayo/")
    p.add_argument("--w", type=int, default=32, help="W del ensayo")
    p.add_argument("--control", action="store_true", help="ensayo con el control de16")
    p.add_argument("--lr", type=float, default=None, help="SOLO para el ensayo; entrenar usa LR")
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
