#!/usr/bin/env python3
"""Curvas de aprendizaje de las dos CNN tradicionales (nn/cnn.py) sobre las MISMAS 60 particiones de la ganancia
(nn/ganancia.py: datasets balanceados T ∈ {500, 1000, 2000, 4000} × p ∈ {2, 4, 10, 20, 50} % × 3 semillas, del pool de
5620): 120 entrenamientos independientes, pensados para correr A LA VEZ en una máquina de Vast (`nn/vast.sh cnn`).
Criterio en instrucciones/02-criterio.md § «Curvas de CNN», escrito antes de correr.

Protocolo, el mismo para las dos (el de ruido-nist): 3996 pasos de lotes de 20 (si N < 20, el lote es N), Adam con el lr
de nn/cnn.py, pesos iniciales torch.manual_seed(sem), SIN selección por validación (se evalúa la red del último paso) y
sin aumento de datos. La CNN de 3 capas ve el 8×8 (cuentas/16, el dato de 8 px reducido aquí del de 32: es el mismo bit a
bit, comprobado en nn/datos.py); LeNet-5 ve el bitmap de 32×32.

    python nn/curvas_cnn.py --comprobar                  particiones = las de la ganancia (huellas) + mecanismo
    python nn/curvas_cnn.py --todas [--procesos K]       los 120, K a la vez (defecto: TRABAJO_VCPU o nproc), 1 hilo cada uno
    python nn/curvas_cnn.py --una lenet5 4000 0.5 1      uno
    python nn/curvas_cnn.py --resumen                    junta nn/curvas-cnn/*/*.json en resultados/curvas-cnn.json

Cada entrenamiento deja nn/curvas-cnn/<modelo>/T<T>-p<p>-s<sem>.json al terminar; si ya existe, se salta (relanzar no
repite lo hecho).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F_

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import cnn as C                                 # noqa: E402
import datos                                    # noqa: E402
import ganancia as Gn                           # noqa: E402

SALIDA = AQUI / "curvas-cnn"
RES = AQUI.parent / "resultados"
PASOS = 3996
LOTE = 20
_DATO: dict = {}


def dato() -> dict:
    """y del pool (orden de la ganancia) y las dos entradas: 32×32 binaria y 8×8 = cuentas 4×4 / 16."""
    if not _DATO:
        d = datos.digitos(con_extra=True)
        x32 = d["x"].astype(np.float32)
        x8 = (x32.reshape(-1, 1, 8, 4, 8, 4).sum((3, 5)) / 16).astype(np.float32)
        _DATO.update({"y": d["y"].astype(np.int64), 32: torch.from_numpy(x32), 8: torch.from_numpy(x8)})
    return _DATO


def ruta(modelo: str, T: int, p: float, sem: int, raiz: Path = SALIDA) -> Path:
    return raiz / modelo / f"T{T}-p{p:g}-s{sem}.json"


def entrenar(modelo: str, T: int, p: float, sem: int, pasos: int = PASOS, raiz: Path = SALIDA) -> dict:
    destino = ruta(modelo, T, p, sem, raiz)
    if destino.exists():
        return json.loads(destino.read_text())
    torch.set_num_threads(1)
    d = dato(); y = d["y"]
    tr, te = Gn.particion(y, T, p, sem)
    clase, lr = C.MODELOS[modelo]
    x = d[clase.entrada]
    xtr, ytr, xte, yte = x[tr], torch.from_numpy(y[tr]), x[te], torch.from_numpy(y[te])
    torch.manual_seed(sem)
    red = clase()
    opt = torch.optim.Adam(red.parameters(), lr=lr)
    rng = np.random.default_rng(10_000 + sem)
    lote = min(LOTE, len(tr)); paso = 0; t0 = time.time()
    red.train()
    while paso < pasos:
        perm = rng.permutation(len(tr))
        for i in range(0, len(perm), lote):
            idx = torch.from_numpy(perm[i:i + lote])
            opt.zero_grad(); F_.cross_entropy(red(xtr[idx]), ytr[idx]).backward(); opt.step()
            paso += 1
            if paso >= pasos:
                break
    red.eval()
    with torch.no_grad():
        out = torch.cat([red(xte[i:i + 2048]) for i in range(0, len(te), 2048)])
        acc = float((out.argmax(1) == yte).float().mean()); ce = float(F_.cross_entropy(out, yte))
        acc_tr = float((red(xtr).argmax(1) == ytr).float().mean())
    Tr, N = len(tr) + len(te), len(tr)
    fila = {"modelo": modelo, "T": T, "p": p, "sem": sem, "T_real": Tr, "N": N, "n_test": len(te), "acc": round(acc, 4),
            "acc_train": round(acc_tr, 4), "ce_test": round(ce, 4), "pct_train": round(N / Tr, 5), "G": round(acc / (N / Tr), 3),
            "pasos": pasos, "lote": lote, "lr": lr, "parametros": C.n_parametros(red), "segundos": round(time.time() - t0, 1)}
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(".tmp"); tmp.write_text(json.dumps(fila) + "\n"); tmp.replace(destino)
    return fila


def _uno(args) -> str:
    modelo, T, p, sem = args
    try:
        f = entrenar(modelo, T, p, sem)
        return f"{modelo} T={T} p={p:g} s={sem}: N={f['N']} acc {f['acc']:.4f} ({f['segundos']} s)"
    except Exception as exc:                        # noqa: BLE001  (un fallo no tumba a los demás; queda sin json)
        return f"✗ {modelo} T={T} p={p:g} s={sem}: {type(exc).__name__}: {exc}"


def tareas() -> list[tuple]:
    # las LeNet primero: son las largas, y así no quedan solas al final
    return [(m, T, p, s) for m in ("lenet5", "cnn3") for T in Gn.T_TAMANOS for p in Gn.FRACCIONES for s in Gn.SEMILLAS]


def todas(procesos: int) -> int:
    t0 = time.time(); fallos = 0
    with Pool(procesos) as pool:
        for i, linea in enumerate(pool.imap_unordered(_uno, tareas()), 1):
            fallos += linea.startswith("✗")
            print(f"[{i:>3}/{len(tareas())} · {time.time() - t0:5.0f} s] {linea}", flush=True)
    print(f"terminado: {len(tareas()) - fallos} bien, {fallos} con fallo, {time.time() - t0:.0f} s con {procesos} procesos")
    return 1 if fallos else 0


def huellas_particiones() -> tuple[str, str]:
    """Las mismas huellas que nn/ganancia.py (mismo orden de bucle): si no casan, no son las mismas particiones."""
    y = dato()["y"]; hp = hashlib.sha256()
    for T in Gn.T_TAMANOS:
        for p in Gn.FRACCIONES:
            for s in Gn.SEMILLAS:
                tr, te = Gn.particion(y, T, p, s)
                hp.update(tr.tobytes()); hp.update(te.tobytes())
    return Gn.huella(y), hp.hexdigest()[:16]


def comprobar() -> int:
    import tempfile
    ok = True

    def prueba(que, bien, det=""):
        nonlocal ok
        ok &= bool(bien); print(f"  [{'ok' if bien else 'FALLA':>5}] {que}" + (f"  {det}" if det else ""))

    g = json.loads((RES / "ganancia.json").read_text())
    hy, hp = huellas_particiones()
    prueba("las etiquetas del pool son las de la ganancia", hy == g["huella_y"], hy)
    prueba("las 60 particiones son las de la ganancia", hp == g["huella_particiones"], hp)
    d = dato()
    z8 = np.load(datos.exigir_dataset(datos.DIGITOS_8PX) / "datos.npz")
    prueba("el 8×8 reducido aquí es el dato publicado de 8 px (windep)",
           np.array_equal((d[8][:1797, 0].numpy() * 16).round().astype(np.uint8), z8["imagenes"]))
    with tempfile.TemporaryDirectory() as tmp:
        for m in C.MODELOS:
            a = entrenar(m, 500, 0.02, 1, pasos=50, raiz=Path(tmp))
            b = entrenar(m, 500, 0.02, 1, pasos=50, raiz=Path(tmp) / "b")
            prueba(f"{m}: 50 pasos con N = {a['N']}, determinista", a["acc"] == b["acc"] and a["N"] == 10, f"acc {a['acc']}")
            c = entrenar(m, 500, 0.02, 1, pasos=50, raiz=Path(tmp))
            prueba(f"{m}: si el json ya existe, se salta (no se repite)", c == a)
    print("el mecanismo funciona." if ok else "✗ algo no funciona")
    return 0 if ok else 1


def resumen() -> int:
    casos = {}
    for m in C.MODELOS:
        filas = [json.loads(f.read_text()) for f in sorted((SALIDA / m).glob("*.json"))]
        casos[C.ETIQUETAS[m]] = filas
        print(f"{C.ETIQUETAS[m]:<30} {len(filas)} de {len(Gn.T_TAMANOS) * len(Gn.FRACCIONES) * len(Gn.SEMILLAS)} entrenamientos")
    hy, hp = huellas_particiones()
    out = {"protocolo": f"{PASOS} pasos de lotes de {LOTE}, Adam, sin selección ni aumento; particiones de nn/ganancia.py",
           "huella_y": hy, "huella_particiones": hp, "casos": casos}
    (RES / "curvas-cnn.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("→ resultados/curvas-cnn.json")
    return 0


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--comprobar", action="store_true"); a.add_argument("--todas", action="store_true")
    a.add_argument("--resumen", action="store_true"); a.add_argument("--una", nargs=4, metavar=("MODELO", "T", "P", "SEM"))
    a.add_argument("--procesos", type=int, default=int(os.environ.get("TRABAJO_VCPU") or os.cpu_count() or 1))
    x = a.parse_args()
    if x.comprobar:
        return comprobar()
    if x.todas:
        return todas(x.procesos)
    if x.resumen:
        return resumen()
    if x.una:
        print(_uno((x.una[0], int(x.una[1]), float(x.una[2]), int(x.una[3]))))
        return 0
    a.print_help(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
