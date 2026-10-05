#!/usr/bin/env python3
"""Curvas de aprendizaje de redes entrenadas de punta a punta (nn/cnn.py) sobre las MISMAS 60 particiones de la ganancia
(nn/ganancia.py: datasets balanceados T ∈ {500, 1000, 2000, 4000} × p ∈ {2, 4, 10, 20, 50} % × 3 semillas, del pool de
5620): 60 entrenamientos independientes por modelo, pensados para correr A LA VEZ en Vast (`nn/vast.sh cnn` /
`nn/vast.sh compositor`). Criterios en instrucciones/02-criterio.md, escritos antes de correr cada tanda.

  cnn3, lenet5                     las dos CNN tradicionales (tanda del 2026-10-05, «Curvas de CNN»)
  cnn3pos, aprendidos13, ajuste13  A, B y C: las CNN con el compositor de los detectores («CNN con compositor»)

Protocolo, el mismo para todas (el de ruido-nist): 3996 pasos de lotes de 20 (si N < 20, el lote es N), Adam, pesos
iniciales torch.manual_seed(sem), SIN selección por validación (se evalúa la red del último paso) y sin aumento de datos.
`ajuste13` arranca de nn/init-detectores-8px.pt y va en dos fases (cnn.AJUSTE): la mitad de los pasos sólo el compositor,
la otra mitad todo. Entra el 8×8 (cuentas/16, reducido aquí del de 32: bit a bit el de 8 px) o el 32×32 según el modelo.

    python nn/curvas_cnn.py --comprobar                        particiones = las de la ganancia (huellas) + mecanismo
    python nn/curvas_cnn.py --exportar-init                    nn/init-detectores-8px.pt desde los detectores de feat-ind
    python nn/curvas_cnn.py --todas --modelos a,b [--procesos K]   todos los de esos modelos, K a la vez, 1 hilo cada uno
    python nn/curvas_cnn.py --una lenet5 4000 0.5 1            uno
    python nn/curvas_cnn.py --resumen                          junta nn/curvas-cnn/*/*.json en resultados/curvas-cnn.json

Cada entrenamiento deja nn/curvas-cnn/<modelo>/T<T>-p<p>-s<sem>.json al terminar; si ya existe, se salta.
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


def huella_tensores(estado: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(estado):
        h.update(k.encode()); h.update(estado[k].detach().contiguous().to(torch.float32).numpy().astype("<f4").tobytes())
    return h.hexdigest()[:16]


def cargar_init() -> dict:
    est = torch.load(AQUI / C.INIT_AJUSTE, map_location="cpu", weights_only=False)
    if huella_tensores(est["detectores"]) != est["huella"]:
        raise RuntimeError(f"{C.INIT_AJUSTE}: la huella no casa. Me niego.")
    return est


def _pasos(red, opt, xtr, ytr, n: int, rng, lote: int) -> None:
    red.train(); hechos = 0
    while hechos < n:
        perm = rng.permutation(len(xtr))
        for i in range(0, len(perm), lote):
            idx = torch.from_numpy(perm[i:i + lote])
            opt.zero_grad(); F_.cross_entropy(red(xtr[idx]), ytr[idx]).backward(); opt.step()
            hechos += 1
            if hechos >= n:
                return


@torch.no_grad()
def _evaluar(red, x, y) -> tuple[float, float]:
    red.eval()
    out = torch.cat([red(x[i:i + 512]) for i in range(0, len(x), 512)])   # 512: 32 bancos a la vez caben en memoria
    return float((out.argmax(1) == y).float().mean()), float(F_.cross_entropy(out, y))


def entrenar_red(modelo: str, T: int, p: float, sem: int, pasos: int = PASOS):
    """Entrena UNA red y la devuelve con su partición: (red, tr, te, extra, t0). Es lo que hace `entrenar` antes de evaluar,
    separado para poder mirar predicciones dígito a dígito (nn/errores_c.py) con EXACTAMENTE el mismo entrenamiento."""
    torch.set_num_threads(1)
    d = dato(); y = d["y"]
    tr, te = Gn.particion(y, T, p, sem)
    clase, lr = C.MODELOS[modelo]
    x = d[clase.entrada]
    xtr, ytr, xte, yte = x[tr], torch.from_numpy(y[tr]), x[te], torch.from_numpy(y[te])
    torch.manual_seed(sem)
    red = clase()
    rng = np.random.default_rng(10_000 + sem)
    lote = min(LOTE, len(tr)); t0 = time.time(); extra = {}
    if modelo == "ajuste13":
        init = cargar_init()
        red.detectores.load_state_dict(init["detectores"])
        a = C.AJUSTE; n1 = round(pasos * a["pasos_congelado"] / PASOS)
        for q in red.detectores.parameters():
            q.requires_grad_(False)
        _pasos(red, torch.optim.Adam(red.compositor.parameters(), lr=a["lr_compositor_congelado"]), xtr, ytr, n1, rng, lote)
        acc1, ce1 = _evaluar(red, xte, yte)
        for q in red.detectores.parameters():
            q.requires_grad_(True)
        opt = torch.optim.Adam([{"params": red.detectores.parameters(), "lr": a["lr_detectores"]},
                                {"params": red.compositor.parameters(), "lr": a["lr_compositor"]}])
        _pasos(red, opt, xtr, ytr, pasos - n1, rng, lote)
        extra = {"acc_congelado": round(acc1, 4), "ce_congelado": round(ce1, 4), "pasos_congelado": n1, "init": init["huella"]}
    else:
        _pasos(red, torch.optim.Adam(red.parameters(), lr=lr), xtr, ytr, pasos, rng, lote)
    return red, tr, te, extra, t0


def entrenar(modelo: str, T: int, p: float, sem: int, pasos: int = PASOS, raiz: Path = SALIDA) -> dict:
    destino = ruta(modelo, T, p, sem, raiz)
    if destino.exists():
        return json.loads(destino.read_text())
    red, tr, te, extra, t0 = entrenar_red(modelo, T, p, sem, pasos)
    d = dato(); y = d["y"]; clase, lr = C.MODELOS[modelo]; x = d[clase.entrada]
    xtr, ytr, xte, yte = x[tr], torch.from_numpy(y[tr]), x[te], torch.from_numpy(y[te])
    lote = min(LOTE, len(tr))
    acc, ce = _evaluar(red, xte, yte)
    acc_tr, _ = _evaluar(red, xtr, ytr)
    Tr, N = len(tr) + len(te), len(tr)
    fila = {"modelo": modelo, "T": T, "p": p, "sem": sem, "T_real": Tr, "N": N, "n_test": len(te), "acc": round(acc, 4),
            "acc_train": round(acc_tr, 4), "ce_test": round(ce, 4), "pct_train": round(N / Tr, 5), "G": round(acc / (N / Tr), 3),
            "pasos": pasos, "lote": lote, "lr": lr if lr else C.AJUSTE, "parametros": C.n_parametros(red),
            "segundos": round(time.time() - t0, 1), **extra}
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


def tareas(modelos: list[str]) -> list[tuple]:
    # en el orden dado (los largos primero, para que no queden solos al final), y dentro, los T grandes primero
    return [(m, T, p, s) for m in modelos for T in sorted(Gn.T_TAMANOS, reverse=True) for p in Gn.FRACCIONES for s in Gn.SEMILLAS]


def todas(modelos: list[str], procesos: int) -> int:
    malos = [m for m in modelos if m not in C.MODELOS]
    if malos:
        raise SystemExit(f"✗ modelos desconocidos: {malos}; los de nn/cnn.py son {list(C.MODELOS)}")
    if "ajuste13" in modelos:
        cargar_init()                               # se niega ANTES de empezar si el init no está o no casa (R2)
    ts = tareas(modelos); t0 = time.time(); fallos = 0
    with Pool(procesos) as pool:
        for i, linea in enumerate(pool.imap_unordered(_uno, ts), 1):
            fallos += linea.startswith("✗")
            print(f"[{i:>3}/{len(ts)} · {time.time() - t0:5.0f} s] {linea}", flush=True)
    print(f"terminado: {len(ts) - fallos} bien, {fallos} con fallo, {time.time() - t0:.0f} s con {procesos} procesos")
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


def exportar_init() -> int:
    """nn/init-detectores-8px.pt: los 13 detectores de la corrida 2 de `feat-ind` (nn/pesos/<f>/best.pt, leídos por el id
    del experimento en el registro), apilados para el Banco13. Comprueba, antes de escribir nada, que el banco da EXACTAMENTE
    los mapas de los 13 detectores por separado y los de resultados/mapas-digitos.npz de feat-ind."""
    sys.path.insert(0, str(AQUI.parent.parent))
    from expcnn.registro import por_id               # noqa: PLC0415
    raiz = por_id("feat-ind").carpeta
    estados, origen = [], {}
    for f in C.FAMILIAS:
        ruta_pt = raiz / "nn" / "pesos" / f / "best.pt"
        est = torch.load(ruta_pt, map_location="cpu", weights_only=False)
        if tuple(est["config"]["canales"]) != (16, 32, 32):
            raise SystemExit(f"✗ {f}: canales {est['config']['canales']}, no (16, 32, 32). Me niego.")
        estados.append(est["modelo"])
        origen[f] = {"fichero": f"feat-ind:nn/pesos/{f}/best.pt", "sha256": hashlib.sha256(ruta_pt.read_bytes()).hexdigest()[:16],
                     "epoca": est["epoca"]}
    banco = C.Banco13(); banco.detectores.load_state_dict(C.Banco13.estado_desde_detectores(estados)); banco.eval()
    x = torch.rand(64, 1, 8, 8)
    with torch.no_grad():
        uno_a_uno = torch.cat([_det(e)(x) for e in estados], 1)
        dif = float((banco.mapas(x) - uno_a_uno).abs().max())
    print(f"  banco agrupado contra los 13 detectores por separado: diferencia máxima {dif:.2e}")
    if dif > 1e-5:
        raise SystemExit("✗ el banco agrupado NO reproduce los 13 detectores. Me niego.")
    mapas = raiz / "resultados" / "mapas-digitos.npz"
    if mapas.is_file():
        m = np.load(mapas)
        x8 = dato()[8][:1797]
        with torch.no_grad():
            dif2 = float(np.abs(torch.sigmoid(banco.mapas(x8)).numpy() - m["sigma"]).max())
        print(f"  σ de los mapas sobre los 1797 dígitos contra feat-ind/resultados/mapas-digitos.npz: diferencia máxima {dif2:.2e}")
        if dif2 > 1e-4:
            raise SystemExit("✗ los mapas no casan con los de feat-ind. Me niego.")
    else:
        print("  (feat-ind/resultados/mapas-digitos.npz no está: se comprueba sólo contra los detectores por separado)")
    estado = {k: v.clone() for k, v in banco.detectores.state_dict().items()}
    torch.save({"detectores": estado, "huella": huella_tensores(estado), "familias": list(C.FAMILIAS), "origen": origen,
                "creado": time.strftime("%Y-%m-%d")}, AQUI / C.INIT_AJUSTE)
    print(f"→ nn/{C.INIT_AJUSTE} · huella {huella_tensores(estado)}")
    return 0


def _det(estado: dict):
    d = C.Detector8(); d.load_state_dict(estado); d.eval()
    return d


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
    try:
        init = cargar_init(); prueba(f"{C.INIT_AJUSTE} carga y su huella casa", True, init["huella"])
    except Exception as exc:                        # noqa: BLE001
        prueba(f"{C.INIT_AJUSTE} carga y su huella casa", False, str(exc))
    with tempfile.TemporaryDirectory() as tmp:
        for m in C.MODELOS:
            a = entrenar(m, 500, 0.02, 1, pasos=50, raiz=Path(tmp))
            b = entrenar(m, 500, 0.02, 1, pasos=50, raiz=Path(tmp) / "b")
            prueba(f"{m}: 50 pasos con N = {a['N']}, determinista", a["acc"] == b["acc"] and a["N"] == 10, f"acc {a['acc']}")
            c = entrenar(m, 500, 0.02, 1, pasos=50, raiz=Path(tmp))
            prueba(f"{m}: si el json ya existe, se salta (no se repite)", c == a)
        f = json.loads(ruta("ajuste13", 500, 0.02, 1, Path(tmp)).read_text())
        prueba("ajuste13: la fase congelada son la mitad de los pasos y deja su acierto", f["pasos_congelado"] == 25 and "acc_congelado" in f,
               f"congelado {f['acc_congelado']} → final {f['acc']}")
    print("el mecanismo funciona." if ok else "✗ algo no funciona")
    return 0 if ok else 1


def resumen() -> int:
    casos = {}
    for m in C.MODELOS:
        filas = [json.loads(f.read_text()) for f in sorted((SALIDA / m).glob("*.json"))]
        if filas:
            casos[C.ETIQUETAS[m]] = filas
            print(f"{C.ETIQUETAS[m]:<40} {len(filas)} de {len(Gn.T_TAMANOS) * len(Gn.FRACCIONES) * len(Gn.SEMILLAS)} entrenamientos")
    hy, hp = huellas_particiones()
    out = {"protocolo": f"{PASOS} pasos de lotes de {LOTE}, Adam, sin selección ni aumento; particiones de nn/ganancia.py",
           "huella_y": hy, "huella_particiones": hp, "casos": casos}
    (RES / "curvas-cnn.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("→ resultados/curvas-cnn.json")
    return 0


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--comprobar", action="store_true"); a.add_argument("--todas", action="store_true")
    a.add_argument("--resumen", action="store_true"); a.add_argument("--exportar-init", action="store_true")
    a.add_argument("--una", nargs=4, metavar=("MODELO", "T", "P", "SEM"))
    a.add_argument("--modelos", default="lenet5,cnn3", help="separados por comas (por defecto, la tanda de las dos CNN)")
    a.add_argument("--procesos", type=int, default=int(os.environ.get("TRABAJO_VCPU") or os.cpu_count() or 1))
    x = a.parse_args()
    if x.comprobar:
        return comprobar()
    if x.exportar_init:
        return exportar_init()
    if x.todas:
        return todas(x.modelos.split(","), x.procesos)
    if x.resumen:
        return resumen()
    if x.una:
        print(_uno((x.una[0], int(x.una[1]), float(x.una[2]), int(x.una[3]))))
        return 0
    a.print_help(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
