"""EMULA los compositores del revisor: guarda en el almacén un `.pt` por receta con la HUELLA correcta pero redes SIN ENTRENAR
(pesos al azar), para probar que un dev nuevo los RESTAURA sin esperar. Pedido por el dueño el 2026-10-16 («emula todos los
compositores, no los entrenes, pues es una prueba; voy a destruir el dev»).

    .venv/bin/python 2026-10-11-rizo-dos/web/emular.py [--seco]

Qué deja: revisor/<receta>.pt (resumen con "emulado": …, val ≈ 0,1 porque son redes al azar), revisor/recetas.json y la caché
de features base (revisor/cache/caract-<hash>-3x3.npz), commiteado y empujado. ⚠ Un `.pt` emulado se PISA con el de verdad
en cuanto ese compositor se entrene (misma receta = mismo fichero): /api/reentrenar en la app, o borrar el .pt y reiniciar.
No calcula las features de extra (eso es entrenar, y no se entrena)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import servidor as S  # noqa: E402


def emular(c: S.Compositor, datos: S.Datos, marca: str):
    todos, Xtr, _ = datos.features(c.celdas); Xtr, Xva = c.corta(Xtr, datos), c.corta(todos[datos.va], datos)
    c.n = Xtr.shape[1]; c.redes = []; preds = []; finos = []; accs = []
    y = c.y_fino(datos)[datos.va] if c.partido else datos.y[datos.va]           # § 107: el 4 y/o el 1 partidos
    for s in c.semillas:
        torch.manual_seed(s); mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        red = torch.nn.Sequential(torch.nn.Linear(c.n, S.OCULTA), torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(S.OCULTA, c.salidas)); red.eval()
        c.redes.append((red, mu, sd))
        with torch.no_grad():
            lg = red(torch.from_numpy((Xva - mu) / sd).float()).numpy()
        pf = lg.argmax(1); finos.append(pf); p = pf if c.partido else c.juntar(pf); preds.append(p); accs.append(float((p == y).mean()))
        if s == c.semillas[0]:
            c.logits_va = lg
    c.preds = np.stack(preds); c.finos = np.stack(finos); c.veces = (c.preds != y).sum(0); p0 = preds[0]; c.segundos = 0.0
    c.resumen = {"nombre": c.nombre, "bloques": c.bloques, "celdas": c.celdas, "semillas": list(c.semillas), "n": int(c.n),
                 "val": round(float(np.mean(accs)), 4), "val_rango": [round(min(accs), 4), round(max(accs), 4)],
                 "fallos_s0": int((p0 != y).sum()), "fallan_en_todas": int((c.veces == len(c.semillas)).sum()), "confusiones": {},
                 "cuota": {b: 0.0 for b in c.bloques}, "segundos": 0.0, "sin_posicion": [b for b in c.bloques if b in S.SIN_POSICION],
                 "sub4": c.sub4, "sub4_resumen": None, "sub1": c.sub1, "sub1_resumen": None, "clases": c.clases_finas(), "extra": c.extra, "n_train": 0, "emulado": marca}
    c.estado = "listo"


def main() -> int:
    seco = "--seco" in sys.argv; alm = S.almacen_revisor()
    if alm is None:
        print("✗ sin repo de datos (foveal-vision-data): no hay almacén donde emular"); return 2
    recetas = S.App.recetas_guardadas(None)                       # no arranca la App: arrancarla encola y ENTRENA
    marca = f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} · redes SIN ENTRENAR (web/emular.py): prueba de restauración"
    print(f"emular {len(recetas)} compositor(es) → {alm}" + (" (SECO)" if seco else "")); t = time.time()
    datos = S.Datos(); datos.features(3)                           # ~30 s la primera vez; va a la caché local y al almacén
    comps = []
    for nombre, r in recetas.items():
        c = S.Compositor(nombre, r["bloques"], r.get("celdas", 3), r.get("semillas", S.SEMILLAS), r.get("sub4", False), r.get("extra", False), sub1=r.get("sub1", False))
        emular(c, datos, marca); comps.append(c)
        print(f"  {c.nombre}: n={c.n} · val {c.resumen['val']} (al azar) → {c.fichero(alm).name}")
        if not seco:
            c.guardar(alm, c.huella(datos))
    if seco:
        print("SECO: no se ha escrito nada"); return 0
    app = S.App.__new__(S.App); app.comps = {c.nombre: c for c in comps}; app.guardar_recetas()
    S.empujar_almacen(f"rizo-2 revisor: {len(comps)} compositor(es) EMULADOS (sin entrenar) para probar la restauración en el dev siguiente")
    print(f"listo en {time.time() - t:.0f} s · huella {comps[0].huella(datos)}"); return 0


if __name__ == "__main__":
    sys.exit(main())
