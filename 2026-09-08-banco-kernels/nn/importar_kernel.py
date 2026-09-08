#!/usr/bin/env python3
"""Saca el kernel APRENDIDO de un checkpoint de otro experimento y lo deja como .npy.

    python nn/importar_kernel.py --de esq-k --brazos k07 k09 k11
    python nn/importar_kernel.py --de esq-k --brazos k07 --clave conv.weight

POR QUE ESTO EXISTE Y POR QUE ES ASI DE PARANOICO
=================================================
El banco es agnostico al origen del kernel (§1) y los metodos para obtenerlos estan
fuera de alcance (§15). Pero los kernels TIENEN que llegar de algun sitio, y el sitio
mas obvio es un experimento que aprendio uno. Esto hace ese trasvase, y nada mas:
lee un `.pt`, saca un tensor de forma (1, 1, k, k), lo guarda como `.npy` y anota de
donde salio.

⚠⚠ SE PIDE AL REGISTRO POR SU `id`, NUNCA POR RUTA (R16). El nombre de una carpeta de
experimento no es su identidad: se puede renombrar y re-ordenar, y `comprobar.py`
falla a proposito si un fichero de fuera nombra una carpeta ajena. Por eso aqui se
usa `expcnn.por_id("esq-k")` y no una ruta escrita a mano.

⚠ NO SE IMPORTA NI UNA LINEA DE CODIGO DEL OTRO EXPERIMENTO (Regla 0 del repo). Se
lee su artefacto con torch y punto. Si el checkpoint tuviera otra forma, se dice y se
para -- no se adivina.

⚠⚠ Y LA FUGA DE DISTRIBUCION SE COMPRUEBA AQUI (§3.7), porque es el unico sitio por
el que puede entrar. Un kernel aprendido con el MISMO generador que el banco tiene
fuga AUNQUE LAS MUESTRAS SEAN DISTINTAS. No se bloquea -- puede ser justo lo que se
quiere medir -- pero queda escrito en el `.json` que acompanya al kernel, para que
viaje con el hasta el veredicto.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
sys.path.insert(0, str(EXP.parent))
KERNELS = EXP / "kernels"

# La reserva del §3.7 de ESTE banco. Un kernel aprendido sobre datos que la usaron
# tiene fuga: no se le prohibe entrar, se le marca.
RESERVA_FUENTE = "LiberationMono"
RESERVA_INTERLINEADO = (1.45, 1.60)


def _origen(exp_id: str):
    from expcnn import por_id                              # noqa: PLC0415
    e = por_id(exp_id)
    if e is None:
        raise SystemExit(f"✗ no hay ningun experimento con id '{exp_id}'")
    return e


def _fuga(e) -> dict:
    """§3.7: ¿el kernel se aprendio con datos del mismo generador?"""
    receta = e.carpeta / "nn" / "receta.json"
    d = {"mismo_generador": None, "usa_la_reserva": None, "detalle": ""}
    if not receta.is_file():
        d["detalle"] = ("no tiene receta.json: no se puede saber que datos uso. "
                        "Se asume fuga (el criterio prudente)")
        d["mismo_generador"] = True
        d["usa_la_reserva"] = None
        return d
    r = json.loads(receta.read_text(encoding="utf-8"))
    d["mismo_generador"] = True                # una receta => salio del generador
    bloque = (r.get("blocks") or [{}])[0]
    tip = bloque.get("typography", {})
    fuentes = r.get("fonts") or tip.get("font_family")
    lh = tip.get("line_height")
    # Sin `fonts` ni `font_family`, el generador sortea de TODAS las familias
    # registradas -- incluida la reservada.
    usa_fuente = fuentes is None or RESERVA_FUENTE in (
        fuentes if isinstance(fuentes, list) else [fuentes])
    # Sin `line_height`, el defecto del generador es Range(1.15, 1.6), que SOLAPA
    # con la banda reservada.
    banda = (1.15, 1.6) if lh is None else (
        tuple(lh["range"]) if isinstance(lh, dict) and "range" in lh else (lh, lh))
    usa_lh = not (banda[1] < RESERVA_INTERLINEADO[0] or banda[0] > RESERVA_INTERLINEADO[1])
    d["usa_la_reserva"] = bool(usa_fuente or usa_lh)
    d["detalle"] = (
        f"familia: {'sin fijar -> sortea TODAS, incluida ' + RESERVA_FUENTE if fuentes is None else fuentes}"
        f" · interlineado: {banda} contra la reserva {RESERVA_INTERLINEADO}"
        f" -> {'SOLAPA' if usa_lh else 'no solapa'}")
    return d


def importar(exp_id: str, brazos: list[str], clave: str, prefijo: str | None) -> int:
    import torch                                            # noqa: PLC0415
    sys.path.insert(0, str(AQUI))
    from pipeline import comprobar_contrato                 # noqa: PLC0415

    e = _origen(exp_id)
    fuga = _fuga(e)
    KERNELS.mkdir(exist_ok=True)
    pref = prefijo or exp_id.replace("-", "")
    print(f"\nde '{exp_id}' ({e.estado}) · dataset de origen: {e.dataset}\n")
    salidas = []
    for b in brazos:
        f = e.carpeta / "nn" / "pesos" / b / "best.pt"
        if not f.is_file():
            print(f"  ✗ {b}: no existe {f.name} en ese brazo")
            continue
        ck = torch.load(f, map_location="cpu", weights_only=False)
        sd = ck.get("modelo", ck) if isinstance(ck, dict) else ck
        if clave not in sd:
            print(f"  ✗ {b}: el checkpoint no tiene '{clave}'. Tiene: {list(sd)[:8]}")
            continue
        w = sd[clave].detach().cpu().numpy()
        if w.ndim != 4 or w.shape[0] != 1 or w.shape[1] != 1:
            print(f"  ✗ {b}: forma {w.shape}; se esperaba (1, 1, k, k) — 1 canal a 1 (§5.1)")
            continue
        k = w[0, 0].astype(np.float32)
        comprobar_contrato(k)                               # §5.1/§5.2/§5.3
        nombre = f"{pref}-{b}"
        np.save(KERNELS / f"{nombre}.npy", k)
        meta = {
            "nombre": nombre, "origen_id": exp_id, "origen_brazo": b,
            "origen_estado": e.estado, "origen_dataset": e.dataset,
            "clave_del_checkpoint": clave, "epoca": ck.get("epoca") if isinstance(ck, dict) else None,
            "k": int(k.shape[0]), "norma_original": float(np.linalg.norm(k)),
            "suma": float(k.sum()),
            "sha256_16": hashlib.sha256(np.ascontiguousarray(k).tobytes()).hexdigest()[:16],
            "§3.7_fuga_de_distribucion": fuga,
        }
        (KERNELS / f"{nombre}.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        salidas.append(nombre)
        print(f"  ok  {nombre:16} k={k.shape[0]:<3} norma={meta['norma_original']:.4f}  "
              f"suma={meta['suma']:+.4f}  sha={meta['sha256_16']}")

    print(f"\n⚠⚠ §3.7 FUGA DE DISTRIBUCION: "
          f"{'SI' if fuga['usa_la_reserva'] else 'no' if fuga['usa_la_reserva'] is False else 'NO SE SABE'}")
    print(f"   {fuga['detalle']}")
    if fuga["usa_la_reserva"]:
        print("   Estos kernels se aprendieron sobre datos del MISMO generador y que")
        print("   ALCANZAN la reserva del §3.7. No se bloquea la evaluacion -- puede ser")
        print("   justo lo que se quiere medir -- pero el resultado sale OPTIMISTA y esa")
        print("   advertencia tiene que viajar con el hasta el informe.")
    print(f"\n{len(salidas)} kernel(s) en {KERNELS}. Evaluar:")
    for n in salidas:
        print(f"   nn/lanzar.sh kernel kernels/{n}.npy")
    return 0 if salidas else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--de", required=True, help="el ID del experimento de origen (no su ruta)")
    ap.add_argument("--brazos", nargs="+", required=True)
    ap.add_argument("--clave", default="conv.weight")
    ap.add_argument("--prefijo")
    a = ap.parse_args()
    return importar(a.de, a.brazos, a.clave, a.prefijo)


if __name__ == "__main__":
    raise SystemExit(main())
