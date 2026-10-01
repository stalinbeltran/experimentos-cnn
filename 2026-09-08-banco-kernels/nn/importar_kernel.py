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

⚠ DE DONDE SE SABE QUE DATOS VIO EL KERNEL (desde el 2026-10-01): primero la
`nn/receta.json` del experimento de origen, si genero el mismo sus datos; si no la
tiene, el MANIFIESTO del dataset que ese experimento DECLARA en su `experimento.json`
(`factores.fuentes` y `factores.interlineado`). Hasta ese dia solo existia la receta,
y un experimento que CONSUME un dataset publicado -- lo normal en este repo -- salia
siempre como «se asume fuga» aunque su dataset la respetara. Sin ninguna de las dos se
sigue asumiendo fuga. Se prueba con `--probar` (las tres ramas).
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


def _usa_reserva(fuentes, lh) -> tuple[bool, str]:
    """La regla del §3.7 sobre lo que declara una receta o un manifiesto.

    Sin familias declaradas, el generador sortea de TODAS las registradas -- incluida
    la reservada. Sin interlineado, su defecto es Range(1.15, 1.6), que SOLAPA con la
    banda reservada. Omitir no es neutro: es usarlas."""
    lista = None if fuentes is None else (fuentes if isinstance(fuentes, list) else [fuentes])
    usa_fuente = lista is None or RESERVA_FUENTE in lista
    if lh is None:
        banda = (1.15, 1.6)
    elif isinstance(lh, dict) and "range" in lh:
        banda = tuple(lh["range"])
    elif isinstance(lh, (list, tuple)) and len(lh) == 2:
        banda = tuple(lh)
    else:
        banda = (lh, lh)
    usa_lh = not (banda[1] < RESERVA_INTERLINEADO[0] or banda[0] > RESERVA_INTERLINEADO[1])
    detalle = (
        f"familia: {'sin fijar -> sortea TODAS, incluida ' + RESERVA_FUENTE if lista is None else lista}"
        f" · interlineado: {banda} contra la reserva {RESERVA_INTERLINEADO}"
        f" -> {'SOLAPA' if usa_lh else 'no solapa'}")
    return bool(usa_fuente or usa_lh), detalle


def _fuga(carpeta: Path, dataset: str | None) -> dict:
    """§3.7: ¿el kernel se aprendio con datos que alcanzan la reserva?

    Dos fuentes, en este orden, y las dos contestan lo mismo -- QUE DATOS VIO:
      1. `nn/receta.json` del experimento de origen, si genero sus datos el mismo;
      2. el MANIFIESTO del dataset que declara (`factores.fuentes`,
         `factores.interlineado`), si consume uno publicado. Es la fuente unica de que
         datos vio: lo escribio quien los genero (R15: un solo mando por hecho).
    Sin ninguna de las dos se asume fuga, que es el criterio prudente."""
    d = {"mismo_generador": None, "usa_la_reserva": None, "detalle": "", "fuente": None}
    receta = carpeta / "nn" / "receta.json"
    if receta.is_file():
        r = json.loads(receta.read_text(encoding="utf-8"))
        tip = (r.get("blocks") or [{}])[0].get("typography", {})
        d["mismo_generador"] = True                # una receta => salio del generador
        d["usa_la_reserva"], d["detalle"] = _usa_reserva(
            r.get("fonts") or tip.get("font_family"), tip.get("line_height"))
        d["fuente"] = "nn/receta.json del experimento de origen"
        return d
    if dataset:
        from expcnn import ruta_dataset                    # noqa: PLC0415
        p = ruta_dataset(dataset)
        if p is not None:
            m = json.loads((p / "manifiesto.json").read_text(encoding="utf-8"))
            fac = m.get("factores") or {}
            d["mismo_generador"] = True            # un manifiesto con factores => generador
            d["usa_la_reserva"], d["detalle"] = _usa_reserva(
                fac.get("fuentes"), fac.get("interlineado"))
            d["fuente"] = f"manifiesto del dataset declarado '{dataset}'"
            return d
    d["mismo_generador"] = True
    d["detalle"] = ("ni receta.json ni un dataset publicado que leer: no se puede saber "
                    "que datos uso. Se asume fuga (el criterio prudente)")
    d["fuente"] = "ninguna"
    return d


def _probar() -> int:
    """Las tres ramas de `_fuga`, contra carpetas y datasets de mentira."""
    import os                                               # noqa: PLC0415
    import tempfile                                         # noqa: PLC0415
    fallos = []

    def caso(que, obtenido, esperado):
        bien = obtenido == esperado
        print(f"  {'ok ' if bien else 'FALLA'}  {que}: {obtenido}")
        if not bien:
            fallos.append(que)

    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        os.environ["EXPCNN_DATOS"] = str(t / "datos")

        def experimento(nombre, receta=None):
            c = t / nombre
            (c / "nn").mkdir(parents=True)
            if receta is not None:
                (c / "nn" / "receta.json").write_text(json.dumps(receta))
            return c

        def dataset(nombre, factores):
            p = t / "datos" / "experimentos-cnn" / nombre
            p.mkdir(parents=True)
            (p / "manifiesto.json").write_text(json.dumps({"factores": factores}))
            return nombre

        limpio = {"fuentes": ["DejaVuSans"], "interlineado": [1.10, 1.42]}
        sucio = {"fuentes": ["DejaVuSans", "LiberationMono"], "interlineado": [1.10, 1.42]}
        ds_limpio, ds_sucio = dataset("limpio", limpio), dataset("sucio", sucio)
        ds_sin_lh = dataset("sin-lh", {"fuentes": ["DejaVuSans"]})

        rec_limpia = {"fonts": ["DejaVuSans"], "blocks": [
            {"typography": {"line_height": {"range": [1.1, 1.42]}}}]}
        rec_sucia = {"blocks": [{"typography": {"line_height": {"range": [1.1, 1.42]}}}]}

        caso("receta que respeta", _fuga(experimento("a", rec_limpia), None)["usa_la_reserva"], False)
        caso("receta sin `fonts` (sortea todas)", _fuga(experimento("b", rec_sucia), None)["usa_la_reserva"], True)
        caso("SIN receta, manifiesto que respeta", _fuga(experimento("c"), ds_limpio)["usa_la_reserva"], False)
        caso("SIN receta, manifiesto con LiberationMono", _fuga(experimento("d"), ds_sucio)["usa_la_reserva"], True)
        caso("SIN receta, manifiesto sin interlineado", _fuga(experimento("e"), ds_sin_lh)["usa_la_reserva"], True)
        caso("la receta MANDA sobre el manifiesto",
             _fuga(experimento("f", rec_sucia), ds_limpio)["usa_la_reserva"], True)
        sin = _fuga(experimento("g"), "no-publicado")
        caso("sin receta ni dataset publicado: se asume fuga",
             (sin["mismo_generador"], sin["usa_la_reserva"], sin["fuente"]), (True, None, "ninguna"))
        caso("sin receta y sin dataset declarado: igual",
             _fuga(experimento("h"), None)["fuente"], "ninguna")
    print(f"\n{'TODO OK' if not fallos else str(len(fallos)) + ' FALLO(S)'}")
    return 1 if fallos else 0


def importar(exp_id: str, brazos: list[str], clave: str, prefijo: str | None) -> int:
    import torch                                            # noqa: PLC0415
    sys.path.insert(0, str(AQUI))
    from pipeline import comprobar_contrato                 # noqa: PLC0415

    e = _origen(exp_id)
    fuga = _fuga(e.carpeta, e.dataset)
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
    print(f"   (de: {fuga['fuente']})")
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
    ap.add_argument("--de", help="el ID del experimento de origen (no su ruta)")
    ap.add_argument("--brazos", nargs="+")
    ap.add_argument("--clave", default="conv.weight")
    ap.add_argument("--prefijo")
    ap.add_argument("--probar", action="store_true",
                    help="comprueba las tres ramas de la deteccion de fuga y sale")
    a = ap.parse_args()
    if a.probar:
        return _probar()
    if not a.de or not a.brazos:
        ap.error("hacen falta --de y --brazos (o --probar)")
    return importar(a.de, a.brazos, a.clave, a.prefijo)


if __name__ == "__main__":
    raise SystemExit(main())
