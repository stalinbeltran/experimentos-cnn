#!/usr/bin/env python3
"""El preflight de este repo: ¿qué hay, qué puede correr, y qué está mal puesto?

Como todos los preflights de este proyecto, **comprueba estado utilizable, no
presencia**, y **crece con cada fallo**: cuando algo se descubra a mitad, la
comprobación se añade aquí en el mismo commit que el arreglo.

    python3 comprobar.py            el informe entero
    python3 comprobar.py --breve    una línea (es lo que lee el ejecutor de Telegram)
    python3 comprobar.py --indice   regenera la tabla del README desde los manifiestos
    python3 comprobar.py --exit0    sale siempre con 0

Sale con 0 si el repo está coherente, 1 si hay algo que arreglar.

⚠ `--exit0` existe por el coordinador, no por gusto: lee cualquier código ≠ 0 como
«el ejecutor falló» y entonces no corre los encargados, así que el informe no
llegaría a Telegram. El veredicto va en el TEXTO, que es donde se lee; fuera de
Telegram el código de salida sigue sirviendo para encadenar. Es la misma decisión
que `cerrable.mjs --exit0` en el coordinador.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# La ÚNICA manipulación de sys.path del repo, y es legítima: este script vive en
# la raíz e importa el paquete de su propio repo. Lo que NO se hace en ningún
# sitio es deducir del disco dónde está OTRO repo — para eso está expcnn.entorno.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from expcnn import (ESTADOS, GASTOS, experimentos, hay_fv, raiz,  # noqa: E402
                    ruta_datos, ruta_fv)
from expcnn.registro import MANIFIESTO  # noqa: E402

TEXTO = {".py", ".md", ".json", ".toml", ".sh", ".mjs", ".txt", ".yaml", ".yml", ".cfg"}
# Excepción declarada: el índice del README nombra las carpetas a propósito, es su
# trabajo — son enlaces. No ata nada porque se REGENERA con `--indice`, así que
# renombrar una carpeta sigue siendo `git mv` + regenerar.
EXENTOS = {"README.md"}
MARCA_INI = "<!-- INDICE: generado por `python3 comprobar.py --indice`. No editar a mano. -->"
MARCA_FIN = "<!-- FIN INDICE -->"
# El nombre de una carpeta de experimento empieza por su fecha. No es estética:
# `_rutas_cableadas` busca el nombre dentro de los ficheros, y un nombre corto
# —el id de un brazo, por ejemplo— aparecería en cualquier docstring o tabla de
# resultados que hable de ese brazo: el aviso que sale siempre. Con la fecha
# delante, el nombre es inequívoco. Renombrar sigue siendo gratis: lo que se
# fija es la FORMA del nombre, no el nombre.
FECHADA = re.compile(r"^\d{4}-\d{2}-\d{2}-.+")


def _problemas_de_manifiestos(exps) -> list[str]:
    malos = []
    vistos: dict[str, str] = {}
    for e in exps:
        d, donde = e.datos, e.rel()
        if e.id in vistos:
            malos.append(f"id duplicado '{e.id}': {vistos[e.id]} y {donde}")
        vistos[e.id] = donde
        if e.estado not in ESTADOS:
            malos.append(f"{donde}: estado '{e.estado}' no es uno de {ESTADOS}")
        if e.gasta not in GASTOS:
            malos.append(f"{donde}: gasta '{e.gasta}' no es uno de {GASTOS}")
        for campo in ("titulo", "pregunta", "creado"):
            if not d.get(campo):
                malos.append(f"{donde}: falta '{campo}' en {MANIFIESTO}")
        if not FECHADA.match(e.carpeta.name):
            malos.append(
                f"{donde}: la carpeta tiene que empezar por su fecha "
                f"(`<AAAA-MM-DD>-<nombre>`); '{e.carpeta.name}' no. Un nombre corto "
                f"aparece por casualidad dentro de otros textos y llena de falsos "
                f"avisos la comprobación que mantiene barato re-ordenar"
            )
        if e.entrada:
            if not (e.carpeta / e.entrada).exists():
                malos.append(f"{donde}: `entrada` apunta a {e.entrada}, que no existe")
            # Contrato con el freno del coordinador (cerrable.mjs:137): lo que
            # entrena largo se tiene que llamar así, o el veredicto "se puede
            # apagar este server" no lo ve y dice CERRABLE con 37 épocas vivas.
            if e.gasta == "entrena-local" and Path(e.entrada).name != "entrenar_local.py":
                malos.append(
                    f"{donde}: gasta='entrena-local' pero la entrada es "
                    f"'{Path(e.entrada).name}'. El freno (cerrable.mjs:137) casa "
                    f"`entrenar_local.py`: con otro nombre, no lo ve"
                )
        elif e.gasta != "no":
            malos.append(f"{donde}: gasta='{e.gasta}' pero no declara `entrada`")
    return malos


def _rutas_cableadas(exps) -> list[str]:
    """R16: nada fuera de una carpeta de experimento puede nombrarla.

    Es lo que hace que re-ordenar sea `git mv` y nada más. Si esto falla, el
    nombre de la carpeta ha vuelto a ser identidad."""
    nombres = {e.carpeta.name: e.rel() for e in exps}
    if not nombres:
        return []
    malos = []
    for f in raiz().rglob("*"):
        if not f.is_file() or ".git" in f.parts or f.suffix not in TEXTO:
            continue
        if f.name in EXENTOS:
            continue
        try:
            txt = f.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            continue
        for nombre, donde in nombres.items():
            if nombre in f.parts:
                continue
            # Por TOKEN, no por subcadena: una carpeta no puede casar dentro del
            # nombre de otra que la tenga por prefijo, que es otra carpeta y otra
            # cosa. (Y por eso los ejemplos de estos comentarios no son nombres
            # verosímiles: este fichero también pasa por la comprobación.)
            if re.search(rf"(?<![\w-]){re.escape(nombre)}(?![\w-])", txt):
                malos.append(f"{f.relative_to(raiz())} nombra la carpeta '{nombre}' ({donde})")
    return malos


def _indice(exps) -> int:
    """Reescribe la tabla del README entre sus marcas, leyendo los manifiestos.

    Transcribir una tabla a mano es como nacen los números que nadie puede
    auditar — la misma razón por la que `comun/serie.py` imprime la suya."""
    readme = raiz() / "README.md"
    txt = readme.read_text(encoding="utf-8")
    if MARCA_INI not in txt or MARCA_FIN not in txt:
        print(f"✗ {readme.name} no tiene las marcas del índice; no toco nada")
        return 1
    if exps:
        filas = ["| experimento | estado | qué pregunta |", "|---|---|---|"]
        for e in sorted(exps, key=lambda x: x.rel()):
            filas.append(
                f"| [`{e.rel()}/`]({e.rel()}/) `{e.id}` | {e.estado} | "
                f"{e.datos.get('pregunta', e.titulo)} |"
            )
        tabla = "\n".join(filas)
    else:
        tabla = "_Todavía no hay ningún experimento._"
    ini, fin = txt.index(MARCA_INI) + len(MARCA_INI), txt.index(MARCA_FIN)
    readme.write_text(txt[:ini] + "\n\n" + tabla + "\n\n" + txt[fin:], encoding="utf-8")
    print(f"índice regenerado: {len(exps)} experimento(s)")
    return 0


def main() -> int:
    breve = "--breve" in sys.argv
    exps = experimentos()
    if "--indice" in sys.argv:
        return _indice(exps)
    malos = _problemas_de_manifiestos(exps) + _rutas_cableadas(exps)
    corribles = [e for e in exps if not e.usa_fv or hay_fv()]
    bloqueados = [e for e in exps if e.usa_fv and not hay_fv()]
    vivos = [e for e in exps if e.estado == "corriendo"]

    if breve:
        estado = "⚠" if malos else "ok"
        trozos = [f"{len(exps)} experimento(s)"]
        if vivos:
            trozos.append(f"{len(vivos)} corriendo: " + ", ".join(x.id for x in vivos))
        if bloqueados:
            trozos.append(f"{len(bloqueados)} necesitan foveal-vision (no está)")
        if malos:
            trozos.append(f"{len(malos)} problema(s)")
        print(f"{estado} experimentos-cnn — " + " · ".join(trozos))
        return 1 if malos else 0

    print(f"\nrepo: {raiz()}")
    fv = ruta_fv()
    print(f"foveal-vision: {fv if fv else '— no está (los experimentos autónomos corren igual)'}")
    datos = ruta_datos()
    print(f"foveal-vision-data: {datos if datos else '— no está (el dato de entrada se lee de ahí)'}\n")

    if not exps:
        print("No hay ningún experimento todavía.")
        print("Para crear uno: copia experimento.ejemplo.json a")
        print("  <fecha>-<nombre>/experimento.json  y rellénalo. La forma está en CLAUDE.md.\n")
    else:
        for e in exps:
            marca = "⛔" if (e.usa_fv and not hay_fv()) else "  "
            print(f"{marca} {e.id:<28} [{e.estado}] gasta={e.gasta:<14} {e.rel()}")
            print(f"     {e.titulo}")
        print(f"\n{len(corribles)} de {len(exps)} se pueden correr aquí ahora mismo.")

    if malos:
        print(f"\n{len(malos)} problema(s):")
        for m in malos:
            print(f"  ✗ {m}")
        return 1
    print("\nTodo coherente.")
    return 0


if __name__ == "__main__":
    codigo = main()
    raise SystemExit(0 if "--exit0" in sys.argv else codigo)
