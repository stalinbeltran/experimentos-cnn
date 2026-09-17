#!/usr/bin/env python3
"""Las comprobaciones de `gauss-p`. Corre sola y sale != 0 si algo falla.

    python nn/probar.py

QUE SE COMPRUEBA Y POR QUE ESO
==============================
La mayoria de este experimento no puede fallar a gritos. Es un visor: si ensenya
la imagen equivocada, se ve igual de bien. Asi que lo que se prueba es lo que
FALLARIA EN SILENCIO:

 1. **el acuerdo con el banco** -- que esta copia del §6 no se haya desviado del
    original. Es el unico fallo que haria elegir mirando una cosa y medir otra;
 2. **la reserva del §3.7** -- que no se cuele una muestra reservada, y que si el
    manifiesto no declara la reserva se NIEGUE en vez de suponer que no hay;
 3. **la puerta del banco** -- que el apreton de manos rechace una huella que no
    coincide, y que rechazando no escriba nada;
 4. **que el dato privado no se sirva sin token**;
 5. **que sin dataset se niegue ANTES de abrir el puerto** (R2), no a mitad.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
EXP = AQUI.parent
REPO = EXP.parent
PY = REPO / ".venv" / "bin" / "python"

# ⚠ EL BANCO SE LOCALIZA POR SU `id`, NUNCA POR EL NOMBRE DE SU CARPETA (R16 del
# repo, que `comprobar.py` hace cumplir). Asi re-ordenar carpetas es un `git mv`
# y nada mas, y este fichero no se entera. Lo cazo el propio `comprobar.py` el
# 2026-09-17, con la ruta escrita a mano.
sys.path.insert(0, str(REPO))
try:
    from expcnn import por_id                              # noqa: E402
    BANCO_K = por_id("banco-k").carpeta
except Exception:                                          # noqa: BLE001
    BANCO_K = REPO / "no-esta-el-banco"

# ⚠ MIS MODULOS PRIMERO, Y POR RUTA PROPIA. `banco-k/nn/` tiene su propio
# `muestras.py`, asi que meter su directorio en `sys.path` ANTES de importar el
# mio lo ensombrece y la prueba del §3.7 se salta sola -- que es exactamente un
# fallo silencioso disfrazado de «SALTA». Lo cazo esta misma suite el 2026-09-17.
sys.path.insert(0, str(AQUI))
import banco                                          # noqa: E402
import forma                                          # noqa: E402
import muestras as mis_muestras                       # noqa: E402

fallos: list[str] = []
saltados: list[str] = []


def _cargar(nombre: str, ruta: Path):
    """Carga un modulo de `banco-k` SIN meter su directorio en `sys.path`.

    Por que asi: `banco-k/nn/` tiene ficheros con los mismos nombres que los de
    aqui (`muestras.py`), y un `sys.path.insert` los ensombrece. La prueba que se
    salta por un choque de nombres se lee como «no aplica» y en realidad es «no se
    comprobo».
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[nombre] = mod
    spec.loader.exec_module(mod)
    return mod


def ok(que: str, bien: bool, detalle: str = "") -> bool:
    print(f"  {'ok   ' if bien else 'FALLA'}  {que}" + (f"   {detalle}" if detalle else ""))
    if not bien:
        fallos.append(que)
    return bien


def saltar(que: str, motivo: str) -> None:
    print(f"  SALTA  {que}   {motivo}")
    saltados.append(f"{que}: {motivo}")


# ------------------------------------------------------- 1. acuerdo con el banco
def acuerdo_con_el_banco() -> None:
    """La prueba que hace honesta la copia. Sin ella, `banco.py` es una promesa."""
    print("\n1. El §6 copiado coincide con el de `banco-k`\n")
    if not (BANCO_K / "nn" / "pipeline.py").is_file():
        return saltar("acuerdo con banco-k", f"no esta {BANCO_K} — la copia queda SIN comprobar")

    orig_pipe = _cargar("bancok_pipeline", BANCO_K / "nn" / "pipeline.py")
    orig_kern = _cargar("bancok_kernels", BANCO_K / "nn" / "kernels.py")

    # las constantes que fijan la geometria
    for nom in ("MARCO", "DESCARTE", "FINAL", "K_MAX"):
        ok(f"constante {nom}", getattr(banco, nom) == getattr(orig_pipe, nom),
           f"{getattr(banco, nom)}")

    # gauss: el mismo array bit a bit, en todo el rango del contrato
    peor = 0.0
    for k in banco.ks_validos():
        for s in (0.3, 0.7, 1.0, k / 6.0, 2.5, 4.0):
            d = float(np.abs(banco.gauss(k, s) - orig_kern.gauss(k, s)).max())
            peor = max(peor, d)
    ok("gauss(k, sigma) identico en todo el contrato", peor == 0.0,
       f"dif max {peor:.1e} sobre {len(banco.ks_validos())*6} pares")

    # aplicar(): el mismo resultado bit a bit sobre datos sinteticos
    rng = np.random.default_rng(0)
    x = (rng.random((4, banco.MARCO, banco.MARCO)).astype(np.float32) * 4080)
    peor = 0.0
    for k in [None] + banco.ks_validos():
        ker = None if k is None else banco.gauss(k, max(0.4, k / 6.0))
        d = float(np.abs(banco.aplicar(x, ker) - orig_pipe.aplicar(x, ker)).max())
        peor = max(peor, d)
    ok("aplicar() identico (identidad y los 9 `k`)", peor == 0.0, f"dif max {peor:.1e}")

    # recorte_de(): la tabla del §6.2 entera
    igual = all(banco.recorte_de(k) == orig_pipe.recorte_de(k)
                for k in [None] + banco.ks_validos())
    ok("recorte_de() identico (tabla del §6.2)", igual)

    # normalizar + estandarizar
    b = rng.standard_normal((9, 9)).astype(np.float32)
    ok("normalizar_kernel() identico",
       float(np.abs(banco.normalizar_kernel(b) - orig_pipe.normalizar_kernel(b)).max()) == 0.0)

    # y la huella, contra el kernel QUE EL BANCO EVALUO DE VERDAD
    res = BANCO_K / "resultados" / "gauss-s3" / "config.json"
    if res.is_file():
        sha_real = json.loads(res.read_text()).get("kernel_sha256_16")
        ok("la huella coincide con el `gauss` YA EVALUADO en el banco",
           forma.huella(banco.gauss(9, 1.5)) == sha_real,
           f"{forma.huella(banco.gauss(9, 1.5))} == {sha_real}")
    else:
        saltar("huella contra el gauss evaluado", "no hay resultados/gauss-s3/config.json")


# ------------------------------------------------------- 2. la reserva del §3.7
def reserva() -> None:
    print("\n2. La reserva del §3.7 se cumple, y se NIEGA si no la puede leer\n")
    Muestras = mis_muestras.Muestras
    comprobar_reserva = mis_muestras.comprobar_reserva
    reserva_del_manifiesto = mis_muestras.reserva_del_manifiesto
    try:
        m = Muestras()
    except Exception as e:                                   # noqa: BLE001
        return saltar("§3.7", f"no esta el dataset publicado: {e}")

    r = m.resumen()
    ok("hay muestras libres tras descartar la reserva", r["n_libres"] > 0,
       f"{r['n_libres']} libres de {r['n_particion']}")
    ok("se descarta de verdad (la reserva no esta vacia)", r["n_reservadas"] > 0,
       f"{r['n_reservadas']} descartadas")
    ok("la particion es `train`, no `eval`", m.particion == "train",
       "elegir mirando `eval` seria fuga del conjunto que declara")

    malos = comprobar_reserva(m.meta, m.fuente_res, m.dens_res)
    ok("ninguna muestra cargada cae en la reserva", not malos,
       "; ".join(malos[:2]) if malos else f"comprobadas {len(m.meta)}")

    elegidas = [m.meta[j] for j in m.elegir(10)]
    malos = comprobar_reserva(elegidas, m.fuente_res, m.dens_res)
    ok("ninguna de las 10 que se ENSENYAN cae en la reserva", not malos)
    ok("la eleccion de muestras es estable (misma semilla, mismas muestras)",
       list(m.elegir(10)) == list(m.elegir(10)))

    # y si el manifiesto no la declara, se NIEGA
    try:
        reserva_del_manifiesto({"nombre": "inventado"})
        ok("un manifiesto SIN reserva se rechaza", False, "no se quejo")
    except RuntimeError as e:
        ok("un manifiesto SIN reserva se rechaza", "§3.7" in str(e))


# ------------------------------------------------------- 3. la puerta del banco
def puerta() -> None:
    print("\n3. El apreton de manos de la puerta del banco\n")
    kern = BANCO_K / "nn" / "kernels.py"
    if not kern.is_file() or not PY.is_file():
        return saltar("puerta del banco", "falta banco-k o el venv")

    def correr(*args):
        return subprocess.run([str(PY), "nn/kernels.py", *args], cwd=BANCO_K,
                              capture_output=True, text=True, timeout=120)

    antes = set(p.name for p in (BANCO_K / "kernels").glob("gauss-k*"))

    r = correr("--gauss", "--k", "9", "--sigma", "1.234",
               "--guardar", "--esperado", "0000000000000000")
    ok("una huella que no coincide se RECHAZA", r.returncode == 2,
       f"exit={r.returncode}")
    ok("y rechazando no escribe NADA",
       set(p.name for p in (BANCO_K / "kernels").glob("gauss-k*")) == antes)

    r = correr("--gauss", "--k", "9")
    ok("--gauss sin --sigma se rechaza", r.returncode != 0,
       "el defecto k/6 es justo el punto ya evaluado")

    r = correr("--gauss", "--k", "8", "--sigma", "1.0", "--guardar",
               "--esperado", forma.huella(banco.gauss(8, 1.0)))
    ok("un `k` PAR se rechaza (§5.3)", r.returncode != 0)

    r = correr("--gauss", "--k", "9", "--sigma", "1.5")
    ok("sin --guardar no escribe (modo seco)",
       r.returncode == 0 and "seco" in r.stdout
       and set(p.name for p in (BANCO_K / "kernels").glob("gauss-k*")) == antes)

    # el camino que SI escribe, en un directorio de usar y tirar
    with tempfile.TemporaryDirectory() as tmp:
        kmod = _cargar("bancok_kernels_w", BANCO_K / "nn" / "kernels.py")
        guardado, kmod.KERNELS = kmod.KERNELS, Path(tmp)
        try:
            rc = kmod.guardar_gauss(9, 1.234, forma.huella(banco.gauss(9, 1.234)))
            escrito = Path(tmp) / "gauss-k09-s1.234.npy"
            ok("con la huella correcta SI escribe", rc == 0 and escrito.is_file())
            if escrito.is_file():
                ok("y el `.npy` es el que se miro",
                   forma.huella(np.load(escrito)) == forma.huella(banco.gauss(9, 1.234)))
                meta = json.loads((Path(tmp) / "gauss-k09-s1.234.json").read_text())
                ok("la receta viaja con el kernel (se puede regenerar)",
                   meta["receta"]["sigma"] == 1.234 and meta["receta"]["k"] == 9)
                ok("y el aviso de sesgo viaja hasta el veredicto",
                   "§3.7" in json.dumps(meta, ensure_ascii=False))
                rc2 = kmod.guardar_gauss(9, 1.234, None)
                ok("no lo reescribe si ya esta y es el mismo", rc2 == 0)
        finally:
            kmod.KERNELS = guardado


# ------------------------------------------------------- 4. la forma
def forma_del_kernel() -> None:
    print("\n4. La forma: `sigma` es el mando, `k` solo es la ventana\n")
    ok("sigma <= 0 se rechaza",
       _lanza(lambda: banco.gauss(9, 0.0), ValueError))
    ok("el `k/6` del banco es la gaussiana mas ancha que cabe en k",
       all(forma.truncamiento(k, banco.sigma_banco(k)) <= forma.TRUNCAMIENTO_AVISO
           for k in banco.ks_validos()),
       "o sea que para ensanchar hace falta subir `k`, y `k` <= 19")

    # a igual sigma, un k mas ancho de lo necesario da el MISMO filtro
    peor = 0.0
    for s in (0.5, 1.0, 1.5):
        km = forma.k_minimo(s)
        g0 = banco.normalizar_kernel(banco.gauss(km, s))
        for k in [x for x in banco.ks_validos() if x > km]:
            gk = banco.normalizar_kernel(banco.gauss(k, s))
            pad = np.zeros((k, k), np.float32)
            o = (k - km) // 2
            pad[o:o + km, o:o + km] = g0
            peor = max(peor, float(np.linalg.norm(gk - pad)))
    ok("con k de sobra sale practicamente el mismo filtro", peor < 5e-3,
       f"peor L2 dif {peor:.1e} — por eso el eje util es `sigma`")

    # y el limite: sigma grande no cabe en el contrato
    ok("un sigma que no cabe ni en k=19 se detecta", forma.k_minimo(5.0) is None,
       f"trunca {forma.truncamiento(19, 5.0):.0%} con k=19 (§5.2, congelado)")
    d = forma.describir(19, 5.0)
    ok("y `describir` lo marca", d["trunca_mucho"] and d["k_minimo"] is None)

    # nada de lo que se ensenya puntua un sigma (R13)
    prohibidas = {"mejor", "peor", "puntuacion", "score", "ranking", "calidad"}
    ok("la pantalla no puntua ningun sigma (R13)",
       not (prohibidas & set(forma.describir(9, 1.5))),
       "el criterio que declara es el del banco, escrito antes de mirar")


def _lanza(f, exc) -> bool:
    try:
        f()
        return False
    except exc:
        return True


# ------------------------------------------------------- 5. la app
def app_web() -> None:
    print("\n5. La app: sin token no sirve dato privado, y sin dataset no abre puerto\n")
    if not PY.is_file():
        return saltar("app", "no hay venv")

    # R2: sin dataset se niega ANTES de abrir el puerto
    with tempfile.TemporaryDirectory() as vacio:
        env = {**os.environ, "EXPCNN_DATOS": vacio}
        r = subprocess.run([str(PY), str(AQUI / "app.py"), "--puerto", "8099"],
                           capture_output=True, text=True, timeout=90, env=env)
        salida = r.stdout + r.stderr
        ok("sin dataset se NIEGA (R2: antes de empezar, no a mitad)",
           r.returncode != 0, f"exit={r.returncode}")
        ok("y dice como se arregla",
           "publicad" in salida.lower() or "dataset" in salida.lower())

    # el token: el dato es privado
    puerto = 8098
    p = subprocess.Popen([str(PY), str(AQUI / "app.py"), "--puerto", str(puerto),
                          "--host", "127.0.0.1"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        url, t0 = None, time.time()
        while time.time() - t0 < 90:
            linea = p.stdout.readline()
            if not linea:
                break
            if "http://" in linea:
                url = linea.strip()
                break
        if not url:
            p.kill()
            return saltar("token", "la app no llego a levantar")
        base = f"http://127.0.0.1:{puerto}"
        token = url.split("?t=")[1]

        ok("sin token, 403", _codigo(f"{base}/api/estado") == 403)
        ok("con un token equivocado, 403", _codigo(f"{base}/api/estado?t=noesta") == 403)
        ok("con el token bueno, 200", _codigo(f"{base}/api/estado?t={token}") == 200)
        ok("una imagen sin token tampoco sale",
           _codigo(f"{base}/img.png?i=0&modo=identidad") == 403)
        ok("/salud contesta sin token (para el `estado` del ejecutor)",
           _codigo(f"{base}/salud") == 200)

        with urllib.request.urlopen(f"{base}/api/estado?t={token}", timeout=30) as resp:
            e = json.loads(resp.read())
        ok("el estado dice cuantas descarto por §3.7", e["n_reservadas"] > 0,
           f"{e['n_reservadas']} de {e['n_particion']}")

        # ⚠ `i=-1` DEVOLVIA 200 CON OTRA MUESTRA. numpy indexa por el final, asi
        # que un indice negativo enseyaba la ultima en vez de fallar: en un visor
        # cuyo unico trabajo es enseyar lo que se pidio, eso es elegir un `sigma`
        # mirando una muestra que no es la de la cabecera. Medido el 2026-09-17.
        img = f"{base}/img.png?t={token}&modo=filtrado"
        ok("un indice NEGATIVO se rechaza (no devuelve otra muestra)",
           _codigo(f"{img}&i=-1&k=9&sigma=1.5") == 400)
        ok("un indice fuera de rango se rechaza",
           _codigo(f"{img}&i=999&k=9&sigma=1.5") == 400)
        ok("un `k` PAR se rechaza tambien aqui (§5.3)",
           _codigo(f"{img}&i=0&k=8&sigma=1.5") == 400)
        ok("un `k` fuera del contrato se rechaza (§5.2)",
           _codigo(f"{img}&i=0&k=99&sigma=1.5") == 400)
        ok("sigma <= 0 se rechaza", _codigo(f"{img}&i=0&k=9&sigma=0") == 400)
        ok("basura donde va un numero se rechaza",
           _codigo(f"{img}&i=abc&k=9&sigma=1.5") == 400)
        ok("y lo valido sigue dando 200",
           _codigo(f"{img}&i=0&k=9&sigma=1.5") == 200)
        ok("la app sigue viva despues de todo eso",
           _codigo(f"{base}/salud") == 200,
           "un error del cliente no puede tumbar el servicio")
        ok("y no filtra el token en el estado", "t" not in e and token not in json.dumps(e))
    finally:
        p.kill()
        p.wait(timeout=10)


def _codigo(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:                                        # noqa: BLE001
        return 0


# ------------------------------------------------------- 6. los modos y el comando
def modos_y_comando() -> None:
    """Dos fallos que existieron de verdad el 2026-09-17 y que no gritaban."""
    print("\n6. Los modos, y que el comando que imprime la app EXISTA\n")
    if not PY.is_file():
        return saltar("modos", "no hay venv")

    def correr(*args, **kw):
        return subprocess.run([str(PY), str(AQUI / "app.py"), *args],
                              capture_output=True, text=True, timeout=120, **kw)

    r = correr("forma")
    ok("`forma` imprime su tabla y no repara el parseo de la app",
       r.returncode == 0 and "k minimo" in r.stdout,
       "estuvo roto: `forma._main()` releia sys.argv y daba «unrecognized arguments»")

    r = correr("borra", "todo")
    ok("un modo desconocido SE NIEGA (no hace nada por defecto)",
       r.returncode == 2 and "no sé qué es" in r.stdout)
    ok("y dice qué modos hay", "url" in r.stdout and "estado" in r.stdout)

    # ⚠ EL COMANDO QUE LA APP ENSENYA TIENE QUE EXISTIR. Hasta el 2026-09-17 el
    # manejador PISABA el comando bueno con uno viejo que apuntaba a un
    # `generar_kernel.py` que nunca existio: la app decia como meter el kernel al
    # banco y ese comando fallaba. No gritaba por ningun lado.
    sys.path.insert(0, str(AQUI))
    import app                                               # noqa: PLC0415
    d = app.datos_kernel(11, 1.8)
    lineas = [l.strip() for l in d["comando"].splitlines() if l.strip()]
    ok("el comando nombra `kernels.py --gauss`, que es la puerta que existe",
       any("kernels.py --gauss" in l for l in lineas),
       "y no un script inventado")
    ok("y lleva la huella, que es lo que hace el apreton de manos",
       d["huella"] in d["comando"])

    # ⚠ UNA VEZ POR (k, sigma), NO UNA POR PETICION. Mover el deslizador dispara
    # 20 peticiones simultaneas del MISMO array; con la cache comprobada fuera del
    # cerrojo, las 20 filtraban las 55 imagenes por su cuenta (5,12 s medidos con
    # k=19 el 2026-09-17, contra 1,32 s con el arreglo).
    import threading                                         # noqa: PLC0415
    import app as _app                                       # noqa: PLC0415
    try:
        v = _app.Visor(4)
    except Exception as e:                                   # noqa: BLE001
        saltar("cache", f"no se pudo crear el visor: {e}")
    else:
        veces = {"n": 0}
        real = v.m.estadisticos

        def contando(ker):
            veces["n"] += 1
            return real(ker)

        v.m.estadisticos = contando
        hilos = [threading.Thread(target=v.z, args=(19, 2.345)) for _ in range(20)]
        for h in hilos:
            h.start()
        for h in hilos:
            h.join()
        ok("20 peticiones del mismo (k, sigma) lo calculan UNA vez",
           veces["n"] == 1, f"se calculo {veces['n']} vez/veces")
        v.m.estadisticos = real

    # ⚠ EL LIMITADOR DE REINICIOS, EN LA SECCION CORRECTA. Puesto en [Service],
    # systemd lo IGNORA con un aviso en el journal que nadie mira, y el freno del
    # bucle de `Restart=always` queda apagado en silencio. Paso aqui el 2026-09-17.
    import app as _a                                         # noqa: PLC0415

    def seccion_de(clave: str) -> str | None:
        """En que seccion del unit cae una clave. Por LINEA, no por `split`:
        un comentario que mencione «[Service]» partiria el texto por donde no es
        -- que es justo lo que hizo fallar a esta prueba al escribirla."""
        actual = None
        for linea in _a.UNIT.splitlines():
            l = linea.strip()
            if l.startswith("[") and l.endswith("]"):
                actual = l
            elif l.startswith(clave + "=") :
                return actual
        return None

    ok("StartLimitIntervalSec va en [Unit], no en [Service]",
       seccion_de("StartLimitIntervalSec") == "[Unit]",
       f"esta en {seccion_de('StartLimitIntervalSec')} — en [Service] systemd lo IGNORA")
    ok("StartLimitBurst tambien", seccion_de("StartLimitBurst") == "[Unit]")
    ok("y la ventana es ALCANZABLE (mayor que RestartSec x burst)",
       300 > 5 * 5, "si no, el limitador no salta nunca y `Restart=` es un bucle")

    # y se ejecuta de verdad, en seco
    cd = [l for l in lineas if l.startswith("cd ")]
    gen = [l for l in lineas if "--gauss" in l]
    if cd and gen and Path(cd[0][3:]).is_dir():
        r = subprocess.run(gen[0].replace("--guardar", ""), shell=True,
                           cwd=cd[0][3:], capture_output=True, text=True, timeout=120)
        ok("y corre de verdad (en seco, sin escribir nada)", r.returncode == 0,
           (r.stdout + r.stderr).strip().splitlines()[-1][:70] if (r.stdout or r.stderr) else "")
    else:
        saltar("el comando corre", "no se pudo resolver la carpeta del banco")


def main() -> int:
    print("\nComprobaciones de `gauss-p`")
    acuerdo_con_el_banco()
    reserva()
    puerta()
    forma_del_kernel()
    app_web()
    modos_y_comando()
    print()
    if saltados:
        print(f"⚠ {len(saltados)} comprobacion(es) SALTADA(s) — no es un ok:")
        for s in saltados:
            print(f"    {s}")
    if fallos:
        print(f"\n✗ {len(fallos)} fallo(s):")
        for f in fallos:
            print(f"    {f}")
        print()
        return 1
    print(f"Todo bien{' (con saltos)' if saltados else ''}.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
