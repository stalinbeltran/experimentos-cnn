#!/usr/bin/env python3
"""Mini app para ELEGIR MIRANDO los parametros de la gaussiana. numpy + pillow.

    python nn/app.py                    la levanta en el 8030
    python nn/app.py --puerto 8031
    python nn/app.py --url              imprime la URL con su token y sale

QUE ES Y QUE NO ES
==================
Es un VISOR. No mide nada, no entrena nada, no gasta nada y no decide nada: pone
delante las mismas 10 muestras pasadas por `gauss(k, sigma)` tal como las vera
`banco-k`, para que la eleccion de `(k, sigma)` la haga un ojo en vez de un
defecto del codigo.

⚠ Lo que se ve NO es un veredicto. Que un `sigma` se vea mejor no dice que el
banco lo mida mejor; para eso hay que evaluarlo alli, y el banco tiene medido
(commit 64b8fbb) que en el ningun kernel llega a declarar. Esto acota DONDE
mirar, no contesta.

LAS CUATRO DECISIONES DE ESTA APP
=================================
1. **Lo que se ensenya es lo que la red RECIBE**, no una version bonita: kernel
   normalizado en L2 (§5.4), convolucion valida, recorte a 128 (§6.2) y
   estandarizado con mu/sigma del §6.5. Un visor que ensenyara la imagen «cruda»
   filtrada mentiria en la direccion mas facil de creer -- la gaussiana apaga el
   contraste, y el §6.5 lo devuelve entero.

2. **La escala de gris es FIJA entre parametros, por defecto.** Auto-ajustar cada
   imagen a su propio rango esconde justo lo que cambia: al desenfocar, la tinta
   se reparte y su z deja de bajar tanto (medido: de -10,9 con la identidad a
   -4,8 con k=19 s=5). Con escala propia las dos se ven igual de negras y esa
   diferencia desaparece de la vista. Se puede cambiar a `escala propia` para
   mirar la ESTRUCTURA, y la app dice cual esta puesta.

3. **Se dibuja la caja de la etiqueta**, que es lo unico que el banco mide. La
   pregunta util no es «se ve bonito» sino «sigue habiendo borde donde la
   etiqueta dice que esta». Con la caja encima, eso se ve; sin ella, se opina.

4. **El token lo crea el servidor al arrancar.** No se configura, no viaja en
   ningun llavero y no puede faltar. Es la leccion de `claude-code-webapp-mobile`
   del 2026-09-15: una app que arranca sin token decide mal su bind y los tres
   mandos dicen verde mientras el movil no entra.

⚠ EL DATO ES PRIVADO. Las imagenes salen de `foveal-vision-data`, que es un repo
PRIVADO, y esta app las sirve por HTTP. Por eso hay token, y por eso esta app no
escribe ni una imagen dentro de `experimentos-cnn`, que es PUBLICO.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import secrets
import socket
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
sys.path.insert(0, str(AQUI.parent.parent))   # para `expcnn`

import banco                                          # noqa: E402
import forma                                          # noqa: E402
from muestras import Muestras, N_POR_DEFECTO, SEMILLA  # noqa: E402

PUERTO = 8030
UNIDAD = "gauss-p"
# El rango de `sigma` que se admite. Por abajo, 0,05 ya es la identidad hasta el
# ultimo bit; por arriba, 60 satura en una caja plana. Fuera de ahi no hay nada
# que mirar, y un numero enorme solo sirve para reventar la memoria.
SIGMA_MIN, SIGMA_MAX = 0.01, 100.0
ESCALA_PNG = 3            # 128 -> 384 px, para que se vea en un movil sin CSS raro


# ------------------------------------------------------------------ el estado
class Visor:
    """Las muestras, sus estadisticos y la cache. Todo lo caro se hace una vez."""

    def __init__(self, n: int = N_POR_DEFECTO, semilla: int = SEMILLA):
        self.m = Muestras()
        self.sel = self.m.elegir(n, semilla)
        self.semilla = semilla
        self.imgs = self.m.imagenes[self.sel]
        self.meta = [self.m.meta[j] for j in self.sel]
        # §6.4: la etiqueta del marco de 584 al de 128
        self.cajas = (self.m.etiquetas[self.sel].astype(np.float32) / 4.0
                      - banco.DESCARTE)
        self._cache: dict[tuple, np.ndarray] = {}
        self._stats: dict[tuple, tuple[float, float]] = {}
        self._lock = threading.Lock()
        # La ventana de gris FIJA sale de la IDENTIDAD: asi la linea base usa el
        # rango entero y todo lo demas se compara contra ella (decision 2).
        z = self.z(None)
        self.ventana = (float(z.min()), float(z.max()))

    def clave(self, k, sigma):
        return (None, None) if k is None else (int(k), round(float(sigma), 4))

    def z(self, k, sigma: float = 0.0) -> np.ndarray:
        """Las muestras elegidas, filtradas y estandarizadas como en el banco.

        ⚠ SE CALCULA UNA VEZ POR (k, sigma), NO UNA VEZ POR PETICION. Mover el
        deslizador dispara **20** peticiones a la vez (10 muestras x 2 imagenes)
        y todas piden el MISMO array: con la comprobacion de cache fuera del
        cerrojo, las 20 entraban a la vez, ninguna veia el resultado de las otras
        y cada una filtraba las 55 imagenes por su cuenta.
        *Medido el 2026-09-17 con k=19: **5,12 s** por cada movimiento del mando;
        con el calculo dentro del cerrojo, **1,32 s**.* Serializar no cuesta nada
        aqui porque el trabajo es de CPU: hacerlo 20 veces en paralelo no es mas
        rapido, es 20 veces el mismo trabajo.
        """
        c = self.clave(k, sigma)
        with self._lock:
            if c in self._cache:
                return self._cache[c]
            ker = None if k is None else banco.gauss(int(k), float(sigma))
            mu, sd = self.m.estadisticos(ker)         # §6.5 sobre las libres
            out = banco.estandarizar(banco.aplicar(self.imgs, ker), mu, sd)
            self._cache[c] = out
            self._stats[c] = (mu, sd)
            if len(self._cache) > 40:                 # no crece sin techo
                for viejo in list(self._cache)[:10]:
                    if viejo != (None, None):         # la identidad no se tira
                        self._cache.pop(viejo, None)
                        self._stats.pop(viejo, None)
        return out

    def stats(self, k, sigma):
        self.z(k, sigma)
        return self._stats[self.clave(k, sigma)]


# ------------------------------------------------------------------ dibujo
def a_png(z: np.ndarray, ventana: tuple[float, float], caja=None,
          escala: int = ESCALA_PNG) -> bytes:
    """Una imagen estandarizada -> PNG en gris, con la caja de la etiqueta."""
    lo, hi = ventana
    if hi <= lo:
        hi = lo + 1e-6
    v = np.clip((z - lo) / (hi - lo), 0.0, 1.0)
    img = Image.fromarray((v * 255).astype(np.uint8), mode="L").convert("RGB")
    if escala > 1:
        img = img.resize((img.width * escala, img.height * escala), Image.NEAREST)
    if caja is not None:
        px = img.load()
        izq, der, sup, inf = [int(round(c * escala)) for c in caja]
        w, h = img.size
        for x in range(max(0, izq), min(w, der + 1)):
            for y in (sup, inf):
                if 0 <= y < h:
                    px[x, y] = (255, 40, 40)
        for y in range(max(0, sup), min(h, inf + 1)):
            for x in (izq, der):
                if 0 <= x < w:
                    px[x, y] = (255, 40, 40)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def perfil_png(k: int, sigma: float, lado: int = 180) -> bytes:
    """El kernel normalizado, como imagen. Su FORMA es lo unico que el banco ve."""
    g = banco.normalizar_kernel(banco.gauss(k, sigma))
    v = (g - g.min()) / max(g.max() - g.min(), 1e-12)
    img = Image.fromarray((v * 255).astype(np.uint8), mode="L").convert("RGB")
    img = img.resize((lado, lado), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def datos_kernel(k: int, sigma: float) -> dict:
    """La forma del kernel (`forma.describir`) mas el comando que lo mete al banco.

    ⚠ La forma NO se vuelve a calcular aqui: se pide a `forma.py`, que es el mismo
    modulo que usa la tabla de la linea de comandos. Dos sitios calculando lo
    mismo divergen y nadie se entera.
    """
    d = forma.describir(k, sigma)
    # ⚠ La carpeta del banco se PREGUNTA por su `id` (R16): escrita a mano, este
    # fichero se rompe en silencio el dia que alguien re-ordene las carpetas.
    try:
        from expcnn import por_id                           # noqa: PLC0415
        bk = por_id("banco-k").carpeta
    except Exception:                                       # noqa: BLE001
        bk = "<la carpeta de `banco-k`>"
    d["comando"] = (
        f"cd {bk}\n"
        f"../.venv/bin/python nn/kernels.py --gauss --k {k} --sigma {sigma:g} "
        f"--guardar --esperado {d['huella']}\n"
        f"BANCOK_SECO=1 nn/lanzar.sh kernel kernels/gauss-k{k:02d}-s{sigma:g}.npy")
    return d


# ------------------------------------------------------------------ el HTML
HTML = r"""<!doctype html>
<html lang="es"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gauss-p · elegir sigma mirando</title>
<style>
 :root { color-scheme: dark; }
 body { margin:0; background:#111; color:#ddd;
        font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
 header { position:sticky; top:0; background:#1a1a1a; border-bottom:1px solid #333;
          padding:10px 12px; z-index:10; }
 h1 { font-size:16px; margin:0 0 8px; font-weight:600; }
 h1 small { font-weight:400; color:#888; }
 .fila { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:6px 0; }
 .ks { display:flex; flex-wrap:wrap; gap:4px; }
 button { background:#2a2a2a; color:#ddd; border:1px solid #444; border-radius:6px;
          padding:6px 10px; font-size:14px; cursor:pointer; }
 button:hover { background:#333; }
 button.on { background:#2d5a8a; border-color:#4a8ac0; color:#fff; }
 input[type=range] { flex:1; min-width:150px; accent-color:#4a8ac0; }
 input[type=number] { width:76px; background:#2a2a2a; color:#ddd; border:1px solid #444;
                      border-radius:6px; padding:5px; font-size:14px; }
 label { color:#999; font-size:13px; }
 .aviso { color:#e0a030; font-size:13px; }
 .ok { color:#5a5; }
 main { padding:12px; }
 .panel { background:#1a1a1a; border:1px solid #333; border-radius:8px;
          padding:10px 12px; margin-bottom:12px; }
 .panel h2 { font-size:14px; margin:0 0 8px; color:#aaa; font-weight:600; }
 table { border-collapse:collapse; font-size:13px; width:100%; }
 td,th { padding:3px 8px 3px 0; text-align:left; }
 th { color:#888; font-weight:500; }
 td.n { font-variant-numeric:tabular-nums; text-align:right; }
 /* SIEMPRE dos columnas, tambien en un movil estrecho: la gracia de esto es
    comparar la identidad con el filtro, y apiladas no se comparan. */
 .par { display:grid; grid-template-columns:1fr 1fr; gap:8px; align-items:start; }
 .muestra { background:#1a1a1a; border:1px solid #333; border-radius:8px;
            padding:8px; margin-bottom:12px; }
 .muestra .cab { font-size:12px; color:#888; margin-bottom:6px; }
 .im { min-width:0; }
 .im img { width:100%; height:auto; display:block; border-radius:4px;
           image-rendering:pixelated; background:#000; }
 .im .pie { font-size:12px; color:#777; margin-top:3px; }
 code { background:#000; padding:2px 5px; border-radius:4px; font-size:12px;
        color:#9c9; word-break:break-all; }
 pre { background:#000; padding:10px; border-radius:6px; overflow-x:auto;
       font-size:12px; color:#9c9; margin:6px 0 0; }
 #cargando { color:#e0a030; font-size:13px; }
</style></head><body>

<header>
 <h1>gauss-p <small>· elegir <b>sigma</b> y <b>k</b> mirando lo que verá el banco</small></h1>
 <div class="fila">
  <label>k</label><div class="ks" id="ks"></div>
 </div>
 <div class="fila">
  <label>sigma</label>
  <input type="range" id="sig" min="0.2" max="8" step="0.05" value="1.5">
  <input type="number" id="sign" min="0.05" max="60" step="0.05" value="1.5">
 </div>
 <div class="fila">
  <button id="bbanco">sigma del banco (k/6)</button>
  <button id="bkmin">ajustar k a este sigma</button>
  <button id="bescala">escala: fija</button>
  <button id="bcaja">caja: sí</button>
  <span id="cargando"></span>
 </div>
</header>

<main>
 <div class="panel">
  <h2>el kernel, tal como el banco lo normaliza (§5.4)</h2>
  <div class="par" style="grid-template-columns:auto 1fr">
   <div><img id="perfil" width="180" height="180"
        style="image-rendering:pixelated;border-radius:4px"></div>
   <div style="flex:1;min-width:220px"><table id="tk"></table></div>
  </div>
 </div>
 <div id="muestras"></div>
 <div class="panel">
  <h2>cuando lo tengas elegido</h2>
  <p style="font-size:13px;color:#999;margin:0 0 6px">
   Esto <b>no</b> evalúa nada: escribe el <code>.npy</code> en
   <code>banco-kernels/kernels/</code> con su receta, para que el kernel se pueda
   regenerar desde código commiteado.</p>
  <pre id="cmd"></pre>
 </div>
 <div class="panel">
  <h2>de dónde salen estas muestras</h2>
  <div id="proc" style="font-size:13px;color:#999"></div>
 </div>
</main>

<script>
const T = new URLSearchParams(location.search).get('t') || '';
let K = 9, S = 1.5, ESCALA = 'fija', CAJA = 1, ST = null;
const $ = id => document.getElementById(id);
const q = o => Object.entries(o).map(([a,b])=>a+'='+encodeURIComponent(b)).join('&');

fetch('/api/estado?t='+T).then(r=>r.json()).then(e=>{
  ST = e;
  $('ks').innerHTML = e.ks.map(k=>`<button data-k="${k}">${k}</button>`).join('');
  $('ks').querySelectorAll('button').forEach(b=>b.onclick=()=>{ K=+b.dataset.k; pinta(); });
  $('proc').innerHTML =
    `<b>${e.dataset}</b> · partición <b>${e.particion}</b><br>`+
    `${e.n_particion} muestras · <b>${e.n_reservadas} descartadas por la reserva §3.7</b> `+
    `(fuente <code>${e.reserva.fuente}</code>, interlineado `+
    `<code>[${e.reserva.interlineado.join(', ')}]</code>) · ${e.n_libres} libres, `+
    `de las que se muestran ${e.n_muestras} (semilla ${e.semilla}).<br>`+
    `<span class="aviso">mu y sigma del §6.5 salen de las ${e.n_libres} libres, no de `+
    `las ${e.n_particion}: la vista previa no es bit a bit la del banco. `+
    `El desvío está medido en <code>nn/muestras.py --desvio</code>.</span>`;
  $('muestras').innerHTML = e.muestras.map((m,i)=>
    `<div class="muestra"><div class="cab">#${i} · i=${m.i} · ${m.fuente} · `+
    `cuerpo ${m.cuerpo} · interlineado ${m.interlineado} · gris ${m.gris}</div>`+
    `<div class="par">`+
    `<div class="im"><img id="ia${i}"><div class="pie">identidad (§6.3) — la línea base</div></div>`+
    `<div class="im"><img id="ib${i}"><div class="pie" id="pb${i}">gauss</div></div>`+
    `</div></div>`).join('');
  pinta();
});

function pinta(){
  $('sig').value = Math.min(Math.max(S,0.2),8); $('sign').value = S.toFixed(2);
  $('ks').querySelectorAll('button').forEach(b=>b.classList.toggle('on', +b.dataset.k===K));
  $('bescala').textContent = 'escala: '+ESCALA;
  $('bcaja').textContent = 'caja: '+(CAJA?'sí':'no');
  $('cargando').textContent = 'calculando…';
  const base = {t:T, k:K, sigma:S, escala:ESCALA, caja:CAJA};
  $('perfil').src = '/perfil.png?'+q({t:T,k:K,sigma:S});
  ST.muestras.forEach((m,i)=>{
    $('ia'+i).src = '/img.png?'+q({...base, i:i, modo:'identidad'});
    $('ib'+i).src = '/img.png?'+q({...base, i:i, modo:'filtrado'});
    $('pb'+i).textContent = `gauss k=${K} sigma=${S.toFixed(2)}`;
  });
  fetch('/api/kernel?'+q({t:T,k:K,sigma:S})).then(r=>r.json()).then(d=>{
    $('cargando').textContent = '';
    const f = (x,n=3)=>Number(x).toFixed(n);
    const pct = x => (100*x).toFixed(x<0.01?3:1)+' %';
    $('tk').innerHTML = [
      ['k · sigma', `${d.k} · ${f(d.sigma,2)}`,
       d.es_el_del_banco?'<span class="ok">este punto el banco YA lo evaluó</span>'
                        :`sin explorar · el banco usa k/6 = ${f(d.sigma_banco,2)}`],
      ['la ventana corta', pct(d.truncamiento),
       d.trunca_mucho
         ? `<span class="aviso">no es una gaussiana de sigma ${f(d.sigma,2)}: está `+
           `recortada. ${d.k_minimo?('haría falta k='+d.k_minimo)
             :'ni k=19 la contiene (§5.2, congelado)'}</span>`
         : 'la ventana la contiene: es la gaussiana entera'],
      ['k de sobra', d.k_de_sobra?'sí':'no',
       d.k_de_sobra
         ? `<span class="aviso">con k=${d.k_minimo} sale el MISMO filtro: el banco `+
           `descarta 9 px sea cual sea k (§6.2)</span>`
         : 'k es el más pequeño que no la corta'],
      ['centro / esquina', f(d.centro_sobre_esquina,1),
       'cuánto pesa el centro frente al borde del kernel'],
      ['masa central', f(d.masa_central), 'fracción del peso en la mitad interior'],
      ['distancia a la identidad', f(d.dist_identidad),
       '0 = no filtra nada · crece con el desenfoque'],
      ['distancia a la caja plana', f(d.dist_caja),
       '0 = ya es un promedio plano: más sigma no cambiaría nada'],
      ['huella (sha256_16)', `<code>${d.huella}</code>`,
       'la que el banco exigirá para confirmar que evalúa ESTO'],
      ['media / desviación (§6.5)', `${f(d.mu,1)} / ${f(d.sd,2)}`,
       'de las libres de train, filtradas con este kernel'],
    ].map(r=>`<tr><th>${r[0]}</th><td class="n">${r[1]}</td><td style="color:#777">${r[2]}</td></tr>`).join('');
    $('cmd').textContent = d.comando;
  });
}

$('sig').oninput = e => { S = +e.target.value; $('sign').value = S.toFixed(2); };
$('sig').onchange = () => pinta();
$('sign').onchange = e => { S = Math.max(0.05, +e.target.value); pinta(); };
$('bbanco').onclick = () => { S = K/6; pinta(); };
$('bkmin').onclick = () => {
  fetch('/api/kernel?'+q({t:T,k:K,sigma:S})).then(r=>r.json()).then(d=>{
    if (d.k_minimo) { K = d.k_minimo; pinta(); }
    else alert('Ni k=19 contiene una gaussiana de sigma '+S.toFixed(2)+
               '. K_MAX está congelado (§5.2 del banco).');
  });
};
$('bescala').onclick = () => { ESCALA = ESCALA==='fija' ? 'propia' : 'fija'; pinta(); };
$('bcaja').onclick = () => { CAJA = CAJA?0:1; pinta(); };
</script></body></html>"""


# ------------------------------------------------------------------ servidor
class Malo(ValueError):
    """Entrada del cliente que no vale. Se contesta 400, no 500.

    ⚠ Y `i` SE VALIDA CONTRA EL RANGO, no sólo como entero. Un `i=-1` indexaba
    numpy por el final y devolvía **200 con OTRA muestra**: en un visor cuyo único
    trabajo es enseñar lo que se pidió, devolver algo distinto en silencio es el
    peor fallo posible -- se elegiría un `sigma` mirando una muestra que no es la
    que dice la cabecera. Medido aquí el 2026-09-17.
    """


def _entero(p, n, d, lo, hi, impar=False):
    try:
        v = int((p.get(n) or [d])[0])
    except (TypeError, ValueError):
        raise Malo(f"`{n}` tiene que ser un entero") from None
    if not (lo <= v <= hi):
        raise Malo(f"`{n}` fuera de rango: {v} no está entre {lo} y {hi}")
    if impar and v % 2 == 0:
        raise Malo(f"`{n}` tiene que ser IMPAR y es {v} (§5.3 del banco)")
    return v


def _real(p, n, d, lo, hi):
    try:
        v = float((p.get(n) or [d])[0])
    except (TypeError, ValueError):
        raise Malo(f"`{n}` tiene que ser un número") from None
    if not (lo <= v <= hi):
        raise Malo(f"`{n}` fuera de rango: {v} no está entre {lo} y {hi}")
    return v


class Handler(BaseHTTPRequestHandler):
    visor: Visor = None            # lo pone servir()
    token: str = ""
    server_version = "gauss-p"

    def log_message(self, fmt, *a):      # una linea por peticion, sin ruido
        sys.stderr.write("  %s %s\n" % (self.address_string(), fmt % a))

    def _envia(self, code, tipo, cuerpo: bytes, cache=False):
        self.send_response(code)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=600")
        self.end_headers()
        self.wfile.write(cuerpo)

    def _json(self, o, code=200):
        self._envia(code, "application/json; charset=utf-8",
                    json.dumps(o, ensure_ascii=False).encode("utf-8"))

    def do_GET(self):                                        # noqa: N802
        u = urllib.parse.urlparse(self.path)
        p = urllib.parse.parse_qs(u.query)
        pide = lambda n, d=None: (p.get(n) or [d])[0]        # noqa: E731

        if u.path == "/salud":                               # sin token, a proposito
            return self._json({"ok": True, "app": "gauss-p"})
        # El dato es PRIVADO: todo lo demas lleva token.
        if not secrets.compare_digest(str(pide("t", "")), self.token):
            return self._json({"error": "falta el token. Pídelo con `nn/app.py --url`"}, 403)

        try:
            if u.path == "/":
                return self._envia(200, "text/html; charset=utf-8", HTML.encode("utf-8"))
            if u.path == "/api/estado":
                v = self.visor
                r = v.m.resumen()
                r.update({
                    "ks": banco.ks_validos(), "n_muestras": len(v.sel),
                    "semilla": v.semilla, "ventana": list(v.ventana),
                    "muestras": [{"i": m["i"], "fuente": m["fuente"],
                                  "cuerpo": m["cuerpo"], "interlineado": m["interlineado"],
                                  "gris": m["gris"]} for m in v.meta],
                })
                return self._json(r)
            if u.path == "/api/kernel":
                k = _entero(p, "k", 9, banco.K_MIN, banco.K_MAX, impar=True)
                s = _real(p, "sigma", 1.5, SIGMA_MIN, SIGMA_MAX)
                d = datos_kernel(k, s)
                mu, sd = self.visor.stats(k, s)
                d["mu"], d["sd"] = mu, sd
                # ⚠ El comando lo pone `datos_kernel`, y aqui NO se vuelve a
                # escribir. Estuvo pisado hasta el 2026-09-17 por una version
                # vieja que apuntaba a un script inexistente: la app decia como
                # generar el kernel y ese comando no existia. Lo caza el test
                # «el comando que imprime la app existe de verdad».
                return self._json(d)
            if u.path == "/perfil.png":
                return self._envia(200, "image/png", perfil_png(
                    _entero(p, "k", 9, banco.K_MIN, banco.K_MAX, impar=True),
                    _real(p, "sigma", 1.5, SIGMA_MIN, SIGMA_MAX)), cache=True)
            if u.path == "/img.png":
                v = self.visor
                i = _entero(p, "i", 0, 0, len(v.sel) - 1)
                k = (None if pide("modo") == "identidad"
                     else _entero(p, "k", 9, banco.K_MIN, banco.K_MAX, impar=True))
                z = v.z(k, _real(p, "sigma", 1.5, SIGMA_MIN, SIGMA_MAX))[i]
                ventana = (v.ventana if pide("escala", "fija") == "fija"
                           else (float(z.min()), float(z.max())))
                caja = v.cajas[i] if pide("caja", "1") == "1" else None
                return self._envia(200, "image/png", a_png(z, ventana, caja), cache=True)
        except Malo as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:                               # nunca tumba el servidor
            import traceback
            traceback.print_exc()
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
        return self._json({"error": "no existe"}, 404)


def _ip() -> str:
    """La IP con la que se sale a la red, para componer la URL del movil."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("1.1.1.1", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def _token_de_disco(crear: bool) -> str:
    """El token vive en `~/.cache/gauss-p.token`, FUERA de los dos repos.

    ⚠ Se crea al arrancar y no se configura (decision 4). `--url` lo LEE: si no
    existe, es que la app no esta levantada, y eso se dice en vez de inventar una
    URL que no va a funcionar.
    """
    p = Path.home() / ".cache" / "gauss-p.token"
    if p.is_file():
        return p.read_text().strip()
    if not crear:
        return ""
    p.parent.mkdir(parents=True, exist_ok=True)
    t = secrets.token_urlsafe(18)
    p.write_text(t)
    p.chmod(0o600)
    return t


def _sh(cmd: list[str]) -> str:
    import subprocess
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:                                        # noqa: BLE001
        return ""


def estado(puerto: int) -> int:
    """¿Se llega a esto desde el movil? Se pregunta al SOCKET y al CORTAFUEGOS.

    ⚠ NO se deduce del disco ni de `systemctl is-active`. Es la leccion del
    2026-09-15, que costo un dev entero por dos puertas distintas: la unidad
    decia `active`, `curl` desde la propia maquina daba 200 --porque ese trafico
    va por `lo` y `ufw` lo deja pasar-- y el movil no entraba. Lo que hay que
    mirar son las TRES cosas de abajo, y las tres tienen que dar verde.
    """
    print(f"\ngauss-p · puerto {puerto}\n")
    problemas = []

    activa = _sh(["systemctl", "is-active", UNIDAD]) or "(no instalada)"
    show = _sh(["systemctl", "show", UNIDAD, "-p", "Result", "-p", "NRestarts"])
    print(f"  unidad     {UNIDAD}: {activa}")
    for l in show.splitlines():
        print(f"             {l}")
    # ⚠ NRestarts NO es cosmetico: una unidad que fallo y se relanzo sola PARECE
    # corriendo y esta repitiendo trabajo desde cero (2026-09-04, 62 veces).
    if activa != "active":
        problemas.append(f"la unidad no esta activa: sudo systemctl start {UNIDAD}")

    ss = _sh(["ss", "-ltn"])
    linea = [l for l in ss.splitlines() if f":{puerto} " in l or l.endswith(f":{puerto}")]
    print(f"  socket     {linea[0].strip() if linea else 'NADA escuchando en ' + str(puerto)}")
    if not linea:
        problemas.append(f"nada escucha en el {puerto}")
    elif "127.0.0.1" in linea[0]:
        problemas.append(f"escucha SOLO en loopback: desde el movil no se llega")

    ufw = _sh(["sudo", "-n", "ufw", "status"])
    abierto = any(str(puerto) in l for l in ufw.splitlines()) if ufw else None
    print(f"  cortafuegos {'abierto' if abierto else 'CERRADO' if ufw else 'no se pudo mirar'}"
          f" para el {puerto}")
    if ufw and not abierto:
        problemas.append(f"ufw no deja pasar el {puerto}: "
                         f"sudo ufw allow {puerto}/tcp   (o `nn/app.py --abrir`)")
    if not ufw:
        problemas.append("no pude leer `ufw status`: NO se si se llega desde fuera")

    hay_token = bool(_token_de_disco(crear=False))
    print(f"  token      {'sí' if hay_token else 'NO'}")
    if not hay_token:
        problemas.append("no hay token: la app no ha arrancado nunca aqui")

    print()
    if problemas:
        print("✗ desde el movil NO se llega:")
        for x in problemas:
            print(f"    {x}")
        print()
        return 1
    print(f"ok  http://{_ip()}:{puerto}/?t={_token_de_disco(crear=False)}\n")
    print("  ⚠ Lo unico que esto NO puede comprobar es la red de en medio.")
    print("    `curl` desde esta maquina va por `lo` y pasa el cortafuegos igual:")
    print("    la prueba de verdad es abrir esa URL en el movil.\n")
    return 0


def cortafuegos(puerto: int, abrir: bool) -> int:
    import subprocess
    accion = "allow" if abrir else "delete allow"
    cmd = ["sudo", "-n", "ufw", *accion.split(), f"{puerto}/tcp"]
    print("  " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    print((r.stdout + r.stderr).strip())
    return r.returncode


UNIT = """[Unit]
Description=gauss-p (visor de parametros de la gaussiana, experimentos-cnn)
After=network-online.target
Wants=network-online.target
# ⚠ ESTAS DOS VAN EN [Unit], NO EN [Service] -- en [Service] systemd las IGNORA con
# un aviso en el journal que nadie mira (comprobado aqui el 2026-09-17). Y la
# ventana tiene que ser ALCANZABLE: con RestartSec=5 y la de por defecto (10 s)
# nunca caben 5 arranques, asi que el limitador no saltaria NUNCA y un fallo al
# arrancar seria un bucle infinito (la leccion del 2026-09-04, 62 relanzamientos).
StartLimitIntervalSec=300
StartLimitBurst=5

[Service]
Type=simple
User={usuario}
WorkingDirectory={repo}
ExecStart=/bin/bash -lc 'exec .venv/bin/python {rel} --puerto {puerto}'
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
"""


def instalar(puerto: int) -> int:
    """Deja la app corriendo y alcanzable en ESTA maquina. Idempotente.

    ⚠ POR QUE HACE FALTA UN COMANDO Y NO VALE «ya esta puesta»: estos servers son
    EFIMEROS y `experimentos-cnn` **no esta en los repos que clona un dev nuevo**
    (`types/dev.json` del lanzador, comprobado el 2026-09-17). O sea que al rehacer
    la maquina esto no existe, y una unidad hecha a mano no se reconstruye sola.
    Escrito como lista de pasos se olvida uno; escrito como comando, no.

        git clone …/experimentos-cnn && cd experimentos-cnn
        uv venv .venv && uv pip install --python .venv/bin/python numpy pillow
        .venv/bin/python <esta carpeta>/nn/app.py instalar
    """
    import getpass
    import subprocess
    repo = AQUI.parent.parent
    unit = UNIT.format(usuario=getpass.getuser(), repo=repo,
                       rel=AQUI.relative_to(repo) / "app.py", puerto=puerto)
    destino = f"/etc/systemd/system/{UNIDAD}.service"
    print(f"\n  escribiendo {destino}")
    r = subprocess.run(["sudo", "-n", "tee", destino], input=unit, text=True,
                       capture_output=True, timeout=60)
    if r.returncode != 0:
        print(f"✗ no pude escribir la unidad: {r.stderr.strip()}")
        print(f"  → hace falta sudo. Contenido, por si lo pones a mano:\n\n{unit}")
        return 1
    for cmd in (["sudo", "-n", "systemctl", "daemon-reload"],
                ["sudo", "-n", "systemctl", "enable", "--now", UNIDAD],
                ["sudo", "-n", "systemctl", "restart", UNIDAD]):
        print("  " + " ".join(cmd[2:]))
        subprocess.run(cmd, capture_output=True, timeout=120)
    # ⚠ El cortafuegos va en el MISMO paso, no «luego». Es la cuarta puerta del
    # 2026-09-15: `ss` decia 0.0.0.0, `curl` local daba 200 --porque va por `lo`--
    # y el movil no entraba, porque el puerto no estaba abierto en `ufw`.
    cortafuegos(puerto, True)
    # ⚠ SE ESPERA AL SOCKET, NO A `is-active`. Con `Type=simple` systemd da la
    # unidad por activa en cuanto lanza el proceso, y esta app tarda unos segundos
    # en cargar el dataset antes de atar el puerto: esperar a `is-active` daba un
    # `✗ nada escucha` que era mentira. Es la misma regla del `estado` de abajo --
    # se mira el dato observable-- aplicada a la espera. Pasó aquí el 2026-09-17.
    import time
    for _ in range(60):
        if any(f":{puerto} " in l or l.endswith(f":{puerto}")
               for l in _sh(["ss", "-ltn"]).splitlines()):
            break
        time.sleep(1)
    print()
    return estado(puerto)


def servir(puerto: int, n: int, semilla: int, host: str = "0.0.0.0") -> int:
    print(f"gauss-p · cargando {n} muestras…", flush=True)
    Handler.visor = Visor(n, semilla)
    Handler.token = _token_de_disco(crear=True)
    r = Handler.visor.m.resumen()
    print(f"  {r['n_particion']} de train · {r['n_reservadas']} descartadas por §3.7 · "
          f"{r['n_libres']} libres · se muestran {len(Handler.visor.sel)}")
    print(f"  ventana de gris fija (de la identidad): "
          f"[{Handler.visor.ventana[0]:.2f}, {Handler.visor.ventana[1]:.2f}]")
    print(f"\n  http://{_ip()}:{puerto}/?t={Handler.token}\n", flush=True)
    srv = ThreadingHTTPServer((host, puerto), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nadiós")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--puerto", type=int, default=int(os.environ.get("GAUSSP_PUERTO", PUERTO)))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--n", type=int, default=N_POR_DEFECTO)
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    ap.add_argument("--url", action="store_true", help="imprime la URL y sale")
    ap.add_argument("--estado", action="store_true",
                    help="¿se llega desde el móvil? pregunta al socket y a ufw")
    ap.add_argument("--abrir", action="store_true", help="abre el puerto en ufw")
    ap.add_argument("--cerrar", action="store_true", help="lo cierra")
    # Lo mismo como PALABRA, que es como se teclea desde Telegram.
    # `nargs="*"` y no `"?"`: desde Telegram se teclea texto libre, y con `"?"`
    # dos palabras daban el «unrecognized arguments» de argparse en vez de la
    # negativa de abajo, que sí dice qué modos hay.
    ap.add_argument("modo", nargs="*", default=[],
                    help="url · estado · instalar · abrir · cerrar · forma · (vacío: servir)")
    a = ap.parse_args()

    # ⚠ EL MODO SE FIJA AQUI Y NO SE VUELVE A LEER DE NINGUN ARGUMENTO, y el
    # ultimo caso SE NIEGA en vez de hacer algo por defecto. Es la leccion del
    # 2026-09-08: un lanzador con varios modos releyo `$1` despues de consumirlo,
    # corrio la calibracion en vez de lo que se le pidio, y salio con
    # `Result=success` -- el falso verde, con todos los indicadores en verde.
    modo = " ".join(a.modo).strip().lower()
    if a.estado:
        modo = "estado"
    elif a.abrir:
        modo = "abrir"
    elif a.cerrar:
        modo = "cerrar"
    elif a.url:
        modo = "url"

    if modo in ("estado", "forma", "url", "abrir", "cerrar", "instalar", "", "servir"):
        print(f"gauss-p: modo `{modo or 'servir'}`", file=sys.stderr)
    if modo == "estado":
        return estado(a.puerto)
    if modo == "instalar":
        return instalar(a.puerto)
    if modo in ("abrir", "cerrar"):
        return cortafuegos(a.puerto, modo == "abrir")
    if modo == "forma":
        return forma.tabla()
    if modo not in ("", "servir", "url"):                    # se NIEGA, no adivina
        print(f"\n✗ no sé qué es «{modo}».\n"
              f"  url · estado · instalar · abrir · cerrar · forma · (nada: levantarla)\n")
        return 2
    if modo == "url":
        t = _token_de_disco(crear=False)
        if not t:
            print("✗ no hay token: la app no se ha levantado nunca en esta máquina.")
            print(f"  → sudo systemctl start {UNIDAD}")
            return 1
        print(f"http://{_ip()}:{a.puerto}/?t={t}")
        return 0
    return servir(a.puerto, a.n, a.semilla, a.host)


if __name__ == "__main__":
    raise SystemExit(main())
