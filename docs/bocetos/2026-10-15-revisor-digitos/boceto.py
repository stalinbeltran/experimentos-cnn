#!/usr/bin/env python3
"""BOCETO (no ejecutado contra datos reales: el dígito y los números son inventados para enseñar la forma) del app web
móvil «revisor de dígitos»: lista de fallos → dígito con sus objetos → rejillas y atribución del compositor → comparar
corridas. Genera 01-pantallas.png y 02-datos.png.    ../../../.venv/bin/python boceto.py
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, Circle

C = dict(rec="#d2501e", arc="#2b7fd6", lazo="#111111", ent=["#d2501e", "#2b7fd6", "#2a9d5c", "#7b3fd1"], rojo="#b22222", verde="#1f8a4c", gris="#777", fondo="#f6f6f6")

def telefono(ax, titulo):
    ax.set_xlim(0, 390); ax.set_ylim(844, 0); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 390, 844, boxstyle="round,pad=0,rounding_size=28", fc="white", ec="#222", lw=2.5))
    ax.add_patch(Rectangle((0, 0), 390, 64, fc="#222"))
    ax.text(16, 40, titulo, color="white", fontsize=11, fontweight="bold", va="center")
    # barra inferior
    ax.add_patch(Rectangle((0, 784), 390, 60, fc="#eee"))
    for i, t in enumerate(["Lista", "Dígito", "Compositor", "Comparar"]):
        ax.text(48 + i * 98, 814, t, ha="center", va="center", fontsize=8.5, color="#222")

def digito6(ax, x0, y0, s, objetos=True, grueso=False):
    """un 6 inventado: tinta gris, cresta, lazo, recta, arco, entrante arriba"""
    t = np.linspace(0, 2 * np.pi, 80); cx, cy = x0 + 0.5 * s, y0 + 0.66 * s
    lazo = np.c_[cx + 0.26 * s * np.cos(t), cy + 0.24 * s * np.sin(t)]
    cola = np.c_[np.linspace(x0 + 0.28 * s, x0 + 0.6 * s, 20), np.linspace(y0 + 0.45 * s, y0 + 0.05 * s, 20)]
    for p in list(lazo) + list(cola):
        ax.add_patch(Circle(p, 0.07 * s if not grueso else 0.11 * s, fc="#c9c9c9", ec="none"))
    if objetos:
        ax.plot(lazo[:, 0], lazo[:, 1], color=C["lazo"], lw=2.2)
        ax.plot(cola[:, 0], cola[:, 1], color=C["rec"], lw=2.2)
        ax.plot(lazo[:8, 0], lazo[:8, 1], color=C["arc"], lw=3)
        tri = Polygon([cola[0], cola[-1], (x0 + 0.75 * s, y0 + 0.42 * s)], closed=True, fc=C["ent"][3], alpha=0.3, ec=C["ent"][3])
        ax.add_patch(tri); ax.plot(cx, cy, "+", color="k", ms=9, mew=2)
    else:
        ax.plot(lazo[:, 0], lazo[:, 1], color="#555", lw=1); ax.plot(cola[:, 0], cola[:, 1], color="#555", lw=1)

def rejilla(ax, x, y, s, vals, col, tit):
    ax.add_patch(Rectangle((x, y), s, s, fc="white", ec=col, lw=1.6))
    for i in range(3):
        for j in range(3):
            v = vals[i][j]
            ax.add_patch(Rectangle((x + j * s / 3, y + i * s / 3), s / 3, s / 3, fc=(1 - v, 1 - v, 1 - v), ec="none"))
            ax.text(x + (j + .5) * s / 3, y + (i + .5) * s / 3, f"{v:.2f}", ha="center", va="center", fontsize=5.2, color="white" if v > .55 else "#222")
    ax.text(x + s / 2, y - 5, tit, ha="center", va="bottom", fontsize=5.8, color=col)

Z = [[0, 0, 0]] * 3
fig, axs = plt.subplots(1, 4, figsize=(20, 11))
# ---- pantalla 1: lista
ax = axs[0]; telefono(ax, "rizo-2 · revisor de dígitos")
ax.add_patch(Rectangle((16, 80), 358, 34, fc=C["fondo"], ec="#ccc")); ax.text(26, 97, "corrida: cresta-lazos-region (144) · val 0,9365  ▾", fontsize=8, va="center")
for i, (t, on) in enumerate([("fallos", 1), ("3 sem.", 0), ("aciertos", 0), ("todos", 0)]):
    ax.add_patch(FancyBboxPatch((16 + i * 90, 124), 82, 26, boxstyle="round,pad=0,rounding_size=13", fc="#222" if on else "white", ec="#222"))
    ax.text(57 + i * 90, 137, t, ha="center", va="center", fontsize=8, color="white" if on else "#222")
ax.text(16, 175, "clase:", fontsize=8, va="center")
for k in range(10):
    ax.add_patch(Circle((70 + k * 31, 175), 12, fc="#222" if k == 6 else "white", ec="#222"))
    ax.text(70 + k * 31, 175, str(k), ha="center", va="center", fontsize=8, color="white" if k == 6 else "#222")
ax.text(16, 205, "confusión: 6→8 ×5 · 6→0 ×3 · 6→1 ×2   (10 fallos de 160)", fontsize=7.5, va="center", color=C["gris"])
for n in range(6):
    r, c = divmod(n, 2); x, y = 20 + c * 180, 225 + r * 180
    ax.add_patch(Rectangle((x, y), 170, 170, fc="white", ec="#ddd"))
    et = ["6→8 · idx 34 · 3/3", "6→0 · idx 444 · 3/3", "6→0 · idx 782 · 2/3", "6→1 · idx 1771 · 3/3", "6→8 · idx 902 · 3/3", "6 ✓ · idx 15 · p 0,98"][n]
    ax.text(x + 6, y + 12, et, fontsize=6.8, color=C["rojo"] if "→" in et else C["verde"])
    digito6(ax, x + 35, y + 30, 110, grueso=(n == 3))
ax.text(195, 775, "toca un dígito para abrirlo", ha="center", fontsize=7.5, color=C["gris"])
# ---- pantalla 2: dígito
ax = axs[1]; telefono(ax, "idx 782 · es un 6 · lo lee 0   (2/3 semillas)")
for i, (t, on) in enumerate([("tinta", 0), ("delgado", 1), ("cresta", 0), ("objetos", 1)]):
    ax.add_patch(FancyBboxPatch((16 + i * 90, 76), 82, 24, boxstyle="round,pad=0,rounding_size=12", fc="#222" if on else "white", ec="#222"))
    ax.text(57 + i * 90, 88, t, ha="center", va="center", fontsize=7.5, color="white" if on else "#222")
ax.add_patch(Rectangle((35, 112), 320, 320, fc="white", ec="#ddd"))
for k in (1, 2):
    ax.plot([35 + k * 320 / 3] * 2, [112, 432], color="#bbb", lw=.7, ls="--"); ax.plot([35, 355], [112 + k * 320 / 3] * 2, color="#bbb", lw=.7, ls="--")
digito6(ax, 85, 135, 230)
ax.text(195, 448, "rejilla 3×3 de fondo · pellizca para ampliar", ha="center", fontsize=7, color=C["gris"])
ax.text(16, 475, "objetos que ve el detector (toca uno para resaltarlo):", fontsize=8, fontweight="bold")
filas = [("■ lazo", "182 px² · centro (16,20) · región: centro 0,67 · abajo-c 0,61", C["lazo"]),
         ("■ recta 45° /", "largo 11 px · centro (10,9) · + 7 restos fundidos", C["rec"]),
         ("■ arco k↑", "barre 64° · largo 14 px · abajo-centro", C["arc"]),
         ("■ arco k←", "barre 180° · lado izquierdo del lazo", C["arc"]),
         ("■ entrante ↑ arriba", "abertura 9 px · presencia 0,97 arriba-centro", C["ent"][3]),
         ("· golpes", "ninguno", C["gris"])]
for i, (a, b, col) in enumerate(filas):
    y = 500 + i * 42
    ax.add_patch(Rectangle((16, y - 14), 358, 38, fc=C["fondo"] if i % 2 == 0 else "white", ec="none"))
    ax.text(24, y, a, fontsize=8, color=col, fontweight="bold", va="center"); ax.text(24, y + 15, b, fontsize=6.8, color="#333", va="center")
ax.text(195, 770, "◀ anterior fallo        siguiente fallo ▶", ha="center", fontsize=8)
# ---- pantalla 3: compositor
ax = axs[2]; telefono(ax, "idx 782 · compositor (144 entradas)")
ax.text(16, 84, "logits · rojo = predicho 0, verde = real 6", fontsize=7.5, va="center")
lg = [4.3, -1, -0.5, -2, -1.5, 0.5, 4.0, -3, 2.1, 0.2]
for k, v in enumerate(lg):
    h = max(v, 0) * 14; col = C["rojo"] if k == 0 else C["verde"] if k == 6 else "#aaa"
    ax.add_patch(Rectangle((22 + k * 36, 165 - h), 26, h, fc=col)); ax.text(35 + k * 36, 176, str(k), ha="center", fontsize=7)
    ax.text(35 + k * 36, 160 - h, f"{v:+.1f}", ha="center", fontsize=5.5)
ax.text(16, 200, "rejillas que recibe (desliza →; toca una celda para ver su peso):", fontsize=7.5, va="center")
tiles = [("recta 0° —", C["rec"], [[0, .02, 0], [0, .13, 0], [0, 0, 0]]), ("recta 45° /", C["arc"], [[0, 0, 0], [.03, .03, 0], [.10, .10, 0]]),
         ("recta 90° |", C["verde"], [[.03, .03, 0], [.28, .28, .05], [.05, .04, .06]]), ("recta 135° \\", C["ent"][3], [[.02, .08, 0], [0, 0, 0], [0, 0, 0]]),
         ("arco k←", C["arc"], [[0, 0, 0], [0, .32, .32], [0, .01, .01]]), ("arco k↑", C["ent"][3], [[0, 0, 0], [0, 0, 0], [0, .75, .11]]),
         ("lazo REGIÓN", C["lazo"], [[0, 0, 0], [.12, .67, .07], [.02, .61, .10]]), ("entrante ↑", C["ent"][3], [[.07, .97, .01], [.07, .90, .01], [0, 0, 0]]),
         ("golpes", C["gris"], Z)]
for n, (t, col, v) in enumerate(tiles):
    r, c = divmod(n, 3); rejilla(ax, 28 + c * 118, 232 + r * 118, 96, v, col, t)
ax.text(16, 600, "qué pesa · rojo empuja al 0, verde al 6", fontsize=7.5, va="center")
ax.text(16, 614, "por bloque: rectas −0,5 · arcos −0,7 · lazo +3,0 · entrantes −1,3", fontsize=6.8, va="center", color="#333")
barras = [("lazo región · centro (0,67)", 2.3), ("lazo región · centro-izq", 1.15), ("arco k→ · cuenta", .7), ("recta 45° · cuenta", .5),
          ("entrante ↑ · arriba-centro", -.65), ("recta 90° | · centro-izq", -.8), ("arco k↑ · largo/32", -.6), ("recta 45° · abajo-izq", -.5)]
barras.sort(key=lambda b: -b[1])
for i, (t, v) in enumerate(barras):
    y = 632 + i * 17; w = abs(v) * 36
    ax.add_patch(Rectangle((200, y), w if v > 0 else -w, 12, fc=C["rojo"] if v > 0 else C["verde"]))
    ax.text(196 if v > 0 else 204, y + 6, t, ha="right" if v > 0 else "left", va="center", fontsize=6.2)
# ---- pantalla 4: comparar
ax = axs[3]; telefono(ax, "idx 782 · el mismo dígito en 3 corridas")
cor = [("presencia (§ 74)", "6 ✓", C["verde"], [[0, 0, 0], [0, .99, 0], [0, .2, 0]], "lazo 0,99 en una celda"),
       ("región (§ 79, aprobada)", "0 ✗", C["rojo"], [[0, 0, 0], [.12, .67, .07], [.02, .61, .10]], "182 px² llena centro y abajo: patrón de 0"),
       ("área (§ 79)", "6 ✓", C["verde"], [[0, 0, 0], [0, .60, 0], [0, .60, 0]], "el rival pasa a ser el 8 (+3,6)")]
for i, (t, ver, col, v, nota) in enumerate(cor):
    y = 84 + i * 200
    ax.add_patch(Rectangle((16, y), 358, 186, fc=C["fondo"], ec="#ddd"))
    ax.text(26, y + 16, t, fontsize=8.5, fontweight="bold", va="center"); ax.text(360, y + 16, ver, fontsize=11, fontweight="bold", color=col, ha="right", va="center")
    digito6(ax, 30, y + 35, 120); rejilla(ax, 190, y + 50, 96, v, C["lazo"], "bloque del lazo")
    ax.text(300, y + 100, "logit 0 / 6", fontsize=6.5, ha="left"); ax.text(300, y + 115, ["+1,9 / +4,1", "+4,3 / +4,0", "+2,0 / +4,4"][i], fontsize=7.5, ha="left", fontweight="bold")
    ax.text(26, y + 170, nota, fontsize=6.8, color="#333", va="center")
ax.text(195, 700, "↑ sólo cambia el bloque que difiere entre corridas;\nlas otras 14 rejillas se marcan «idénticas»", ha="center", fontsize=7, color=C["gris"])
ax.text(195, 745, "resumen de la corrida: val · ciega · fallos · confusiones", ha="center", fontsize=7, color=C["gris"])
fig.suptitle("BOCETO del app móvil «revisor de dígitos» (rizo-2) — dígito y números INVENTADOS para enseñar la forma, no ejecutado contra datos", fontsize=12, y=0.995)
fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig("01-pantallas.png", dpi=110)

# ---- figura 2: datos y conclusiones
fig, ax = plt.subplots(figsize=(14, 8.5)); ax.set_xlim(0, 140); ax.set_ylim(85, 0); ax.axis("off")
def caja(x, y, w, h, tit, lineas, fc="white", ec="#222"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5", fc=fc, ec=ec, lw=1.6))
    ax.text(x + 1.5, y + 3.2, tit, fontsize=10, fontweight="bold", va="center")
    for i, l in enumerate(lineas): ax.text(x + 1.5, y + 7 + i * 3.6, l, fontsize=8.2, va="center")
caja(2, 3, 40, 30, "1 · dev: Python exporta (nn/exportar_revisor.py)", [
    "por corrida (40-entrada-*.json) y por idx de val:",
    "· delgado 32×32 (PNG) · tinta · cresta",
    "· objetos: tipo, centro, largo, orientación, área…",
    "· las 144 entradas con nombre",
    "· logits, predicho, real, p, semillas que fallan",
    "· atribución por entrada (ya la calcula desglose.py)",
    "→ resultados/revisor/<corrida>/{idx}.json + .png"], fc="#fff7ee")
caja(50, 3, 40, 30, "2 · commit + push al repo", [
    "el export es DATO: 1617 × ~3 KB ≈ 5 MB/corrida",
    "(estimado, no medido; PNG de 32 px ≈ 300 B)",
    "una corrida nueva = correr el export y empujar",
    "sin torch ni numpy en el navegador:",
    "la app sólo LEE json y png estáticos"], fc="#eef5ff")
caja(98, 3, 40, 30, "3 · mini: app estática (HTML+JS)", [
    "servicio systemd, puerto 8050, token al arrancar",
    "ufw en el MISMO paso que la unidad",
    "`estado` pregunta a ss y a ufw (las 4 puertas)",
    "git pull del repo = datos nuevos, sin reiniciar",
    "/use revisor → url (bot Lanzador)"], fc="#eefbf0")
for x in (42, 90): ax.annotate("", (x + 8, 18), (x, 18), arrowprops=dict(arrowstyle="->", lw=2))
caja(2, 40, 136, 42, "Lo que el app contesta, pantalla por pantalla (sale de las figuras 75–83 de ayer)", [
    "LISTA     — ¿qué dígitos fallan en esta corrida, por clase y por confusión? (fig. 75: 4 por clase; aquí: todos, filtrables, y también aciertos)",
    "DÍGITO    — ¿qué objetos ve el detector y dónde? (fig. 75/77: objetos pintados sobre el delgado + lista con sus números; cada objeto se toca)",
    "COMPOSITOR— ¿qué recibe y qué pesa? (fig. 82/83: las 15 rejillas 3×3, logits de las 10 clases, barras rojo/verde por entrada y suma por bloque)",
    "COMPARAR  — ¿cómo lee el MISMO dígito cada corrida? (fig. 80/81: presencia / región / área; sólo se despliega el bloque que cambia)",
    "",
    "Decisiones a confirmar:  (a) la atribución se precalcula en el dev (no hay torch en el móvil)   (b) val entero + a ciegas NO (no se mira)",
    "(c) vive en experimentos-cnn/2026-10-11-rizo-dos/web/ junto al export; se sirve desde el mini como claude-web   (d) qué corridas entran: las 40-entrada-* con 144",
    "Coste: cero alquiler. Export ≈ 2 min por corrida (estimado: desglose.py entrena el MLP una vez, ~118 s según el JSON) · sin ejecutar nada todavía"])
fig.savefig("02-datos.png", dpi=110)
print("ok")
