"""Un dígito, de cerca: qué objetos ve el detector y cuánto aporta cada feature a CADA clase por separado (el dueño,
2026-10-17, con el 973: «un 5 reconocido como 3; las concavidades no me convencen; analiza los pesos que se le dan al 5 y
al 3 por separado; si es un cono de 60°, dibújalo así»).

    ../.venv/bin/python nn/analizar_digito.py <idx> <clase A> <clase B> "<compositor>" ["<compositor>"…]

Saca, en resultados/:
  100-<idx>-objetos.png        el dígito con sus concavidades y entrantes, cada uno con el SECTOR de su clase de abertura
                               dibujado: el cuadrante de 90° de `concavidades` y el cono de `entrantes60` (±60° en
                               horizontal, 60° en vertical), y una flecha con la dirección exacta.
  100-<idx>-pesos-<comp>.png   por cada rejilla del compositor, en tres filas: lo que aporta al logit de A, al de B, y A − B.

«Aporte» = gradiente × entrada del logit de ESA clase respecto de la entrada normalizada, media de las semillas. Es la
misma atribución que la app («qué pesa»), pero por clase en vez de por diferencia. ⚠ No es una descomposición exacta: la
red tiene una ReLU, así que la suma de aportes aproxima (no iguala) el logit menos el de una entrada media.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "web")); sys.path.insert(0, str(AQUI / "nn"))
import servidor as S  # noqa: E402
import cresta_curvas as CC  # noqa: E402

DIRS = "→←↓↑"
COLOR = {0: "#d4541a", 1: "#2a78d6", 2: "#1b7f3b", 3: "#7c3aed"}     # → ← ↓ ↑, los de la app


CORTO = {"lazos_region": "lazo región", "concavidades": "concavidad", "entrantes60": "cono 60°", "entrantes60f": "cono 60° fuerza",
         "entrantes": "entrante", "vertice_1": "vértice 1 celda", "vertice": "vértice", "rectas_todo": "rectas∪restos"}   # títulos que caben


def sector(clase: int, ancho_h: float) -> tuple[float, float]:
    """Ángulos (grados, 0 = derecha, positivo hacia ABAJO como en la imagen) del sector de una clase de abertura.
    ancho_h = medio ángulo horizontal: 45 para `concavidades` (cuadrantes), 60 para `entrantes60` (conos)."""
    return {0: (-ancho_h, ancho_h), 1: (180 - ancho_h, 180 + ancho_h), 2: (ancho_h, 180 - ancho_h), 3: (-180 + ancho_h, -ancho_h)}[clase]


def compositor(nombre: str, d: S.Datos):
    alm = S.almacen_revisor(); rec = {**json.loads((alm / "recetas.json").read_text()), **S.recetas_guardadas()}
    bl = rec[nombre]
    c = S.Compositor(nombre, bl["bloques"], bl.get("celdas", 3), bl.get("semillas", S.SEMILLAS), bl.get("sub4", False), bl.get("extra", False))
    if not c.restaurar(alm, c.huella(d)):
        raise SystemExit(f"✗ «{nombre}»: sin pesos válidos en el almacén")
    return c


def aportes(c: S.Compositor, d: S.Datos, i: int, clases: list[int]):
    todos, _, _ = d.features(c.celdas); x = c.corta(todos[i:i + 1])[0]; por = {k: [] for k in clases}; lg = []; P = []
    for red, mu, sd in c.redes:
        z = torch.from_numpy((x - mu) / sd).float()[None].requires_grad_(True); out = red(z)[0]
        lg.append(out.detach().numpy()); P.append(torch.softmax(out, 0).detach().numpy())
        for k in clases:
            g, = torch.autograd.grad(out[k], z, retain_graph=True); por[k].append((g[0] * z[0]).detach().numpy())
    return x, {k: np.mean(v, 0) for k, v in por.items()}, np.mean(lg, 0), np.mean(P, 0)


def fig_objetos(d: S.Datos, i: int, salida: Path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    ob = CC.objetos(d.M[i]); conc = CC.concavidades(ob["arcos"], ob["lazos"]); ent = ob["entrantes"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.4), dpi=150)
    for ax, objs, ancho, clase, titulo in ((axs[0], conc, 45, lambda q: q["o"], "CONCAVIDADES de arco\n(sector = cuadrante de 90°: la clase de abertura)"),
                                            (axs[1], ent, 60, CC.cono, "ENTRANTES (arcos + esquinas)\n(sector = cono de entrantes60: ±60° horizontal, 60° vertical)")):
        ax.imshow(d.tinta[i], cmap="Greys", vmin=0, vmax=4, alpha=1)               # la tinta, gris claro
        ax.imshow(np.ma.masked_less(d.M[i], 0.05), cmap="Greys", vmin=0, vmax=2.4)  # el delgado (la cresta), encima
        for k in (1, 2):
            ax.axhline(32 * k / 3 - 0.5, color="#9ca3af", ls="--", lw=0.8); ax.axvline(32 * k / 3 - 0.5, color="#9ca3af", ls="--", lw=0.8)
        for q in objs:
            cl = clase(q); col = COLOR[cl]; cx, cy = q["cx"] - 0.5, q["cy"] - 0.5
            ax.add_patch(Polygon(np.asarray(q["pts"]) - 0.5, closed=True, fc=col, ec=col, alpha=0.30, lw=1.5))
            a0, a1 = sector(cl, ancho); t = np.radians(np.linspace(a0, a1, 40)); r = 7.0
            ax.add_patch(Polygon(np.vstack([[cx, cy], np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]]), closed=True, fc=col, alpha=0.18, ec=col, lw=1.2, ls=":"))
            ax.annotate("", xy=(cx + 6 * q["dir"][0], cy + 6 * q["dir"][1]), xytext=(cx, cy), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
            ang = np.degrees(np.arctan2(q["dir"][1], q["dir"][0]))
            ax.text(cx, cy - 1.2, f"{DIRS[cl]}  área {q['area']:.0f}  {ang:.0f}°" + ("  esquina" if q.get("s") == -1 else ""), color=col,
                    fontsize=8, fontweight="bold", ha="center", va="bottom", bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))
        ax.set_title(titulo + f"\n{len(objs)} detectado(s)", fontsize=9, loc="left"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5)
    fig.suptitle(f"idx {i} · es un {d.y[i]} · flecha = dirección exacta de la abertura · sector punteado = la clase en la que cae", fontsize=10, x=0.01, ha="left")
    fig.savefig(salida, bbox_inches="tight", pad_inches=0.15); plt.close(fig)


def fig_pesos(c: S.Compositor, d: S.Datos, i: int, A: int, B: int, salida: Path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    K = c.CLASES_FINAS if c.sub4 else c.CLASES[:10]
    x, ap, lg, P = aportes(c, d, i, [A, B]); grupos = []; resto = []; n = 0
    for b in c.bloques:
        base, idx, cols = S.columnas_de(b, c.celdas); et = CORTO.get(base, S.BLOQUES[base][0].split(" (")[0])
        for sub in dict.fromkeys(cc["sub"] for cc in cols):
            ks = [n + k for k, cc in enumerate(cols) if cc["sub"] == sub and cc["tipo"] == "rejilla"]
            if ks:
                grupos.append((f"{et} {sub}".strip(), ks))
            resto += [(f"{et} {sub} {cc['nombre']}".strip(), n + k) for k, cc in enumerate(cols) if cc["sub"] == sub and cc["tipo"] != "rejilla"]
        n += len(cols)
    filas = [("valor que recibe", x, "Greys"), (f"aporte al {K[A]}", ap[A], "RdBu_r"), (f"aporte al {K[B]}", ap[B], "RdBu_r"), (f"{K[A]} − {K[B]}", ap[A] - ap[B], "RdBu_r")]
    lim = max(1e-6, max(np.abs(ap[A]).max(), np.abs(ap[B]).max(), np.abs(ap[A] - ap[B]).max()))
    fig, axs = plt.subplots(len(filas), len(grupos), figsize=(1.55 * len(grupos) + 1.4, 1.75 * len(filas) + 1.0), dpi=150, squeeze=False)
    for fi, (nom, v, cmap) in enumerate(filas):
        for gi, (gn, ks) in enumerate(grupos):
            ax = axs[fi, gi]; lado = int(round(len(ks) ** 0.5)); m = np.asarray(v)[ks].reshape(lado, lado)
            ax.imshow(m, cmap=cmap, vmin=(0 if fi == 0 else -lim), vmax=(max(1.0, np.abs(x).max()) if fi == 0 else lim))
            for r in range(lado):
                for q in range(lado):
                    val = m[r, q]
                    if fi == 0 or abs(val) >= 0.05:
                        ax.text(q, r, f"{val:.2f}" if fi == 0 else f"{val:+.1f}", ha="center", va="center", fontsize=7.5,
                                color="white" if (fi == 0 and val > 0.6) or (fi > 0 and abs(val) > 0.6 * lim) else "#111")
            ax.set_xticks([]); ax.set_yticks([])
            if fi == 0:
                ax.set_title(gn, fontsize=8)
            if gi == 0:
                ax.set_ylabel(nom, fontsize=9)
            if fi > 0:
                ax.set_xlabel(f"Σ {np.asarray(v)[ks].sum():+.1f}", fontsize=8)
    tot = lambda k: ap[k].sum()
    extra = ("  ·  sin rejilla: " + ", ".join(f"{nm} {x[k]:.2f} ({ap[A][k]:+.1f} / {ap[B][k]:+.1f})" for nm, k in resto)) if resto else ""
    fig.suptitle(f"idx {i} (es un {d.y[i]}) en «{c.nombre}» · lee {K[int(P.argmax())]} · p({K[A]}) {P[A]:.2f}  p({K[B]}) {P[B]:.2f} · logit {K[A]} {lg[A]:+.1f}  {K[B]} {lg[B]:+.1f}\n"
                 f"rojo = sube ese logit, azul = lo baja · suma de aportes: {K[A]} {tot(A):+.1f}, {K[B]} {tot(B):+.1f}{extra}", fontsize=9, x=0.01, ha="left")
    fig.savefig(salida, bbox_inches="tight", pad_inches=0.15); plt.close(fig)
    return {"compositor": c.nombre, "lee": K[int(P.argmax())], "p": {K[A]: float(P[A]), K[B]: float(P[B])}, "logit": {K[A]: float(lg[A]), K[B]: float(lg[B])},
            "suma_aportes": {K[A]: float(tot(A)), K[B]: float(tot(B))},
            "por_rejilla": {gn.replace("\n", " "): {K[A]: float(ap[A][ks].sum()), K[B]: float(ap[B][ks].sum())} for gn, ks in grupos}}


if __name__ == "__main__":
    i, A, B = int(sys.argv[1]), sys.argv[2], sys.argv[3]; comps = sys.argv[4:]
    d = S.Datos(); out = AQUI / "resultados"; out.mkdir(exist_ok=True); fig_objetos(d, i, out / f"100-{i}-objetos.png"); res = {}
    for nm in comps:
        c = compositor(nm, d); K = c.CLASES_FINAS if c.sub4 else c.CLASES[:10]
        slug = "".join(ch if ch.isalnum() else "-" for ch in nm.lower()).strip("-")[:30]
        res[nm] = fig_pesos(c, d, i, K.index(A), K.index(B), out / f"100-{i}-pesos-{slug}.png")
    (out / f"100-{i}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1)); print(json.dumps(res, ensure_ascii=False, indent=1))
