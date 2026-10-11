#!/usr/bin/env python3
"""lazos — calibrar y evaluar el detector de lazos (ver REGLAS.md e instrucciones/02-criterio.md).

    python nn/lazos.py   → resultados/1-sinteticos.png · 2-digitos.png · 3-estadisticas.png · metricas.json  (~90 s)
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import detector as DT                                               # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

RES = AQUI / "resultados"
DATASET = "uci-optdigits-orig-32px-r20261005"
REJILLA = {"sigma_v": (1.0, 1.5), "v_min": (3, 5, 8, 12)}
N = {"calibrar": 150, "probar": 300}
UMBRALES = {"encontrados": 90.0, "falsos": 10.0, "centro_px": 1.5, "cierre": 0.15, "orientacion": 30.0,
            "cerrado_0_6_9": 80.0, "dos_en_8": 60.0, "ninguno_1_7": 80.0}
DS = list(range(-4, 5))
SUP, T1, T2, AZU, NAR, VERDE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#c2410c", "#1b7f3b"
_YY, _XX = np.mgrid[0:32, 0:32] + 0.5


# ─── sintéticos con su verdad ─────────────────────────────────────────────────────────────────────────────────────
def lazo(cx, cy, R, cob, abertura, g) -> np.ndarray:
    """Arco de círculo: cubre `cob` de la vuelta, con el hueco centrado en `abertura` (grados; 0 = →, 90 = ↓)."""
    d = np.hypot(_XX - cx, _YY - cy); a = np.degrees(np.arctan2(_YY - cy, _XX - cx))
    lejos = np.abs((a - abertura + 180) % 360 - 180) > (1 - cob) * 180
    return ((np.abs(d - R) <= g / 2) & lejos).astype(np.uint8)


def recta(cx, cy, L, ang, g) -> np.ndarray:
    t = np.deg2rad(ang); px, py = _XX - cx, _YY - cy
    a = px * np.cos(t) + py * np.sin(t); p = -px * np.sin(t) + py * np.cos(t)
    return ((np.abs(a) <= L / 2) & (np.abs(p) <= g / 2)).astype(np.uint8)


def sinteticos(n: int, semilla: int) -> list[dict]:
    rng = np.random.default_rng(semilla); out = []
    while len(out) < n:
        R, g, cob = rng.uniform(4, 10), int(rng.integers(1, 4)), rng.uniform(0.5, 1.0)
        cx, cy = rng.uniform(R + 3, 32 - R - 3, 2); ab = rng.uniform(0, 360)
        x = lazo(cx, cy, R, cob, ab, g)
        if x.sum() >= 10:
            out.append({"tipo": "lazo", "x": x, "centro": (cx, cy), "radio": R, "cierre": (cob - 0.5) / 0.5,
                        "abertura": ab, "grosor": g})
    for k in range(n):
        if k % 3 == 0:
            x = recta(*rng.uniform(10, 22, 2), rng.uniform(12, 24), rng.uniform(0, 180), int(rng.integers(1, 4)))
        elif k % 3 == 1:
            R = rng.uniform(6, 15); cx, cy = rng.uniform(10, 22, 2)
            x = lazo(cx, cy, R, rng.uniform(0.1, 0.28), rng.uniform(0, 360), int(rng.integers(1, 4)))
        else:
            x = (rng.random((32, 32)) < 0.01).astype(np.uint8)
        out.append({"tipo": "negativo", "x": x})
    return out


def _dang(a, b) -> float:
    return abs((a - b + 180) % 360 - 180)


def evaluar(figs, sigma_v, v_min) -> dict:
    enc, fp, ec, ecie, eori = [], [], [], [], []
    for f in figs:
        L = DT.lazos(f["x"], sigma_v, v_min)
        if f["tipo"] == "negativo":
            fp.append(bool(L)); continue
        dist = [np.hypot(l["centro"][0] - f["centro"][0], l["centro"][1] - f["centro"][1]) for l in L]
        if not dist or min(dist) > 3:
            enc.append(False); continue
        enc.append(True); l = L[int(np.argmin(dist))]
        ec.append(min(dist)); ecie.append(abs(l["cierre"] - f["cierre"]))
        if f["cierre"] < 0.9 and l["orientacion"] is not None:
            eori.append(_dang(l["orientacion"], f["abertura"]))
    r = {"encontrados": round(100 * float(np.mean(enc)), 1), "falsos": round(100 * float(np.mean(fp)), 1),
         "centro_px": round(float(np.median(ec)), 2) if ec else None,
         "cierre": round(float(np.median(ecie)), 3) if ecie else None,
         "orientacion": round(float(np.median(eori)), 1) if eori else None}
    r["puntos"] = round((r["encontrados"] + 100 - r["falsos"]) / 2, 1)
    return r


# ─── dibujo pedido: dígito en gris · cruz azul en el centro · arco delgado del lazo ───────────────────────────────
def dibujar(ax, x, L, titulo="", color=T1):
    ax.imshow(np.where(x > 0, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#b9b8b4"]), vmin=0, vmax=1)
    for l in L:
        cx, cy = l["centro"][0] - 0.5, l["centro"][1] - 0.5; R = l["radio"]
        span = (0.5 + 0.5 * l["cierre"]) * 360
        mitad = l["orientacion"] + 180 if l["orientacion"] is not None else 0.0     # el arco, opuesto a la abertura
        t = np.deg2rad(np.linspace(mitad - span / 2, mitad + span / 2, 90))
        ax.plot(cx + R * np.cos(t), cy + R * np.sin(t), color=NAR, lw=1.4)
        ax.plot(cx, cy, "+", color=AZU, ms=10, mew=2)
    ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(titulo, fontsize=8, pad=2, color=color)


def main() -> int:
    torch.set_num_threads(2); RES.mkdir(exist_ok=True); t0 = time.time()
    plt.rcParams.update({"font.size": 8, "figure.facecolor": SUP, "axes.facecolor": SUP, "axes.titlecolor": T1})
    cal, pru = sinteticos(N["calibrar"], 101), sinteticos(N["probar"], 202)
    out = {"umbrales": UMBRALES, "detector_curvas": {"lam": DT.LAM, "tau": DT.TAU, "sigma_e": DT.SIGMA_E,
                                                      "coh_min": DT.COH_MIN, "kappa_min": DT.KAPPA_MIN},
           "calibracion": []}
    mejor, mp = None, -1.0
    for sv, vm in itertools.product(*REJILLA.values()):
        r = evaluar(cal, sv, vm); out["calibracion"].append({"sigma_v": sv, "v_min": vm, **r})
        if r["puntos"] > mp:
            mp, mejor = r["puntos"], {"sigma_v": sv, "v_min": vm}
    out["elegida"] = mejor
    print(f"calibración ({time.time() - t0:.0f} s): elegida {mejor} · {out['calibracion']}", flush=True)
    sv, vm = mejor["sigma_v"], mejor["v_min"]
    out["sinteticos_prueba"] = evaluar(pru, sv, vm)
    print(f"sintéticos de prueba: {out['sinteticos_prueba']}", flush=True)

    d = dict(np.load(exigir_dataset(DATASET) / "datos.npz"))
    va = np.flatnonzero(d["particion"] == "val")
    X, Y = (d["imagenes"][va] > 0).astype(np.uint8), d["etiquetas"][va].astype(int)
    L0 = [DT.lazos(x, sv, vm) for x in X]
    n_l = np.array([len(l) for l in L0])
    cerrado = np.array([any(l["cierre"] >= 0.5 for l in ls) for ls in L0])
    clase = {}
    for k in range(10):
        m = Y == k
        clase[str(k)] = {"n": int(m.sum()), "lazos_0": round(100 * float((n_l[m] == 0).mean()), 1),
                         "lazos_1": round(100 * float((n_l[m] == 1).mean()), 1),
                         "lazos_2": round(100 * float((n_l[m] == 2).mean()), 1),
                         "lazos_3+": round(100 * float((n_l[m] >= 3).mean()), 1),
                         "con_cerrado": round(100 * float(cerrado[m].mean()), 1),
                         "cierre_medio": round(float(np.mean([l["cierre"] for ls, y in zip(L0, Y) if y == k for l in ls]
                                                             or [np.nan])), 2)}
    out["digitos_por_clase"] = clase
    out["digitos_criterio"] = {"cerrado_0": clase["0"]["con_cerrado"], "cerrado_6": clase["6"]["con_cerrado"],
                               "cerrado_9": clase["9"]["con_cerrado"], "dos_en_8": clase["8"]["lazos_2"],
                               "ninguno_1": clase["1"]["lazos_0"], "ninguno_7": clase["7"]["lazos_0"]}
    print(f"dígitos ({time.time() - t0:.0f} s): {out['digitos_criterio']}", flush=True)

    # desplazamientos (horizontal; lo que sale por un borde se pierde)
    def mover(a, dx):
        o = np.zeros_like(a)
        if dx >= 0:
            o[:, dx:] = a[:, :32 - dx]
        else:
            o[:, :32 + dx] = a[:, -dx:]
        return o
    cambia, err_c, recort = [], [], []
    for dx in DS:
        Xm = np.stack([mover(x, dx) for x in X]); Lm = [DT.lazos(x, sv, vm) for x in Xm]
        cambia.append(round(100 * float(np.mean([len(a) != len(b) for a, b in zip(Lm, L0)])), 1))
        e = [max(a, key=lambda z: z["votos"])["centro"][0] - max(b, key=lambda z: z["votos"])["centro"][0] - dx
             for a, b in zip(Lm, L0) if a and b]
        err_c.append(round(float(np.median(np.abs(e))), 2) if e else None)
        recort.append(round(100 * float(np.mean([xm.sum() < x.sum() for xm, x in zip(Xm, X)])), 1))
    out["desplazamientos"] = {"d": DS, "cambia_n_lazos_pc": cambia, "error_centro_principal_px": err_c, "recortados_pc": recort}
    print(f"desplazamientos: {out['desplazamientos']}", flush=True)

    s, dc = out["sinteticos_prueba"], out["digitos_criterio"]
    out["pasa"] = {"1_encontrados": s["encontrados"] >= UMBRALES["encontrados"], "2_falsos": s["falsos"] <= UMBRALES["falsos"],
                   "3_centro": (s["centro_px"] or 99) <= UMBRALES["centro_px"], "4_cierre": (s["cierre"] or 9) <= UMBRALES["cierre"],
                   "5_orientacion": (s["orientacion"] or 999) <= UMBRALES["orientacion"],
                   "6_cerrado_0": dc["cerrado_0"] >= 80, "6_cerrado_6": dc["cerrado_6"] >= 80, "6_cerrado_9": dc["cerrado_9"] >= 80,
                   "7_dos_en_8": dc["dos_en_8"] >= 60, "8_ninguno_1": dc["ninguno_1"] >= 80, "8_ninguno_7": dc["ninguno_7"] >= 80}
    (RES / "metricas.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # ── 1 · sintéticos de prueba ──
    rng = np.random.default_rng(5)
    lz = [f for f in pru if f["tipo"] == "lazo"]; ng = [f for f in pru if f["tipo"] == "negativo"]
    ej = [lz[i] for i in rng.choice(len(lz), 18, replace=False)] + [ng[i] for i in rng.choice(len(ng), 6, replace=False)]
    fig, axs = plt.subplots(3, 8, figsize=(16, 6.6))
    for ax, f in zip(axs.flat, ej):
        L = DT.lazos(f["x"], sv, vm)
        if f["tipo"] == "lazo":
            dist = [np.hypot(l["centro"][0] - f["centro"][0], l["centro"][1] - f["centro"][1]) for l in L]
            if dist and min(dist) <= 3:
                l = L[int(np.argmin(dist))]
                tit = f"cierre {l['cierre']:.2f} (real {f['cierre']:.2f})"
                ok = abs(l["cierre"] - f["cierre"]) <= 0.15
            else:
                tit, ok = f"NO encontrado (cierre real {f['cierre']:.2f})", False
        else:
            tit, ok = ("negativo: sin lazo ✓" if not L else f"negativo: {len(L)} lazo(s) ✗"), not L
        dibujar(ax, f["x"], L, tit, VERDE if ok else NAR)
    fig.suptitle(f"lazos · 1 · sintéticos de prueba (semilla 202) · gris = figura · cruz azul = centro del lazo · arco = el "
                 f"lazo, con su hueco hacia la abertura\nencontrados {s['encontrados']} % · falsos {s['falsos']} % · error "
                 f"mediano: centro {s['centro_px']} px · cierre {s['cierre']} · orientación {s['orientacion']}°",
                 fontsize=10, color=T1)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(RES / "1-sinteticos.png", dpi=100); plt.close(fig)

    # ── 2 · 100 dígitos de val, 10 por clase ──
    fig, axs = plt.subplots(10, 10, figsize=(13, 15))
    for k in range(10):
        ids = np.flatnonzero(Y == k)[:10]
        for j, i in enumerate(ids):
            L = L0[i]
            dibujar(axs[k, j], X[i], L, f"{k} · {len(L)} lazo(s)" + (" · " + " ".join(f"{l['cierre']:.1f}" for l in L) if L else ""))
    fig.suptitle("lazos · 2 · los 10 primeros dígitos de val de cada clase · gris = dígito · cruz azul = centro del lazo · "
                 "arco = el lazo, hueco hacia la abertura · en el título, el cierre de cada lazo (0 = media vuelta, 1 = "
                 "completo)", fontsize=10, color=T1)
    fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(RES / "2-digitos.png", dpi=90); plt.close(fig)

    # ── 3 · estadísticas ──
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    a = axs[0]; base = np.zeros(10)
    for clave, col, et in (("lazos_0", "#d9d8d4", "0 lazos"), ("lazos_1", AZU, "1"), ("lazos_2", NAR, "2"), ("lazos_3+", T2, "3+")):
        v = np.array([clase[str(k)][clave] for k in range(10)])
        a.bar(range(10), v, bottom=base, color=col, label=et); base += v
    a.set_xticks(range(10)); a.set_ylim(0, 100); a.legend(frameon=False, fontsize=8, ncol=4, loc="upper center")
    a.set_title("cuántos lazos ve en cada clase (val, % de dígitos)", loc="left", fontsize=9.5)
    a = axs[1]; a.axis("off")
    lin = [f"sintéticos de prueba ({N['probar']} lazos + {N['probar']} negativos):",
           f"  encontrados {s['encontrados']} %   (≥ 90)  {'✓' if out['pasa']['1_encontrados'] else '✗'}",
           f"  falsos      {s['falsos']} %   (≤ 10)  {'✓' if out['pasa']['2_falsos'] else '✗'}",
           f"  centro      {s['centro_px']} px  (≤ 1,5) {'✓' if out['pasa']['3_centro'] else '✗'}",
           f"  cierre      {s['cierre']}    (≤ 0,15) {'✓' if out['pasa']['4_cierre'] else '✗'}",
           f"  orientación {s['orientacion']}°   (≤ 30) {'✓' if out['pasa']['5_orientacion'] else '✗'}", "",
           "dígitos de val:"] + \
          [f"  {n:12s} {v:5.1f} %  {'✓' if out['pasa'][p] else '✗'}" for n, v, p in
           (("0 cerrado", dc["cerrado_0"], "6_cerrado_0"), ("6 cerrado", dc["cerrado_6"], "6_cerrado_6"),
            ("9 cerrado", dc["cerrado_9"], "6_cerrado_9"), ("8 con 2", dc["dos_en_8"], "7_dos_en_8"),
            ("1 sin lazo", dc["ninguno_1"], "8_ninguno_1"), ("7 sin lazo", dc["ninguno_7"], "8_ninguno_7"))] + \
          ["", f"calibración: σ_v {sv} · V_min {vm}"]
    a.text(0, 1, "\n".join(lin), va="top", fontsize=9, family="monospace", color=T1)
    a = axs[2]
    a.plot(DS, out["desplazamientos"]["cambia_n_lazos_pc"], "o-", color=NAR, label="% con otro número de lazos")
    a.plot(DS, out["desplazamientos"]["recortados_pc"], ":", color=T2, label="% recortados")
    a.set_xticks(DS); a.set_ylim(0, 100); a.legend(frameon=False, fontsize=8); a.set_xlabel("desplazamiento horizontal (px)")
    a.set_title("desplazamientos (val)", loc="left", fontsize=9.5)
    for a in axs[[0, 2]]:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("lazos · 3 · estadísticas", fontsize=11, color=T1)
    fig.tight_layout(); fig.savefig(RES / "3-estadisticas.png", dpi=100); plt.close(fig)
    print(f"pasa: {out['pasa']} · total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
