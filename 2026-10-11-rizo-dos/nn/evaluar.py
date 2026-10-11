#!/usr/bin/env python3
"""rizo-2 — las cuatro estrategias con los MISMOS números (instrucciones/02-criterio.md): A sintéticos de prueba,
B los 48 doses etiquetados, C las demás clases, D desplazamientos. Figuras 1–5 y resultados/metricas.json.

    python nn/evaluar.py [--rapido]     (~10 min; --rapido: la mitad de dígitos en C)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

AQUI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AQUI / "nn")); sys.path.insert(0, str(AQUI.parent))
import detector as DT                                               # noqa: E402
import plantillas as PL                                             # noqa: E402
import rayos as RY                                                  # noqa: E402
import sinteticos as SN                                             # noqa: E402
from etiquetas import arco_etiqueta, etiquetas                      # noqa: E402
from modelo import BRAZOS, Rizo                                     # noqa: E402
from expcnn import exigir_dataset                                   # noqa: E402

RES = AQUI / "resultados"
DATASET = "uci-optdigits-orig-32px-r20261005"
TOL = 2.5
DS = list(range(-4, 5))
SUP, T1, T2, AZU, NAR, VERDE = "#fcfcfb", "#0b0b0b", "#52514e", "#2a78d6", "#c2410c", "#1b7f3b"
UMBRAL = {"1_encontrados_sint": (">=", 90), "2_falsos_sint": ("<=", 10), "3_centro_sint": ("<=", 1.5), "4_radio_sint": ("<=", 1.0),
          "5_cierre_sint": ("<=", 0.15), "6_abertura_sint": ("<=", 30), "7_encontrados_2": (">=", 85), "8_centro_2": ("<=", 1.5),
          "9_radio_2": ("<=", 1.0), "10_abertura_2": ("<=", 40), "11_de_mas_2": ("<=", 20), "12_uno": ("<=", 15), "12_cuatro": ("<=", 15),
          "12_siete": ("<=", 30), "14_desplaz_caida": ("<=", 5), "15_desplaz_centro": ("<=", 1.0)}


# ─── la estrategia C (CNN) como detector ─────────────────────────────────────────────────────────────────────────
class EstrategiaC:
    def __init__(self, brazo: str, p: float | None = None):
        self.brazo = brazo; carpeta = AQUI / "nn" / "pesos" / brazo
        cfg = json.loads((carpeta / "config.json").read_text())
        self.red = Rizo(**BRAZOS[cfg.get("arq", brazo)]); self.red.load_state_dict(torch.load(carpeta / "best.pt", map_location="cpu")); self.red.eval()
        self.p = p

    @torch.no_grad()
    def mapas(self, X: np.ndarray) -> np.ndarray:
        out = self.red(torch.from_numpy(X.astype(np.float32))[:, None]).numpy()
        out[:, 0] = 1 / (1 + np.exp(-out[:, 0])); return out

    def de_mapa(self, m: np.ndarray, x: np.ndarray, p: float) -> list[dict]:
        h = m[0]; pico = (h == maximum_filter(h, size=5)) & (h >= p) & (x == 0)
        ys, xs = np.nonzero(pico); out = []
        for i in np.argsort(-h[ys, xs]):
            y, xx = ys[i], xs[i]
            ori = float(np.degrees(np.arctan2(m[4, y, xx], m[3, y, xx])) % 360)
            out.append({"centro": (float(xx + 0.5), float(y + 0.5)), "radio": float(np.clip(m[1, y, xx] * 7, 1, 10)),
                        "cierre": float(np.clip(m[2, y, xx], 0, 1)), "orientacion": ori, "puntuacion": float(h[y, xx])})
        return out

    def __call__(self, x: np.ndarray) -> list[dict]:
        return self.de_mapa(self.mapas(np.asarray(x)[None])[0], np.asarray(x) > 0, self.p)

    def calibrar(self, conj: list[dict]) -> dict:
        M = self.mapas(np.stack([f["x"] for f in conj])); out, mejor = [], None
        for p in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            enc, lim, extra = [], [], []
            for f, m in zip(conj, M):
                r = self.de_mapa(m, f["x"] > 0, p)
                if f["tipo"] == "rizo":
                    d = [np.hypot(c["centro"][0] - f["centro"][0], c["centro"][1] - f["centro"][1]) for c in r]
                    enc.append(any(e <= TOL for e in d)); extra.append(sum(e > TOL for e in d))
                else:
                    lim.append(not r)
            fila = {"p": p, "encontrados": round(100 * float(np.mean(enc)), 1), "negativos_limpios": round(100 * float(np.mean(lim)), 1),
                    "extra_por_positivo": round(float(np.mean(extra)), 2),
                    "puntos": round(50 * float(np.mean(enc)) + 50 * float(np.mean(lim)) - 2 * float(np.mean(extra)), 2)}
            out.append(fila)
            if mejor is None or fila["puntos"] > mejor["puntos"]:
                mejor = fila
        self.p = mejor["p"]; return {"rejilla": out, "elegida": {"p": mejor["p"]}, "mejor": mejor}


# ─── medidas ─────────────────────────────────────────────────────────────────────────────────────────────────────
def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _dang(a, b):
    return None if a is None or b is None else float(abs((a - b + 180) % 360 - 180))


def medir_contra_verdad(resultados: list[list[dict]], verdad: list[dict], solo_arriba=False) -> dict:
    """Encontrados (≤ TOL), errores (mediana) del centro, radio, cierre y abertura, y rizos de más."""
    enc, ec, er, ez, ea, extra = [], [], [], [], [], []
    for r, v in zip(resultados, verdad):
        d = [(_dist(c["centro"], v["centro"]), c) for c in r]
        cerca = [c for e, c in d if e <= TOL]
        enc.append(bool(cerca))
        otros = [c for e, c in d if e > TOL and (not solo_arriba or c["centro"][1] < 16)]
        extra.append(bool(otros))
        if cerca:
            c = max(cerca, key=lambda z: z["puntuacion"])
            ec.append(_dist(c["centro"], v["centro"])); er.append(abs(c["radio"] - v["radio"])); ez.append(abs(c["cierre"] - v["cierre"]))
            if v["cierre"] < 0.9:
                a = _dang(c["orientacion"], v["abertura"])
                if a is not None:
                    ea.append(a)
    med = lambda z: (round(float(np.median(z)), 2) if z else None)   # noqa: E731
    return {"n": len(verdad), "encontrados": round(100 * float(np.mean(enc)), 1), "centro_px": med(ec), "radio_px": med(er),
            "cierre": med(ez), "abertura_deg": med(ea), "de_mas_pc": round(100 * float(np.mean(extra)), 1), "encontrado": enc}


def mover(a, dx=0, dy=0):
    o = np.zeros_like(a); H, W = a.shape
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    o[yd, xd] = a[ys, xs]; return o


def dibujar(ax, x, rizos, etiqueta=None, titulo="", color=T1):
    ax.imshow(np.where(x > 0, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#b9b8b4"]), vmin=0, vmax=1)
    if etiqueta is not None:
        arco_etiqueta(ax, etiqueta, VERDE, 1.0)
    for l in rizos:
        cx, cy = l["centro"][0] - 0.5, l["centro"][1] - 0.5; R = l["radio"]
        span = (0.5 + 0.5 * l["cierre"]) * 360
        mitad = l["orientacion"] + 180 if l["orientacion"] is not None else 0.0
        t = np.deg2rad(np.linspace(mitad - span / 2, mitad + span / 2, 90))
        ax.plot(cx + R * np.cos(t), cy + R * np.sin(t), color=NAR, lw=1.3)
        ax.plot(cx, cy, "+", color=AZU, ms=9, mew=1.8)
    ax.set_xlim(-0.5, 31.5); ax.set_ylim(31.5, -0.5); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(titulo, fontsize=7, pad=2, color=color)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--rapido", action="store_true"); a = ap.parse_args()
    torch.set_num_threads(2); RES.mkdir(exist_ok=True); t0 = time.time()
    plt.rcParams.update({"font.size": 8, "figure.facecolor": SUP, "axes.facecolor": SUP, "axes.titlecolor": T1})
    cal_r, cal_p = RY.cargar_calibracion(), PL.cargar_calibracion()
    est = {"R": lambda x: RY.rizos(x, **cal_r), "P": lambda x: PL.rizos(x, cal_p), "V": DT.estrategia_v}
    out = {"tolerancia_px": TOL, "calibracion": {"R": cal_r, "P": {"s": cal_p}, "V": "la de lazos (σ_v 1, V_min 3, rayos 0,97)"}}
    cal = SN.conjunto(150, 150, 101)
    for brazo in sorted(p.name for p in (AQUI / "nn" / "pesos").glob("*") if (p / "best.pt").exists()):
        if True:
            c = EstrategiaC(brazo); out["calibracion"][f"C:{brazo}"] = c.calibrar(cal); est[f"C:{brazo}"] = c
            print(f"C {brazo}: p {c.p} → {out['calibracion'][f'C:{brazo}']['mejor']}", flush=True)
    nombres = list(est)

    # A · sintéticos de prueba
    pru = SN.conjunto(300, 300, 202); pos = [f for f in pru if f["tipo"] == "rizo"]; neg = [f for f in pru if f["tipo"] != "rizo"]
    out["A_sinteticos"] = {}
    for n in nombres:
        rp = [est[n](f["x"]) for f in pos]; rn = [est[n](f["x"]) for f in neg]
        m = medir_contra_verdad(rp, pos); m.pop("encontrado"); m["falsos"] = round(100 * float(np.mean([bool(r) for r in rn])), 1)
        out["A_sinteticos"][n] = m; print(f"A {n}: {m}  ({time.time() - t0:.0f} s)", flush=True)

    # B · los doses etiquetados
    d = dict(np.load(exigir_dataset(DATASET) / "datos.npz")); IM = (d["imagenes"] > 0).astype(np.uint8)
    E = etiquetas(); E_ok = [e for e in E if not e["dudoso"]]
    X2 = [IM[e["i"]] for e in E_ok]
    out["B_doses"] = {}; R2 = {}
    for n in nombres:
        R2[n] = [est[n](x) for x in X2]
        m = medir_contra_verdad(R2[n], E_ok, solo_arriba=True); out["B_doses"][n] = m
        print(f"B {n}: { {k: v for k, v in m.items() if k != 'encontrado'} }", flush=True)

    # C · por clase (val)
    va = np.flatnonzero(d["particion"] == "val"); Y = d["etiquetas"][va].astype(int)
    if a.rapido:
        va, Y = va[::2], Y[::2]
    out["C_clases"] = {}; tiempos = {}
    for n in nombres:
        t1 = time.time(); arriba = np.array([any(r["centro"][1] < 16 for r in est[n](IM[i])) for i in va])
        tiempos[n] = round(1000 * (time.time() - t1) / len(va), 1)
        out["C_clases"][n] = {str(k): round(100 * float(arriba[Y == k].mean()), 1) for k in range(10)}
        print(f"C {n}: {out['C_clases'][n]} · {tiempos[n]} ms/dígito", flush=True)
    out["ms_por_digito"] = tiempos

    # D · desplazamientos (los doses etiquetados; la verdad se mueve d)
    out["D_desplazamientos"] = {"d": DS, "horizontal": {}, "vertical": {}, "recortados_pc": {"horizontal": [], "vertical": []}}
    for eje in ("horizontal", "vertical"):
        rec = []
        for dd in DS:
            Xm = [mover(x, dx=dd) if eje == "horizontal" else mover(x, dy=dd) for x in X2]
            Em = [{**e, "centro": [e["centro"][0] + (dd if eje == "horizontal" else 0), e["centro"][1] + (dd if eje == "vertical" else 0)]} for e in E_ok]
            # «recortado» = se pierde tinta DEL RIZO (a ≤ R + 3 px de su centro): los dígitos tocan arriba y abajo, así que
            # perder tinta de la base al subir el dígito no invalida la medida del rizo
            no_rec = []
            for xm, x, e in zip(Xm, X2, E_ok):
                vuelta = mover(xm, dx=-dd) if eje == "horizontal" else mover(xm, dy=-dd)
                py, px = np.nonzero((x > 0) & (vuelta == 0))
                no_rec.append(not np.any(np.hypot(px + 0.5 - e["centro"][0], py + 0.5 - e["centro"][1]) <= e["radio"] + 3))
            rec.append(round(100 * (1 - float(np.mean(no_rec))), 1))
            for n in nombres:
                res = [est[n](x) for x in Xm]
                m = medir_contra_verdad([r for r, ok in zip(res, no_rec) if ok], [e for e, ok in zip(Em, no_rec) if ok])
                out["D_desplazamientos"][eje].setdefault(n, {"encontrados": [], "centro_px": []})
                out["D_desplazamientos"][eje][n]["encontrados"].append(m["encontrados"]); out["D_desplazamientos"][eje][n]["centro_px"].append(m["centro_px"])
        out["D_desplazamientos"]["recortados_pc"][eje] = rec
        print(f"D {eje}: recortados {rec} · " + " · ".join(f"{n} {out['D_desplazamientos'][eje][n]['encontrados']}" for n in nombres), flush=True)

    # los 15 umbrales
    out["criterio"] = {}
    for n in nombres:
        A, B, C, D = out["A_sinteticos"][n], out["B_doses"][n], out["C_clases"][n], out["D_desplazamientos"]
        caida = max(max(D[e][n]["encontrados"][4] - v for v in D[e][n]["encontrados"]) for e in ("horizontal", "vertical"))
        cen = [v for e in ("horizontal", "vertical") for v in D[e][n]["centro_px"] if v is not None]
        med = {"1_encontrados_sint": A["encontrados"], "2_falsos_sint": A["falsos"], "3_centro_sint": A["centro_px"], "4_radio_sint": A["radio_px"],
               "5_cierre_sint": A["cierre"], "6_abertura_sint": A["abertura_deg"], "7_encontrados_2": B["encontrados"], "8_centro_2": B["centro_px"],
               "9_radio_2": B["radio_px"], "10_abertura_2": B["abertura_deg"], "11_de_mas_2": B["de_mas_pc"], "12_uno": C["1"], "12_cuatro": C["4"],
               "12_siete": C["7"], "14_desplaz_caida": round(caida, 1), "15_desplaz_centro": (max(cen) if cen else None)}
        pasa = {}
        for k, (op, u) in UMBRAL.items():
            v = med[k]; pasa[k] = (v is not None) and ((v >= u) if op == ">=" else (v <= u))
        out["criterio"][n] = {"medido": med, "pasa": pasa, "pasados": int(sum(pasa.values())), "de": len(UMBRAL)}
    orden = sorted(nombres, key=lambda n: (-out["criterio"][n]["pasados"], -out["B_doses"][n]["encontrados"], out["B_doses"][n]["de_mas_pc"]))
    out["ganadora"] = orden[0] if out["criterio"][orden[0]]["pasa"]["7_encontrados_2"] else None
    out["orden"] = orden
    print("criterio: " + " · ".join(f"{n} {out['criterio'][n]['pasados']}/{len(UMBRAL)}" for n in orden) + f" → ganadora {out['ganadora']}", flush=True)

    # ── figuras ──
    # 1 · sintéticos: 6 ejemplos × (verdad + estrategias)
    ej = pos[:5] + neg[:1]; cols = 1 + len(nombres)
    fig, axs = plt.subplots(len(ej), cols, figsize=(1.6 * cols, 1.75 * len(ej)))
    for r, f in enumerate(ej):
        v = f if f["tipo"] == "rizo" else None
        dibujar(axs[r, 0], f["x"], [], v, "verdad" if r == 0 else "")
        for c, n in enumerate(nombres):
            rr = est[n](f["x"]); dibujar(axs[r, c + 1], f["x"], rr, None, (n if r == 0 else "") + f" · {len(rr)}")
    fig.suptitle("rizo-2 · 1 · sintéticos de prueba: verdad (verde) y lo que ve cada estrategia (cruz azul = centro, arco = rizo)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(RES / "1-sinteticos.png", dpi=100); plt.close(fig)

    # 2 · los 48 doses, por estrategia
    k = len(nombres); por_fila = 4; filas = int(np.ceil(len(E) / por_fila))
    fig, axs = plt.subplots(filas, por_fila * k, figsize=(1.15 * por_fila * k, 1.3 * filas + 0.8))
    for j, e in enumerate(E):
        x = IM[e["i"]]
        for c, n in enumerate(nombres):
            ax = axs[j // por_fila, (j % por_fila) * k + c]
            rr = est[n](x); ok = any(_dist(r["centro"], e["centro"]) <= TOL for r in rr)
            dibujar(ax, x, rr, e, f"{n} {e['i']}" + (" ✓" if ok else " ✗"), VERDE if ok else NAR)
    for ax in axs.flat[len(E) * k:]:
        ax.axis("off")
    fig.suptitle(f"rizo-2 · 2 · los 48 doses etiquetados: verde = la etiqueta a mano; azul/naranja = lo detectado por cada estrategia "
                 f"({' · '.join(nombres)}) · ✓ = algún candidato a ≤ {TOL} px", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(RES / "2-doses-etiquetados.png", dpi=85); plt.close(fig)

    # 3 · por clase
    fig, ax = plt.subplots(figsize=(11, 4)); w = 0.8 / k
    cols_ = ["#b9b8b4", NAR, AZU, VERDE, "#7c3aed", "#0f766e"]
    for c, n in enumerate(nombres):
        ax.bar(np.arange(10) + (c - (k - 1) / 2) * w, [out["C_clases"][n][str(i)] for i in range(10)], w, color=cols_[c], label=n)
    ax.set_xticks(range(10)); ax.set_ylim(0, 105); ax.legend(frameon=False, fontsize=8, ncol=k)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("rizo-2 · 3 · % de dígitos de val con algún rizo en la MITAD SUPERIOR, por clase y estrategia (1 y 4 ≤ 15 %, 7 ≤ 30 %)",
                 loc="left", fontsize=9.5)
    fig.tight_layout(); fig.savefig(RES / "3-por-clase.png", dpi=100); plt.close(fig)

    # 4 · desplazamientos
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, eje in zip(axs, ("horizontal", "vertical")):
        for c, n in enumerate(nombres):
            ax.plot(DS, out["D_desplazamientos"][eje][n]["encontrados"], "o-", color=cols_[c], label=n, ms=4)
        ax2 = ax.twinx(); ax2.bar(DS, out["D_desplazamientos"]["recortados_pc"][eje], 0.5, color="#e5e3df", zorder=0)
        ax2.set_ylim(0, 100); ax2.set_ylabel("% recortados (gris)", color=T2, fontsize=8); ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
        ax.set_ylim(0, 105); ax.set_xlabel(f"d px ({eje})"); ax.set_ylabel("% encontrados (sin recortados)"); ax.legend(frameon=False, fontsize=8)
        ax.spines[["top"]].set_visible(False)
    fig.suptitle("rizo-2 · 4 · resistencia a los DESPLAZAMIENTOS en los 47 doses etiquetados (la verdad se mueve d)", fontsize=9.5)
    fig.tight_layout(); fig.savefig(RES / "4-desplazamientos.png", dpi=100); plt.close(fig)

    # 5 · fallos de la mejor
    mejor = max(nombres, key=lambda n: out["B_doses"][n]["encontrados"] - out["B_doses"][n]["de_mas_pc"]); fallos = [(e, r) for e, r, ok in zip(E_ok, R2[mejor], out["B_doses"][mejor]["encontrado"]) if not ok]
    if fallos:
        nc = min(12, len(fallos)); nf = int(np.ceil(len(fallos) / nc))
        fig, axs = plt.subplots(nf, nc, figsize=(1.5 * nc + 0.5, 1.7 * nf + 0.6), squeeze=False)
        for ax, (e, r) in zip(axs.flat, fallos):
            dibujar(ax, IM[e["i"]], r, e, f"idx {e['i']}")
        for ax in axs.flat[len(fallos):]:
            ax.axis("off")
        fig.suptitle(f"rizo-2 · 5 · los doses que la mejor estrategia ({mejor}) NO encuentra: verde = etiqueta, azul = lo que vio", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(RES / "5-fallos.png", dpi=100); plt.close(fig)
    for n in nombres:
        out["B_doses"][n].pop("encontrado")
    (RES / "metricas.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"→ resultados/metricas.json · figuras 1–5 · total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
